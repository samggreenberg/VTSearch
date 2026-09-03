"""``vg_scale``: one pickle holding every ``(class, band)`` cell of #3156.

The construction, in one paragraph: one image pool and one class list
(:data:`pile_config.SCALE_CLASSES`); for a class *c* and band *B* an image is a
**positive** when its compact union box for *c* falls in *B*, a **negative**
when it holds no instance of any class in *C*, and **excluded** otherwise -- it
holds *c* at some other size, so scoring it as a negative would penalise a
detector for finding a real bus, which is what #3156 is about. Exclusion is
carried per media as ``evaluable_categories`` and honoured by
``vtscore.eval.labels.evaluable_pool``.

Cells are *designated* rather than inferred: exactly ``SCALE_N_POS`` positives
and one shared pool of ``SCALE_N_NEG`` negatives each. Every cell therefore has
identical prevalence and identical negatives, so a small-vs-large difference is
a paired contrast on one class rather than two datasets of different difficulty.

**The labels are COCO's, and the pool is the half of VG that can carry them.**
VG's own annotation is not exhaustive and measurably fails this construction --
see :func:`anchor_to_coco` and ``coco_anchor.py``.

The build is six passes, and they are named rather than inlined because two of
them are where this dataset's expensive bugs have lived: :func:`apply_corrections`
is the single point at which a box crosses from normalised into pixel space
(#3281), and :func:`designate_cells` is what decides whether a rebuild keeps the
images a human already reviewed. Each takes what it reads and returns what it
produces, so both can be exercised without the 100 GB of VG source the loader
otherwise needs. See ``tests_lib/meta/test_pile_vg_scale.py``.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import pile_config as pc

from pilebuild.corrections import correction_boxes_px, load_corrections
from pilebuild.env import log
from pilebuild.geometry import region_geometry_problems
from pilebuild.vgsource import vg_boxes_by_name, vg_image_paths, vg_objects_json, vg_source


def read_vg_labels(
    records: list, paths: dict, dims: dict[int, tuple[int, int]], wanted: set[str]
) -> dict[int, dict[str, list[list[float]]]]:
    """``{image_id: {class: [box_px]}}`` for every usable image in the VG source.

    An image with none of the classes still gets an entry (an empty dict): it is
    a candidate *negative*, and dropping it here would shrink the shared pool to
    the images that already hold something.
    """
    labels: dict[int, dict[str, list[list[float]]]] = {}
    for rec in records:
        iid = int(rec["image_id"])
        if iid not in paths or dims.get(iid) is None:
            continue
        labels[iid] = vg_boxes_by_name(rec, wanted)
    return labels


def anchor_to_coco(
    labels: dict[int, dict[str, list[list[float]]]],
    dims: dict[int, tuple[int, int]],
    coco_of: dict[int, int],
    truth: dict[int, dict[str, list[list[float]]]],
    coco_dims: dict[int, tuple[int, int]],
    wanted: set[str],
) -> tuple[dict[int, tuple[int, int]], set[int], int, int]:
    """Replace VG's labels with COCO's wherever COCO annotates the same image.

    Returns ``(box_dims, exhaustive, n_anchored, n_reframed)`` and edits
    *labels* in place.

    VG's own annotation is not exhaustive: measured against COCO its recall over
    C is 0.61, and 1.35% of the images it treats as negatives actually hold the
    object -- ~54 hidden positives against 100 labelled ones per cell, and 4.1%
    for ``backpack`` puts more real backpacks among the negatives than among the
    positives (``coco_anchor.py``, issue #3156).

    48% of VG's images ARE COCO images, and COCO annotates C exhaustively, so on
    that half the repair is free and total. The other half has no reference and
    is what the human slates are for (``make_audit_slate.py``); its verdicts
    arrive through the corrections file. The pool stays the whole of VG either
    way -- restricting it to the COCO half would make this a COCO subset with
    extra steps, losing VG's non-COCO diversity for nothing.

    ``box_dims`` is the pixel space each image's boxes live in, and is the whole
    reason this pass returns anything. VG ships DOWNSCALED copies of the COCO
    originals -- 500 px wide against COCO's 640, on 95% of the overlap -- so a
    COCO box normalised by the VG file's dimensions lands in the wrong place and
    with the wrong extent. Every box is normalised by the dimensions of the
    image its coordinates were measured on.
    """
    box_dims: dict[int, tuple[int, int]] = dict(dims)
    exhaustive: set[int] = set()
    n_anchored = 0
    n_reframed = 0
    for iid in labels:
        cid = coco_of.get(iid)
        ref = truth.get(cid) if cid is not None else None
        if ref is None:
            continue
        wh = coco_dims.get(cid)
        if wh is None:
            continue
        # A normalised box only transfers between two copies of an image if they
        # frame the same thing. 49 of the 51,497 overlaps disagree on aspect
        # ratio -- some are transposed (VG 500x375 against COCO 375x500), i.e. a
        # rotated or re-cropped copy -- and there COCO's box does not describe
        # VG's pixels at all. Those keep VG's own labels and stay unanchored
        # rather than importing a box that points at the wrong part of a
        # different framing.
        vw, vh = dims[iid]
        if abs((vw / vh) - (wh[0] / wh[1])) / (wh[0] / wh[1]) > pc.MAX_ASPECT_DRIFT:
            n_reframed += 1
            continue
        box_dims[iid] = wh
        # COCO's annotation REPLACES VG's for this image rather than merging
        # with it: the two disagree in both directions, and only one of them is
        # exhaustive. Keeping VG's extra boxes would reintroduce exactly the
        # unverifiable labels this is repairing.
        labels[iid] = {name: bs for name, bs in ref.items() if name in wanted}
        exhaustive.add(iid)
        n_anchored += 1
    return box_dims, exhaustive, n_anchored, n_reframed


def apply_corrections(
    labels: dict[int, dict[str, list[list[float]]]],
    corrections: dict[tuple[int, str], dict],
    box_dims: dict[int, tuple[int, int]],
    exhaustive: set[int],
) -> set[tuple[int, str]]:
    """Fold human verdicts into *labels*, and return the pairs that cannot be banded.

    **This is the only place a correction's box changes space**, and it is the
    reason the pass has a name. The box, when there is one, is NORMALISED, and
    everything in *labels* is in pixels -- so it is converted here, once, against
    the same ``(W, H)`` the region write later divides by. Merging it unconverted
    is #3281: the region write then normalises a normalised coordinate, which
    divides it by ~500 and parks the box on the frame origin. Nothing downstream
    could see that, because the BAND is derived from the same corrupted box, so
    the cell name and its boxes stayed consistent with each other all the way
    into the study.

    A pair a reviewer ruled "present" without drawing a box cannot be banded: a
    band is a claim about size, and no size was measured. It leaves every cell of
    that class instead -- neither a positive nor a negative -- which is precisely
    what the third value is for.
    """
    unbanded: set[tuple[int, str]] = set()
    for (iid, name), verdict in corrections.items():
        if iid not in labels:
            continue
        if verdict.get("present"):
            boxes = correction_boxes_px(verdict, *box_dims[iid])
            if boxes:
                labels[iid][name] = boxes
            else:
                labels[iid].pop(name, None)
                unbanded.add((iid, name))
        else:
            labels[iid].pop(name, None)
        exhaustive.add(iid)  # someone looked at this pair
    return unbanded


def band_candidates(
    labels: dict[int, dict[str, list[list[float]]]],
    box_dims: dict[int, tuple[int, int]],
    unbanded: set[tuple[int, str]],
) -> tuple[dict[str, dict[str, list[int]]], dict[tuple[int, str], list[list[float]]], list[int]]:
    """Sort every image into ``(class, band)`` supply, or into the clean pool.

    Returns ``(supply, boxes_for, clean)``. An image with no instance of any
    class in C joins ``clean`` -- unless a reviewer said one *is* present without
    drawing it, which makes it neither a positive nor a true negative.
    """
    supply: dict[str, dict[str, list[int]]] = {c: {b: [] for b in pc.BOX_BANDS} for c in pc.SCALE_CLASSES}
    boxes_for: dict[tuple[int, str], list[list[float]]] = {}
    clean: list[int] = []

    for iid, by_name in labels.items():
        W, H = box_dims[iid]
        area = float(W * H)
        if not by_name:
            # Only a true negative for every class in C may join the shared pool.
            if not any((iid, c) in unbanded for c in pc.SCALE_CLASSES):
                clean.append(iid)
            continue
        for name, bs in by_name.items():
            ux0 = min(b[0] for b in bs)
            uy0 = min(b[1] for b in bs)
            ux1 = max(b[2] for b in bs)
            uy1 = max(b[3] for b in bs)
            union = max(0.0, ux1 - ux0) * max(0.0, uy1 - uy0) / area
            largest = max((b[2] - b[0]) * (b[3] - b[1]) for b in bs) / area
            # Scattered instances in *this* image: the union box describes the
            # scatter rather than the object, so the image is excluded from
            # every band of this class rather than banded by a box no user
            # would drag.
            if union > largest * pc.BAND_MAX_INFLATION:
                continue
            for band, (lo, hi) in pc.BOX_BANDS.items():
                if lo <= union < hi:
                    supply[name][band].append(iid)
                    boxes_for[(iid, pc.scale_cell(name, band))] = bs
                    break
    return supply, boxes_for, clean


def rank(cell: str, iid: int) -> str:
    """A cell-local ordering key that does not move when the pool changes.

    Selection must be stable under a changing candidate list, not merely
    deterministic. ``rng.sample`` is deterministic given the same list, but any
    edit to the pool -- a label fix, an image excluded as a re-framed copy --
    reshuffles the entire draw, and a rebuild then silently retires images a
    human already reviewed (49 of 360 in one such rebuild). Ranking each
    candidate by a hash of ``(cell, image_id)`` instead means adding or removing
    one image changes only that image's membership.
    """
    return hashlib.sha1(f"{cell}:{iid}".encode()).hexdigest()  # noqa: S324 - not security


def designate_cells(
    supply: dict[str, dict[str, list[int]]],
    corrections: dict[tuple[int, str], dict],
    roster: dict,
) -> dict[str, list[int]]:
    """Choose each cell's ``SCALE_N_POS`` positives, preserving reviewed membership.

    A roster pins the membership a review was actually carried out against.
    Without it, switching selection rules retires images a human has already
    judged -- the hash draw and the earlier random draw share only ~228 of 3,900
    negatives, so the entire negative review would have been orphaned. Entries
    that are no longer eligible (a correction moved or removed them) drop out and
    the shortfall is backfilled by :func:`rank`, so the roster adapts without
    ever reshuffling what it can keep.
    """
    chosen: dict[str, list[int]] = {}
    for c in pc.SCALE_CLASSES:
        for band in pc.BOX_BANDS:
            pool = sorted(supply[c][band])
            cell = pc.scale_cell(c, band)
            if len(pool) < pc.SCALE_N_POS:
                # Say so rather than quietly building a smaller cell: unequal
                # prevalence between bands is the defect this construction
                # exists to remove.
                log(f"  UNDER-SUPPLIED {cell}: {len(pool)} positives (wanted {pc.SCALE_N_POS})")
            eligible = set(pool)
            pinned = [i for i in roster.get("cells", {}).get(cell, []) if i in eligible]
            # A correction can move an image to another band -- that is the
            # point of re-drawing a box. If the destination cell is already full
            # of images nobody has looked at, the reviewed one lands nowhere and
            # the review quietly stops covering it (99 of 360 boxed positives,
            # first time round). Reviewed images therefore outrank unreviewed
            # ones for a seat, wherever their box now puts them.
            reviewed = {i for i in eligible if (i, c) in corrections}
            order = (
                [i for i in pinned if i in reviewed]
                + sorted(reviewed - set(pinned), key=lambda i: rank(cell, i))
                + [i for i in pinned if i not in reviewed]
                + sorted(eligible - reviewed - set(pinned), key=lambda i: rank(cell, i))
            )
            chosen[cell] = order[: pc.SCALE_N_POS]
    return chosen


def draw_negatives(clean: list[int], roster: dict) -> tuple[list[int], list[int]]:
    """The shared negative pool and its spares, drawn from the clean images.

    Spares are drawn beyond the designated pool on purpose: a human verdict can
    retire a contaminated negative later, and re-designating from spares costs a
    relabel rather than a re-embed of every cell.
    """
    want_neg = pc.SCALE_N_NEG + pc.SCALE_N_NEG_SPARE
    clean_set = set(clean)
    drawn = [i for i in roster.get("negatives", []) + roster.get("spares", []) if i in clean_set]
    if len(drawn) < want_neg:
        extra = sorted(clean_set - set(drawn), key=lambda i: rank("__negatives__", i))
        drawn += extra[: want_neg - len(drawn)]
    drawn = drawn[:want_neg]
    return drawn[: pc.SCALE_N_NEG], drawn[pc.SCALE_N_NEG :]


def _emit_medias(
    medias: dict[int, dict],
    paths: dict,
    chosen: dict[str, list[int]],
    negatives: list[int],
    spares: list[int],
    boxes_for: dict[tuple[int, str], list[list[float]]],
    box_dims: dict[int, tuple[int, int]],
    exhaustive: set[int],
    cells: list[str],
    embedder_name: str,
) -> None:
    """Read the pixels and write one media dict per designated image."""
    from PIL import Image  # noqa: PLC0415

    # media id -> the cells it is a positive for. Negatives get every cell.
    positive_in: dict[int, list[str]] = defaultdict(list)
    for cell, ids in chosen.items():
        for iid in ids:
            positive_in[iid].append(cell)
    neg_set = set(negatives)

    for iid in sorted(set(positive_in) | set(negatives) | set(spares)):
        path = paths[iid]
        try:
            with Image.open(path) as im:
                vw, vh = im.size
            data = path.read_bytes()
        except Exception:  # noqa: BLE001 - a corrupt file just drops out
            continue
        if vw <= 0 or vh <= 0:
            continue
        # Normalised region boxes are resolution-independent, so they must be
        # divided by the size of the image the coordinates came from -- which is
        # the COCO original for a repaired image, not the VG copy carrying the
        # pixels.
        W, H = box_dims[iid]
        cats = sorted(positive_in.get(iid, []))
        regions = [
            {"box": [b[0] / W, b[1] / H, b[2] / W, b[3] / H], "label": cell}
            for cell in cats
            for b in boxes_for.get((iid, cell), [])
        ]
        medias[iid] = {
            "id": iid,
            "media_type": "image",
            "embedder": embedder_name,
            "duration": 0,
            "file_size": 0,
            "md5": "",
            "embeddings": {},
            "media_bytes": data,
            "media_string": None,
            "filename": path.name,
            "category": cats[0] if cats else "",
            "categories": cats,
            # A designated cell membership, not a closed world: a positive is
            # scorable only in the cells it was drawn for, and the shared
            # negatives are scorable everywhere.
            "evaluable_categories": cats if cats else (list(cells) if iid in neg_set else []),
            # Whether this image's labels rest on an exhaustive reference (COCO,
            # or a human who looked). False means VG's silence is the only
            # evidence of absence -- which is what the review slates target.
            "labels_exhaustive": iid in exhaustive,
            "regions": regions,
            "origin": {"importer": "vg_scale", "params": {"embedder": embedder_name, "labels": "coco"}},
            "origin_name": str(path),
        }


def load(dataset: str, medias: dict[int, dict], embedder_name: str) -> None:
    """Run the six passes over the VG source and write the designated medias."""
    import coco_anchor as ca  # noqa: PLC0415

    wanted = set(pc.SCALE_CLASSES)
    cells = [pc.scale_cell(c, b) for c in pc.SCALE_CLASSES for b in pc.BOX_BANDS]

    paths = vg_image_paths()
    _, records, dims = vg_source()

    image_data, instances = ca.ensure_sources(pc.PILE / "coco_anchor", fetch=False)
    truth = ca.coco_truth(instances, wanted)
    with image_data.open() as fh:
        coco_of = {int(m["image_id"]): int(m["coco_id"]) for m in json.load(fh) if m.get("coco_id")}

    corrections = load_corrections()
    log(f"  {len(coco_of)} VG images carry a coco_id; {len(corrections)} human verdicts on file")

    labels = read_vg_labels(records, paths, dims, wanted)
    box_dims, exhaustive, n_anchored, n_reframed = anchor_to_coco(labels, dims, coco_of, truth, ca.COCO_DIMS, wanted)
    unbanded = apply_corrections(labels, corrections, box_dims, exhaustive)
    log(
        f"  labels: {len(labels)} VG images, {n_anchored} repaired from COCO, "
        f"{len(exhaustive)} with a verified pair, {n_reframed} skipped as re-framed copies"
    )

    supply, boxes_for, clean = band_candidates(labels, box_dims, unbanded)

    roster = {}
    if pc.ROSTER.exists():
        roster = json.loads(pc.ROSTER.read_text())
        log(f"  roster: {pc.ROSTER.name} pins {len(roster.get('cells', {}))} cells")

    chosen = designate_cells(supply, corrections, roster)
    clean.sort()
    negatives, spares = draw_negatives(clean, roster)
    pc.ROSTER.write_text(json.dumps({"cells": chosen, "negatives": negatives, "spares": spares}, indent=1) + "\n")
    log(
        f"  {sum(len(v) for v in chosen.values())} positives over {len(cells)} cells, "
        f"{len(negatives)} shared negatives + {len(spares)} spares (from {len(clean)} clean images)"
    )

    _emit_medias(medias, paths, chosen, negatives, spares, boxes_for, box_dims, exhaustive, cells, embedder_name)

    # Refuse to embed a pickle whose boxes are impossible. `--verify` runs the
    # same check, but only after the GPU hours are spent and the cell is on
    # disk; #3281 got as far as three published studies that way.
    bad = region_geometry_problems(medias)
    if bad:
        raise SystemExit("vg_scale: " + "; ".join(bad))


def check(dataset: str) -> str:
    """What a ``vg_scale`` rebuild reads, without parsing the multi-GB source."""
    objects_json = vg_objects_json()
    if not objects_json.exists():
        raise SystemExit(f"{dataset}: missing {objects_json}")
    n_cells = len(pc.SCALE_CLASSES) * len(pc.BOX_BANDS)
    roster = "roster present" if pc.ROSTER.exists() else "NO ROSTER (membership would be redrawn)"
    return f"VG source present, {n_cells} cells, {roster}"
