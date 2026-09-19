"""The scale-family core: banding, designation, the pool draw, the media shape.

Lifted out of `pilebuild/loaders/vg_scale.py` when Visual Genome was retired
(#4035-era cleanup). None of it is about VG: it takes boxes in pixel space and a
label dict and answers *which cell does this image belong to*, which is the same
question over any exhaustively annotated source.

**It lives apart from any loader on purpose.** `coco_quarry` and the retired
`vg_scale` were only ever comparable because they banded, designated, drew
negatives and built their media dicts through the *same objects* rather than the
same intentions -- the reason :func:`band_for` was split out in the first place,
and the reason `coco_only_supply.py` imports it rather than restating it. Keeping
the core inside a source-specific loader made that coupling invisible and made
retiring the source look like it would take the core with it.

What is emphatically NOT here: reading a free-text vocabulary, guessing which
spelling means the class, withholding the ambiguous ones, anchoring one source's
labels to another's, or applying a corrections file. Those existed to repair an
annotation source that could not answer the question asked of it, and they went
with it.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict

import pile_config as pc

from pilebuild.env import log

SCATTERED = "scattered"
OVERSIZE = "oversize"


def band_for(boxes: list[list[float]], W: int, H: int) -> str:
    """The band one class's boxes put an image in, or why they put it in none.

    Returns a key of :data:`pile_config.BOX_BANDS`, or :data:`SCATTERED` /
    :data:`OVERSIZE`. Split out of :func:`band_candidates` so that anything
    asking "what band would these boxes imply?" -- the builder, and
    ``audit_band_drift.py``, which re-bands the same images off a second
    annotation source -- asks it of one implementation. A second copy of this
    rule would answer a drift question with its own drift.

    *boxes* must be non-empty and in the pixel space of ``(W, H)``.
    """
    area = float(W * H)
    ux0 = min(b[0] for b in boxes)
    uy0 = min(b[1] for b in boxes)
    ux1 = max(b[2] for b in boxes)
    uy1 = max(b[3] for b in boxes)
    union = max(0.0, ux1 - ux0) * max(0.0, uy1 - uy0) / area
    largest = max((b[2] - b[0]) * (b[3] - b[1]) for b in boxes) / area
    # Scattered instances in *this* image: the union box describes the scatter
    # rather than the object, so the image is excluded from every band of this
    # class rather than banded by a box no user would drag.
    if union > largest * pc.BAND_MAX_INFLATION:
        return SCATTERED
    for band, (lo, hi) in pc.BOX_BANDS.items():
        if lo <= union < hi:
            return band
    return OVERSIZE


def band_candidates(
    labels: dict[int, dict[str, list[list[float]]]],
    box_dims: dict[int, tuple[int, int]],
    unbanded: set[tuple[int, str]],
    classes: tuple[str, ...] | None = None,
    designated: dict[tuple[int, str], list[list[float]]] | None = None,
) -> tuple[dict[str, dict[str, list[int]]], dict[tuple[int, str], list[list[float]]], list[int]]:
    """Sort every image into ``(class, band)`` supply, or into the clean pool.

    Returns ``(supply, boxes_for, clean)``. *designated* carries a reviewer's
    chosen box per ``(image, class)`` (#3726): it decides the **band**, while
    ``boxes_for`` still carries every instance, so the cell knows what the class
    has in the image and the band still says which one was picked.

    Returns ``(supply, boxes_for, clean)``. An image with no instance of any
    class in C joins ``clean`` -- unless its ``(image, class)`` pair is in
    *unbanded*, which makes it neither a positive nor a true negative. Two things
    put a pair there: a reviewer who said one *is* present without drawing it (no
    size was measured, and a band is a claim about size), and an ambiguous VG
    spelling that may or may not be the class (:func:`lift_ambiguous`).

    *classes* defaults to :data:`pile_config.SCALE_CLASSES`, i.e. the built
    dataset. It is a parameter so a class being *considered* for C can be banded
    by this exact rule -- the scatter filter and the band edges included --
    rather than by a second implementation in the slate builder that would be
    free to drift from it (#3588). The default keeps every existing caller
    byte-identical.
    """
    classes = tuple(classes) if classes is not None else pc.SCALE_CLASSES
    supply: dict[str, dict[str, list[int]]] = {c: {b: [] for b in pc.BOX_BANDS} for c in classes}
    boxes_for: dict[tuple[int, str], list[list[float]]] = {}
    clean: list[int] = []

    for iid, by_name in labels.items():
        W, H = box_dims[iid]
        if not by_name:
            # Only a true negative for every class in C may join the shared pool.
            if not any((iid, c) in unbanded for c in classes):
                clean.append(iid)
            continue
        for name, bs in by_name.items():
            # The band is a claim about the object this cell is about, so a
            # reviewer's designation decides it and the other instances do not
            # drag it (#3726). Without a designation this is exactly the old
            # behaviour: the union over everything the class has here.
            picked = (designated or {}).get((iid, name))
            band = band_for(picked or bs, W, H)
            if band not in pc.BOX_BANDS:  # scattered, or bigger than a region
                continue
            supply[name][band].append(iid)
            boxes_for[(iid, pc.scale_cell(name, band))] = bs
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


def disqualified_negatives(roster: dict, clean: set[int]) -> list[int]:
    """Rostered negatives that can no longer BE negatives, accumulated.

    An image holding a class in *C* cannot serve as a negative for anything, so a
    rebuild is right to drop it -- but `check_review_coverage.py` sees only that
    a reviewed image is gone and reads a correct rebuild as lost review.
    Expanding *C* from twelve to 25 disqualified 1,530 of them at once (#3588).

    **Accumulated, not recomputed, and that is the whole point of the function.**
    ``load`` runs once per embedder and rewrites the roster each time, so the
    first cell sees the old negatives and records the disqualification while
    every later cell reads a roster whose negatives are already clean, computes
    an empty set, and overwrites the fact. The value survived exactly one cell of
    a five-cell build before this existed. Same shape as the counter that was
    placed before the pass discarding its work (#3637).

    Spares count too: they are drawn into the pickle and can be designated later,
    so one becoming ineligible is the same event.
    """
    was = {int(i) for i in roster.get("disqualified", [])}
    pool = list(roster.get("negatives", [])) + list(roster.get("spares", []))
    return sorted(was | {int(i) for i in pool if int(i) not in clean})


def draw_negatives(
    clean: list[int], roster: dict, coco_scored: set[int], coco_fraction: float
) -> tuple[list[int], list[int]]:
    """The shared negative pool and its spares, drawn from the clean images.

    Spares are drawn beyond the designated pool on purpose: a human verdict can
    retire a contaminated negative later, and re-designating from spares costs a
    relabel rather than a re-embed of every cell.

    The draw is **stratified by provenance** (#3670). *coco_fraction* is the share
    of the pool that must come from *coco_scored* -- images COCO annotated, where
    "holds none of C" is a fact rather than VG's silence. The caller derives it
    from :data:`pile_config.SCALE_NEG_COMPOSITION`: 1.0 for an all-provable pool,
    the positives' own COCO share for a provenance-matched one.

    ``coco_scored`` is deliberately **not** the ``exhaustive`` set the rest of
    the build carries. That one also holds every image a human looked at, and a
    human who looked for a `bus` established nothing about the other eleven
    classes -- so admitting those here would put images in an "all-provable"
    pool whose absence claim is still VG's silence for most of *C*. The
    distinction costs nothing: 34,071 clean images carry a COCO pairing against
    the 9,900 the pool needs.

    Stratifying rather than filtering is what keeps the roster honest. A pinned
    image that is still clean keeps its seat **within its own stratum**, so a
    composition change retires only the images the new composition cannot hold
    instead of reshuffling the whole draw -- the failure that orphaned 49 of 360
    reviewed images the last time a selection rule changed.
    """
    clean_set = set(clean)
    want_prov = round(pc.SCALE_N_NEG * coco_fraction)
    strata = [
        (clean_set & coco_scored, want_prov),
        (clean_set - coco_scored, pc.SCALE_N_NEG - want_prov),
    ]
    pinned = [i for i in roster.get("negatives", []) + roster.get("spares", []) if i in clean_set]

    def draw(pool: set[int], want: int, taken: set[int]) -> list[int]:
        """*want* images from *pool*, roster pins first, then hash-ranked."""
        avail = pool - taken
        out = [i for i in pinned if i in avail][:want]
        if len(out) < want:
            rest = sorted(avail - set(out), key=lambda i: rank("__negatives__", i))
            out += rest[: want - len(out)]
        return out

    negatives: list[int] = []
    for pool, want in strata:
        negatives += draw(pool, want, set(negatives))

    # Spares come from the same strata in the same proportion, so promoting one
    # later cannot change what the pool is made of.
    spares: list[int] = []
    for pool, want in strata:
        share = round(pc.SCALE_N_NEG_SPARE * want / pc.SCALE_N_NEG) if pc.SCALE_N_NEG else 0
        spares += draw(pool, share, set(negatives) | set(spares))
    # A short pool yields what there is rather than silently rebalancing: the
    # builder reports an under-supplied pool, it does not paper over one.
    short = pc.SCALE_N_NEG + pc.SCALE_N_NEG_SPARE - len(negatives) - len(spares)
    if short > 0:
        spares += draw(clean_set, short, set(negatives) | set(spares))
    return negatives, spares


def _cells_by_class(cells: list[str]) -> dict[str, set[str]]:
    """``{class: {cell, ...}}`` for a cell list, whatever its keying.

    ``vg_scale`` cells are ``class@band`` and ``vg_scale_deep`` cells are the
    bare class, so the split is on the suffix separator :func:`pile_config.scale_cell`
    writes. A class name never contains it -- `stop sign` has a space, not an
    `@` -- so the head of the split is the class in both spellings.
    """
    out: dict[str, set[str]] = defaultdict(set)
    for cell in cells:
        out[cell.split("@", 1)[0]].add(cell)
    return dict(out)


def _evaluable(
    iid: int,
    cats: list[str],
    cells: list[str],
    neg_set: set[int],
    labels: dict[int, dict[str, list[list[float]]]],
    coco_scored: set[int],
    reviewed_absent: set[tuple[int, str]],
    reviewed_present: set[tuple[int, str]],
) -> list[str]:
    """Which cells this image can be SCORED in -- positive or negative.

    ``categories`` says what an image *is* a positive for; this says where it is
    allowed to be judged at all. A shared negative is judged everywhere. A
    positive was, until #3667, judged **only in its own cells** -- so an image
    holding a book and no bus was neither a bus positive nor a bus negative, and
    41.9% of the pile fell out of every class's evaluation.

    The exclusion is right for the same class at another size and wrong for
    every other class, so it is now applied per class: a class this image does
    not hold contributes all of its cells, and the image scores there as the
    negative it is.

    **Gated per class on who actually answered, which is not the same as who
    looked.** On a COCO-annotated image "holds no bus" is a fact about all eighty
    classes at once, so ``coco_scored`` admits every class. Off COCO it is VG's
    silence, which #3588 measured wrong 0.5-2.5% of the time, and importing that
    into the negatives is a separate decision -- ``SCALE_CROSS_CLASS_NEGATIVES``
    turns the whole thing off rather than pretending the two halves are alike.

    This used to read ``exhaustive``, and that set has **two** populations in it:
    images COCO answered, and *any image a human looked at*, because
    :func:`apply_corrections` adds every reviewed image to it. A reviewer asked
    "does this hold a `car`?" established a fact about `car` and nothing about
    `bus` -- so the rule promoted 467 review-only images into cross-class
    negatives they had not earned, 322 of them designated positives, 4.5% of the
    designated positive set (#3697). The error ran in the flattering direction:
    an image reviewed as holding a `car` scored as a *confirmed* negative for
    `truck`, which is exactly the hard negative a detector has not been shown.

    ``reviewed_absent`` keeps what a review really did establish, per class: a
    verdict of absent on ``(image, class)`` admits that class's cells and no
    others. That matters because a **group** pass -- the #3588 negative pass
    asked "do you see NONE of these twenty-five?" -- legitimately answers for
    every member it named, and dropping human evidence wholesale would throw
    those away to fix the one-class case.

    ``reviewed_present`` is the guard the other way, and it is not hypothetical.
    A **boxless** ``present`` verdict -- "it is here, I cannot draw one box for
    it" -- makes :func:`apply_corrections` POP the class from ``labels``, so the
    image then looks to this function exactly like an image that does not hold
    it. On a COCO-anchored image that would admit the class as a *confirmed
    negative* on the strength of a human saying it is present. A pair a reviewer
    touched is never admitted by absence, whatever the anchor says.

    **The cells a class owns are read off ``cells``, never spelled.** This
    function serves two datasets that name their cells differently --
    ``vg_scale`` keys on ``class@band`` and ``vg_scale_deep`` on the bare class
    -- and the first cut of #3667 spelled ``scale_cell(c, band)`` inline. On the
    deep sibling that wrote 36 band-suffixed names that are not cells of that
    dataset into every COCO-exhaustive positive, including the image's **own**
    class at other bands: inert, because nothing matches them, but it is the
    #3156 guarantee stated backwards in a shipped pickle. Deriving the map from
    the caller's own cell list is what makes the rule dataset-agnostic.
    """
    if not cats:
        return list(cells) if iid in neg_set else []
    out = set(cats)
    if pc.SCALE_CROSS_CLASS_NEGATIVES:
        held = set(labels.get(iid, {}))
        anchored = iid in coco_scored
        for c, owned in _cells_by_class(cells).items():
            if c in held or (iid, c) in reviewed_present:
                continue
            if anchored or (iid, c) in reviewed_absent:
                out |= owned
    return sorted(out)


def scale_media(
    *,
    iid: int,
    data: bytes,
    filename: str,
    origin_name: str,
    box_dims: tuple[int, int],
    cats: list[str],
    boxes_for: dict[tuple[int, str], list[list[float]]],
    cells: list[str],
    neg_set: set[int],
    labels: dict[int, dict[str, list[list[float]]]],
    coco_scored: set[int],
    exhaustive: set[int],
    reviewed_absent: set[tuple[int, str]],
    reviewed_present: set[tuple[int, str]],
    embedder_name: str,
    importer: str,
) -> dict | None:
    """One scale-family media dict, or ``None`` if the bytes do not decode.

    **Single-sourced because comparability lives in the shape, not just the
    rule.** `coco_quarry` and `vg_scale` are only comparable if a cell means the
    same thing in both, and that is as true of `evaluable_categories` and
    `coco_scored` as it is of the band: a second copy of this dict would drift
    in a field nobody diffs. The same argument split :func:`band_for` out, and
    the loaders differ in the one place they genuinely must -- where the pixels
    come from. `vg_scale` reads a file per image; `coco_quarry` reads a member
    out of a staged zip (#3991).

    *data* is decoded header-only as a corruption check: a file that will not
    open here would fail later inside the embedder, where the failure is a
    stack trace in a six-hour job rather than one skipped image.
    """
    import io  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    try:
        with Image.open(io.BytesIO(data)) as im:
            vw, vh = im.size
    except Exception:  # noqa: BLE001 - a corrupt file just drops out
        return None
    if vw <= 0 or vh <= 0:
        return None

    # Normalised region boxes are resolution-independent, so they must be
    # divided by the size of the image the coordinates came from -- which is
    # the COCO original for a repaired image, not the VG copy carrying the
    # pixels.
    W, H = box_dims
    regions = [
        {"box": [b[0] / W, b[1] / H, b[2] / W, b[3] / H], "label": cell}
        for cell in cats
        for b in boxes_for.get((iid, cell), [])
    ]
    return {
        "id": iid,
        "media_type": "image",
        "embedder": embedder_name,
        "duration": 0,
        "file_size": 0,
        "md5": "",
        "embeddings": {},
        "media_bytes": data,
        "media_string": None,
        "filename": filename,
        "category": cats[0] if cats else "",
        "categories": cats,
        # A designated cell membership, not a closed world: a positive is
        # scorable only in the cells it was drawn for, and the shared
        # negatives are scorable everywhere.
        "evaluable_categories": _evaluable(
            iid, cats, cells, neg_set, labels, coco_scored, reviewed_absent, reviewed_present
        ),
        # Whether this image's labels rest on an exhaustive reference (COCO,
        # or a human who looked). False means VG's silence is the only
        # evidence of absence -- which is what the review slates target.
        "labels_exhaustive": iid in exhaustive,
        # The strict half of the flag above, and the one #3670's composition
        # is defined on. `labels_exhaustive` is also set by a human looking
        # at ONE class; this says COCO answered for all eighty at once, which
        # is what makes a negative provable rather than merely reviewed.
        "coco_scored": iid in coco_scored,
        "regions": regions,
        "origin": {"importer": importer, "params": {"embedder": embedder_name, "labels": "coco"}},
        "origin_name": origin_name,
    }
