"""The read-only modes: ``--verify``, ``--rebuildable``, ``--bands``, ``--list``.

``--verify`` and ``--rebuildable`` are complements, and the two answer different
questions. ``--verify`` asks whether the cells on disk are usable;
``--rebuildable`` asks whether they could be produced again. A cell that loads
says nothing about whether it can be rebuilt -- the paths share no code -- which
is how the ``vg_box_*`` rebuild sat broken for eleven days behind a pile that
verified clean (#3297).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pile_config as pc

from pilebuild.env import cells_io, experiment_config, log
from pilebuild.loaders import loader_for


def label_problems(ds: str, medias: dict) -> list[str]:
    """Whether a boxed dataset's medias still carry their labels (#4117).

    Every boxed loader writes ``categories`` and ``regions`` on every media,
    negatives included (as empty lists). A cell without them loads, has the right
    media count, vectors and patch grids, and is wrong for every study that reads
    it: ``media_is_positive`` falls back to the single ``category``, and a region
    arm has no box to drag. A rebuild through the app's demo cache produced
    exactly that on all 4,193 ``visual_genome_m`` medias, and no other check here
    could see it -- the cross-cell media counts still agreed.
    """
    if not pc.DATASETS.get(ds, {}).get("boxed") or not medias:
        return []
    n = len(medias)
    bare = {f: sum(1 for m in medias.values() if m.get(f) is None) for f in ("categories", "regions")}
    out = [f"{k}/{n} medias carry no `{f}`" for f, k in bare.items() if k]
    if not out and not any(m.get("regions") for m in medias.values()):
        out.append(f"none of {n} medias carries a region box")
    return out


def coco_held_by() -> dict[int, list[str]]:
    """``VG image id -> the classes of C COCO annotates it with``, empty for none.

    Keyed on VG's ids because that is what a cell carries, and *absent* rather
    than empty for an image COCO never scored: "annotated and holds none" and
    "never annotated" are the two facts this whole pool rests on distinguishing,
    so they must not be the same value here either.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import coco_anchor as ca  # noqa: PLC0415

    try:
        image_data, instances = ca.ensure_sources(pc.PILE / "coco_anchor", fetch=False)
    except SystemExit:
        # An empty map, not an exit: every caller draws only from images COCO
        # answered for, so with no sources it draws nothing rather than guessing.
        log("NOTE: coco_anchor sources are not staged, so no image has a COCO answer")
        return {}
    truth = ca.coco_truth(instances, set(pc.SCALE_CLASSES))
    with image_data.open() as fh:
        coco_of = {int(m["image_id"]): int(m["coco_id"]) for m in json.load(fh) if m.get("coco_id")}
    return {i: sorted(c for c, boxes in truth[cid].items() if boxes) for i, cid in coco_of.items() if cid in truth}


def verify() -> int:
    """Load every present cell and check it is usable. Returns an exit code."""
    io = cells_io()
    problems: list[str] = []
    rows = []
    counts_by_dataset: dict[str, dict[str, int]] = defaultdict(dict)
    for ds, emb in pc.cells():
        path = pc.cell_path(ds, emb)
        if not path.exists():
            rows.append((ds, emb, "MISSING", "", "", ""))
            continue
        # Raw, never repaired: this is the audit of what the pile STORES.
        medias = io.load_medias(path, repair=False)
        n = len(medias)
        counts_by_dataset[ds][emb] = n
        n_patch = sum(1 for m in medias.values() if m.get("patch_grid") is not None)
        first = next(iter(medias.values()), None)
        dim = ""
        if first is not None:
            from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

            vec = media_embedding(first)
            dim = str(len(vec)) if vec is not None else "NO-VECTOR"
        want_region = pc.region_capable(ds, emb)
        # The app holds every vector unit-norm; a cell that does not is a cell
        # whose harness runs measured a detector nobody ships (#4095, #4099).
        from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

        norms = [
            float(np.linalg.norm(np.asarray(v, dtype=np.float32)))
            for m in list(medias.values())[:200]
            if (v := media_embedding(m)) is not None
        ]
        off_unit = [x for x in norms if abs(x - 1.0) > io.UNIT_NORM_TOL]
        unlabelled = label_problems(ds, medias)
        state = "ok"
        if n == 0:
            state = "EMPTY"
            problems.append(f"{ds} x {emb}: 0 medias")
        elif unlabelled:
            state = "NO-LABELS"
            problems += [f"{ds} x {emb}: {u}" for u in unlabelled]
        elif dim in ("", "NO-VECTOR"):
            state = "NO-VECTOR"
            problems.append(f"{ds} x {emb}: medias carry no embedding")
        elif want_region and n_patch < n:
            state = "PATCH-GAP"
            problems.append(f"{ds} x {emb}: region-capable but patch_grid on only {n_patch}/{n}")
        elif not pc.is_patch_embedder(emb) and n_patch:
            state = "UNEXPECTED-PATCH"
            problems.append(f"{ds} x {emb}: single-vector embedder carries patch grids")
        elif off_unit:
            state = "NOT-UNIT"
            problems.append(
                f"{ds} x {emb}: {len(off_unit)}/{len(norms)} sampled vectors not unit-norm "
                f"(e.g. {off_unit[0]:.2f}); normalise the cell"
            )
        rows.append((ds, emb, state, str(n), f"{n_patch}/{n}", dim))

    # A dataset's cells must all cover the same medias, or cross-embedder
    # comparisons silently compare different populations. This is not
    # hypothetical: a datadir missing its demo-source symlink sent the loader
    # off to re-download the dataset, and it embedded a truncated 1662-media
    # subset of a 4193-media dataset into a cell that otherwise looked healthy.
    for ds, per_emb in counts_by_dataset.items():
        if len(set(per_emb.values())) > 1:
            majority = max(set(per_emb.values()), key=list(per_emb.values()).count)
            odd = {e: n for e, n in per_emb.items() if n != majority}
            problems.append(
                f"{ds}: cells disagree on media count (most are {majority}); "
                f"rebuild {', '.join(f'{e} ({n})' for e, n in sorted(odd.items()))}"
            )

    log(f"{'dataset':18s} {'embedder':14s} {'state':16s} {'medias':>7s} {'patch':>12s} {'dim':>6s}")
    for ds, emb, state, n, patch, dim in rows:
        log(f"{ds:18s} {emb:14s} {state:16s} {n:>7s} {patch:>12s} {dim:>6s}")

    # Coverage is not implied by anything above: a cell can be structurally
    # perfect and no longer contain the images a human reviewed. Reported here
    # so a rebuild cannot be declared healthy without it being looked at.
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import check_review_coverage  # noqa: PLC0415

        if pc.cell_path("vg_scale", "siglip").exists():
            log("")
            log("review coverage:")
            sys.argv = ["check_review_coverage"]
            if check_review_coverage.main() != 0:
                problems.append("vg_scale: the rebuild retired images that had been reviewed")
    except Exception as exc:  # noqa: BLE001 - an absent review is not a build failure
        log(f"  (review-coverage check skipped: {exc})")

    # The human record is the one input a rebuild cannot regenerate, and it
    # lives on the same purgeable mount as the cells (#3729). Checked here
    # because this is where a pile is declared healthy, and an uncommitted
    # verdict is the kind of loss nobody notices until the mount is cleared.
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import verdict_store  # noqa: PLC0415

        log("")
        log("human record:")
        if verdict_store.do_check(strict=False) != 0:
            problems.append("the human record on disk is not in the repository -- run `verdict_store.py export`")
    except Exception as exc:  # noqa: BLE001 - a missing store is not a build failure
        log(f"  (human-record check skipped: {exc})")

    if problems:
        log("")
        for p in problems:
            log(f"PROBLEM: {p}")
        return 1
    log("all present cells verified")
    return 0


def rebuildable(datasets: list[str] | None = None) -> int:
    """Exercise every dataset's *selection* step without embedding anything.

    The pile documents itself as purgeable -- ``pile_config``: "every cell must
    be rebuildable from sources that are **not** on scratch". Nothing checked
    that. ``--verify`` loads the built cells, and a cell that loads says
    nothing about whether it can be rebuilt: the two paths share no code, so a
    rebuild path can rot for months behind a pile that verifies clean. It did
    (#3297): a scan-format change on 2026-08-17 outran the scan file the
    ``vg_box_*`` cells were selected from, and the break surfaced only when
    somebody asked for a rebuild eleven days later.

    So this is the canary that would have caught it the same day, for a few
    seconds. It runs the part of each build that reads sources and decides
    *what goes in the cell*, and skips the part that costs GPU-hours.

    Each dataset kind answers for itself, through the ``check()`` of the very
    module whose ``load()`` builds it (:mod:`pilebuild.loaders`). That identity
    is the point rather than tidiness: when this check spelled its own source
    paths it named ``COCO_IMAGES`` while the builder opened ``val2017.zip``, and
    reported ``coco_val`` REBUILD-BROKEN against a staging area that was present
    and fine (#3299). A canary that names a different path than the build is not
    a canary.

    A banded dataset goes one question further, via
    :func:`pilebuild.loaders.vg_band._vocab_drift`: not only "would a rebuild
    run?" but "would a rebuild produce *this*?". A repair that restores the
    former while quietly changing the latter is the expensive kind, and it is
    invisible from the built cell.

    Deliberately does not parse the multi-GB sources (VG's ``objects.json``,
    the COCO zip). A canary nobody runs is worth nothing, and the way to make
    it run is to keep it cheap enough to sit in front of every build.
    """
    for bad in [d for d in (datasets or []) if d not in pc.DATASETS]:
        raise SystemExit(f"unknown dataset {bad!r}; known: {sorted(pc.DATASETS)}")
    wanted = list(datasets) if datasets else list(pc.DATASETS)

    problems: list[str] = []

    for ds in wanted:
        try:
            ok = loader_for(ds, pc.DATASETS[ds].get("kind")).check(ds)
            log(f"  {ds:18s} ok       {ok}")
        except SystemExit as exc:  # the loaders' own way of reporting a bad source
            log(f"  {ds:18s} BROKEN   {exc}")
            problems.append(f"{ds}: {exc}")

    if problems:
        log(f"{len(problems)} dataset(s) CANNOT be rebuilt from their sources:")
        for p in problems:
            log(f"REBUILD-BROKEN: {p}")
        return 1
    log("every dataset's selection step runs against its current sources")
    return 0


def report_bands() -> int:
    """Report voted-box scale-band populations for each boxed dataset.

    The bands are anchored to the patch embedder's geometry: ``sub_patch`` is
    "smaller than one DINOv3 patch", i.e. below what the patch grid can resolve
    at all. That anchoring is the band's whole meaning, so a thin ``sub_patch``
    is a fact about the data, not a threshold to tune — widening the edge would
    inflate the count with objects that *are* resolvable.

    Reads the smallest available cell for each dataset: scale stats need only
    ``regions``, which every cell carries, so there is no reason to page in the
    multi-GB patch cell.
    """
    io = cells_io()
    cfg = experiment_config()
    from vtscore.eval.labels import category_scale_stats  # noqa: PLC0415

    boxed = [ds for ds, info in pc.DATASETS.items() if info.get("boxed")]
    if not boxed:
        log("no boxed datasets in the pile; nothing to stratify")
        return 0

    for ds in boxed:
        present = [(pc.cell_path(ds, e).stat().st_size, e) for e in pc.EMBEDDERS if pc.cell_path(ds, e).exists()]
        if not present:
            log(f"{ds}: no cells present")
            continue
        _, emb = min(present)
        medias = io.load_medias(pc.cell_path(ds, emb))

        counts: dict[str, int] = defaultdict(int)
        for m in medias.values():
            for c in m.get("categories") or [m.get("category")]:
                if c:
                    counts[c] += 1

        selected, report = cfg.select_categories_by_scale(medias, dict(counts))
        log("")
        log(f"=== {ds}: {len(medias)} medias, {len(counts)} categories (via {emb}) ===")
        dropped = report.get("dropped_above_max_voted_area") or []
        log(f"  dropped above max_voted_area={report.get('max_voted_area')}: {len(dropped)}")
        for name, info in (report.get("bands") or {}).items():
            lo, hi = info["range"]
            flag = "  ** UNDER-POPULATED **" if info["under_populated"] else ""
            log(
                f"  {name:14s} [{lo * 100:5.2f}%, {hi * 100:6.2f}%): "
                f"{len(info['selected'])}/{info['target']} of {info['n_candidates']} candidates{flag}"
            )
            log(f"      {info['selected']}")

        # When a band is starved, say whether the min-count filter is even the
        # binding constraint. Measured on the first run it was not: the
        # sub_patch pool held 5 categories (VG) and 1 (COCO) at every
        # min_count from 5 to 30, so lowering it recovers nothing.
        starved = [n for n, i in (report.get("bands") or {}).items() if i["under_populated"]]
        if starved:
            stats = {c: s for c in counts if (s := category_scale_stats(medias, c)) is not None}
            for name in starved:
                lo, hi = report["bands"][name]["range"]
                pools = {
                    mc: sum(1 for c, s in stats.items() if counts[c] >= mc and lo <= s["voted_area"] < hi)
                    for mc in (5, 10, 20, 30)
                }
                spread = "same at every min_count" if len(set(pools.values())) == 1 else str(pools)
                log(f"  {name}: candidate pool by min category count -> {spread}")
        log(f"  -> selected {len(selected)} categories")
    return 0


def list_cells() -> None:
    log(f"pile: {pc.PILE}")
    for ds, emb in pc.cells():
        path = pc.cell_path(ds, emb)
        # An `on_request` cell that is absent is absent BY DESIGN, and reading
        # it as MISSING is how a foot-gun gets "fixed" by building it.
        on_request = pc.DATASETS.get(ds, {}).get("on_request")
        mark = "present" if path.exists() else ("on-request" if on_request else "MISSING")
        size = f"{path.stat().st_size / 1e6:8.0f} MB" if path.exists() else " " * 11
        region = " region-voting" if pc.region_capable(ds, emb) else ""
        log(f"  {ds:18s} x {emb:14s} {mark:8s} {size}{region}")
