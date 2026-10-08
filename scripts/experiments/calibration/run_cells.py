"""Calibration study: one SLURM-array task per (dataset, embedder, category, seed).

Each task runs *all* styles for that embedder inside it (they share the one
loaded pickle), so an embedder's arms are paired on identical data, sim/test
splits, and the identical startup exemplar.  Every style emits the #2781
calibration metrics (regret + its rule-inefficiency/calibration-shift
decomposition, threshold provenance, the degenerate flag) and the near-free
inclusion-budget sweep; the raw-patch tree style additionally re-pools its own
per-node scores under ``topk`` / ``pnorm`` (extra rows tagged ``pool_variant``).

Main rows -> ``results/cells/task_<idx>.csv``; the inclusion-budget sweep ->
``results/cells/task_<idx>__sweep.csv``; the #2836 cut decomposition ->
``__cutdiag.csv``; the #2865 cut-rule x inclusion sweep -> ``__cutincl.csv``.

Run directly with ``--index N`` for one cell, or via SLURM with
``$SLURM_ARRAY_TASK_ID``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402

import experiment_config as cfg  # noqa: E402


def _categories_by_dataset(prepare_info: dict) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    for ds, per_emb in prepare_info.get("datasets", {}).items():
        out[ds] = {emb: cfg.with_train_mixes(entry.get("selected_categories", [])) for emb, entry in per_emb.items()}
    return out


def ab_pair_keys(cells: list[dict]) -> set[tuple]:
    """The units ``analyze_ab.py`` pairs a trajectory A/B on, over *cells*.

    ``analyze_ab`` joins two grids on ``(arm, category, seed)`` with ``arm =
    dataset/embedder/style``.  So every style a task runs is its own paired
    cell: a patch embedder on a boxed dataset running ``whole_image,max_patch``
    gives two per ``(category, seed)``.  #3796's calibration-seed draws of one
    ``(category, seed)`` share a key and are averaged into one.  ``--print-cells``
    counts array tasks, which is neither.  It sizes the array; this sizes the
    A/B (preflight ``--resolve-delta``, #4111).
    """
    return {
        (c["dataset"], c["embedder"], style, c["category"], c["seed"])
        for c in cells
        for style in cfg.cell_styles(c["dataset"], c["embedder"])
    }


def paired_cell_count(prepare_info: dict) -> int:
    """How many paired cells an A/B of this grid against a same-shaped one holds."""
    return len(ab_pair_keys(cfg.array_cells(_categories_by_dataset(prepare_info))))


def _seed_query_text(ds: str, cat: str) -> str:
    """The text a user would type to find *cat* in *ds*, or "" if none is known.

    Delegates to :func:`experiment_config.seed_query_text`, the single
    implementation ``prepare_data.py`` filters on and ``preflight.sh`` checks
    against.  It used to be inlined here; a second copy of a lookup is how a
    preflight gate comes to pass while the run seeds differently.
    """
    return cfg.seed_query_text(ds, cat)


def _text_seed_vectors(ds: str, emb: str, medias: dict) -> tuple[dict, str]:
    """``(medias_to_rank, provenance)`` for the opening's cosine sort.

    For an ordinary embedder that is the cell's own medias -- one pickle, one
    space.  For a **paired** embedder (``siglip+dinov3_patch``) the opening runs
    in the text half's space, and there are two ways to have those vectors:

    * the media already carries them.  ``media["embeddings"]`` is a dict keyed
      by embedder name (three-slot embedders, ``docs/plans/patch-embedder.md``),
      so one media can hold a SigLIP vector and a DINOv3 vector at once.  This
      is the shape production would ship, and it needs no second file.
    * the pile stores one embedder per cell pickle, which is what it actually
      does today -- so the text half's pickle is opened and its vectors used.

    Preferring the first means the harness reads a multi-vector media correctly
    the day the pile writes one, instead of silently ranking against a stale
    side file.

    Either way the two must describe the same medias.  A seed sort covering a
    different set than the run walks is not the app's opening at all: the ids it
    ranks are the ids the autopilot steps through, so a media missing from the
    text side is a media the opening can never reach -- silently, in every
    column.  It is asserted per cell rather than assumed, because "the flag you
    passed is not the property you got" is how #2877, #2897 and #2905 each went
    wrong.
    """
    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

    if not cfg.is_paired(emb):
        return medias, "cell"

    text_emb = cfg.text_embedder(emb)
    probe = next(iter(medias.values()), None)
    if probe is not None and media_embedding(probe, text_emb) is not None:
        common.log(f"  paired opening: {text_emb} vectors already on the cell's medias")
        return medias, "multi_vector"

    from vtscore.config import EMBEDDINGS_DIR  # noqa: PLC0415

    from _cells_io import load_medias  # noqa: PLC0415

    text_pkl = EMBEDDINGS_DIR / cfg.text_pickle_name(ds, emb)
    if not text_pkl.exists():
        raise FileNotFoundError(
            f"paired embedder {emb!r} needs {text_emb} vectors: neither the cell's medias nor "
            f"{text_pkl.name} (which does not exist) supplies them"
        )
    text_medias = load_medias(text_pkl, repair=True)  # as the app holds them (#4095)
    missing = sorted(set(medias) - set(text_medias))
    if missing:
        raise ValueError(
            f"{text_pkl.name} is missing {len(missing)} of {len(medias)} medias in "
            f"{cfg.pickle_name(ds, emb)} (e.g. {missing[:5]}); the opening would rank "
            "a different set than the run walks"
        )
    common.log(f"  paired opening: ranked in {text_emb} space from {text_pkl.name}")
    return text_medias, text_pkl.name


def _text_seed_scores(ds: str, emb: str, cat: str, medias: dict) -> "dict[int, float] | None":
    """The app's text sort: cosine from the typed query to every media.

    This is what a real user starts from -- they type "boat" and vote down the
    ranking -- and it is what ``seed_scores`` has always been documented to hold
    (``al_strategies``, ``EVAL.md``, ``voting_iterations`` all say "similarity to
    the typed query").

    The query is embedded by the **text half** of the embedder name and scored
    against that half's media vectors, while the ids returned are the *cell's*
    ids -- so a paired arm hands the autopilot a SigLIP ranking of exactly the
    medias it will then learn about in DINOv3 space.

    Returns ``None`` when no query is defined for the cell, or when the text
    embedder has no text tower -- DINOv3 does not, so ``embed_text`` is the base
    class's ``return None`` and ``embed_text_query`` yields nothing.  ``None`` is
    the signal for the autopilot to seed from *three random known-good examples*
    instead, the app's other real start ("3 random examples pulled from the
    Good").  Both are things a user does; ranking by cosine to a cropped box is
    not, which is why this no longer seeds from crops.
    """
    from vtscore.embedding.helpers import embed_text_query  # noqa: PLC0415
    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

    text_emb = cfg.text_embedder(emb)
    text = _seed_query_text(ds, cat)
    if not text:
        return None
    qvec = embed_text_query(text, "image", enrich=cfg.SEED_ENRICH, embedder_name=text_emb)
    if qvec is None:
        return None

    def _unit(vec):
        v = np.asarray(vec, dtype=np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-12 else v

    ids = list(medias.keys())
    if not ids:
        return None
    vectors, _provenance = _text_seed_vectors(ds, emb, medias)
    # Named on the paired path, primary on the ordinary one.  Asking for the
    # name is what makes a wrong-space vector a KeyError-shaped None rather than
    # a plausible ranking built from the wrong embedder.
    rows = []
    for c in ids:
        vec = media_embedding(vectors[c], text_emb) if cfg.is_paired(emb) else media_embedding(vectors[c])
        if vec is None:
            raise ValueError(f"media {c} carries no {text_emb} vector for the opening of {ds}:{cat}")
        rows.append(_unit(vec))
    matrix = np.stack(rows)
    cos = matrix @ _unit(qvec)
    return {ids[k]: float(cos[k]) for k in range(len(ids))}


def check_declared_opening(ds: str, emb: str, cat: str, seed_mode: str) -> None:
    """Raise unless this cell opened the way the study said it would (#3278).

    Which start a cell takes is decided silently, by whether its (dataset,
    category) has a query and whether its embedder's text half has a tower.  So
    a grid mixing SigLIP and DINOv3 arms opens two different ways along one axis
    and nothing anywhere says so -- the confound
    ``lessons/2026-08-27-the-region-arm-could-not-open-the-way-the-app-does.md``
    describes.  ``CALIB_REQUIRE_OPENING`` is the study saying which opening it
    means; this is the per-cell half of enforcing it, beside preflight check 14's
    per-grid half.

    It raises rather than warning for the reason the paired-arm guard in
    :func:`main` does: a cell missing from the array is visible in any count of
    it, and a cell that ran under the wrong opening is not.  ``mixed`` (and unset) assert
    nothing -- a re-runner mirroring a completed grid legitimately holds both.
    """
    if cfg.REQUIRE_OPENING not in ("text", "known_good") or seed_mode == cfg.REQUIRE_OPENING:
        return
    raise RuntimeError(
        f"cell {ds}x{emb}:{cat} opened on {seed_mode!r} but this study declares "
        f"CALIB_REQUIRE_OPENING={cfg.REQUIRE_OPENING!r} "
        f"(query={_seed_query_text(ds, cat)!r}, text half={cfg.text_embedder(emb)})"
    )


def cell_progress(idx: int, n_launched: int, results: Path | None = None) -> str:
    """``cell i/N`` for the task log, qualified when the study was truncated (#3736).

    ``n_launched`` is ``len(cells)``: the grid this task recomputes from its own
    environment, which is the grid that was *launched*.  A study cut mid-run --
    ``scancel`` on the tail of the array plus a rewritten ``grid_shape.json`` --
    leaves every surviving task still reporting that larger number, so a reader
    watching a log sees a denominator no one intends to reach.  Job 622816 was
    cut from 3,750 cells to 1,875 and a peer session read ``cell 1496/3750`` as
    ~40% done; it was 76% done.  The number is well-formed, plausible, and wrong
    only in the direction that makes a run look less finished than it is, which
    is the direction that gets believed.

    ``results/grid_shape.json`` is the file that knows: ``launch_scale.sh``
    writes it and ``analyze_scale.py`` reads it for exactly this denominator, so
    the authoritative count already exists and the task log simply never asked.

    **The enumeration is not truncated to match, and must not be.** ``idx``
    indexes into ``cells`` to resolve ``(dataset, embedder, category, seed)``;
    shortening that list would shift the mapping and run the wrong cells
    silently.  Only the reported denominator is at issue, so only the report is
    qualified -- both numbers are named, and the launched one keeps its place.

    A missing, unreadable, or malformed shape file leaves the plain form: a log
    line is not worth failing a cell over, and a task run by hand from a bare
    results dir has no shape file by construction.
    """
    plain = f"cell {idx}/{n_launched}"
    shape = (common.RESULTS if results is None else results) / "grid_shape.json"
    try:
        n_shape = int(json.loads(shape.read_text())["n_cells"])
    except (OSError, ValueError, KeyError, TypeError):
        return plain
    if n_shape == n_launched:
        return plain
    verdict = "study truncated" if n_shape < n_launched else "grid_shape.json disagrees"
    return f"{plain} launched (grid_shape.json: {n_shape} — {verdict})"


def cell_file(path: Path) -> Path:
    """*path*, gzipped when ``CALIB_CELLS_GZIP=1`` (#4184).  pandas picks the codec off the suffix."""
    return path.with_name(path.name + ".gz") if cfg.CELLS_GZIP else path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibration: one cell (dataset,embedder,category,seed).")
    parser.add_argument("--index", type=int, default=None, help="Cell index; defaults to $SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--outdir", default=str(common.RESULTS / "cells"))
    parser.add_argument("--print-cells", action="store_true", help="Print the total cell count and exit.")
    parser.add_argument(
        "--print-paired-cells",
        action="store_true",
        help="Print how many cells analyze_ab.py pairs this grid on (each style its own cell) and exit.",
    )
    args = parser.parse_args(argv)

    prepare_info = json.loads((common.RESULTS / "prepare_info.json").read_text())
    cells = cfg.array_cells(_categories_by_dataset(prepare_info))

    if args.print_cells:
        print(len(cells))
        return 0
    if args.print_paired_cells:
        print(len(ab_pair_keys(cells)))
        return 0

    # CALIB_INDEX_OFFSET lets a second array reach cells past SLURM's MaxArraySize
    # (10100 on GRID): task i runs cell i + offset (#4159's overnight seeds).
    idx = (
        args.index
        if args.index is not None
        else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")) + int(os.environ.get("CALIB_INDEX_OFFSET", "0"))
    )
    if idx >= len(cells):
        common.log(f"index {idx} >= {len(cells)} cells; nothing to do")
        return 0
    cell = cells[idx]
    ds, emb, cat, seed = cell["dataset"], cell["embedder"], cell["category"], cell["seed"]
    # `None` on every grid but #3796's, where it is the swept axis.  Passed
    # through as-is: `simulate_voting_iterations` resolves None to the app's own
    # CALIBRATION_SPLIT_SEED, so the harness never writes that constant down.
    cal_seed = cell.get("calibration_seed")
    # A standalone trainer (#3959) has no head for a detection style to drive,
    # so it runs style-less: the whole-image path, which is the only one it has.
    styles = cfg.cell_styles(ds, emb)
    region_voting = cfg.region_voting_for(ds, emb)
    common.log(
        f"{cell_progress(idx, len(cells))}: dataset={ds} embedder={emb} "
        f"(learn={cfg.learn_embedder(emb)} text={cfg.text_embedder(emb)}) category={cat} seed={seed} "
        f"styles={styles} head={cfg.HEAD or 'default (production)'} safe_thresholds={cfg.SAFE_THRESHOLDS} "
        f"trainer={cfg.TRAINER} strategy={cfg.STRATEGY} standalone_cut={cfg.STANDALONE_CUT} "
        f"haystack_prevalence={cfg.HAYSTACK_PREVALENCE or 'natural'} "
        f"target_prevalence={cfg.TARGET_PREVALENCE or 'natural'} "
        f"calibrate_count={cfg.CALIBRATE_COUNT} fold_counts={cfg.FOLD_COUNTS or 'off'} "
        f"fold_count_schedule={cfg.FOLD_COUNT_SCHEDULE or 'off'} "
        f"sim_fraction={cfg.SIM_FRACTION} exclusion={cfg.exclusion_arm_name()} "
        f"cut_incl_ks={cfg.CUT_INCLUSION_KS or 'off'} live_cut_rule={cfg.LIVE_CUT_RULE or 'app default'} "
        f"live_threshold={cfg.LIVE_THRESHOLD or 'shipped'} "
        f"skyline_arms={cfg.SKYLINE_ARMS or 'off'} "
        f"acq_inclusion_offset={cfg.ACQ_INCLUSION_OFFSET} acq_rank_percentile={cfg.ACQ_RANK_PERCENTILE} "
        f"acq_p_crossing={cfg.ACQ_P_CROSSING} acq_origin={cfg.ACQ_ORIGIN} acq_target_p={cfg.ACQ_TARGET_P} smart_gate={cfg.SMART_GATE} label_quota={cfg.LABEL_QUOTA} "
        f"startup_schedule={cfg.STARTUP_SCHEDULE or 'app default'} "
        f"opening_diversity={cfg.OPENING_DIVERSITY or 'off'} "
        f"calibration_seed={cal_seed if cal_seed is not None else 'app pin'}"
    )

    import pandas as pd

    from vtscore.eval.voting_columns import (
        CALIBRATION_COLUMNS,
        CUT_DIAGNOSTIC_COLUMNS,
        CUT_INCLUSION_COLUMNS,
        FIT_QUALITY_ROW_COLUMNS,
        INCLUSION_SWEEP_COLUMNS,
        LINE_TEST_COLUMNS,
        PICK_COLUMNS,
        RANK_FRAME_COLUMNS,
    )
    from vtscore.eval.voting_iterations import simulate_voting_iterations

    from vtscore.config import EMBEDDINGS_DIR  # isort: skip

    from _cells_io import load_medias  # noqa: PLC0415

    pkl = EMBEDDINGS_DIR / cfg.pickle_name(ds, emb)
    medias: dict[int, dict] = load_medias(pkl, repair=True)  # as the app holds them (#4095)
    common.log(f"loaded {len(medias)} medias from {pkl}")

    seed_scores = _text_seed_scores(ds, emb, cat, medias)
    seed_mode = "text" if seed_scores is not None else "known_good"
    seed_query = _seed_query_text(ds, cat) if seed_scores is not None else ""
    seed_embedder = cfg.text_embedder(emb) if seed_scores is not None else ""
    if cfg.is_paired(emb) and seed_mode != "text":
        # A pair exists FOR the text sort.  Falling back to known-goods here
        # would run an arm that is identical to the bare learn embedder while
        # being labelled as something else -- a cell that looks like the
        # experiment and is not it.  Fail the cell instead: a missing cell is
        # visible and a mislabelled one is not.
        raise RuntimeError(
            f"paired embedder {emb!r} fell back to the known-good start for {ds}:{cat} "
            f"(query={_seed_query_text(ds, cat)!r}); the pair exists to take the text sort"
        )
    # After the pair guard, which says the same thing about a paired arm in more
    # useful words.
    check_declared_opening(ds, emb, cat, seed_mode)
    common.log(
        f"seed: mode={seed_mode} embedder={seed_embedder or '-'} query={seed_query!r} "
        f"declared={cfg.REQUIRE_OPENING or 'nothing'}"
    )

    all_rows: list[dict] = []
    all_sweep: list[dict] = []
    all_cutdiag: list[dict] = []
    all_cutincl: list[dict] = []
    all_fitq: list[dict] = []
    all_picks: list[dict] = []
    all_pframes: list[dict] = []
    all_rankframes: list[dict] = []
    all_linetests: list[dict] = []
    all_testscores: dict[str, dict] = {}
    for style in styles:
        variants = cfg.REPOOL_VARIANTS if style == cfg.REPOOL_STYLE else []
        sweep_local: list[dict] = []
        cutdiag_local: list[dict] = []
        cutincl_local: list[dict] = []
        fitq_local: list[dict] = [] if cfg.FIT_QUALITY else None
        picks_local: list[dict] | None = [] if cfg.EMIT_PICKS else None
        pframes_local: list[dict] | None = [] if cfg.PFRAME_STEPS else None
        rankframes_local: list[dict] | None = [] if cfg.RANK_FRAME_STEPS else None
        testscores_local: list[dict] | None = [] if cfg.SAVE_TEST_SCORES else None
        linetest_local: list[dict] | None = [] if cfg.LINE_TEST else None
        rows = simulate_voting_iterations(
            medias,
            target_category=cat,
            seed=seed,
            dataset_name=ds,
            inclusion=cfg.INCLUSION,
            sim_fraction=cfg.SIM_FRACTION,
            safe_thresholds=cfg.SAFE_THRESHOLDS,
            calibrate_count=cfg.CALIBRATE_COUNT,
            calibration_fraction=cfg.CALIBRATION_FRACTION,
            exclusion_min_remainder=cfg.EXCLUSION_MIN_REMAINDER,
            beta=cfg.BETA,
            live_cut_rule=cfg.LIVE_CUT_RULE,
            live_threshold=cfg.LIVE_THRESHOLD,
            region_voting=region_voting,
            max_steps=cfg.MAX_STEPS,
            seed_scores=seed_scores,
            trainer=cfg.TRAINER,
            strategy=cfg.STRATEGY,
            standalone_cut=cfg.STANDALONE_CUT,
            haystack_prevalence=cfg.HAYSTACK_PREVALENCE,
            target_prevalence=cfg.TARGET_PREVALENCE,
            head=cfg.HEAD,
            style=style,
            test_bands=cfg.TEST_BANDS,
            test_band_auroc=cfg.TEST_BAND_AUROC,
            train_mix=cfg.train_mix_for(cat),
            emit_calibration_metrics=True,
            repool_variants=variants,
            repool_topk=cfg.REPOOL_TOPK,
            inclusion_sweep_ks=cfg.INCLUSION_SWEEP_KS,
            sweep_sink=sweep_local,
            blend_schedule=cfg.BLEND_SCHEDULE,
            schedule_variants=cfg.SCHEDULE_VARIANTS,
            cut_diag_sink=cutdiag_local,
            fit_quality_sink=fitq_local,
            fit_quality_stride=cfg.FIT_QUALITY_STRIDE,
            anchored_thresholds=cfg.ANCHORED,
            anchored_weights=cfg.ANCHORED_WEIGHTS,
            anchored_rules=cfg.ANCHORED_RULES,
            anchored_fold_arms=cfg.ANCHORED_FOLD_ARMS,
            anchored_fold_combines=cfg.ANCHORED_FOLD_COMBINES,
            fold_count_variants=cfg.FOLD_COUNTS or None,
            skyline_arms=cfg.SKYLINE_ARMS or None,
            fold_count_schedule=cfg.FOLD_COUNT_SCHEDULE,
            cut_inclusion_ks=cfg.CUT_INCLUSION_KS or None,
            cut_inclusion_sink=cutincl_local,
            cut_inclusion_qtilt_steps=cfg.CUT_INCLUSION_QTILT_STEPS or None,
            acq_inclusion_offset=cfg.ACQ_INCLUSION_OFFSET,
            acq_rank_percentile=cfg.ACQ_RANK_PERCENTILE,
            acq_p_crossing=cfg.ACQ_P_CROSSING,
            acq_origin=cfg.ACQ_ORIGIN,
            acq_target_p=cfg.ACQ_TARGET_P,
            smart_gate=cfg.SMART_GATE,
            label_quota=cfg.LABEL_QUOTA,
            walk_picks=cfg.WALK_PICKS,
            walk_tol=cfg.WALK_TOL,
            walk_fine=cfg.WALK_FINE,
            walk_guard=cfg.WALK_GUARD,
            walk_shape=cfg.WALK_SHAPE,
            **({"spot_check": cfg.SPOT_CHECK} if cfg.SPOT_CHECK is not None else {}),
            **({"weak_separation": cfg.WEAK_D} if cfg.WEAK_D is not None else {}),
            **({"weak_min_t": cfg.WEAK_MIN_T} if cfg.WEAK_MIN_T is not None else {}),
            **({"weak_repeat": cfg.WEAK_REPEAT} if cfg.WEAK_REPEAT is not None else {}),
            **({"weak_phase": cfg.WEAK_PHASE} if cfg.WEAK_PHASE is not None else {}),
            startup_schedule=cfg.STARTUP_SCHEDULE,
            opening_diversity=cfg.OPENING_DIVERSITY,
            pick_sink=picks_local,
            precision_frame_sink=pframes_local,
            precision_frame_steps=cfg.PFRAME_STEPS or None,
            rank_frame_sink=rankframes_local,
            test_score_sink=testscores_local,
            line_test_sink=linetest_local,
            sim_size=cfg.SIM_SIZE,
            rank_frame_steps=cfg.RANK_FRAME_STEPS or None,
            calibration_seed=cal_seed,
        )
        # The recorded fraction is the one the run actually used: an explicit
        # CALIB_CALIBRATION_FRACTION pin verbatim, else the per-space default
        # the harness resolved for this cell's embedder (#3290) - recorded
        # through the same production table so the column can't drift from
        # what simulate_voting_iterations resolved.
        if cfg.CALIBRATION_FRACTION is not None:
            cell_calibration_fraction = cfg.CALIBRATION_FRACTION
        else:
            from vtscore.training.thresholds import production_split_for

            cell_calibration_fraction = production_split_for(patch_space=cfg.is_patch_embedder(emb))
        # The exclusion arm, recorded as BOTH a label and the number the floor
        # actually used (#3312).  Resolving the number here - through the app's
        # own `resolve_exclusion_floor` when the arm is the default - is what
        # lets the analyzer recompute, per step, whether the floor bound;
        # reading the arm label alone would leave `app(...)` un-numbered in a
        # frame concatenated across arms.
        from vtscore.training.thresholds import resolve_exclusion_floor

        exclusion_arm = cfg.exclusion_arm_name()
        exclusion_floor = resolve_exclusion_floor(cfg.EXCLUSION_MIN_REMAINDER)
        # The live cut rule, resolved to a NAME even on the default arm (#3557),
        # for the same reason as the exclusion floor above: a frame concatenated
        # across run-level arms has no other record of which rule cut it.
        from vtscore.training.thresholds import FOLD_ANCHOR_CUT_RULE

        live_cut_rule = cfg.LIVE_CUT_RULE or FOLD_ANCHOR_CUT_RULE
        # The retired live rule (#4184), named on every row for the same reason:
        # "shipped" on the default arm, so a pooled frame never reads a blank as
        # a rule.
        live_threshold = cfg.LIVE_THRESHOLD or "shipped"
        for r in rows:
            r["embedder"] = emb
            r["seed_mode"] = seed_mode
            r["seed_query"] = seed_query
            r["seed_embedder"] = seed_embedder
            r["calibration_fraction"] = cell_calibration_fraction
            r["sim_fraction"] = cfg.SIM_FRACTION
            r["exclusion_arm"] = exclusion_arm
            r["exclusion_min_remainder"] = exclusion_floor
            r["live_cut_rule"] = live_cut_rule
            r["live_threshold"] = live_threshold
            r["standalone_cut"] = cfg.STANDALONE_CUT
        for sr in sweep_local:
            sr["embedder"] = emb
        for dr in cutdiag_local:
            dr["embedder"] = emb
        for cr in cutincl_local:
            cr["embedder"] = emb
        for pr in picks_local or []:
            pr["embedder"] = emb
        for fr in fitq_local or []:
            fr["embedder"] = emb
        for rf in rankframes_local or []:
            rf["embedder"] = emb
        all_rows.extend(rows)
        all_sweep.extend(sweep_local)
        all_cutdiag.extend(cutdiag_local)
        all_cutincl.extend(cutincl_local)
        all_picks.extend(picks_local or [])
        all_pframes.extend({**f, "style": np.array(style or "")} for f in pframes_local or [])
        all_fitq.extend(fitq_local or [])
        all_rankframes.extend(rankframes_local or [])
        for lt in linetest_local or []:
            lt["embedder"] = emb
        all_linetests.extend(linetest_local or [])
        if testscores_local:
            # The last ordinary step and the last row (after the check): what Train left and what Find applies.
            # The full-label ceiling's snapshot (#4490) is kept apart, never read as a session's step; and
            # CALIB_SAVE_TEST_SCORES_AT keeps the first ordinary step at or past each listed click (``t<N>``).
            ceiling = [f for f in testscores_local if f["phase"] == "ceiling"]
            session = [f for f in testscores_local if f["phase"] != "ceiling"]
            ordinary = [f for f in session if f["phase"] != "check"]
            keep = {}
            if session:
                keep = {"last": ordinary[-1] if ordinary else session[-1], "final": session[-1]}
            for at in cfg.SAVE_TEST_SCORES_AT:
                hit = next((f for f in ordinary if int(f["t"]) >= at), None)
                if hit is not None:
                    keep[f"t{at}"] = hit
            if ceiling:
                keep["ceiling"] = ceiling[-1]
            all_testscores.update({f"{emb}/{style}/{k}": v for k, v in keep.items()})
        common.log(
            f"  style={style}: {len(rows)} rows, {len(sweep_local)} sweep rows, "
            f"{len(cutdiag_local)} cut-diagnostic rows, {len(cutincl_local)} cut-inclusion rows, "
            f"{len(fitq_local or [])} fit-quality rows"
        )

    outdir = common.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # `seed_embedder` joins `seed_mode`/`seed_query` for the same reason those
    # two exist: the #3156 rerun's root cause was that how a run started was
    # unnameable after the fact.  A paired arm's opening lives in a different
    # space than its `embedder` column implies, so the space has to be a column.
    # `calibration_fraction` joins them for the same reason (issue #3287): it is
    # a run-level knob, so the only other record of which arm a cell belongs to
    # is the directory it was read out of - and a frame that has been
    # concatenated across arms no longer has one.
    main_cols = [
        *CALIBRATION_COLUMNS,
        "embedder",
        "seed_mode",
        "seed_query",
        "seed_embedder",
        "calibration_fraction",
        "sim_fraction",
        "exclusion_arm",
        "exclusion_min_remainder",
        "live_cut_rule",
        "live_threshold",
        "standalone_cut",
    ]
    out = cell_file(outdir / f"task_{idx:04d}.csv")
    pd.DataFrame(all_rows, columns=pd.Index(main_cols)).to_csv(out, index=False)
    sweep_cols = [*INCLUSION_SWEEP_COLUMNS, "embedder"]
    sweep_out = cell_file(outdir / f"task_{idx:04d}__sweep.csv")
    pd.DataFrame(all_sweep, columns=pd.Index(sweep_cols)).to_csv(sweep_out, index=False)
    # The #2836 cut-decomposition frame (one row per step per fit geometry).
    cutdiag_cols = [*CUT_DIAGNOSTIC_COLUMNS, "embedder"]
    cutdiag_out = cell_file(outdir / f"task_{idx:04d}__cutdiag.csv")
    pd.DataFrame(all_cutdiag, columns=pd.Index(cutdiag_cols)).to_csv(cutdiag_out, index=False)
    # The #2865 cut-rule x inclusion frame (one row per step per arm per k).
    # Written unconditionally, like the frames above: an empty CSV with the
    # right header is what tells the analyzer the run had the sweep switched
    # off, rather than that its cells silently failed.
    cutincl_cols = [*CUT_INCLUSION_COLUMNS, "embedder"]
    cutincl_out = cell_file(outdir / f"task_{idx:04d}__cutincl.csv")
    pd.DataFrame(all_cutincl, columns=pd.Index(cutincl_cols)).to_csv(cutincl_out, index=False)
    # The #3267 per-click pick log.  Written unconditionally, like the frames
    # above, so an empty file with the right header says "the log was off"
    # rather than "the cell failed".
    picks_cols = [*PICK_COLUMNS, "embedder"]
    picks_out = cell_file(outdir / f"task_{idx:04d}__picks.csv")
    pd.DataFrame(all_picks, columns=pd.Index(picks_cols)).to_csv(picks_out, index=False)
    # The #3329 goodness-of-fit frame (one row per step per scope).  Same
    # unconditional-write rule as every frame above.
    fitq_cols = [*FIT_QUALITY_ROW_COLUMNS, "embedder"]
    fitq_out = cell_file(outdir / f"task_{idx:04d}__fitq.csv")
    pd.DataFrame(all_fitq, columns=pd.Index(fitq_cols)).to_csv(fitq_out, index=False)
    # The #4357 rank frames: where the positives sit in each ranking.  Same
    # unconditional-write rule as every CSV frame above.
    rankframes_cols = [*RANK_FRAME_COLUMNS, "embedder"]
    rankframes_out = cell_file(outdir / f"task_{idx:04d}__rankframes.csv")
    pd.DataFrame(all_rankframes, columns=pd.Index(rankframes_cols)).to_csv(rankframes_out, index=False)
    # The #4523 Test arm: one row per session.  Same unconditional-write rule:
    # an empty file with the header says the arm was off, not that the cell failed.
    linetest_cols = [*LINE_TEST_COLUMNS, "embedder"]
    linetest_out = cell_file(outdir / f"task_{idx:04d}__linetest.csv")
    pd.DataFrame(all_linetests, columns=pd.Index(linetest_cols)).to_csv(linetest_out, index=False)
    # The #4220 precision frames: arrays, not a table, so one npz per cell with
    # each frame's fields prefixed by its step (``t150/test_scores``).  Written
    # only when asked for - unlike the CSV side frames, an absent file here means
    # "off", because nothing reads it by default.
    if cfg.SAVE_TEST_SCORES and all_testscores:
        # #4452's Find scenarios: per kept snapshot its scores, labels and a JSON of the rest.
        ts_out = outdir / f"task_{idx:04d}__testscores.npz"
        packed_ts: dict[str, object] = {}
        for key, snap in all_testscores.items():
            arrays = {k: v for k, v in snap.items() if isinstance(v, np.ndarray)}
            for name, arr in arrays.items():
                packed_ts[f"{key}/{name}"] = arr
            meta = {k: v for k, v in snap.items() if k not in arrays}
            packed_ts[f"{key}/meta"] = np.array(json.dumps(meta))
        np.savez_compressed(ts_out, **packed_ts)
        common.log(f"wrote {len(all_testscores)} test-score snapshots to {ts_out}")
    if cfg.PFRAME_STEPS:
        pframes_out = outdir / f"task_{idx:04d}__pframes.npz"
        packed = {f"t{int(f['t'])}/{k}": v for f in all_pframes for k, v in f.items()}
        np.savez_compressed(pframes_out, **packed)
        common.log(f"wrote {len(all_pframes)} precision frames to {pframes_out}")
    common.log(
        f"wrote {len(all_rows)} rows to {out}, {len(all_sweep)} sweep rows to {sweep_out}, "
        f"{len(all_cutdiag)} cut-diagnostic rows to {cutdiag_out}, "
        f"{len(all_cutincl)} cut-inclusion rows to {cutincl_out}, "
        f"{len(all_picks)} pick rows to {picks_out}, "
        f"{len(all_fitq)} fit-quality rows to {fitq_out}, "
        f"and {len(all_rankframes)} rank frames to {rankframes_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
