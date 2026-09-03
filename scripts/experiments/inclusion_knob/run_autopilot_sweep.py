"""Selection-bias sweep: does the conformal Inclusion budget survive real voting?

Asks whether ``alpha(k) = 0.25 * 2^-k`` — a **split-conformal** guarantee, so it
assumes the calibration votes are exchangeable with the inference set — still
holds when the votes are chosen by the detector's own sort.  The comparison the
concluded study made (Autopilot vs. exchangeable random voting) is recorded in
[`SELECTION-BIAS.md`](../../../docs/experiments/2026-07-27-inclusion-knob/SELECTION-BIAS.md);
what this script is *for* now is measuring that budget under the **shipped**
detector, which is what the open items in
``docs/plans/inclusion-calibration-bias.md`` need.

**This is a driver, not a simulation.**  Every vote, fit, calibration and cut is
:func:`vtscore.eval.voting_iterations.simulate_voting_iterations`'s; this file
only builds a media dict from an arm's vectors, calls it once per (arm, seed),
and writes the two frames out.  That is deliberate and is the whole point of
issue #3408: the previous version hand-rebuilt the vote loop, and by the time
anyone looked it was measuring a detector nobody ships — the pre-#2877
parity-interleaved Hard/New order instead of the app's phase machine, the
retired ``"mlp"`` head instead of ``linear_svm``, no
``ACQUISITION_INCLUSION_OFFSET``, and a hardcoded ``calibration_fraction=0.5``
where production resolves 0.3 in a single-vector space.  Nothing under
``scripts/`` is covered by ``scripts/check-eval-app-sync.py``, so none of that
drift tripped a gate.  A driver cannot drift: it has no copy to go stale.

Consequently **this script does not reproduce the committed
``autopilot_sweep.csv``**, which is the 2026-07-30 run's record and stays as it
is.  The new frames are written under the ``autopilot_prod_`` prefix.

What the harness resolves, that the old loop got wrong (all of it by *not*
passing an argument — every default below is the app's):

* the Autopilot phase machine (``autopilot_fidelity``), via
  :class:`vtscore.eval.autopilot_flow.AutopilotFlow`;
* the shipped head, :data:`vtscore.eval.step_model.PRODUCTION_HEAD`;
* the acquisition cut, taken
  :data:`~vtscore.training.thresholds.ACQUISITION_INCLUSION_OFFSET` inclusion
  steps below the reporting cut, so the Hard phase measures ``|p - t|`` against
  the threshold the app actually selects on;
* ``calibration_fraction=None`` → ``production_split_for(patch_space=False)``;
* the safe blend, on by default since #3400.

Arms are the study's own: 4 AG News one-vs-rest categories on real E5 passage
embeddings (text-seeded, as a user typing a query would seed Autopilot) plus 3
synthetic separability levels (no text, so the selector takes its designed
random-known-good seed path).  ``whole_image`` is passed explicitly because both
side frames are gated on a resolved style; it *is* the binary geometry these
single-vector arms train in, and it is the same arm the calibration study runs
its binary cells under.

Two frames are written:

* ``autopilot_prod_steps.csv`` — one row per (arm, seed, step) at the reporting
  inclusion, from :data:`~vtscore.eval.voting_columns.CALIBRATION_COLUMNS`
  (filtered to the base ``gmm_variant``).  Carries ``threshold``,
  ``xcal_threshold``, ``oracle_threshold``, ``phase`` and the operating point.
* ``autopilot_prod_budget.csv`` — one row per (arm, seed, step, ``k``), from
  :data:`~vtscore.eval.voting_columns.INCLUSION_SWEEP_COLUMNS`.  ``alpha``,
  ``sweep_fnr`` and ``excess_fnr`` are this study's headline metric, computed by
  the harness rather than restated here.

Every step is emitted, not just the checkpoints: one trajectory yields them all
now that the loop is not being re-simulated per vote count, and the cold-start
question in ``docs/plans/inclusion-calibration-bias.md`` lives in the steps the
old checkpoint grid skipped.

Usage::

    python run_autopilot_sweep.py [--quick] [--steps-out CSV] [--budget-out CSV]
"""

from __future__ import annotations

import argparse
import itertools

import common

common.setup_env()

import numpy as np  # noqa: E402

SEEDS = range(4)
#: Inclusion stops the budget frame is swept at.  The study's own set, kept so
#: the new frames stay readable against the committed tables.
INCLUSIONS = (-10, -7, -5, -3, -1, 0, 1, 3, 5, 7, 10)
MAX_VOTES = 100
QUICK_MAX_VOTES = 24
SIM_FRACTION = 0.5  # half the items votable, half held out for metrics
#: The binary geometry.  Passed explicitly because both side frames are gated on
#: a resolved style, and a single-vector pool resolves none on its own.
STYLE = "whole_image"
#: The positive/negative category names the binarised arms are relabelled to.
TARGET, OTHER = "target", "other"
#: Text a user would plausibly type per AG News category, embedded with E5's
#: ``"query: "`` prefix to seed the Autopilot text sort (mirrors
#: :func:`vtscore.eval.seed_scores.build_seed_scores`).
AGNEWS_QUERIES = {
    "Business": "business and finance news",
    "Sci/Tech": "science and technology news",
    "Sports": "sports news",
    "World": "world news and international affairs",
}


def _load_arms() -> dict[str, tuple[np.ndarray, np.ndarray] | str]:
    """Arm name -> (X, y_binary) for AG News; synthetic arms resolve per-seed."""
    arms: dict = {}
    npz = common.CACHE / "agnews_e5.npz"
    if npz.exists():
        ag = np.load(npz, allow_pickle=True)
        X, y, cats = ag["X"], ag["y"], [str(c) for c in ag["categories"]]
        for ci, cat in enumerate(cats):
            arms[f"agnews:{cat}"] = (X, (y == ci).astype(np.int8))
    else:
        common.log(f"WARNING: {npz} missing - run prepare_agnews.py first; sweeping synthetic arms only")
    for level in ("easy", "medium", "hard"):
        arms[f"synth:{level}"] = level
    return arms


def _clips(X: np.ndarray, y: np.ndarray) -> dict[int, dict]:
    """A media dict over *X* — one single-vector media per row, ids from 1.

    The minimum a media needs to be votable: an embedder name, that embedder's
    vector, and the ``category`` the ground truth is read from.  No
    ``patch_grid``, so the harness resolves the single-vector production
    defaults (``whole_image`` geometry, the 0.3 Train/Calibrate split, the
    binary blend schedule).
    """
    return {
        i + 1: {
            "id": i + 1,
            "embedder": "e5",
            "embeddings": {"e5": X[i]},
            "category": TARGET if y[i] == 1 else OTHER,
        }
        for i in range(len(X))
    }


#: Per-arm text-sort ranking cache: the query embedding depends only on the
#: arm's category, so every seed reuses it instead of reloading E5.
_SEED_SCORE_CACHE: dict[str, dict[int, float]] = {}


def _agnews_seed_scores(X: np.ndarray, arm: str) -> dict[int, float] | None:
    """Cosine of every media to a real E5-embedded text query, or ``None``.

    Keyed by media id (``row + 1``, matching :func:`_clips`), which is what
    ``simulate_voting_iterations``' ``seed_scores`` is indexed by.

    Only AG News arms get a text sort: they are a text dataset, so a typed
    query is exactly how a user seeds Autopilot.  Synthetic arms have no text,
    which is the case the selector's random-known-good seed path exists for.
    """
    if not arm.startswith("agnews:"):
        return None
    if arm in _SEED_SCORE_CACHE:
        return _SEED_SCORE_CACHE[arm]
    category = arm.split(":", 1)[1]
    from sentence_transformers import SentenceTransformer

    from vtscore.config import E5_MODEL_ID

    model = SentenceTransformer(E5_MODEL_ID)
    q = model.encode(f"query: {AGNEWS_QUERIES[category]}", normalize_embeddings=True)
    sims = X @ np.asarray(q, dtype=np.float32)
    _SEED_SCORE_CACHE[arm] = {int(i) + 1: float(s) for i, s in enumerate(sims)}
    return _SEED_SCORE_CACHE[arm]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the autopilot selection-bias sweep.")
    parser.add_argument("--quick", action="store_true", help=f"1 seed, {QUICK_MAX_VOTES} votes (smoke test)")
    parser.add_argument("--steps-out", default=str(common.RESULTS / "autopilot_prod_steps.csv"))
    parser.add_argument("--budget-out", default=str(common.RESULTS / "autopilot_prod_budget.csv"))
    args = parser.parse_args(argv)

    import pandas as pd

    import synthetic
    from vtscore.embedding.loader import ensure_torch_configured
    from vtscore.eval.voting_iterations import simulate_voting_iterations

    ensure_torch_configured()

    seeds = [0] if args.quick else list(SEEDS)
    max_votes = QUICK_MAX_VOTES if args.quick else MAX_VOTES

    arms = _load_arms()
    step_rows: list[dict] = []
    budget_rows: list[dict] = []
    cells = list(itertools.product(sorted(arms), seeds))
    for ci, (arm, seed) in enumerate(cells):
        spec = arms[arm]
        X, y = synthetic.make_synthetic(spec, seed) if isinstance(spec, str) else spec
        sweep_sink: list[dict] = []
        with common.timed(f"{arm} seed={seed}"):
            rows = simulate_voting_iterations(
                _clips(X, y),
                target_category=TARGET,
                seed=seed,
                dataset_name=arm,
                sim_fraction=SIM_FRACTION,
                max_steps=max_votes,
                seed_scores=_agnews_seed_scores(X, arm),
                style=STYLE,
                emit_calibration_metrics=True,
                inclusion_sweep_ks=list(INCLUSIONS),
                sweep_sink=sweep_sink,
            )
        # The base row is the shipped cut; the other `gmm_variant`s are #2799's
        # alternative safe-threshold arms, which ride along whenever calibration
        # metrics are emitted and are not this study's question.
        base = [r for r in rows if not r.get("gmm_variant")]
        step_rows.extend(base)
        budget_rows.extend(sweep_sink)
        last = base[-1] if base else {}
        common.log(
            f"[{ci + 1}/{len(cells)}] {arm} seed={seed}: {len(base)} steps, {len(sweep_sink)} budget rows, "
            f"final phase={last.get('phase', '-')} good/bad={last.get('n_good', '-')}/{last.get('n_bad', '-')} "
            f"recall={last.get('recall', float('nan')):.3f}"
        )

    for frame, path, label in (
        (pd.DataFrame(step_rows), args.steps_out, "steps"),
        (pd.DataFrame(budget_rows), args.budget_out, "budget"),
    ):
        frame.to_csv(path, index=False)
        common.log(f"wrote {path}: {len(frame)} {label} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
