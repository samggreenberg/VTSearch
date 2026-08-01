"""Stage A of the threshold-stability study (#2790): frozen-trace threshold replay.

Records nothing itself — it *consumes* a run's ``--labeling-trace`` output plus the
run's npz feature cache, and at every labeling step ``t`` re-derives the decision
threshold under each candidate rule (``argmin`` / ``conformal`` / ``rank-transfer``,
optionally med3-smoothed) over the **frozen** vote set at that step. Because the
votes are frozen, arms and RNG seeds are exactly paired, so the step-to-step
variance decomposes:

* variance across **fold-split seeds** at fixed ``t`` → split noise (plan S2/S4);
* variance across **trainer seeds** at fixed ``(t, split)`` → fold-fit noise (S3);
* ``Δthreshold`` across adjacent ``t`` at fixed seeds → label-increment sensitivity;
* rule differences at identical inputs → the fidelity gap (S1: argmin vs the
  conformal rule production Autopilot actually runs).

Replay cannot see selection feedback (S5) — the vote *sequence* is fixed by the
recorded trace — so it brackets the effect together with the live-loop Stage B; it
is the cheap, exactly-paired half.

**Two layers, so the science is testable without the cache.** The pure core
(:func:`reconstruct_vote_sets`, :func:`replay_step_thresholds`,
:func:`decompose_variance`) is model-/cache-free and unit-tested on synthetic
traces. The cache glue (:class:`CacheVectorLoader`) reads the run's npz feature
cache written by ``scripts/sod/features.py`` and is the one piece that must be
validated against a real cache on the Grid (there is no cache on a laptop). It is
injected into the core as a ``vote_vectors`` callable so the two never entangle.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# scripts/sod is import-pathed as a package sibling of vtscore on the sweep venv.
from vtscore.eval.threshold_rules import RULES, calibrated_threshold, median_smooth, rank_transfer_cut

#: A per-step frozen vote set: the good/bad image ids accumulated through step t.
VoteSet = tuple[list[int], list[int]]

#: Loader contract: given the good/bad image ids voted so far, return the training
#: matrices ``(X, y)`` exactly as the live loop's ``_train_pool_head`` builds them
#: for the whole/box-pool path (pos = each good image's exemplar rows, neg = each
#: bad image's whole vector), plus the final-model pool scores for rank-transfer.
VoteVectorsFn = Callable[[list[int], list[int]], "StepVectors | None"]


@dataclass
class StepVectors:
    """Frozen training inputs for one replay step."""

    X: np.ndarray
    y: np.ndarray
    #: The whole-image score pool for the current model over the ranking pool,
    #: used only by ``rank-transfer`` (may be empty → rank-transfer == conformal).
    final_pool_scores: np.ndarray = field(default_factory=lambda: np.empty(0))


def reconstruct_vote_sets(trace_rows: Sequence[dict]) -> list[tuple[int, VoteSet]]:
    """Accumulate the frozen ``(good_ids, bad_ids)`` vote set at each labeled step.

    The recorded trace has one row per labeled item, in labeling order, each with
    an ``image_id`` and a ``gt_label`` (``"good"``/``"bad"`` or 1/0). One item is
    revealed per step, so the vote set at step ``t`` is every item labeled at rows
    ``<= t``. Returns ``[(t, (good_ids, bad_ids)), ...]`` in step order. Rows are
    read in the order given (the trace is already ordered); a missing/duplicate
    ``image_id`` is tolerated (dedup keeps first vote for an id).
    """
    good: list[int] = []
    bad: list[int] = []
    seen: set[int] = set()
    out: list[tuple[int, VoteSet]] = []
    for row in trace_rows:
        iid = int(row["image_id"])
        label = row.get("gt_label")
        is_good = label in (1, 1.0, "good", "1", "True", True)
        if iid not in seen:
            seen.add(iid)
            (good if is_good else bad).append(iid)
        t = int(row.get("t", len(out)))
        out.append((t, (list(good), list(bad))))
    return out


def replay_step_thresholds(
    vote_sets: Sequence[tuple[int, VoteSet]],
    vote_vectors: VoteVectorsFn,
    trainer_fn_factory: Callable[[], Callable],
    *,
    rules: Sequence[str] = RULES,
    fold_seeds: Sequence[int] = tuple(range(10)),
    trainer_seeds: Sequence[int] = (0,),
    inclusion_value: int = 0,
    calibrate_count: int = 2,
    cal_fraction: float = 0.5,
    smooth: str = "none",
) -> list[dict]:
    """Recompute the threshold at every step × rule × fold-seed × trainer-seed.

    ``vote_vectors(good_ids, bad_ids)`` supplies the frozen ``StepVectors`` for a
    step (``None`` → the step is skipped, e.g. a cold-start step with no trained
    head). ``trainer_fn_factory()`` returns a fresh ``trainer_fn`` (``(X, y, seed)
    -> predict``); the ``argmin``/``conformal`` fold machinery drives it.
    ``fold_seeds`` vary the split RNG (isolating split noise); ``trainer_seeds``
    are folded into the fold RNG offset (isolating fit noise). With
    ``smooth="med3"`` each (rule, fold_seed, trainer_seed) trajectory is passed
    through :func:`median_smooth` over its own raw-threshold history.

    Returns one row per (t, rule, fold_seed, trainer_seed) with ``threshold`` (raw)
    and ``threshold_smoothed``; :func:`decompose_variance` aggregates these.
    """
    # Per-trajectory raw-threshold history, keyed by (rule, fold_seed, trainer_seed).
    history: dict[tuple[str, int, int], list[float]] = {}
    rows: list[dict] = []
    for t, (good_ids, bad_ids) in vote_sets:
        sv = vote_vectors(good_ids, bad_ids)
        if sv is None:
            continue
        for rule in rules:
            for fs in fold_seeds:
                for ts in trainer_seeds:
                    trainer_fn = trainer_fn_factory()
                    raw = calibrated_threshold(
                        sv.X,
                        sv.y,
                        trainer_fn,
                        seed=fs * 1000 + ts,
                        rule=rule,
                        inclusion_value=inclusion_value,
                        calibrate_count=calibrate_count,
                        cal_fraction=cal_fraction,
                    )
                    if rule == "rank-transfer" and sv.final_pool_scores.size:
                        # Map the conformal cut into the final model's score pool.
                        pooled = _pooled_fold_scores(
                            sv, trainer_fn_factory(), fs * 1000 + ts, calibrate_count, cal_fraction
                        )
                        raw = rank_transfer_cut(raw, pooled, sv.final_pool_scores)
                    key = (rule, fs, ts)
                    hist = history.setdefault(key, [])
                    hist.append(float(raw))
                    smoothed = median_smooth(hist) if smooth == "med3" else float(raw)
                    rows.append(
                        {
                            "t": t,
                            "rule": rule,
                            "fold_seed": fs,
                            "trainer_seed": ts,
                            "threshold": float(raw),
                            "threshold_smoothed": float(smoothed),
                            "n_good": len(good_ids),
                            "n_bad": len(bad_ids),
                        }
                    )
    return rows


def _pooled_fold_scores(
    sv: StepVectors, trainer_fn: Callable, seed: int, calibrate_count: int, cal_fraction: float
) -> list[float]:
    """Pooled held-out fold scores for rank-transfer's source distribution."""
    from vtscore.eval.threshold_rules import stratified_fold_orderings  # noqa: PLC0415

    orderings = stratified_fold_orderings(
        sv.X, sv.y, trainer_fn, seed, calibrate_count=calibrate_count, cal_fraction=cal_fraction
    )
    return [s for scores, _ in orderings for s in scores]


def decompose_variance(rows: Sequence[dict], *, warmup_t: int = 20, spike_delta: float = 0.1) -> list[dict]:
    """Aggregate replay rows into the pre-registered per-(rule, t) variance stats.

    For each (rule, t) past ``warmup_t`` (the GMM ramp), reports:

    * ``sd_threshold`` — spread of the raw threshold across all fold/trainer seeds
      (the paired threshold noise the study wants shrunk);
    * ``mean_threshold`` / ``mean_threshold_smoothed``;
    * ``spike_rate`` — fraction of seeds whose ``|Δthreshold|`` from the previous
      ``t`` exceeds ``spike_delta`` (single-step excursions, #2790's symptom).

    ``Δthreshold`` is computed within a (fold_seed, trainer_seed) trajectory so it
    measures label-increment sensitivity, not cross-seed spread.
    """
    import statistics  # noqa: PLC0415

    by_rt: dict[tuple[str, int], list[dict]] = {}
    for r in rows:
        by_rt.setdefault((r["rule"], r["t"]), []).append(r)

    # Previous-t threshold per trajectory, for Δ / spike computation.
    prev: dict[tuple[str, int, int], float] = {}
    out: list[dict] = []
    for (rule, t), group in sorted(by_rt.items()):
        thrs = [g["threshold"] for g in group]
        spikes = 0
        for g in group:
            key = (g["rule"], g["fold_seed"], g["trainer_seed"])
            if key in prev and abs(g["threshold"] - prev[key]) > spike_delta:
                spikes += 1
            prev[key] = g["threshold"]
        if t < warmup_t:
            continue
        out.append(
            {
                "rule": rule,
                "t": t,
                "n_seeds": len(group),
                "mean_threshold": statistics.fmean(thrs),
                "sd_threshold": statistics.pstdev(thrs) if len(thrs) > 1 else 0.0,
                "mean_threshold_smoothed": statistics.fmean([g["threshold_smoothed"] for g in group]),
                "spike_rate": spikes / len(group),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Cache glue — the one Grid-validated layer (no cache exists on a laptop).
# --------------------------------------------------------------------------- #


class CacheVectorLoader:
    """Reads frozen vote vectors from a run's npz feature cache (whole path).

    Mirrors ``_train_pool_head``'s whole/box-pool assembly using the npz files
    ``scripts/sod/features.py::FeatureCache`` wrote: a **good** image contributes
    its exemplar rows (``exemplars/<ds>/<class_slug>/<embedder>/<slug>/<id>_*.npz``,
    globbed so the gt-box-hash suffix need not be recomputed), a **bad** image
    contributes its whole vector (``regions/<ds>/<embedder>/<slug>/<id>.npz`` →
    ``whole_vec``). Reads only cache hits (no image pixels, no embedder, no GPU),
    so it runs on the Grid over the #2790 SigLIP2/whole cache as-is.

    VALIDATE-ON-GRID: the exact pooling in ``_train_pool_head`` (does a good vote
    use all exemplar rows or the first?) and the ``final_pool_scores`` source must
    be confirmed against one real cell before a full launch; this loader implements
    the documented whole-path shape and is deliberately isolated so that check is a
    one-file edit. See docs REPORT for the checklist.
    """

    def __init__(self, cache_dir: Path, dataset: str, embedder: str, class_slug: str, proposal_slug: str = "whole"):
        self.regions_dir = Path(cache_dir) / "regions" / dataset / embedder / proposal_slug
        self.exem_dir = Path(cache_dir) / "exemplars" / dataset / class_slug / embedder / proposal_slug

    def _whole_vec(self, image_id: int) -> np.ndarray | None:
        p = self.regions_dir / f"{image_id}.npz"
        if not p.exists():
            return None
        with np.load(p) as z:
            return np.asarray(z["whole_vec"], dtype=np.float64)

    def _exemplars(self, image_id: int) -> np.ndarray | None:
        matches = sorted(self.exem_dir.glob(f"{image_id}_*.npz"))
        if not matches:
            return None
        with np.load(matches[0]) as z:
            return np.asarray(z["exemplars"], dtype=np.float64)

    def __call__(self, good_ids: list[int], bad_ids: list[int]) -> StepVectors | None:
        pos_rows = [self._exemplars(i) for i in good_ids]
        neg_rows = [self._whole_vec(i) for i in bad_ids]
        pos = [r for r in pos_rows if r is not None]
        neg = [r.reshape(1, -1) for r in neg_rows if r is not None]
        if not pos or not neg:
            return None
        X = np.concatenate(pos + neg, axis=0)
        y = np.concatenate([np.ones(sum(len(r) for r in pos)), np.zeros(len(neg))])
        return StepVectors(X=X, y=y)


def _read_trace(seed_dir: Path) -> list[dict]:
    tj = seed_dir / "trace.json"
    if tj.exists():
        return json.loads(tj.read_text())
    raise FileNotFoundError(f"no trace.json under {seed_dir}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Replay decision thresholds over a frozen labeling trace (#2790).")
    ap.add_argument("--trace-dir", required=True, help="A .../labeling_trace/<slug>/<config>/seed<N> dir.")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--embedder", required=True)
    ap.add_argument("--class-slug", required=True)
    ap.add_argument("--proposal-slug", default="whole")
    ap.add_argument("--rules", default=",".join(RULES))
    ap.add_argument("--fold-seeds", type=int, default=10)
    ap.add_argument("--trainer-seeds", type=int, default=10)
    ap.add_argument("--inclusion", type=int, default=0)
    ap.add_argument("--calibrate-count", type=int, default=2)
    ap.add_argument("--smooth", choices=["none", "med3"], default="none")
    ap.add_argument("--out", required=True, help="Output CSV path for the decomposition.")
    args = ap.parse_args(argv)

    from vtscore.training.mlp import train_model  # noqa: PLC0415

    def trainer_fn_factory():
        def trainer_fn(X, y, seed):  # noqa: ARG001 - signature fixed by TrainerFn
            import torch  # noqa: PLC0415

            model = train_model(
                torch.tensor(np.asarray(X), dtype=torch.float32),
                torch.tensor(np.asarray(y), dtype=torch.float32).unsqueeze(1),
                X.shape[1],
            )

            def predict(Xc):
                with torch.no_grad():
                    return model(torch.tensor(np.asarray(Xc), dtype=torch.float32)).squeeze(-1).numpy()

            return predict

        return trainer_fn

    trace = _read_trace(Path(args.trace_dir))
    vote_sets = reconstruct_vote_sets(trace)
    loader = CacheVectorLoader(Path(args.cache_dir), args.dataset, args.embedder, args.class_slug, args.proposal_slug)
    rows = replay_step_thresholds(
        vote_sets,
        loader,
        trainer_fn_factory,
        rules=[r for r in args.rules.split(",") if r],
        fold_seeds=range(args.fold_seeds),
        trainer_seeds=range(args.trainer_seeds),
        inclusion_value=args.inclusion,
        calibrate_count=args.calibrate_count,
        smooth=args.smooth,
    )
    agg = decompose_variance(rows)
    _write_csv(Path(args.out), agg)
    print(f"replayed {len(vote_sets)} steps -> {len(rows)} rows -> {len(agg)} (rule,t) cells -> {args.out}")
    return 0


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
