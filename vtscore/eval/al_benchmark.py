"""Hermetic benchmark harness for active-learning acquisition strategies.

Wires the acquisition strategies in :mod:`vtscore.eval.al_strategies` to the
voting-iterations evaluator (:mod:`vtscore.eval.voting_iterations`) over
**data sources that need no model downloads**, so a strategy sweep runs in the
plain test container:

- ``synthetic`` — Gaussian category clusters generated on the fly.  Fully
  parameterised (item count, dimensionality, class count, separation, noise) and
  seeded, so a run is reproducible and its difficulty is a dial.
- ``precomputed`` — feature vectors + string labels supplied directly (in
  memory) or loaded from a ``.npz`` file (keys ``features`` / ``labels``).  This
  is how you benchmark on *real* embeddings without re-embedding: dump a demo
  dataset's vectors once, then sweep strategies over the frozen features.

Both sources yield the ``{dataset_name: {media_id: media}}`` mapping
:func:`vtscore.eval.voting_iterations.run_voting_iterations_eval` consumes, where
each media carries an ``embeddings`` dict and a ``category`` — exactly the shape
a real loaded dataset has, minus the pixels.

Run it as a module::

    python -m vtscore.eval.al_benchmark --source synthetic \\
        --strategies random margin entropy bald coreset --seeds 0 1 2 \\
        --plot-dir al_out --output al_results.csv

    python -m vtscore.eval.al_benchmark --source precomputed \\
        --features-path features.npz --strategies all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np

from vtscore.eval.al_strategies import available_strategies
from vtscore.eval.visualize import plot_voting_iterations
from vtscore.eval.voting_iterations import run_voting_iterations_eval

if TYPE_CHECKING:
    import pandas as pd

#: Feature key used for the single stored vector on each synthetic/precomputed
#: media.  ``media_embedding`` resolves the sole entry as the primary vector.
_FEATURE_KEY = "feature"

# The benchmark's coverage-atlas floor is well below the production default (20)
# because hermetic sources are small: a lower floor lets the ``density_*``
# strategies actually resolve distinct cells over a few dozen items.
_DEFAULT_ATLAS_MIN_NODE_SIZE = 8

# A fast default sweep that skips ``eig`` (the costliest sampler, ``2 x K``
# retrains per step).  Pass ``--strategies all`` to include it.
_DEFAULT_STRATEGIES = ["random", "margin", "entropy", "bald", "coreset"]

MediaDict = dict[int, dict[str, Any]]
DatasetClips = dict[str, MediaDict]


# ------------------------------------------------------------------
# Data sources
# ------------------------------------------------------------------


def synthetic_source(
    *,
    n_per_cat: int = 40,
    dim: int = 32,
    n_categories: int = 2,
    separation: float = 1.0,
    noise: float = 0.25,
    seed: int = 0,
    name: str = "synthetic",
) -> DatasetClips:
    """Build a hermetic dataset of Gaussian category clusters.

    Each category is centred on its own random unit direction scaled by
    *separation*; every item is that centre plus isotropic Gaussian *noise*.
    Larger *separation* (or smaller *noise*) makes the classes easier to tell
    apart.  Everything is drawn from a *seed*-seeded generator, so the dataset —
    and therefore the benchmark — is reproducible.

    Returns the ``{name: {id: media}}`` mapping the voting-iterations evaluator
    consumes; each media carries a single ``embeddings`` vector and a
    ``category`` label (``"cat0"`` .. ``"cat<n-1>"``).
    """
    if n_categories < 2:
        raise ValueError(f"n_categories must be >= 2, got {n_categories}")
    if n_per_cat < 1:
        raise ValueError(f"n_per_cat must be >= 1, got {n_per_cat}")

    rng = np.random.default_rng(seed)
    means = rng.standard_normal((n_categories, dim)).astype(np.float32)
    means /= np.maximum(np.linalg.norm(means, axis=1, keepdims=True), 1e-12)
    means *= separation

    medias: MediaDict = {}
    media_id = 1
    for c in range(n_categories):
        category = f"cat{c}"
        for _ in range(n_per_cat):
            vec = (means[c] + rng.standard_normal(dim).astype(np.float32) * noise).astype(np.float32)
            medias[media_id] = {"id": media_id, "embeddings": {_FEATURE_KEY: vec}, "category": category}
            media_id += 1
    return {name: medias}


def precomputed_source(
    features: Any,
    labels: Sequence[Any],
    *,
    name: str = "precomputed",
    ids: Optional[Sequence[int]] = None,
) -> DatasetClips:
    """Wrap precomputed feature vectors + labels into a benchmark dataset.

    Args:
        features: An ``(n_items, dim)`` array-like of feature vectors.
        labels: One category label per row (stringified).
        name: Dataset name in the returned mapping.
        ids: Optional explicit media ids (default ``1 .. n``).

    Returns the ``{name: {id: media}}`` mapping.  Raises ``ValueError`` on a
    non-2D feature matrix or a label-count mismatch.
    """
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"features must be 2D (n_items, dim), got shape {features.shape}")
    if len(labels) != features.shape[0]:
        raise ValueError(f"labels length {len(labels)} does not match {features.shape[0]} feature rows")

    id_seq = list(ids) if ids is not None else list(range(1, features.shape[0] + 1))
    if len(id_seq) != features.shape[0]:
        raise ValueError(f"ids length {len(id_seq)} does not match {features.shape[0]} feature rows")

    medias: MediaDict = {}
    for media_id, vec, label in zip(id_seq, features, labels, strict=True):
        mid = int(media_id)
        medias[mid] = {
            "id": mid,
            "embeddings": {_FEATURE_KEY: np.asarray(vec, dtype=np.float32)},
            "category": str(label),
        }
    return {name: medias}


def precomputed_source_from_npz(
    path: str | Path,
    *,
    name: Optional[str] = None,
    features_key: str = "features",
    labels_key: str = "labels",
) -> DatasetClips:
    """Load a ``precomputed`` source from a ``.npz`` file.

    The archive must hold a 2D ``features`` array and a matching ``labels``
    array.  Loaded **without** ``allow_pickle`` (labels ride as a numpy string
    array), so an untrusted archive cannot execute code on load.  The dataset
    name defaults to the file stem.
    """
    path = Path(path)
    with np.load(path) as data:  # allow_pickle defaults to False
        if features_key not in data or labels_key not in data:
            raise ValueError(f"{path} must contain '{features_key}' and '{labels_key}' arrays")
        features = data[features_key]
        labels = [str(x) for x in data[labels_key].tolist()]
    return precomputed_source(features, labels, name=name or path.stem)


#: Names of the hermetic data sources, for CLI/help discoverability.
DATA_SOURCES = ("synthetic", "precomputed")


# ------------------------------------------------------------------
# Benchmark runner
# ------------------------------------------------------------------


def run_al_benchmark(
    dataset_clips: DatasetClips,
    *,
    strategies: list[str],
    seeds: list[int],
    categories: Optional[dict[str, list[str]]] = None,
    inclusion: int = 0,
    sim_fraction: float = 0.5,
    max_steps: Optional[int] = None,
    calibrate_count: int = 2,
    atlas_min_node_size: int = _DEFAULT_ATLAS_MIN_NODE_SIZE,
    plot_dir: Optional[str | Path] = None,
) -> "pd.DataFrame":
    """Run *strategies* over *dataset_clips* and return the results frame.

    A thin orchestration over
    :func:`~vtscore.eval.voting_iterations.run_voting_iterations_eval`: it adds
    the strategy axis, applies the benchmark's lower coverage-atlas floor, and
    (when *plot_dir* is given) renders the strategy-faceted charts.  The returned
    :class:`~pandas.DataFrame` carries the usual voting-iterations columns plus
    ``strategy``.
    """
    df = run_voting_iterations_eval(
        dataset_clips,
        seeds=seeds,
        categories=categories,
        inclusion=inclusion,
        sim_fraction=sim_fraction,
        calibrate_count=calibrate_count,
        strategies=strategies,
        max_steps=max_steps,
        atlas_min_node_size=atlas_min_node_size,
    )
    if plot_dir is not None:
        plot_voting_iterations(df, output_dir=plot_dir)
    return df


def _final_cost_summary(df: "pd.DataFrame") -> "pd.DataFrame":
    """Return each strategy's mean cost at its last recorded step, ascending.

    Averages over the deepest ``t`` reached per (seed, dataset, category) run so
    strategies are ranked by where they *end up* — lower is better.
    """
    import pandas as pd  # noqa: PLC0415

    if df.empty:
        return pd.DataFrame(columns=pd.Index(["strategy", "final_cost"]))
    last = df.sort_values("t").groupby(["strategy", "seed", "dataset", "category"]).tail(1)
    records = [
        {"strategy": strategy, "final_cost": float(group["cost"].mean())} for strategy, group in last.groupby("strategy")
    ]
    summary = pd.DataFrame(records, columns=pd.Index(["strategy", "final_cost"]))
    return summary.sort_values("final_cost").reset_index(drop=True)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def _resolve_strategies(requested: list[str]) -> list[str]:
    """Expand the special ``all`` token and validate every requested name."""
    if requested == ["all"]:
        return available_strategies()
    unknown = [s for s in requested if s not in available_strategies()]
    if unknown:
        raise SystemExit(f"Unknown strategies {unknown}; available: {', '.join(available_strategies())} (or 'all')")
    return requested


def _build_source(args: argparse.Namespace) -> DatasetClips:
    if args.source == "synthetic":
        return synthetic_source(
            n_per_cat=args.n_per_cat,
            dim=args.dim,
            n_categories=args.n_categories,
            separation=args.separation,
            noise=args.noise,
            seed=args.data_seed,
        )
    if args.source == "precomputed":
        if not args.features_path:
            raise SystemExit("--features-path is required for --source precomputed")
        return precomputed_source_from_npz(args.features_path)
    raise SystemExit(f"Unknown source {args.source!r}")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m vtscore.eval.al_benchmark",
        description="Benchmark active-learning acquisition strategies on hermetic data sources.",
    )
    parser.add_argument(
        "--source", choices=DATA_SOURCES, default="synthetic", help="Hermetic data source (default: synthetic)."
    )
    # Synthetic-source knobs.
    parser.add_argument("--n-per-cat", type=int, default=40, help="Synthetic items per category (default: 40).")
    parser.add_argument("--dim", type=int, default=32, help="Synthetic embedding dimensionality (default: 32).")
    parser.add_argument("--n-categories", type=int, default=2, help="Synthetic category count (default: 2).")
    parser.add_argument("--separation", type=float, default=1.0, help="Synthetic class-mean separation (default: 1.0).")
    parser.add_argument("--noise", type=float, default=0.25, help="Synthetic isotropic noise scale (default: 0.25).")
    parser.add_argument("--data-seed", type=int, default=0, help="Seed for the synthetic data (default: 0).")
    # Precomputed-source input.
    parser.add_argument(
        "--features-path",
        type=str,
        default=None,
        metavar="NPZ",
        help="'.npz' with 'features'/'labels' for --source precomputed.",
    )
    # Benchmark axes.
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=_DEFAULT_STRATEGIES,
        metavar="NAME",
        help="Strategies to compare, or 'all' (default: random margin entropy bald coreset).",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Simulation seeds (default: 0 1 2).")
    parser.add_argument("--inclusion", type=int, default=0, help="Inclusion setting in [-10, 10] (default: 0).")
    parser.add_argument(
        "--sim-fraction", type=float, default=0.5, help="Fraction of items used for simulated voting (default: 0.5)."
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Cap on voting steps per run (default: all).")
    parser.add_argument(
        "--calibrate-count",
        type=int,
        default=2,
        help="Random Train/Calibrate splits for threshold calibration (default: 2).",
    )
    parser.add_argument(
        "--atlas-min-node-size",
        type=int,
        default=_DEFAULT_ATLAS_MIN_NODE_SIZE,
        help=f"Coverage-atlas leaf floor for density_* strategies (default: {_DEFAULT_ATLAS_MIN_NODE_SIZE}).",
    )
    parser.add_argument(
        "--output", type=str, default=None, metavar="CSV", help="Write the full results frame to a CSV file."
    )
    parser.add_argument(
        "--plot-dir", type=str, default=None, metavar="DIR", help="Render strategy-faceted charts into DIR."
    )

    args = parser.parse_args(argv)
    strategies = _resolve_strategies(args.strategies)
    dataset_clips = _build_source(args)

    df = run_al_benchmark(
        dataset_clips,
        strategies=strategies,
        seeds=args.seeds,
        inclusion=args.inclusion,
        sim_fraction=args.sim_fraction,
        max_steps=args.max_steps,
        calibrate_count=args.calibrate_count,
        atlas_min_node_size=args.atlas_min_node_size,
        plot_dir=args.plot_dir,
    )

    print(f"\n{'=' * 60}")
    print("ACTIVE-LEARNING BENCHMARK — final cost by strategy (lower is better)")
    print(f"{'=' * 60}")
    summary = _final_cost_summary(df)
    for _, row in summary.iterrows():
        print(f"  {row['strategy']:18s}  final_cost={row['final_cost']:.4f}")

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nResults written to {args.output}")
    if args.plot_dir:
        print(f"Plots written to {args.plot_dir}/")

    if df.empty:
        print("\nNo rows produced (every split lacked a usable test set).")
        sys.exit(1)


if __name__ == "__main__":
    main()
