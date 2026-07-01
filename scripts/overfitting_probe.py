#!/usr/bin/env python
"""Shuffled-label control: is the detector MLP overfitting, or actually learning?

Motivation
----------
A very common worry: "I click ~200 Goods and Bads and the resulting sort
*always* puts every Good above every Bad - even for a category I made up on the
spot. Doesn't that mean the MLP has too many parameters and is over-training?"

The answer is no, and this script demonstrates why. Separating your *labeled*
examples is uninformative: with 512-768-dim embeddings, ~200 points are almost
always separable for **any** labeling (including pure noise), so the model fits
its training labels perfectly whether or not it learned anything real. Lowering
the parameter count would not change that symptom. Overfitting is only visible
on items the model did **not** train on.

This script makes the distinction concrete by running two experiments on the
same dataset, same train/held-out split, and same model:

* **REAL** - label each item by its true category (a signal that exists).
* **NOISE** - label each item by a fixed coin flip (a signal that does not).

For each it reports, averaged over seeds:

* ``train_auc`` / ``train_acc`` - how well the model ranks/classifies its **own
  training labels**. This is the "all Goods above all Bads" symptom. It is
  ~1.0 for **both** experiments. That is the whole point: perfect training
  separation happens even for pure noise, so it proves nothing.
* ``test_auc`` / ``test_f1`` / ``test_cost`` - the same on **held-out** items.
  REAL generalizes (high AUC/F1, low cost); NOISE collapses to chance
  (AUC ~ 0.5, cost ~ chance) because there is nothing to generalize to.

The verdict you should read off the output:

* REAL test AUC well above 0.5  -> the detector is genuinely learning; the
  perfect training separation you saw is *good behavior*, not over-training.
* NOISE test AUC ~ 0.5 while its train AUC ~ 1.0 -> the model is *not* a
  memorizer that fakes held-out signal; it correctly fails to generalize noise.
  (If NOISE test AUC were high, *that* would be real overfitting / leakage.)

Usage::

    python scripts/overfitting_probe.py
    python scripts/overfitting_probe.py --dataset caltech101_s --n-good 100 --n-bad 100
    python scripts/overfitting_probe.py --dataset esc50_s --category rain --seeds 1 2 3 42 100

To instead answer "is the hidden-layer width right?", sweep capacity while
holding the split/votes/seeds fixed and watch held-out AUC per width::

    python scripts/overfitting_probe.py --sweep-hidden 4 8 16 32 64 128 256

The defaults download a small demo dataset (cached after first run) and take a
minute or two. This is a diagnostic / teaching script, not a CI gate.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np


def _auc(scores: list[float], labels: list[int]) -> float:
    """Threshold-free ranking AUC = P(score(pos) > score(neg)).

    Computed via the rank-sum (Mann-Whitney U) identity so it needs no
    sklearn and handles ties correctly. Returns ``nan`` when one class is
    absent (AUC is undefined without both a positive and a negative).
    """
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # Average ranks within tie groups so tied scores don't bias the AUC.
    scores_arr = np.asarray(scores, dtype=np.float64)[order]
    i = 0
    while i < len(scores_arr):
        j = i
        while j + 1 < len(scores_arr) and scores_arr[j + 1] == scores_arr[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    labels_arr = np.asarray(labels)
    sum_ranks_pos = ranks[labels_arr == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _score(model: Any, embs: np.ndarray) -> list[float]:
    """Sigmoid MLP scores for a stack of embeddings."""
    import torch  # noqa: PLC0415

    X = torch.tensor(embs, dtype=torch.float32)
    with torch.no_grad():
        X = X.to(next(model.parameters()).device)
        return torch.sigmoid(model(X)).squeeze(1).cpu().tolist()


def _assign_labels(
    ids: list[int],
    clips: dict[int, dict[str, Any]],
    mode: str,
    category: str,
    rng: np.random.RandomState,
    positive_rate: float,
) -> dict[int, int]:
    """Return ``{media_id: 0/1}`` for every id under REAL or NOISE labeling.

    REAL uses the ground-truth category. NOISE draws an independent coin flip
    per id at ``positive_rate`` (matched to REAL's base rate) so the two
    experiments share class balance and differ only in whether the label
    carries signal.
    """
    from vtscore.eval.labels import media_is_positive  # noqa: PLC0415

    if mode == "real":
        return {cid: int(media_is_positive(clips[cid], category)) for cid in ids}
    return {cid: int(rng.random() < positive_rate) for cid in ids}


def _run_one(
    clips: dict[int, dict[str, Any]],
    category: str,
    mode: str,
    seed: int,
    n_good: int,
    n_bad: int,
    inclusion: int,
    hidden_dim: int | None = None,
) -> dict[str, float] | None:
    """One (experiment, seed) trial. Returns per-trial metrics or ``None``.

    Splits the dataset into a train pool and a held-out pool, labels **both**
    pools under *mode*, samples up to ``n_good``/``n_bad`` training votes, trains
    the MLP, and measures train-set vs held-out ranking.

    When *hidden_dim* is ``None`` the production auto-sizing applies
    (``max(4, min(32, n_train//3))``); pass an explicit width to hold every
    other variable fixed and isolate the effect of hidden-layer capacity.
    """
    import torch  # noqa: PLC0415

    from vtscore.embedding.media_vectors import media_embedding
    from vtscore.eval.labels import media_is_positive
    from vtscore.training.mlp import train_model
    from vtscore.training.thresholds import calculate_cross_calibration_threshold

    rng = np.random.RandomState(seed)
    all_ids = [cid for cid in sorted(clips) if media_embedding(clips[cid]) is not None]
    shuffled = rng.permutation(all_ids).tolist()
    split = len(shuffled) // 2
    train_pool, test_pool = shuffled[:split], shuffled[split:]

    # Base rate from the REAL category, so NOISE matches its class balance.
    positive_rate = float(np.mean([media_is_positive(clips[c], category) for c in all_ids]))
    if positive_rate in (0.0, 1.0):
        return None  # category absent or universal - nothing to separate

    labels = _assign_labels(all_ids, clips, mode, category, rng, positive_rate)

    # Sample the training votes from the train pool, capped at n_good / n_bad.
    pos_ids = [c for c in train_pool if labels[c] == 1]
    neg_ids = [c for c in train_pool if labels[c] == 0]
    rng.shuffle(pos_ids)
    rng.shuffle(neg_ids)
    good = pos_ids[:n_good]
    bad = neg_ids[:n_bad]
    if not good or not bad:
        return None

    train_ids = good + bad
    X_list = [media_embedding(clips[c]) for c in train_ids]
    y_list = [1.0] * len(good) + [0.0] * len(bad)
    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
    input_dim = X.shape[1]

    threshold = calculate_cross_calibration_threshold(X_list, y_list, input_dim, inclusion, rng=rng)
    model = train_model(X, y, input_dim, hidden_dim=hidden_dim)

    # In-sample: how well the model separates the votes it trained on.
    train_scores = _score(model, np.array(X_list))
    train_labels = [int(v) for v in y_list]
    train_auc = _auc(train_scores, train_labels)
    train_acc = float(np.mean([(s >= threshold) == bool(t) for s, t in zip(train_scores, train_labels)]))

    # Held-out: the only view that can reveal overfitting.
    test_ids = [c for c in test_pool if labels[c] in (0, 1)]
    test_labels = [labels[c] for c in test_ids]
    if sum(test_labels) == 0 or sum(test_labels) == len(test_labels):
        return None
    test_scores = _score(model, np.array([media_embedding(clips[c]) for c in test_ids]))
    test_auc = _auc(test_scores, test_labels)
    preds = [int(s >= threshold) for s in test_scores]

    from vtscore.eval.metrics import compute_binary_classification_metrics  # noqa: PLC0415

    _, _, _, test_f1 = compute_binary_classification_metrics(preds, test_labels)

    fn = sum(p == 0 and gt == 1 for p, gt in zip(preds, test_labels))
    fp = sum(p == 1 and gt == 0 for p, gt in zip(preds, test_labels))
    n_pos = sum(test_labels)
    n_neg = len(test_labels) - n_pos
    fpr = fp / n_neg if n_neg else 0.0
    fnr = fn / n_pos if n_pos else 0.0
    fpr_w, fnr_w = (1.0, 2.0**inclusion) if inclusion >= 0 else (2.0 ** (-inclusion), 1.0)
    test_cost = fpr_w * fpr + fnr_w * fnr

    return {
        "train_auc": train_auc,
        "train_acc": train_acc,
        "test_auc": test_auc,
        "test_f1": test_f1,
        "test_cost": test_cost,
    }


def _mean(rows: list[dict[str, float]], key: str) -> float:
    vals = [r[key] for r in rows if not np.isnan(r[key])]
    return float(np.mean(vals)) if vals else float("nan")


def _std(rows: list[dict[str, float]], key: str) -> float:
    vals = [r[key] for r in rows if not np.isnan(r[key])]
    return float(np.std(vals)) if vals else float("nan")


def _run_hidden_sweep(
    clips: dict[int, dict[str, Any]],
    category: str,
    seeds: list[int],
    n_good: int,
    n_bad: int,
    inclusion: int,
    hidden_sizes: list[int],
) -> None:
    """Sweep hidden-layer width and print held-out ranking per size.

    For every width, both REAL and NOISE are re-run over all *seeds* on the
    same splits/votes, varying only the hidden layer. Held-out ``test_auc`` is
    the capacity signal (threshold-free, so it isolates ranking quality from
    threshold calibration); its std across seeds flags instability that big
    layers can introduce even when the mean looks fine.
    """
    header = (
        f"{'hidden':>7} {'REAL_train':>11} {'REAL_test':>11} {'±std':>7} "
        f"{'NOISE_train':>12} {'NOISE_test':>11} {'±std':>7}"
    )
    print(header)
    print("-" * len(header))
    for hd in hidden_sizes:
        real = [
            r
            for s in seeds
            if (r := _run_one(clips, category, "real", s, n_good, n_bad, inclusion, hidden_dim=hd)) is not None
        ]
        noise = [
            r
            for s in seeds
            if (r := _run_one(clips, category, "noise", s, n_good, n_bad, inclusion, hidden_dim=hd)) is not None
        ]
        if not real or not noise:
            print(f"{hd:>7} {'(no valid trials)':>60}")
            continue
        print(
            f"{hd:>7} "
            f"{_mean(real, 'train_auc'):>11.3f} {_mean(real, 'test_auc'):>11.3f} {_std(real, 'test_auc'):>7.3f} "
            f"{_mean(noise, 'train_auc'):>12.3f} {_mean(noise, 'test_auc'):>11.3f} {_std(noise, 'test_auc'):>7.3f}"
        )

    print(
        "\nHow to read this:\n"
        "  * REAL_train stays ~1.0 at every width - training separation is not a\n"
        "    capacity signal, so it can't tell you which width is right.\n"
        "  * REAL_test is what matters. If it is flat across a wide band of widths,\n"
        "    the task is (nearly) linearly separable in embedding space and the hidden\n"
        "    layer is not the bottleneck - any width on the plateau is 'right', and the\n"
        "    default cap of 32 sits safely on it. A peak that then declines would mean\n"
        "    larger widths overfit; a rising curve would mean 32 is too small.\n"
        "  * NOISE_test should sit at ~0.5 for every width: no capacity level can\n"
        "    generalize a signal that isn't there. Watch NOISE_train instead climb\n"
        "    toward 1.0 as width grows - that is pure memorization, exactly the thing\n"
        "    held-out AUC refuses to reward.\n"
        "  * Prefer the smallest width on the REAL_test plateau (also the lowest ±std):\n"
        "    equal quality, less compute, more stable. That is the case for the 4-32\n"
        "    auto-sizing - it tracks the bottom of the plateau."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="caltech101_s", help="Demo dataset id (default: caltech101_s)")
    parser.add_argument(
        "--category",
        default="",
        help="Target category for the REAL experiment (default: the dataset's most common category)",
    )
    parser.add_argument("--n-good", type=int, default=100, help="Max Good training votes (default: 100)")
    parser.add_argument("--n-bad", type=int, default=100, help="Max Bad training votes (default: 100)")
    parser.add_argument("--inclusion", type=int, default=0, help="Inclusion value in [-10, 10] (default: 0)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 42, 100], help="Random seeds to average")
    parser.add_argument(
        "--sweep-hidden",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="Instead of REAL-vs-NOISE, sweep these hidden-layer widths and report held-out AUC "
        "per width (e.g. --sweep-hidden 4 8 16 32 64 128 256). Isolates MLP capacity.",
    )
    args = parser.parse_args()

    from vtscore.embedding import initialize_models

    initialize_models()

    from vtscore.datasets.loader import load_demo_dataset

    print(f"Loading demo dataset '{args.dataset}' (downloads + embeds on first run)...")
    clips: dict[int, dict[str, Any]] = {}
    load_demo_dataset(args.dataset, clips)

    category = args.category
    if not category:
        counts: dict[str, int] = {}
        for c in clips.values():
            for cat in c.get("categories") or [c.get("category")]:
                if cat:
                    counts[cat] = counts.get(cat, 0) + 1
        category = max(counts, key=counts.get)  # type: ignore[arg-type]
    print(
        f"REAL category: '{category}'   |   votes: up to {args.n_good} good / {args.n_bad} bad   |   seeds: {args.seeds}\n"
    )

    if args.sweep_hidden:
        print(f"Sweeping hidden-layer widths: {args.sweep_hidden}\n")
        _run_hidden_sweep(clips, category, args.seeds, args.n_good, args.n_bad, args.inclusion, args.sweep_hidden)
        return

    results: dict[str, list[dict[str, float]]] = {"real": [], "noise": []}
    for mode in ("real", "noise"):
        for seed in args.seeds:
            row = _run_one(clips, category, mode, seed, args.n_good, args.n_bad, args.inclusion)
            if row is not None:
                results[mode].append(row)

    header = (
        f"{'experiment':<10} {'train_auc':>10} {'train_acc':>10} {'test_auc':>10} {'test_f1':>10} {'test_cost':>10}"
    )
    print(header)
    print("-" * len(header))
    for mode, label in (("real", "REAL"), ("noise", "NOISE")):
        rows = results[mode]
        if not rows:
            print(f"{label:<10} {'(no valid trials)':>52}")
            continue
        print(
            f"{label:<10} "
            f"{_mean(rows, 'train_auc'):>10.3f} {_mean(rows, 'train_acc'):>10.3f} "
            f"{_mean(rows, 'test_auc'):>10.3f} {_mean(rows, 'test_f1'):>10.3f} {_mean(rows, 'test_cost'):>10.3f}"
        )

    print(
        "\nHow to read this:\n"
        "  * train_auc/train_acc ~ 1.0 for BOTH rows is the 'all Goods above all Bads'\n"
        "    symptom. It appears even for NOISE, so it does NOT indicate over-training.\n"
        "  * REAL test_auc well above 0.5 -> the detector genuinely learns; the perfect\n"
        "    training separation is good behavior, not a too-big MLP.\n"
        "  * NOISE test_auc ~ 0.5 (with train_auc ~ 1.0) -> the model correctly fails to\n"
        "    generalize a signal that isn't there. If NOISE test_auc were high, THAT would\n"
        "    be real overfitting/leakage worth chasing.\n"
        "  * To probe capacity directly, edit MLP_HIDDEN_MAX in vtscore/config.py down to\n"
        "    a handful of neurons and rerun: REAL test_auc barely moves and train_auc stays\n"
        "    ~1.0, confirming the symptom is embedding dimensionality, not parameter count."
    )


if __name__ == "__main__":
    main()
