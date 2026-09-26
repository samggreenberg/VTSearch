"""Literal examples for #3945: the same Autopilot votes, fitted at C = 1 and at a smaller C.

For the sessions where the replayed C path says C = 1 loses most to the smaller
C at click ``--t``, refit both on that session's votes and list the held-out
images whose rank the two fits disagree on most, with filenames: the positives
the smaller C lifts (its wins) and the negatives it lifts (its losses).  A
"win" that is really a mislabelled image is checkable by eye.

    python examples.py --replay-root /expscratch/$USER/overtrain-3945/stageD \
        --rpath /expscratch/$USER/overtrain-3945/R400 --t 300 --out EXAMPLES.md
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "svm_vs_logistic"))
sys.path.insert(0, str(HERE.parent / "calibration"))

from c_path import fit_svm_w  # noqa: E402
from stage_a import avg_prec, harness_split, load_cell  # noqa: E402


def ranks(s: np.ndarray) -> np.ndarray:
    """1 = top of the haystack."""
    order = np.argsort(-s, kind="mergesort")
    r = np.empty(len(s), dtype=int)
    r[order] = np.arange(1, len(s) + 1)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-root", required=True, type=Path)
    ap.add_argument("--rpath", required=True, type=Path, help="c_path.py output dir")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--t", type=int, default=150)
    ap.add_argument("--small-c", type=float, default=0.1)
    ap.add_argument("--n-cells", type=int, default=4)
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()
    from vtscore.eval.labels import media_is_positive

    rp = pd.concat(pd.read_csv(f) for f in glob.glob(str(args.rpath / "*.csv.gz")))
    rp = rp[(rp["t"] == args.t) & (rp["one_class"] == 0)]
    keys = ["dataset", "embedder", "category", "seed"]
    w = rp[np.isclose(rp["C"], 1.0) | np.isclose(rp["C"], args.small_c)].pivot_table(
        index=keys, columns="C", values="test_oracle_cost"
    )
    w["gap"] = w[1.0] - w[args.small_c]
    w = w.reset_index().sort_values("gap", ascending=False)
    chosen = w.groupby(["dataset", "embedder"]).head(1).head(args.n_cells)
    picks = pd.read_csv(args.replay_root / "svm" / "picks_all.csv.gz")

    lines = [
        f"# Literal examples (#3945): the same {args.t} Autopilot votes at C = 1 and C = {args.small_c:g}",
        "",
        f"Each block refits the shipped head at C = 1 and at C = {args.small_c:g} on the vote set the shipped "
        f"arm's Autopilot collected by click {args.t}, and scores the harness's held-out half. The sessions are the "
        f"ones where C = {args.small_c:g} beats C = 1 by the most oracle cost at that click, one per environment. "
        "Ranks are 1 = top of the haystack. `labels` is the image's full annotation, so a 'loss' that plainly "
        "contains the category is an annotation gap, not a model error.",
        "",
    ]
    for _, c in chosen.iterrows():
        X, ids, medias = load_cell(c.dataset, c.embedder)
        labels = np.array([1 if media_is_positive(medias[i], c.category) else 0 for i in ids])
        pos_of = {cid: i for i, cid in enumerate(ids)}
        _, test = harness_split(ids, int(c.seed))
        te = np.array([pos_of[x] for x in test])
        p = picks[
            (picks.dataset == c.dataset)
            & (picks.embedder == c.embedder)
            & (picks.category == c.category)
            & (picks.seed == c.seed)
            & (picks.t <= args.t)
        ]
        idx = np.array([pos_of[x] for x in p.picked_id])
        y = p.picked_label.to_numpy()
        w1, b1 = fit_svm_w(X[idx], y, 1.0)
        ws, bs = fit_svm_w(X[idx], y, args.small_c)
        s1, ss = X[te] @ w1 + b1, X[te] @ ws + bs
        r1, rs = ranks(s1), ranks(ss)
        yt = labels[te]
        lines += [
            f"## {c.dataset} × {c.embedder}: `{c.category}`, seed {int(c.seed)}",
            "",
            f"Votes: {int(y.sum())} Good / {int(len(y) - y.sum())} Bad. Held-out: {int(yt.sum())} positives in "
            f"{len(te)}. AP: C = 1 {avg_prec(s1, yt):.2f}, C = {args.small_c:g} {avg_prec(ss, yt):.2f}. "
            f"Oracle cost gap: {c.gap:+.2f}.",
            "",
            f"| kind | file | labels | rank at C = 1 | rank at C = {args.small_c:g} |",
            "|---|---|---|---|---|",
        ]
        d = r1 - rs  # > 0: the smaller C ranks it higher
        for kind, mask in (
            (f"C = {args.small_c:g} wins (positive lifted)", yt == 1),
            (f"C = {args.small_c:g} losses (negative lifted)", yt == 0),
        ):
            cand = np.where(mask)[0]
            for j in cand[np.argsort(-d[cand])][: args.top]:
                m = medias[ids[te[j]]]
                lab = ", ".join(sorted(m.get("categories") or [m.get("category") or ""])[:8])
                name = m.get("filename") or m.get("origin_name") or ids[te[j]]
                lines.append(f"| {kind} | `{name}` | {lab} | {r1[j]} | {rs[j]} |")
        lines.append("")
    args.out.write_text("\n".join(lines))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
