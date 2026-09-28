"""Literal examples for #3197: where the two heads rank the same test images differently.

For a handful of cells, refit the shipped SVM and the shipped logistic head on
the SAME Autopilot vote set (the ``svm`` arm's picks at click ``--t``) and list
the held-out images whose rank the two heads disagree on most - the positives
the SVM ranks far higher than the logistic head (its wins) and the negatives it
ranks far higher (its losses) - with filenames, so a reader can check whether a
"win" is a mislabelled image.

    python examples.py --stageB /expscratch/$USER/svmlog-3197/stageB --out FILE.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "calibration"))

import heads as H  # noqa: E402
from stage_a import Cell, load_cell  # noqa: E402


#: The head each ``--other`` arm names, refitted the way that arm fits it in the loop.
OTHER_FIT = {
    "linear": H.fit_lr_shipped,
    "lrconv": lambda X, y: H.fit_lr_sklearn(X, y, C=1.0),
}
OTHER_DESC = {
    "linear": "the early-stopped logistic head the SVM replaced",
    "lrconv": "logistic regression fitted to convergence, C = 1",
}


def ranks(s: np.ndarray) -> np.ndarray:
    """1 = top of the haystack."""
    order = np.argsort(-s, kind="mergesort")
    r = np.empty(len(s), dtype=int)
    r[order] = np.arange(1, len(s) + 1)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stageB", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--t", type=int, default=40)
    ap.add_argument("--n-cells", type=int, default=6)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument(
        "--other",
        default="linear",
        choices=tuple(OTHER_FIT),
        help="the arm compared with the SVM: `linear` (#3197) or `lrconv` (#4114)",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="list only images in the top K of either head, where a reader looks (0 = anywhere; #3197's reading)",
    )
    ap.add_argument(
        "--four-ways",
        action="store_true",
        help="list positives and negatives each head ranks higher, not only the SVM's side",
    )
    ap.add_argument(
        "--both-ways",
        action="store_true",
        help="per stratum, also the cell where the OTHER arm led most (#4114: the gap changes sign by band)",
    )
    args = ap.parse_args()
    other = args.other
    fit_other = OTHER_FIT[other]
    from vtscore.eval.labels import media_is_positive

    picks = pd.read_csv(args.stageB / "svm" / "picks_all.csv.gz")
    # The cells where the two Stage B arms' oracle cost at t differs most in the
    # SVM's favour - the cells that carry the in-loop gap - one per env.
    import _cells_io

    snaps = []
    for arm in ("svm", other):
        df, _ = _cells_io.load_arm(args.stageB / arm / "results")
        s = df[df["t"] <= args.t].sort_values("t").groupby(["dataset", "embedder", "category", "seed"]).tail(1)
        s = s[["dataset", "embedder", "category", "seed", "oracle_cost"]].assign(arm=arm)
        snaps.append(s)
    w = pd.concat(snaps).pivot_table(
        index=["dataset", "embedder", "category", "seed"], columns="arm", values="oracle_cost"
    )
    w["gap"] = w[other] - w["svm"]
    w = w.reset_index().sort_values("gap", ascending=False)
    from analyze_stage_b import env_of

    w["env"] = env_of(w)
    chosen = w.groupby("env").head(1).head(args.n_cells)
    if args.both_ways:
        chosen = pd.concat([chosen, w.iloc[::-1].groupby("env").head(1).head(args.n_cells)])

    lines = [
        f"# Literal examples: same {args.t} Autopilot votes, the SVM and `{other}`",
        "",
        f"Each block refits the shipped SVM and the `{other}` head ({OTHER_DESC[other]}) on the vote set the "
        f"**SVM arm's** Autopilot collected by click {args.t}, and scores the harness's held-out test half. "
        "The cells are the ones where the two in-loop arms' oracle costs differed most in the SVM's favour "
        "(one per environment"
        + (", then one per environment in the other arm's favour" if args.both_ways else "")
        + f"). Ranks are 1 = top of the haystack; `in-loop gap` is the `{other}` arm's "
        "oracle cost minus the SVM arm's at that click in Stage B.",
        "",
    ]
    for _, c in chosen.iterrows():
        X, ids, medias = load_cell(c.dataset, c.embedder)
        labels = np.array([1 if media_is_positive(medias[i], c.category) else 0 for i in ids])
        cell = Cell(X, ids, labels, int(c.seed))
        p = picks[
            (picks.dataset == c.dataset)
            & (picks.embedder == c.embedder)
            & (picks.category == c.category)
            & (picks.seed == c.seed)
            & (picks.t <= args.t)
        ]
        pos_of = {cid: i for i, cid in enumerate(ids)}
        idx = np.array([pos_of[x] for x in p.picked_id])
        y = p.picked_label.to_numpy()
        svm, lr = H.fit_svm_shipped(X[idx], y), fit_other(X[idx], y)
        te = cell.test
        rs, rl = ranks(svm.score(X[te])), ranks(lr.score(X[te]))
        yt = labels[te]
        from stage_a import avg_prec

        lines += [
            f"## {c.dataset} × {c.embedder} — `{c.category}`, seed {int(c.seed)}",
            "",
            f"Votes: {int(y.sum())} Good / {int(len(y) - y.sum())} Bad. Held-out: {int(yt.sum())} positives in {len(te)}. "
            f"AP on these votes: SVM {avg_prec(svm.score(X[te]), yt):.2f}, {other} {avg_prec(lr.score(X[te]), yt):.2f}. "
            f"In-loop gap at click {args.t}: {c.gap:+.2f}.",
            "",
            f"| kind | file | label | SVM rank | {other} rank |",
            "|---|---|---|---|---|",
        ]
        d = rl - rs  # > 0: the SVM ranks it higher
        near_top = np.ones(len(d), dtype=bool) if args.top_k <= 0 else np.minimum(rs, rl) <= args.top_k
        kinds = [
            ("SVM wins (positive ranked higher)", yt == 1, 1),
            ("SVM losses (negative ranked higher)", yt == 0, 1),
        ]
        if args.four_ways:
            kinds += [
                (f"{other} wins (positive ranked higher)", yt == 1, -1),
                (f"{other} losses (negative ranked higher)", yt == 0, -1),
            ]
        for kind, mask, sign in kinds:
            cand = np.where(mask & near_top & (sign * d > 0))[0]
            top = cand[np.argsort(-sign * d[cand])][: args.top]
            for j in top:
                m = medias[ids[te[j]]]
                lab = ", ".join((m.get("categories") or [m.get("category")])[:6])
                lines.append(f"| {kind} | `{m.get('filename') or m.get('origin_name')}` | {lab} | {rs[j]} | {rl[j]} |")
        lines.append("")
    args.out.write_text("\n".join(lines))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
