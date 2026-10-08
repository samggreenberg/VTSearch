"""Literal sessions for the #4383 report: what each rule returns on named cells."""

import sys
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
r = pd.read_csv(sys.argv[1], dtype={"size": str})
rules = ["fixed", "grow", "gmm", "gmm+shift5", "gmm-grow"]
cells = [
    ("0.44%", "airplane@large"),
    ("0.44%", "toothbrush@medium"),
    ("0.44%", "tv@small"),
    ("5%", "dog@large"),
    ("0.1%", "airplane@medium"),
]
for w, cat in cells:
    for size in ("320", "full", "x10"):
        g = r[
            (r.world == w)
            & (r.category == cat)
            & (r.seed == 0)
            & (r.t == 150)
            & (r["size"] == size)
            & (r.floor == 0.5)
            & (r.draw == 0)
            & r.rule.isin(rules)
        ]
        if g.empty:
            continue
        n, npos, ok = int(g.n_corpus.iloc[0]), int(g.n_pos.iloc[0]), int(g.oracle_k.iloc[0])
        print(
            f"== {w} {cat} seed 0, size {size}: n={n}, positives={npos}, oracle keeps {ok} (recall {g.oracle_recall.iloc[0]:.2f}, best F1 {g.oracle_f1.iloc[0]:.2f})"
        )
        print(g[["rule", "k", "precision", "recall", "f1", "votes", "est"]].round(2).to_string(index=False))
