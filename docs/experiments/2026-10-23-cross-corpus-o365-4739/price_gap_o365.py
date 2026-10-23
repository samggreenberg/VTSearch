"""#4739 Leg 1: per run, the walk's price on o365 minus its in-corpus price, paired on (category, seed), 107 cells.

    python price_gap_o365.py <repo>/scripts/experiments/calibration <cross-corpus root>

Fills each run's votes as analyze_autopilot_4671 does (text_baseline anchor in-corpus, the run's own query rows on the
Find corpus), takes session means over votes 1-150 and the windows, and reports mean +- SE over runs.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(sys.argv[1])  # scripts/experiments/calibration
sys.path.insert(0, str(HERE))
import common  # noqa: E402

common.setup_env()
import _cells_io  # noqa: E402
import analyze_autopilot_4671 as ap  # noqa: E402
import curves  # noqa: E402
from analyze_buildup_4668 import text_fbeta  # noqa: E402
from analyze_progression_4184 import CELL, fbeta, filled_matrix  # noqa: E402
from _rank_metrics import beta_tag  # noqa: E402

R = Path(sys.argv[2])  # launch_cross_corpus_4739.sh's ROOT, holding paired-107.txt and text_baseline.csv
keep = ap.read_categories(R / "paired-107.txt")
bl = curves.text_sort_baseline(R / "text_baseline.csv")
bl = bl[bl["category"].isin(keep)]
rows = []
for p, beta in ap.PRESETS.items():
    obj_anchor = text_fbeta(bl, f"text_line_precision_{beta_tag(beta)}", f"text_line_recall_{beta_tag(beta)}", beta)
    m = {}
    for arm in ("ctl", "nowalk"):
        frame, _ = _cells_io.load_arm(R / "o365" / f"{arm}_{p}" / "results")
        frame = frame[frame["category"].isin(keep)].copy()
        frame["objective"] = fbeta(frame["precision"], frame["recall"], beta)
        cells = frame[CELL].drop_duplicates().sort_values(CELL).reset_index(drop=True)
        ins = filled_matrix(frame, cells, {}, 150, metric="objective", baseline_metric=obj_anchor)[0]
        fnd = filled_matrix(
            frame, cells, {}, 150, metric="find_fbeta", baseline_metric=ap.find_anchor(frame, "find_fbeta")
        )[0]
        n_f = filled_matrix(
            frame, cells, {}, 150, metric="find_n_flagged", baseline_metric=ap.find_anchor(frame, "find_n_flagged")
        )[0]
        m[arm] = (ins, fnd, n_f)
    if p == "b1":
        cols = [c for c in m["ctl"][0].columns if 1 <= int(c) <= 150]
        ex = pd.DataFrame(
            {
                "ctl_in": m["ctl"][0][cols].mean(axis=1),
                "nowalk_in": m["nowalk"][0][cols].mean(axis=1),
                "ctl_o365": m["ctl"][1][cols].mean(axis=1),
                "nowalk_o365": m["nowalk"][1][cols].mean(axis=1),
                "ctl_o365_t100": m["ctl"][1]["100"] if "100" in m["ctl"][1] else m["ctl"][1][100],
                "nowalk_o365_t100": m["nowalk"][1]["100"] if "100" in m["nowalk"][1] else m["nowalk"][1][100],
                "ctl_returned_t100": m["ctl"][2]["100"] if "100" in m["ctl"][2] else m["ctl"][2][100],
                "nowalk_returned_t100": m["nowalk"][2]["100"] if "100" in m["nowalk"][2] else m["nowalk"][2][100],
            }
        )
        ex = pd.concat([cells, ex.reset_index(drop=True)], axis=1)
        ex["price_o365"] = ex["nowalk_o365"] - ex["ctl_o365"]
        ex["price_in"] = ex["nowalk_in"] - ex["ctl_in"]
        ex = ex.sort_values("price_o365")
        ex.to_csv(R / "analysis-o365" / "examples_b1.csv", index=False, float_format="%.4g")
        print(pd.concat([ex.head(3), ex.tail(3)]).drop(columns=["dataset", "embedder"]).round(3).to_string(index=False))
    for lo, hi in ((1, 150), (1, 50), (51, 150)):
        cols = [c for c in m["ctl"][0].columns if lo <= int(c) <= hi]
        d_in = m["nowalk"][0][cols].mean(axis=1) - m["ctl"][0][cols].mean(axis=1)
        d_f = m["nowalk"][1][cols].mean(axis=1) - m["ctl"][1][cols].mean(axis=1)
        g = (d_f - d_in).dropna()
        rows.append(
            {
                "preset": p,
                "window": f"{lo}-{hi}",
                "runs": len(g),
                "price_in": d_in.mean(),
                "price_o365": d_f.mean(),
                "gap": g.mean(),
                "gap_se": g.std(ddof=1) / np.sqrt(len(g)),
            }
        )
out = pd.DataFrame(rows)
out.to_csv(R / "analysis-o365" / "price_gap.csv", index=False, float_format="%.6g")
print(out.round(4).to_string(index=False))
