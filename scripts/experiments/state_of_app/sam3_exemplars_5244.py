"""#5244: do a Region session's Good (and Bad) vote boxes, pasted beside the target as SAM3 exemplars, beat the text prompt?

Reads ``calibration/sam3_exemplars_5244.py``'s per-cell scores (six arms on two canvases), #5213's native
text-only scores (the same noun prompts, the image alone at full resolution), and #4711's shipped ``max_patch``
Region run. Every number is seed 0 on the cell's own withheld half, so the arms are the same images cell for cell.

Per cell and arm: ``AP``, ``oracle`` (best F1 of any cut on the ranking) and ``F_native`` (F1 of SAM3's own
operating point, a detection over 0.5). Region's ``AP`` and ``oracle`` are read at the vote the arm's last
exemplar arrived (the k-th Good, or the Bad if it came later): the detector the same session had at that point.

Paired gains are means over cells with SEs over classes (a class's bands are not independent), as in #5213.

    python sam3_exemplars_5244.py --root <root> --native <sam3-5213>/scores --region <mp run>/results/cells --out <dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calibration"))

from sam3_analyze_5213 import band_of, class_of, paired, summarize  # noqa: E402

ARMS = ("native", "c1", "g1", "g1b1", "c3", "g3", "g3b1")
LABEL = {
    "native": "SAM3 text, image alone (#5213)",
    "c1": "canvas A, text only",
    "g1": "canvas A, text + 1 Good box",
    "g1b1": "canvas A, text + 1 Good + 1 Bad box",
    "c3": "canvas B, text only",
    "g3": "canvas B, text + 3 Good boxes",
    "g3b1": "canvas B, text + 3 Good + 1 Bad box",
}


def oracle_f1(y: np.ndarray, s: np.ndarray) -> float:
    o = np.argsort(-s, kind="stable")
    tp = np.cumsum(y[o])
    k = np.arange(1, len(y) + 1)
    f = 2 * tp / (k + y.sum())
    return float(f.max()) if len(f) else float("nan")


def native_f1(y: np.ndarray, s: np.ndarray, cut: float = 0.5) -> tuple[float, float, float]:
    pred = s > cut
    tp = float((pred & (y == 1)).sum())
    p = tp / pred.sum() if pred.sum() else 0.0
    r = tp / y.sum()
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def region_at(cells: Path) -> dict[str, pd.DataFrame]:
    """cell -> its seed-0 ordinary rows (t, AP, oracle F1), one per vote."""
    out = {}
    for f in sorted(glob.glob(f"{cells}/task_*.csv*")):
        if "__" in Path(f).name:
            continue
        d = pd.read_csv(
            f, usecols=["seed", "category", "t", "phase", "embedder", "average_precision", "oracle_fbeta_b1"]
        )
        d = d[(d["seed"] == 0) & (d["embedder"] == "siglip+dinov3_patch") & (d["phase"].astype(str) != "check")]
        for cat, g in d.groupby("category"):
            out[cat] = g.drop_duplicates("t", keep="last").sort_values("t")[
                ["t", "average_precision", "oracle_fbeta_b1"]
            ]
    return out


def at_vote(g: pd.DataFrame, t: int) -> tuple[float, float]:
    """Region's AP and oracle F1 after vote *t* (the last row at or before it)."""
    h = g[g["t"] <= t]
    if h.empty:
        return float("nan"), float("nan")
    return float(h["average_precision"].iloc[-1]), float(h["oracle_fbeta_b1"].iloc[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--native", required=True, help="#5213's score shards (sam3_score_5213.py output)")
    ap.add_argument("--region", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from sam3_scores import load_sam3  # noqa: PLC0415

    native = load_sam3(Path(a.native), "noun")
    region = region_at(Path(a.region))
    rows, diag, cost = [], [], []
    for f in sorted((Path(a.root) / "scores").glob("*.npz")):
        if "_limit" in f.name:
            continue
        z = np.load(f, allow_pickle=True)
        meta = json.loads(str(z["meta"][0]))
        cat = meta["category"]
        ids, y = z["ids"], z["labels"].astype(int)
        nat, _ = native(cat, {int(i): None for i in ids})
        scores = {"native": np.asarray([nat[int(i)] for i in ids], np.float64)}
        for j, arm in enumerate(z["arms"]):
            scores[str(arm)] = z["score"][:, j].astype(np.float64)
        tg = [g["t"] for g in meta["goods"]]
        tb = meta["bad"]["t"]
        vote = {"native": 0, "c1": tg[0], "g1": tg[0], "g1b1": max(tg[0], tb)}
        if len(tg) >= 3:
            vote.update({"c3": tg[2], "g3": tg[2], "g3b1": max(tg[2], tb)})
        for arm, s in scores.items():
            fn, pn, rn = native_f1(y, s)
            rows.append(
                {
                    "category": cat,
                    "seed": 0,
                    "arm": arm,
                    "vote": vote.get(arm),
                    "AP": average_precision_score(y, s),
                    "oracle": oracle_f1(y, s),
                    "F_native": fn,
                    "P_native": pn,
                    "R_native": rn,
                    "k_native": int((s > 0.5).sum()),
                    "n_test": len(y),
                    "n_pos": int(y.sum()),
                }
            )
            if arm != "native" and cat in region:
                rap, ror = at_vote(region[cat], vote[arm])
                rows.append(
                    {"category": cat, "seed": 0, "arm": f"region@{arm}", "vote": vote[arm], "AP": rap, "oracle": ror}
                )
        for j, arm in enumerate(z["arms"]):
            diag.append(
                {
                    "category": cat,
                    "arm": str(arm),
                    "on_ref": float(z["on_ref"][:, j].mean()),
                    "presence_mean": float(z["presence"][:, j].mean()),
                    "presence_neg_mean": float(z["presence"][y == 0, j].mean()),
                    "presence_pos_mean": float(z["presence"][y == 1, j].mean()),
                    "bad_sam3_score": meta["bad"]["sam3_score"],
                }
            )
        el, tv, td, n, ncanv, narms, nprom = z["timing"].tolist()
        cost.append(
            {
                "category": cat,
                "gpu": meta["gpu"],
                "n": n,
                "s_per_img": el / n,
                "vision_ms_per_canvas": tv / n / ncanv * 1e3,
                "decode_ms_per_img_arm_prompt": td / n / narms / nprom * 1e3,
                "canvases": ncanv,
                "prompts": nprom,
            }
        )

    d = pd.DataFrame(rows)
    d["band"] = d["category"].map(band_of)
    d["class"] = d["category"].map(class_of)
    d.to_csv(out / "sam3_5244_per_cell.csv", index=False)
    cols = ["AP", "oracle", "F_native", "P_native", "R_native", "k_native", "vote"]
    s = summarize(d, cols)
    s.to_csv(out / "sam3_5244_summary.csv", index=False)

    # Compare on the cells every arm has (canvas B needs 3 Goods within 150 votes).
    b_cells = set(d.loc[d["arm"] == "g3", "category"])
    common_d = d[d["category"].isin(b_cells)]
    s_common = summarize(common_d, cols)
    s_common.to_csv(out / "sam3_5244_summary_common.csv", index=False)

    pairs = []
    for band in ("all", "small", "medium", "large"):
        sub = d if band == "all" else d[d["band"] == band]
        for x, ref in [
            ("g1", "native"),
            ("g1", "c1"),
            ("g1b1", "g1"),
            ("c1", "native"),
            ("g3", "native"),
            ("g3", "c3"),
            ("g3b1", "g3"),
            ("c3", "native"),
            ("g3", "g1"),
            ("g1", "region@g1"),
            ("g3", "region@g3"),
            ("g1b1", "region@g1b1"),
            ("g3b1", "region@g3b1"),
            ("native", "region@g1"),
            ("native", "region@g3"),
        ]:
            for col in ("AP", "oracle", "F_native"):
                if col == "F_native" and x.startswith("region") | ref.startswith("region"):
                    continue
                r = paired(sub, x, ref, col)
                r["group"] = band
                pairs.append(r)
    p = pd.DataFrame(pairs)
    p.to_csv(out / "sam3_5244_paired.csv", index=False)
    dg = pd.DataFrame(diag)
    dg.to_csv(out / "sam3_5244_diag.csv", index=False)
    dg.groupby("arm").mean(numeric_only=True).to_csv(out / "sam3_5244_diag_summary.csv")
    c = pd.DataFrame(cost)
    c.to_csv(out / "sam3_5244_cost.csv", index=False)
    pc = d[d["arm"].isin(ARMS)].pivot_table(index="class", columns="arm", values="AP")
    pc.to_csv(out / "sam3_5244_per_class_AP.csv")
    (out / "sam3_5244_counts.json").write_text(
        json.dumps(
            {"cells": int(d["category"].nunique()), "cells_canvas_b": len(b_cells), "arms": list(d["arm"].unique())}
        )
    )
    _figure(s, out)
    pd.set_option("display.width", 250)
    print(s.round(3).to_string())
    print(p[p["group"] == "all"].round(3).to_string())
    print(dg.groupby("arm").mean(numeric_only=True).round(3).to_string())
    print(c.groupby("gpu").median(numeric_only=True).round(3).to_string())
    return 0


def _figure(s: pd.DataFrame, out: Path) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    bands = ("all", "small", "medium", "large")
    arms = [x for x in ARMS if x in set(s["arm"])]
    colors = {
        "native": "#111827",
        "c1": "#9ca3af",
        "g1": "#2563eb",
        "g1b1": "#93c5fd",
        "c3": "#d1d5db",
        "g3": "#15803d",
        "g3b1": "#86efac",
    }
    for ax, metric in zip(axes, ("AP", "oracle")):
        w = 0.8 / (len(arms) + 2)
        for i, arm in enumerate(arms + ["region@g1", "region@g3"]):
            v = [s.loc[(s["group"] == b) & (s["arm"] == arm), metric] for b in bands]
            e = [s.loc[(s["group"] == b) & (s["arm"] == arm), f"{metric}_se"] for b in bands]
            v = [float(x.iloc[0]) if len(x) else np.nan for x in v]
            e = [float(x.iloc[0]) if len(x) else np.nan for x in e]
            col = colors.get(arm, "#dc2626" if arm == "region@g1" else "#f97316")
            ax.bar(
                np.arange(4) + (i - len(arms) / 2) * w,
                v,
                w,
                yerr=e,
                color=col,
                label=LABEL.get(arm, f"Region at that vote ({arm[7:]})"),
            )
        ax.set_xticks(np.arange(4), bands)
        ax.set_title({"AP": "AP on the withheld half", "oracle": "Oracle F1 (best cut)"}[metric])
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "sam3_5244_bands.png", dpi=130)


if __name__ == "__main__":
    sys.exit(main())
