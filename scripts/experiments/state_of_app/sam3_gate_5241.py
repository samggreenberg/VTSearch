"""#5241: a label-only rule for when the head takes over from SAM3's set, priced on the SAM3 arm's Binary sessions.

SAM3's set at 0.5 beats today's SigLIP opening on Binary, the head never catches it on average, but per cell the
better of the two at each click is worth +0.07 F1 more than holding SAM3 (``sam3_opening_5241.py``'s ``best``). A
rule may only use what the app has: the labels. #4776's judge does that for the centroid: each labelled Good is held
out, the head is refitted on the rest, and the Good's rank in the unvoted pool under the head is set beside its rank
under the centroid (``__handover.csv``). Here the other side is SAM3's sort, which is not fitted on the labels, so a
Good's rank under it is read straight off the pool. The judge prefers the head when its ranks are better
(``mrr``/``logrank``/``median``/``wins``, as #4776), plain at each row or sticky once it has preferred it; count
rules (``goods>=N``, the Goods the user has labelled, check votes included) are priced beside it on both paths.
The judge's rows exist only on single-vector cells: Binary only.

Picks never depend on what Test shows, so every rule is a choice between the two rows of one session.

    python sam3_gate_5241.py collect --out DIR   # per (cell, click): head ranks and SAM3 ranks of the same Goods
    python sam3_gate_5241.py price --out DIR     # F1 shown under each rule -> DIR/sam3_5241_gate.csv

``collect`` loads the cells' media pickle and SAM3's shards (run it under ``sam3_5213.sh``'s environment).
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CAL = HERE.parent / "calibration"
ROOT = Path("/expscratch/sgreenberg/state-of-the-app")
RUN = "2026-10-26-sam3open5241-{arm}-b1"
STATS = ("mrr", "logrank", "median", "wins")
GOODS = (4, 8, 12, 16, 20, 24, 32, 48)
PATHS = {"siglip": "Binary", "siglip+dinov3_patch": "Region"}


def _ranks(s: object) -> np.ndarray:
    return np.array([int(x) for x in str(s).split()], dtype=float) if isinstance(s, str) and s.strip() else np.empty(0)


def collect(out: Path, scores_root: Path) -> None:
    sys.path.insert(0, str(CAL))
    import common  # noqa: PLC0415

    common.setup_env()
    import experiment_config as cfg  # noqa: PLC0415
    from _cells_io import load_medias  # noqa: PLC0415
    from sam3_scores import load_sam3  # noqa: PLC0415

    from vtscore.config import EMBEDDINGS_DIR  # noqa: PLC0415
    from vtscore.eval.labels import evaluable_pool  # noqa: PLC0415
    from vtscore.eval.voting_iterations import _split_media_ids  # noqa: PLC0415

    cells = Path(ROOT / RUN.format(arm="sam3") / "results" / "cells")
    medias = load_medias(EMBEDDINGS_DIR / cfg.text_pickle_name("coco_better", "siglip"))
    sam3 = load_sam3(scores_root, "typed")
    rows = []
    for hp in sorted(glob.glob(f"{cells}/task_*__handover.csv*")):
        ho = pd.read_csv(hp)
        if ho.empty:
            continue
        picks = pd.read_csv(hp.replace("__handover", "__picks"))
        cat, seed = str(ho["category"].iloc[0]), int(ho["seed"].iloc[0])
        pool_cat = evaluable_pool(medias, cat)
        sim, _ = _split_media_ids(pool_cat, cfg.SIM_FRACTION, np.random.RandomState(seed))
        s, _ = sam3(cat, pool_cat)
        corpus = np.sort(np.array([s[i] for i in sim]))
        picks = picks.sort_values("t")
        for _, r in ho.drop_duplicates("t", keep="last").iterrows():
            t = int(r["t"])
            voted = picks[picks["t"] <= t]
            # good_votes is a dict in vote order, so the k-th held-out rank is the k-th Good voted.
            goods = list(dict.fromkeys(int(i) for i in voted.loc[voted["picked_label"] == 1, "picked_id"]))
            if len(goods) != int(r["n_good"]) or len(sim) - len(set(voted["picked_id"])) != int(r["n_pool"]):
                raise SystemExit(f"{hp} t={t}: votes do not match the judge's row (goods {len(goods)} vs "
                                 f"{r['n_good']}, pool {len(sim) - len(set(voted['picked_id']))} vs {r['n_pool']})")  # fmt: skip
            vs = np.array([s[int(i)] for i in set(voted["picked_id"])])
            sam3_ranks = [
                1 + int(len(corpus) - np.searchsorted(corpus, s[g], side="right")) - int(np.sum(vs > s[g]))
                for g in goods
            ]
            rows.append({"category": cat, "seed": seed, "t": t, "n_good": int(r["n_good"]), "n_bad": int(r["n_bad"]),
                         "head_ranks": r["head_ranks"], "sam3_ranks": " ".join(map(str, sam3_ranks))})  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "sam3_5241_gate_evidence.csv", index=False)
    common.log(f"{len(rows)} judge rows from {len(set(r['category'] for r in rows))} cells")


def prefers_head(sam3: np.ndarray, head: np.ndarray, stat: str) -> bool | None:
    """#4776's ``judge_prefers_head`` with SAM3 in the centroid's seat; ``None`` with nothing held out."""
    if not len(head) or len(head) != len(sam3):
        return None
    if stat == "mrr":
        return float(np.mean(1.0 / head)) > float(np.mean(1.0 / sam3))
    if stat == "logrank":
        return float(np.mean(np.log2(head))) < float(np.mean(np.log2(sam3)))
    if stat == "median":
        return float(np.median(head)) < float(np.median(sam3))
    return int(np.sum(head < sam3)) > int(np.sum(head > sam3))


def price(out: Path) -> None:
    sys.path.insert(0, str(HERE))
    from early_check_4664 import area  # noqa: PLC0415
    from sam3_opening_5241 import HOLD_TAG, T, curves, load, mean_se  # noqa: PLC0415

    ev = pd.read_csv(out / "sam3_5241_gate_evidence.csv")
    sig = load(ROOT / RUN.format(arm="sig") / "results" / "cells", 8)
    cells = ROOT / RUN.format(arm="sam3") / "results" / "cells"
    sam3 = load(cells, 8)
    s_shown, q_shown, q_head = curves(sig, ""), curves(sam3, ""), curves(sam3, HOLD_TAG)
    # Goods labelled by each click, per (cell, path), off the session's own picks.
    n_goods: dict[tuple[str, str], np.ndarray] = {}
    for pp in sorted(glob.glob(f"{cells}/task_*__picks.csv*")):
        pk = pd.read_csv(pp, usecols=["category", "embedder", "t", "picked_label"])
        if pk.empty:
            continue
        g = np.zeros(T + 1)
        for t in pk.loc[(pk["picked_label"] == 1) & (pk["t"] <= T), "t"].astype(int):
            g[t:] += 1
        n_goods[(str(pk["category"].iloc[0]), str(pk["embedder"].iloc[0]))] = g
    rows, per_click, info = [], [], {"judge_rows": len(ev)}
    for emb, path in PATHS.items():
        keys = sorted(k for k in set(s_shown) & set(q_shown) & set(q_head) & set(n_goods) if k[1] == emb)
        cats = [k[0] for k in keys]
        cls = np.array([c.rpartition("@")[0] for c in cats])
        band = np.array([c.rpartition("@")[2] for c in cats])
        sig_f = np.vstack([s_shown[k][0] for k in keys])
        q0 = np.vstack([q_shown[k][0] for k in keys])[:, 1]
        head = np.vstack([q_head[k][0] for k in keys])
        has = np.isfinite(head)
        good = np.vstack([n_goods[k] for k in keys])
        rules: dict[str, np.ndarray] = {"hold": np.zeros_like(has), "quota": has, "best": has & (head > q0[:, None])}
        for g in GOODS:
            rules[f"goods>={g}"] = has & (good >= g)
        if path == "Binary":
            for s in STATS:
                rules[f"judge:{s}"] = np.zeros_like(has)
                rules[f"judge:{s} sticky"] = np.zeros_like(has)
            for k, c in enumerate(cats):
                e = ev[ev["category"] == c].set_index("t").sort_index()
                dec = {s: np.zeros(T + 1, bool) for s in STATS}
                for t, r in e.iterrows():
                    if t > T:
                        continue
                    for s in STATS:
                        p = prefers_head(_ranks(r["sam3_ranks"]), _ranks(r["head_ranks"]), s)
                        dec[s][t:] = bool(p)  # carried to the next judged click; None (too few Goods) keeps SAM3
                for s in STATS:
                    rules[f"judge:{s}"][k] = has[k] & dec[s]
                    rules[f"judge:{s} sticky"][k] = has[k] & np.maximum.accumulate(dec[s])
        info[path] = {"cells": len(keys), "goods_at_150_median": float(np.median(good[:, T]))}
        hold_f = np.repeat(q0[:, None], T + 1, axis=1)
        for t in range(1, T + 1):
            per_click.append({"path": path, "t": t, "sig": float(np.nanmean(sig_f[:, t])),
                              **{r: float(np.nanmean(np.where(sh, head, q0[:, None])[:, t])) for r, sh in rules.items()}})  # fmt: skip
        for name, m in (("all", np.ones(len(keys), bool)), *((b, band == b) for b in ("small", "medium", "large"))):
            for rule, show in rules.items():
                f = np.where(show, head, q0[:, None])
                row = {"path": path, "band": name, "rule": rule, "cells": int(m.sum()),
                       "share_head_150": float(show[m][:, 1:].mean())}  # fmt: skip
                for lo, hi in ((1, 50), (1, 150)):
                    a = area(f[m], lo, hi)
                    row[f"F_{lo}_{hi}"], row[f"se_{lo}_{hi}"] = mean_se(a, cls[m])
                    d_hold, d_sig = a - area(hold_f[m], lo, hi), a - area(sig_f[m], lo, hi)
                    row[f"vs_hold_{lo}_{hi}"], row[f"vs_hold_se_{lo}_{hi}"] = mean_se(d_hold, cls[m])
                    row[f"vs_sig_{lo}_{hi}"], row[f"vs_sig_se_{lo}_{hi}"] = mean_se(d_sig, cls[m])
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "sam3_5241_gate.csv", index=False)
    pc = pd.DataFrame(per_click)
    pc.to_csv(out / "sam3_5241_gate_per_click.csv", index=False)
    figure(pc, out / "sam3_5241_gate.png")
    (out / "sam3_5241_gate_info.json").write_text(json.dumps(info, indent=1) + "\n")
    cols = ["path", "rule", "share_head_150", "F_1_50", "F_1_150", "vs_hold_1_150", "vs_hold_se_1_150",
            "vs_sig_1_150", "vs_sig_se_1_150"]  # fmt: skip
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(df[df["band"] == "all"][cols].round(3).to_string(index=False))


def figure(pc: pd.DataFrame, path: Path, rule: str = "goods>=20") -> None:
    """F1 shown per click on each path: today, SAM3 held, SAM3 to the head at the quota and at *rule*."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from early_check_4664 import MUTED, _axes_style  # noqa: PLC0415

    lines = (("sig", "SigLIP opening (today)", "#8a8985", 2), ("hold", "SAM3 set, held", "#2a78d6", 2),
             ("quota", "SAM3 -> head at quota", "#d65f2a", 1.2), (rule, f"SAM3 -> head at {rule}", "#3a9a5b", 2.4),
             ("best", "better of the two (bound)", "#b8b6b0", 1))  # fmt: skip
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (name, g) in zip(axes, pc.groupby("path", sort=False), strict=False):
        for col, label, color, lw in lines:
            ax.plot(g["t"], g[col], color=color, lw=lw, label=label, ls=":" if col == "best" else "-")
        ax.set_title(name, loc="left", fontsize=11)
        ax.set_xlabel("votes", color=MUTED)
        _axes_style(ax)
    axes[0].set_ylabel("F1 shown (mean over cells)", color=MUTED)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "price"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scores", type=Path, default=Path("/expscratch/sgreenberg/sam3-5213/scores"))
    a = ap.parse_args()
    collect(a.out, a.scores) if a.mode == "collect" else price(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
