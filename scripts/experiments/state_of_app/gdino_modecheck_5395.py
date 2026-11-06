"""#5395: which GroundingDINO protocol ranks the positives better, on a subset of the pool?

One prompt per forward pass and prompts packed into a caption rank the pool very differently (median Spearman ~0.3
over 50 images), so agreement cannot choose between them; the labels can. Each dir holds ``gdino_score_5395.py``
shards over the same subset; ``--sam3`` adds #5213's scores on those images as the anchor.

The grade is class level: an image is a positive for a class when it holds it at any size (the cells' bands are a
pool rule, not the detector's question), and AP is over the subset, per class and prompt set (a merged class takes
the max over its prompts, as ``sam3_scores.load_sam3`` does). SEs are over classes.

    python gdino_modecheck_5395.py --dirs tiny-c250=DIR tiny-single=DIR ... --sam3 <sam3>/scores --out CSV
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load(root: Path):
    ids, top1, prompts = [], [], None
    for p in sorted(root.glob("shard_*_of_*.npz")):
        z = np.load(p, allow_pickle=True)
        prompts = list(z["prompts"])
        ids.extend(z["ids"].tolist())
        top1.append(z["top1"].astype(np.float32))
    table = json.loads((root / "prompts.json").read_text())["classes"]
    return ids, np.concatenate(top1), {q: j for j, q in enumerate(prompts)}, table


def main() -> int:
    import common  # noqa: PLC0415

    common.setup_env()
    from sklearn.metrics import average_precision_score  # noqa: PLC0415

    from _cells_io import load_medias  # noqa: PLC0415
    from vtscore.config import EMBEDDINGS_DIR  # noqa: PLC0415

    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="label=dir")
    ap.add_argument("--sam3", default=None, help="#5213's scores dir")
    ap.add_argument("--dataset", default="coco_better")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    runs = {lab: load(Path(d)) for lab, d in (x.split("=", 1) for x in a.dirs)}
    ids = runs[next(iter(runs))][0]
    for lab, r in runs.items():
        if r[0] != ids:
            raise SystemExit(f"{lab}: a different image list")
    if a.sam3:
        s_ids, s_mat, s_col, s_table = load(Path(a.sam3))
        row = {i: k for k, i in enumerate(s_ids)}
        runs["sam3"] = (ids, s_mat[[row[i] for i in ids]], s_col, s_table)

    medias = load_medias(EMBEDDINGS_DIR / f"{a.dataset}__siglip.pkl")
    cats = [set(medias[i].get("categories") or [medias[i].get("category")]) for i in ids]
    del medias

    rows = []
    table = runs[next(iter(runs))][3]
    for cls, kinds in table.items():
        y = np.asarray([any(c.rpartition("@")[0] == cls or c == cls for c in cs) for cs in cats], int)
        if y.sum() == 0:
            continue
        for kind in ("typed", "noun"):
            for lab, (_, mat, col, _) in runs.items():
                s = mat[:, [col[q] for q in kinds[kind]]].max(axis=1)
                rows.append(
                    {"class": cls, "set": kind, "arm": lab, "n_pos": int(y.sum()), "AP": average_precision_score(y, s)}
                )
    d = pd.DataFrame(rows)
    d.to_csv(a.out, index=False)
    print(f"{len(ids)} images, {d['class'].nunique()} classes with a positive")
    piv = d.pivot_table(index=["set", "class"], columns="arm", values="AP")
    for kind in ("typed", "noun"):
        p = piv.loc[kind]
        se = p.std(ddof=1) / np.sqrt(len(p))
        print(f"\n{kind}: mean AP (SE over {len(p)} classes)")
        print(pd.DataFrame({"AP": p.mean(), "se": se}).round(3).to_string())
        arms = list(p.columns)
        for i, x in enumerate(arms):
            for z in arms[i + 1 :]:
                dd = p[x] - p[z]
                print(
                    f"  {x} - {z}: {dd.mean():+.3f} +- {dd.std(ddof=1) / np.sqrt(len(dd)):.3f}  ({(dd > 0).mean():.0%} ahead)"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
