"""#5490: does SAM3 suffer more than SigLIP on a class no model knows? Grading half.

``zeroshot``: every model's sort at every prompt level of ``pilebuild/loaders/fgvc_aircraft.py``, graded per cell
(a variant with siblings) at four depths:

- ``ap_variant``: the cell's 100 photos among the whole pool (20,000 images), what the opening has to find;
- ``ap_sibling``: the cell's photos among its family's photos only, so whether the text tells a 737-400 from a
  737-300 (chance is 1 / the family's variant count);
- ``ap_family``: the family's photos among the whole pool, so whether the text at least finds the right family;
- ``auroc_aircraft``: FGVC's 10,000 photos against the 10,000 COCO distractors, so whether it finds aircraft at all.

SAM3 and GroundingDINO score from their shards (``sam3_score_5213.py`` / ``gdino_score_5395.py`` run with
``--dataset fgvc_aircraft``: one prompt per level per cell); SigLIP scores as the app's typed sort does
(``embed_text_query`` into the whole-image style's ``exemplar_sims``, as ``text_baseline.py``). Two more columns
show a detector's collapse: ``n_cut`` (how many images clear its own cut, 0.5 for SAM3 and 0.35 for GroundingDINO)
and ``pos_median`` (the median score of the cell's photos).

A family's cells share every level's text but ``variant``, so a paired difference is averaged over cells and its
SE taken over the 16 families (cluster SE): 46 cells are not 46 independent prompts.

    python unknown_class_5490.py zeroshot --results <grid>/results --scores sam3=DIR gdino=DIR --out DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

#: Each detector's own cut (#5213: SAM3 0.5; #5395: GroundingDINO's 0.35).
NATIVE_CUT = {"sam3": 0.5, "gdino": 0.35}


def _shards(root: Path):
    """``(ids, top1 matrix, {prompt: column})`` from a scorer's full shards."""
    shards = sorted(p for p in root.glob("shard_*_of_*.npz") if "limit" not in p.name)
    n = int(shards[0].name.split("_of_")[1].split(".")[0])
    if len(shards) != n:
        raise SystemExit(f"{root}: {len(shards)} of {n} shards present")
    ids, top1, prompts = [], [], None
    for p in shards:
        z = np.load(p, allow_pickle=True)
        prompts = list(z["prompts"]) if prompts is None else prompts
        if list(z["prompts"]) != prompts:
            raise SystemExit(f"{p}: prompt list differs from the first shard's")
        ids.extend(z["ids"].tolist())
        top1.append(z["top1"].astype(np.float32))
    return ids, np.concatenate(top1), {q: j for j, q in enumerate(prompts)}


def _cluster_se(diff: pd.Series, fam: pd.Series) -> float:
    """SE of a cell mean, clustered by family: the SE of the family means, weighted by family size."""
    g = pd.DataFrame({"d": diff.values, "f": fam.values}).groupby("f")["d"]
    k = g.ngroups
    if k < 2:
        return float("nan")
    resid = (g.sum() - g.count() * diff.mean()).to_numpy()
    return float(np.sqrt(k / (k - 1) * (resid**2).sum()) / len(diff))


def zeroshot(a) -> int:
    import common  # noqa: PLC0415

    common.setup_env()
    from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: PLC0415

    import experiment_config as cfg  # noqa: PLC0415
    from _cells_io import load_medias  # noqa: PLC0415
    from vtscore.config import EMBEDDINGS_DIR  # noqa: PLC0415
    from vtscore.embedding import embed_text_query  # noqa: PLC0415
    from vtscore.eval.patch_styles import resolve_style  # noqa: PLC0415

    sys.path.append(str(Path(cfg.__file__).resolve().parent.parent / "pile"))
    from pilebuild.loaders import fgvc_aircraft as fa  # noqa: PLC0415

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prepare = json.loads((Path(a.results) / "prepare_info.json").read_text())
    cells = sorted(next(iter(prepare["datasets"]["fgvc_aircraft"].values()))["selected_categories"])
    if cells != fa.cells():
        raise SystemExit("the grid's cells are not the loader's")

    medias = load_medias(EMBEDDINGS_DIR / "fgvc_aircraft__siglip.pkl")
    ids = sorted(medias)
    lab = [medias[i].get("fgvc") or {} for i in ids]
    variant = np.asarray([x.get("variant", "") for x in lab])
    family = np.asarray([x.get("family", "") for x in lab])
    aircraft = np.asarray([bool(x) for x in lab])

    # Every model's score matrix over `ids`, one column per prompt.
    texts = sorted({t for c in cells for t in fa.prompt_levels(c).values()})
    style = resolve_style("whole_image")
    sig = np.zeros((len(ids), len(texts)), np.float32)
    for j, t in enumerate(texts):
        vec = embed_text_query(t, "image", enrich=cfg.SEED_ENRICH, embedder_name="siglip")
        sims = style.exemplar_sims(medias, np.asarray(vec, dtype=np.float32))
        sig[:, j] = [float(sims[i]) for i in ids]
    models = {"siglip": (sig, {t: j for j, t in enumerate(texts)})}
    del medias
    for spec in a.scores:
        name, root = spec.split("=", 1)
        if not Path(root).is_dir():
            common.log(f"SKIP {name}: no {root}")
            continue
        s_ids, mat, col = _shards(Path(root))
        row = {i: k for k, i in enumerate(s_ids)}
        missing = [i for i in ids if i not in row]
        if missing:
            raise SystemExit(f"{name}: {len(missing)} pool images unscored (e.g. {missing[:3]})")
        models[name] = (mat[[row[i] for i in ids]], col)

    rows = []
    for c in cells:
        fam = fa.family_of(c)
        pos = variant == c
        in_fam = family == fam
        for level, text in fa.prompt_levels(c).items():
            for name, (mat, col) in models.items():
                s = mat[:, col[text]]
                cut = NATIVE_CUT.get(name)
                rows.append(
                    {
                        "model": name,
                        "level": level,
                        "cell": c,
                        "family": fam,
                        "n_siblings": len(fa.FAMILIES[fam].variants),
                        "text": text,
                        "ap_variant": average_precision_score(pos, s),
                        "ap_sibling": average_precision_score(pos[in_fam], s[in_fam]),
                        "ap_family": average_precision_score(in_fam, s),
                        "auroc_aircraft": roc_auc_score(aircraft, s),
                        "n_cut": int((s > cut).sum()) if cut is not None else np.nan,
                        "pos_median": float(np.median(s[pos])),
                        "neg_median": float(np.median(s[~aircraft])),
                    }
                )
    d = pd.DataFrame(rows)
    d.to_csv(out / "zeroshot_per_cell.csv", index=False)

    metrics = ["ap_variant", "ap_sibling", "ap_family", "auroc_aircraft", "n_cut", "pos_median"]
    summ = []
    for (m, lv), g in d.groupby(["model", "level"], sort=False):
        r = {"model": m, "level": lv, "n_cells": len(g)}
        for k in metrics:
            r[k] = g[k].mean()
            r[f"{k}_se"] = _cluster_se(g[k], g["family"])
        summ.append(r)
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "zeroshot_summary.csv", index=False)

    # Chance for each depth, per cell, averaged the same way.
    n_pool = len(ids)
    chance = pd.DataFrame(
        {
            "ap_variant": [100 / n_pool for _ in cells],
            "ap_sibling": [1 / len(fa.FAMILIES[fa.family_of(c)].variants) for c in cells],
            "ap_family": [float((family == fa.family_of(c)).sum()) / n_pool for c in cells],
        }
    ).mean()
    chance.to_csv(out / "zeroshot_chance.csv", header=["chance"])

    paired = []
    for other in [m for m in models if m != "siglip"]:
        for lv in fa.LEVELS:
            x = d[(d.model == other) & (d.level == lv)].set_index("cell")
            y = d[(d.model == "siglip") & (d.level == lv)].set_index("cell").loc[x.index]
            for k in ("ap_variant", "ap_sibling", "ap_family", "auroc_aircraft"):
                diff = x[k] - y[k]
                paired.append(
                    {
                        "model": other,
                        "vs": "siglip",
                        "level": lv,
                        "metric": k,
                        "diff": diff.mean(),
                        "se": _cluster_se(diff, x["family"]),
                        "cells_ahead": float((diff > 0).mean()),
                    }
                )
    pd.DataFrame(paired).to_csv(out / "zeroshot_paired.csv", index=False)

    pd.set_option("display.width", 200)
    print(summ[["model", "level"] + metrics].round(3).to_string(index=False))
    print("\nchance:", chance.round(4).to_dict())
    print(pd.DataFrame(paired).round(3).to_string(index=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    z = sub.add_parser("zeroshot")
    z.add_argument("--results", required=True, help="the grid's results dir (prepare_info.json names the cells)")
    z.add_argument("--scores", nargs="*", default=[], help="name=dir of a scorer's shards")
    z.add_argument("--out", required=True)
    a = ap.parse_args()
    return {"zeroshot": zeroshot}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
