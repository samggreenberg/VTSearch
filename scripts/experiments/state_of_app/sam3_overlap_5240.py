"""#5240: is #5213's SAM3 lead memorisation? Grade it apart on the COCO images SAM3 trained on.

SAM3's training sets (arXiv 2511.16719, Table 26) name no COCO, but SA-Co/EXT holds Visual Genome, and VG's
images are COCO 2014 images (51,208 carry a ``coco_id``). So #5213's pool splits, inside the same cells, into
images SAM3 saw with VG's dense noun labels and images it was never given labels for. A remembered COCO image
lifts SAM3 on the first half and not the second; SigLIP (no VG labels) is the control for any difference
between the halves themselves.

    sam3_overlap_5240.py ids --out DIR        DIR/{vg,novg}_ids.txt: the dataset's media ids by VG membership
    sam3_overlap_5240.py analyze --root DIR --out DIR
        reads DIR/{siglip,sam3_typed,sam3_noun}_{all,vg,novg}.csv (``text_baseline.py --keep-ids`` rows)

The two id files partition every test half, so a cell's ``vg`` and ``novg`` rows are disjoint slices of its
#5213 row. SEs are over classes, as in #5213.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

VG_IMAGES = Path("/exp/scale26/datasets/external/VisualGenome/annotations/image_data.json")
ARMS = ("siglip", "sam3_typed", "sam3_noun")
HALVES = ("all", "vg", "novg")
BANDS = ("all", "small", "medium", "large")


def write_ids(out: Path, dataset: str, embedder: str) -> None:
    import common  # noqa: PLC0415

    common.setup_env()
    import experiment_config as cfg  # noqa: PLC0415
    from _cells_io import load_medias  # noqa: PLC0415

    from vtscore.config import EMBEDDINGS_DIR  # noqa: PLC0415

    vg = {d["coco_id"] for d in json.loads(VG_IMAGES.read_text()) if d.get("coco_id")}
    medias = load_medias(EMBEDDINGS_DIR / cfg.text_pickle_name(dataset, embedder))
    split: dict[str, list[int]] = {"vg": [], "novg": []}
    for mid, m in medias.items():
        hit = re.search(r"(\d+)\.jpg$", m["origin_name"])
        if not hit:
            raise SystemExit(f"{mid}: no COCO id in {m['origin_name']!r}")
        split["vg" if int(hit.group(1)) in vg else "novg"].append(mid)
    out.mkdir(parents=True, exist_ok=True)
    for k, v in split.items():
        (out / f"{k}_ids.txt").write_text("".join(f"{i}\n" for i in sorted(v)))
    n = len(medias)
    print(f"{n} medias: {len(split['vg'])} in VG ({len(split['vg']) / n:.1%}), {len(split['novg'])} not")


def load(root: Path) -> pd.DataFrame:
    frames = []
    for arm in ARMS:
        for half in HALVES:
            d = pd.read_csv(root / f"{arm}_{half}.csv")
            if arm == "siglip":
                d = d[d["embedder"] == "siglip"]
            frames.append(
                pd.DataFrame(
                    {
                        "category": d["category"],
                        "seed": d["seed"],
                        "arm": arm,
                        "half": half,
                        "n_test": d["n_test"],
                        "n_pos": d["n_test_pos"],
                        "prevalence": d["prevalence"],
                        "AP": d["text_AP"],
                        "auroc": d["text_auroc"],
                        "oracle": d["text_oracle_fbeta_b1"],
                        "F_native": d.get("text_native_f1", np.nan),
                    }
                )
            )
    df = pd.concat(frames, ignore_index=True)
    df["band"] = df["category"].str.rpartition("@")[2]
    df["cls"] = df["category"].str.rpartition("@")[0]
    return df


def class_se(g: pd.DataFrame, col: str) -> tuple[float, float, int]:
    """Mean over cells, SE over class means."""
    per = g.groupby("cls")[col].mean().dropna()
    se = float(per.std(ddof=1) / np.sqrt(len(per))) if len(per) > 1 else np.nan
    return float(g[col].mean()), se, len(per)


def analyze(root: Path, out: Path) -> None:
    df = load(root)
    out.mkdir(parents=True, exist_ok=True)
    # Paired frame: one row per (cell, seed) present in BOTH halves for every arm, so no half wins by losing cells.
    wide = df.pivot_table(index=["category", "seed", "band", "cls"], columns=["arm", "half"], values="AP")
    wide = wide.dropna()
    counts = df[df["arm"] == "siglip"].pivot_table(
        index=["category", "seed"], columns="half", values=["n_test", "n_pos"]
    )
    rows = []
    for metric in ("AP", "auroc", "oracle", "F_native"):
        w = df.pivot_table(index=["category", "seed", "band", "cls"], columns=["arm", "half"], values=metric)
        w = w.loc[wide.index].reset_index()
        for band in BANDS:
            sub = w if band == "all" else w[w["band"] == band]
            for arm in ARMS:
                if metric == "F_native" and arm == "siglip":
                    continue
                for half in HALVES:
                    g = pd.DataFrame({"cls": sub["cls"], "v": sub[(arm, half)]})
                    m, se, nc = class_se(g, "v")
                    rows.append(
                        {
                            "metric": metric,
                            "band": band,
                            "arm": arm,
                            "half": half,
                            "mean": m,
                            "se": se,
                            "cells": len(sub),
                            "classes": nc,
                        }
                    )
                # The memorisation read: SAM3's lead over SigLIP on seen images minus its lead on unseen ones.
                if arm != "siglip" and metric != "F_native":
                    lead_vg = sub[(arm, "vg")] - sub[("siglip", "vg")]
                    lead_no = sub[(arm, "novg")] - sub[("siglip", "novg")]
                    for name, v in (
                        ("lead_vg", lead_vg),
                        ("lead_novg", lead_no),
                        ("did", lead_vg - lead_no),
                        ("self_vg_minus_novg", sub[(arm, "vg")] - sub[(arm, "novg")]),
                    ):
                        m, se, nc = class_se(pd.DataFrame({"cls": sub["cls"], "v": v}), "v")
                        rows.append(
                            {
                                "metric": metric,
                                "band": band,
                                "arm": arm,
                                "half": name,
                                "mean": m,
                                "se": se,
                                "cells": len(sub),
                                "classes": nc,
                            }
                        )
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "sam3_5240_summary.csv", index=False)

    # Per class (all bands, seed-averaged): where would memorisation show if it were concentrated?
    w = wide.reset_index()
    per = []
    for arm in ARMS[1:]:
        d = pd.DataFrame(
            {
                "cls": w["cls"],
                "did": (w[(arm, "vg")] - w[("siglip", "vg")]) - (w[(arm, "novg")] - w[("siglip", "novg")]),
                "self": w[(arm, "vg")] - w[(arm, "novg")],
            }
        )
        g = d.groupby("cls").mean()
        g["arm"] = arm
        per.append(g.reset_index())
    pd.concat(per).to_csv(out / "sam3_5240_per_class.csv", index=False)

    c = counts.reset_index()
    info = {
        "paired_cells": int(wide.reset_index()["category"].nunique()),
        "paired_rows": int(len(wide)),
        "dropped_rows": int(df[(df["arm"] == "siglip") & (df["half"] == "all")].shape[0] - len(wide)),
        "test_items_vg": int(c[("n_test", "vg")].sum()),
        "test_items_novg": int(c[("n_test", "novg")].sum()),
        "test_pos_vg": int(c[("n_pos", "vg")].sum()),
        "test_pos_novg": int(c[("n_pos", "novg")].sum()),
        "prevalence_vg": float(c[("n_pos", "vg")].sum() / c[("n_test", "vg")].sum()),
        "prevalence_novg": float(c[("n_pos", "novg")].sum() / c[("n_test", "novg")].sum()),
    }
    (out / "sam3_5240_counts.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info, indent=1))
    show = summ[(summ["metric"] == "AP")]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(show.pivot_table(index=["band", "arm"], columns="half", values="mean").round(3))
        print(show.pivot_table(index=["band", "arm"], columns="half", values="se").round(3))


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ids")
    a.add_argument("--out", required=True, type=Path)
    a.add_argument("--dataset", default="coco_better")
    a.add_argument("--embedder", default="siglip")
    b = sub.add_parser("analyze")
    b.add_argument("--root", required=True, type=Path)
    b.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    if args.cmd == "ids":
        write_ids(args.out, args.dataset, args.embedder)
    else:
        analyze(args.root, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
