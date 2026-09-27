"""#3959 interactive viewers, one per dataset, through the one implementation (``calibration/viewer.py``).

``coco_better``'s 144 cells are class x size band, and a page carrying every band
as its own category (288 groups x 8 arms x every metric x 151 clicks) is 10 MB.
The band is a stratum of the bench here, not a variable of the study, so the
page folds it into the run: category = class, and each band's seed becomes its
own run (``seed = 10 * band + seed``).  No run is dropped; the class mean pools
its bands, and the page's subtitle says so.  Even folded it sits at the repo's
4 MB file cap, so ``coco_better`` gets one page per embedder - which loses nothing,
because the viewer never averages embedders anyway.

    python viewer_3959.py --root /expscratch/$USER/gp-grid-3959 --out docs/experiments/2026-09-26-gp-head-grid-3959
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "calibration"))
sys.path.insert(0, str(HERE))

from analyze_grid_3959 import ARMS  # noqa: E402

BANDS = {"small": 0, "medium": 1, "large": 2}
DATASET_OF = {"better": "coco_better", "natural": "coco_val"}


def _fold_bands(df):
    split = df["category"].astype(str).str.split("@", n=1, expand=True)
    band = split[1].map(BANDS).fillna(0).astype(int)
    df = df.copy()
    df["category"] = split[0]
    df["seed"] = 10 * band + df["seed"].astype(int)
    return df


def main() -> int:
    import curves
    import viewer

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    for env, ds in DATASET_OF.items():
        base = args.root / env
        dirs = [f"{a}/results" for a in ARMS if (base / a / "results" / "cells").exists()]
        frame = curves._load(base, dirs)
        frame["arm"] = frame["arm"].str.replace("/results", "", regex=False)
        bl_path = base / "prepare" / "results" / "text_baseline.csv"
        baseline = curves.text_sort_baseline(bl_path) if bl_path.exists() else None
        subtitle = "8 arms, paired on the cell; one panel per embedder"
        if env == "better":
            frame = _fold_bands(frame)
            if baseline is not None:
                baseline = _fold_bands(baseline)
            subtitle += "; each class pools its three size bands (a run = one band x one seed)"
        pages = [(frame, baseline, args.out / ds / "viewer.html", ds)]
        if env == "better":
            pages = [
                (
                    frame[frame["embedder"] == emb],
                    None if baseline is None else baseline[baseline["embedder"] == emb],
                    args.out / ds / emb / "viewer.html",
                    f"{ds} · {emb}",
                )
                for emb in sorted(frame["embedder"].unique())
            ]
        for page_frame, page_baseline, out, label in pages:
            out.parent.mkdir(parents=True, exist_ok=True)
            viewer.build_viewer(
                page_frame,
                out,
                arms=[a for a in ARMS if a in set(page_frame["arm"])],
                baseline=page_baseline,
                title=f"GP head vs the shipped SVM, {label} (#3959)",
                subtitle=subtitle,
                runs_budget_mb=1.0,
                build={"results": str(base), "arms": ",".join(dirs), "baseline": str(bl_path)},
            )
            print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
