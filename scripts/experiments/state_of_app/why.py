#!/usr/bin/env python3
"""State of the App: WHY a class does poorly, read off what the loop showed the user.

    python why.py --exp <run dir> --analysis <run dir>/analysis-binary --out <dir> [--classes a,b,...]

Two questions per class, answered from the runs themselves:

* **Could it be learned?** The full-label ceiling says whether the head can
  separate the class at all with every label. A high ceiling AP and a low final
  AP is the LOOP's failure (it never collected the labels); a low ceiling is
  the embedding's.
* **What did the user get shown instead?** Every Bad click is an image the
  detector ranked highly that was not the class. Pooled over seeds, the most
  often picked negatives are the class's standing confusions, and what COCO
  says those images DO hold (their largest annotated object, any of COCO's 80
  categories) names the confusion, ranked by LIFT over the corpus.

Writes ``why.csv`` (per class: ceiling, final, positives found, the share of
Bad clicks by what they hold) and ``confusers_<class>.jpg`` (the most-picked
negatives, whole and small, for the report).
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pile"))

import pile_config as pc  # noqa: E402


def _dominant(anchor: Path) -> dict[int, str]:
    """Each image's largest non-crowd COCO object, over all 80 categories."""
    best: dict[int, tuple[float, str]] = {}
    for split in ("val2017", "train2017"):
        d = json.loads((anchor / f"instances_{split}.json").read_text())
        cats = {c["id"]: c["name"] for c in d["categories"]}
        for a in d["annotations"]:
            if a.get("iscrowd"):
                continue
            area = a["bbox"][2] * a["bbox"][3]
            iid = int(a["image_id"])
            if area > best.get(iid, (0.0, ""))[0]:
                best[iid] = (area, cats[a["category_id"]])
    return {iid: name for iid, (_, name) in best.items()}


def _sheet(ids: list[int], path: Path, px: int = 150, cols: int = 6) -> None:
    from PIL import Image  # noqa: PLC0415

    members = {}
    for zp in (pc.COCO_VAL_ZIP, pc.COCO_TRAIN_ZIP):
        zf = zipfile.ZipFile(zp)
        for nm in zf.namelist():
            if nm.endswith(".jpg"):
                members[Path(nm).name] = (zf, nm)
    tiles = []
    for iid in ids:
        zf, nm = members[f"{iid:012d}.jpg"]
        im = Image.open(io.BytesIO(zf.read(nm))).convert("RGB")
        im.thumbnail((px, px))
        tile = Image.new("RGB", (px, px), (255, 255, 255))
        tile.paste(im, ((px - im.width) // 2, (px - im.height) // 2))
        tiles.append(tile)
    rows = (len(tiles) + cols - 1) // cols
    out = Image.new("RGB", (cols * px, rows * px), (255, 255, 255))
    for k, t in enumerate(tiles):
        out.paste(t, ((k % cols) * px, (k // cols) * px))
    out.save(path, quality=72, optimize=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", type=Path, required=True)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--classes", default="", help="default: the ten worst classes by final AP")
    ap.add_argument("--sheet", type=int, default=12, help="confusers per contact sheet")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cells = pd.read_csv(args.analysis / "cells.csv")
    cells = cells[cells["arm"].str.contains("SigLIP")]
    per = cells.groupby("class")[["text_ap", "final_ap", "ceiling_ap", "positives_found"]].mean()
    classes = [c.strip() for c in args.classes.split(",") if c.strip()] or list(per.sort_values("final_ap").index[:10])
    dom = _dominant(pc.COCO_ANCHOR_DIR)
    n_img = 123_287
    base = collections.Counter(dom.values())
    base["(none of COCO's 80)"] = n_img - len(dom)
    base = {k: v / n_img for k, v in base.items()}

    bad: dict[str, collections.Counter] = {c: collections.Counter() for c in classes}
    good: dict[str, int] = collections.Counter()
    for p in sorted((args.exp / "results" / "cells").glob("task_*__picks.csv")):
        df = pd.read_csv(p, usecols=["category", "picked_id", "picked_label", "embedder"])
        df = df[df["embedder"] == "siglip"]
        if df.empty:
            continue
        cls = df["category"].iloc[0].split("@")[0]
        if cls not in bad:
            continue
        bad[cls].update(int(i) for i in df.loc[df["picked_label"] == 0, "picked_id"])
        good[cls] += int((df["picked_label"] == 1).sum())

    rows = []
    for cls in classes:
        n_bad = sum(bad[cls].values())
        holds = collections.Counter()
        for iid, n in bad[cls].items():
            holds[dom.get(iid, "(none of COCO's 80)")] += n
        # Rank by LIFT over the corpus, not by share: `person` is the largest object
        # in a third of COCO, so it tops every class's Bad clicks by share alone.
        top = sorted(
            ((k, v / n_bad, (v / n_bad) / base[k]) for k, v in holds.items() if v / n_bad >= 0.02),
            key=lambda t: -t[2],
        )[:5]
        rows.append(
            {
                "class": cls,
                **per.loc[cls].round(3).to_dict(),
                "good_clicks": good[cls],
                "bad_clicks": n_bad,
                "bad_holds": "; ".join(f"{k} {share:.0%} (x{lift:.0f})" for k, share, lift in top),
            }
        )
        _sheet([i for i, _ in bad[cls].most_common(args.sheet)], args.out / f"confusers_{cls.replace(' ', '_')}.jpg")
        print(f"{cls}: {rows[-1]['bad_holds']}")
    pd.DataFrame(rows).to_csv(args.out / "why.csv", index=False)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
