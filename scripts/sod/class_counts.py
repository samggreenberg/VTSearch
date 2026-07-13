#!/usr/bin/env python3
"""Per-class image/instance counts for the SOD datasets (coco / lvis / vg).

Streams each dataset's derived extract once and tallies, per class, the number of
distinct images (the positive-image count that caps K and sets prevalence in the
sweep) and the number of instance rows. Classes are keyed exactly as
``SodDataset`` matches them: normalized ``name`` for coco/lvis; ``synset`` first
token (else normalized ``name``) for vg. Writes a full CSV per dataset and prints
a summary + the top classes.
"""

from __future__ import annotations

import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

from datasets import _CONFIG, _norm  # noqa: PLC0415


def _class_key(row: dict, kind: str) -> str:
    if kind == "coco_lvis":
        return _norm(row.get("name", ""))
    syn = str(row.get("synset", "")).strip().lower()
    return syn.split(".")[0] if syn else _norm(row.get("name", ""))


def count(name: str, out_dir: Path, top: int = 30) -> None:
    cfg = _CONFIG[name]
    extract = cfg["extract"]
    if not extract.exists():
        print(f"[{name}] missing extract {extract} — skipping", flush=True)
        return
    images: dict[str, set[int]] = defaultdict(set)
    instances: dict[str, int] = defaultdict(int)
    n_rows = 0
    with gzip.open(extract, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            key = _class_key(row, cfg["kind"])
            if not key:
                continue
            images[key].add(int(row["image_id"]))
            instances[key] += 1
            n_rows += 1

    rows = sorted(((k, len(v), instances[k]) for k, v in images.items()), key=lambda r: (-r[1], r[0]))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}_class_counts.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "n_images", "n_instances"])
        w.writerows(rows)

    total_imgs = len({i for s in images.values() for i in s})
    print(f"\n===== {name}  ({cfg['kind']}, negatives_exhaustive={cfg['negatives_exhaustive']}) =====", flush=True)
    print(f"classes={len(rows)}  distinct_images≈{total_imgs}  instance_rows={n_rows}  -> {csv_path}", flush=True)
    print(f"top {min(top, len(rows))} by image count:", flush=True)
    for k, ni, ninst in rows[:top]:
        print(f"  {ni:6d} imgs  {ninst:7d} inst  {k}", flush=True)


def main() -> int:
    # Repo-root-relative so output lands in the same place regardless of cwd.
    out_dir = Path(__file__).resolve().parents[2] / "docs/experiments/dataset-class-counts"
    names = sys.argv[1:] or ["coco", "lvis", "vg"]
    for name in names:
        try:
            count(name, out_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[{name}] error: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
