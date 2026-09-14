#!/usr/bin/env python3
"""Every pre-slate `bowl` and `vase` positive, not just the ones a detector could see.

The first cut of this slate carried only the 17 whose boxed vessel an
open-vocabulary detector called a flower pot. The reviewer then pointed out the
hole in that: an EMPTY, bowl-shaped planter looks like a bowl, so the screen is
blindest exactly where the error is. A visual model cannot separate "a bowl" from
"a planter shaped like a bowl" -- that is a fact about the objects, not a
threshold to tune.

80 images is cheaper than the argument, so all of them go in front of the
reviewer and the screen is dropped. Its guess is kept in the manifest only so the
two can be compared afterwards.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

OUT = Path("/expscratch/sgreenberg/vlm-3720/pot_recheck")
ROOT = Path("/exp/scale26/datasets/external/vtsearch-demos/visual_genome")
NAME = "planter recheck: is the boxed vessel a flower pot?"

_SQUEUE = shutil.which("squeue") or "/usr/bin/squeue"
_ARGS = [_SQUEUE, "-u", "sgreenberg", "-h", "-n", "vtsearch", "-o", "%N"]
node = (
    subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
        _ARGS, capture_output=True, text=True, check=False
    )
    .stdout.strip()
    .split()[0]
)
BASE = f"http://{node}:11850"
_H = {"Content-Type": "application/json"}


def _req(p, payload, method):
    url = BASE + p
    data = json.dumps(payload).encode() if payload is not None else None
    return urllib.request.Request(url, data=data, method=method, headers=_H)  # noqa: S310


def api(p, payload=None, method="GET"):
    try:
        with urllib.request.urlopen(_req(p, payload, method), timeout=180) as fh:  # noqa: S310
            return fh.getcode(), json.loads(fh.read().decode() or "{}")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()[:200]


def main() -> int:
    sys.path.insert(0, "scripts/experiments/pile")
    from make_positive_slate import draw_with_inset  # noqa: PLC0415

    owl = {
        int(f["image_id"]): f["top"]
        for f in json.loads(Path("/expscratch/sgreenberg/vlm-3720/old_pots.json").read_text())["flagged"]
    }
    corr = json.loads(Path("/expscratch/sgreenberg/vts-cache/corrections.json").read_text())

    imgs = OUT / "images"
    if imgs.exists():
        shutil.rmtree(imgs)
    imgs.mkdir(parents=True, exist_ok=True)
    manifest, skipped = [], 0
    for r in corr:
        cls = r.get("class")
        if cls not in ("bowl", "vase") or not r.get("present") or not r.get("boxes"):
            continue
        iid = int(r["image_id"])
        src = next(
            (ROOT / s / f"{iid}.jpg" for s in ("VG_100K", "VG_100K_2") if (ROOT / s / f"{iid}.jpg").exists()), None
        )
        if src is None:
            skipped += 1
            continue
        # Filenames carry the CLASS as well as the id: this slate mixes two
        # classes, and one image (2398885) is a positive for both with different
        # boxes, so id-only names silently dropped one of them.
        fname = f"{cls}_{iid}.jpg"
        try:
            draw_with_inset(src, tuple(r["boxes"][0]), imgs / fname)
        except Exception as exc:  # noqa: BLE001
            print(f"  {iid}: {exc}")
            skipped += 1
            continue
        manifest.append(
            {
                "filename": fname,
                "image_id": iid,
                "class": cls,
                "box": r["boxes"][0],
                "owl_says": owl.get(iid, "not flagged"),
            }
        )
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    n_bowl = sum(1 for m in manifest if m["class"] == "bowl")
    print(f"{len(manifest)} boxed images ({n_bowl} bowl, {len(manifest) - n_bowl} vase), {skipped} skipped")
    print(f"  of these, {sum(1 for m in manifest if m['owl_says'] != 'not flagged')} were what the detector flagged")

    # the dataset already exists with 17; replace it so the pair stays 1:1
    ds = {d["name"]: d for d in api("/api/datasets/registry")[1]["datasets"]}
    if NAME in ds:
        api(f"/api/datasets/registry/{ds[NAME]['id']}", method="DELETE")
        print("  removed the 17-image version")
    api(
        "/api/dataset/import/server_folder",
        {"path": str(imgs), "media_type": "image", "recursive": "false", "dig_archives": "false", "dataset_name": NAME},
        method="POST",
    )
    for _ in range(160):
        time.sleep(3)
        d = {x["name"]: x for x in api("/api/datasets/registry")[1]["datasets"]}.get(NAME)
        if d and sum(int(v) for v in (d.get("file_type_counts") or {}).values()) >= len(manifest):
            print(f"  imported {NAME!r}: {len(manifest)} items")
            break
    else:
        print("  import TIMED OUT")
    print("\nGood = it IS a planter (retire the old positive); Bad = it is not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
