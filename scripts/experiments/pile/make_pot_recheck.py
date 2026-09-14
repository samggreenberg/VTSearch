#!/usr/bin/env python3
"""A 17-image slate of the old positives that may be planters (#3778).

The planter ruling put a vessel made to hold a growing plant outside C. `bowl`
never admitted planters so nothing is owed there by the rule, and `vase` did, so
its old rows are the ones the ruling actually moves. Rather than re-review all 80
pre-slate positives, an open-vocabulary pass flagged the 17 whose boxed vessel
looks like a flower pot or potted plant -- and a detector's guess is a proposal,
not a verdict, so those 17 go in front of the reviewer with the box drawn.

Both classes share one slate because the question is the same for both: is the
boxed vessel a planter? Its own class is recorded in the manifest so a Bad here
retires the right row.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
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

    flagged = json.loads(Path("/expscratch/sgreenberg/vlm-3720/old_pots.json").read_text())["flagged"]
    corr = json.loads(Path("/expscratch/sgreenberg/vts-cache/corrections.json").read_text())
    box_of = {}
    for r in corr:
        if r.get("class") in ("bowl", "vase") and r.get("present") and r.get("boxes"):
            box_of[(r["class"], int(r["image_id"]))] = r["boxes"][0]

    imgs = OUT / "images"
    if imgs.exists():
        shutil.rmtree(imgs)
    imgs.mkdir(parents=True, exist_ok=True)
    manifest = []
    for f in flagged:
        cls, iid = f["class"], int(f["image_id"])
        b = box_of.get((cls, iid))
        src = next(
            (ROOT / s / f"{iid}.jpg" for s in ("VG_100K", "VG_100K_2") if (ROOT / s / f"{iid}.jpg").exists()), None
        )
        if not b or src is None:
            print(f"  skip {cls} {iid}")
            continue
        draw_with_inset(src, tuple(b), imgs / f"{iid}.jpg")
        manifest.append({"image_id": iid, "class": cls, "box": b, "owl_says": f["top"]})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{len(manifest)} boxed images -> {imgs}")

    have = {d["name"] for d in api("/api/datasets/registry")[1]["datasets"]}
    if NAME not in have:
        api(
            "/api/dataset/import/server_folder",
            {
                "path": str(imgs),
                "media_type": "image",
                "recursive": "false",
                "dig_archives": "false",
                "dataset_name": NAME,
            },
            method="POST",
        )
        import time

        for _ in range(120):
            time.sleep(3)
            d = {x["name"]: x for x in api("/api/datasets/registry")[1]["datasets"]}.get(NAME)
            if d and sum(int(v) for v in (d.get("file_type_counts") or {}).values()) >= len(manifest):
                print(f"dataset imported: {NAME!r}")
                break
        else:
            print("dataset import TIMED OUT")
    dets = {d["name"] for d in api("/api/detectors/registry")[1]["detectors"]}
    if NAME not in dets:
        code, _ = api(
            "/api/detectors/registry",
            {
                "name": NAME,
                "media_type": "image",
                "embedder_type": "semantic",
                "text_query": "flower pot",
                "examples": [{"type": "text", "value": "flower pot"}],
            },
            method="POST",
        )
        print(f"detector created ({code})")
    print("\nGood = it IS a planter (so the old positive should be retired); Bad = it is not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
