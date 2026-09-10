#!/usr/bin/env python3
"""Re-ask the CLASS question of the old positives after the planter ruling (#3778).

The first version of this slate asked "is the boxed vessel a flower pot?" -- a
question about a category that does not exist in *C*, so no answer to it records
anything the dataset can act on. The reviewer's correction is the right frame:
the ruling changed what a Bowl and a Vase ARE, so the question to re-ask is the
class question itself.

**Image-level, and no box.** The old row asserts `present: true` for a class,
and what has to be re-established is whether that still holds -- "does this photo
contain a Bowl under the new rule?". A drawn box would push the answer toward one
vessel, when an image may hold a planter (out of *C*) AND a real bowl; the photo
is then still a Bowl positive and only its box is wrong, which is a separate and
smaller question.

One dataset per class, each named for its rule, because that name is the only
wording a reviewer reads while voting (#3612).
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

OUT = Path("/expscratch/sgreenberg/vlm-3720/class_recheck")
ROOT = Path("/exp/scale26/datasets/external/vtsearch-demos/visual_genome")
OLD = "planter recheck: is the boxed vessel a flower pot?"

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
    import pile_config as pc  # noqa: PLC0415

    corr = json.loads(Path("/expscratch/sgreenberg/vts-cache/corrections.json").read_text())
    by_cls: dict[str, list[dict]] = {"bowl": [], "vase": []}
    for r in corr:
        if r.get("class") in by_cls and r.get("present"):
            by_cls[r["class"]].append(r)

    if OUT.exists():
        shutil.rmtree(OUT)
    made = {}
    for cls, rows in by_cls.items():
        d = OUT / cls / "images"
        d.mkdir(parents=True, exist_ok=True)
        man = []
        for r in rows:
            iid = int(r["image_id"])
            src = next(
                (ROOT / s / f"{iid}.jpg" for s in ("VG_100K", "VG_100K_2") if (ROOT / s / f"{iid}.jpg").exists()), None
            )
            if src is None:
                continue
            dest = d / f"{iid}.jpg"
            if not dest.exists():  # one image can be positive for both
                dest.symlink_to(src)
            man.append({"image_id": iid, "old_box": (r.get("boxes") or [None])[0]})
        (OUT / cls / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
        made[cls] = len(man)
        print(f"  {cls}: {len(man)} images")

    # retire the mis-framed slate
    ds = {d["name"]: d for d in api("/api/datasets/registry")[1]["datasets"]}
    dt = {d["name"]: d for d in api("/api/detectors/registry")[1]["detectors"]}
    for reg, table in (("datasets", ds), ("detectors", dt)):
        if OLD in table:
            n = (table[OLD].get("num_training") or 0) if reg == "detectors" else 0
            if n:
                print(f"  REFUSING to delete {reg} {OLD!r}: holds {n} verdicts")
                continue
            api(f"/api/{reg}/registry/{table[OLD]['id']}", method="DELETE")
            print(f"  removed the mis-framed {reg[:-1]}")

    for cls, n in made.items():
        rule = getattr(pc.SCALE_CLASS_RULES.get(cls), "name", "") or cls
        name = f"{rule} -- recheck: is there one in this image?"
        folder = OUT / cls / "images"
        if name not in ds:
            api(
                "/api/dataset/import/server_folder",
                {
                    "path": str(folder),
                    "media_type": "image",
                    "recursive": "false",
                    "dig_archives": "false",
                    "dataset_name": name,
                },
                method="POST",
            )
            for _ in range(160):
                time.sleep(3)
                d = {x["name"]: x for x in api("/api/datasets/registry")[1]["datasets"]}.get(name)
                if d and sum(int(v) for v in (d.get("file_type_counts") or {}).values()) >= n:
                    print(f"  imported {name[:58]!r}: {n}")
                    break
            else:
                print(f"  import TIMED OUT for {cls}")
        if name not in dt:
            code, _ = api(
                "/api/detectors/registry",
                {
                    "name": name,
                    "media_type": "image",
                    "embedder_type": "semantic",
                    "text_query": cls,
                    "examples": [{"type": "text", "value": cls}],
                },
                method="POST",
            )
            print(f"  detector created ({code})")
    print("\nGood = the photo DOES contain one under the new rule; Bad = it does not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
