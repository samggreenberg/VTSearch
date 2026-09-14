#!/usr/bin/env python3
"""Repair the two faults in the rename pass (#3720).

* `dog` gained a written rule on dev while this ran (#3771), so the name the
  rename computed no longer matched the detector it was looking for, and the
  dog datasets were renamed against detectors that kept their old names.
* the recreate step seeded `text_query` from the first WORD of the old detector
  name, which silently truncates every multi-word class: `cell phone` became
  "cell", `stop sign` "stop", `fire hydrant` "fire". That is the string the
  ranking is seeded from, so it quietly demotes the class it is meant to find.

Only empty detectors are recreated; any detector holding verdicts keeps its name
and its dataset is renamed to match it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

_SQ = shutil.which("squeue") or "/usr/bin/squeue"
node = (
    subprocess.run(  # noqa: S603 - fixed argv, no shell
        [_SQ, "-u", "sgreenberg", "-h", "-n", "vtsearch", "-o", "%N"], capture_output=True, text=True, check=False
    )
    .stdout.strip()
    .split()[0]
)
BASE = f"http://{node}:11850"


def api(p, payload=None, method="GET"):
    req = urllib.request.Request(  # noqa: S310 - our own app
        BASE + p,
        data=json.dumps(payload).encode() if payload is not None else None,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as fh:  # noqa: S310
            return fh.getcode(), json.loads(fh.read().decode() or "{}")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()[:200]


def main() -> int:
    sys.path.insert(0, "scripts/experiments/pile")
    import pile_config as pc  # noqa: PLC0415

    ds = {d["name"]: d for d in api("/api/datasets/registry")[1]["datasets"]}
    dets = {d["name"]: d for d in api("/api/detectors/registry")[1]["detectors"]}

    # every dataset name we intend, and the class it belongs to
    want: dict[str, str] = {}
    for cls in pc.SCALE_CLASSES:
        rule = getattr(pc.SCALE_CLASS_RULES.get(cls), "name", "") or cls
        want[f"{rule} (confirm the box)"] = cls
        want[f"{rule} (any in image, no box)"] = cls

    fixed = 0
    # 1. any detector that is empty and whose text_query is not its class
    for name, d in sorted(dets.items()):
        cls = want.get(name)
        if cls is None or (d.get("num_training") or 0):
            continue
        if d.get("text_query") == cls:
            continue
        api(f"/api/detectors/registry/{d['id']}", method="DELETE")
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
        print(f"  text_query {d.get('text_query')!r} -> {cls!r}  {name[:48]}")
        fixed += code == 201

    # 2. datasets whose detector does not exist under the same name
    dets = {d["name"]: d for d in api("/api/detectors/registry")[1]["detectors"]}
    made = 0
    for dsname, d in sorted(ds.items()):
        cls = want.get(dsname)
        if cls is None or dsname in dets:
            continue
        code, _ = api(
            "/api/detectors/registry",
            {
                "name": dsname,
                "media_type": "image",
                "embedder_type": "semantic",
                "text_query": cls,
                "examples": [{"type": "text", "value": cls}],
            },
            method="POST",
        )
        print(f"  created missing detector for {dsname[:52]}")
        made += code == 201

    # 3. an old jargon-named detector holding verdicts keeps its name; its
    #    dataset must carry that name too, or the pair no longer matches
    dets = {d["name"]: d for d in api("/api/detectors/registry")[1]["detectors"]}
    ds = {d["name"]: d for d in api("/api/datasets/registry")[1]["datasets"]}
    for name, t in sorted(dets.items()):
        if not (t.get("num_training") or 0) or name in ds:
            continue
        print(f"  NOTE: detector {name[:52]!r} holds {t['num_training']} verdicts but has no dataset of that name")

    ds = {d["name"] for d in api("/api/datasets/registry")[1]["datasets"]}
    dt = {d["name"] for d in api("/api/detectors/registry")[1]["detectors"]}
    print(f"\n{fixed} text_query repairs, {made} detectors created")
    print(f"{len(ds)} datasets, {len(dt)} detectors")
    print(f"datasets with no matching detector: {sorted(ds - dt)}")
    print(f"detectors with no matching dataset: {sorted(dt - ds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
