#!/usr/bin/env python3
"""Give every dataset and its detector one shared, self-explanatory name (#3720).

The dashboard grew two kinds of pair with jargon names -- `[slate]` and
`[below-cut: any in image]` -- which say how the images were selected rather than
what the reviewer is being asked. The reviewer reads the name while voting and
nothing else (#3612), so it has to carry the QUESTION:

    "<rule> (confirm the box)"        -- a detection is drawn; is that box one?
    "<rule> (any in image, no box)"   -- nothing is drawn; is there one anywhere?

Datasets can be renamed in place. Detectors cannot -- the registry offers DELETE
and no rename -- so an empty detector is deleted and recreated under the new
name, and a detector **carrying verdicts is never touched**; its dataset is
renamed to match it instead. Losing a person's judgement to a cosmetic rename
would be an absurd trade.
"""

from __future__ import annotations

import json
import shutil
import subprocess
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
    import sys

    sys.path.insert(0, "scripts/experiments/pile")
    import pile_config as pc  # noqa: PLC0415

    print(f"app node: {node}\n")
    ds = {d["name"]: d for d in api("/api/datasets/registry")[1]["datasets"]}
    dets = {d["name"]: d for d in api("/api/detectors/registry")[1]["detectors"]}

    plan = []
    for cls in pc.SCALE_CLASSES:
        rule = getattr(pc.SCALE_CLASS_RULES.get(cls), "name", "") or cls
        plan.append((f"vgscale {cls} candidates", f"{rule} [slate]", f"{rule} (confirm the box)"))
        if f"vgscale {cls} below-cut" in ds:
            plan.append(
                (f"vgscale {cls} below-cut", f"{rule} [below-cut: any in image]", f"{rule} (any in image, no box)")
            )

    renamed = recreated = frozen = 0
    for old_ds, old_det, new in plan:
        d = ds.get(old_ds)
        t = dets.get(old_det)
        if not d:
            continue
        held = (t.get("num_training") or 0) if t else 0
        target = old_det if held else new  # a detector with work keeps its name
        if d["name"] != target:
            code, _ = api(f"/api/datasets/registry/{d['id']}/rename", {"name": target}, method="PUT")
            renamed += code == 200
        if t and not held:
            api(f"/api/detectors/registry/{t['id']}", method="DELETE")
            code, _ = api(
                "/api/detectors/registry",
                {
                    "name": new,
                    "media_type": "image",
                    "embedder_type": "semantic",
                    "text_query": pc.SCALE_CLASS_RULES and old_det.split()[0],
                    "examples": [{"type": "text", "value": old_det.split()[0]}],
                },
                method="POST",
            )
            recreated += code == 201
            print(f"  {new[:62]:<64} renamed pair")
        elif held:
            frozen += 1
            print(f"  {target[:62]:<64} {held} verdicts - detector left alone")

    print(
        f"\n{renamed} datasets renamed, {recreated} detectors recreated, "
        f"{frozen} pairs left alone because the detector holds verdicts"
    )
    after_ds = api("/api/datasets/registry")[1]["datasets"]
    after_dt = {d["name"] for d in api("/api/detectors/registry")[1]["detectors"]}
    unmatched = [d["name"] for d in after_ds if d["name"] not in after_dt]
    print(
        f"{len(after_ds)} datasets, {len(after_dt)} detectors, "
        f"{len(unmatched)} datasets without an identically named detector"
    )
    for u in unmatched:
        print(f"   UNMATCHED: {u}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
