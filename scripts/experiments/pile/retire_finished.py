#!/usr/bin/env python3
"""Remove finished dataset/detector pairs from the dashboard (#3720).

A pair is finished when every image in its dataset carries a verdict. Deleting is
only safe because the verdicts are in the repository -- so the committed copy is
re-compared against the LIVE one immediately before each delete, not once at the
start. The reviewer keeps working while this runs, and a verdict cast in between
would otherwise be destroyed with nothing to show it existed.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request

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
REPO = pathlib.Path("/exp/sgreenberg/projects/vts-annq-3720/scripts/experiments/pile/human_record")


_HDRS = {"Content-Type": "application/json"}


def api(p, method="GET"):
    req = urllib.request.Request(BASE + p, method=method, headers=_HDRS)  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=180) as fh:  # noqa: S310 - our own app
            return fh.getcode(), json.loads(fh.read().decode() or "{}")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()[:150]


def main() -> int:
    ds = {d["name"]: d for d in api("/api/datasets/registry")[1]["datasets"]}
    dets = {d["name"]: d for d in api("/api/detectors/registry")[1]["detectors"]}
    removed = kept = 0
    for name, d in sorted(ds.items()):
        size = sum(int(v) for v in (d.get("file_type_counts") or {}).values())
        t = dets.get(name)
        done = (t.get("num_training") or 0) if t else 0
        if not size or done < size:
            kept += 1
            if t:
                print(f"  keep    {name[:56]:<58} {done}/{size}")
            continue

        # re-verify the bank against live, right now
        anyq = ("any in image" in name) or ("below-cut" in name)
        kind = "belowcut" if anyq else "slate"
        cls = t.get("text_query") or name.split(" [")[0].split(" (")[0].split()[0]
        f = REPO / f"LABELSETS__{kind}__{cls.replace(' ', '_')}.json"
        live = api(f"/api/detectors/{urllib.parse.quote(name)}/labels-detail")[1]
        lg = {r["filename"] for r in live.get("good", [])}
        lb = {r["filename"] for r in live.get("bad", [])}
        if not f.exists():
            print(f"  KEEP    {name[:56]:<58} not banked ({f.name})")
            kept += 1
            continue
        doc = json.loads(f.read_text())
        if {r["filename"] for r in doc["good"]} != lg or {r["filename"] for r in doc["bad"]} != lb:
            print(f"  KEEP    {name[:56]:<58} live moved since banking")
            kept += 1
            continue

        api(f"/api/detectors/registry/{t['id']}", method="DELETE")
        api(f"/api/datasets/registry/{d['id']}", method="DELETE")
        print(f"  removed {name[:56]:<58} {len(lg)}g/{len(lb)}b banked")
        removed += 1

    after_ds = api("/api/datasets/registry")[1]["datasets"]
    after_dt = api("/api/detectors/registry")[1]["detectors"]
    print(f"\nremoved {removed} finished pairs, kept {kept}")
    print(f"{len(after_ds)} datasets, {len(after_dt)} detectors remain")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
