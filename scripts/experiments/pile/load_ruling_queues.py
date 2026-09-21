#!/usr/bin/env python3
"""Register the class-ruling queues: one dataset plus one EMPTY detector each.

Two traps this avoids, both paid for already on this project:

* ``/api/detectors`` writes a detector file without registering it, so the UI
  never lists it. Detector writes go through ``/api/detectors/registry``.
* an import returns as soon as it is QUEUED. Reporting that as success is how a
  run gets called done with nothing ingested, so every import is polled to
  completion and then re-counted from the registry.

It only ever ADDS. Nothing here deletes or rewrites a dataset or a detector: the
dashboard is shared, and a detector is often the only copy of a human review.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

QUESTION = {
    "tv": "coco_quarry tv - is this a tv?",
    "dining table": "coco_quarry dining table - is this a dining table?",
}
TEXT = {"tv": "a tv", "dining table": "a dining table"}


def api(base: str, path: str, payload=None, method="GET", timeout=180):
    req = urllib.request.Request(  # noqa: S310 - our own app on the cluster
        base.rstrip("/") + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as fh:  # noqa: S310
            body = fh.read().decode()
    except urllib.error.HTTPError as exc:
        return {"_error": f"{exc.code}: {exc.read().decode()[:300]}"}
    return json.loads(body) if body.strip() else {}


def count_of(d: dict) -> int:
    fc = d.get("file_type_counts") or {}
    return sum(int(v) for v in fc.values()) if fc else int(d.get("num_items") or d.get("count") or 0)


def datasets(base: str) -> dict:
    got = api(base, "/api/datasets/registry")
    rows = got.get("datasets", got if isinstance(got, list) else [])
    return {d.get("name", ""): d for d in rows if isinstance(d, dict)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://rack5n04:11850")
    ap.add_argument("--queues", type=Path, default=Path("/expscratch/sgreenberg/ruling-queues"))
    ap.add_argument("--wait", type=int, default=600)
    args = ap.parse_args()

    have = datasets(args.api)
    dets = {d["name"] for d in api(args.api, "/api/detectors/registry").get("detectors", [])}
    print(f"{len(have)} datasets, {len(dets)} detectors already registered")

    rc = 0
    for klass, name in QUESTION.items():
        folder = args.queues / klass.replace(" ", "_") / "images"
        n = len(list(folder.glob("*.jpg")))
        if n == 0:
            print(f"  {name}: SKIP, no images at {folder}")
            rc = 1
            continue
        if name in have:
            print(f"  {name}: dataset exists ({count_of(have[name])} items)")
        else:
            resp = api(
                args.api,
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
            if "_error" in resp:
                print(f"  {name}: IMPORT FAILED {resp['_error'][:100]}")
                rc = 1
                continue
            t0, landed = time.time(), None
            while time.time() - t0 < args.wait:
                time.sleep(3)
                d = datasets(args.api).get(name)
                if d and count_of(d) >= n * 0.95:
                    landed = d
                    break
            if not landed:
                print(f"  {name}: TIMED OUT after {args.wait}s (may still be ingesting)")
                rc = 1
                continue
            print(f"  {name}: dataset OK {count_of(landed)}/{n}")

        if name in dets:
            print(f"  {name}: detector exists, left alone")
            continue
        resp = api(
            args.api,
            "/api/detectors/registry",
            {
                "name": name,
                "media_type": "image",
                "embedder_type": "semantic",
                "text_query": TEXT[klass],
                "examples": [{"type": "text", "value": TEXT[klass]}],
            },
            method="POST",
        )
        if "_error" in resp or not resp.get("ok", True):
            print(f"  {name}: DETECTOR FAILED {str(resp)[:140]}")
            rc = 1
        else:
            print(f"  {name}: empty detector created")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
