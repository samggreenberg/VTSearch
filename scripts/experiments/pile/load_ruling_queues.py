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

#: The question is the RETRIEVAL one, not a definitional one about objects.
#: "Is this a tv?" cannot be answered for COCO's `tv`, because the class is 50%
#: television and 47% computer monitor and the honest answer depends on USE --
#: the same panel is a tv with a console on it and not a tv with a spreadsheet
#: on it, which is a property COCO's annotators never conditioned on. The
#: benchmark issues the query `a tv` (verbatim from `_COCO_TEXTS`) against
#: COCO's designation, and the two disagree on ~47% of the positives. Asking
#: what the query should return measures the thing the cell actually scores, and
#: it absorbs the use distinction for free.
QUESTION = {
    "tv": "coco_quarry tv - would you want this back searching for a tv?",
    "dining table": "coco_quarry dining table - would you want this back searching for a dining table?",
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
    ap.add_argument(
        "--name-template",
        default="",
        help="e.g. 'coco_quarry {cls} - is the red box around ONE object?'. When given, every "
        "subdirectory of --queues is registered under it and QUESTION is ignored.",
    )
    ap.add_argument("--wait", type=int, default=600)
    ap.add_argument(
        "--replace",
        action="store_true",
        help="delete an existing queue of the same name first; REFUSED if it holds any image vote",
    )
    ap.add_argument("--also-drop", action="append", default=[], help="an older queue name to remove too")
    args = ap.parse_args()

    have = datasets(args.api)
    det_rows = {d["name"]: d for d in api(args.api, "/api/detectors/registry").get("detectors", [])}
    dets = set(det_rows)
    print(f"{len(have)} datasets, {len(dets)} detectors already registered")

    # A queue is only ever removed when it is provably unanswered. The seeded
    # text query is not a vote; an image example is. Banking after a clear once
    # cost 1,725 answers, so the guard is on the votes, not on who made it.
    for name in list(args.also_drop) + ([q for q in QUESTION.values()] if args.replace else []):
        d = det_rows.get(name)
        if d is not None:
            votes = [e for e in (d.get("examples") or []) if e.get("type") != "text"]
            if votes:
                print(f"  {name}: REFUSING to replace, it holds {len(votes)} image vote(s)")
                return 1
            api(args.api, f"/api/detectors/registry/{d['id']}", method="DELETE")
            print(f"  {name}: detector removed (0 votes)")
        ds = have.get(name)
        if ds is not None:
            api(args.api, f"/api/datasets/registry/{ds.get('id')}", method="DELETE")
            print(f"  {name}: dataset removed")
    if args.also_drop or args.replace:
        have = datasets(args.api)
        dets = {d["name"] for d in api(args.api, "/api/detectors/registry").get("detectors", [])}

    rc = 0
    questions = QUESTION
    if args.name_template:
        # Discover, so a new task needs no edit here. The class comes from the
        # manifest rather than the directory name, because the directory is
        # slugged and the class is what the question has to say.
        questions = {}
        for d in sorted(p for p in args.queues.iterdir() if p.is_dir()):
            man = d / "manifest.json"
            if not man.exists():
                continue
            cls = json.loads(man.read_text())["class"]
            questions[cls] = args.name_template.format(cls=cls)
    for klass, name in questions.items():
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
                "text_query": TEXT.get(klass, f"a {klass}"),
                "examples": [{"type": "text", "value": TEXT.get(klass, f"a {klass}")}],
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
