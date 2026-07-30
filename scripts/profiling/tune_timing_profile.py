#!/usr/bin/env python3
"""Build a per-environment timing profile for every long-running VTSearch task.

A progress bar's ETA is only as good as its idea of what the *next* phase costs.
Shipped defaults are one cluster's measurements; your hardware, disks, and
network are not that cluster. This script measures yours and writes the JSON that
``VTSEARCH_TIMING_PROFILE`` points at, after which every instance in the
environment predicts its own timings (see :mod:`vtscore.timing`).

There are two ways to gather the timings, and they produce the same file.

**1. Observe real usage (recommended).** Start the server with the recorder armed
and let people use it::

    VTSEARCH_TIMING_RECORD=/var/lib/vtsearch/timings.jsonl python app.py
    # ... a day or a week of normal work ...
    python scripts/profiling/tune_timing_profile.py --fit-only \\
        --out /etc/vtsearch/timing-profile.json /var/lib/vtsearch/timings.jsonl

This has no side effects, needs no exemplar data, and measures the datasets your
users actually load at the sizes they actually are — a mix no synthetic sweep
reproduces. It is the right default for a production system.

**2. Drive the workloads.** When you want numbers *now* — commissioning a new
node, or after a hardware change — ``--drive`` exercises the task families
against datasets and detectors you name::

    python scripts/profiling/tune_timing_profile.py --drive \\
        --out timing-profile.json --datasets ds-a,ds-b,ds-c --reps 3

By default ``--drive`` runs only the **read-only** families: opening a dataset,
text search, and Find. The rest mutate state — loading a detector seeds example
votes, train-and-score *overwrites the active dataset's labels*, promote creates
a dataset, and an import writes one — so they need ``--allow-mutating`` and
should be pointed at a scratch ``--data-dir``, never at live user data.

Both modes end the same way: the fit runs, the profile is written, and a coverage
report says exactly which task families got measured and which fell back to the
built-in defaults. Deploy it with::

    VTSEARCH_TIMING_PROFILE=/etc/vtsearch/timing-profile.json

Re-run it whenever the hardware changes. The profile only ever affects how
progress bars pace and predict; a stale or missing one costs accuracy, never
correctness.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ``python scripts/profiling/x.py`` only puts the script's own directory on
# sys.path, so make the repo root (where app.py lives) importable from any cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

#: Families whose driver only reads: safe to run against live production data.
_READ_ONLY_TASKS = ("dataset_open", "text_sort", "find")

#: Families whose driver writes something a user would notice. Gated behind
#: ``--allow-mutating`` with the specific hazard named, because "the tuning
#: script wiped my votes" is not a tradeoff anyone would have accepted if asked.
_MUTATING_TASKS = {
    "detector_load": "loads detectors and seeds their example votes into the active dataset",
    "train_and_score": "OVERWRITES every label in the active dataset (replace_all)",
    "dataset_promote": "creates new datasets in the registry",
    "dataset_load": "imports demo datasets, downloading and writing them",
}

#: How long to wait for one background task before giving up on the cell.
_TASK_TIMEOUT_S = 3600


def _log(*parts: object) -> None:
    print("[tune]", *parts, file=sys.stderr, flush=True)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("jsonl", nargs="*", help="recorded JSONL to fit (with --fit-only)")
    ap.add_argument("--out", required=True, help="profile JSON to write")
    ap.add_argument(
        "--record",
        default="",
        help="JSONL sink the drivers record into (default: <out>.jsonl)",
    )
    ap.add_argument(
        "--fit-only",
        action="store_true",
        help="fit the given JSONL without driving anything (the observe-real-usage flow)",
    )
    ap.add_argument("--drive", action="store_true", help="exercise the task families in-process")
    ap.add_argument(
        "--tasks",
        default="",
        help=f"comma list of task families to drive (default: {','.join(_READ_ONLY_TASKS)})",
    )
    ap.add_argument(
        "--allow-mutating",
        action="store_true",
        help="permit drivers that change state; point --data-dir at scratch data first",
    )
    ap.add_argument("--datasets", default="", help="comma list of registry dataset ids or names")
    ap.add_argument("--detectors", default="", help="comma list of registry detector ids or names")
    ap.add_argument("--demo", default="", help="comma list of demo dataset ids for dataset_load")
    ap.add_argument(
        "--queries",
        default="a person,an outdoor scene,music",
        help="comma list of text-search queries for text_sort",
    )
    ap.add_argument("--reps", type=int, default=2, help="repetitions per cell (default 2)")
    ap.add_argument(
        "--min-samples",
        type=int,
        default=2,
        help="drop fitted cells with fewer runs than this (default 2)",
    )
    ap.add_argument("--data-dir", default="", help="VTSEARCH_DATA_DIR for the driven run")
    ap.add_argument("--notes", default="", help="free-text note stored in the profile")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    return ap.parse_args()


def _split(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _resolve_tasks(args: argparse.Namespace) -> list[str]:
    """Resolve which families to drive, refusing mutating ones unless allowed."""
    from vtscore.timing import known_tasks  # noqa: PLC0415

    wanted = _split(args.tasks) or list(_READ_ONLY_TASKS)
    valid = set(known_tasks())
    resolved: list[str] = []
    for task in wanted:
        if task not in valid:
            _log(f"SKIPPED {task}: not a known task family")
            continue
        hazard = _MUTATING_TASKS.get(task)
        if hazard and not args.allow_mutating:
            _log(f"SKIPPED {task}: {hazard}. Pass --allow-mutating (with a scratch --data-dir) to include it.")
            continue
        resolved.append(task)
    return resolved


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def _wait_for_task(registry, task_id: str, label: str) -> bool:
    """Block until a background task finishes. Returns ``False`` on timeout."""
    if not task_id:
        return False
    deadline = time.monotonic() + _TASK_TIMEOUT_S
    while time.monotonic() < deadline:
        if registry.is_finished(task_id):
            return True
        time.sleep(0.25)
    _log(f"TIMEOUT waiting for {label}")
    return False


def _resolve_datasets(names: list[str]) -> list[dict]:
    """Registry entries for the requested ids/names (all accessible if empty)."""
    from vtscore.datasets.registry import list_datasets  # noqa: PLC0415

    entries = list_datasets()
    if not names:
        return entries
    wanted = set(names)
    picked = [e for e in entries if e.get("id") in wanted or e.get("name") in wanted]
    for name in wanted - {e.get("id") for e in picked} - {e.get("name") for e in picked}:
        _log(f"SKIPPED dataset {name!r}: not in the registry")
    return picked


def _resolve_detectors(names: list[str]) -> list[dict]:
    """Registry entries for the requested detector ids/names (all if empty)."""
    from vtscore.detectors.registry import list_detectors  # noqa: PLC0415

    entries = list_detectors()
    if not names:
        return entries
    wanted = set(names)
    picked = [e for e in entries if e.get("id") in wanted or e.get("name") in wanted]
    for name in wanted - {e.get("id") for e in picked} - {e.get("name") for e in picked}:
        _log(f"SKIPPED detector {name!r}: not in the registry")
    return picked


def _open_dataset(client, dataset_id: str, *, reopen: bool) -> bool:
    """Load a dataset, optionally unloading first so the load is really measured."""
    from vtscore.concurrency.progress import loading_tasks  # noqa: PLC0415

    if reopen:
        client.post(f"/api/datasets/registry/{dataset_id}/unload")
    resp = client.post(f"/api/datasets/registry/{dataset_id}/load")
    if resp.status_code != 200:
        _log(f"SKIPPED dataset {dataset_id}: load returned {resp.status_code}")
        return False
    task_id = (resp.get_json() or {}).get("task_id") or ""
    if not task_id:
        return True  # already loaded; nothing was measured but the state is right
    return _wait_for_task(loading_tasks, task_id, f"dataset {dataset_id}")


def drive_dataset_open(client, datasets: list[dict], reps: int) -> None:
    """Measure opening each dataset from its pkl (read + dedup, coverage atlas)."""
    for entry in datasets:
        for rep in range(reps):
            _log(f"dataset_open {entry.get('name')} rep {rep + 1}/{reps}")
            _open_dataset(client, entry["id"], reopen=True)


def drive_text_sort(client, datasets: list[dict], queries: list[str], reps: int) -> None:
    """Measure a text search (encoder load, query embed, score) per dataset."""
    for entry in datasets:
        if not _open_dataset(client, entry["id"], reopen=False):
            continue
        headers = {"X-Dataset-Id": entry["id"]}
        for rep in range(reps):
            for query in queries:
                _log(f"text_sort {entry.get('name')} {query!r} rep {rep + 1}/{reps}")
                resp = client.post("/api/sort", json={"text": query}, headers=headers)
                if resp.status_code != 200:
                    _log(f"SKIPPED text_sort on {entry.get('name')}: {resp.status_code}")
                    break


def drive_find(client, datasets: list[dict], detectors: list[dict], reps: int) -> None:
    """Measure a Find pass over every (dataset, detector) media-type match."""
    if not detectors:
        _log("SKIPPED find: no detectors in the registry")
        return
    for entry in datasets:
        matching = [d for d in detectors if d.get("media_type") == entry.get("media_type")]
        if not matching:
            _log(f"SKIPPED find on {entry.get('name')}: no detector of media type {entry.get('media_type')!r}")
            continue
        body = {"dataset_ids": [entry["id"]], "detector_ids": [d["id"] for d in matching]}
        for rep in range(reps):
            _log(f"find {entry.get('name')} × {len(matching)} detectors rep {rep + 1}/{reps}")
            resp = client.post("/api/find", json=body)
            if resp.status_code != 200:
                _log(f"SKIPPED find on {entry.get('name')}: {resp.status_code}")
                break


def drive_detector_load(client, datasets: list[dict], detectors: list[dict], reps: int) -> None:
    """Measure loading each detector (restore labels, seed examples, train)."""
    from vtscore.concurrency.progress import detector_loading_tasks  # noqa: PLC0415

    for detector in detectors:
        host = next((d for d in datasets if d.get("media_type") == detector.get("media_type")), None)
        if host is None or not _open_dataset(client, host["id"], reopen=False):
            _log(f"SKIPPED detector_load {detector.get('name')}: no loadable dataset of its media type")
            continue
        headers = {"X-Dataset-Id": host["id"]}
        for rep in range(reps):
            _log(f"detector_load {detector.get('name')} rep {rep + 1}/{reps}")
            client.post(f"/api/detectors/registry/{detector['id']}/unload", headers=headers)
            resp = client.post(
                "/api/detectors/registry/load",
                json={"detector_id": detector["id"]},
                headers=headers,
            )
            if resp.status_code != 200:
                _log(f"SKIPPED detector_load {detector.get('name')}: {resp.status_code}")
                break
            task_id = (resp.get_json() or {}).get("task_id") or ""
            _wait_for_task(detector_loading_tasks, task_id, f"detector {detector['id']}")


def drive_train_and_score(client, datasets: list[dict], detectors: list[dict], reps: int) -> None:
    """Measure a train-and-score pass. Destroys the active dataset's labels."""
    for entry in datasets:
        matching = [d for d in detectors if d.get("media_type") == entry.get("media_type")]
        if not matching or not _open_dataset(client, entry["id"], reopen=False):
            continue
        headers = {"X-Dataset-Id": entry["id"], "X-Detector-Id": matching[0]["id"]}
        for rep in range(reps):
            _log(f"train_and_score {entry.get('name')} / {matching[0].get('name')} rep {rep + 1}/{reps}")
            resp = client.post("/api/find-label", json={"detector_id": matching[0]["id"]}, headers=headers)
            if resp.status_code != 200:
                _log(f"SKIPPED train_and_score on {entry.get('name')}: {resp.status_code}")
                break


def drive_dataset_promote(client, datasets: list[dict], reps: int) -> None:
    """Measure promoting a subset into a new dataset (atlas, serialize, register)."""
    from vtscore.concurrency.progress import loading_tasks  # noqa: PLC0415

    for entry in datasets:
        if not _open_dataset(client, entry["id"], reopen=False):
            continue
        headers = {"X-Dataset-Id": entry["id"]}
        listing = client.get("/api/medias", headers=headers)
        if listing.status_code != 200:
            _log(f"SKIPPED dataset_promote on {entry.get('name')}: cannot list medias")
            continue
        media_ids = [m["id"] for m in (listing.get_json() or {}).get("medias", [])]
        if not media_ids:
            _log(f"SKIPPED dataset_promote on {entry.get('name')}: no medias")
            continue
        for rep in range(reps):
            _log(f"dataset_promote {entry.get('name')} ({len(media_ids)} items) rep {rep + 1}/{reps}")
            resp = client.post(
                "/api/dataset/promote",
                json={"name": f"_tuning_{entry['id'][:8]}_{rep}", "media_ids": media_ids},
                headers=headers,
            )
            if resp.status_code != 200:
                _log(f"SKIPPED dataset_promote on {entry.get('name')}: {resp.status_code}")
                break
            task_id = (resp.get_json() or {}).get("task_id") or ""
            _wait_for_task(loading_tasks, task_id, f"promote {entry['id']}")
    _log("NOTE: dataset_promote left '_tuning_*' datasets in the registry; delete them when done.")


def drive_dataset_load(demo_ids: list[str], reps: int) -> None:
    """Measure importing demo datasets end to end (acquire, load, embed, finalize)."""
    from vtscore.concurrency.progress import loading_tasks  # noqa: PLC0415
    from vtscore.datasets.config import DEMO_DATASETS  # noqa: PLC0415
    from vtscore.datasets.importers import get_importer  # noqa: PLC0415
    from vtscore.datasets.load_pipeline import _run_importer_in_background  # noqa: PLC0415

    if not demo_ids:
        _log("SKIPPED dataset_load: no --demo ids given")
        return
    importer = get_importer("demo")
    for demo_id in demo_ids:
        info = DEMO_DATASETS.get(demo_id)
        if info is None:
            _log(f"SKIPPED demo {demo_id}: unknown demo dataset")
            continue
        for rep in range(reps):
            _log(f"dataset_load {demo_id} rep {rep + 1}/{reps}")
            task_id = _run_importer_in_background(
                importer,
                {"name": demo_id, "media_type": info.get("media_type", ""), "embedder": ""},
            )
            _wait_for_task(loading_tasks, task_id, f"demo {demo_id}")


def run_drivers(args: argparse.Namespace, tasks: list[str]) -> None:
    """Dispatch each requested family's driver against the resolved inputs."""
    import app as app_module  # noqa: PLC0415 - wires Flask + every plugin registry

    app_module.app.config["TESTING"] = True
    datasets = _resolve_datasets(_split(args.datasets))
    detectors = _resolve_detectors(_split(args.detectors))
    if not datasets and tasks != ["dataset_load"]:
        _log("nothing to drive: no datasets matched. Load or name at least one registry dataset.")
    with app_module.app.test_client() as client:
        if "dataset_open" in tasks:
            drive_dataset_open(client, datasets, args.reps)
        if "text_sort" in tasks:
            drive_text_sort(client, datasets, _split(args.queries), args.reps)
        if "find" in tasks:
            drive_find(client, datasets, detectors, args.reps)
        if "detector_load" in tasks:
            drive_detector_load(client, datasets, detectors, args.reps)
        if "train_and_score" in tasks:
            drive_train_and_score(client, datasets, detectors, args.reps)
        if "dataset_promote" in tasks:
            drive_dataset_promote(client, datasets, args.reps)
    if "dataset_load" in tasks:
        drive_dataset_load(_split(args.demo), args.reps)


# ---------------------------------------------------------------------------
# Fit + emit
# ---------------------------------------------------------------------------


def write_profile(args: argparse.Namespace, jsonl_paths: list[str]) -> int:
    """Fit the recorded rows, write the profile, and print the coverage report."""
    from vtscore.timing.fit import coverage_report, fit_profile, load_rows  # noqa: PLC0415

    existing = [p for p in jsonl_paths if Path(p).is_file()]
    missing = [p for p in jsonl_paths if p not in existing]
    for path in missing:
        _log(f"SKIPPED {path}: no such file")
    if not existing:
        _log("no timing rows to fit — nothing written")
        return 1

    rows = load_rows(existing)
    _log(f"read {len(rows)} recorded rows from {len(existing)} file(s)")
    profile = fit_profile(
        rows,
        min_samples=args.min_samples,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        host=socket.gethostname(),
        notes=args.notes,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _log(f"wrote {out}")
    _log("coverage:")
    for line in coverage_report(rows, profile):
        print(line, file=sys.stderr)
    if not profile.get("tasks"):
        _log("WARNING: no cell had enough runs to fit. The profile is empty and every")
        _log("         task will keep using its built-in defaults. Record more runs,")
        _log("         or lower --min-samples if you accept thinner evidence.")
        return 1
    _log(f"deploy with: VTSEARCH_TIMING_PROFILE={out.resolve()}")
    return 0


def main() -> int:
    args = _parse_args()
    record_path = args.record or f"{args.out}.jsonl"

    if not args.fit_only and not args.drive:
        _log("pass --drive to exercise the workloads, or --fit-only to fit already-recorded JSONL.")
        return 2

    if args.fit_only:
        sources = args.jsonl or ([args.record] if args.record else [])
        if not sources:
            _log("--fit-only needs at least one recorded JSONL path (or --record).")
            return 2
        if args.dry_run:
            _log(f"would fit {sources} into {args.out}")
            return 0
        return write_profile(args, sources)

    # Drive mode: arm the recorder and isolate the data dir *before* importing
    # the app, since both are read at import time.
    os.environ["VTSEARCH_TIMING_RECORD"] = os.path.abspath(record_path)
    if args.data_dir:
        Path(args.data_dir).mkdir(parents=True, exist_ok=True)
        os.environ["VTSEARCH_DATA_DIR"] = os.path.abspath(args.data_dir)
    # A profile already in the environment would pace the very loads being
    # measured, folding the old profile's opinions into the new one's timings.
    os.environ.pop("VTSEARCH_TIMING_PROFILE", None)

    tasks = _resolve_tasks(args)
    if not tasks:
        _log("no task families left to drive")
        return 2
    _log(f"driving: {', '.join(tasks)}  (reps={args.reps}, record={record_path})")
    if args.dry_run:
        return 0

    run_drivers(args, tasks)
    return write_profile(args, [record_path] + list(args.jsonl))


if __name__ == "__main__":
    raise SystemExit(main())
