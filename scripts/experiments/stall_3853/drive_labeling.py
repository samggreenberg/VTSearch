#!/usr/bin/env python3
"""Drive a labeling session against a running VTSearch the way the SPA does.

Issue #3853: a rare multi-second stall after a vote.  This replays the
per-vote request chain the frontend issues (vote POST, then the fan-out of
``/api/votes``, ``/api/labeling-status``, ``/api/jobs/active``,
``/api/achievements``, three ``GET /api/detectors/<name>``, a ``medias/batch``
and the next image, then ``POST /api/learned-sort`` 300 ms later polled to
completion) at a chosen cadence, timing every request from the client side,
and reports the outliers the same way the issue's comments did:
*keypress -> panel filled* (vote start to image end) and *sort wait*.

Stdlib only, so it runs from a GRID login node's system Python as well as
from a laptop through the tunnel.

Typical use::

    python drive_labeling.py --base-url http://localhost:5000 \
        --dataset-id <id> --detector-id <id> --votes 400 --region \
        --out reqlog.jsonl

``--flip-every N`` re-votes an earlier item with the opposite label every N
votes.  That is the one client action that truncates the labeling-status
cache and makes its worker replay the history under ``_progress_lock``
(see ``vtscore/detectors/labeling_progress.py``), which is a candidate
mechanism for the stall; a run with and without it separates that.

Media ids default to ``0..num_medias-1`` from ``/api/dataset/status``;
pass ``--ids lo:hi`` when the dataset is keyed differently.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class Client:
    """Tiny HTTP client that records every request's timing."""

    def __init__(self, base_url: str, dataset_id: str, detector_id: str, out_path: str | None) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "X-Dataset-Id": dataset_id,
            "X-Detector-Id": detector_id,
            "Content-Type": "application/json",
        }
        self.records: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._out = open(out_path, "a", encoding="utf-8") if out_path else None  # noqa: SIM115
        self.t_origin = time.time()

    def close(self) -> None:
        if self._out is not None:
            self._out.close()

    def call(self, method: str, path: str, body: Any = None, *, vote_index: int, tag: str) -> dict[str, Any]:
        data = json.dumps(body).encode() if body is not None else None
        # Detector names carry spaces and parentheses ("knife incl butter
        # knives (confirm the box)"); the SPA percent-encodes them and
        # urllib refuses a raw space, so quote the path the same way.
        url = self.base_url + urllib.parse.quote(path, safe="/?=&%")
        req = urllib.request.Request(url, data=data, method=method, headers=self.headers)  # noqa: S310
        t0 = time.time()
        status = 0
        payload: Any = None
        request_id = ""
        error = ""
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:  # noqa: S310 - operator-supplied URL
                status = resp.status
                request_id = resp.headers.get("X-Request-Id", "")
                raw = resp.read()
                if resp.headers.get("Content-Type", "").startswith("application/json"):
                    payload = json.loads(raw) if raw else None
                else:
                    payload = len(raw)
        except urllib.error.HTTPError as exc:
            status = exc.code
            error = exc.read().decode(errors="replace")[:200]
        except Exception as exc:  # noqa: BLE001 - keep driving, record the failure
            error = repr(exc)
        t1 = time.time()
        rec = {
            "t0": t0,
            "t1": t1,
            "ms": (t1 - t0) * 1000.0,
            "method": method,
            "path": path,
            "status": status,
            "vote_index": vote_index,
            "tag": tag,
            "request_id": request_id,
        }
        if error:
            rec["error"] = error
        with self._lock:
            self.records.append(rec)
            if self._out is not None:
                self._out.write(json.dumps(rec) + "\n")
                self._out.flush()
        rec["payload"] = payload
        return rec


def _get_json(client: Client, path: str) -> Any:
    return client.call("GET", path, vote_index=-1, tag="setup")["payload"]


def _entries(payload: Any, key: str) -> list[dict[str, Any]]:
    """Registry listings come back as ``{key: [...]}``; tolerate a bare list too."""
    if isinstance(payload, dict):
        payload = payload.get(key, [])
    return [e for e in (payload or []) if isinstance(e, dict)]


def _resolve_ids(client: Client, spec: str | None) -> list[int]:
    if spec:
        lo, _, hi = spec.partition(":")
        return list(range(int(lo), int(hi)))
    stubs = _get_json(client, "/api/medias/ids") or []
    ids = [int(s["id"]) for s in stubs if isinstance(s, dict) and "id" in s]
    if not ids:
        sys.exit("dataset reports no medias; pass --ids lo:hi")
    return ids


def _detector_name(client: Client, detector_id: str) -> str:
    for entry in _entries(_get_json(client, "/api/detectors/registry"), "detectors"):
        if entry.get("id") == detector_id:
            return entry.get("name", detector_id)
    return detector_id


def _resolve_dataset(base_url: str, spec: str) -> str:
    """``auto`` picks the first loaded dataset in the registry."""
    if spec != "auto":
        return spec
    probe = Client(base_url, "", "", None)
    entries = _entries(_get_json(probe, "/api/datasets/registry"), "datasets")
    loaded = [e for e in entries if e.get("loaded")]
    if not loaded:
        sys.exit("no loaded dataset in the registry; load one or pass --dataset-id")
    print(f"dataset: {loaded[0].get('name')} ({loaded[0]['id']})")
    return loaded[0]["id"]


def _ensure_detector(
    base_url: str,
    dataset_id: str,
    detector_id: str | None,
    create: str | None,
    embedder_type: str = "patch_semantic",
    text_query: str = "",
) -> str:
    """Return a loaded detector id, registering *create* and loading it if asked."""
    probe = Client(base_url, dataset_id, "", None)
    if create:
        body = {
            "name": create,
            "media_type": "image",
            "trainable": True,
            "embedder_type": embedder_type,
            "text_query": text_query,
        }
        made = probe.call("POST", "/api/detectors/registry", body, vote_index=-1, tag="setup")
        if made["status"] == 201:
            detector_id = ((made["payload"] or {}).get("detector") or {}).get("id")
        elif made["status"] == 409:
            for e in _entries(_get_json(probe, "/api/detectors/registry"), "detectors"):
                if e.get("name") == create:
                    detector_id = e["id"]
        else:
            sys.exit(f"detector registration failed: {made['status']} {made.get('error')}")
    if not detector_id:
        entries = [e for e in _entries(_get_json(probe, "/api/detectors/registry"), "detectors") if e.get("loaded")]
        if not entries:
            sys.exit("no loaded detector; pass --detector-id or --create-detector NAME")
        detector_id = entries[0]["id"]
    loaded = {e["id"]: e.get("loaded") for e in _entries(_get_json(probe, "/api/detectors/registry"), "detectors")}
    if not loaded.get(detector_id):
        probe.call("POST", "/api/detectors/registry/load", {"detector_id": detector_id}, vote_index=-1, tag="setup")
        deadline = time.time() + 120
        while time.time() < deadline:
            time.sleep(0.5)
            loaded = {
                e["id"]: e.get("loaded") for e in _entries(_get_json(probe, "/api/detectors/registry"), "detectors")
            }
            if loaded.get(detector_id):
                break
        else:
            sys.exit("detector did not finish loading in 120s")
    print(f"detector: {detector_id} loaded")
    return detector_id


def _fanout(client: Client, vote_index: int, det_name: str, next_ids: list[int]) -> list[dict[str, Any]]:
    """The requests the SPA fires after a vote lands, concurrently like a browser."""
    calls: list[tuple[str, str, Any, str]] = [
        ("GET", "/api/votes", None, "votes"),
        ("GET", "/api/labeling-status", None, "labeling-status"),
        ("GET", "/api/jobs/active", None, "jobs-active"),
        ("GET", "/api/achievements", None, "achievements"),
        ("GET", f"/api/detectors/{det_name}", None, "detector-1"),
        ("GET", f"/api/detectors/{det_name}", None, "detector-2"),
        ("GET", f"/api/detectors/{det_name}", None, "detector-3"),
        ("POST", "/api/medias/batch", {"ids": next_ids[:8]}, "batch"),
    ]
    if next_ids:
        calls.append(("GET", f"/api/medias/{next_ids[0]}/image", None, "image"))
    results: list[dict[str, Any]] = []
    lock = threading.Lock()

    def run(method: str, path: str, body: Any, tag: str) -> None:
        rec = client.call(method, path, body, vote_index=vote_index, tag=tag)
        with lock:
            results.append(rec)

    threads = [threading.Thread(target=run, args=c) for c in calls]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def _learned_sort(client: Client, vote_index: int) -> tuple[float | None, str]:
    """POST the sort and poll it to completion; returns (seconds waited, status)."""
    t0 = time.time()
    first = client.call("POST", "/api/learned-sort", {"wait": False}, vote_index=vote_index, tag="sort")
    payload = first["payload"] or {}
    if first["status"] != 200:
        return None, f"http {first['status']}"
    if payload.get("status") == "done":
        return time.time() - t0, "cached"
    job_id = payload.get("job_id")
    if not job_id:
        return None, "no job id"
    delay = 0.2
    deadline = time.time() + 120
    while time.time() < deadline:
        time.sleep(delay)
        poll = client.call("GET", f"/api/learned-sort/result?job_id={job_id}", vote_index=vote_index, tag="sort-poll")
        st = (poll["payload"] or {}).get("status") if poll["status"] == 200 else f"http {poll['status']}"
        if st in ("done", "error", "cancelled") or (isinstance(st, str) and st.startswith("http")):
            return time.time() - t0, str(st)
        delay = min(delay * 1.5, 0.5)
    return time.time() - t0, "timeout"


def _pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def _fmt_stats(name: str, values: list[float]) -> str:
    if not values:
        return f"{name:>22}: n=0"
    return (
        f"{name:>22}: n={len(values):<5d} p50={_pct(values, 0.5):7.0f}ms  p90={_pct(values, 0.9):7.0f}ms  "
        f"p99={_pct(values, 0.99):7.0f}ms  max={max(values):7.0f}ms"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:5000")
    ap.add_argument("--dataset-id", default="auto", help="registry id, or 'auto' for the first loaded dataset")
    ap.add_argument("--detector-id", default=None, help="registry id (default: the first loaded detector)")
    ap.add_argument("--create-detector", default=None, help="register (image, --embedder-type) and load this detector")
    ap.add_argument(
        "--embedder-type",
        default="patch_semantic",
        help="embedder type for --create-detector: semantic (the VG slates as labeled) or patch_semantic",
    )
    ap.add_argument("--text-query", default="", help="text query for --create-detector, as the SPA sets it")
    ap.add_argument("--votes", type=int, default=200)
    ap.add_argument("--cadence", type=float, default=0.0, help="seconds of think time between votes (after the chain)")
    ap.add_argument("--good-frac", type=float, default=0.5)
    ap.add_argument("--region", action="store_true", help="send a region box with every good vote (patch datasets)")
    ap.add_argument("--flip-every", type=int, default=0, help="every N votes, relabel an earlier item the other way")
    ap.add_argument("--no-sort", action="store_true", help="skip the learned sort after each vote")
    ap.add_argument("--ids", default=None, help="media id range lo:hi (default: 0..num_medias)")
    ap.add_argument("--seed", type=int, default=3853)
    ap.add_argument("--out", default=None, help="JSONL file to append every request record to")
    args = ap.parse_args()

    dataset_id = _resolve_dataset(args.base_url, args.dataset_id)
    detector_id = _ensure_detector(
        args.base_url, dataset_id, args.detector_id, args.create_detector, args.embedder_type, args.text_query
    )
    client = Client(args.base_url, dataset_id, detector_id, args.out)
    rng = random.Random(args.seed)
    ids = _resolve_ids(client, args.ids)
    rng.shuffle(ids)
    det_name = _detector_name(client, detector_id)
    print(
        f"driving {args.votes} votes on {len(ids)} medias; detector {det_name!r}; sort={'off' if args.no_sort else 'on'}"
    )

    voted: list[tuple[int, str]] = []
    panel: list[float] = []
    sort_waits: list[float] = []
    slow_events: list[tuple[float, int, str, float]] = []  # (ms, vote_index, what, t0)
    try:
        for i in range(args.votes):
            if args.flip_every and voted and i % args.flip_every == args.flip_every - 1:
                media_id, prev = rng.choice(voted[: max(1, len(voted) // 2)])
                target = "bad" if prev == "good" else "good"
                kind = "flip"
            else:
                media_id = ids[i % len(ids)]
                target = "good" if rng.random() < args.good_frac else "bad"
                kind = "vote"
            body: dict[str, Any] = {"target": target}
            if args.region and target == "good":
                x0, y0 = rng.uniform(0.0, 0.6), rng.uniform(0.0, 0.6)
                body["region_box"] = [x0, y0, x0 + 0.3, y0 + 0.3]
            vote = client.call("POST", f"/api/medias/{media_id}/vote", body, vote_index=i, tag=kind)
            if vote["status"] == 200:
                voted.append((media_id, target))
            next_ids = ids[(i + 1) % len(ids) : (i + 1) % len(ids) + 8]
            fan = _fanout(client, i, det_name, next_ids)
            image = next((r for r in fan if r["tag"] == "image"), None)
            if image is not None:
                p_ms = (image["t1"] - vote["t0"]) * 1000.0
                panel.append(p_ms)
                if p_ms >= 1000:
                    slow_events.append((p_ms, i, "keypress->panel", vote["t0"]))
            for r in [vote, *fan]:
                if r["ms"] >= 1000:
                    slow_events.append((r["ms"], i, f"{r['method']} {r['path']}", r["t0"]))
            if not args.no_sort:
                time.sleep(0.3)
                waited, st = _learned_sort(client, i)
                if waited is not None:
                    sort_waits.append(waited * 1000.0)
                    if waited >= 2.0:
                        slow_events.append((waited * 1000.0, i, f"sort ({st})", vote["t0"]))
            if i % 25 == 0:
                print(
                    f"  vote {i:4d}: vote {vote['ms']:5.0f}ms  panel {panel[-1] if panel else float('nan'):6.0f}ms  "
                    f"sort {sort_waits[-1] if sort_waits else float('nan'):6.0f}ms  labels={len(voted)}",
                    flush=True,
                )
            if args.cadence > 0:
                time.sleep(args.cadence)
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        client.close()

    recs = [r for r in client.records if r["vote_index"] >= 0]
    by_tag: dict[str, list[float]] = {}
    for r in recs:
        by_tag.setdefault(r["tag"], []).append(r["ms"])
    print()
    print(_fmt_stats("keypress->panel", panel))
    print(_fmt_stats("sort wait", sort_waits))
    for tag in sorted(by_tag):
        print(_fmt_stats(tag, by_tag[tag]))
    errors = [r for r in recs if r.get("error") or (r["status"] and r["status"] >= 400)]
    if errors:
        print(f"\n{len(errors)} failed requests, first: {errors[0]}")
    if slow_events:
        print("\nslowest events (>=1s request or panel wait, >=2s sort):")
        for ms, vi, what, t0 in sorted(slow_events, reverse=True)[:15]:
            print(f"  {ms:8.0f}ms  vote {vi:4d}  {what}  at {time.strftime('%H:%M:%S', time.localtime(t0))}")
    # Head-of-line signature: several requests that overlapped in time and all
    # ended within 100 ms of each other after being slow.
    ends = sorted((r["t1"], r) for r in recs if r["ms"] >= 1000)
    clusters = 0
    j = 0
    while j < len(ends):
        k = j
        while k + 1 < len(ends) and ends[k + 1][0] - ends[j][0] < 0.1:
            k += 1
        if k - j + 1 >= 3:
            clusters += 1
            names = ", ".join(f"{r['tag']} {r['ms']:.0f}ms" for _, r in ends[j : k + 1])
            print(f"  head-of-line cluster at {time.strftime('%H:%M:%S', time.localtime(ends[j][0]))}: {names}")
        j = k + 1
    if recs:
        total_s = recs[-1]["t1"] - recs[0]["t0"]
        print(
            f"\n{len(recs)} requests in {total_s:.0f}s; mean vote POST {statistics.fmean(by_tag.get('vote', [0])):.0f}ms"
        )
    print(f"head-of-line clusters: {clusters}")


if __name__ == "__main__":
    main()
