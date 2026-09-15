#!/usr/bin/env python3
"""Read a VTSearch JSON log and show what surrounded every stall (#3853).

The app logs one JSON object per line (``VTSEARCH_LOG_FORMAT=json``, the
default).  The lines this cares about are the ones the stall diagnostics
emit at WARNING, all present at the default log level:

* ``slow request: METHOD /path -> status in NNNms (request_id=…)`` from
  ``vtsearch.hooks`` (threshold ``VTSEARCH_SLOW_REQUEST_MS``);
* ``stall: heartbeat late by NNNms; …`` from the watchdog;
* ``gc pause: generation N took NNNms …``;
* ``slow phase: NAME total NNNms (…)`` and ``lock wait: NAME waited NNNms``;
* at INFO, ``rehydrating votes …`` and ``progress cache truncated …``.

Interleaved with those, ``faulthandler`` writes its thread dump as plain
text (``Timeout (0:00:01)!`` then one ``Thread 0x… (most recent call
first):`` block per thread).  Those lines are not JSON; they are collected
as dump blocks and attached to the stall that follows them.

Usage::

    python analyze_app_log.py data/logs/app-node-20260914-120000.log [--window 15]
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from typing import Any

_SLOW_REQ = re.compile(r"slow request: (\S+) (\S+) -> (\d+) in (\d+)ms \(request_id=([^)]*)\)")
_STALL = re.compile(r"stall: heartbeat late by (\d+)ms")
_MS = re.compile(r"(\d+)ms")


def _parse_ts(raw: str) -> float:
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _kind(msg: str) -> str | None:
    for prefix, kind in (
        ("slow request:", "request"),
        ("stall:", "stall"),
        ("gc pause:", "gc"),
        ("slow phase:", "phase"),
        ("lock wait:", "lock"),
        ("rehydrating votes", "rehydrate"),
        ("progress cache truncated", "truncate"),
        ("stall watchdog", "watchdog"),
    ):
        if msg.startswith(prefix):
            return kind
    return None


def load(path: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(events, dumps)``; each dump is ``{"line": n, "threads": [...]}``."""
    events: list[dict[str, Any]] = []
    dumps: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    thread: dict[str, Any] | None = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for n, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if line.startswith("{"):
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                msg = str(rec.get("msg", ""))
                kind = _kind(msg)
                if kind is None:
                    continue
                events.append({"line": n, "t": _parse_ts(str(rec.get("ts", ""))), "kind": kind, "msg": msg})
                continue
            # faulthandler output
            if line.startswith("Timeout (") or line.startswith("Fatal Python error"):
                current = {"line": n, "threads": []}
                dumps.append(current)
                thread = None
            elif line.startswith("Thread 0x") or line.startswith("Current thread 0x"):
                if current is None:
                    current = {"line": n, "threads": []}
                    dumps.append(current)
                thread = {"header": line, "frames": []}
                current["threads"].append(thread)
            elif thread is not None and line.startswith("  File "):
                thread["frames"].append(line.strip())
    return events, dumps


def _dump_summary(dump: dict[str, Any], frames_per_thread: int = 3) -> str:
    out = [f"  thread dump at log line {dump['line']} ({len(dump['threads'])} threads):"]
    for th in dump["threads"]:
        frames = th["frames"][:frames_per_thread]
        if not frames:
            continue
        # Skip threads parked in the obvious wait spots; they are not the story.
        top = frames[0]
        if any(s in top for s in ("threading.py", "socketserver.py", "selectors.py", "queue.py")):
            continue
        out.append(f"    {th['header']}")
        out.extend(f"      {f}" for f in frames)
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--window", type=float, default=15.0, help="seconds shown either side of a stall")
    ap.add_argument("--all", action="store_true", help="also list every slow request, not only those near a stall")
    args = ap.parse_args()

    events, dumps = load(args.log)
    counts: dict[str, int] = {}
    for e in events:
        counts[e["kind"]] = counts.get(e["kind"], 0) + 1
    print(
        f"{len(events)} diagnostic lines, {len(dumps)} thread dumps: "
        + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    )

    requests = [e for e in events if e["kind"] == "request"]
    if requests:
        durations = []
        for e in requests:
            m = _SLOW_REQ.search(e["msg"])
            if m:
                durations.append((int(m.group(4)), m.group(1), m.group(2), e["t"]))
        durations.sort(reverse=True)
        print("\nslowest requests:")
        for ms, method, path, t in durations[:10]:
            print(
                f"  {ms:7d}ms  {method} {path}  at {datetime.fromtimestamp(t, tz=timezone.utc).strftime('%H:%M:%S')}Z"
            )

    stalls = [e for e in events if e["kind"] == "stall"]
    if not stalls:
        print("\nno stall lines: the watchdog never missed a beat in this log")
    for s in stalls:
        m = _STALL.search(s["msg"])
        lag = int(m.group(1)) if m else 0
        t_end = s["t"]
        t_start = t_end - lag / 1000.0
        print("\n" + "=" * 78)
        print(
            f"STALL {lag}ms ending {datetime.fromtimestamp(t_end, tz=timezone.utc).strftime('%H:%M:%S')}Z (log line {s['line']})"
        )
        print("  " + s["msg"])
        near = [e for e in events if e is not s and t_start - args.window <= e["t"] <= t_end + args.window]
        for e in sorted(near, key=lambda e: e["t"]):
            rel = e["t"] - t_start
            print(f"  {rel:+8.2f}s  [{e['kind']:9s}] {e['msg'][:160]}")
        # The dump that fired during this stall is the last one logged before
        # the report line (faulthandler writes it mid-stall, the report after).
        before = [d for d in dumps if d["line"] < s["line"]]
        if before:
            print(_dump_summary(before[-1]))
        else:
            print("  (no thread dump found before this stall line)")

    if args.all and requests:
        print("\nall slow requests:")
        for e in requests:
            print(f"  {datetime.fromtimestamp(e['t'], tz=timezone.utc).strftime('%H:%M:%S')}Z  {e['msg']}")


if __name__ == "__main__":
    main()
