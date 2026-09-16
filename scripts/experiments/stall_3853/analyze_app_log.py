#!/usr/bin/env python3
"""Read a VTSearch JSON log and show what surrounded every stall (#3853).

The app logs one JSON object per line (``VTSEARCH_LOG_FORMAT=json``, the
default).  The lines this cares about are the ones the stall diagnostics
emit at WARNING, all present at the default log level:

* ``slow request: METHOD /path -> status in NNNms cpu=NNms gc=NNms
  (request_id=…)`` from ``vtsearch.hooks`` (threshold
  ``VTSEARCH_SLOW_REQUEST_MS``), and at ``VTSEARCH_LOG_LEVEL=INFO`` the same
  figures for *every* request as ``request trace: …``;
* ``stall: heartbeat late by NNNms; …`` from the watchdog;
* ``gc pause: generation N took NNNms …``;
* ``slow phase: NAME total NNNms (…)`` and ``lock wait: NAME waited NNNms``;
* at INFO, ``rehydrating votes …`` and ``progress cache truncated …``.

With the ``request trace`` lines present this also reconstructs a **per-vote
budget** - what each vote cycle spent, split into time the server had a
request in flight and time it did not.  That is the one question a threshold
cannot answer: #3853's residue is a felt half-second made of a chain of
requests, a retrain and a collection that are each below every bar, so the
analysis has to add the trace up rather than look for an outlier in it.

Interleaved with those, ``faulthandler`` writes its thread dump as plain
text (``Timeout (0:00:01)!`` then one ``Thread 0x… (most recent call
first):`` block per thread).  Those lines are not JSON; they are collected
as dump blocks and attached to the stall that follows them.

Usage::

    python analyze_app_log.py data/logs/app-node-20260914-120000.log [--window 15]
    python analyze_app_log.py data/logs/app-*.log --votes 10   # worst vote cycles
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from typing import Any

_SLOW_REQ = re.compile(
    r"(?:slow request|request trace): (\S+) (\S+) -> (\d+) in (\d+)ms"
    r"(?: cpu=(\d+)ms gc=(\d+)ms)? \(request_id=([^)]*)\)"
)
_VOTE_PATH = re.compile(r"^/api/medias/[^/]+/vote$")
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
        ("request trace:", "trace"),
        ("diagnostics config:", "config"),
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
    ap.add_argument("--votes", type=int, default=8, help="how many worst vote cycles to break down")
    args = ap.parse_args()

    events, dumps = load(args.log)

    # Print the bars first: every absence below is relative to them, and a
    # reader who does not know them can misread "no slow requests" as "nothing
    # was slow" when the bar was simply high (#3853, the `car` session).
    configs = [e for e in events if e["kind"] == "config"]
    if configs:
        for e in configs:
            print(f"{_hms(e['t'])}Z  {e['msg']}")
    else:
        print(
            "no `diagnostics config` line: this log predates it, so the "
            "thresholds it was written at are unknown -- read every absence with that in mind."
        )

    counts: dict[str, int] = {}
    for e in events:
        counts[e["kind"]] = counts.get(e["kind"], 0) + 1
    print(
        f"{len(events)} diagnostic lines, {len(dumps)} thread dumps: "
        + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    )

    requests = [e for e in events if e["kind"] == "request"]
    parsed = _requests(events)
    if parsed:
        print("\nslowest requests:")
        for r in sorted(parsed, key=lambda r: r["ms"], reverse=True)[:10]:
            print(f"  {r['ms']:7.0f}ms  {_cpu_gc(r)}  {r['method']} {r['path']}  at {_hms(r['end'])}Z")

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

    # A stall that is I/O rather than the interpreter - a vote's fsync to a
    # slow filesystem, a stat under ``_state_lock`` - never trips the
    # watchdog, so it shows only as slow phases, lock waits and the requests
    # queued behind them.  List those too, so "no stall lines" is not read as
    # "nothing happened".
    waits = [e for e in events if e["kind"] in ("phase", "lock")]
    if waits:
        print(f"\nslow phases and lock waits ({len(waits)}), slowest first:")
        for e in sorted(waits, key=lambda e: _duration_ms(e["msg"]), reverse=True)[:15]:
            print(f"  {datetime.fromtimestamp(e['t'], tz=timezone.utc).strftime('%H:%M:%S')}Z  {e['msg'][:170]}")

    _vote_budgets(events, parsed, worst=args.votes)

    if args.all and requests:
        print("\nall slow requests:")
        for e in requests:
            print(f"  {datetime.fromtimestamp(e['t'], tz=timezone.utc).strftime('%H:%M:%S')}Z  {e['msg']}")


# ---------------------------------------------------------------------------
# Per-vote budgets
# ---------------------------------------------------------------------------


def _hms(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%H:%M:%S")


def _cpu_gc(r: dict[str, Any]) -> str:
    if r["cpu"] is None:
        return "cpu=?     "
    return f"cpu={r['cpu']:.0f}ms gc={r['gc']:.0f}ms"


def _requests(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every timed request in the log, slow-line or trace-line alike.

    The log record is written when the request *finishes*, so its timestamp
    is the end and the start is that minus the duration.  Getting this the
    wrong way round would shift every request by its own length, which is
    exactly the scale of the thing being measured.
    """
    out: list[dict[str, Any]] = []
    for e in events:
        if e["kind"] not in ("request", "trace"):
            continue
        m = _SLOW_REQ.search(e["msg"])
        if not m:
            continue
        ms = float(m.group(4))
        cpu = float(m.group(5)) if m.group(5) is not None else None
        gc_ms = float(m.group(6)) if m.group(6) is not None else None
        out.append(
            {
                "method": m.group(1),
                "path": m.group(2),
                "status": m.group(3),
                "ms": ms,
                "cpu": cpu,
                "gc": gc_ms,
                "request_id": m.group(7),
                "end": e["t"],
                "start": e["t"] - ms / 1000.0,
                "slow": e["kind"] == "request",
            }
        )
    out.sort(key=lambda r: r["start"])
    return out


def _union_ms(intervals: list[tuple[float, float]]) -> float:
    """Wall time covered by at least one interval, in ms.

    A *union*, not a sum: a vote cycle fires ~10 requests and several overlap,
    so summing their durations reports more busy time than the clock has and
    makes every cycle look saturated.
    """
    total = 0.0
    end = None
    for lo, hi in sorted(intervals):
        if end is None or lo > end:
            total += hi - lo
            end = hi
        elif hi > end:
            total += hi - end
            end = hi
    return total * 1000.0


def _vote_budgets(events: list[dict[str, Any]], reqs: list[dict[str, Any]], worst: int = 8) -> None:
    """Break the trace into vote cycles and print the most expensive ones.

    A cycle runs from one vote POST's start to the next one's, which is the
    reviewer's own unit: keypress to keypress.  Within it, ``busy`` is the
    wall time the server had at least one request in flight and ``gap`` is
    the rest - client-side work, the browser's own queueing, and think time,
    which is why a large ``gap`` alone proves nothing.  What the split does
    prove is where a *felt* pause is not: a cycle whose span is 2 s with
    200 ms of busy was not the server.
    """
    votes = [r for r in reqs if r["method"] == "POST" and _VOTE_PATH.match(r["path"])]
    if len(votes) < 2:
        if not any(r["cpu"] is not None for r in reqs):
            print(
                "\nno per-vote budget: this log has no `request trace` lines. "
                "Re-run with VTSEARCH_LOG_LEVEL=INFO to record every request."
            )
        return

    cycles = []
    for vote, nxt in zip(votes, votes[1:]):
        lo, hi = vote["start"], nxt["start"]
        inside = [r for r in reqs if r["start"] >= lo and r["start"] < hi]
        span_ms = (hi - lo) * 1000.0
        busy_ms = _union_ms([(r["start"], r["end"]) for r in inside])
        gc_lines = [e for e in events if e["kind"] == "gc" and lo <= e["t"] < hi]
        # GC and phase lines are exact and non-overlapping among themselves,
        # unlike the per-request ``gc=`` column, which several concurrent
        # requests each charge for the same collection.
        gc_ms = sum(_duration_ms(e["msg"]) for e in gc_lines)
        marks = [e for e in events if e["kind"] in ("phase", "lock", "stall") and lo <= e["t"] < hi]
        sort_ms = _sort_wait_ms(inside)
        cycles.append(
            {
                "vote": vote,
                "span": span_ms,
                "busy": busy_ms,
                "gap": max(0.0, span_ms - busy_ms),
                "n": len(inside),
                "gc": gc_ms,
                "sort": sort_ms,
                "inside": inside,
                "marks": marks,
            }
        )

    busies = sorted(c["busy"] for c in cycles)
    print(f"\nper-vote budget over {len(cycles)} vote cycles (keypress to keypress):")
    print(
        f"  server busy per cycle: p50 {_pct(busies, 50):.0f}ms  p90 {_pct(busies, 90):.0f}ms  max {busies[-1]:.0f}ms"
    )
    votes_ms = sorted(c["vote"]["ms"] for c in cycles)
    print(
        f"  vote POST itself:      p50 {_pct(votes_ms, 50):.0f}ms  p90 {_pct(votes_ms, 90):.0f}ms  max {votes_ms[-1]:.0f}ms"
    )
    print(f"  requests per cycle:    {sum(c['n'] for c in cycles) / len(cycles):.1f} mean")

    print(f"\n  worst {min(worst, len(cycles))} cycles by server-busy time:")
    for c in sorted(cycles, key=lambda c: c["busy"], reverse=True)[:worst]:
        print(
            f"\n  {_hms(c['vote']['start'])}Z  span={c['span']:.0f}ms busy={c['busy']:.0f}ms "
            f"gap={c['gap']:.0f}ms reqs={c['n']} gc={c['gc']:.0f}ms sort_wait={c['sort']:.0f}ms"
        )
        for r in sorted(c["inside"], key=lambda r: r["ms"], reverse=True)[:6]:
            print(f"      {r['ms']:6.0f}ms {_cpu_gc(r)}  {r['method']} {r['path']}")
        for e in c["marks"]:
            print(f"      [{e['kind']}] {e['msg'][:170]}")


def _sort_wait_ms(inside: list[dict[str, Any]]) -> float:
    """Learned-sort POST start to the last result poll's end, within a cycle."""
    post = [r for r in inside if r["method"] == "POST" and r["path"].endswith("/api/learned-sort")]
    polls = [r for r in inside if "/api/learned-sort/result" in r["path"]]
    if not post:
        return 0.0
    end = max([r["end"] for r in polls], default=post[0]["end"])
    return max(0.0, (end - post[0]["start"]) * 1000.0)


def _pct(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    i = min(len(sorted_vals) - 1, int(round(pct / 100.0 * (len(sorted_vals) - 1))))
    return sorted_vals[i]


_MS = re.compile(r"(\d+)ms")


def _duration_ms(msg: str) -> int:
    """The first ``NNNms`` in a phase / lock-wait line (its total)."""
    m = _MS.search(msg)
    return int(m.group(1)) if m else 0


if __name__ == "__main__":
    main()
