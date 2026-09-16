#!/usr/bin/env python3
"""Sample the things a stall shows up in but the app cannot see (#3853).

The app's own instruments stop at the process boundary.  Two of the leading
explanations in that issue live outside it:

* **filesystem latency** - the detector write, the registry read-modify-write
  and every page fault of the venv go to a shared NFS export whose tail runs
  to hundreds of milliseconds, while its median is ~1 ms.  A burst probe
  cannot rule that in or out; only a sampler running *through* the session
  can say whether the mount was slow at the second the reviewer felt a pause.
* **memory pressure** - a process that is not being scheduled, or is faulting
  pages back in, freezes in exactly the observed shape, and the watchdog's
  `majflt` line only fires when the interpreter itself stalls.

So this samples both on a fixed interval and writes one JSON object per
sample, to be read against the app log by timestamp.  It replaces the two
ad-hoc shell samplers that ran from a scratch directory during the
2026-09-15 sessions and were never version-controlled.

Stdlib only, no vtscore import: it has to run in a bare shell on a node that
may not have the venv activated, and it must not perturb what it measures.

Sample::

    # run beside a labeling session; Ctrl-C or --duration stops it
    python sample_host.py --mount /exp/$USER --mount /expscratch/$USER \\
        --match 'python app.py' --interval 5 --out nfs-$(hostname -s).jsonl

    # afterwards, the tables that go in an issue comment
    python sample_host.py --summarize nfs-rack7n06.jsonl

NFS figures come from the kernel's own per-mount RPC accounting
(``/proc/self/mountstats``), **differenced per interval**.  Reading those
counters raw gives the average since the mount was made -- 82 days in one
case, dominated by past bulk writes -- which is how a 75 ms "WRITE average"
got into this issue's record when the steady-state cost was ~1 ms.  Every
number this prints is per-interval; none is a lifetime average.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Iterable

#: Per-op fields in a ``mountstats`` line, in order.  Times are milliseconds
#: and every one is cumulative since mount.
_OP_FIELDS = ("ops", "trans", "timeouts", "bytes_sent", "bytes_recv", "queue_ms", "rtt_ms", "exec_ms", "errors")

#: Ops worth watching by default.  The full set is ~60 per mount and would
#: dominate the file; these are the three a vote actually performs.
DEFAULT_OPS = ("READ", "WRITE", "GETATTR", "ACCESS", "LOOKUP", "COMMIT")


# ---------------------------------------------------------------------------
# /proc parsing (pure functions, so they can be tested without an NFS mount)
# ---------------------------------------------------------------------------


def parse_mountstats(text: str) -> dict[str, dict[str, dict[str, int]]]:
    """``{mountpoint: {OP: {field: counter}}}`` for every NFS mount in *text*.

    Non-NFS mounts are skipped: they have no per-op section, and including
    them would make a caller's "mount not found" ambiguous between "not
    mounted" and "not NFS".
    """
    out: dict[str, dict[str, dict[str, int]]] = {}
    current: dict[str, dict[str, int]] | None = None
    in_per_op = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("device "):
            in_per_op = False
            current = None
            # device <src> mounted on <mountpoint> with fstype <fs> [statvers=…]
            parts = line.split()
            try:
                mount = parts[parts.index("on") + 1]
                fstype = parts[parts.index("fstype") + 1]
            except (ValueError, IndexError):
                continue
            if not fstype.startswith("nfs"):
                continue
            current = {}
            out[mount] = current
            continue
        if current is None:
            continue
        if line.startswith("per-op statistic"):
            in_per_op = True
            continue
        if not in_per_op or ":" not in line:
            continue
        name, _, rest = line.partition(":")
        values = rest.split()
        if not name.isupper() or not values or not all(v.lstrip("-").isdigit() for v in values):
            continue
        current[name] = {f: int(v) for f, v in zip(_OP_FIELDS, values)}
    return out


def diff_ops(before: dict[str, dict[str, int]], after: dict[str, dict[str, int]], ops: Iterable[str]) -> dict[str, Any]:
    """Per-interval deltas, with the per-operation times a reader wants.

    ``exec`` is the whole round trip as the client saw it, ``queue`` the part
    spent waiting to be sent (client-side backlog) and ``rtt`` the part on the
    wire.  Reporting all three is what distinguishes a slow server from a
    saturated client.
    """
    out: dict[str, Any] = {}
    for op in ops:
        b, a = before.get(op), after.get(op)
        if not b or not a:
            continue
        n = a["ops"] - b["ops"]
        if n <= 0:
            continue
        row: dict[str, Any] = {"ops": n}
        for field, label in (("exec_ms", "exec"), ("queue_ms", "queue"), ("rtt_ms", "rtt")):
            row[label] = round((a[field] - b[field]) / n, 3)
        row["bytes"] = (a["bytes_sent"] - b["bytes_sent"]) + (a["bytes_recv"] - b["bytes_recv"])
        out[op] = row
    return out


def parse_proc_stat(text: str) -> dict[str, int] | None:
    """``minflt``/``majflt``/``rss_kb`` out of a ``/proc/<pid>/stat`` line.

    The ``comm`` field is parenthesised and may contain spaces, so the split
    has to start after the last ``)`` rather than at the second field.
    """
    tail = text.rsplit(")", 1)[-1].split()
    # Fields after comm, 0-indexed: state(0) … minflt(7) … majflt(9) … rss(21)
    if len(tail) < 22:
        return None
    try:
        page_kb = os.sysconf("SC_PAGE_SIZE") // 1024
    except (AttributeError, ValueError, OSError):  # pragma: no cover - non-POSIX
        page_kb = 4
    return {"minflt": int(tail[7]), "majflt": int(tail[9]), "rss_kb": int(tail[21]) * page_kb}


def find_pids(needle: str) -> list[int]:
    """Every pid whose cmdline contains *needle*; the app restarts, so the
    sampler resolves this on every sample rather than pinning one pid."""
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as fh:
                cmdline = fh.read().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if needle in cmdline:
            found.append(int(entry))
    return sorted(found)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _read(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return None


def sample_once(mounts: list[str], pids: list[int], ops: Iterable[str], prev: dict[str, Any] | None) -> dict[str, Any]:
    """One sample.  ``prev`` carries the raw counters the deltas are against."""
    now = time.time()
    raw = parse_mountstats(_read("/proc/self/mountstats") or "")
    row: dict[str, Any] = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)), "mounts": {}, "procs": {}}
    for mount in mounts:
        cur = raw.get(mount)
        if cur is None:
            row["mounts"][mount] = {"error": "not an nfs mount here"}
            continue
        if prev and mount in prev.get("_raw", {}):
            row["mounts"][mount] = diff_ops(prev["_raw"][mount], cur, ops)
    for pid in pids:
        stat = _read(f"/proc/{pid}/stat")
        parsed = parse_proc_stat(stat) if stat else None
        if parsed:
            row["procs"][str(pid)] = parsed
    row["_raw"] = raw
    return row


def run(args: argparse.Namespace) -> None:
    out = open(args.out, "a", encoding="utf-8") if args.out else sys.stdout
    deadline = time.time() + args.duration if args.duration else None
    prev: dict[str, Any] | None = None
    try:
        while deadline is None or time.time() < deadline:
            pids = list(args.pid)
            if args.match:
                pids.extend(p for p in find_pids(args.match) if p not in pids)
            row = sample_once(args.mount, pids, args.ops, prev)
            prev = row
            emitted = {k: v for k, v in row.items() if not k.startswith("_")}
            # The first sample has no interval to difference against, so it
            # carries no rates; skip it rather than print an empty one.
            if emitted["mounts"] or emitted["procs"]:
                out.write(json.dumps(emitted) + "\n")
                out.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        if out is not sys.stdout:
            out.close()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _pct(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    i = min(len(ordered) - 1, int(round(pct / 100.0 * (len(ordered) - 1))))
    return ordered[i]


def summarize(path: str) -> None:
    """Per-op percentiles per mount, and the fault/RSS range per process."""
    per_mount: dict[str, dict[str, dict[str, list[float]]]] = {}
    totals: dict[str, dict[str, int]] = {}
    procs: dict[str, list[dict[str, int]]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            for mount, ops in (row.get("mounts") or {}).items():
                for op, fields in ops.items():
                    if not isinstance(fields, dict) or "exec" not in fields:
                        continue
                    bucket = per_mount.setdefault(mount, {}).setdefault(op, {"exec": [], "queue": [], "rtt": []})
                    for key in bucket:
                        bucket[key].append(float(fields.get(key, 0.0)))
                    totals.setdefault(mount, {}).setdefault(op, 0)
                    totals[mount][op] += int(fields.get("ops", 0))
            for pid, fields in (row.get("procs") or {}).items():
                procs.setdefault(pid, []).append(fields)

    for mount, ops in sorted(per_mount.items()):
        print(f"\n{mount} - per-interval averages, so no lifetime counter is in here")
        print(f"  {'op':<10} {'ops':>8} {'exec p50':>10} {'p90':>8} {'max':>8} {'queue p50':>10} {'rtt p50':>9}")
        for op, fields in sorted(ops.items()):
            print(
                f"  {op:<10} {totals[mount][op]:>8} {_pct(fields['exec'], 50):>9.2f}ms "
                f"{_pct(fields['exec'], 90):>7.2f}ms {max(fields['exec']):>7.2f}ms "
                f"{_pct(fields['queue'], 50):>9.2f}ms {_pct(fields['rtt'], 50):>8.2f}ms"
            )
    for pid, rows in sorted(procs.items()):
        if not rows:
            continue
        majflt = [r["majflt"] for r in rows]
        rss = [r["rss_kb"] for r in rows]
        print(
            f"\npid {pid}: majflt {majflt[0]} -> {majflt[-1]} (+{majflt[-1] - majflt[0]}), "
            f"rss {min(rss) // 1024}-{max(rss) // 1024} MB"
        )
        if majflt[-1] == majflt[0]:
            # Worth stating rather than leaving to inference: it rules out the
            # paging explanation for every stall inside the window.
            print("  no major faults in this window: nothing was paged back in from disk")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mount", action="append", default=[], help="mountpoint to sample (repeatable)")
    ap.add_argument("--pid", action="append", type=int, default=[], help="pid to sample (repeatable)")
    ap.add_argument("--match", help="also sample every process whose cmdline contains this")
    ap.add_argument(
        "--ops", nargs="*", default=list(DEFAULT_OPS), help=f"NFS ops to record (default: {' '.join(DEFAULT_OPS)})"
    )
    ap.add_argument("--interval", type=float, default=5.0, help="seconds between samples")
    ap.add_argument("--duration", type=float, default=0.0, help="stop after this many seconds (0 = until Ctrl-C)")
    ap.add_argument("--out", help="append JSONL here (default stdout)")
    ap.add_argument("--summarize", help="read a JSONL written earlier and print the tables instead of sampling")
    args = ap.parse_args()

    if args.summarize:
        summarize(args.summarize)
        return
    if not args.mount and not args.pid and not args.match:
        ap.error("nothing to sample: pass --mount, --pid or --match")
    run(args)


if __name__ == "__main__":
    main()
