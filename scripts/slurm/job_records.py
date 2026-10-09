#!/usr/bin/env python3
"""Refuse a job array that would fill the cluster's job-record table.

Slurm's ``MaxJobCount`` caps how many jobs ``slurmctld`` holds at once: pending,
running, and finished ones not yet aged out. It is one number for the **whole
cluster**, and every array task counts against it, pending ones included. A
``%6`` throttle limits *running* tasks, not queued records, so a small throttle
does not make a large array cheap here. When the table is full, ``sbatch``
prints ``Slurm temporarily unable to accept job, sleeping and retrying`` and
gives up, for everyone's arrays and not only for yours.

On 2026-10-08 (#4668) sixteen 720-task arrays went in at ``%6``. The first
twelve held 8,500 of the cluster's ~9,200 records, and the last four were
refused (#4701). So, before an array is submitted:

* the cluster's records plus this array must stay within ``--cluster-pct`` of
  ``MaxJobCount`` (default 50%), and
* this user's records plus this array must stay within ``--user-pct`` of it
  (default 25%: half of the cluster's half).

The records are read with ``squeue -h -r -t all``: ``-r`` puts each array task
on its own line, the way ``MaxJobCount`` counts them, and ``-t all`` includes
finished jobs that ``slurmctld`` still holds.

Usage::

    python3 scripts/slurm/job_records.py --tasks 720

The verdict is the first line of stdout. Any lines after it start with ``->``
and say what to do:

* ``FITS: ...``: exit 0.
* ``OVER: ...``: exit 1. The array would cross a ceiling. The line says how
  many tasks fit right now.
* ``UNKNOWN: ...``: exit 3. ``scontrol`` or ``squeue`` could not be read. A
  ceiling nobody could read is not a passing one.

Environment:

``VTS_RECORDS_CLUSTER_PCT`` / ``VTS_RECORDS_USER_PCT``
    Defaults for ``--cluster-pct`` / ``--user-pct``.
"""

from __future__ import annotations

import argparse
import getpass
import os
import subprocess  # noqa: S404 -- runs fixed `scontrol`/`squeue` argvs, no shell, no user input
import sys
from dataclasses import dataclass

DEFAULT_CLUSTER_PCT = 50.0
DEFAULT_USER_PCT = 25.0

EXIT_FITS = 0
EXIT_OVER = 1
EXIT_UNKNOWN = 3


@dataclass(frozen=True)
class Snapshot:
    """What the scheduler holds right now."""

    max_jobs: int
    cluster: int
    mine: int
    # `PrivateData=jobs` hides other users' jobs from squeue, so `cluster` is
    # then only a lower bound.
    private: bool = False


@dataclass(frozen=True)
class Verdict:
    tag: str
    summary: str
    advice: tuple[str, ...]
    fits: int

    def render(self) -> str:
        return "\n".join([f"{self.tag}: {self.summary}", *(f"-> {a}" for a in self.advice)])


def parse_config(text: str) -> tuple[int | None, bool]:
    """``(MaxJobCount, PrivateData hides jobs)`` from ``scontrol show config``."""
    max_jobs: int | None = None
    private = False
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key, value = key.strip(), value.strip()
        if key == "MaxJobCount":
            try:
                max_jobs = int(value)
            except ValueError:
                max_jobs = None
        elif key == "PrivateData":
            private = "jobs" in {v.strip().lower() for v in value.split(",")}
    return max_jobs, private


def count_records(squeue_users: str, user: str) -> tuple[int, int]:
    """``(cluster, mine)`` from ``squeue -h -r -t all -o %u``, one user name per record."""
    names = [n.strip() for n in squeue_users.splitlines() if n.strip()]
    return len(names), sum(1 for n in names if n == user)


def judge(tasks: int, snap: Snapshot, cluster_pct: float, user_pct: float) -> Verdict:
    """Whether *tasks* more records stay under both ceilings."""
    cluster_cap = int(snap.max_jobs * cluster_pct / 100)
    user_cap = int(snap.max_jobs * user_pct / 100)
    fits = max(0, min(cluster_cap - snap.cluster, user_cap - snap.mine))
    where = (
        f"cluster {snap.cluster:,} + {tasks:,} = {snap.cluster + tasks:,} records (ceiling {cluster_cap:,}), "
        f"you {snap.mine:,} + {tasks:,} = {snap.mine + tasks:,} (ceiling {user_cap:,}), "
        f"MaxJobCount {snap.max_jobs:,}"
    )
    advice: list[str] = []
    if snap.private:
        advice.append("PrivateData=jobs: squeue shows only your own jobs, so the cluster count is a lower bound")
    if tasks <= fits:
        return Verdict("FITS", f"{tasks:,} array tasks fit: {where}", tuple(advice), fits)
    crossed = []
    if snap.cluster + tasks > cluster_cap:
        crossed.append(f"the cluster past {cluster_pct:g}%")
    if snap.mine + tasks > user_cap:
        crossed.append(f"you past {user_pct:g}%")
    advice += [
        f"{fits:,} tasks fit right now; each array task is a record from the moment it is queued",
        "a %N throttle limits running tasks, not records, so lowering it does not help",
        "chunk the array (submit what fits, the next chunk as one drains), or pack several cells per task",
    ]
    summary = f"{tasks:,} array tasks would take {' and '.join(crossed)} of MaxJobCount: {where}"
    return Verdict("OVER", summary, tuple(advice), fits)


def _run(argv: list[str]) -> str | None:
    """stdout of *argv*, or None if it could not be run or failed."""
    # A SQUEUE_USERS (or SQUEUE_STATES) in the caller's environment is a default
    # filter squeue applies silently; with it, "the cluster" would be one user.
    env = {k: v for k, v in os.environ.items() if not k.startswith("SQUEUE_")}
    try:
        proc = subprocess.run(  # noqa: S603 -- fixed argv, shell=False
            argv,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def snapshot(user: str) -> Snapshot | str:
    """The scheduler's state, or a string saying why it could not be read."""
    config = _run(["scontrol", "show", "config"])  # noqa: S607 -- resolved via PATH like every slurm call here
    if config is None:
        return "could not run `scontrol show config`"
    max_jobs, private = parse_config(config)
    if not max_jobs:
        return "`scontrol show config` has no MaxJobCount"
    users = _run(["squeue", "-h", "-r", "-t", "all", "-o", "%u"])  # noqa: S607 -- as above
    if users is None:
        return "could not run `squeue`"
    cluster, mine = count_records(users, user)
    return Snapshot(max_jobs=max_jobs, cluster=cluster, mine=mine, private=private)


def _pct(raw: str) -> float:
    value = float(raw)
    if not 0 < value <= 100:
        raise argparse.ArgumentTypeError(f"wants a percentage in (0, 100], got {raw}")
    return value


def _tasks(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError(f"wants a task count >= 0, got {raw}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", type=_tasks, required=True, help="array tasks about to be submitted")
    parser.add_argument(
        "--cluster-pct",
        type=_pct,
        # A string default goes through `type`, so a bad env value is a usage error.
        default=os.environ.get("VTS_RECORDS_CLUSTER_PCT") or str(DEFAULT_CLUSTER_PCT),
        help="ceiling on the cluster's records, as a percentage of MaxJobCount (default 50)",
    )
    parser.add_argument(
        "--user-pct",
        type=_pct,
        default=os.environ.get("VTS_RECORDS_USER_PCT") or str(DEFAULT_USER_PCT),
        help="ceiling on this user's records, as a percentage of MaxJobCount (default 25)",
    )
    parser.add_argument("--user", default=os.environ.get("USER") or getpass.getuser())
    args = parser.parse_args(argv)

    snap = snapshot(args.user)
    if isinstance(snap, str):
        print(f"UNKNOWN: {snap}, so the job-record ceilings were NOT checked")
        return EXIT_UNKNOWN
    verdict = judge(args.tasks, snap, args.cluster_pct, args.user_pct)
    print(verdict.render())
    return EXIT_FITS if verdict.tag == "FITS" else EXIT_OVER


if __name__ == "__main__":
    sys.exit(main())
