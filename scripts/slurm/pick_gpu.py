#!/usr/bin/env python3
"""Pick which SLURM GPU *type* to request, from what is actually free right now.

This cluster rejects an untyped ``--gres=gpu:1``, so every launcher has to name
a type -- and a named type is a pin that quietly outlives the reason it was
chosen. Every pile cell built before 2026-08-17 landed on a V100 because
``launch_pile.sh`` hardcoded ``v100`` while L40S/A100 nodes sat idle; on the
same 384 images, same fp32 code, L40S embedded ``siglip`` 1.7x and ``siglip2_l``
**2.3x** faster (issue #3144). The opposite pin is no better: pinning ``l40s``
once meant ~5-day waits back when only two L40S nodes existed. A pin is wrong in
one direction or the other; asking the scheduler is not.

So: prefer the **fastest type that has enough free GPUs to start now**, and only
fall back toward availability when the fast types are busy. Selection order:

1. ``$VTS_GPU``, if set -- an explicit request always wins, nothing is queried.
2. The first type in the candidate list (fastest first) with ``--need`` GPUs
   free on healthy nodes of the partition.
3. Failing that, the type with the most free GPUs -- something startable beats a
   fast queue slot.
4. Failing that (nothing free anywhere), the type with the largest total pool,
   since the biggest pool drains soonest.
5. Failing that (``scontrol`` unreachable, none of the candidates exist here,
   or no node reports its GPU usage in a form this can read),
   ``$VTS_GPU_FALLBACK``.

The chosen type goes to **stdout**; the reason goes to **stderr**, so callers can
use ``$(pick_gpu.py)`` and still have the reasoning land in their log.

Usage::

    python3 scripts/slurm/pick_gpu.py                  # one GPU, partition gpu
    python3 scripts/slurm/pick_gpu.py --need 3         # about to submit 3 jobs
    python3 scripts/slurm/pick_gpu.py --explain        # per-type free/total table

Environment:

``VTS_GPU``
    Explicit type (``a100``). Set it and no query happens -- this is the escape
    hatch when you know better than the heuristic.
``VTS_PART``
    Partition to look at. Default ``gpu``.
``VTS_GPU_TYPES``
    Space- or comma-separated candidates, **fastest first**. Default
    ``a100 l40s v100``. Only the last place is measured: V100 is the slow one.
    The A100/L40S order is a judgement call and rarely binds, since we only fall
    past the first candidate when it has nothing free. ``h100``/``h200`` are
    deliberately absent -- the HLTCOE ``4gpu_tier`` QOS caps them at 0, so
    requesting one pends forever (GRID-PLAYBOOK section 2). Set this to match
    your own QOS.
``VTS_GPU_FALLBACK``
    Type to name when nothing can be learned. Default ``l40s``.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess  # noqa: S404 -- runs a fixed `scontrol` argv, no shell, no user input
import sys
from dataclasses import dataclass

DEFAULT_TYPES: tuple[str, ...] = ("a100", "l40s", "v100")
DEFAULT_FALLBACK = "l40s"
DEFAULT_PARTITION = "gpu"

# A node in any of these states cannot take a new job (or is being held for one
# that is not ours), so its GPUs are not "free" no matter what GresUsed says.
# Matched case-insensitively as substrings of scontrol's `State=` field, which
# combines flags with `+` (e.g. `IDLE+DRAIN`, `DOWN+NOT_RESPONDING`).
_UNUSABLE_STATE = re.compile(
    r"DOWN|DRAIN|DRNG|FAIL|MAINT|INVAL|UNKNOWN|NO_RESPOND|NOT_RESPONDING|POWER|RESERVED|PLANNED",
    re.IGNORECASE,
)

# `gpu:l40s:8`, and the decorated forms scontrol emits: `gpu:l40s:8(S:0-1)` in
# Gres and `gpu:l40s:3(IDX:0-2)` in GresUsed. An untyped `gpu:8` deliberately
# does not match -- this cluster cannot be asked for one, so it is not a
# candidate even when it is idle.
_GRES_RE = re.compile(r"gpu:([A-Za-z0-9_.-]+):(\d+)")

# The same counts as they appear in `AllocTRES=`: `gres/gpu:a100=4`. The untyped
# `gres/gpu=4` alongside it is a total over types and deliberately does not
# match, or every allocated GPU would be counted twice.
_ALLOC_GPU_RE = re.compile(r"gres/gpu:([A-Za-z0-9_.-]+)=(\d+)")


@dataclass(frozen=True)
class Availability:
    """Free/total GPU counts for one type, summed over usable nodes."""

    gpu_type: str
    free: int
    total: int


def _field_opt(line: str, key: str) -> str | None:
    """``key=value`` from one node record, or None if the key is absent.

    The distinction matters: ``AllocTRES=`` is written *empty* on a node with
    nothing allocated, which is a measurement ("zero used"), while a key that
    does not appear at all is the absence of a measurement. Collapsing the two
    is what let :func:`_node_gpus_used` read "no usage field" as "nothing is
    used" and report a fully-booked cluster as fully free (#3299).

    Regex rather than splitting on whitespace because a few values on that line
    (``Reason=``, ``OS=``) contain spaces, which would shift every field after
    them. Anchoring on a word boundary keeps ``Gres=`` from matching inside
    ``GresUsed=``.
    """
    match = re.search(rf"(?:^|\s){re.escape(key)}=(\S*)", line)
    return match.group(1) if match else None


def _field(line: str, key: str) -> str:
    """``key=value``, with an absent key flattened to the empty string."""
    return _field_opt(line, key) or ""


def _gpu_counts(gres: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for gpu_type, count in _GRES_RE.findall(gres):
        counts[gpu_type] = counts.get(gpu_type, 0) + int(count)
    return counts


def _node_gpus_used(line: str) -> dict[str, int] | None:
    """How many GPUs of each type this node has allocated, or None if unknowable.

    Two fields can answer it, and which one exists depends on the Slurm build:

    * ``GresUsed=gpu:a100:4(IDX:0-3)`` -- the direct answer, when present.
    * ``AllocTRES=cpu=32,mem=384G,gres/gpu=4,gres/gpu:a100=4`` -- the same
      number by another route. Written **empty** on a node with nothing
      allocated, which is still an answer: zero.

    The cluster this runs on (Slurm 23.11.6) emits ``GresUsed`` on *no* node in
    ``scontrol show node --oneliner``, so the original reader -- which looked
    only there and treated a missing field as an empty string -- computed
    ``free == total`` for every node on the cluster and always returned the
    first candidate type. It reported "a100 23/23 free" against 23 allocated
    A100s and 109 idle V100s, and the resulting job was told it would start in
    24 hours (#3299). Every test fixture wrote a ``GresUsed`` field, so nothing
    caught it: the fixtures described a cluster the code never met.

    Returning None rather than ``{}`` when neither field is present is the
    point of the repair. "I cannot measure usage" must not be spellable as
    "nothing is used", because those two produce opposite launches.
    """
    used = _field_opt(line, "GresUsed")
    if used is not None:
        return _gpu_counts(used)
    alloc = _field_opt(line, "AllocTRES")
    if alloc is not None:
        return {t: int(n) for t, n in _ALLOC_GPU_RE.findall(alloc)}
    return None


def parse_node_availability(scontrol_output: str, partition: str = DEFAULT_PARTITION) -> dict[str, Availability]:
    """Sum free/total GPUs per type over the usable nodes of ``partition``.

    ``scontrol_output`` is ``scontrol show node --oneliner`` (one node per line).
    Nodes outside the partition, in an unusable state, or whose GPU usage
    cannot be read at all contribute nothing -- not even to ``total`` -- so a
    drained node full of idle A100s never advertises itself as the shortest
    queue, and neither does a node this parser cannot understand.
    """
    free: dict[str, int] = {}
    total: dict[str, int] = {}

    for line in scontrol_output.splitlines():
        line = line.strip()
        if not line.startswith("NodeName="):
            continue
        partitions = _field(line, "Partitions")
        if partition not in partitions.split(","):
            continue
        if _UNUSABLE_STATE.search(_field(line, "State")):
            continue

        node_total = _gpu_counts(_field(line, "Gres"))
        node_used = _node_gpus_used(line)
        if node_used is None:
            # Unmeasurable: contribute nothing at all, exactly as a drained node
            # does. A node whose usage cannot be read is not evidence of a free
            # GPU, and pretending otherwise is the #3299 failure.
            continue
        for gpu_type, count in node_total.items():
            total[gpu_type] = total.get(gpu_type, 0) + count
            # A node can report more used than configured only if the two fields
            # disagree mid-update; clamp so a transient never yields a negative.
            free[gpu_type] = free.get(gpu_type, 0) + max(0, count - node_used.get(gpu_type, 0))

    return {t: Availability(t, free.get(t, 0), n) for t, n in total.items()}


def select_gpu_type(
    availability: dict[str, Availability],
    types: tuple[str, ...] = DEFAULT_TYPES,
    need: int = 1,
    fallback: str = DEFAULT_FALLBACK,
) -> tuple[str, str]:
    """Choose a GPU type. Returns ``(type, human-readable reason)``.

    ``types`` is ordered fastest-first and is the *only* thing that expresses
    speed; ``availability`` only knows about counts.
    """
    known = [(t, availability[t]) for t in types if t in availability]

    for gpu_type, avail in known:
        if avail.free >= need:
            return gpu_type, f"{avail.free}/{avail.total} free -- fastest type with {need} free to start on now"

    startable = [(t, a) for t, a in known if a.free > 0]
    if startable:
        gpu_type, avail = max(startable, key=lambda pair: pair[1].free)
        return gpu_type, f"no type has {need} free; {gpu_type} has the most ({avail.free}/{avail.total})"

    if known:
        gpu_type, avail = max(known, key=lambda pair: pair[1].total)
        return gpu_type, f"nothing free; {gpu_type} has the largest pool ({avail.total}) so it should drain soonest"

    return fallback, "no candidate type visible in this partition; using the fallback"


def _query_scontrol() -> str | None:
    """Return ``scontrol show node --oneliner`` output, or None if unavailable."""
    try:
        proc = subprocess.run(  # noqa: S603 -- fixed argv, shell=False
            ["scontrol", "show", "node", "--oneliner"],  # noqa: S607 -- resolved via PATH like every other slurm call here
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def _split_types(raw: str) -> tuple[str, ...]:
    return tuple(t for t in re.split(r"[,\s]+", raw.strip()) if t)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--need",
        type=int,
        default=1,
        help="how many GPUs of one type are about to be requested (default 1); a type with fewer free is skipped",
    )
    parser.add_argument("--partition", default=os.environ.get("VTS_PART", DEFAULT_PARTITION), help="SLURM partition")
    parser.add_argument("--types", default=os.environ.get("VTS_GPU_TYPES", ""), help="candidate types, fastest first")
    parser.add_argument("--explain", action="store_true", help="print the per-type free/total table to stderr")
    args = parser.parse_args(argv)

    explicit = os.environ.get("VTS_GPU", "").strip()
    if explicit:
        print(explicit)
        print(f"pick_gpu: {explicit} (VTS_GPU is set; not querying the scheduler)", file=sys.stderr)
        return 0

    types = _split_types(args.types) if args.types else DEFAULT_TYPES
    fallback = os.environ.get("VTS_GPU_FALLBACK", "").strip() or DEFAULT_FALLBACK

    output = _query_scontrol()
    if output is None:
        print(fallback)
        print(f"pick_gpu: {fallback} (scontrol unavailable; using the fallback)", file=sys.stderr)
        return 0

    availability = parse_node_availability(output, args.partition)
    if args.explain:
        for gpu_type in sorted(availability, key=lambda t: (types.index(t) if t in types else len(types), t)):
            avail = availability[gpu_type]
            mark = "" if gpu_type in types else "   (not a candidate)"
            print(
                f"pick_gpu:   {avail.gpu_type:<8} {avail.free:>3} free / {avail.total:>3} total{mark}", file=sys.stderr
            )

    gpu_type, reason = select_gpu_type(availability, types, need=max(1, args.need), fallback=fallback)
    print(gpu_type)
    print(f"pick_gpu: {gpu_type} -- {reason}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
