"""Several ``sota_documents.py`` arms in one process, over one load of the tier (#4934).

A tier-``m`` run holds ~120 GB, nearly all of it the tier's features and tiles, and the GRID's
per-user memory cap fits only a few. So one process loads the tier once and runs every
``(arm, replicate, class)`` as its own ``sota_documents.main`` call, sharing the loaded pages as
the app shares a dataset across detectors. The calls interleave arm by arm within each class, so a
job that dies leaves every arm on the same classes, and a re-run skips every call that finished.

Each call writes ``<out>/<arm>/rep<r>/<class>/`` (``steps.csv`` and the rest) and a ``done``
marker. Replicate 2 is ``--swap-halves``. An arm is ``name=flags``, its flags for
``sota_documents.py``:

    python sota_documents_arms.py --out <dir> --classes a,b --reps 1,2 \\
        --arm all= --arm strong2048="--keep-kp 2048 --kp-order strength --strength <file>" \\
        -- --tier m --max-v 50 --matrix <votes-4162>/matrix-m --feature-cache <dir>
"""

from __future__ import annotations

import argparse
import shlex
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sota_documents as sd  # noqa: E402

_MEMO: dict[Any, Any] = {}


def _memoized(name: str) -> None:
    """Make ``sd.<name>`` return its first result for the same tier on every later call."""
    fn = getattr(sd, name)

    def once(ids: list[str], *args: Any) -> Any:
        key = (name, len(ids), hash(tuple(ids)), *(str(a) for a in args[-2:]))
        if key not in _MEMO:
            _MEMO[key] = fn(ids, *args)
        return _MEMO[key]

    setattr(sd, name, once)


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    passthrough: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, passthrough = argv[:cut], argv[cut + 1 :]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--classes", required=True, help="comma-separated class ids this process runs")
    ap.add_argument("--reps", default="1,2", help="replicates: 1 clicks the click half, 2 swaps the halves")
    ap.add_argument("--arm", action="append", required=True, help="name=flags (flags for sota_documents.py)")
    args = ap.parse_args(argv)
    if any(a in passthrough for a in ("--out", "--classes", "--swap-halves")):
        ap.error("--out, --classes and --swap-halves are set per call; leave them out after --")

    arms = []
    for spec in args.arm:
        name, _, flags = spec.partition("=")
        if not name or "/" in name:
            ap.error(f"bad arm {spec!r}")
        arms.append((name, shlex.split(flags)))
    reps = [int(r) for r in args.reps.split(",") if r]
    classes = [c for c in args.classes.split(",") if c]
    for name in ("load_or_extract", "_cached_tiles"):
        _memoized(name)

    t0 = time.time()
    for cid in classes:
        slug = cid.replace("/", "__", 1)
        for rep in reps:
            for arm, flags in arms:
                out = args.out / arm / f"rep{rep}" / slug
                if (out / "done").exists():
                    continue
                print(f"== {arm} rep{rep} {cid} ({time.time() - t0:.0f}s)", flush=True)
                swap = ["--swap-halves"] if rep == 2 else []
                rc = sd.main([*passthrough, *flags, *swap, "--classes", cid, "--out", str(out)])
                if rc:
                    return rc
                (out / "done").write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n", encoding="utf-8")
    sd.restore_verify_many()
    print(f"all {len(classes)} classes x {len(reps)} replicates x {len(arms)} arms done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
