"""Grow the FullMarks roster without a rebuild, on a staging copy of the corpus (#4143).

A roster change is a major version (v5.0): new classes, and so a new ground
truth for every study that reads the roster.  It is done on a **staging** copy
of the corpus's JSON stores -- the page images are shared, since
``corpus.jsonl`` names them by absolute path -- so the live corpus and its frozen
version stay valid and scorable until the new version is finished and swapped
in.

``stage``
    Copies the stores the audit tools read and write into a new directory.

``admit``
    Admits candidate classes the way ``build_corpus.py --roster`` would: the
    same :func:`build_corpus.admit_classes` over the current pages, the same
    :func:`build_corpus.write_query_crops` for their query crops.  The
    candidates' marks already carry their cluster ids in ``corpus.jsonl``, so
    nothing is re-clustered and no existing class moves.  The new classes join
    ``classes.json`` and ``roster.json``; a dry run by default.

Everything after admission -- identity, membership, completeness, query crops --
is the audit pipeline the existing classes went through, run against the
staging corpus with ``--corpus <staging>``.

    python roster_grow.py stage --out /expscratch/$USER/fullmarks/corpus-v5
    python roster_grow.py admit --corpus /expscratch/$USER/fullmarks/corpus-v5 --classes a,b,c [--apply]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fullmarks_config as cfg  # noqa: E402

#: The stores the builder replays and the audit tools read or write.
STORES = (
    "corpus.jsonl",
    "classes.json",
    "roster.json",
    "adjudications.json",
    "added_marks.json",
    "reviewed_negatives.json",
    "box_overrides.json",
    "query_crops.json",
)


def stage(src: Path, out: Path) -> list[str]:
    """Copy the stores and query crops of *src* into a new *out*; refuses an existing one."""
    if out.exists():
        raise SystemExit(f"{out} exists; a staging corpus is made once, from the live one")
    out.mkdir(parents=True)
    copied = []
    for name in STORES:
        if (src / name).exists():
            shutil.copy2(src / name, out / name)
            copied.append(name)
    shutil.copytree(src / "queries", out / "queries")
    (out / "audit").mkdir()
    (out / "STAGING.md").write_text(
        f"Staging copy of {src}, made by roster_grow.py stage.\n"
        "Page images are shared with the live corpus (corpus.jsonl paths are absolute).\n",
        encoding="utf-8",
    )
    return copied


def admit(corpus: Path, candidates: Sequence[str]) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    """``(new class metadata, needs_hand_crop, warnings)`` for *candidates*, written nowhere."""
    import build_corpus as bc  # noqa: PLC0415
    from sources._common import read_manifest  # noqa: PLC0415

    classes = json.loads((corpus / "classes.json").read_text(encoding="utf-8"))
    already = [c for c in candidates if c in classes]
    if already:
        raise SystemExit(f"already in classes.json: {already}")
    pages = list(read_manifest(corpus / "corpus.jsonl"))
    inventory = {cid: refs for cid, refs in bc.class_inventory(pages).items() if cid in set(candidates)}
    missing = sorted(set(candidates) - set(inventory))
    if missing:
        raise SystemExit(f"no marks carry these class ids (merged away?): {missing}")
    admitted, rejected = bc.admit_classes(
        pages, inventory, min_instances=None, min_mark_px=cfg.MIN_MARK_PX, roster=set(candidates)
    )
    if rejected:
        raise SystemExit(f"rejected by the builder: {rejected}")
    needs_hand_crop, warnings = bc.write_query_crops(
        pages, inventory, admitted, corpus / "queries", backend=cfg.CLUSTER_BACKEND
    )
    return admitted, needs_hand_crop, warnings


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage")
    s.add_argument("--src", type=Path, default=cfg.OUT)
    s.add_argument("--out", type=Path, required=True)
    a = sub.add_parser("admit")
    a.add_argument("--corpus", type=Path, required=True, help="the STAGING corpus; never the live one")
    a.add_argument("--classes", required=True, help="comma-separated candidate class ids")
    a.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "stage":
        copied = stage(args.src, args.out)
        print(f"staged {len(copied)} store(s) and queries/ -> {args.out}")
        return 0

    if args.corpus.resolve() == cfg.OUT.resolve() or not (args.corpus / "STAGING.md").exists():
        raise SystemExit("admit runs on a staging corpus (roster_grow.py stage), never on the live one")
    candidates = [c.strip() for c in args.classes.split(",") if c.strip()]
    admitted, needs_hand_crop, warnings = admit(args.corpus, candidates)
    for cid, meta in sorted(admitted.items()):
        print(
            f"  {cid}: {meta['n_instances']} instance(s), median {meta['median_mark_px']} px, "
            f"crop {meta.get('query_crop')}" + (f", caveats {meta['caveats']}" if meta["caveats"] else "")
        )
    for w in warnings:
        print(f"  WARNING: {w}")
    if needs_hand_crop:
        print(f"  needs a hand-drawn crop: {needs_hand_crop}")
    if not args.apply:
        print("dry run -- pass --apply to write classes.json and roster.json (query crops were cut already)")
        return 0
    import roster as _roster  # noqa: PLC0415

    classes = json.loads((args.corpus / "classes.json").read_text(encoding="utf-8"))
    classes.update(admitted)
    (args.corpus / "classes.json").write_text(json.dumps(classes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    current = _roster.load(args.corpus / "roster.json")
    current.classes = list(dict.fromkeys(current.classes + sorted(admitted)))
    _roster.save(current, args.corpus / "roster.json")
    print(f"admitted {len(admitted)} class(es): {len(classes)} in classes.json, {len(current.classes)} on the roster")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
