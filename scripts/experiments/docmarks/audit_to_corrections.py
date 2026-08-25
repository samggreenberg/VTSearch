#!/usr/bin/env python
"""Fold filled-in audit verdicts back into the corpus.

    python audit_to_corrections.py --task cluster --apply
    python audit_to_corrections.py --task distinctive --apply
    python audit_to_corrections.py --task letterhead          # dry run (default)

Without ``--apply`` it prints what it would change and touches nothing.

Verdicts are additive and idempotent: they are recorded in ``classes.json``
under each class's ``audit`` block, and re-running with the same verdict file is
a no-op.  Nothing is ever deleted — a class judged ``generic`` keeps all its
instances and simply stops being part of the headline stratum, so both numbers
stay available and the decision stays visible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import docmarks_config as cfg  # noqa: E402
from sources._common import Mark, Page, read_manifest, write_manifest  # noqa: E402


def load_verdicts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"no verdict file at {path} — run make_audit_slate.py first")
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                if str(row.get("verdict", "")).strip():
                    out.append(row)
    return out


def apply_cluster(
    pages: list[Page], classes: dict[str, Any], verdicts: list[dict[str, Any]]
) -> tuple[list[str], list[str]]:
    """``ok`` / ``split`` / ``merge_into:<id>`` / ``drop`` on derived classes."""
    changes: list[str] = []
    problems: list[str] = []

    for row in verdicts:
        class_id = row["class_id"]
        verdict = str(row["verdict"]).strip()
        meta = classes.get(class_id)
        if meta is None:
            problems.append(f"{class_id}: not in classes.json")
            continue

        if verdict == "ok":
            meta["audit"]["cluster_ok"] = True
            changes.append(f"{class_id}: confirmed")
        elif verdict == "split":
            # A split needs per-instance labels, which a contact sheet cannot
            # express.  Mark it unusable rather than guessing a partition: a
            # silently wrong split is worse than a missing class.
            meta["audit"]["cluster_ok"] = False
            meta["audit"]["notes"] = (row.get("notes") or "flagged for split; excluded pending re-cluster").strip()
            changes.append(f"{class_id}: flagged as over-merged (excluded)")
        elif verdict.startswith("merge_into:"):
            target = verdict.split(":", 1)[1].strip()
            if target not in classes:
                problems.append(f"{class_id}: merge target {target!r} does not exist")
                continue
            for page in pages:
                for i, mark in enumerate(page.marks):
                    if mark.class_id == class_id:
                        page.marks[i] = Mark(mark.kind, mark.box, target, mark.provenance)
            classes[target]["n_instances"] += meta["n_instances"]
            classes[target]["page_ids"] = sorted(set(classes[target]["page_ids"]) | set(meta["page_ids"]))
            classes[target]["audit"]["cluster_ok"] = True
            classes.pop(class_id)
            changes.append(f"{class_id}: merged into {target}")
        elif verdict == "drop":
            for page in pages:
                page.marks = [m for m in page.marks if m.class_id != class_id]
            classes.pop(class_id)
            changes.append(f"{class_id}: dropped")
        else:
            problems.append(f"{class_id}: unrecognised verdict {verdict!r}")

    return changes, problems


def apply_distinctive(classes: dict[str, Any], verdicts: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    changes: list[str] = []
    problems: list[str] = []
    for row in verdicts:
        class_id = row["class_id"]
        verdict = str(row["verdict"]).strip().lower()
        meta = classes.get(class_id)
        if meta is None:
            problems.append(f"{class_id}: not in classes.json")
            continue
        if verdict not in ("distinctive", "generic"):
            problems.append(f"{class_id}: unrecognised verdict {verdict!r}")
            continue
        meta["audit"]["distinctive"] = verdict == "distinctive"
        changes.append(f"{class_id}: {verdict}")
    return changes, problems


def apply_letterhead(classes: dict[str, Any], verdicts: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    changes: list[str] = []
    problems: list[str] = []
    for row in verdicts:
        class_id = row["class_id"]
        meta = classes.get(class_id)
        if meta is None:
            problems.append(f"{class_id}: not in classes.json")
            continue
        try:
            hits = int(str(row["verdict"]).strip())
        except ValueError:
            problems.append(f"{class_id}: verdict must be the count of pages carrying the mark")
            continue
        sampled = int(row.get("sampled", 0)) or 1
        precision = hits / sampled
        meta["audit"]["letterhead_precision"] = round(precision, 3)
        meta["audit"]["letterhead_sampled"] = sampled
        flag = "" if precision >= 0.8 else "  <- below 0.8, treat this class as noisy"
        changes.append(f"{class_id}: label precision {precision:.2f} ({hits}/{sampled}){flag}")
    return changes, problems


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=("cluster", "distinctive", "letterhead"))
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--apply", action="store_true", help="write the changes (default is a dry run)")
    args = ap.parse_args(argv)

    classes_path = args.corpus / "classes.json"
    manifest_path = args.corpus / "corpus.jsonl"
    classes = json.loads(classes_path.read_text(encoding="utf-8"))
    verdicts = load_verdicts(args.corpus / "audit" / args.task / "verdicts.jsonl")

    pages = list(read_manifest(manifest_path)) if args.task == "cluster" else []

    if args.task == "cluster":
        changes, problems = apply_cluster(pages, classes, verdicts)
    elif args.task == "distinctive":
        changes, problems = apply_distinctive(classes, verdicts)
    else:
        changes, problems = apply_letterhead(classes, verdicts)

    for c in changes:
        print(f"  {c}")
    for p in problems:
        print(f"  PROBLEM: {p}")
    print(f"\n{len(changes)} change(s), {len(problems)} problem(s) from {len(verdicts)} filled verdict(s)")

    if not args.apply:
        print("dry run — pass --apply to write")
        return 1 if problems else 0

    classes_path.write_text(json.dumps(classes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.task == "cluster":
        write_manifest(pages, manifest_path)
    print(f"wrote {classes_path}" + (f" and {manifest_path}" if args.task == "cluster" else ""))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
