#!/usr/bin/env python3
"""#4183: rename "COCO Quarry" to "COCO Better" and "DocMarks" to "FullMarks" -- files, then text.

    python scripts/experiments/pile/rename_4183.py            # dry run: what would change
    python scripts/experiments/pile/rename_4183.py --apply    # git mv + rewrite, in this checkout

Idempotent, so it can be re-run after a branch that still says "quarry" merges.
Repo text only; the on-disk pile is ``migrate_4183.py``'s job. Rules, in
order (the owner's ruling on #4183: ``coco_better`` in code, "COCO Better" in
prose, and every reference renamed, past reports included):

* identifiers keep their spelling style: ``coco_quarry`` -> ``coco_better``,
  ``COCO_QUARRY`` -> ``COCO_BETTER``, ``coco-quarry`` -> ``coco-better``;
* prose in any capitalisation -> "COCO Better";
* the bare word "quarry", which only ever meant this benchmark, -> its
  ``coco_better`` / "COCO Better" form;
* DocMarks -> FullMarks (owner, 2026-09-25): ``docmarks`` -> ``fullmarks``,
  ``DOCMARKS`` -> ``FULLMARKS``, "DocMarks" -> "FullMarks".

A line containing ``rename: keep`` is left alone: the legacy-name maps that let
old results load must keep spelling the old names.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

#: Ordered: the specific spellings first, so the bare-word rules only see leftovers.
RULES: list[tuple[str, str]] = [
    (r"coco_quarry", "coco_better"),
    (r"COCO_QUARRY", "COCO_BETTER"),
    (r"coco-quarry", "coco-better"),
    (r"COCO_Quarry", "COCO_Better"),
    (r"docmarks", "fullmarks"),
    (r"(?i:coco)[ -](?i:quarry)", "COCO Better"),
    # Bare "quarry" identifiers (slide figures, the export tool, the fragment).
    (r"data-set-quarry", "data-set-coco-better"),
    (r"quarry_export", "coco_better_export"),
    (r"QUARRY_", "COCO_BETTER_"),
    (r"CocoQuarry", "CocoBetter"),
    (r"_quarry_", "_coco_better_"),
    (r"\bquarry_", "coco_better_"),
    (r"_quarry\b", "_coco_better"),
    (r"DocMarks", "FullMarks"),
    (r"docmarks", "fullmarks"),
    (r"DOCMARKS", "FULLMARKS"),
    (r"Docmarks", "Fullmarks"),
    # A dict key or CLI choice is an identifier, not prose.
    (r'"quarry":', '"coco-better":'),
    # Bare prose.
    (r"\bThe quarry class\b", "The COCO Better class"),
    (r"\bthe quarry's\b", "COCO Better's"),
    (r"\bThe quarry's\b", "COCO Better's"),
    (r"\bthe quarry\b", "COCO Better"),
    (r"\bThe quarry\b", "COCO Better"),
    (r"\bquarry\b", "COCO Better"),
    (r"\bQuarry\b", "COCO Better"),
]
#: Files never rewritten: this script (its rules must keep the old spelling).
SKIP = {"scripts/experiments/pile/rename_4183.py", "scripts/experiments/pile/migrate_4183.py"}


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout  # noqa: S603, S607


def new_path(path: str) -> str:
    out = path
    for pat, rep in RULES[:13]:
        out = re.sub(pat, rep, out)
    return out


def rewrite(text: str) -> str:
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if "rename: keep" in line:
            continue
        for pat, rep in RULES:
            line = re.sub(pat, rep, line)
        lines[i] = line
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    root = Path(_git("rev-parse", "--show-toplevel", cwd=Path.cwd()).strip())
    files = [f for f in _git("ls-files", cwd=root).splitlines() if f not in SKIP]

    moves = [(f, new_path(f)) for f in files if new_path(f) != f]
    for old, new in moves:
        print(f"mv  {old} -> {new}")
        if args.apply:
            (root / new).parent.mkdir(parents=True, exist_ok=True)
            _git("mv", old, new, cwd=root)
    moved = dict(moves)

    changed = 0
    for f in files:
        path = root / (moved.get(f, f) if args.apply else f)
        try:
            text = path.read_text()
        except (UnicodeDecodeError, IsADirectoryError, FileNotFoundError):
            continue
        new = rewrite(text)
        if new != text:
            changed += 1
            print(f"txt {moved.get(f, f)}: {sum(1 for _ in re.finditer('(?i)quarry', text))} mentions")
            if args.apply:
                path.write_text(new)
    print(f"{len(moves)} files moved, {changed} rewritten{'' if args.apply else ' (dry run)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
