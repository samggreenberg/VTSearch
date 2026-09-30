#!/usr/bin/env python3
"""Docs ⇄ screenshot-manifest wiring check (no browser needed).

Asserts three invariants from docs/plans/user-docs-screenshots.md:

  (a) every shot id in docs/user/screenshots.manifest.ts has BOTH theme files
      (`<id>.light.webp` and `<id>.dark.webp`) on disk under docs/user/assets/;
  (b) every screenshot the user-facing docs embed (USER_GUIDE.md, the how-to
      pages under docs/user/howto/, README.md, demos.md) — i.e. each
      `assets/<id>.<theme>.webp` reference — resolves to a real manifest id;
  (c) every shot named by an entry in the reshoot queue (docs/reshoot-queue/,
      one file per change) resolves to a real manifest id, or, spelled
      `slides:<group>`, to a group slides/figs/src/shoot-ui-figs.mjs takes on
      its command line, so the queue can't reference a renamed or deleted shot.

This catches docs/manifest drift without rendering anything, so it is cheap
enough to gate in run-tests.sh. It does NOT render or diff pixels (that is
check.sh, which needs chromium and stays a manual chore).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs" / "user" / "screenshots.manifest.ts"
ASSETS = ROOT / "docs" / "user" / "assets"
RESHOOT_QUEUE = ROOT / "docs" / "reshoot-queue"
SLIDE_SHOOTER = ROOT / "slides" / "figs" / "src" / "shoot-ui-figs.mjs"
SLIDE_PREFIX = "slides:"
THEMES = ("light", "dark")
HOWTO = ROOT / "docs" / "user" / "howto"
DOCS = [
    ROOT / "docs" / "user" / "USER_GUIDE.md",
    *sorted(HOWTO.glob("*.md")),
    ROOT / "README.md",
    ROOT / "docs" / "demos.md",
]

# `id: 'kebab-case'` inside a SHOTS entry. The Shot interface uses
# `id: string;` (no quotes), so it is not matched.
ID_RE = re.compile(r"^\s*id:\s*'([a-z0-9-]+)'", re.MULTILINE)
# Any embedded asset reference, e.g. assets/dashboard-loaded.dark.webp (a
# how-to page writes it ../assets/…, which this matches too)
REF_RE = re.compile(r"assets/([a-z0-9-]+)\.(light|dark)\.webp")
# A reshoot-queue entry's bullet: the shot id is the first backticked token on
# a line starting `- `. Any backticked token counts, so a malformed id is
# reported rather than skipped.
QUEUE_ITEM_RE = re.compile(r"^- [^`\n]*`([^`\n]+)`", re.MULTILINE)
# The slide shooter's groups: the intro session's list, plus every group it
# tests for by name.
SLIDE_INTRO_RE = re.compile(r"const INTRO = \[([^\]]*)\]")
SLIDE_WANTED_RE = re.compile(r"wanted\('([a-z0-9-]+)'\)")


def manifest_ids() -> list[str]:
    text = MANIFEST.read_text(encoding="utf-8")
    ids = ID_RE.findall(text)
    if not ids:
        sys.exit(f"wiring-check: no shot ids found in {MANIFEST}")
    return ids


def slide_groups() -> set[str]:
    text = SLIDE_SHOOTER.read_text(encoding="utf-8")
    groups = set(SLIDE_WANTED_RE.findall(text))
    intro = SLIDE_INTRO_RE.search(text)
    if intro:
        groups.update(re.findall(r"'([a-z0-9-]+)'", intro.group(1)))
    if not groups:
        sys.exit(f"wiring-check: no shot groups found in {SLIDE_SHOOTER}")
    return groups


def queue_entries(queue: Path = RESHOOT_QUEUE) -> dict[Path, list[str]]:
    """Each entry file in the reshoot queue, mapped to the shot ids it names.

    The directory's README.md documents the format, and its example is not an
    entry.
    """
    if not queue.is_dir():
        return {}
    return {
        entry: QUEUE_ITEM_RE.findall(entry.read_text(encoding="utf-8"))
        for entry in sorted(queue.glob("*.md"))
        if entry.name != "README.md"
    }


def queue_errors(entries: dict[Path, list[str]], shot_ids: set[str], groups: set[str]) -> list[str]:
    errors: list[str] = []
    for entry, queued in entries.items():
        name = entry.relative_to(ROOT) if entry.is_relative_to(ROOT) else entry
        if not queued:
            errors.append(f"{name} names no shot: add one '- `<id>` — why' bullet per shot")
        for queue_id in queued:
            if queue_id.startswith(SLIDE_PREFIX):
                if queue_id.removeprefix(SLIDE_PREFIX) not in groups:
                    errors.append(
                        f"{name} queues '{queue_id}', which names no shoot-ui-figs.mjs group"
                        f" (one of: {', '.join(sorted(groups))})"
                    )
            elif queue_id not in shot_ids:
                errors.append(f"{name} queues reshoot for '{queue_id}' with no matching manifest id")
    return errors


def main() -> int:
    ids = manifest_ids()
    id_set = set(ids)
    errors: list[str] = []

    # (a) every manifest id has both theme files on disk.
    for sid in ids:
        for theme in THEMES:
            path = ASSETS / f"{sid}.{theme}.webp"
            if not path.exists():
                errors.append(f"missing asset for manifest id '{sid}': {path.relative_to(ROOT)}")

    # (b) every embedded reference in the docs maps to a manifest id.
    for doc in DOCS:
        if not doc.exists():
            continue
        for match in REF_RE.finditer(doc.read_text(encoding="utf-8")):
            ref_id = match.group(1)
            if ref_id not in id_set:
                errors.append(f"{doc.relative_to(ROOT)} embeds 'assets/{ref_id}.*.webp' with no matching manifest id")

    # (c) every shot queued for reshoot maps to a manifest id or a slide group.
    entries = queue_entries()
    errors.extend(queue_errors(entries, id_set, slide_groups()))
    queued = sum(len(names) for names in entries.values())

    if errors:
        print("wiring-check FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print(
        f"wiring-check OK: {len(ids)} shots, both themes present, docs consistent,"
        f" {queued} reshoot(s) queued in {len(entries)} queue file(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
