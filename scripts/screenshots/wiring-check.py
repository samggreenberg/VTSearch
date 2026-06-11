#!/usr/bin/env python3
"""Docs ⇄ screenshot-manifest wiring check (no browser needed).

Asserts two invariants from docs/plans/user-docs-screenshots.md:

  (a) every shot id in docs/user/screenshots.manifest.ts has BOTH theme files
      (`<id>.light.png` and `<id>.dark.png`) on disk under docs/user/assets/;
  (b) every screenshot the user-facing docs embed (USER_GUIDE.md, README.md,
      demos.md) — i.e. each `assets/<id>.<theme>.png` reference — resolves to a
      real manifest id.

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
THEMES = ("light", "dark")
DOCS = [
    ROOT / "docs" / "user" / "USER_GUIDE.md",
    ROOT / "README.md",
    ROOT / "docs" / "demos.md",
]

# `id: 'kebab-case'` inside a SHOTS entry. The Shot interface uses
# `id: string;` (no quotes), so it is not matched.
ID_RE = re.compile(r"^\s*id:\s*'([a-z0-9-]+)'", re.MULTILINE)
# Any embedded asset reference, e.g. assets/dashboard-loaded.dark.png
REF_RE = re.compile(r"assets/([a-z0-9-]+)\.(light|dark)\.png")


def manifest_ids() -> list[str]:
    text = MANIFEST.read_text(encoding="utf-8")
    ids = ID_RE.findall(text)
    if not ids:
        sys.exit(f"wiring-check: no shot ids found in {MANIFEST}")
    return ids


def main() -> int:
    ids = manifest_ids()
    id_set = set(ids)
    errors: list[str] = []

    # (a) every manifest id has both theme files on disk.
    for sid in ids:
        for theme in THEMES:
            png = ASSETS / f"{sid}.{theme}.png"
            if not png.exists():
                errors.append(f"missing asset for manifest id '{sid}': {png.relative_to(ROOT)}")

    # (b) every embedded reference in the docs maps to a manifest id.
    for doc in DOCS:
        if not doc.exists():
            continue
        for match in REF_RE.finditer(doc.read_text(encoding="utf-8")):
            ref_id = match.group(1)
            if ref_id not in id_set:
                errors.append(f"{doc.relative_to(ROOT)} embeds 'assets/{ref_id}.*.png' with no matching manifest id")

    if errors:
        print("wiring-check FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print(f"wiring-check OK: {len(ids)} shots, both themes present, docs consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
