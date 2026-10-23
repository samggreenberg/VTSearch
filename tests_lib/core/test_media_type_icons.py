"""Every registered media type's ``icon`` names an SVG the frontend ships.

``MediaType.icon`` is an `IconComponent` type name, served by
``GET /api/media-types`` and drawn as ``<vt-icon [type]="…">``, which falls back
to the generic file glyph for a name it has no SVG for.  So a type whose icon
the frontend does not ship would not error anywhere: it would just look like a
file, the same quiet wrongness as the dashboard's Type column showing no icon
for Face rows while every other type had one (#5178).

The SVG tables are read as text, stdlib only, so the check stays in the
library tier.
"""

from __future__ import annotations

import re
from pathlib import Path

from vtscore.media import all_types

REPO = Path(__file__).resolve().parents[2]
ICON_DIR = REPO / "frontend" / "src" / "app" / "components" / "icon"
SVG_TABLES = ("icon-svgs.ts", "icon-svgs-extended.ts")

#: A table entry's key, alone on its line: ``  audio:`` or ``  'file-text':``.
_KEY = re.compile(r"^  '?([a-z][a-z0-9-]*)'?:$", re.MULTILINE)


def _shipped_icon_names() -> set[str]:
    names: set[str] = set()
    for table in SVG_TABLES:
        names |= set(_KEY.findall((ICON_DIR / table).read_text(encoding="utf-8")))
    return names


def test_the_svg_tables_parse() -> None:
    """Guard the parser: an empty or partial set would pass the check below vacuously."""
    shipped = _shipped_icon_names()
    assert {"file", "audio", "file-text", "face"} <= shipped


def test_every_media_type_icon_has_an_svg() -> None:
    shipped = _shipped_icon_names()
    missing = {mt.type_id: mt.icon for mt in all_types() if mt.icon not in shipped}
    assert not missing, f"media types whose icon the frontend has no SVG for: {missing}"
