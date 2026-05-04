"""Regression test: dashboard column resize handle stays centered on the divider.

This bug has shipped twice already. The two failure modes look almost identical
to a casual eye, so a fix that targets one symptom keeps re-introducing the
other:

1. ``right: 5px; width: 5px``  - grab zone entirely on the LEFT of the divider.
2. ``right: -6px; width: 12px`` - grab zone straddles the divider, but the next
   ``<th>`` (``position: sticky; z-index: 1``) creates a stacking context that
   paints over the right half. Effective grab zone is left-only again.
3. ``right: 0; width: 12px``    - same as (1): grab zone entirely on the LEFT.

The structurally correct fix hosts the handle on the LEFT edge of the cell
AFTER the divider (so the previous cell's stacking context cannot cover it)
and offsets it by ``-6px`` so the 12px grab zone is symmetric across the
divider, with the 2px visual indicator centered on top.

If you are tempted to "tidy" the SCSS to remove the negative offset: don't.
Read the comment in ``dashboard.component.scss`` first.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCSS_FILES = [
    REPO / "frontend/src/app/components/dashboard/dashboard.component.scss",
    REPO
    / "frontend/src/app/components/dashboard/dataset-importer-modal/dataset-importer-modal.component.scss",
]


def _extract_handle_block(text: str) -> str:
    """Return the body of the ``.col-resize-handle { ... }`` rule (no nesting)."""
    match = re.search(r"\.col-resize-handle\s*\{", text)
    assert match, "Could not find .col-resize-handle rule"
    start = match.end()
    depth = 1
    i = start
    while i < len(text) and depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start : i - 1]


def _extract_after_block(handle_block: str) -> str:
    """Return the body of the nested ``&::after { ... }`` rule."""
    match = re.search(r"&::after\s*\{", handle_block)
    assert match, "Could not find &::after rule inside .col-resize-handle"
    start = match.end()
    depth = 1
    i = start
    while i < len(handle_block) and depth:
        if handle_block[i] == "{":
            depth += 1
        elif handle_block[i] == "}":
            depth -= 1
        i += 1
    return handle_block[start : i - 1]


def _decl(block: str, prop: str) -> str | None:
    """Return the value for ``prop:`` in a CSS block, or None if absent."""
    match = re.search(rf"(?<![\w-]){re.escape(prop)}\s*:\s*([^;]+);", block)
    return match.group(1).strip() if match else None


@pytest.mark.parametrize("scss_path", SCSS_FILES, ids=lambda p: p.name)
class TestResizeHandleCentering:
    def test_handle_grab_zone_straddles_divider(self, scss_path: Path) -> None:
        """The 12px grab zone must extend equally to both sides of the divider."""
        block = _extract_handle_block(scss_path.read_text())
        # Hosted on the LEFT of the cell after the divider, offset -6px so the
        # 12px width is symmetric across the divider line at the cell's left edge.
        assert _decl(block, "left") == "-6px", (
            f"{scss_path.name}: .col-resize-handle must be `left: -6px` so the "
            "grab zone straddles the divider. If you changed this, re-read the "
            "comment above the rule — `left: 0`/`right: 0` puts the grab zone "
            "entirely on one side, and `right: -6px` is covered by the next "
            "<th>'s stacking context."
        )
        assert _decl(block, "width") == "12px", (
            f"{scss_path.name}: .col-resize-handle must be `width: 12px`."
        )
        # The handle must NOT also pin to the right edge — that would shrink or
        # mis-center the grab zone.
        assert _decl(block, "right") is None, (
            f"{scss_path.name}: .col-resize-handle should not set `right`; "
            "use only `left: -6px` + `width: 12px` to keep the grab zone centered."
        )

    def test_visual_line_centered_on_divider(self, scss_path: Path) -> None:
        """The 2px indicator line must sit on the divider, not flush left/right."""
        block = _extract_handle_block(scss_path.read_text())
        after = _extract_after_block(block)
        # Handle starts at cell-x = -6, line at handle-x = 5 = cell-x = -1, width 2
        # → line spans cell-x [-1, +1], centered on the divider at cell-x = 0.
        assert _decl(after, "left") == "5px", (
            f"{scss_path.name}: .col-resize-handle::after must be `left: 5px` "
            "so the 2px visual line is centered on the divider (handle starts "
            "at cell-x -6, so handle-x 5 = divider)."
        )
        assert _decl(after, "width") == "2px", (
            f"{scss_path.name}: .col-resize-handle::after must be `width: 2px`."
        )
        assert _decl(after, "right") is None, (
            f"{scss_path.name}: .col-resize-handle::after should not set "
            "`right`; pinning to the right edge of the host cell sits the line "
            "one cell-width away from the divider."
        )
