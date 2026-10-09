"""``docs/hints.md`` must chart exactly the hints the frontend gives.

The chart is the reference for ordering hint changes, so a row that names a
hint the app no longer has, misses one it does, or shows the wrong Toasty is
worse than no chart.  Three sources have to agree:

* ``HINT_IDS`` in ``hints.service.ts`` -- every hint id, in order;
* the ``<vt-toasty-hint>`` tags in the templates -- one per id, each with a
  static ``hintId`` and the ``face`` it shows (``happy`` when unset);
* the chart's rows -- the same ids in the same order, with the same face,
  named in words and pictured by that face's file in ``TOASTY_FACES``.

One tag per id is the rule the chart rests on: an id is one bubble with one
text, so a hint that says something else in another state is a second hint.

Stdlib only, so it runs in the library tier.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
APP = REPO / "frontend" / "src" / "app"
HINTS_SERVICE = APP / "services" / "hints.service.ts"
TOASTY_FACES = APP / "utils" / "toasty-faces.ts"
CHART = REPO / "docs" / "hints.md"

_HINT_IDS_RE = re.compile(r"export const HINT_IDS = \[(.*?)\] as const;", re.DOTALL)
_TAG_RE = re.compile(r"<vt-toasty-hint\b([^>]*)>")
_ROW_RE = re.compile(r"^\| `([a-z-]+)` \| ([^|]*) \|", re.MULTILINE)
_FACE_RE = re.compile(r"\b(happy|sad|surprised)\b")
_IMG_RE = re.compile(r'<img src="\.\./frontend/public/([^"]+)"')


def _hint_ids() -> list[str]:
    match = _HINT_IDS_RE.search(HINTS_SERVICE.read_text())
    assert match, f"no HINT_IDS array in {HINTS_SERVICE.relative_to(REPO)}"
    return re.findall(r"'([a-z-]+)'", match.group(1))


def _template_faces() -> dict[str, list[str]]:
    """Each hint id to the face of every tag that renders it."""
    faces: dict[str, list[str]] = {}
    for html in sorted(APP.rglob("*.html")):
        for attrs in _TAG_RE.findall(html.read_text()):
            hint_id = re.search(r'(?<![\w\[])hintId="([^"]+)"', attrs)
            assert hint_id, f'{html.relative_to(REPO)}: a <vt-toasty-hint> without a static hintId="..."'
            face = re.search(r'(?<![\w\[])face="([^"]+)"', attrs)
            faces.setdefault(hint_id.group(1), []).append(face.group(1) if face else "happy")
    return faces


def _face_files() -> dict[str, str]:
    match = re.search(r"TOASTY_FACES\b.*?\{(.*?)\}", TOASTY_FACES.read_text(), re.DOTALL)
    assert match, f"no TOASTY_FACES map in {TOASTY_FACES.relative_to(REPO)}"
    return dict(re.findall(r"(\w+): '([^']+)'", match.group(1)))


def _chart_rows() -> list[tuple[str, str]]:
    """Each row's hint id and the face its Toasty cell names in words."""
    files = _face_files()
    rows = []
    for hint_id, toasty in _ROW_RE.findall(CHART.read_text()):
        # The word, not the picture's filename (``toasty-surprised.png``).
        face = _FACE_RE.search(re.sub(r"<[^>]*>", "", toasty))
        assert face, f"docs/hints.md: the {hint_id!r} row names no Toasty face"
        pictured = _IMG_RE.findall(toasty)
        assert pictured == [files[face.group(1)]], (
            f"docs/hints.md: the {hint_id!r} row says {face.group(1)} but pictures {pictured}"
        )
        rows.append((hint_id, face.group(1)))
    return rows


def test_every_hint_id_is_one_tag() -> None:
    ids = _hint_ids()
    faces = _template_faces()
    assert sorted(faces) == sorted(ids), "the <vt-toasty-hint> tags and HINT_IDS name different hints"
    shared = {hint_id: len(tags) for hint_id, tags in faces.items() if len(tags) > 1}
    assert not shared, f"one tag per hint id; give each text its own id: {shared}"


def test_chart_lists_every_hint_in_order() -> None:
    assert [hint_id for hint_id, _ in _chart_rows()] == _hint_ids()


def test_chart_shows_each_hints_toasty() -> None:
    faces = {hint_id: tags[0] for hint_id, tags in _template_faces().items()}
    assert dict(_chart_rows()) == faces
