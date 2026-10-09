"""``docs/hints.md`` must chart exactly the hints the frontend gives.

The chart is the reference for ordering hint changes, so a row that names a
hint the app no longer has, misses one it does, or shows the wrong Toasty or
the wrong words is worse than no chart.  Three sources have to agree:

* ``HINT_IDS`` in ``hints.service.ts`` -- every hint id, in order;
* the ``<vt-toasty-hint>`` tags in the templates -- one per id, each with a
  static ``hintId``, the ``face`` it shows (``happy`` when unset) and the
  text it says;
* the chart's rows -- the same ids in the same order, with the same face and
  the same text.

One tag per id is the rule the chart rests on: an id is one bubble with one
text, so a hint that says something else in another state is a second hint.

Text is compared after rendering the template's markup the way the chart
writes it: ``<strong>`` as ``**bold**``, entities as their characters, runs
of whitespace as one space, and every ``{{ expression }}`` and every chart
``{placeholder}`` as ``{}``, so the chart may name an interpolation however
reads best.

Stdlib only, so it runs in the library tier.
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import NamedTuple

REPO = Path(__file__).resolve().parents[2]
APP = REPO / "frontend" / "src" / "app"
HINTS_SERVICE = APP / "services" / "hints.service.ts"
CHART = REPO / "docs" / "hints.md"

_HINT_IDS_RE = re.compile(r"export const HINT_IDS = \[(.*?)\] as const;", re.DOTALL)
_TAG_RE = re.compile(r"<vt-toasty-hint\b([^>]*)>(.*?)</vt-toasty-hint>", re.DOTALL)
_ROW_RE = re.compile(r"^\| `([a-z-]+)` \| ([^|]*) \| ([^|]*) \| [^|]* \|$", re.MULTILINE)


class Hint(NamedTuple):
    face: str
    text: str


def _same_spacing(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _template_text(body: str, where: str) -> str:
    text = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)
    text = re.sub(r"<strong>(.*?)</strong>", r"**\1**", text, flags=re.DOTALL)
    text = re.sub(r"\{\{.*?\}\}", "{}", text, flags=re.DOTALL)
    assert not re.search(r"<\w|@(if|else|switch|for)\b", text), (
        f"{where}: markup or control flow inside a hint's text; branch outside the tag, one hint per text"
    )
    return _same_spacing(html.unescape(text))


def _hint_ids() -> list[str]:
    match = _HINT_IDS_RE.search(HINTS_SERVICE.read_text())
    assert match, f"no HINT_IDS array in {HINTS_SERVICE.relative_to(REPO)}"
    return re.findall(r"'([a-z-]+)'", match.group(1))


def _template_hints() -> dict[str, list[Hint]]:
    """Each hint id to every tag that renders it."""
    hints: dict[str, list[Hint]] = {}
    for path in sorted(APP.rglob("*.html")):
        where = str(path.relative_to(REPO))
        for attrs, body in _TAG_RE.findall(path.read_text()):
            hint_id = re.search(r'(?<![\w\[])hintId="([^"]+)"', attrs)
            assert hint_id, f'{where}: a <vt-toasty-hint> without a static hintId="..."'
            face = re.search(r'(?<![\w\[])face="([^"]+)"', attrs)
            hint = Hint(face.group(1) if face else "happy", _template_text(body, f"{where} ({hint_id.group(1)})"))
            hints.setdefault(hint_id.group(1), []).append(hint)
    return hints


def _chart_rows() -> list[tuple[str, Hint]]:
    rows = []
    for hint_id, face, text in _ROW_RE.findall(CHART.read_text()):
        rows.append((hint_id, Hint(face.strip(), _same_spacing(re.sub(r"\{[^}]*\}", "{}", text)))))
    return rows


def test_every_hint_id_is_one_tag() -> None:
    ids = _hint_ids()
    hints = _template_hints()
    assert sorted(hints) == sorted(ids), "the <vt-toasty-hint> tags and HINT_IDS name different hints"
    shared = {hint_id: len(tags) for hint_id, tags in hints.items() if len(tags) > 1}
    assert not shared, f"one tag per hint id; give each text its own id: {shared}"


def test_chart_lists_every_hint_in_order() -> None:
    assert [hint_id for hint_id, _ in _chart_rows()] == _hint_ids()


def test_chart_shows_each_hints_toasty_and_text() -> None:
    charted = dict(_chart_rows())
    for hint_id, (hint, *_) in _template_hints().items():
        assert charted.get(hint_id) == hint, f"docs/hints.md: the {hint_id!r} row differs from its template"
