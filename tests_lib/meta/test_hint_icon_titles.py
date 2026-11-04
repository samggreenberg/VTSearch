"""The text beside a (?) answers on hover too (issue #5536).

A ``<vt-field-hint-icon>`` is the (?) the SPA puts after a label or heading
to explain it at length. The Threshold heading had one, and nothing on the
word ``Threshold:`` itself: pointing at the word showed nothing, pointing six
pixels to its right showed a paragraph. The rule (``docs/style-guide.md``
§3.5) is that the text carries a short ``title`` of its own and the (?)
carries the long version, so every (?) in a template needs a titled
neighbour.

"The text" is found structurally, since a template can't say which words a
(?) explains:

- the element holding the (?), or any *inline* element between it and the
  nearest block (a ``<span>`` inside a ``<label>`` climbs to the label), or
  that block itself: a ``<label>``, a heading, a ``<p>``, a ``<div>`` row;
- or the element just before the (?) among its siblings, which is how a row
  that also holds other controls does it (the Threshold heading's
  ``.balance-label``): a ``title`` on the row would leak onto those
  controls, since a native tooltip is inherited.

Either a static ``title="…"`` or a bound ``[title]="…"`` counts. A ``vt-*``
component's ``title`` is an input (``<vt-modal title="…">`` names the
dialog), never a native tooltip, so it never counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FRONTEND_SRC = REPO / "frontend/src/app"

HINT_TAG = "vt-field-hint-icon"
HINT_COMPONENT = FRONTEND_SRC / "components/field-hint-icon/field-hint-icon.component.ts"

#: Elements that never hold children, so never go on the open-element stack.
VOID = frozenset(
    "area base br col embed hr img input link meta param source track wbr".split()
)

#: Elements the climb passes through on its way to the block holding the text.
INLINE = frozenset("span strong em b i small a code abbr".split())


@dataclass
class _Element:
    tag: str
    titled: bool
    #: Whether the last element child closed so far carried a title.
    last_child_titled: bool = False
    children: list[str] = field(default_factory=list)


def _has_native_title(tag: str, attrs: list[tuple[str, str | None]]) -> bool:
    if tag.startswith("vt-"):
        return False
    return any(name in ("title", "[title]", "[attr.title]") and value for name, value in attrs)


class _HintScanner(HTMLParser):
    """Record each (?) whose text carries no ``title``, by line."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[_Element] = [_Element("#root", titled=False)]
        self.untitled: list[int] = []
        self.hints = 0

    def _check_hint(self) -> None:
        self.hints += 1
        parent = self.stack[-1]
        if parent.last_child_titled:
            return
        for el in reversed(self.stack):
            if el.titled:
                return
            if el.tag not in INLINE:
                break
        self.untitled.append(self.getpos()[0])

    def _closed_child(self, titled: bool) -> None:
        self.stack[-1].last_child_titled = titled

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == HINT_TAG:
            self._check_hint()
        titled = _has_native_title(tag, attrs)
        if tag in VOID:
            self._closed_child(titled)
            return
        self.stack.append(_Element(tag, titled))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # Angular's self-closing ``<vt-field-hint-icon … />``.
        if tag == HINT_TAG:
            self._check_hint()
        self._closed_child(_has_native_title(tag, attrs))

    def handle_endtag(self, tag: str) -> None:
        if tag in VOID:
            return
        # Pop to the matching open element; a stray close tag is ignored.
        for depth in range(len(self.stack) - 1, 0, -1):
            if self.stack[depth].tag == tag:
                closed = self.stack[depth]
                del self.stack[depth:]
                self._closed_child(closed.titled)
                return

    def handle_data(self, data: str) -> None:
        # Words between the last element and the (?) are the (?)'s own text,
        # not the earlier element's: ``<span title>A</span> B (?)`` explains B.
        if data.strip() and not data.strip().startswith(("@", "}")):
            self.stack[-1].last_child_titled = False


def _templates_with_hints() -> list[Path]:
    found = [
        path
        for path in sorted(FRONTEND_SRC.rglob("*.html"))
        if f"<{HINT_TAG}" in path.read_text(encoding="utf-8")
    ]
    assert found, f"no template uses <{HINT_TAG}> — has the SPA moved?"
    return found


def _untitled_hints(text: str) -> list[int]:
    scanner = _HintScanner()
    scanner.feed(text)
    scanner.close()
    return scanner.untitled


def test_no_inline_template_uses_the_hint_icon() -> None:
    """The scan reads ``.html`` templates; a (?) in an inline one would escape it."""
    offenders = [
        str(path.relative_to(REPO))
        for path in sorted(FRONTEND_SRC.rglob("*.ts"))
        if path != HINT_COMPONENT
        and not path.name.endswith(".spec.ts")
        and f"<{HINT_TAG}" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, (
        f"<{HINT_TAG}> in an inline template, where this check can't see it: "
        f"{offenders}. Move the template to a .html file."
    )


def test_every_hint_icon_has_titled_text() -> None:
    offenders = [
        f"{path.relative_to(REPO)}:{line}"
        for path in _templates_with_hints()
        for line in _untitled_hints(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, (
        "A (?) explains text that has no tooltip of its own. Give the label or "
        "heading it follows a short title=, and keep the long version in the "
        "(?)'s hint (docs/style-guide.md §3.5):\n  " + "\n  ".join(offenders)
    )


def test_the_hint_icon_masks_an_inherited_title() -> None:
    """Hovering the (?) shows its long version only, not its label's short one too.

    A native tooltip comes from the nearest ancestor with a ``title``, so a
    (?) inside a titled ``<label>`` would show both. An empty ``title`` on the
    icon stops the inheritance.
    """
    source = HINT_COMPONENT.read_text(encoding="utf-8")
    assert 'class="field-hint-icon"' in source
    assert 'title=""' in source, "the (?) must set title=\"\" so its label's title doesn't show over it"


# -- The scanner itself ------------------------------------------------------


def test_scanner_flags_bare_text() -> None:
    assert _untitled_hints('<div><span>Threshold:</span>\n<vt-field-hint-icon hint="x" /></div>') == [2]


def test_scanner_accepts_a_titled_previous_sibling() -> None:
    assert _untitled_hints('<div><span title="a">Threshold:</span><vt-field-hint-icon hint="x" /></div>') == []


def test_scanner_accepts_a_titled_label_through_an_inline_span() -> None:
    html = '<label title="a"><input type="checkbox" /><span>Signposts<vt-field-hint-icon hint="x"></vt-field-hint-icon></span></label>'
    assert _untitled_hints(html) == []


def test_scanner_accepts_a_bound_title_across_control_flow() -> None:
    html = (
        '<label [title]="f.label">{{ f.label }}:\n'
        '  @if (f.required) { <span class="required">*</span> }\n'
        '  @if (f.hint) { <vt-field-hint-icon [hint]="f.hint"></vt-field-hint-icon> }\n'
        "</label>"
    )
    assert _untitled_hints(html) == []


def test_scanner_stops_at_the_nearest_block() -> None:
    html = '<div title="row"><p>Sentence.<vt-field-hint-icon hint="x" /></p></div>'
    assert _untitled_hints(html) == [1]


def test_scanner_ignores_a_component_title_input() -> None:
    html = '<vt-modal title="Settings"><h3>Server<vt-field-hint-icon hint="x" /></h3></vt-modal>'
    assert _untitled_hints(html) == [1]


def test_scanner_does_not_credit_a_titled_sibling_with_later_words() -> None:
    html = '<p><span title="a">A</span> then more words<vt-field-hint-icon hint="x" /></p>'
    assert _untitled_hints(html) == [1]
