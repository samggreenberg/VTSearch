"""The slide shoot's vote overrides agree with the intro figure's own verdicts.

`slides/figs/src/make-book-figs.py` draws the deck's opening argument: ten
frames, six of which it marks as *not* a book (`RANKING`'s ``False`` half — a
magazine, a DVD box set, a spiral notebook, a boxed game manual, a newspaper).
Two slides later `shoot-ui-figs.mjs` drives a real voting session, and the
button it clicks is decided by which COCO folder the frame is filed in — which
counts every one of those as a book, because COCO's annotators did.

So the Book example carries an override list (`NOT_A_BOOK` in
`scripts/screenshots/book-example.mjs`), and the deck contradicts itself in front of the
room the moment the two drift apart: #3779 caught the session voting **Good**
on the very shelf of box sets the intro had just held up as the canonical
not-a-book. Neither file can import the other — one is Python for matplotlib,
one is JavaScript for Playwright — so the agreement is checked here.

Read by parsing, not by importing: `make-book-figs` pulls in matplotlib and
`coco_fixture` at module scope, and this tier imports neither.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BOOK_FIGS = REPO / "slides" / "figs" / "src" / "make-book-figs.py"
BOOK_EXAMPLE = REPO / "scripts" / "screenshots" / "book-example.mjs"


def _literal(source: str, name: str) -> object:
    """The value of a module-level ``name = <literal>`` assignment."""
    for node in ast.parse(source).body:
        targets = getattr(node, "targets", [])
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not a module-level literal in {BOOK_FIGS.name}")


def _not_books() -> set[str]:
    """The COCO files the intro figure marks as not a book."""
    source = BOOK_FIGS.read_text()
    tiles = _literal(source, "TILES")
    ranking = _literal(source, "RANKING")
    assert isinstance(tiles, dict) and isinstance(ranking, tuple)
    return {tiles[name][0] for name, is_book in ranking if not is_book}


def _overrides() -> set[str]:
    """The COCO files the Book example refuses to vote Good on."""
    block = re.search(r"const NOT_A_BOOK = new Set\(\[(.*?)\]\);", BOOK_EXAMPLE.read_text(), re.DOTALL)
    assert block, "book-example.mjs no longer declares a NOT_A_BOOK set"
    return set(re.findall(r"'([^']+\.jpg)'", block.group(1)))


def test_shoot_overrides_match_the_intro_figures_verdicts() -> None:
    figure, shoot = _not_books(), _overrides()
    assert shoot == figure, (
        "the voting session and the intro figure disagree about what a book is.\n"
        f"  only the figure says not-a-book: {sorted(figure - shoot)}\n"
        f"  only the shoot says not-a-book:  {sorted(shoot - figure)}\n"
        "Update NOT_A_BOOK in scripts/screenshots/book-example.mjs, or RANKING in make-book-figs.py."
    )


def test_the_override_list_is_not_empty() -> None:
    # A regex that quietly stopped matching would pass the test above by
    # comparing two empty sets, which is exactly the failure this gate exists
    # to catch.
    assert _not_books(), "the intro figure marks nothing as not-a-book"
