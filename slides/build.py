#!/usr/bin/env python3
"""Assemble a Marp deck from a .deck manifest and a library of slide fragments.

A manifest names slides in order; this script concatenates them with `---`
separators, prepends Marp front matter, and preflights everything that would
otherwise fail late (missing fragment, missing figure, a stray `---` inside a
fragment that would silently split one slide into two, a build marker naming a
missing stage figure).

A fragment may carry `<!-- build -->` / `<!-- build: figs/x.png -->` markers:
progressive-reveal chop points. The audience build expands each marker into an
earlier stage of the slide (the content above the marker, the figure swapped
when the marker names one), all sharing one page number; the speaker build
keeps one page per fragment — the final stage. See slides/README.md.

    ./build.py hold-the-line         # -> _build/hold-the-line.md
    ./build.py --speaker hold-the-line   # -> _build/hold-the-line.speaker.md
    ./build.py --all
    ./build.py --check               # preflight only, write nothing
    ./build.py --list                # decks, slide counts, unused fragments

The --speaker variant renders presenter notes *visibly*, PowerPoint
notes-page style: each speaker page shows a miniature of the real rendered
slide beside the notes for it. Notes are the HTML comments that are not Marp
directives — the same comments Marp exports as PPTX/HTML presenter notes, so
they are authored once. The speaker build references per-slide PNGs of the
audience deck under _build/imgs/, which render.sh produces first; run
`./render.sh <deck> pdf --speaker` rather than calling this mode directly.
The audience build is untouched.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SLIDES = ROOT / "fragments"
DECKS = ROOT / "decks"
BUILD = ROOT / "_build"

# ![alt](path)  — captures the path, ignoring any "title" suffix.
IMAGE_RE = re.compile(r"!\[[^\]]*\]\(\s*([^)\s]+)")
# src="path" — the speaker build's frame strip writes raw <img> tags, because a
# markdown image cannot sit inside a <figure> without a blank line either side.
SRC_RE = re.compile(r'src="([^"]+)"')
# A build marker, alone on its line: `<!-- build -->` repeats the slide in the
# audience deck with only the content above the marker; `<!-- build: figs/x.png -->`
# additionally swaps the slide's (first) figure for that stage's figure. The
# speaker build ignores markers and keeps one page per fragment.
BUILD_RE = re.compile(r"^\s*<!--\s*build(?:\s*:\s*(\S+))?\s*-->\s*$")
# The same marker as a comment body, so notes extraction can skip it.
BUILD_BODY_RE = re.compile(r"\s*build(?:\s*:\s*\S+)?\s*")
# Which of the two frame-overview shapes a fragment's reveals get on the
# speaker page, alone on its line: `<!-- frames: equal -->` for reveals of
# equal standing (a click-through, a run of votes), `<!-- frames: build -->`
# for one picture drawn in stages. Build-up is the default because that is what
# a build marker usually means; the declaration exists because the difference
# is semantic — later frames of a build *contain* the earlier ones — and no
# amount of looking at the PNGs can be trusted to tell. Stripped from every
# emitted deck, and never a presenter note.
FRAMES_RE = re.compile(r"^\s*<!--\s*frames\s*:\s*(\S+)\s*-->\s*$")
FRAMES_BODY_RE = re.compile(r"\s*frames\s*:\s*\S+\s*")
FRAME_STYLES = ("build", "equal")
# The page number, emitted by this script rather than by Marpit. Marpit can
# only count pages or hold the previous count, and this deck needs neither: the
# title slide takes no number at all (so the first real slide is 1, not 2), and
# a fragment shown six times over is *one* slide shown six ways, which wants one
# number and six letters however far apart the six pages fall. Both are rules
# about fragments, which Marpit cannot see, so `paginate: false` goes in the
# front matter and the number is drawn here.
PAGENO_DIV = '<div class="pageno">{}</div>'
# Emitted on every page of a numbering group, and nowhere else: the letter that
# distinguishes 5a from 5b. Absolutely positioned by the theme just right of the
# page number, and drawn fainter than it, so the pair reads as one label with
# the letter subordinate to the number.
LETTER_DIV = '<div class="pageno-letter">{}</div>'
# A fragment that opts out of numbering entirely — the title slide. It keeps
# Marpit's own spelling because that is what the directive means; what this
# script adds is that such a slide does not *consume* a number either.
UNPAGINATED_RE = re.compile(r"<!--\s*_paginate:\s*false\s*-->")
# A fragment's own per-slide class directive, merged into the injected one.
CLASS_RE = re.compile(r"<!--\s*_class:\s*(.+?)\s*-->")
# A line that is exactly a Marp slide separator.
RULE_RE = re.compile(r"^-{3,}\s*$")
#: A line break inside a headline. See `check_headline`.
BR_RE = re.compile(r"<br\s*/?>", re.I)
# An HTML comment, possibly spanning lines.
COMMENT_RE = re.compile(r"<!--(.*?)-->", re.DOTALL)
# A line that sets a Marp/Marpit directive (`_class: lead`, `paginate: false`,
# ...). A comment whose every non-blank line matches is a directive comment;
# any other comment is a presenter note (mirrors Marp's own reading).
DIRECTIVE_LINE_RE = re.compile(
    r"^\s*_?(?:marp|theme|style|class|paginate|header|footer|color|transition"
    r"|headingDivider|math|lang|size|background[A-Za-z]*)\s*:"
)

# Front-matter keys we default when a manifest doesn't set them.
DEFAULT_FRONTMATTER = {"marp": "true", "theme": "vtsearch", "paginate": "false"}

# Extra CSS for the editable-PowerPoint cut, injected as a `style:` front-matter
# key so the theme itself is untouched.
#
# Marp builds an editable `.pptx` by rendering the deck to PDF and importing it
# into Impress with `impress_pdf_import`; that importer reconstructs shapes from
# the PDF's drawing operators, and it reconstructs a **CSS text shadow** as one
# more text frame plus a greyscale bitmap of the glyphs, per shadow layer. The
# headline's white halo has four layers, so every full-bleed title arrived in
# PowerPoint as five stacked copies of itself behind four alpha masks — which
# is what made the editable export look broken, and the only thing that did:
# the figures themselves import at exactly the right size and position (#3779).
#
# Dropping the halo costs this deck almost nothing. It exists to separate the
# headline from a screenshot's own chrome, and the screenshots are composed with
# the app's left edge at 375px while the title notch ends at 360 — so on every
# slide in this deck the headline already sits on plain white.
EDITABLE_STYLE = "section.full h2 { text-shadow: none; }"


class DeckError(Exception):
    pass


def parse_manifest(path: Path) -> tuple[dict[str, str], list[tuple[str, list[str]]]]:
    """Return (front-matter dict, ordered `(slide name, extra classes)` pairs).

    Format: `key: value` lines, then a `slides:` line, then one slide name per
    line. `#` starts a comment anywhere, so a slide can be parked by commenting
    it out rather than deleting it.

    A slide line may carry trailing `+class` tokens — `outline-foo +at2` — which
    are merged into that *use* of the fragment. It exists for the one thing a
    fragment cannot say about itself: a deck that shows its outline again
    before each section wants the same five lines six times over, with a
    different one marked each time. Six near-identical fragments would be six
    copies to keep in step; one fragment used six ways cannot drift.
    """
    front: dict[str, str] = {}
    slides: list[tuple[str, list[str]]] = []
    in_slides = False

    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line == "slides:":
            in_slides = True
            continue
        if in_slides:
            name, *tokens = line.split()
            bad = [t for t in tokens if not t.startswith("+")]
            if bad:
                raise DeckError(f"{path.name}:{lineno}: expected `+class` after the fragment name, got {bad[0]!r}")
            slides.append((name, [t[1:] for t in tokens]))
        elif ":" in line:
            key, value = line.split(":", 1)
            front[key.strip()] = value.strip()
        else:
            raise DeckError(f"{path.name}:{lineno}: expected `key: value`, got {line!r}")

    if not in_slides:
        raise DeckError(f"{path.name}: no `slides:` section")
    if not slides:
        raise DeckError(f"{path.name}: `slides:` section is empty")
    return front, slides


def with_extra_classes(text: str, extras: list[str]) -> str:
    """Append a `_class` directive merging *extras* into whatever the fragment sets.

    Marpit takes the last `_class` on a slide, so the new directive has to
    restate the fragment's own classes rather than only adding to them.
    """
    if not extras:
        return text
    own = CLASS_RE.findall(text)
    merged = " ".join(dict.fromkeys((own[-1].split() if own else []) + extras))
    return f"{text}\n\n<!-- _class: {merged} -->"


def yaml_scalar(value: str) -> str:
    """Quote a front-matter value if bare YAML would misread it."""
    if value and (value[0] in "\"'" or re.fullmatch(r"[\w./+-]+", value)):
        return value
    return '"' + value.replace('"', '\\"') + '"'


def rewrite_images(text: str) -> str:
    """Repoint image paths from repo-root-relative to _build/-relative.

    Fragments are authored with paths like `figs/x.png` relative to the repo
    root, but the assembled deck lives in _build/, so Marp would resolve them
    one directory too deep. Marp only *warns* about missing local files and
    still exits 0, so this must be right rather than merely checked.
    """

    def repoint(match: re.Match[str]) -> str:
        target = match.group(1)
        if target.startswith(("http://", "https://", "data:", "/")):
            return match.group(0)
        relative = os.path.relpath(ROOT / target, BUILD)
        return match.group(0).replace(target, relative)

    return SRC_RE.sub(repoint, IMAGE_RE.sub(repoint, text))


def stage_letter(index: int) -> str:
    """The suffix a build's *index*-th page carries after the slide number.

    Pages of one build group share a page number (`_paginate: hold`), which is
    what makes "the slide with the mixture plot" name one slide — but it also
    leaves the room and the presenter with no way to say *which* reveal. So
    every page of a group carries a letter after the number: 5a, 5b, 5c. A
    fragment with no build markers is one page and carries no letter.
    """
    letters = ""
    while True:
        index, remainder = divmod(index, 26)
        letters = chr(ord("a") + remainder) + letters
        if index == 0:
            return letters
        index -= 1


def is_directive_comment(body: str) -> bool:
    lines = [line for line in body.splitlines() if line.strip()]
    return bool(lines) and all(DIRECTIVE_LINE_RE.match(line) for line in lines)


def fragment_notes(text: str) -> list[str]:
    """A fragment's presenter notes, one entry per paragraph.

    Notes are the non-directive HTML comments; note text is markdown, and the
    continuation-line indentation of the comment style is stripped so it can't
    be misread as a code block. The return value is flat — a paragraph, not a
    comment, is the unit — because that is the granularity the speaker build
    has to page at: one comment holding a slide's whole narration is longer
    than any page can show, so a splitter working comment by comment would
    have nothing to split.
    """
    notes: list[str] = []
    for match in COMMENT_RE.finditer(text):
        body = match.group(1)
        if is_directive_comment(body) or BUILD_BODY_RE.fullmatch(body) or FRAMES_BODY_RE.fullmatch(body):
            continue
        # Reflow: Marp renders single newlines as hard breaks, so joining the
        # comment's wrapped lines with "\n" would keep its ragged wrapping.
        # Collapse each blank-line-separated block to one line instead.
        for block in re.split(r"\n\s*\n", body):
            paragraph = " ".join(line.strip() for line in block.split("\n") if line.strip())
            if paragraph:
                notes.append(paragraph)
    return notes


def strip_notes(text: str) -> str:
    """Remove presenter-note comments, keeping directive comments."""

    def drop(match: re.Match[str]) -> str:
        return match.group(0) if is_directive_comment(match.group(1)) else ""

    return COMMENT_RE.sub(drop, text)


def swap_figure(body: str, figure: str) -> str:
    """Repoint the first image in *body* at *figure* (a slides/-relative path)."""
    match = IMAGE_RE.search(body)
    if not match:
        return body  # check_fragment already reported the missing image
    return body[: match.start()] + match.group(0).replace(match.group(1), figure) + body[match.end() :]


def expand_builds(text: str) -> list[str]:
    """Expand a fragment's build markers into its audience slide sequence.

    The fragment is authored as the *final* slide; each `<!-- build -->` marker
    chops an earlier reveal out of it: a slide holding only the content above
    the marker, with the figure swapped when the marker names one, and with
    presenter notes stripped (they belong to the final slide alone). The final
    slide — the full fragment, markers removed — comes last. Every slide (the
    final one included) gets the theme's top-anchoring `build` class, so a
    reveal adds ink below what is already on screen instead of re-centring the
    column between pages. A fragment with no markers returns itself.

    Page numbers and letters are *not* set here: they belong to the fragment's
    whole numbering group, which `assemble` knows about and one fragment does
    not — a fragment shown six times over is six pages of one slide.
    """
    # The frames directive steers the speaker build only; Marp would read it as
    # a presenter note, so it never reaches an emitted deck.
    lines = [line for line in text.splitlines() if not FRAMES_RE.match(line)]
    markers = [(i, m.group(1)) for i, m in ((i, BUILD_RE.match(line)) for i, line in enumerate(lines)) if m]
    if not markers:
        return ["\n".join(lines)]

    # Appended last so it wins over any `_class` the fragment sets itself,
    # which is why it must also carry those classes forward.
    fragment_class = CLASS_RE.search(text)
    build_class = f"<!-- _class: {fragment_class.group(1)} build -->" if fragment_class else "<!-- _class: build -->"

    slides: list[str] = []
    for count, (cut, figure) in enumerate(markers):
        kept = [line for line in lines[:cut] if not BUILD_RE.match(line)]
        body = strip_notes("\n".join(kept)).strip("\n")
        if figure:
            body = swap_figure(body, figure)
        slides.append(f"{body}\n\n{build_class}")
    final = "\n".join(line for line in lines if not BUILD_RE.match(line)).strip("\n")
    slides.append(f"{final}\n\n{build_class}")
    return slides


def fragment_frame_style(text: str) -> str:
    """Which overview shape a fragment's frames get. See `frame_overview`."""
    for line in text.splitlines():
        match = FRAMES_RE.match(line)
        if match:
            return match.group(1)
    return "build"


def page_count(text: str) -> int:
    """How many audience pages one *use* of a fragment renders as."""
    return len([line for line in text.splitlines() if BUILD_RE.match(line)]) + 1


#: Speaker-page geometry, in CSS px at 1280x720. Every number here mirrors a
#: rule in the theme's `section.speaker` block: they are one layout described
#: twice, once to draw it and once to decide what fits in it, so a change to
#: either half is a change to both.
SPEAKER_PAD_X, SPEAKER_PAD_Y = 42, 36
SPEAKER_W = 1280 - 2 * SPEAKER_PAD_X
SPEAKER_H = 720 - 2 * SPEAKER_PAD_Y
#: The visual column — the miniature and the frame overview — floats left at
#: this share of the page, and the notes wrap around it and then *under* it.
#: That reflow is what buys a wordy slide its one page: a note that starts
#: beside a 478px column finishes across the full 1196.
VISUAL_FRACTION = 0.40
#: An equal-weight overview gets a wider column, because it is carrying the
#: whole slide rather than a hero plus footnotes: at 40% its frames would be
#: half the size of the hero they replaced, which is the opposite of the point.
#: The notes lose a little width beside it and get it all back underneath.
VISUAL_FRACTION_EQUAL = 0.52
VISUAL_W = VISUAL_FRACTION * SPEAKER_W
VISUAL_GUTTER = 34
FRAME_GAP = 10
FRAME_BORDER = 2
#: Gap between the hero miniature and the strip of frames under it.
STRIP_TOP = 18
SLIDE_ASPECT = 9 / 16

#: Presenter notes are 18px, not the deck's 20px floor. The floor is about a
#: projector at the back of a room; this page is a PDF on the presenter's own
#: laptop, half a metre away, and the 20px version could not fit a long slide's
#: narration on one page however the pictures were arranged. See STYLE.md.
NOTES_FONT_PX = 18
NOTES_LINE_HEIGHT = 1.38
NOTES_LINE_PX = NOTES_FONT_PX * NOTES_LINE_HEIGHT
#: Average glyph width of the notes face. Measured against the theme: 20px
#: Helvetica in a 636px column fits about 67 characters, i.e. 9.5px each, and
#: the face scales linearly. Deliberately a touch wide — overestimating the
#: width fails a deck that would in fact have fitted, which costs an edit,
#: while underestimating it clips a sentence the presenter cannot read.
NOTES_PX_PER_CHAR = 9.5 * NOTES_FONT_PX / 20
NOTES_LINES = int(SPEAKER_H // NOTES_LINE_PX)
#: A paragraph's bottom margin, in lines.
NOTES_PARAGRAPH_GAP = 0.4


#: How tall a build-up's strip of earlier frames may stand, in px. A budget
#: rather than a column count, because the two things the strip trades against
#: each other — how big a thumbnail is, and how many lines of notes run full
#: width under it — are both measured in pixels. Set where a ten-frame build
#: still leaves the notes two thirds of the page.
STRIP_MAX_H = 150


def frame_columns(count: int, style: str) -> int:
    """How many frames a row of the overview holds.

    The two styles want opposite things from the grid. An **equal-weight**
    overview *is* the slide — every frame carries its own content — so it goes
    as wide as it can without leaving a widow: two up to four (a square), three
    beyond, and it never shrinks itself to buy room for prose.

    A **build-up** strip is an afterthought under the hero, and the presenter
    only has to recognise those frames rather than read them, so it is sized to
    a height budget instead: the fewest columns — hence the largest thumbnails
    — whose strip still fits `STRIP_MAX_H`. That keeps a ten-frame build from
    spending most of the page on thumbnails of one picture, and it is why a
    long build's frames come out smaller than a short one's.
    """
    if style == "equal":
        return count if count <= 3 else (2 if count == 4 else 3)
    # Never fewer than three columns: at two, a two-frame build's lone earlier
    # frame comes out half the size of the hero above it and stops reading as
    # subordinate to it. Six is the ceiling because the theme defines --c2
    # through --c6; past twelve frames the strip grows a third row rather than a
    # seventh column, which is the right trade anyway — a seventh column is 66px.
    for cols in range(3, 6):
        if _grid_height(count, cols) <= STRIP_MAX_H:
            return cols
    return 6


def _grid_height(count: int, cols: int, width: float = VISUAL_W) -> float:
    rows = math.ceil(count / cols)
    cell = (width - (cols - 1) * FRAME_GAP) / cols
    return rows * _cell_height(cell) + (rows - 1) * FRAME_GAP


def visual_width(style: str) -> float:
    """How wide the floated visual column stands, in px. See `frame_overview`."""
    return (VISUAL_FRACTION_EQUAL if style == "equal" else VISUAL_FRACTION) * SPEAKER_W


def _cell_height(width: float) -> float:
    return width * SLIDE_ASPECT + FRAME_BORDER


def speaker_visual_height(frames: int, style: str) -> float:
    """How tall the floated visual column stands, in px.

    This is what decides how much of the notes column is narrow (beside the
    pictures) and how much is full width (below them), so it has to agree with
    the theme to the pixel rather than approximately.
    """
    width = visual_width(style)
    if frames < 2:
        return _cell_height(width)
    if style == "equal":
        return _grid_height(frames, frame_columns(frames, style), width)
    return _cell_height(width) + STRIP_TOP + _grid_height(frames - 1, frame_columns(frames - 1, style))


def notes_lines_used(notes: list[str], visual_height: float, visual_width_px: float = VISUAL_W) -> float:
    """Estimate how many lines *notes* occupy beside and below the visual column.

    Lines above the float's bottom edge are narrow; everything after it runs
    the full width of the page. The estimate walks a paragraph line by line so
    one that *starts* beside the pictures and finishes under them is costed at
    both widths, which is the common case and the whole reason the float is
    worth having.
    """
    beside = math.ceil(visual_height / NOTES_LINE_PX)
    narrow = SPEAKER_W - visual_width_px - VISUAL_GUTTER
    line = 0.0
    for index, note in enumerate(notes):
        if index:
            line += NOTES_PARAGRAPH_GAP
        remaining = len(note)
        while True:
            width = narrow if line < beside else SPEAKER_W
            remaining -= max(1, int(width / NOTES_PX_PER_CHAR))
            line += 1
            if remaining <= 0:
                break
    return line


def notes_overflow(notes: list[str], frames: int, style: str) -> float:
    """Lines by which *notes* miss fitting one speaker page; 0.0 when they fit."""
    used = notes_lines_used(notes, speaker_visual_height(frames, style), visual_width(style))
    return max(0.0, used - NOTES_LINES)


def _frame_cell(deck: str, page: int, letter: str) -> str:
    return f'<figure><img src="_build/imgs/{deck}.{page:03d}.png"><figcaption>{letter}</figcaption></figure>'


def frame_overview(deck: str, pages: list[int], group: list[int], style: str) -> str:
    """The floated visual column: this showing's slide, and its sibling frames.

    Two shapes, because a group's frames are not all the same kind of thing.

    A **build-up** — the default, and what a `<!-- build -->` marker usually
    means — reveals one picture in stages, so only the last frame holds all of
    it. That frame is the hero, drawn big, and the rest sit under it small: the
    presenter needs to *read* the slide they are on and only needs to
    *recognise* the steps that got there.

    An **equal-weight** group has no such last frame. Its reveals are different
    pictures of equal standing — three screens of a click-through, six votes in
    a row — so a hero would be an arbitrary one of them blown up while the ones
    that carry the argument stay thumbnails. Every frame is drawn the same size
    instead, which is also *bigger*, because the hero's space is shared out.

    A fragment that renders as one page has no overview at all; it is the hero
    and nothing else.
    """
    hero = pages[-1]
    if len(group) < 2:
        return f'<figure class="speaker-hero"><img src="_build/imgs/{deck}.{hero:03d}.png"></figure>\n'
    letters = {page: stage_letter(index) for index, page in enumerate(group)}
    if style == "equal":
        shown, head = group, ""
    else:
        # Every frame but the one drawn big — "earlier" for a real build, "the
        # other showings" for a fragment the deck comes back to.
        shown = [page for page in group if page != hero]
        head = (
            f'<figure class="speaker-hero"><img src="_build/imgs/{deck}.{hero:03d}.png">'
            f"<figcaption>{letters[hero]}</figcaption></figure>\n"
        )
    cols = frame_columns(len(shown), style)
    cells = "\n".join(_frame_cell(deck, page, letters[page]) for page in shown)
    return f'{head}<div class="speaker-frames speaker-frames--c{cols}">\n{cells}\n</div>\n'


def notes_for_showing(notes: list[str], letters: set[str], first: bool) -> list[str]:
    """The notes belonging to one *showing* of a fragment shown several times.

    A fragment used once narrates all of itself on its one speaker page. A
    fragment used six times — the outline, coming back before each section —
    does not: the presenter reaching section 3 wants the line about section 3,
    not the four paragraphs they read at section 1 for the third time (#3265).

    So a note that names a letter goes to the showing that prints that letter,
    and a note that names none is general to the slide and goes to its first
    showing only.
    """
    kept = []
    for note in notes:
        named = set(NOTE_LETTER_RE.findall(note))
        if named & letters or (not named and first):
            kept.append(note)
    return kept


def showing_notes(text: str, pages: list[int], group: list[int]) -> list[str]:
    """The notes one showing of a fragment narrates."""
    notes = fragment_notes(text)
    if pages == group:
        return notes
    letters = {stage_letter(group.index(page)) for page in pages}
    return notes_for_showing(notes, letters, first=pages[0] == group[0])


def speaker_page(deck: str, pages: list[int], group: list[int], text: str) -> str:
    """One showing of a fragment as one speaker page: pictures beside notes.

    The miniature is the per-slide PNG of the audience deck (rendered by
    render.sh into _build/imgs/ before this runs), so the speaker sees exactly
    what the audience sees, pixel for pixel — page number included. When the
    fragment renders as more than one page, `frame_overview` draws the rest of
    the group with it, in whichever of the two shapes the fragment declares.

    *pages* is this showing's audience pages; *group* is every page the
    fragment renders as across the deck. They differ only for a fragment shown
    more than once, where the overview still shows the whole slide and the
    notes are narrowed to this showing.

    Always exactly one page. Notes that would not fit are a deck error raised
    by `check_speaker_fit`, not a continuation page: a presenter mid-sentence
    does not turn over, so notes split across two pages are notes half read.
    """
    style = fragment_frame_style(text)
    visual = frame_overview(deck, pages, group, style)
    notes = showing_notes(text, pages, group) or ["*(no presenter notes on this slide)*"]
    return (
        "<!-- _class: speaker -->\n<!-- _paginate: false -->\n\n"
        f'<div class="speaker-page">\n<div class="speaker-visual speaker-visual--{style}">\n'
        f"{visual}"
        '</div>\n<div class="speaker-notes">\n\n' + "\n\n".join(notes) + "\n\n</div>\n</div>"
    )


def check_build_markers(name: str, text: str, problems: list[str]) -> None:
    """Preflight a fragment's build markers: syntax, stage figures, swappability."""
    saw_image = False
    for lineno, line in enumerate(text.splitlines(), 1):
        saw_image = saw_image or bool(IMAGE_RE.search(line))
        marker = BUILD_RE.match(line)
        if marker is None:
            if line.strip().startswith("<!-- build"):
                problems.append(
                    f"fragments/{name}.md:{lineno}: malformed build marker — expected "
                    f"`<!-- build -->` or `<!-- build: figs/x.png -->` alone on its line"
                )
            continue
        figure = marker.group(1)
        if figure is None:
            continue
        if not (ROOT / figure).exists():
            problems.append(f"fragments/{name}.md:{lineno}: build figure not found: {figure}")
        if not saw_image:
            problems.append(
                f"fragments/{name}.md:{lineno}: build marker names a figure, but no image appears above it to swap"
            )


# A page letter as a presenter note refers to it: `**c** — the same cut, ...`.
NOTE_LETTER_RE = re.compile(r"\*\*([a-z])\*\*")


def check_note_letters(name: str, text: str, pages: int, problems: list[str]) -> None:
    """Preflight that a numbering group's notes name every page of it.

    The audience deck prints a letter on every page of a group (5a, 5b, 5c) and
    the speaker build labels every frame of its contact sheet with the same
    letter, so a presenter reads the notes *against* those letters — which only
    works if there is a note for each. A frame nobody wrote a line for is the
    failure this catches, and it is the kind that is invisible until you are
    standing in front of a room (#3246).

    *pages* is the group's size across the whole deck, not the fragment's own
    reveal count, so a fragment shown six times owes six lettered notes even
    though it carries no build markers — and gets one note per showing rather
    than the same four paragraphs six times over.

    A single note may name several letters (`**a**, **b** — recapitulation`);
    what is checked is coverage, not one note per page.
    """
    if pages < 2:
        return
    named = {m.group(1) for comment in COMMENT_RE.finditer(text) for m in NOTE_LETTER_RE.finditer(comment.group(1))}
    missing = [stage_letter(i) for i in range(pages) if stage_letter(i) not in named]
    if missing:
        problems.append(
            f"fragments/{name}.md: this slide is {pages} pages, but its presenter notes "
            f"never mention page {', '.join(missing)} — write a note per reveal, named by "
            f"the letter the deck prints on it (`**{missing[0]}** — ...`)"
        )
    stray = sorted(letter for letter in named if letter not in {stage_letter(i) for i in range(pages)})
    if stray:
        problems.append(
            f"fragments/{name}.md: presenter notes name page {', '.join(stray)}, but this "
            f"slide is only {pages} pages (a-{stage_letter(pages - 1)})"
        )


def check_headline(name: str, text: str, problems: list[str]) -> None:
    """A headline may break once, and only once.

    The half of `slides/STYLE.md`'s *A title is two lines at most* that can be
    seen from the markdown. **Where** a two-line headline breaks is a fact about
    rendered pixels and belongs to `balance-titles.mjs`, which needs a browser
    and so cannot run in this gate — but two `<br>`s are three lines however
    they measure, and that is worth catching here rather than in a render
    somebody has to look at.
    """
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.startswith("## "):
            continue
        breaks = len(BR_RE.findall(line))
        if breaks > 1:
            problems.append(
                f"fragments/{name}.md:{lineno}: headline breaks {breaks} times, so it is "
                f"{breaks + 1} lines; a title is two lines at most (slides/STYLE.md). "
                f"Use fewer words, not a third line"
            )


def check_frames_directive(name: str, text: str, problems: list[str]) -> None:
    """Preflight a fragment's `<!-- frames: ... -->` declaration."""
    seen = []
    for lineno, line in enumerate(text.splitlines(), 1):
        match = FRAMES_RE.match(line)
        if match:
            seen.append((lineno, match.group(1)))
        elif line.strip().startswith("<!-- frames"):
            problems.append(
                f"fragments/{name}.md:{lineno}: malformed frames directive — expected "
                f"`<!-- frames: build -->` or `<!-- frames: equal -->` alone on its line"
            )
    for lineno, style in seen:
        if style not in FRAME_STYLES:
            problems.append(
                f"fragments/{name}.md:{lineno}: unknown frame style {style!r} — "
                f"expected one of {', '.join(FRAME_STYLES)}"
            )
    if len(seen) > 1:
        lines = ", ".join(str(lineno) for lineno, _ in seen)
        problems.append(f"fragments/{name}.md: more than one frames directive (lines {lines})")


def check_speaker_fit(
    showings: list[Showing], texts: dict[str, str], group: dict[str, list[int]], problems: list[str]
) -> None:
    """Preflight that every showing's notes fit on its one speaker page.

    Run on every build, the audience one included, because the fit is a
    property of the fragment and not of the deck you happen to be rendering. A
    slide whose narration only fits across two pages is one nobody can present
    from — the presenter turns over mid-sentence, or more often does not notice
    the second page at all — and discovering that at `--speaker` time means
    discovering it the evening before the talk.
    """
    for name, _extras, _stages, pages in showings:
        style = fragment_frame_style(texts[name])
        notes = showing_notes(texts[name], pages, group[name])
        over = notes_overflow(notes, len(group[name]), style)
        if not over:
            continue
        characters = math.ceil(over * SPEAKER_W / NOTES_PX_PER_CHAR)
        letters = "".join(stage_letter(group[name].index(page)) for page in pages)
        where = f" (showing {letters})" if pages != group[name] else ""
        problems.append(
            f"fragments/{name}.md{where}: presenter notes overflow the speaker page by about "
            f"{over:.0f} of {NOTES_LINES} lines — trim roughly {characters} characters, or move "
            f"detail into the report the note cites. A speaker page never continues onto a "
            f"second one; see slides/README.md."
        )


def check_fragment(name: str, text: str, problems: list[str]) -> None:
    for lineno, line in enumerate(text.splitlines(), 1):
        if RULE_RE.match(line):
            problems.append(
                f"fragments/{name}.md:{lineno}: bare `---` splits this fragment into two "
                f"slides; use `***` for a horizontal rule"
            )
    check_headline(name, text, problems)
    check_build_markers(name, text, problems)
    check_frames_directive(name, text, problems)
    for match in IMAGE_RE.finditer(text):
        target = match.group(1)
        if target.startswith(("http://", "https://", "data:")):
            continue
        if not (ROOT / target).exists():
            line = text[: match.start()].count("\n") + 1
            problems.append(f"fragments/{name}.md:{line}: figure not found: {target}")


#: One showing of a fragment: its name, that use's extra classes, the audience
#: slides it renders as, and the audience page numbers they occupy.
Showing = tuple[str, list[str], list[str], list[int]]


def lay_out(
    deck: str, names: list[tuple[str, list[str]]], problems: list[str]
) -> tuple[list[Showing], dict[str, str], dict[str, list[int]]]:
    """Read every fragment and lay out the deck's audience pages.

    Returns the showings in manifest order, each fragment's text, and each
    fragment's whole numbering *group* — every page it renders as across the
    deck. The group is what a fragment cannot know about itself: shown more
    than once, it is one slide shown several ways, and its number and letters
    belong to all its showings together.
    """
    showings: list[Showing] = []
    texts: dict[str, str] = {}
    group: dict[str, list[int]] = {}
    page = 0  # audience-deck page count, builds included
    for name, extras in names:
        fragment = SLIDES / f"{name}.md"
        if not fragment.exists():
            problems.append(f"{deck}.deck: missing fragment: fragments/{name}.md")
            continue
        text = texts.setdefault(name, fragment.read_text().strip("\n"))
        check_fragment(name, text, problems)
        stages = [with_extra_classes(stage, extras) for stage in expand_builds(text)]
        pages = list(range(page + 1, page + 1 + len(stages)))
        page += len(stages)
        group.setdefault(name, []).extend(pages)
        showings.append((name, extras, stages, pages))
    return showings, texts, group


def audience_bodies(
    showings: list[Showing], texts: dict[str, str], group: dict[str, list[int]], pageno: bool = True
) -> list[str]:
    """The audience deck's slides, each carrying its page number and letter.

    A slide's number is claimed by its fragment's first showing and reused by
    the rest; a fragment marked `_paginate: false` — the title slide — takes no
    number and does not consume one, so the first real slide is 1 rather than 2.

    `pageno=False` draws neither, for a deck being handed over rather than
    presented. The numbers are earned — they are how a question from the room
    names a slide, and how this repo's own review comments do — so this is an
    export option and not a style choice: nothing else about the deck changes,
    and the numbering is still computed, so a page's *address* is the same
    whether or not it is printed on it.
    """
    numbers: dict[str, int] = {}
    bodies: list[str] = []
    for name, _extras, stages, pages in showings:
        numbered = not UNPAGINATED_RE.search(texts[name])
        if numbered and name not in numbers:
            numbers[name] = len(numbers) + 1
        for offset, stage in enumerate(stages):
            marks = ""
            if numbered and pageno:
                marks = "\n\n" + PAGENO_DIV.format(numbers[name])
                if len(group[name]) > 1:
                    marks += "\n\n" + LETTER_DIV.format(stage_letter(group[name].index(pages[offset])))
            bodies.append(stage + marks)
    return bodies


def speaker_bodies(
    deck: str,
    showings: list[Showing],
    texts: dict[str, str],
    group: dict[str, list[int]],
    write: bool,
    problems: list[str],
) -> list[str]:
    """The speaker deck's pages: exactly one per showing.

    The miniature is the *final* stage of the audience build, which is the page
    the fragment's notes narrate, and `frame_overview` draws the rest of the
    numbering group beside or beneath it. Notes that would not fit one page are
    a deck error (`check_speaker_fit`), so this never emits two pages for one
    slide.
    """
    bodies: list[str] = []
    for name, _extras, _stages, pages in showings:
        for number in pages if write else []:
            if not (BUILD / "imgs" / f"{deck}.{number:03d}.png").exists():
                problems.append(
                    f"{deck}.deck: missing slide image _build/imgs/{deck}.{number:03d}.png — "
                    f"the speaker build needs the audience deck rendered to per-slide PNGs "
                    f"first; use `./render.sh {deck} pdf --speaker`, which does both"
                )
        bodies.append(speaker_page(deck, pages, group[name], texts[name]))
    return bodies


def assemble(deck: str, write: bool, speaker: bool = False, pageno: bool = True, editable: bool = False) -> list[str]:
    """Preflight one deck; write _build/<deck>[.speaker|.editable|.unnumbered].md unless write=False.

    Returns the list of problems found (empty means the deck is clean).
    """
    manifest = DECKS / f"{deck}.deck"
    if not manifest.exists():
        raise DeckError(f"no such deck: {manifest.relative_to(ROOT)}")

    front, names = parse_manifest(manifest)
    problems: list[str] = []
    showings, texts, group = lay_out(deck, names, problems)

    for name, text in texts.items():
        check_note_letters(name, text, len(group[name]), problems)
    check_speaker_fit(showings, texts, group, problems)

    if speaker:
        bodies = speaker_bodies(deck, showings, texts, group, write, problems)
    else:
        bodies = audience_bodies(showings, texts, group, pageno)

    if problems or not write:
        return problems

    merged = dict(DEFAULT_FRONTMATTER)
    merged.update(front)
    if editable:
        merged["style"] = EDITABLE_STYLE
    header = "\n".join(f"{k}: {yaml_scalar(v)}" for k, v in merged.items())

    BUILD.mkdir(exist_ok=True)
    parts = [""] if speaker else [p for p, on in ((".editable", editable), (".unnumbered", not pageno)) if on]
    suffix = ".speaker" if speaker else "".join(parts)
    out = BUILD / f"{deck}{suffix}.md"
    body = rewrite_images("\n\n---\n\n".join(bodies))
    out.write_text(f"---\n{header}\n---\n\n{body}\n")

    # Verify against the emitted file, not the sources: the paths that matter
    # are the ones Marp will resolve, from _build/.
    problems += [f"{deck}.deck: unresolvable from _build/: {target}" for target in unresolvable_images(body)]
    if problems:
        out.unlink()
        return problems

    print(f"built {out.relative_to(ROOT)}  ({len(bodies)} slides)")
    return []


def unresolvable_images(body: str) -> list[str]:
    """Local image paths in an assembled deck body that don't resolve from _build/."""
    missing: list[str] = []
    for pattern in (IMAGE_RE, SRC_RE):
        for match in pattern.finditer(body):
            target = match.group(1)
            if target.startswith(("http://", "https://", "data:")):
                continue
            if not (BUILD / target).exists():
                missing.append(target)
    return missing


def all_decks() -> list[str]:
    return sorted(p.stem for p in DECKS.glob("*.deck"))


def cmd_list() -> None:
    used: set[str] = set()
    for deck in all_decks():
        try:
            _, names = parse_manifest(DECKS / f"{deck}.deck")
        except DeckError as exc:
            print(f"  {deck:<24} !! {exc}")
            continue
        used.update(name for name, _ in names)
        pages = 0
        for name, _ in names:
            fragment = SLIDES / f"{name}.md"
            pages += page_count(fragment.read_text().strip("\n")) if fragment.exists() else 1
        builds = f"  ({pages} pages with builds)" if pages != len(names) else ""
        print(f"  {deck:<24} {len(names):>2} slides{builds}")

    orphans = sorted(p.stem for p in SLIDES.glob("*.md") if p.stem not in used)
    if orphans:
        print("\nfragments in no deck (fine — a library holds spares):")
        for name in orphans:
            print(f"  fragments/{name}.md")


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("deck", nargs="?", help="deck name (without .deck)")
    parser.add_argument(
        "--speaker",
        action="store_true",
        help="render presenter notes visibly; writes _build/<deck>.speaker.md",
    )
    parser.add_argument(
        "--editable",
        action="store_true",
        help="build the cut Marp's --pptx-editable imports cleanly; writes _build/<deck>.editable.md",
    )
    parser.add_argument(
        "--no-pageno",
        dest="pageno",
        action="store_false",
        help="draw no page numbers; writes _build/<deck>.unnumbered.md",
    )
    parser.add_argument("--all", action="store_true", help="build every deck")
    parser.add_argument("--check", action="store_true", help="preflight only")
    parser.add_argument("--list", action="store_true", help="show decks and orphans")
    args = parser.parse_args()

    if args.list:
        cmd_list()
        return 0

    if args.speaker and args.editable:
        parser.error("--editable and --speaker are mutually exclusive: the speaker view is a PDF, not a deck to edit")

    if args.speaker and not args.pageno:
        parser.error(
            "--no-pageno and --speaker are mutually exclusive: the speaker view is "
            "navigated by those numbers — its contact sheet labels every frame with "
            "the letter the audience deck prints beside them."
        )

    targets = all_decks() if (args.all or args.check) else [args.deck] if args.deck else []
    if not targets:
        parser.error("give a deck name, --all, --check, or --list")

    problems: list[str] = []
    for deck in targets:
        try:
            problems += assemble(
                deck, write=not args.check, speaker=args.speaker, pageno=args.pageno, editable=args.editable
            )
        except DeckError as exc:
            problems.append(str(exc))

    # A fragment shared by several decks reports once per deck; dedupe so the
    # count reflects things to fix, not decks affected.
    unique = list(dict.fromkeys(problems))
    if unique:
        print(f"\n{len(unique)} problem(s):", file=sys.stderr)
        for problem in unique:
            print(f"  {problem}", file=sys.stderr)
        return 1

    if args.check:
        print(f"{len(targets)} deck(s) OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
