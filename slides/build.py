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
# Injected on every slide of a build group after the first, so the whole
# progression shares one page number (Marpit's `paginate: hold`).
HOLD = "<!-- _paginate: hold -->"
# Emitted on every page of a build group, and nowhere else: the letter that
# distinguishes 5a from 5b. Absolutely positioned by the theme just right of
# Marp's own page number, and drawn fainter than it, so the pair reads as one
# label with the letter subordinate to the number.
LETTER_DIV = '<div class="pageno-letter">{}</div>'
# A fragment's own per-slide class directive, merged into the injected one.
CLASS_RE = re.compile(r"<!--\s*_class:\s*(.+?)\s*-->")
# A line that is exactly a Marp slide separator.
RULE_RE = re.compile(r"^-{3,}\s*$")
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
DEFAULT_FRONTMATTER = {"marp": "true", "theme": "vtsearch", "paginate": "true"}


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
        if is_directive_comment(body) or BUILD_BODY_RE.fullmatch(body):
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
    slide — the full fragment, markers removed — comes last. Every slide after
    the first holds the page number, so the whole progression reads as one
    slide to the audience, and every slide (the final one included) gets the
    theme's top-anchoring `build` class, so a reveal adds ink below what is
    already on screen instead of re-centring the column between pages. A
    fragment with no markers returns itself.
    """
    lines = text.splitlines()
    markers = [(i, m.group(1)) for i, m in ((i, BUILD_RE.match(line)) for i, line in enumerate(lines)) if m]
    if not markers:
        return [text]

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
        prefix = f"{HOLD}\n\n" if count else ""
        slides.append(f"{prefix}{body}\n\n{build_class}\n\n{LETTER_DIV.format(stage_letter(count))}")
    final = "\n".join(line for line in lines if not BUILD_RE.match(line)).strip("\n")
    slides.append(f"{HOLD}\n\n{final}\n\n{build_class}\n\n{LETTER_DIV.format(stage_letter(len(markers)))}")
    return slides


#: What one speaker page's notes column holds, measured against the theme:
#: notes are 20px on a 1.38 line height in a 636px column, which is about 62
#: characters a line and 21 lines a page. Deliberately conservative — the cost
#: of underestimating is one extra continuation page, and the cost of
#: overestimating is a sentence the presenter cannot read, which is the bug
#: this exists to make impossible (#3246).
NOTES_CHARS_PER_LINE = 67
NOTES_LINES = 23
#: A paragraph's bottom margin, in lines.
NOTES_PARAGRAPH_GAP = 0.4


def _notes_pages(notes: list[str]) -> list[list[str]]:
    """Split *notes* into as many speaker pages as it takes for all of it to fit.

    A speaker page that clips its own notes mid-sentence is worse than useless
    — the presenter cannot tell that anything is missing. The type floor rules
    out shrinking to fit, so the overflow goes onto a continuation page
    instead. A single paragraph longer than a whole page still overflows; the
    estimator reports that by giving it its own page, which is the loudest
    thing a build step can do without failing a deck for being wordy.
    """
    pages: list[list[str]] = []
    current: list[str] = []
    used = 0.0
    for note in notes:
        cost = max(1, math.ceil(len(note) / NOTES_CHARS_PER_LINE)) + NOTES_PARAGRAPH_GAP
        if current and used + cost > NOTES_LINES:
            pages.append(current)
            current, used = [], 0.0
        current.append(note)
        used += cost
    pages.append(current or ["*(no presenter notes on this slide)*"])
    return pages


def _frame_strip(deck: str, pages: list[int]) -> str:
    """The build group's reveals, as a lettered contact sheet.

    A speaker page used to spend a sentence of its notes saying "in the
    audience deck this slide is a seven-page build" — prose standing in for a
    picture, in a column that had none to spare, beside a quarter of the page
    that was empty. The frames themselves say it better and for free: the
    presenter sees what each advance puts on screen, and every one carries the
    letter the notes refer to it by.

    Empty for a fragment that is one page — there is nothing to contact-sheet.
    """
    if len(pages) < 2:
        return ""
    cells = "\n".join(
        f'<figure><img src="_build/imgs/{deck}.{page:03d}.png"><figcaption>{stage_letter(i)}</figcaption></figure>'
        for i, page in enumerate(pages)
    )
    return f'<div class="speaker-frames">\n{cells}\n</div>\n'


def speaker_page(deck: str, pages: list[int], text: str) -> list[str]:
    """Build the speaker pages for one fragment: the slide beside its notes.

    The miniature is the per-slide PNG of the audience deck (rendered by
    render.sh into _build/imgs/ before this runs), so the speaker sees exactly
    what the audience sees, pixel for pixel — page number included. When the
    fragment is a build, the frames of the whole group follow it as a lettered
    contact sheet, filling the space the single miniature left empty. The paths
    are written repo-`slides/`-relative like every fragment figure, and
    rewrite_images repoints them for _build/.

    Returns one page per chunk of notes: usually one, more when the notes are
    too long to fit at the type floor.
    """
    last = pages[-1]
    image = f"_build/imgs/{deck}.{last:03d}.png"
    chunks = _notes_pages(fragment_notes(text))
    strip = _frame_strip(deck, pages)
    out: list[str] = []
    for number, chunk in enumerate(chunks):
        # The contact sheet goes on the first page only: a continuation page is
        # more notes about the same slide, not a second slide.
        left = f"![Slide {last}]({image})\n\n" + (strip if number == 0 else "")
        out.append(
            "<!-- _class: speaker -->\n<!-- _paginate: false -->\n\n"
            '<div class="speaker-page">\n<div class="speaker-slide">\n\n'
            f"{left}"
            '</div>\n<div class="speaker-notes">\n\n' + "\n\n".join(chunk) + "\n\n</div>\n</div>"
        )
    return out


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


def check_fragment(name: str, text: str, problems: list[str]) -> None:
    for lineno, line in enumerate(text.splitlines(), 1):
        if RULE_RE.match(line):
            problems.append(
                f"fragments/{name}.md:{lineno}: bare `---` splits this fragment into two "
                f"slides; use `***` for a horizontal rule"
            )
    check_build_markers(name, text, problems)
    for match in IMAGE_RE.finditer(text):
        target = match.group(1)
        if target.startswith(("http://", "https://", "data:")):
            continue
        if not (ROOT / target).exists():
            line = text[: match.start()].count("\n") + 1
            problems.append(f"fragments/{name}.md:{line}: figure not found: {target}")


def assemble(deck: str, write: bool, speaker: bool = False) -> list[str]:
    """Preflight one deck; write _build/<deck>[.speaker].md unless write=False.

    Returns the list of problems found (empty means the deck is clean).
    """
    manifest = DECKS / f"{deck}.deck"
    if not manifest.exists():
        raise DeckError(f"no such deck: {manifest.relative_to(ROOT)}")

    front, names = parse_manifest(manifest)
    problems: list[str] = []
    bodies: list[str] = []
    page = 0  # audience-deck page count, builds included

    for name, extras in names:
        fragment = SLIDES / f"{name}.md"
        if not fragment.exists():
            problems.append(f"{deck}.deck: missing fragment: fragments/{name}.md")
            continue
        text = fragment.read_text().strip("\n")
        check_fragment(name, text, problems)
        stages = [with_extra_classes(stage, extras) for stage in expand_builds(text)]
        pages = list(range(page + 1, page + 1 + len(stages)))
        page += len(stages)
        if not speaker:
            bodies.extend(stages)
            continue
        # One speaker page per fragment (more when its notes overflow); the
        # miniature is the *final* stage of the audience build, which is the
        # page the fragment's notes narrate, and the lettered contact sheet
        # beneath it is every page of the group.
        for number in pages if write else []:
            if not (BUILD / "imgs" / f"{deck}.{number:03d}.png").exists():
                problems.append(
                    f"{deck}.deck: missing slide image _build/imgs/{deck}.{number:03d}.png — "
                    f"the speaker build needs the audience deck rendered to per-slide PNGs "
                    f"first; use `./render.sh {deck} pdf --speaker`, which does both"
                )
        bodies.extend(speaker_page(deck, pages, text))

    if problems or not write:
        return problems

    merged = dict(DEFAULT_FRONTMATTER)
    merged.update(front)
    header = "\n".join(f"{k}: {yaml_scalar(v)}" for k, v in merged.items())

    BUILD.mkdir(exist_ok=True)
    out = BUILD / (f"{deck}.speaker.md" if speaker else f"{deck}.md")
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
            pages += len(expand_builds(fragment.read_text().strip("\n"))) if fragment.exists() else 1
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
    parser.add_argument("--all", action="store_true", help="build every deck")
    parser.add_argument("--check", action="store_true", help="preflight only")
    parser.add_argument("--list", action="store_true", help="show decks and orphans")
    args = parser.parse_args()

    if args.list:
        cmd_list()
        return 0

    targets = all_decks() if (args.all or args.check) else [args.deck] if args.deck else []
    if not targets:
        parser.error("give a deck name, --all, --check, or --list")

    problems: list[str] = []
    for deck in targets:
        try:
            problems += assemble(deck, write=not args.check, speaker=args.speaker)
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
