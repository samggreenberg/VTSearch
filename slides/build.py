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

    ./build.py scale26-review        # -> _build/scale26-review.md
    ./build.py --speaker scale26-review  # -> _build/scale26-review.speaker.md
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


def parse_manifest(path: Path) -> tuple[dict[str, str], list[str]]:
    """Return (front-matter dict, ordered slide names).

    Format: `key: value` lines, then a `slides:` line, then one slide name per
    line. `#` starts a comment anywhere, so a slide can be parked by commenting
    it out rather than deleting it.
    """
    front: dict[str, str] = {}
    slides: list[str] = []
    in_slides = False

    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line == "slides:":
            in_slides = True
            continue
        if in_slides:
            slides.append(line)
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

    return IMAGE_RE.sub(repoint, text)


def is_directive_comment(body: str) -> bool:
    lines = [line for line in body.splitlines() if line.strip()]
    return bool(lines) and all(DIRECTIVE_LINE_RE.match(line) for line in lines)


def fragment_notes(text: str) -> list[str]:
    """Extract a fragment's presenter notes (non-directive HTML comments).

    Note text is markdown; the continuation-line indentation of the comment
    style is stripped so it can't be misread as a code block.
    """
    notes: list[str] = []
    for match in COMMENT_RE.finditer(text):
        body = match.group(1)
        if is_directive_comment(body) or BUILD_BODY_RE.fullmatch(body):
            continue
        # Reflow: Marp renders single newlines as hard breaks, so joining the
        # comment's wrapped lines with "\n" would keep its ragged wrapping.
        # Collapse each blank-line-separated block to one line instead.
        paragraphs = [
            " ".join(line.strip() for line in block.split("\n") if line.strip()) for block in re.split(r"\n\s*\n", body)
        ]
        cleaned = "\n\n".join(p for p in paragraphs if p)
        if cleaned:
            notes.append(cleaned)
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
        slides.append(f"{prefix}{body}\n\n{build_class}")
    final = "\n".join(line for line in lines if not BUILD_RE.match(line)).strip("\n")
    slides.append(f"{HOLD}\n\n{final}\n\n{build_class}")
    return slides


def speaker_page(deck: str, index: int, text: str) -> str:
    """Build one speaker page: the rendered slide beside its notes.

    The miniature is the per-slide PNG of the audience deck (rendered by
    render.sh into _build/imgs/ before this runs), so the speaker sees exactly
    what the audience sees, pixel for pixel — page number included. The path
    is written repo-`slides/`-relative like every fragment figure, and
    rewrite_images repoints it for _build/.
    """
    image = f"_build/imgs/{deck}.{index:03d}.png"
    notes = fragment_notes(text) or ["*(no presenter notes on this slide)*"]
    return (
        "<!-- _class: speaker -->\n<!-- _paginate: false -->\n\n"
        '<div class="speaker-page">\n<div class="speaker-slide">\n\n'
        f"![Slide {index}]({image})\n\n"
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

    for name in names:
        fragment = SLIDES / f"{name}.md"
        if not fragment.exists():
            problems.append(f"{deck}.deck: missing fragment: fragments/{name}.md")
            continue
        text = fragment.read_text().strip("\n")
        check_fragment(name, text, problems)
        stages = expand_builds(text)
        page += len(stages)
        if not speaker:
            bodies.extend(stages)
            continue
        # One speaker page per fragment; the miniature is the *final* stage of
        # the audience build, which is the page the fragment's notes narrate.
        if write and not (BUILD / "imgs" / f"{deck}.{page:03d}.png").exists():
            problems.append(
                f"{deck}.deck: missing slide image _build/imgs/{deck}.{page:03d}.png — "
                f"the speaker build needs the audience deck rendered to per-slide PNGs "
                f"first; use `./render.sh {deck} pdf --speaker`, which does both"
            )
        bodies.append(speaker_page(deck, page, text))

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
    for match in IMAGE_RE.finditer(body):
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
        used.update(names)
        pages = 0
        for name in names:
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
