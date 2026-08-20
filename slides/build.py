#!/usr/bin/env python3
"""Assemble a Marp deck from a .deck manifest and a library of slide fragments.

A manifest names slides in order; this script concatenates them with `---`
separators, prepends Marp front matter, and preflights everything that would
otherwise fail late (missing fragment, missing figure, a stray `---` inside a
fragment that would silently split one slide into two).

    ./build.py scale26-review        # -> _build/scale26-review.md
    ./build.py --all
    ./build.py --check               # preflight only, write nothing
    ./build.py --list                # decks, slide counts, unused fragments
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
# A line that is exactly a Marp slide separator.
RULE_RE = re.compile(r"^-{3,}\s*$")

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


def check_fragment(name: str, text: str, problems: list[str]) -> None:
    for lineno, line in enumerate(text.splitlines(), 1):
        if RULE_RE.match(line):
            problems.append(
                f"fragments/{name}.md:{lineno}: bare `---` splits this fragment into two "
                f"slides; use `***` for a horizontal rule"
            )
    for match in IMAGE_RE.finditer(text):
        target = match.group(1)
        if target.startswith(("http://", "https://", "data:")):
            continue
        if not (ROOT / target).exists():
            line = text[: match.start()].count("\n") + 1
            problems.append(f"fragments/{name}.md:{line}: figure not found: {target}")


def assemble(deck: str, write: bool) -> list[str]:
    """Preflight one deck; write _build/<deck>.md unless write=False.

    Returns the list of problems found (empty means the deck is clean).
    """
    manifest = DECKS / f"{deck}.deck"
    if not manifest.exists():
        raise DeckError(f"no such deck: {manifest.relative_to(ROOT)}")

    front, names = parse_manifest(manifest)
    problems: list[str] = []
    bodies: list[str] = []

    for name in names:
        fragment = SLIDES / f"{name}.md"
        if not fragment.exists():
            problems.append(f"{deck}.deck: missing fragment: fragments/{name}.md")
            continue
        text = fragment.read_text().strip("\n")
        check_fragment(name, text, problems)
        bodies.append(text)

    if problems or not write:
        return problems

    merged = dict(DEFAULT_FRONTMATTER)
    merged.update(front)
    header = "\n".join(f"{k}: {yaml_scalar(v)}" for k, v in merged.items())

    BUILD.mkdir(exist_ok=True)
    out = BUILD / f"{deck}.md"
    body = rewrite_images("\n\n---\n\n".join(bodies))
    out.write_text(f"---\n{header}\n---\n\n{body}\n")

    # Verify against the emitted file, not the sources: the paths that matter
    # are the ones Marp will resolve, from _build/.
    for match in IMAGE_RE.finditer(body):
        target = match.group(1)
        if target.startswith(("http://", "https://", "data:")):
            continue
        if not (BUILD / target).exists():
            problems.append(f"{deck}.deck: unresolvable from _build/: {target}")
    if problems:
        out.unlink()
        return problems

    print(f"built {out.relative_to(ROOT)}  ({len(bodies)} slides)")
    return []


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
        print(f"  {deck:<24} {len(names):>2} slides")

    orphans = sorted(p.stem for p in SLIDES.glob("*.md") if p.stem not in used)
    if orphans:
        print("\nfragments in no deck (fine — a library holds spares):")
        for name in orphans:
            print(f"  fragments/{name}.md")


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("deck", nargs="?", help="deck name (without .deck)")
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
            problems += assemble(deck, write=not args.check)
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
