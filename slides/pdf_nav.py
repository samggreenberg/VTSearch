#!/usr/bin/env python3
"""Write a rendered deck's bookmarks and page labels into its PDF.

    ./pdf_nav.py _out/hold-the-line.pdf _build/hold-the-line.nav.json
    ./pdf_nav.py _out/hold-the-line.speaker.pdf _build/hold-the-line.speaker.nav.json \\
        --probe _build/hold-the-line.probe.pdf

`render.sh` runs this after every PDF render. Chromium, which Marp prints
through, writes a link for every `<a>` it draws — which is all the audience
deck's clickable outline needs — but has no way to be told about the two things
a reader navigates a long PDF by: **bookmarks** (the sections, and the slides
in each) and **page labels** (so the viewer's page box reads `17c`, the
address printed on the slide, not `95`). `build.py` works both out from the
manifest, since they are facts about fragments that no rendered page carries,
and leaves them in `_build/<deck>[...].nav.json` for this script to write in.
It also sets the PDF to open with the bookmarks showing.

The speaker deck needs one thing more. Its outline is a *picture* — a PNG of
the audience slide beside the notes — so the links the audience deck gets for
free have to be laid over it by hand. `--probe` names a PDF of the outline
slides alone (`build.py`'s `probe_bodies`), whose links Chromium has measured
and which point at speaker pages; each one is scaled from its slide into the
miniature drawn on the matching speaker page.

Needs PyMuPDF, which the project already depends on (the `agpl` extra), or
`pip install pymupdf`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import pymupdf
except ImportError:
    sys.exit(
        "pdf_nav.py: PyMuPDF is not installed, so the deck cannot get its bookmarks and page labels.\n"
        "  It is in the project's `agpl` extra, or: pip install pymupdf"
    )

# The probe's links name their speaker page with this scheme; build.py's
# `PROBE_SCHEME` writes it.
PROBE_RE = re.compile(r"^vtsnav:(\d+)$")


def page_labels(labels: list[str]) -> list[dict[str, object]]:
    """One labelling range per page, each a bare prefix with no counter.

    A range normally runs a counter (`D` for 1, 2, 3 …) from a start page, but
    this deck's addresses are not a counter — the outline is `1a` on page 2 and
    `1d` on page 76 — so every page is its own range whose prefix is the whole
    label.
    """
    return [{"startpage": index, "prefix": label, "style": "", "firstpagenum": 1} for index, label in enumerate(labels)]


def hero_rect(page: pymupdf.Page, slide: pymupdf.Rect) -> pymupdf.Rect:
    """Where a speaker page draws its slide miniature: the largest slide-shaped image on it.

    The frame strip's thumbnails are slide-shaped too, but every one is smaller
    than the miniature it sits under. "Slide-shaped" is not decoration: Chromium
    prints the miniature's drop shadow as an image of its own, bigger than the
    miniature, and taking the largest image of any shape lays every link over
    the shadow — off by a growing amount from the first line to the last.
    """
    aspect = slide.width / slide.height
    rects = [
        rect
        for rect in (pymupdf.Rect(info["bbox"]) for info in page.get_image_info())
        if rect.height and abs(rect.width / rect.height - aspect) < 0.01
    ]
    if not rects:
        raise SystemExit(f"pdf_nav.py: speaker page {page.number + 1} draws no slide miniature to link")
    return max(rects, key=lambda rect: rect.width * rect.height)


def link_miniatures(doc: pymupdf.Document, probe: pymupdf.Document, pages: list[int]) -> int:
    """Lay the probe's outline links over the miniatures they were measured on.

    *pages* is the speaker page each probe page is drawn on, in order. Returns
    how many links were written.
    """
    if probe.page_count != len(pages):
        raise SystemExit(
            f"pdf_nav.py: the probe has {probe.page_count} pages but the speaker deck expects {len(pages)} — "
            f"it is stale; re-render with ./render.sh <deck> pdf --speaker"
        )
    written = 0
    for probe_page, speaker_page in zip(probe, pages):
        slide = probe_page.rect
        hero = hero_rect(doc[speaker_page - 1], slide)
        # The slide's coordinates onto the miniature's: scale, then shift.
        onto = pymupdf.Matrix(hero.width / slide.width, 0, 0, hero.height / slide.height, hero.x0, hero.y0)
        for link in probe_page.get_links():
            target = PROBE_RE.match(link.get("uri") or "")
            if not target:
                continue
            doc[speaker_page - 1].insert_link(
                {
                    "kind": pymupdf.LINK_GOTO,
                    "from": link["from"] * onto,
                    "page": int(target.group(1)) - 1,
                    "to": pymupdf.Point(0, 0),
                }
            )
            written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("pdf", type=Path, help="the rendered deck, rewritten in place")
    parser.add_argument("nav", type=Path, help="the _build/<deck>[...].nav.json build.py wrote beside it")
    parser.add_argument("--probe", type=Path, help="the rendered outline probe (speaker deck only)")
    args = parser.parse_args()

    nav = json.loads(args.nav.read_text())
    doc = pymupdf.open(args.pdf)
    # A nav file from a different build of the deck would label and bookmark
    # the wrong pages without any error, so the one cheap invariant is checked.
    if doc.page_count != len(nav["labels"]):
        sys.exit(
            f"pdf_nav.py: {args.pdf} has {doc.page_count} pages but {args.nav} describes "
            f"{len(nav['labels'])} — they come from different builds of the deck"
        )

    doc.set_toc(nav["toc"], collapse=1)
    # Open with the bookmarks showing: a reader who never thinks to open the
    # sidebar never learns the deck has sections to jump between. Acrobat and
    # Preview honour it; Chrome's viewer ignores it, harmlessly.
    doc.set_pagemode("UseOutlines")
    doc.set_page_labels(page_labels(nav["labels"]))
    links = 0
    if nav.get("probe"):
        if args.probe is None:
            sys.exit(f"pdf_nav.py: {args.nav} has outline links to lay over miniatures; pass --probe")
        with pymupdf.open(args.probe) as probe:
            links = link_miniatures(doc, probe, nav["probe"])

    # An incremental save appends the new objects instead of rewriting a file
    # of a hundred megabytes of figures, which is both faster and leaves every
    # byte Chromium wrote exactly where it was.
    if doc.can_save_incrementally():
        doc.saveIncr()
    else:
        tmp = args.pdf.with_suffix(".nav-tmp.pdf")
        doc.save(tmp, garbage=1)
        doc.close()
        tmp.replace(args.pdf)
    sections = sum(1 for level, _title, _page in nav["toc"] if level == 1)
    extra = f", {links} outline links on the miniatures" if links else ""
    print(f"navigation: {len(nav['toc'])} bookmarks ({sections} top-level), {len(nav['labels'])} page labels{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
