#!/usr/bin/env python
"""Render a look-at-it report for a built DocMarks corpus.

    python make_report.py --corpus <dir> --out docs/experiments/<date>-docmarks/report.html

Counts tell you a corpus is the right size; only pictures tell you it is the
right corpus.  This renders both, with the images inline so the file is one
self-contained artifact that survives being emailed, archived, or opened six
months later on a machine with no access to ``/expscratch``.

Sections, in the order a reader needs them:

1. **What is in it** — pages per source, marks per kind, class inventory.
2. **Full pages** — marks boxed in situ, because the single most important
   property of this task is how *small* the target is against the page, and no
   crop conveys that.
3. **The Goods** — every roster/candidate class as a strip of its own instances,
   so within-class variation is visible at a glance.
4. **The Bads** — distractor pages, and specifically the near-misses: other
   marks that are not the class being searched for.
5. **Scale** — the mark-size distribution against the ~32 px structural floor
   the 2026-07-13 study measured.

Images are JPEG-compressed and size-capped; a 200k-page corpus still produces a
report of a few MB because it samples rather than dumps.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import docmarks_config as cfg  # noqa: E402
from sources._common import Page, read_manifest  # noqa: E402

#: Thumbnail long side, in px, per figure kind.  Kept small deliberately: the
#: report is meant to be scrolled, and a 4000px scan inlined at full size makes
#: a file nobody can open.
THUMB_INSTANCE = 150
THUMB_PAGE = 460
JPEG_QUALITY = 72


# --------------------------------------------------------------------------
# Image helpers
# --------------------------------------------------------------------------


def _b64(image: Any, *, long_side: int, quality: int = JPEG_QUALITY) -> str:
    """A PIL image as an inline ``data:`` URI, downscaled and JPEG-compressed."""
    im = image.convert("RGB").copy()
    im.thumbnail((long_side, long_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _crop(page: Page, box: tuple[int, int, int, int], pad_frac: float = 0.12) -> Any:
    from PIL import Image

    x, y, w, h = box
    pad = int(round(max(w, h) * pad_frac))
    with Image.open(page.path) as im:
        return im.convert("RGB").crop(
            (max(0, x - pad), max(0, y - pad), min(im.width, x + w + pad), min(im.height, y + h + pad))
        )


def _page_with_boxes(page: Page, *, highlight: Optional[str] = None) -> Any:
    """The whole page with every mark outlined, so scale is visible in context."""
    from PIL import Image, ImageDraw

    with Image.open(page.path) as src:
        im = src.convert("RGB")
    draw = ImageDraw.Draw(im)
    width = max(2, int(round(max(im.size) / 400)))
    for mark in page.marks:
        if mark.area() <= 0:
            continue
        x, y, w, h = mark.box
        hot = highlight is not None and mark.class_id == highlight
        colour = (215, 25, 28) if hot else (70, 130, 200)
        draw.rectangle([x, y, x + w, y + h], outline=colour, width=width * (2 if hot else 1))
    return im


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

_CSS = """
:root { --ink:#1a1a1a; --dim:#666; --line:#ddd; --hot:#d7191c; --cool:#4682c8; --bg:#fff; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
  font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; }
main { max-width:1180px; margin:0 auto; padding:32px 28px 80px; }
h1 { font-size:30px; margin:0 0 4px; letter-spacing:-0.01em; }
h2 { font-size:22px; margin:44px 0 6px; padding-top:18px; border-top:2px solid var(--ink); letter-spacing:-0.01em; }
h3 { font-size:16px; margin:26px 0 8px; }
.sub { color:var(--dim); margin:0 0 22px; }
p { max-width:74ch; }
.note { color:var(--dim); font-size:13.5px; max-width:74ch; }
table { border-collapse:collapse; margin:14px 0 20px; font-size:14px; }
th,td { text-align:left; padding:5px 14px 5px 0; border-bottom:1px solid var(--line); }
th { font-weight:600; border-bottom:1.5px solid var(--ink); white-space:nowrap; }
td.num, th.num { text-align:right; }
code { font:13px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace; background:#f4f4f4;
  padding:1px 4px; border-radius:3px; }
.strip { display:flex; flex-wrap:wrap; gap:6px; margin:8px 0 4px; }
.strip img { height:96px; width:auto; border:1px solid var(--line); border-radius:2px; background:#fafafa; }
.pages { display:flex; flex-wrap:wrap; gap:14px; margin:14px 0 6px; }
.pages figure { margin:0; width:300px; }
.pages img { width:100%; border:1px solid var(--line); }
.pages figcaption { font-size:12px; color:var(--dim); margin-top:4px; }
.cls { border-left:3px solid var(--line); padding:2px 0 2px 14px; margin:22px 0; }
.cls h3 { margin:0 0 2px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:14px; }
.cls .meta { color:var(--dim); font-size:13px; margin:0 0 6px; }
.bar { height:9px; background:var(--cool); display:inline-block; vertical-align:middle; border-radius:1px; }
.warn { background:#fff6f5; border-left:3px solid var(--hot); padding:10px 14px; margin:16px 0; font-size:14px; }
.warn strong { color:var(--hot); }
"""


def _esc(text: Any) -> str:
    return html.escape(str(text))


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]], numeric: Sequence[int] = ()) -> str:
    num = set(numeric)
    head = "".join(
        f'<th class="num">{_esc(h)}</th>' if i in num else f"<th>{_esc(h)}</th>" for i, h in enumerate(headers)
    )
    body = []
    for row in rows:
        cells = "".join(
            f'<td class="num">{_esc(c)}</td>' if i in num else f"<td>{_esc(c)}</td>" for i, c in enumerate(row)
        )
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


def section_overview(pages: Sequence[Page], classes: dict[str, Any], report: dict[str, Any]) -> str:
    by_source = Counter(p.source for p in pages)
    marked = Counter(p.source for p in pages if any(m.class_id for m in p.marks))
    kinds = Counter(m.kind for p in pages for m in p.marks)
    provenances = Counter(m.provenance for p in pages for m in p.marks)

    out = ["<h2>What is in it</h2>"]
    out.append(
        _table(
            ["source", "pages", "pages carrying a labelled mark", "share"],
            [[s, f"{by_source[s]:,}", f"{marked[s]:,}", f"{marked[s] / by_source[s]:.0%}"] for s in sorted(by_source)],
            numeric=(1, 2, 3),
        )
    )
    out.append("<h3>Marks by kind, and by how the label was arrived at</h3>")
    out.append(
        _table(
            ["kind", "marks"], [[k, f"{v:,}"] for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])], numeric=(1,)
        )
    )
    out.append(
        _table(
            ["provenance", "marks", "what it means"],
            [
                [k, f"{v:,}", _PROVENANCE_MEANING.get(k, "")]
                for k, v in sorted(provenances.items(), key=lambda kv: -kv[1])
            ],
            numeric=(1,),
        )
    )
    verified = sum(1 for m in classes.values() if m.get("audit", {}).get("membership_verified"))
    out.append(
        f"<p><strong>{len(classes)}</strong> class(es); <strong>{verified}</strong> with every instance "
        f"hand-verified. Tier budget reached: "
        + ", ".join(f"<code>{t}</code>={report.get('tier_cumulative', {}).get(t, 0):,}" for t in cfg.TIER_ORDER)
        + ".</p>"
    )
    if not verified:
        out.append(
            '<div class="warn"><strong>Nothing here is verified yet.</strong> Every class below was proposed '
            "by clustering, which is exactly the step a previous study skipped when it published per-class "
            "numbers on a derived inventory. Treat this report as a look at what the pipeline produced, not "
            "as ground truth.</div>"
        )
    return "\n".join(out)


_PROVENANCE_MEANING = {
    "gt": "a box the source itself ships",
    "clustered": "identity derived here by clustering — a proposal until audited",
    "clustered_band": "identity derived from a coarse top-of-page strip, so the box locates a region not a mark",
    "candidate": "a pool member with no identity yet",
    "synthetic": "pasted at a known position — true by construction",
}


def section_full_pages(pages: Sequence[Page], n: int, seed: int) -> str:
    """Whole pages with marks boxed — the only way to see how small the target is."""
    rng = random.Random(seed)
    marked = [p for p in pages if any(m.area() > 0 and m.class_id for m in p.marks)]
    picks = rng.sample(marked, min(n, len(marked))) if marked else []

    figs = []
    for page in picks:
        biggest = max((m for m in page.marks if m.area() > 0), key=lambda m: m.area())
        frac = biggest.area() / float(page.width * page.height)
        figs.append(
            f'<figure><img src="{_b64(_page_with_boxes(page, highlight=biggest.class_id), long_side=THUMB_PAGE)}">'
            f"<figcaption>{_esc(page.page_id)} — largest mark is "
            f"{biggest.longest_side()}px, {frac:.2%} of the page</figcaption></figure>"
        )
    return (
        "<h2>Whole pages, marks boxed</h2>"
        "<p>The defining property of this task is the ratio between the target and the page. "
        "Red is the largest labelled mark; blue is every other mark the source annotates. "
        "A crop gallery makes these look easy; this is what the matcher actually sees.</p>"
        f'<div class="pages">{"".join(figs)}</div>'
    )


def section_goods(pages: Sequence[Page], classes: dict[str, Any], max_classes: int, max_inst: int) -> str:
    """Each class as a strip of its own instances."""
    by_id = {p.page_id: p for p in pages}
    ordered = sorted(classes.items(), key=lambda kv: -kv[1]["n_instances"])[:max_classes]

    blocks = []
    for class_id, meta in ordered:
        thumbs = []
        for page_id in meta["page_ids"][:max_inst]:
            page = by_id.get(page_id)
            if page is None:
                continue
            for mark in page.marks:
                if mark.class_id == class_id:
                    thumbs.append(f'<img src="{_b64(_crop(page, mark.box), long_side=THUMB_INSTANCE)}">')
                    break
        if not thumbs:
            continue
        px = meta.get("median_mark_px")
        extra = f"median {px}px" if px else f"located by {meta.get('located_by', '?')}"
        shown = "" if len(meta["page_ids"]) <= max_inst else f" (first {max_inst} of {meta['n_instances']})"
        blocks.append(
            f'<div class="cls"><h3>{_esc(class_id)}</h3>'
            f'<p class="meta">{meta["n_instances"]} instance(s) · {extra} · '
            f"{_esc(meta.get('kind', '?'))} · {_esc(', '.join(meta.get('provenance', [])))}{shown}</p>"
            f'<div class="strip">{"".join(thumbs)}</div></div>'
        )
    return (
        "<h2>The Goods — every class, every instance</h2>"
        "<p>One row per class. What to look for: do all the crops in a row show the "
        "<em>same</em> mark? A row that mixes two marks is an over-merged cluster; two rows showing the "
        "same mark are a split that should be merged. Both are what the "
        "<code>cluster</code> and <code>confusable</code> audits exist to catch.</p>" + "".join(blocks)
    )


def section_bads(pages: Sequence[Page], classes: dict[str, Any], n: int, seed: int) -> str:
    """Distractors, and the near-miss marks that make them hard."""
    rng = random.Random(seed + 1)
    unlabelled = [p for p in pages if not any(m.class_id for m in p.marks)]
    picks = rng.sample(unlabelled, min(n, len(unlabelled))) if unlabelled else []
    thumbs = "".join(
        f'<figure><img src="{_b64(_page_with_boxes(p), long_side=THUMB_PAGE)}">'
        f"<figcaption>{_esc(p.page_id)} · {_esc(p.source)}</figcaption></figure>"
        for p in picks
    )

    # Near misses: a mark from some *other* class is the hardest negative there
    # is, because it is a real mark on a real page and only its identity differs.
    by_id = {p.page_id: p for p in pages}
    near = []
    for class_id, meta in sorted(classes.items())[:12]:
        page = by_id.get(meta["page_ids"][0]) if meta["page_ids"] else None
        if page is None:
            continue
        for mark in page.marks:
            if mark.class_id == class_id:
                near.append(
                    f'<img src="{_b64(_crop(page, mark.box), long_side=THUMB_INSTANCE)}" title="{_esc(class_id)}">'
                )
                break

    by_source = Counter(p.source for p in unlabelled)
    return (
        "<h2>The Bads — what the haystack looks like</h2>"
        f"<p><strong>{len(unlabelled):,}</strong> pages carry no labelled mark and serve as distractors: "
        + ", ".join(f"{v:,} from <code>{k}</code>" for k, v in sorted(by_source.items()))
        + ". These need no labels — only to be <em>safe to score against</em>, which the contamination "
        "rules decide.</p>"
        f'<div class="pages">{thumbs}</div>'
        "<h3>Near misses — the negatives that actually matter</h3>"
        "<p>A blank page is a trivial negative. The hard one is a real mark on a real scan that simply "
        "is not the mark being searched for. Each of these is a positive for one class and a negative "
        "for every other, which is why exhaustive verification turns a same-source page from a "
        "contamination risk into the most valuable negative available.</p>"
        f'<div class="strip">{"".join(near)}</div>'
    )


def section_scale(pages: Sequence[Page], classes: dict[str, Any]) -> str:
    """Mark size against the measured structural floor."""
    sides = [m.longest_side() for p in pages for m in p.marks if m.class_id and m.area() > 0]
    if not sides:
        return ""
    bands = [(0, 32), (32, 64), (64, 128), (128, 256), (256, 512), (512, 10**9)]
    counts = [sum(1 for s in sides if lo <= s < hi) for lo, hi in bands]
    peak = max(counts) or 1
    labels = ["<32", "32-64", "64-128", "128-256", "256-512", "512+"]
    rows = [
        [
            lab,
            f"{c:,}",
            f"{c / len(sides):.0%}",
            f'<span class="bar" style="width:{int(200 * c / peak)}px"></span>',
        ]
        for lab, c in zip(labels, counts)
    ]
    below = counts[0] / len(sides)

    fracs = sorted(m.area() / float(p.width * p.height) for p in pages for m in p.marks if m.class_id and m.area() > 0)
    median_frac = fracs[len(fracs) // 2]

    table = "<table><thead><tr><th>longest side (px)</th><th class='num'>marks</th>"
    table += "<th class='num'>share</th><th></th></tr></thead><tbody>"
    for lab, c, share, bar in rows:
        table += f"<tr><td>{lab}</td><td class='num'>{c}</td><td class='num'>{share}</td><td>{bar}</td></tr>"
    table += "</tbody></table>"

    return (
        "<h2>Scale — how big are these things?</h2>"
        "<p>The 2026-07-13 study measured a hard floor near <strong>32px</strong>: below it no structural "
        "pipeline recovered anything, tiling and backends alike. A class built from sub-floor instances "
        "measures that floor rather than the method, which is why the builder refuses one by default.</p>"
        + table
        + f"<p class='note'>{below:.0%} of labelled marks fall below the 32px floor. "
        f"The median mark covers {median_frac:.2%} of its page.</p>"
    )


def section_per_class_table(classes: dict[str, Any]) -> str:
    rows = []
    for class_id, meta in sorted(classes.items(), key=lambda kv: -kv[1]["n_instances"]):
        audit = meta.get("audit", {})
        rows.append(
            [
                class_id,
                meta.get("kind", "?"),
                f"{meta['n_instances']:,}",
                meta.get("median_mark_px") or "—",
                meta.get("located_by", "box"),
                "yes" if audit.get("membership_verified") else "no",
                len(meta.get("distinct_from", [])),
                "; ".join(meta.get("caveats", [])) or "—",
            ]
        )
    return "<h2>Per-class inventory</h2>" + _table(
        ["class", "kind", "instances", "median px", "located by", "verified", "known-distinct from", "caveats"],
        rows,
        numeric=(2, 3, 6),
    )


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def build_html(
    pages: list[Page],
    classes: dict[str, Any],
    report: dict[str, Any],
    *,
    title: str,
    max_classes: int,
    max_instances: int,
    n_full_pages: int,
    n_bad_pages: int,
    seed: int,
) -> str:
    parts = [
        section_overview(pages, classes, report),
        section_full_pages(pages, n_full_pages, seed),
        section_goods(pages, classes, max_classes, max_instances),
        section_bads(pages, classes, n_bad_pages, seed),
        section_scale(pages, classes),
        section_per_class_table(classes),
    ]
    sources = ", ".join(sorted({p.source for p in pages}))
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{_esc(title)}</title><style>{_CSS}</style></head><body><main>"
        f"<h1>{_esc(title)}</h1>"
        f"<p class='sub'>{len(pages):,} pages · {len(classes)} classes · sources: {_esc(sources)} · "
        f"built by <code>scripts/experiments/docmarks/</code>, rendered by <code>make_report.py</code></p>"
        + "\n".join(parts)
        + "</main></body></html>"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default="DocMarks v0 — what the corpus looks like")
    ap.add_argument("--max-classes", type=int, default=40)
    ap.add_argument("--max-instances", type=int, default=20)
    ap.add_argument("--full-pages", type=int, default=12)
    ap.add_argument("--bad-pages", type=int, default=12)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args(argv)

    pages = list(read_manifest(args.corpus / "corpus.jsonl"))
    classes_path = args.corpus / "classes.json"
    classes = json.loads(classes_path.read_text(encoding="utf-8")) if classes_path.exists() else {}
    report_path = args.corpus / "build_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}

    if not pages:
        raise SystemExit(f"no pages in {args.corpus / 'corpus.jsonl'}")

    doc = build_html(
        pages,
        classes,
        report,
        title=args.title,
        max_classes=args.max_classes,
        max_instances=args.max_instances,
        n_full_pages=args.full_pages,
        n_bad_pages=args.bad_pages,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(doc, encoding="utf-8")
    print(f"wrote {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
