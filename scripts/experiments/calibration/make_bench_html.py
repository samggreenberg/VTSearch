"""Build the study's reading copy: one self-contained HTML page from REPORT.md.

`docs/experiments/<study>/REPORT.md` is the archival record and renders fine on
GitHub. It is not a comfortable *read*: bitmap plots that cannot be zoomed, and a
figure that is only a link away from the claim it supports. This turns the same
markdown into the `docs/reports/` convention — a single file with every figure
inlined (SVG where one exists, so plots stay crisp at any zoom, base64 otherwise)
and every photograph embedded.

**Generated, not hand-written.** The narrative has exactly one source, so the
page cannot drift from the report the way a hand-maintained second copy would.
Re-run it after editing the report:

    python make_bench_html.py --report docs/experiments/2026-08-12-overview-bench/REPORT.md \\
        --out docs/reports/2026-08-17-overview-bench.html \\
        --subtitle "What a user actually gets from each shipped configuration"

The markdown subset is the one experiment reports actually use: headings, pipe
tables, fenced code, block quotes, bullet and numbered lists, images with an
italic caption underneath, and inline bold/italic/code/links. It is deliberately
not a general markdown implementation — a report that needs more should say so by
failing here rather than by rendering wrongly.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import re
from pathlib import Path

from PIL import Image

INLINE_CODE = re.compile(r"`([^`]+)`")
BOLD = re.compile(r"\*\*([^*]+)\*\*")
EM = re.compile(r"(?<![*\w])\*([^*\n]+)\*(?!\*)")
LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
IMAGE = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$")
HEADING = re.compile(r"^(#{1,4})\s+(.*)$")
TABLE_SEP = re.compile(r"^\|[\s:|-]+\|$")

MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif"}

#: Inline an SVG only while it is cheaper than a downscaled raster of the same
#: figure. The 170-trace spaghetti plot is ~700 KB of path data, which would put
#: the page over the repo's large-file limit on its own; everything else is a
#: line or bar plot where the vector version is both smaller and zoomable.
SVG_INLINE_MAX = 350_000
#: Rasters are re-encoded for the page: the committed copies stay full quality
#: for the markdown, and base64 costs a third on top of whatever is embedded.
RASTER_MAX_WIDTH = 1500
RASTER_QUALITY = 82  # matches the committed sheets, so embedding is not a second loss

CSS = """
:root {
  --bg: #fbfaf8; --surface: #ffffff; --ink: #1c1b19; --ink-2: #45423d; --ink-3: #d9d5cd;
  --accent: #8a5a2b; --accent-soft: #f3ece2; --quote: #f6f2ea;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #16171a; --surface: #1d1f23; --ink: #e9e6e1; --ink-2: #b3aea6; --ink-3: #3a3d43;
    --accent: #d9a25f; --accent-soft: #26262a; --quote: #212227;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1080px; margin: 0 auto; padding: 48px 28px 96px; }
header.masthead { border-bottom: 2px solid var(--ink-3); padding-bottom: 20px; margin-bottom: 8px; }
header.masthead .kicker { color: var(--accent); font-weight: 650; letter-spacing: .08em;
  text-transform: uppercase; font-size: 12px; }
header.masthead h1 { font-size: 34px; line-height: 1.2; margin: 8px 0 6px; }
header.masthead p { color: var(--ink-2); margin: 0; }
h1 { font-size: 27px; margin: 52px 0 14px; padding-bottom: 6px; border-bottom: 1px solid var(--ink-3); }
h2 { font-size: 21px; margin: 36px 0 10px; }
h3 { font-size: 17px; margin: 28px 0 8px; color: var(--ink); }
h4 { font-size: 15px; margin: 22px 0 6px; }
p { margin: 12px 0; }
a { color: var(--accent); }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .88em;
  background: var(--accent-soft); padding: .12em .35em; border-radius: 4px; }
pre { background: var(--surface); border: 1px solid var(--ink-3); border-radius: 8px;
  padding: 14px 16px; overflow-x: auto; }
pre code { background: none; padding: 0; font-size: .82em; line-height: 1.5; }
blockquote { margin: 18px 0; padding: 12px 18px; background: var(--quote);
  border-left: 3px solid var(--accent); border-radius: 0 8px 8px 0; }
blockquote p:first-child { margin-top: 0; } blockquote p:last-child { margin-bottom: 0; }
.tablewrap { overflow-x: auto; margin: 18px 0; }
table { border-collapse: collapse; width: 100%; font-size: 14px; background: var(--surface); }
th, td { border: 1px solid var(--ink-3); padding: 7px 10px; text-align: left; }
th { background: var(--accent-soft); font-weight: 650; }
tbody tr:nth-child(even) td { background: color-mix(in srgb, var(--surface) 92%, var(--ink) 8%); }
figure { margin: 26px 0; }
figure svg, figure img { display: block; width: 100%; height: auto;
  background: #fff; border: 1px solid var(--ink-3); border-radius: 8px; }
figure figcaption { color: var(--ink-2); font-size: 13.5px; margin-top: 8px; font-style: italic; }
ul, ol { padding-left: 24px; } li { margin: 7px 0; }
hr { border: 0; border-top: 1px solid var(--ink-3); margin: 40px 0; }
nav.toc { background: var(--surface); border: 1px solid var(--ink-3); border-radius: 10px;
  padding: 14px 20px; margin: 26px 0 8px; font-size: 14.5px; }
nav.toc strong { display: block; margin-bottom: 6px; font-size: 12px; letter-spacing: .07em;
  text-transform: uppercase; color: var(--ink-2); }
nav.toc ol { padding-left: 20px; margin: 0; } nav.toc li { margin: 3px 0; }
footer.foot { margin-top: 56px; padding-top: 18px; border-top: 1px solid var(--ink-3);
  color: var(--ink-2); font-size: 13.5px; }
"""


def slugify(text: str) -> str:
    """GitHub's heading-slug algorithm, so the report's own #anchors still work.

    Runs of hyphens are *preserved* — `## a / b — c` slugs to `a--b--c`, not
    `a-b-c` — which is the difference between an in-page link resolving and
    landing nowhere. Same rule as `scripts/check-docs.py`, which is what
    validates those links in the markdown.
    """
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)  # link text only
    text = text.replace("`", "").replace("*", "").lower()
    text = "".join(ch for ch in text if ch.isalnum() or ch in " -_")
    return text.strip().replace(" ", "-")


def inline(text: str) -> str:
    """Inline markdown → HTML, with code spans protected from further passes."""
    spans: list[str] = []

    def stash(match: re.Match[str]) -> str:
        spans.append(f"<code>{html.escape(match.group(1))}</code>")
        return f"\x00{len(spans) - 1}\x00"

    text = INLINE_CODE.sub(stash, text)
    text = html.escape(text)
    text = LINK.sub(lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', text)
    text = BOLD.sub(r"<strong>\1</strong>", text)
    text = EM.sub(r"<em>\1</em>", text)
    return re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], text)


def embed_figure(src: str, alt: str, base: Path) -> str:
    """Inline a small SVG (crisp at any zoom), else embed a downscaled raster."""
    path = (base / src).resolve()
    svg = path.with_suffix(".svg")
    if svg.exists() and svg.stat().st_size <= SVG_INLINE_MAX:
        markup = svg.read_text()
        markup = markup[markup.index("<svg") :]  # drop the XML prolog / DOCTYPE
        return re.sub(r'(<svg[^>]*?)\swidth="[^"]*"\sheight="[^"]*"', r"\1", markup, count=1)
    if not path.exists():
        return f"<p><em>missing figure: {html.escape(src)}</em></p>"
    return f'<img alt="{html.escape(alt)}" src="{data_uri(path)}">'


def data_uri(path: Path) -> str:
    """Downscale and JPEG-encode for embedding; PNG only when transparency matters."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.width > RASTER_MAX_WIDTH:
            im = im.resize((RASTER_MAX_WIDTH, round(im.height * RASTER_MAX_WIDTH / im.width)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=RASTER_QUALITY, optimize=True, progressive=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def convert(md: str, base: Path) -> tuple[str, list[tuple[str, str]]]:
    out: list[str] = []
    toc: list[tuple[str, str]] = []
    lines = md.split("\n")
    i = 0
    pending_figure = False  # last block was a figure, so an italic line is its caption

    def para(buf: list[str]) -> None:
        nonlocal pending_figure
        if not buf:
            return
        text = " ".join(buf).strip()
        stripped = text.startswith("*") and text.endswith("*") and not text.startswith("**")
        if pending_figure and stripped:
            out.append(f"<figcaption>{inline(text[1:-1])}</figcaption></figure>")
            pending_figure = False
        else:
            if pending_figure:
                out.append("</figure>")
                pending_figure = False
            out.append(f"<p>{inline(text)}</p>")
        buf.clear()

    buf: list[str] = []
    while i < len(lines):
        line = lines[i]

        if not line.strip():
            para(buf)
            i += 1
            continue

        image = IMAGE.match(line)
        if image:
            para(buf)
            if pending_figure:
                out.append("</figure>")
            out.append("<figure>" + embed_figure(image.group(2), image.group(1), base))
            pending_figure = True
            i += 1
            continue

        heading = HEADING.match(line)
        if heading:
            para(buf)
            if pending_figure:
                out.append("</figure>")
                pending_figure = False
            level, text = len(heading.group(1)), heading.group(2)
            slug = slugify(text)
            if level <= 2:
                toc.append((slug, re.sub(r"`|\*\*", "", text)))
            out.append(f'<h{level} id="{slug}">{inline(text)}</h{level}>')
            i += 1
            continue

        if line.startswith("```"):
            para(buf)
            i += 1
            code: list[str] = []
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            out.append(f"<pre><code>{html.escape(chr(10).join(code))}</code></pre>")
            continue

        if line.startswith("|"):
            para(buf)
            rows: list[str] = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            out.append(render_table(rows))
            continue

        if line.startswith("> "):
            para(buf)
            quote: list[str] = []
            while i < len(lines) and lines[i].startswith(">"):
                quote.append(lines[i].lstrip(">").strip())
                i += 1
            out.append(f"<blockquote><p>{inline(' '.join(quote))}</p></blockquote>")
            continue

        if re.match(r"^(-|\d+\.)\s", line):
            para(buf)
            ordered = not line.startswith("-")
            items: list[str] = []
            while i < len(lines):
                match = re.match(r"^(-|\d+\.)\s+(.*)$", lines[i])
                if match:
                    items.append(match.group(2))
                    i += 1
                elif lines[i].startswith("  ") and items:  # continuation of the last item
                    items[-1] += " " + lines[i].strip()
                    i += 1
                else:
                    break
            tag = "ol" if ordered else "ul"
            body = "".join(f"<li>{inline(x)}</li>" for x in items)
            out.append(f"<{tag}>{body}</{tag}>")
            continue

        if line.strip() == "---":
            para(buf)
            if pending_figure:
                out.append("</figure>")
                pending_figure = False
            out.append("<hr>")
            i += 1
            continue

        buf.append(line.strip())
        i += 1

    para(buf)
    if pending_figure:
        out.append("</figure>")
    return "\n".join(out), toc


def render_table(rows: list[str]) -> str:
    def cells(row: str) -> list[str]:
        return [c.strip() for c in row.strip().strip("|").split("|")]

    header = cells(rows[0])
    body_rows = rows[2:] if len(rows) > 1 and TABLE_SEP.match(rows[1]) else rows[1:]
    aligns = []
    if len(rows) > 1 and TABLE_SEP.match(rows[1]):
        for spec in cells(rows[1]):
            aligns.append("right" if spec.endswith(":") and not spec.startswith(":") else "left")
    aligns += ["left"] * (len(header) - len(aligns))

    head = "".join(f'<th style="text-align:{aligns[n]}">{inline(c)}</th>' for n, c in enumerate(header))
    body = ""
    for row in body_rows:
        tds = cells(row)
        body += (
            "<tr>"
            + "".join(
                f'<td style="text-align:{aligns[n] if n < len(aligns) else "left"}">{inline(c)}</td>'
                for n, c in enumerate(tds)
            )
            + "</tr>"
        )
    return f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--kicker", default="VTSearch · experiment report")
    args = ap.parse_args()

    report = Path(args.report)
    md = report.read_text()
    lines = md.split("\n")
    title = args.title or lines[0].lstrip("# ").strip()
    body_md = "\n".join(lines[1:])  # the H1 becomes the masthead

    body, toc = convert(body_md, report.parent)
    toc_html = "".join(f'<li><a href="#{slug}">{html.escape(text)}</a></li>' for slug, text in toc)
    toc_block = f'<nav class="toc"><strong>Contents</strong><ol>{toc_html}</ol></nav>'
    # The lead - everything before the first section heading - goes ABOVE the
    # contents. A reader's first screen should be the findings, not a 22-item
    # index of where the findings are.
    split = body.find("<h1 ")
    body = (body[:split] + toc_block + body[split:]) if split > 0 else toc_block + body
    generated_from = report.as_posix()

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<header class="masthead">
  <div class="kicker">{html.escape(args.kicker)}</div>
  <h1>{html.escape(title)}</h1>
  <p>{inline(args.subtitle)}</p>
</header>
{body}
<footer class="foot">
  Generated from <code>{html.escape(generated_from)}</code> by
  <code>scripts/experiments/calibration/make_bench_html.py</code> — edit the report,
  not this page, and re-run the script.
</footer>
</div>
</body>
</html>
"""
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # No trailing whitespace: the repo's pre-commit hook would strip it, and a
    # hook-rewritten page no longer matches what this script emits.
    out.write_text("\n".join(line.rstrip() for line in page.split("\n")))
    print(f"wrote {out} ({len(page) / 1_000_000:.2f} MB, {len(toc)} sections)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
