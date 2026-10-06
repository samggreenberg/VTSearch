#!/usr/bin/env python3
"""Build the static site that gives the newest State of the App report a stable URL.

Each kind of review (Binary Photo, Region Photo, Document Logo, ...) gets two
links that always point at its newest report:

    https://samggreenberg.github.io/VTSearch/sota/binary-photo/
        redirects to GitHub's own rendering of the newest REPORT.md
    https://samggreenberg.github.io/VTSearch/sota/binary-photo/viewer.html
        that report's interactive viewer, served as a page rather than as
        source, so it opens in the browser with no download

and ``https://samggreenberg.github.io/VTSearch/sota/`` lists every kind with
its earlier reports.

The slide decks get their stable URL from a rolling release
(``scripts/publish-slides.sh``), but that cannot work here: GitHub serves a
release asset as ``application/octet-stream``, so an ``.html`` downloads
instead of rendering. Pages is the one GitHub surface that serves HTML as HTML.
Nothing is copied into the site except the viewers; the reports stay on GitHub,
which renders their markdown, figures and relative links as it always has.

**What counts as a report** is read from the tree, so a new review needs no
registration here. A directory under ``docs/experiments/`` qualifies when its
name starts with a ``YYYY-MM-DD-`` date and its ``REPORT.md`` opens with::

    # State of the App: <Kind> — <date>

The kind is the text after the colon, up to the first `` — ``, ``,`` or `` (``,
so ``Document Logo, round 3`` files under Document Logo. The date prefix
decides which report is newest. A write-up titled any other way (the per-floor
control is "State of the App per floor: ...") is reported as skipped, and
``tests_lib/meta/test_publish_sota_site.py`` fails on a skip it does not
expect, so a review with a mistyped title cannot silently fall off the links.

The viewer is ``viewer.html`` beside ``REPORT.md``. When the newest report of a
kind has none, its ``viewer.html`` link serves a short page saying so, rather
than an older report's viewer under a URL that promises the newest.

.github/workflows/publish-sota.yml runs this on every push to ``main`` that
touches a report, and deploys the result. To look at the site locally::

    python scripts/publish-sota-site.py --out _site
    python -m http.server -d _site      # then open http://localhost:8000/sota/
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / "docs" / "experiments"

_DATED_DIR = re.compile(r"^(\d{4}-\d{2}-\d{2})-")
_TITLE = re.compile(r"^#\s+State of the App:\s*(?P<rest>.+?)\s*$")
# The kind ends at the first em dash, comma or opening parenthesis.
_KIND_END = re.compile(r"\s+[—–-]\s+|,|\s+\(")

# A kind that was renamed after its first report keeps one set of links. The
# first Document Logo round (2026-10-01) carried the path's old name.
RENAMED = {"Structural Document": "Document Logo"}


@dataclass(frozen=True)
class Report:
    kind: str
    date: str
    directory: str  # name of the report's directory under docs/experiments/
    title: str  # the H1, without the leading '# '
    has_viewer: bool


@dataclass
class Series:
    kind: str
    reports: list[Report] = field(default_factory=list)  # newest first

    @property
    def slug(self) -> str:
        return slugify(self.kind)

    @property
    def latest(self) -> Report:
        return self.reports[0]


def slugify(kind: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", kind.lower()).strip("-")


def parse_kind(first_line: str) -> str | None:
    """The review kind a REPORT.md's first line names, or None if it names none."""
    m = _TITLE.match(first_line)
    if not m:
        return None
    kind = _KIND_END.split(m.group("rest"), maxsplit=1)[0].strip()
    if not kind:
        return None
    return RENAMED.get(kind, kind)


def discover(experiments: Path) -> tuple[list[Series], list[str]]:
    """Every State of the App series under ``experiments``, and the skipped dirs.

    A directory is *skipped* when its name says it is a State of the App write-up
    but its title does not name a kind, so the caller can tell a deliberate
    side-report from a review whose title was mistyped.
    """
    by_kind: dict[str, list[Report]] = {}
    skipped: list[str] = []
    for report_md in sorted(experiments.glob("*/REPORT.md")):
        directory = report_md.parent.name
        dated = _DATED_DIR.match(directory)
        with report_md.open(encoding="utf-8") as fh:
            first_line = fh.readline().rstrip("\n")
        kind = parse_kind(first_line)
        if kind is None or dated is None:
            if "state-of-the-app" in directory:
                skipped.append(directory)
            continue
        by_kind.setdefault(kind, []).append(
            Report(
                kind=kind,
                date=dated.group(1),
                directory=directory,
                title=first_line.lstrip("#").strip(),
                has_viewer=(report_md.parent / "viewer.html").is_file(),
            )
        )
    series = [
        Series(kind, sorted(reports, key=lambda r: (r.date, r.directory), reverse=True))
        for kind, reports in sorted(by_kind.items())
    ]
    slugs = [s.slug for s in series]
    clashes = sorted({s for s in slugs if slugs.count(s) > 1})
    if clashes:
        raise SystemExit(f"two kinds share a URL slug: {clashes}; retitle one or add it to RENAMED")
    return series, skipped


def report_url(repo: str, ref: str, report: Report) -> str:
    return f"https://github.com/{repo}/blob/{ref}/docs/experiments/{report.directory}/REPORT.md"


_STYLE = """
  :root { color-scheme: light dark; --bg: #fcfcfb; --ink: #0b0b0b; --muted: #6b6a66;
          --rule: #e1e0d9; --link: #2a78d6; }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #141413; --ink: #ececea; --muted: #9a9993; --rule: #2e2e2c; --link: #6aa6ef; }
  }
  body { background: var(--bg); color: var(--ink); margin: 0;
         font: 16px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 46rem; margin: 3rem auto; padding: 0 1.5rem; }
  h1 { font-size: 1.6rem; margin: 0 0 .25rem; }
  h2 { font-size: 1.15rem; margin: 2rem 0 .25rem; }
  a { color: var(--link); }
  .muted, footer { color: var(--muted); }
  .links a { margin-right: 1.25rem; font-weight: 600; }
  ul.earlier { margin: .5rem 0 0; padding-left: 1.25rem; }
  footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--rule); font-size: .85rem; }
"""


def _page(title: str, body: str, *, redirect: str | None = None) -> str:
    head = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>{html.escape(title)}</title>",
    ]
    if redirect:
        url = html.escape(redirect, quote=True)
        head += [f'<meta http-equiv="refresh" content="0; url={url}">', f'<link rel="canonical" href="{url}">']
    head += [f"<style>{_STYLE}</style>", "</head>"]
    return "\n".join([*head, "<body>", "<main>", body, "</main>", "</body>", "</html>", ""])


def _redirect_page(report: Report, target: str) -> str:
    link = f'<a href="{html.escape(target, quote=True)}">{html.escape(report.title)}</a>'
    return _page(report.title, f"<p>Opening {link} on GitHub…</p>", redirect=target)


def _no_viewer_page(series: Series, repo: str, ref: str) -> str:
    report = series.latest
    target = html.escape(report_url(repo, ref, report), quote=True)
    body = (
        f"<h1>No viewer for this report</h1>\n"
        f"<p>The newest State of the App: {html.escape(series.kind)} report "
        f"({html.escape(report.date)}) has no <code>viewer.html</code> beside its "
        f"<code>REPORT.md</code>, so there is no interactive view to show.</p>\n"
        f'<p><a href="{target}">Read the report</a> · <a href="../">all reports</a></p>'
    )
    return _page(f"{series.kind}: no viewer", body)


def _index_page(series: list[Series], repo: str, ref: str, sha: str, built: str) -> str:
    parts = [
        "<h1>State of the App</h1>",
        '<p class="muted">The newest review of each kind. These links follow '
        f"<code>{html.escape(ref)}</code>, so they move when a release carries a new report there.</p>",
    ]
    for s in series:
        latest = s.latest
        viewer = '<a href="{0}/viewer.html">Viewer</a>'.format(s.slug)
        if not latest.has_viewer:
            viewer = '<span class="muted">no viewer</span>'
        parts += [
            f"<h2>{html.escape(s.kind)}</h2>",
            f'<p class="muted">{html.escape(latest.date)} · <code>{html.escape(latest.directory)}</code></p>',
            f'<p class="links"><a href="{s.slug}/">Report</a>{viewer}</p>',
        ]
        if len(s.reports) > 1:
            items = "".join(
                f'<li><a href="{html.escape(report_url(repo, ref, r), quote=True)}">{html.escape(r.date)}</a></li>'
                for r in s.reports[1:]
            )
            parts.append(f'<details><summary class="muted">Earlier</summary><ul class="earlier">{items}</ul></details>')
    commit = html.escape(sha)
    if re.fullmatch(r"[0-9a-f]{7,40}", sha):
        commit = f'<a href="https://github.com/{html.escape(repo)}/commit/{sha}"><code>{sha[:9]}</code></a>'
    parts.append(
        f"<footer>Built {html.escape(built)} from {commit} by <code>scripts/publish-sota-site.py</code>.</footer>"
    )
    return _page("State of the App", "\n".join(parts))


def build(out: Path, *, experiments: Path, repo: str, ref: str, sha: str) -> tuple[list[Series], list[str]]:
    """Write the site to ``out`` (replacing it) and return what it published."""
    series, skipped = discover(experiments)
    if not series:
        raise SystemExit(f"no State of the App reports found under {experiments}")
    if out.exists():
        shutil.rmtree(out)
    sota = out / "sota"
    sota.mkdir(parents=True)
    built = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    (out / "index.html").write_text(
        _page("State of the App", '<p><a href="sota/">State of the App</a></p>', redirect="sota/")
    )
    (sota / "index.html").write_text(_index_page(series, repo, ref, sha, built))
    for s in series:
        d = sota / s.slug
        d.mkdir()
        (d / "index.html").write_text(_redirect_page(s.latest, report_url(repo, ref, s.latest)))
        if s.latest.has_viewer:
            shutil.copyfile(experiments / s.latest.directory / "viewer.html", d / "viewer.html")
        else:
            (d / "viewer.html").write_text(_no_viewer_page(s, repo, ref))
    return series, skipped


def _head_sha() -> str:
    try:
        return subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "rev-parse", "HEAD"],  # noqa: S607 - git from PATH, as every script here
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, type=Path, help="directory to write the site to (replaced)")
    ap.add_argument("--experiments", type=Path, default=EXPERIMENTS, help="where the report directories live")
    ap.add_argument("--repo", default="samggreenberg/VTSearch", help="owner/name the report links point at")
    ap.add_argument("--ref", default="main", help="branch the report links point at")
    ap.add_argument("--sha", default=None, help="commit the site is built from (default: HEAD)")
    args = ap.parse_args(argv)

    series, skipped = build(
        args.out, experiments=args.experiments, repo=args.repo, ref=args.ref, sha=args.sha or _head_sha()
    )
    for s in series:
        viewer = "viewer" if s.latest.has_viewer else "no viewer"
        print(f"sota/{s.slug}/  ->  {s.latest.directory}  ({viewer}; {len(s.reports)} report(s))")
    for directory in skipped:
        print(f"skipped {directory}: its title names no kind", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
