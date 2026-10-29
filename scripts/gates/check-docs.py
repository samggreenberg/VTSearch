#!/usr/bin/env python3
"""Documentation drift gate — links, anchors, code-path references, path leaks.

Nine invariants over the docs. All of them are *pure*: they compare the docs
against the repo as it exists right now, so there is nothing to re-pin and no
maintenance beyond the allowlists at the top of this file.

  1. LINK    every relative markdown link resolves to a real file or directory;
  2. ANCHOR  every ``#anchor`` resolves to a real heading in the target document,
             using GitHub's heading-slug algorithm;
  3. PATH    every backticked repo-path reference (``foo/bar.py``) resolves to a
             real file — with an allowlist for runtime-generated and
             deliberately-fictional example paths;
  4. LEAK    no absolute path out of somebody's home directory appears in
             committed prose;
  5. PLAN    every ``docs/plans/*.md`` path mentioned *anywhere in the repo* —
             including Python docstrings and shell comments — resolves, because
             plan files get deleted when their work ships and the citations that
             point at them are exactly the "why is this code shaped like this"
             pointers a maintainer follows;
  6. FENCE   no code fence is preceded by text on the same line (``-> ```json``
             renders as a paragraph, silently swallowing the block);
  7. STUDY   every ``docs/experiments/`` study directory is named
             ``YYYY-MM-DD-<slug>``, so ``ls`` alone sorts the archive
             chronologically. There is deliberately no index file to keep in
             step (#4138);
  8. GATES   the stage tables in ``docs/TESTING.md`` list exactly the gates
             ``run-tests.sh`` runs, each in the stage and lane the script puts
             it in. The table is a hand-written restatement of the script, and
             nothing else ties the two together;
  9. LINENO  no doc cites code by line number, in either spelling: a file name
             with a ``:NNN`` suffix, or a parenthesised ``(line NNN)``. A line
             number is wrong by the next edit to the file it points into, so
             cite the module and the symbol instead (#5290).

Dependency-free, imports nothing from the app, and reads each file once, so it
costs well under a second and can sit early in ``run-tests.sh``.

Run from anywhere:

    python scripts/gates/check-docs.py          # report failures, exit 1 if any
    python scripts/gates/check-docs.py -v       # also print per-check counts
"""

from __future__ import annotations

import argparse
import re
import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------
# Allowlists
# --------------------------------------------------------------------------
# Paths that are *supposed* not to exist in a clean checkout. Two kinds:
# runtime-generated files, and deliberately fictional paths in "here is how you
# would write your own plugin" examples. Matching is exact on the backticked
# token (after stripping any `:NNN` line suffix); a trailing `/` means the whole
# subtree is exempt.
ALLOWED_PATHS: dict[str, str] = {
    # Runtime-generated: written by a running app, absent from a clean checkout.
    "data/": "runtime data directory (gitignored); settings, detectors, datasets, uploads",
    "static/": "Angular build output (gitignored)",
    "vtsearch/_version.txt": "baked into Docker images only; gitignored",
    "frontend/node_modules/": "npm install output",
    "frontend/src/app/api/": "generated API client (npm run generate-api-client)",
    # Written by frontend/scripts/build-stamp.mjs from the prebuild/pretest
    # hooks, so it is absent until the frontend has been built once -- and this
    # gate runs before that build, so a cold container would otherwise fail here.
    "frontend/src/app/generated/": "generated build stamp (frontend prebuild/pretest hook); gitignored",
    # Marp build products. Both are gitignored, so they exist only on a machine
    # that has rendered a deck since checkout -- which made this gate pass on
    # the author's box and fail on a fresh clone.
    "slides/_build/": "assembled deck markdown (slides/build.py); gitignored",
    "slides/_out/": "rendered decks (slides/render.sh); gitignored",
    # Another branch's tree. Backticked because it reads as a path, but it
    # exists only on the evaluation-framework branch, never on dev or main.
    "scripts/sod/": "lives only on the evaluation-framework branch, not dev (#2847)",
    # Deliberately fictional paths in "here is how you would write your own"
    # examples. These must stay non-existent — that is the point of the example.
    "vtsearch/auth/my_provider.py": "fictional path in the auth extension guide",
    "vtscore/media/image/embedder_myclip.py": "fictional path in the embedder extension guide",
    "vtscore/media/text/embedder_minilm.py": "fictional path in the embedder extension guide",
    "vtscore/labels/sources/sqlite/": "fictional path in the labelset-source guide",
    # Retired code that finished study reports still cite as the code they ran.
    # The admin timing profile, its fitter, its tuning script and the studies
    # that measured them were removed with the dataset-import ETA (#4667); the
    # reports link the last tree that held them (80e7724ce).
    "vtscore/timing/fit.py": "timing-profile fitter, retired in #4667",
    "scripts/profiling/tune_timing_profile.py": "timing-profile tuning script, retired in #4667",
    "tests_lib/core/test_timing_branches.py": "timing-profile branch tests, retired in #4667",
    "scripts/experiments/drive_cold/": "timing-profile study scripts (#3521, #3595), retired in #4667",
}

# Documents exempt from the PATH check (only PATH — links, anchors, leaks and
# fences are still policed everywhere):
#
#   docs/plans/       describes work not yet done, so it names the files and
#                     report paths the work *will* create. Demanding those exist
#                     would invert the point of a plan.
#
# `docs/experiments/` and `scripts/experiments/` used to be exempt too, on the
# grounds that a run record cites its cluster scratch dir and the throwaway
# scripts that drove it. The scratch dirs are fine without an exemption — they
# are absolute paths, or relative ones like `agg/x.csv` whose first component is
# not a tracked top-level directory, so `looks_like_repo_path` never claims
# them. The *scripts* were the real hole, and they should not be throwaway: the
# overview-bench report cited an analysis script that had never been committed,
# which made its most checkable finding impossible to reproduce and the gate
# said nothing. A report may only cite analysis code that is in the tree.
PATH_SKIP_PREFIXES = ("docs/plans/",)

# Documents whose relative links resolve against a directory other than their
# own. Slide fragments are assembled into `slides/_build/<deck>.md` by
# `slides/build.py`, which repoints every image path as it goes, so a fragment
# writes `figs/x.webp` relative to `slides/` rather than to its own directory.
# Resolving those against `slides/fragments/` would be wrong rather than
# lenient, so the LINK check follows the convention instead of skipping it —
# these links stay fully policed, just against the base Marp will see.
LINK_BASE_OVERRIDES: dict[str, str] = {
    "slides/fragments/": "slides",
}

# Exempt from the PLAN check: this checker and its test are the one place in the
# tree that must name plan paths which deliberately do not resolve — an example
# in a comment, a synthetic fixture — because naming them is how the check is
# explained and proven.
PLAN_SELF_REFERENCE = frozenset(
    {
        "scripts/gates/check-docs.py",
        "tests_lib/meta/test_docs_gate.py",
    }
)

# Documents allowed to cite code by line number (only LINENO; every other check
# still applies). Matching is on the document's repo-relative path; a trailing
# `/` exempts the whole subtree.
ALLOWED_LINE_ANCHORS: dict[str, str] = {
    "docs/experiments/": (
        "a study record describes the code as it was at the commit it ran on, so its "
        "line numbers are pinned to that commit, not to the moving tree"
    ),
}

# Absolute-path leaks that are legitimate: experiment reports record the cluster
# directory a run actually lived in, which is provenance, not a leaked checkout.
ALLOWED_LEAKS: dict[str, str] = {
    "docs/experiments/2026-08-07-acquisition-inclusion/REPORT_REGION_VOTING.md": (
        "records the GRID scratch directory the run lived in (provenance)"
    ),
}

# Only tokens whose final component carries one of these suffixes (or that end
# in `/`) are treated as repo-path references. Everything else backticked is
# prose: `foo/bar` module notation, `application/json`, `and/or`, and so on.
PATH_SUFFIXES = (
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".mjs",
    ".html",
    ".scss",
    ".css",
    ".md",
    ".sh",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".cfg",
    ".ini",
    ".txt",
    ".pkl",
    ".npz",
    ".csv",
    ".sql",
    ".lock",
)

# Markdown files we do not police. Vendored or generated prose only.
SKIP_DIR_PARTS = frozenset({"node_modules", ".venv", "venv", "site-packages", ".git"})

# --------------------------------------------------------------------------
# Regexes
# --------------------------------------------------------------------------
FENCE_RE = re.compile(r"^(\s*)(```+|~~~+)(.*)$")
BLOCKQUOTE_RE = re.compile(r"^(?:\s*>\s?)+")
BROKEN_FENCE_RE = re.compile(r"^\s*(\S.*?)\s+(?:```+|~~~+)\s*\w*\s*$")
# `[text](target)` and `![alt](target)`, plus reference definitions `[k]: target`.
INLINE_LINK_RE = re.compile(r"!?\[(?P<text>(?:[^\][]|\[[^\]]*\])*)\]\((?P<target>[^()\s]*(?:\([^()]*\)[^()\s]*)*)")
REF_DEF_RE = re.compile(r"^\s{0,3}\[(?P<label>[^\]]+)\]:\s*(?P<target>\S+)")
INLINE_CODE_RE = re.compile(r"`([^`]+)`")
ATX_HEADING_RE = re.compile(r"^(\s{0,3})(#{1,6})\s+(?P<text>.*?)\s*#*\s*$")
HTML_ANCHOR_RE = re.compile(r"<a\s[^>]*\b(?:name|id)\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
# `docs/plans/foo.md`, wherever it appears — prose, docstrings, shell comments.
PLAN_REF_RE = re.compile(r"docs/plans/[A-Za-z0-9_.-]+\.md")
# An absolute path rooted in somebody's home directory.
LEAK_RE = re.compile(r"(?:/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+)/\S*")
# A `:NNN` or `:NNN-MMM` (or comma-separated) line-number suffix on a path.
LINE_SUFFIX_RE = re.compile(r":\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*$")
# A line-number anchor, in either spelling. The first is a repo file name with a
# `:NNN` suffix; the second a parenthesised `(line NNN)`, `(lines NNN-MMM)` or
# `(~line NNN)`, which the old vtscore-only check let through.
LINE_ANCHOR_RES = (
    re.compile(r"[\w./-]*\.(?:" + "|".join(re.escape(s[1:]) for s in PATH_SUFFIXES) + r"):\d+"),
    re.compile(r"\(\s*~?\s*lines?\s+\d+", re.IGNORECASE),
)
# Markdown emphasis/links inside heading text, stripped before slugging.
HEADING_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")

# `docs/experiments/` holds one directory per study, named for the day its report
# first landed. The date is what makes the archive browsable — `ls` alone answers
# "what is the latest result?" — and it only stays true if nothing can be added
# without one.
EXPERIMENTS_DIR = "docs/experiments"
STUDY_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-[a-z0-9]+(?:-[a-z0-9]+)*$")

# The gate chain, stated twice: once as code in `run-tests.sh`, once as the
# stage tables under `### The stages` in `docs/TESTING.md`. Every gate from
# stage 2 on is started through `_start_lane <kind> "<name>" <function>`, so
# those calls are the script's own list, and the name is what the run prints
# for the gate. A table row's Gate cell carries that name; a stage-3 row's Lane
# cell opens with the kind. Stage 1 runs outside the lanes, so its rows are
# matched on the script they run.
RUN_TESTS = "run-tests.sh"
TESTING_DOC = "docs/TESTING.md"
STAGES_HEADING = "The stages"
LANE_STAGE = {"gate": 2, "gating": 3, "free": 3}
START_LANE_RE = re.compile(r'^\s*_start_lane\s+(?P<kind>\S+)\s+"(?P<name>[^"]*)"')
# A shell expansion inside a lane name (`pyright ($_pyright_threads thread(s))`)
# is written `N` in the table, as the table's commands already do.
SHELL_EXPANSION_RE = re.compile(r"\$\{[^}]*\}|\$[A-Za-z_][A-Za-z0-9_]*")
SCRIPT_STAGE_RE = re.compile(r"^#\s*Stage (\d+):")
SCRIPT_PY_RE = re.compile(r"\bpython3?\s+(\S+\.py)\b")
DOC_STAGE_RE = re.compile(r"^\*\*Stage (\d+)\b")
TABLE_SEPARATOR_RE = re.compile(r"^:?-+:?$")


class Failure:
    __slots__ = ("check", "path", "line", "message")

    def __init__(self, check: str, path: Path, line: int, message: str) -> None:
        self.check = check
        self.path = path
        self.line = line
        self.message = message

    def __str__(self) -> str:
        rel = self.path.relative_to(ROOT) if self.path.is_absolute() else self.path
        return f"[{self.check}] {rel}:{self.line}: {self.message}"


# --------------------------------------------------------------------------
# Repo inventory
# --------------------------------------------------------------------------
def tracked_files() -> list[str]:
    """Every git-tracked path, repo-relative, POSIX-separated."""
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
            ["git", "-C", str(ROOT), "ls-files", "-z"],  # noqa: S607 - git resolved from PATH
            capture_output=True,
            check=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - git always present in-repo
        return [str(p.relative_to(ROOT).as_posix()) for p in ROOT.rglob("*") if p.is_file()]
    return [p for p in out.split("\0") if p]


def build_inventory(files: Iterable[str]) -> tuple[frozenset[str], frozenset[str]]:
    """Return (file paths, directory paths) as repo-relative POSIX strings."""
    file_set = set(files)
    dirs: set[str] = set()
    for f in file_set:
        parts = f.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            dirs.add("/".join(parts[:i]))
    return frozenset(file_set), frozenset(dirs)


# --------------------------------------------------------------------------
# Markdown structure
# --------------------------------------------------------------------------
def strip_quote(line: str) -> str:
    """Drop leading blockquote markers, so `> ```bash` reads as a normal fence."""
    return BLOCKQUOTE_RE.sub("", line)


def split_fences(text: str) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Split into (prose lines, fenced-code lines), each as (lineno, line).

    Fenced blocks are excluded from every prose-level check: a shell snippet may
    legitimately name a file that does not exist, and a JSON example may carry an
    absolute path. Blockquoted fences count — a code block inside a `>` quote is
    still a code block.
    """
    prose: list[tuple[int, str]] = []
    code: list[tuple[int, str]] = []
    fence: str | None = None
    for lineno, raw in enumerate(text.splitlines(), 1):
        match = FENCE_RE.match(strip_quote(raw))
        if fence is None:
            if match:
                fence = match.group(2)
                code.append((lineno, raw))
                continue
            prose.append((lineno, raw))
        else:
            code.append((lineno, raw))
            # A fence closes on a marker of at least the opening length, same char.
            if (
                match
                and match.group(2)[0] == fence[0]
                and len(match.group(2)) >= len(fence)
                and not match.group(3).strip()
            ):
                fence = None
    return prose, code


def prose_only_text(text: str, prose: list[tuple[int, str]]) -> str:
    """The document with every fenced-code line blanked out.

    Line numbering is preserved (blank lines stand in for the code), so an
    offset into the result maps back to a real line in the file.
    """
    keep = dict(prose)
    return "\n".join(keep.get(lineno, "") for lineno in range(1, len(text.splitlines()) + 1))


def slugify(heading: str) -> str:
    """GitHub's heading-slug algorithm.

    Strip markdown formatting, lowercase, drop everything that is not a letter,
    digit, space, hyphen or underscore, then swap spaces for hyphens. Runs of
    hyphens are *preserved*, which is exactly the trap that killed the
    USER_GUIDE table of contents: `## Autopilot - the guided workflow` slugs to
    `autopilot---the-guided-workflow`, with three hyphens, not two.
    """
    text = HEADING_LINK_RE.sub(r"\1", heading)
    text = text.replace("`", "").replace("*", "").replace("_", "_")
    text = re.sub(r"<[^>]+>", "", text)
    text = text.lower()
    text = "".join(ch for ch in text if ch.isalnum() or ch in " -_")
    return text.strip().replace(" ", "-")


def anchors_of(text: str) -> frozenset[str]:
    """Every anchor a reader can link to in this document."""
    prose, _ = split_fences(text)
    seen: dict[str, int] = {}
    anchors: set[str] = set()
    for _lineno, line in prose:
        for match in HTML_ANCHOR_RE.finditer(line):
            anchors.add(match.group(1))
        heading = ATX_HEADING_RE.match(line)
        if not heading:
            continue
        slug = slugify(heading.group("text"))
        if not slug:
            continue
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        anchors.add(slug if count == 0 else f"{slug}-{count}")
    return frozenset(anchors)


def iter_links(prose: list[tuple[int, str]]) -> Iterator[tuple[int, str]]:
    """Yield (lineno, target) for every markdown link and reference definition."""
    for lineno, line in prose:
        ref = REF_DEF_RE.match(line)
        if ref:
            yield lineno, ref.group("target")
            continue
        for match in INLINE_LINK_RE.finditer(line):
            target = match.group("target").strip()
            # Strip an optional link title: [x](path "Title")
            target = re.split(r"\s+[\"']", target, maxsplit=1)[0].strip()
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1]
            yield lineno, target


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
def is_external(target: str) -> bool:
    return bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target)) or target.startswith("//")


def allowlisted(key: str, table: dict[str, str]) -> bool:
    """True if `key` is in `table`, or falls under one of its `/`-ending prefixes."""
    if key in table:
        return True
    return any(prefix.endswith("/") and key.startswith(prefix) for prefix in table)


def allowed_path(token: str) -> bool:
    return allowlisted(token, ALLOWED_PATHS)


def link_base(rel: str, doc: Path) -> Path:
    """The directory a document's relative links resolve against.

    Normally the document's own directory; see :data:`LINK_BASE_OVERRIDES` for
    the trees that deliberately author links against a different base.
    """
    for prefix, base in LINK_BASE_OVERRIDES.items():
        if rel.startswith(prefix):
            return ROOT / base
    return doc.parent


def resolve_repo_path(token: str, doc: Path, files: frozenset[str], dirs: frozenset[str]) -> bool:
    """True if `token` names something that exists, read from the repo root or
    from the citing document's own directory."""
    bare = token.rstrip("/") or token
    if bare in files or bare in dirs:
        return True

    if "*" in bare or "?" in bare:
        pattern = re.escape(bare).replace(r"\*", "[^/]*").replace(r"\?", "[^/]")
        matcher = re.compile(rf"^{pattern}$")
        return any(matcher.match(p) for p in files) or any(matcher.match(d) for d in dirs)

    doc_rel = doc.parent.relative_to(ROOT).as_posix()
    joined = f"{doc_rel}/{bare}" if doc_rel != "." else bare
    return joined in files or joined in dirs or (doc.parent / bare).resolve().exists()


def check_markdown(
    doc: Path,
    text: str,
    files: frozenset[str],
    dirs: frozenset[str],
    top_level: frozenset[str],
    anchor_cache: dict[Path, frozenset[str]],
) -> list[Failure]:
    failures: list[Failure] = []
    rel = doc.relative_to(ROOT).as_posix()
    prose, _code = split_fences(text)
    # Seed the cache with the text we were handed, so in-page anchors are
    # checked against *this* version of the document rather than whatever is
    # currently on disk under the same name.
    anchor_cache.setdefault(doc, anchors_of(text))

    def anchors_for(target_doc: Path) -> frozenset[str]:
        if target_doc not in anchor_cache:
            try:
                anchor_cache[target_doc] = anchors_of(target_doc.read_text(encoding="utf-8"))
            except OSError:
                anchor_cache[target_doc] = frozenset()
        return anchor_cache[target_doc]

    # (1) LINK / (2) ANCHOR
    for lineno, target in iter_links(prose):
        if not target or is_external(target):
            continue
        path_part, _, anchor = target.partition("#")
        path_part = path_part.strip()
        if not path_part:
            if anchor and anchor not in anchors_for(doc):
                failures.append(Failure("ANCHOR", doc, lineno, f"'#{anchor}' matches no heading in this file"))
            continue
        if path_part.startswith("/"):
            failures.append(Failure("LINK", doc, lineno, f"'{target}' is an absolute link; use a relative path"))
            continue
        resolved = (link_base(rel, doc) / unquote(path_part)).resolve()
        if not resolved.exists():
            failures.append(Failure("LINK", doc, lineno, f"'{path_part}' does not exist"))
            continue
        if anchor and resolved.is_file() and resolved.suffix == ".md":
            if anchor not in anchors_for(resolved):
                where = resolved.relative_to(ROOT).as_posix() if ROOT in resolved.parents else resolved.name
                failures.append(Failure("ANCHOR", doc, lineno, f"'#{anchor}' matches no heading in {where}"))

    # (3) PATH — backticked repo-path references. Scanned over the whole
    # document rather than line by line: a code span may wrap across a line
    # break, and pairing backticks per line mis-reads every span after one that
    # does, silently skipping the references on those lines.
    if not rel.startswith(PATH_SKIP_PREFIXES):
        prose_text = prose_only_text(text, prose)
        for match in INLINE_CODE_RE.finditer(prose_text):
            token = " ".join(match.group(1).split())
            token = LINE_SUFFIX_RE.sub("", token).rstrip(".,;:")
            if token.startswith("./"):
                token = token[2:]
            if not looks_like_repo_path(token, top_level) or allowed_path(token):
                continue
            if not resolve_repo_path(token, doc, files, dirs):
                lineno = prose_text.count("\n", 0, match.start()) + 1
                failures.append(Failure("PATH", doc, lineno, f"`{token}` does not exist"))

    # (4) LEAK — absolute home-directory paths in prose.
    if rel not in ALLOWED_LEAKS:
        for lineno, line in prose:
            leak = LEAK_RE.search(line)
            if leak:
                failures.append(
                    Failure("LEAK", doc, lineno, f"absolute machine path '{leak.group(0)}' in committed prose")
                )

    # (6) FENCE — a fence opener with prose in front of it on the same line, e.g.
    # `-> ```json`. Markdown reads that as a paragraph, so the whole block that
    # follows renders as running text.
    for lineno, line in prose:
        broken = BROKEN_FENCE_RE.match(strip_quote(line))
        if broken and "`" not in broken.group(1):
            failures.append(
                Failure(
                    "FENCE", doc, lineno, f"code fence preceded by text ({line.strip()!r}); the block will not render"
                )
            )

    # (9) LINENO — a citation by line number. Fenced blocks are skipped like
    # every prose check: quoted tool output (a pytest failure line, a
    # traceback) carries line numbers of its own and is not a pointer to follow.
    if not allowlisted(rel, ALLOWED_LINE_ANCHORS):
        for lineno, line in prose:
            for pattern in LINE_ANCHOR_RES:
                anchor = pattern.search(line)
                if anchor:
                    failures.append(
                        Failure(
                            "LINENO",
                            doc,
                            lineno,
                            f"line-number anchor '{anchor.group(0)}'; cite the module and symbol instead, "
                            "since a line number rots on the next edit",
                        )
                    )

    return failures


def looks_like_repo_path(token: str, top_level: frozenset[str]) -> bool:
    """True if a backticked token is claiming to name a path *in this repo*.

    Anchoring on a real top-level entry is what keeps this rule quiet enough to
    gate. Without it the check drowns in tokens that contain a slash but name
    nothing in the tree: `application/json`, `and/or`, module notation like
    `vtscore.datasets/`, and the output directories experiment reports describe
    (`results/summary.json`, `agg/rate_*.csv`) which live in a scratch dir on a
    cluster, not here. A token that does not start with a real top-level entry is
    not making a claim this repo can falsify, so it is left alone.
    """
    if "/" not in token or len(token) > 200:
        return False
    if any(ch in token for ch in " \t\"'`$|<>{}()[]=@!&;\\^%+"):
        return False
    if "://" in token or token.startswith(("http", "//", "#")):
        return False
    if "..." in token:
        return False  # elided path, e.g. `frontend/.../foo.component.ts`
    if token.startswith("./"):
        token = token[2:]
    if token.startswith(("~", "/", "../")):
        return False  # outside the repo, or a relative link the LINK check owns
    if token.split("/", 1)[0] not in top_level:
        return False
    if token.endswith("/"):
        return True
    return token.rsplit("/", 1)[-1].endswith(PATH_SUFFIXES)


def check_plan_refs_in(path: Path, text: str, files: frozenset[str]) -> list[Failure]:
    """(5) PLAN — every docs/plans/*.md this file cites must exist."""
    if "docs/plans/" not in text:
        return []
    failures: list[Failure] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for ref in PLAN_REF_RE.findall(line):
            if ref not in files:
                failures.append(Failure("PLAN", path, lineno, f"cites deleted plan file '{ref}'"))
    return failures


def check_plan_refs(files: frozenset[str]) -> list[Failure]:
    """Run the PLAN check across the whole tracked tree, not just markdown.

    Module docstrings and inline comments cite plans far more often than other
    plans do, and those citations rot invisibly when a shipped plan is deleted —
    a dangling `See docs/plans/<name>.md` is exactly the "why is this code shaped
    like this" pointer a maintainer follows, and it now leads nowhere.
    """
    failures: list[Failure] = []
    for rel in sorted(files):
        if rel in PLAN_SELF_REFERENCE:
            continue
        path = ROOT / rel
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        failures.extend(check_plan_refs_in(path, text, files))
    return failures


def check_study_dirs(dirs: frozenset[str]) -> list[Failure]:
    """(7) STUDY — experiment study directories are dated.

    The date prefix is what puts the newest run at the end of an `ls`, and with
    each report opening on its question and verdict, `ls` plus `head` is the
    whole index. A hand-kept index table used to sit beside it; every study PR
    added a row at the same spot, so parallel studies always conflicted (#4138).
    """
    prefix = f"{EXPERIMENTS_DIR}/"
    failures: list[Failure] = []
    for rel in sorted(d for d in dirs if d.startswith(prefix) and d.count("/") == 2):
        name = rel[len(prefix) :]
        if not STUDY_DIR_RE.match(name):
            failures.append(Failure("STUDY", ROOT / rel, 0, f"study directory '{name}' is not named YYYY-MM-DD-<slug>"))
    return failures


def gate_key(name: str) -> str:
    """How a gate's name is compared: shell expansions read as `N`, case and spacing ignored."""
    return " ".join(SHELL_EXPANSION_RE.sub("N", name).replace("`", "").lower().split())


def script_lanes(script_text: str) -> dict[str, tuple[str, set[str], int]]:
    """Every lane ``run-tests.sh`` starts, as key -> (name, kinds, first line).

    A lane started from two places (Vitest starts either beside the build or
    once the build is green) is one gate.
    """
    lanes: dict[str, tuple[str, set[str], int]] = {}
    for lineno, line in enumerate(script_text.splitlines(), 1):
        match = START_LANE_RE.match(line)
        if not match:
            continue
        name = SHELL_EXPANSION_RE.sub("N", match.group("name"))
        lanes.setdefault(gate_key(name), (name, set(), lineno))[1].add(match.group("kind"))
    return lanes


def script_stage1(script_text: str) -> dict[str, int] | None:
    """The Python scripts ``run-tests.sh`` runs in its stage-1 block, as path -> line.

    None when the script has no ``# Stage 1:`` block to read.
    """
    scripts: dict[str, int] = {}
    stage: int | None = None
    found = False
    for lineno, line in enumerate(script_text.splitlines(), 1):
        header = SCRIPT_STAGE_RE.match(line.strip())
        if header:
            stage = int(header.group(1))
            found = found or stage == 1
            continue
        if stage != 1 or line.lstrip().startswith("#"):
            continue
        for path in SCRIPT_PY_RE.findall(line):
            scripts.setdefault(path, lineno)
    return scripts if found else None


def table_cells(line: str) -> list[str]:
    """A markdown table row's cells. GFM splits on every unescaped pipe, code spans included."""
    cells = [cell.strip() for cell in re.split(r"(?<!\\)\|", line.strip())]
    if cells and not cells[0]:
        cells = cells[1:]
    if cells and not cells[-1]:
        cells = cells[:-1]
    return cells


def doc_gate_rows(doc_text: str) -> list[tuple[int | None, int, dict[str, str]]] | None:
    """Every row of the tables under ``### The stages``, as (stage, line, cells by column).

    A table belongs to the ``**Stage N`` paragraph above it. None when the
    section is missing.
    """
    prose, _code = split_fences(doc_text)
    rows: list[tuple[int | None, int, dict[str, str]]] = []
    in_section = found = False
    stage: int | None = None
    header: list[str] | None = None
    for lineno, line in prose:
        heading = ATX_HEADING_RE.match(line)
        if heading:
            if in_section and len(heading.group(2)) <= 3:
                break
            if heading.group("text") == STAGES_HEADING:
                in_section = found = True
            continue
        if not in_section:
            continue
        marker = DOC_STAGE_RE.match(line)
        if marker:
            stage = int(marker.group(1))
        if not line.lstrip().startswith("|"):
            header = None
            continue
        cells = table_cells(line)
        if header is None:
            header = [cell.lower() for cell in cells]
        elif not all(TABLE_SEPARATOR_RE.match(cell) for cell in cells):
            rows.append((stage, lineno, dict(zip(header, cells))))
    return rows if found else None


def check_gate_tables(
    script_text: str,
    doc_text: str,
    script_path: Path = ROOT / RUN_TESTS,
    doc_path: Path = ROOT / TESTING_DOC,
) -> list[Failure]:
    """(8) GATES — the stage tables in docs/TESTING.md list the gates run-tests.sh runs.

    Matched by name, which is what a reader holds: the run prints a gate's
    name in its per-gate report and its ``TESTS BLOCKED`` / ``RUN FAILED``
    lines, and the table is where they look it up.
    """
    lanes = script_lanes(script_text)
    stage1 = script_stage1(script_text)
    rows = doc_gate_rows(doc_text)
    failures: list[Failure] = []
    if not lanes:
        failures.append(Failure("GATES", script_path, 0, "starts no `_start_lane` gate, so there is nothing to check"))
    if stage1 is None:
        failures.append(Failure("GATES", script_path, 0, "has no '# Stage 1:' block to read stage-1 gates from"))
    if rows is None:
        failures.append(Failure("GATES", doc_path, 0, f"has no '### {STAGES_HEADING}' section to read the gates from"))
    if stage1 is None or rows is None or failures:
        return failures

    for name, kinds, lineno in lanes.values():
        unknown = sorted(kinds - LANE_STAGE.keys())
        if unknown:
            failures.append(
                Failure(
                    "GATES",
                    script_path,
                    lineno,
                    f"'{name}' is started as a '{unknown[0]}' lane, which check-docs.py does not place in a stage",
                )
            )

    documented: dict[str, int] = {}
    documented_stage1: set[str] = set()
    for stage, lineno, cells in rows:
        gate = cells.get("gate", "")
        row = f"stage-{stage} row '{gate}'" if stage is not None else f"row '{gate}'"
        if stage == 1:
            command = INLINE_CODE_RE.search(cells.get("command it runs", ""))
            script = command.group(1).split()[0] if command else ""
            if script in stage1:
                documented_stage1.add(script)
            else:
                failures.append(
                    Failure(
                        "GATES", doc_path, lineno, f"{row} runs `{script}`, which {RUN_TESTS} does not run in stage 1"
                    )
                )
            continue
        key = gate_key(gate)
        if key in documented:
            failures.append(Failure("GATES", doc_path, lineno, f"{row} repeats the row on line {documented[key]}"))
            continue
        documented[key] = lineno
        if key not in lanes:
            failures.append(
                Failure(
                    "GATES",
                    doc_path,
                    lineno,
                    f"{row} names no gate {RUN_TESTS} starts; the Gate cell is the name the run prints for it",
                )
            )
            continue
        name, kinds, _first = lanes[key]
        stages = sorted({LANE_STAGE[kind] for kind in kinds if kind in LANE_STAGE})
        if stages and stage not in stages:
            failures.append(
                Failure("GATES", doc_path, lineno, f"{row} is in stage {stages[0]} in {RUN_TESTS}, not stage {stage}")
            )
            continue
        if stage == 3:
            word = re.match(r"[a-z]+", cells.get("lane", "").lower())
            lane = word.group(0) if word else ""
            if {lane} != kinds:
                started = " and ".join(sorted(kinds))
                failures.append(
                    Failure(
                        "GATES",
                        doc_path,
                        lineno,
                        f"{row} gives its lane as '{lane}', but {RUN_TESTS} starts it as {started}",
                    )
                )

    for key, (name, kinds, lineno) in lanes.items():
        if key not in documented:
            kind = " and ".join(sorted(kinds))
            failures.append(
                Failure(
                    "GATES",
                    script_path,
                    lineno,
                    f"gate '{name}' ({kind} lane) has no row in {TESTING_DOC}'s stage tables",
                )
            )
    for script, lineno in stage1.items():
        if script not in documented_stage1:
            failures.append(
                Failure("GATES", script_path, lineno, f"stage 1 runs `{script}`, which has no row in {TESTING_DOC}")
            )
    return failures


def check_gate_tables_in_repo() -> list[Failure]:
    script_path, doc_path = ROOT / RUN_TESTS, ROOT / TESTING_DOC
    try:
        script_text = script_path.read_text(encoding="utf-8")
        doc_text = doc_path.read_text(encoding="utf-8")
    except OSError as exc:
        return [Failure("GATES", ROOT, 0, str(exc))]
    return check_gate_tables(script_text, doc_text, script_path, doc_path)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def markdown_docs(files: frozenset[str]) -> list[Path]:
    docs = []
    for rel in sorted(files):
        if not rel.endswith(".md"):
            continue
        if SKIP_DIR_PARTS & set(rel.split("/")):
            continue
        docs.append(ROOT / rel)
    return docs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-v", "--verbose", action="store_true", help="print per-check counts")
    args = parser.parse_args(argv)

    files, dirs = build_inventory(tracked_files())
    # Top-level entries a doc could be naming: every tracked top-level directory,
    # plus the gitignored runtime/build directories that exist at run time.
    top_level = frozenset({d for d in dirs if "/" not in d} | {"data", "static"})
    docs = markdown_docs(files)

    failures: list[Failure] = []
    anchor_cache: dict[Path, frozenset[str]] = {}
    for doc in docs:
        try:
            text = doc.read_text(encoding="utf-8")
        except OSError as exc:  # pragma: no cover - tracked file always readable
            failures.append(Failure("READ", doc, 0, str(exc)))
            continue
        failures.extend(check_markdown(doc, text, files, dirs, top_level, anchor_cache))
    failures.extend(check_plan_refs(files))
    failures.extend(check_study_dirs(dirs))
    failures.extend(check_gate_tables_in_repo())

    if failures:
        print("check-docs FAILED:")
        for failure in sorted(failures, key=lambda f: (str(f.path), f.line)):
            print(f"  - {failure}")
        print(f"\n{len(failures)} documentation problem(s) across {len(docs)} markdown file(s).")
        return 1

    if args.verbose:
        print(f"check-docs: scanned {len(docs)} markdown files, {len(files)} tracked paths")
    print(f"check-docs OK: {len(docs)} markdown files, 9 invariants, no drift.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
