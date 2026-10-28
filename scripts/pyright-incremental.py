#!/usr/bin/env python3
"""Run pyright on the files whose diagnostics could have changed since the last green run.

A full ``pyright`` over this repository is the longest serial gate in
``run-tests.sh``: ~85s on four threads of a 4-vCPU box (2026-10-26; ~125s
with the frontend lanes beside it), against ~410s for the whole of pytest.
Pyright's own statistics say where that goes.
Of a 283s single-thread run, before the local ``transformers`` stub (#5077)
shrank the import closure, ~65s parsed and bound the 6,400-file import closure
(transformers and torch included) and ~215s *checked* the 1,208 files that
``pyrightconfig.json`` includes. Checking is per file, and pyright binds only
what the files it is asked to check reach, so ``pyright <one changed test
file>`` takes about 3s. A session's second and later full runs usually change
a handful of files; this script makes them pay for those files alone.

The premise, and why it is exact rather than a heuristic
--------------------------------------------------------

A file's pyright diagnostics are a function of four inputs: its own text; the
text of every module it imports, transitively, through ``import`` and
``from ... import`` statements (pyright evaluates types by following exactly
those -- an ``importlib.import_module`` string or a ``mock.patch`` target is
typed as ``ModuleType`` / ``Any`` and creates no dependency); the installed
packages and stubs; and the configuration. So a file none of whose inputs
changed since a green run is still green, and the files whose inputs did
change are exactly the changed files plus everything that transitively
imports one. That is the set this script re-checks. It is the same
invalidation pyright's language server performs on every edit.

Four guards keep the premise honest:

* **A baseline is only ever written by a green run.** The baseline (by
  default ``.gates-cache/pyright-baseline.json`` under the repo root, a
  gitignored runtime file) records the blob hash of every ``.py`` / ``.pyi``
  the repo tracks or carries untracked, the two config files, an environment
  key (the pyright version pin, the Python version, every installed
  distribution) and each blob's parsed import statements. A red run leaves it
  alone, so the baseline always describes a tree the whole of which pyright
  passed -- by induction: an unchanged file's inputs were green under the
  previous baseline and did not change; a changed file, or a dependent of
  one, was just checked.
* **Anything that changes every file's inputs forces a full run**: no
  baseline (a fresh clone, the release gate), a different environment key, a
  change to ``pyrightconfig.json`` or ``pyproject.toml``, ``--full`` /
  ``VTSEARCH_FRESH=1``, or a scoped target set over 60% of the checkable files
  (at which point the scoped run would not be faster anyway).
* **Resolution is conservative.** Import names are resolved against the same
  roots pyright uses -- the repository root, ``extraPaths`` and ``stubPath``
  read from ``pyrightconfig.json`` -- and every candidate that exists becomes
  an edge: ``import a.b.c`` depends on ``a/__init__``, ``a/b/__init__``,
  ``a/b.py`` and ``a/b/c`` alike; ``from a.b import c`` also on ``a/b/c.py``
  when there is one; a name that is both ``x.py`` and ``x/`` depends on both.
  The closure is taken over the union of the old and the new import graph, so
  a deleted module's importers and a new module's importers are both
  re-checked. An extra edge can only enlarge the target set.
* **Any failure to read git, the config or the baseline falls back to a full
  run**, never to a smaller one.

Files outside the ``include`` set (``app.py``, ``scripts/experiments/...``)
are graph nodes but never targets: pyright does not check them today, and a
change to one still re-checks every included file that imports it.

Usage::

    python scripts/pyright-incremental.py [--threads N] [--full] [--dry-run]
                                          [--baseline PATH] [--repo PATH]

The exit code is pyright's. ``--dry-run`` prints the plan and exits 0 without
running pyright or touching the baseline. Pyright's own output streams
through unchanged, below a one-line summary of what this run decided.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import json
import os
import posixpath
import re
import subprocess
import sys
import tempfile
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

BASELINE_FORMAT = 1
DEFAULT_BASELINE = Path(".gates-cache") / "pyright-baseline.json"
CONFIG_FILES = ("pyrightconfig.json", "pyproject.toml")
SOURCE_SUFFIXES = (".py", ".pyi")
#: Above this share of the checkable files, a scoped run is not worth its bookkeeping.
FULL_THRESHOLD = 0.6
#: Pyright's own default exclusions, applied only when the config names none.
DEFAULT_EXCLUDE = ("**/node_modules", "**/__pycache__", "**/.*")

#: An import statement as stored in the baseline: (module, level, imported names).
ImportSpec = tuple[str, int, tuple[str, ...]]
#: Runs pyright with the given argv in the given directory; returns its exit code.
Runner = Callable[[list[str], Path], int]


class Fallback(Exception):
    """The scoped path cannot be trusted; the caller runs pyright in full."""


# ---------------------------------------------------------------------------
# git
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str, stdin: str | None = None) -> str:
    try:
        out = subprocess.run(  # noqa: S603  # fixed argv, no shell
            ["git", *args],  # noqa: S607  # git resolves on PATH, as everywhere else in this repo
            cwd=repo,
            input=stdin,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise Fallback(f"git unavailable: {exc}") from exc
    if out.returncode != 0:
        raise Fallback(f"git {args[0]} failed: {out.stderr.strip() or out.returncode}")
    return out.stdout


def source_paths(repo: Path) -> list[str]:
    """Every ``.py`` / ``.pyi`` and config file the repo tracks or carries untracked.

    Untracked files count because pyright checks them; ignored ones (a venv,
    build output) do not. A tracked file deleted from the working tree is
    dropped: it is gone for pyright too.
    """
    raw = _git(repo, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    seen: set[str] = set()
    for rel in raw.split("\0"):
        if not rel or rel in seen:
            continue
        if not (rel.endswith(SOURCE_SUFFIXES) or rel in CONFIG_FILES):
            continue
        if not (repo / rel).is_file():
            continue
        seen.add(rel)
    return sorted(seen)


def manifest(repo: Path, paths: list[str]) -> dict[str, str]:
    """``path -> git blob hash`` for *paths*, in one git call."""
    if not paths:
        return {}
    out = _git(repo, "hash-object", "--stdin-paths", stdin="\n".join(paths) + "\n")
    hashes = out.split()
    if len(hashes) != len(paths):
        raise Fallback("git hash-object returned a different number of hashes than paths")
    return dict(zip(paths, hashes, strict=True))


def environment_key() -> str:
    """Everything outside the tree that pyright's answer depends on, hashed."""
    pin = os.environ.get("PYRIGHT_PYTHON_FORCE_VERSION")
    if pin:
        pyright_version = f"pin {pin}"
    else:
        try:
            out = subprocess.run(  # noqa: S603  # fixed argv, no shell
                ["pyright", "--version"],  # noqa: S607  # pyright resolves on PATH, as run-tests.sh runs it
                capture_output=True,
                text=True,
                check=False,
            )
            pyright_version = out.stdout.strip() or f"exit {out.returncode}"
        except OSError:
            pyright_version = "unknown"
    from importlib import metadata

    dists: set[str] = set()
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            dists.add(f"{name}=={dist.version}")
    digest = hashlib.sha256()
    for part in (pyright_version, sys.version, *sorted(dists)):
        digest.update(part.encode())
        digest.update(b"\0")
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# pyrightconfig.json
# ---------------------------------------------------------------------------

# pyrightconfig.json is JSON with comments and tolerates a trailing comma. The
# alternation keeps string literals intact so a `//` inside one is not a comment.
_JSONC_COMMENT = re.compile(r'("(?:[^"\\]|\\.)*")|//[^\n]*')
_JSONC_TRAILING_COMMA = re.compile(r'("(?:[^"\\]|\\.)*")|,(\s*[}\]])')


def _strip_jsonc(text: str) -> str:
    text = _JSONC_COMMENT.sub(lambda m: m.group(1) or "", text)
    return _JSONC_TRAILING_COMMA.sub(lambda m: m.group(1) or m.group(2), text)


def _norm(rel: str) -> str:
    rel = rel.replace("\\", "/").strip()
    while rel.startswith("./"):
        rel = rel[2:]
    return rel.strip("/")


@dataclass(frozen=True)
class PyrightConfig:
    """The three things this script reads from ``pyrightconfig.json``."""

    include: tuple[str, ...]
    exclude: tuple[str, ...]
    #: Import search roots as repo-relative prefixes; ``""`` is the repo root.
    roots: tuple[str, ...]


def load_config(repo: Path) -> PyrightConfig:
    path = repo / "pyrightconfig.json"
    try:
        data = json.loads(_strip_jsonc(path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        raise Fallback(f"cannot read pyrightconfig.json: {exc}") from exc
    if not isinstance(data, dict):
        raise Fallback("pyrightconfig.json is not an object")
    include = tuple(_norm(p) for p in data.get("include", ["."]))
    exclude = tuple(_norm(p) for p in data.get("exclude", list(DEFAULT_EXCLUDE)))
    roots: list[str] = [""]
    for extra in data.get("extraPaths", []):
        roots.append(_norm(extra))
    for implicit in (_norm(str(data.get("stubPath", "typings"))), "src"):
        if (repo / implicit).is_dir():
            roots.append(implicit)
    deduped = tuple(dict.fromkeys(roots))
    return PyrightConfig(include=include, exclude=exclude, roots=deduped)


def _glob_regex(pattern: str) -> re.Pattern[str]:
    """A pyright-style path glob as a regex matching the path or anything under it."""
    out = "^"
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out += "(?:.*/)?"
            i += 3
        elif pattern.startswith("**", i):
            out += ".*"
            i += 2
        elif pattern[i] == "*":
            out += "[^/]*"
            i += 1
        elif pattern[i] == "?":
            out += "[^/]"
            i += 1
        else:
            out += re.escape(pattern[i])
            i += 1
    return re.compile(out + "(?:/.*)?$")


def _matches_any(patterns: Iterable[str], rel: str) -> bool:
    for pattern in patterns:
        if pattern in ("", "."):
            return True
        if any(ch in pattern for ch in "*?["):
            if _glob_regex(pattern).match(rel) or fnmatch.fnmatchcase(rel, pattern):
                return True
        elif rel == pattern or rel.startswith(pattern + "/"):
            return True
    return False


def is_checkable(rel: str, config: PyrightConfig) -> bool:
    """Would pyright check *rel* on a bare run? Mirrors ``include`` minus ``exclude``."""
    if not rel.endswith(SOURCE_SUFFIXES):
        return False
    return _matches_any(config.include, rel) and not _matches_any(config.exclude, rel)


# ---------------------------------------------------------------------------
# the import graph
# ---------------------------------------------------------------------------


def parse_imports(text: str) -> list[ImportSpec]:
    """Every import statement in *text*, at any nesting depth.

    A file that does not parse has no imports here; it is still its own
    target when it changed, and pyright reports the syntax error itself.
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    specs: list[ImportSpec] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                specs.append((alias.name, 0, ()))
        elif isinstance(node, ast.ImportFrom):
            specs.append((node.module or "", node.level or 0, tuple(alias.name for alias in node.names)))
    return specs


def _candidates(base: str, parts: list[str]) -> list[str]:
    """Every file that could stand for dotted *parts* under the directory *base*.

    Each prefix of the dotted name yields both a module file and a package
    ``__init__`` -- the conservative rule: if ``a/b.py`` and ``a/b/`` both exist,
    an import of ``a.b.c`` depends on both.
    """
    out: list[str] = []
    cur = base
    for part in parts:
        cur = posixpath.join(cur, part)
        for suffix in SOURCE_SUFFIXES:
            out.append(cur + suffix)
            out.append(posixpath.join(cur, "__init__" + suffix))
    return out


def resolve(spec: ImportSpec, importer: str, files: set[str], roots: Iterable[str]) -> set[str]:
    """The repo files *spec* (an import in *importer*) can refer to; empty means external."""
    module, level, names = spec
    parts = [p for p in module.split(".") if p]
    found: set[str] = set()
    if level == 0:
        bases = list(roots)
    else:
        base = posixpath.dirname(importer)
        for _ in range(level - 1):
            if not base:
                return found  # climbed above the repo root
            base = posixpath.dirname(base)
        bases = [base]
        for suffix in SOURCE_SUFFIXES:
            init = posixpath.join(base, "__init__" + suffix)
            if init in files:
                found.add(init)
    for base in bases:
        for cand in _candidates(base, parts):
            if cand in files:
                found.add(cand)
        for name in names:
            if name == "*":
                continue
            for cand in _candidates(base, [*parts, name]):
                if cand in files:
                    found.add(cand)
    return found


def build_graph(
    files: set[str], specs_of: Callable[[str], list[ImportSpec]], roots: Iterable[str]
) -> dict[str, set[str]]:
    """``importer -> repo files it imports`` over every source file in *files*."""
    roots = tuple(roots)
    graph: dict[str, set[str]] = {}
    for rel in files:
        if not rel.endswith(SOURCE_SUFFIXES):
            continue
        deps: set[str] = set()
        for spec in specs_of(rel):
            deps |= resolve(spec, rel, files, roots)
        deps.discard(rel)
        graph[rel] = deps
    return graph


def reverse_closure(changed: Iterable[str], graphs: Iterable[dict[str, set[str]]]) -> set[str]:
    """*changed* plus everything that transitively imports one of them, over the union of *graphs*."""
    importers: dict[str, set[str]] = {}
    for graph in graphs:
        for importer, deps in graph.items():
            for dep in deps:
                importers.setdefault(dep, set()).add(importer)
    seen = set(changed)
    queue = deque(seen)
    while queue:
        for importer in importers.get(queue.popleft(), ()):
            if importer not in seen:
                seen.add(importer)
                queue.append(importer)
    return seen


# ---------------------------------------------------------------------------
# the baseline
# ---------------------------------------------------------------------------


def _specs_to_json(specs: list[ImportSpec]) -> list[list[object]]:
    return [[module, level, list(names)] for module, level, names in specs]


def _specs_from_json(raw: object) -> list[ImportSpec]:
    specs: list[ImportSpec] = []
    if not isinstance(raw, list):
        raise Fallback("baseline import data is malformed")
    for item in raw:
        if not (isinstance(item, list) and len(item) == 3):
            raise Fallback("baseline import data is malformed")
        module, level, names = item
        specs.append((str(module), int(level), tuple(str(n) for n in names)))
    return specs


def read_baseline(path: Path) -> dict | None:
    """The baseline as a dict, ``None`` when there is none; malformed raises :class:`Fallback`."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Fallback(f"baseline unreadable: {exc}") from exc
    if not isinstance(data, dict) or data.get("format") != BASELINE_FORMAT:
        raise Fallback("baseline has an unknown format")
    for key in ("env", "manifest", "imports", "written"):
        if key not in data:
            raise Fallback(f"baseline lacks {key!r}")
    return data


def write_baseline(
    path: Path, env: str, new_manifest: dict[str, str], specs_by_hash: dict[str, list[ImportSpec]]
) -> str:
    """Atomically record a tree pyright has just passed in full. Returns the timestamp written."""
    written = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    kept_hashes = {h for rel, h in new_manifest.items() if rel.endswith(SOURCE_SUFFIXES)}
    data = {
        "format": BASELINE_FORMAT,
        "written": written,
        "env": env,
        "manifest": dict(sorted(new_manifest.items())),
        "imports": {h: _specs_to_json(specs_by_hash[h]) for h in sorted(kept_hashes) if h in specs_by_hash},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".pyright-baseline-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, separators=(",", ":"))
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return written


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------


@dataclass
class Plan:
    """What this run will ask pyright to do, and what a green result may record."""

    mode: str  # "full" | "scoped"
    reason: str
    targets: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    checkable: int = 0
    baseline_written: str = ""
    #: Set when a green run may write a new baseline; None when the tree could not be read.
    env: str | None = None
    new_manifest: dict[str, str] | None = None
    specs_by_hash: dict[str, list[ImportSpec]] | None = None

    @property
    def summary(self) -> str:
        if self.mode == "full":
            return f"pyright (full): {self.reason}"
        what = f"{len(self.targets)} to check" if self.targets else "nothing to check"
        return (
            f"pyright (incremental): {len(self.changed)} changed, {what} of {self.checkable} checkable;"
            f" baseline {self.baseline_written}"
        )


def _read_tree(repo: Path, baseline: dict | None) -> tuple[dict[str, str], dict[str, list[ImportSpec]]]:
    """The current manifest, and the import specs of every source blob in it (parsed or cached)."""
    paths = source_paths(repo)
    new_manifest = manifest(repo, paths)
    cached: dict = baseline["imports"] if baseline else {}
    specs_by_hash: dict[str, list[ImportSpec]] = {}
    for rel, blob in new_manifest.items():
        if not rel.endswith(SOURCE_SUFFIXES) or blob in specs_by_hash:
            continue
        if blob in cached:
            specs_by_hash[blob] = _specs_from_json(cached[blob])
        else:
            try:
                text = (repo / rel).read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                raise Fallback(f"cannot read {rel}: {exc}") from exc
            specs_by_hash[blob] = parse_imports(text)
    return new_manifest, specs_by_hash


def make_plan(repo: Path, baseline_path: Path, *, force_full: bool = False) -> Plan:
    """Decide between a full and a scoped run; never raises, a problem means full."""
    # A baseline that cannot be read is treated as absent: the tree is still
    # read, so the full run this forces can record a sound one in its place.
    baseline_problem = ""
    try:
        baseline = read_baseline(baseline_path)
    except Fallback as exc:
        baseline, baseline_problem = None, str(exc)
    try:
        new_manifest, specs_by_hash = _read_tree(repo, baseline)
        env = environment_key()
    except Fallback as exc:
        print(f"pyright-incremental: {exc}; running pyright in full", file=sys.stderr)
        return Plan(mode="full", reason=f"fallback ({exc})")
    recordable = dict(env=env, new_manifest=new_manifest, specs_by_hash=specs_by_hash)

    if force_full:
        return Plan(mode="full", reason="forced (--full or VTSEARCH_FRESH=1)", **recordable)
    if baseline is None:
        if baseline_problem:
            print(f"pyright-incremental: {baseline_problem}; running pyright in full", file=sys.stderr)
            return Plan(mode="full", reason=f"fallback ({baseline_problem})", **recordable)
        return Plan(mode="full", reason="no baseline", **recordable)
    if baseline["env"] != env:
        return Plan(mode="full", reason="environment changed since the baseline", **recordable)
    old_manifest: dict[str, str] = baseline["manifest"]
    for rel in CONFIG_FILES:
        if old_manifest.get(rel) != new_manifest.get(rel):
            return Plan(mode="full", reason=f"{rel} changed since the baseline", **recordable)

    try:
        config = load_config(repo)
        old_specs: dict = baseline["imports"]

        def old_specs_of(rel: str) -> list[ImportSpec]:
            blob = old_manifest[rel]
            if blob not in old_specs:
                raise Fallback(f"baseline lacks import data for {rel}")
            return _specs_from_json(old_specs[blob])

        def new_specs_of(rel: str) -> list[ImportSpec]:
            return specs_by_hash[new_manifest[rel]]

        old_files = {rel for rel in old_manifest if rel.endswith(SOURCE_SUFFIXES)}
        new_files = {rel for rel in new_manifest if rel.endswith(SOURCE_SUFFIXES)}
        changed = sorted(rel for rel in old_files | new_files if old_manifest.get(rel) != new_manifest.get(rel))
        old_graph = build_graph(old_files, old_specs_of, config.roots)
        new_graph = build_graph(new_files, new_specs_of, config.roots)
    except Fallback as exc:
        print(f"pyright-incremental: {exc}; running pyright in full", file=sys.stderr)
        return Plan(mode="full", reason=f"fallback ({exc})", **recordable)

    closure = reverse_closure(changed, (old_graph, new_graph))
    checkable = {rel for rel in new_files if is_checkable(rel, config)}
    targets = sorted(closure & checkable)
    if checkable and len(targets) > FULL_THRESHOLD * len(checkable):
        return Plan(
            mode="full",
            reason=f"{len(targets)} of {len(checkable)} checkable files affected, over the"
            f" {int(FULL_THRESHOLD * 100)}% threshold",
            changed=changed,
            checkable=len(checkable),
            **recordable,
        )
    return Plan(
        mode="scoped",
        reason="",
        targets=targets,
        changed=changed,
        checkable=len(checkable),
        baseline_written=str(baseline["written"]),
        **recordable,
    )


# ---------------------------------------------------------------------------
# running it
# ---------------------------------------------------------------------------


def run_pyright(argv: list[str], cwd: Path) -> int:
    """The default runner: pyright on PATH, output streaming through, cwd at the repo root."""
    try:
        return subprocess.run(argv, cwd=cwd, check=False).returncode  # noqa: S603  # fixed argv, no shell
    except OSError as exc:
        print(f"pyright-incremental: cannot run {argv[0]}: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None, runner: Runner = run_pyright) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=None, help="passed through to pyright")
    ap.add_argument("--full", action="store_true", help="check everything, as a bare pyright would")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; run nothing, write nothing")
    ap.add_argument("--baseline", type=Path, default=None, help=f"baseline file (default {DEFAULT_BASELINE})")
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args(argv)

    repo = args.repo.resolve()
    baseline_path = args.baseline if args.baseline is not None else DEFAULT_BASELINE
    if not baseline_path.is_absolute():
        baseline_path = repo / baseline_path
    force_full = bool(args.full) or os.environ.get("VTSEARCH_FRESH") == "1"

    plan = make_plan(repo, baseline_path, force_full=force_full)
    print(plan.summary, flush=True)
    if args.dry_run:
        for target in plan.targets:
            print(f"  {target}", flush=True)
        return 0

    if plan.mode == "scoped" and not plan.targets:
        status = 0
    else:
        cmd = ["pyright"]
        if args.threads:
            cmd += ["--threads", str(args.threads)]
        if plan.mode == "scoped":
            cmd += plan.targets
        status = runner(cmd, repo)

    if status == 0 and plan.env is not None and plan.new_manifest is not None and plan.specs_by_hash is not None:
        try:
            write_baseline(baseline_path, plan.env, plan.new_manifest, plan.specs_by_hash)
        except OSError as exc:
            print(f"pyright-incremental: could not write {baseline_path}: {exc}", file=sys.stderr)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
