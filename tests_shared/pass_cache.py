"""Skip tests that already passed on this exact tree: the pass cache.

The recommended loop is ``./run-tests.sh <group>`` and then a full
``./run-tests.sh``, and the full run used to repeat every test the group run
had just passed -- for ``detectors`` that is more than half of pytest's wall
time, spent proving a fact the session already held.  This plugin records which
tests passed under a key that pins *everything a test can observe* -- the
committed tree, every uncommitted and untracked change, the interpreter, the
installed packages and the ``VTSEARCH_*`` environment -- and a later run under
the same key deselects them.  A key that has not changed means the inputs have
not changed, so a pass recorded under it is as good as a pass this run would
produce; a key that has changed, by so much as one byte in one file, reuses
nothing.  That is the whole soundness argument: the cache never guesses which
tests a change can reach, it only declines to repeat a run whose inputs are
identical.

What it stores, and where: pytest's own cache (``.pytest_cache/v/vtsearch/
pass-cache``, gitignored with the rest of that directory), as
``{"keys": {<key>: {"when": <UTC ISO time>, "nodeids": [...]}}}``, pruned to
the four most recent keys.  Only the controller writes it, at session end, and a
red run still records the tests that did pass.

Hooks.  Both conftests delegate to the functions below from two-line hook
bodies (the ``tests_shared`` pattern); each is idempotent, because both
conftests fire on one session.

* :func:`configure_pass_cache` (``pytest_configure``) computes the key on the
  controller and exports it as ``VTSEARCH_PASS_CACHE_KEY`` so xdist workers,
  spawned afterwards with the controller's environment, read the same key
  without re-deriving it.
* :func:`deselect_cached_passes` (``pytest_collection_modifyitems``) removes the
  recorded passes from the collection.  It runs where collection runs -- in
  each worker under xdist -- and every worker reads the same file under the
  same key, so their collections stay identical.  A worker leaves its count in
  ``config.workeroutput``, which xdist hands the controller.
* :func:`record_pass_report` (``pytest_runtest_logreport``) notes, on the
  controller, each test whose setup, call and teardown all passed.  Skips,
  xfails, xpasses and teardown errors never count.
* :func:`note_worker_reused` (``pytest_testnodedown``) reads a finished
  worker's count back.
* :func:`finish_pass_cache` (``pytest_sessionfinish``) merges the new passes
  into the key's entry, writes the file, and turns pytest's "no tests
  collected" exit (5) into 0 when the only reason nothing ran is that every
  selected test was already recorded.

``VTSEARCH_FRESH=1`` disables the deselection (every test runs) but still
records, so the next run can reuse.  A session without a usable key -- no git,
no commit, ``-p no:cacheprovider`` -- records and reuses nothing, and says so
once.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import pytest

#: Where the entries live inside pytest's cache (``.pytest_cache/v/<this>``).
CACHE_KEY = "vtsearch/pass-cache"
#: The controller exports the tree key here; xdist workers read it back.
KEY_ENV = "VTSEARCH_PASS_CACHE_KEY"
#: Set to ``1`` to run every selected test regardless of the cache.
FRESH_ENV = "VTSEARCH_FRESH"
#: How many tree keys the file keeps.
KEEP_KEYS = 4
#: The slot a worker uses in ``config.workeroutput``.
WORKEROUTPUT_KEY = "vtsearch_pass_cache"
#: ``VTSEARCH_*`` variables that are per-session temp dirs (tests_shared/data_dir.py),
#: not inputs a test's outcome can depend on.
EXCLUDED_ENV = frozenset({"VTSEARCH_DATA_DIR", "VTSEARCH_TEST_DATA_DIR", KEY_ENV})

_PHASES = ("setup", "call", "teardown")


class PassCacheState:
    """Everything the hooks share for one session, in one process."""

    def __init__(self) -> None:
        self.key: str | None = None
        self.is_worker = False
        #: Tests deselected because they were recorded under this key.
        self.reused = 0
        #: The ``when`` of the entry those passes came from.
        self.previous_when: str | None = None
        #: nodeid -> phase -> passed, for the reports this controller saw.
        self.reports: dict[str, dict[str, bool]] = {}
        #: Tests that passed in this run and were written to the entry.
        self.newly_recorded = 0
        self.deselected_done = False
        self.finished = False


# One session per process; the hooks that pytest calls without a ``config``
# (``pytest_runtest_logreport``, ``pytest_testnodedown``) find it here.
_state: PassCacheState | None = None


# ---------------------------------------------------------------------------
# The tree key
# ---------------------------------------------------------------------------


def _git(root: Path, *args: str, stdin: bytes | None = None) -> bytes:
    out = subprocess.run(  # noqa: S603  # fixed argv, no shell
        ["git", *args],  # noqa: S607  # git resolves on PATH, as everywhere else in this repo
        cwd=root,
        input=stdin,
        capture_output=True,
        check=True,
    )
    return out.stdout


def installed_distributions_digest() -> str:
    """A digest of every installed distribution's name and version."""
    names = set()
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"] if dist.metadata else None
        if name:
            names.add(f"{name}=={dist.version}")
    return hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()


def tree_key(root: Path, environ: Mapping[str, str] | None = None, *, dists_digest: str | None = None) -> str | None:
    """The key under which this tree's passes are recorded, or ``None`` if it cannot be derived.

    Covers the commit at HEAD, every staged and unstaged change to a tracked
    file (``git diff HEAD --binary``), every untracked file that is not ignored
    (path and blob hash), the interpreter, the installed distributions and the
    ``VTSEARCH_*`` environment minus the per-session temp dirs.  Ignored files
    are deliberately outside it: ``static/``, the generated API client and
    ``node_modules`` are functions of tracked inputs, and ``data/`` is a fresh
    temp dir per session.

    ``None`` when git is missing, *root* is not a repository, or it has no
    commit yet -- the plugin then neither reuses nor records.
    """
    env: Mapping[str, str] = os.environ if environ is None else environ
    try:
        head = _git(root, "rev-parse", "HEAD")
        diff = _git(root, "diff", "HEAD", "--no-ext-diff", "--binary")
        others = [p for p in _git(root, "ls-files", "-z", "--others", "--exclude-standard").split(b"\0") if p]
        hashes = _git(root, "hash-object", "--stdin-paths", stdin=b"\n".join(others) + b"\n") if others else b""
    except (OSError, subprocess.CalledProcessError):
        return None
    h = hashlib.sha256()
    for label, blob in (
        (b"head", head),
        (b"diff", diff),
        (b"untracked", b"\0".join(p + b"=" + hsh for p, hsh in zip(others, hashes.split(), strict=True))),
        (b"python", sys.version.encode()),
        (b"dists", (dists_digest or installed_distributions_digest()).encode()),
        (
            b"env",
            "\0".join(
                f"{k}={v}" for k, v in sorted(env.items()) if k.startswith("VTSEARCH_") and k not in EXCLUDED_ENV
            ).encode(),
        ),
    ):
        h.update(label + b"\0" + blob + b"\0\0")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Pure pieces the hooks are built from
# ---------------------------------------------------------------------------


def passed_nodeids(reports: dict[str, dict[str, bool]]) -> set[str]:
    """The nodeids whose setup, call and teardown all passed."""
    return {nodeid for nodeid, phases in reports.items() if all(phases.get(phase) for phase in _PHASES)}


def prune(keys: dict[str, dict[str, Any]], keep: int = KEEP_KEYS) -> dict[str, dict[str, Any]]:
    """Keep the *keep* most recently written entries."""
    ordered = sorted(keys.items(), key=lambda kv: str(kv[1].get("when") or ""), reverse=True)
    return dict(ordered[:keep])


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read(config) -> dict[str, Any]:
    data = config.cache.get(CACHE_KEY, {})
    if not isinstance(data, dict) or not isinstance(data.get("keys", {}), dict):
        return {"keys": {}}
    data.setdefault("keys", {})
    return data


def _usable(config, state: PassCacheState) -> bool:
    return state.key is not None and hasattr(config, "cache")


# ---------------------------------------------------------------------------
# Hook bodies
# ---------------------------------------------------------------------------


def configure_pass_cache(config) -> PassCacheState:
    """``pytest_configure``: derive the key (controller) or read it (worker)."""
    global _state
    existing = getattr(config, "_vtsearch_pass_cache", None)
    if existing is not None:
        return existing
    state = PassCacheState()
    config._vtsearch_pass_cache = state
    _state = state
    state.is_worker = hasattr(config, "workerinput")
    if not config.pluginmanager.hasplugin("cacheprovider"):
        return state
    if state.is_worker:
        state.key = os.environ.get(KEY_ENV) or None
        return state
    state.key = tree_key(Path(str(config.rootpath)))
    if state.key is None:
        os.environ.pop(KEY_ENV, None)
        print("pass cache: no git tree key for this run; nothing is reused or recorded", file=sys.stderr)
    else:
        os.environ[KEY_ENV] = state.key
    return state


def deselect_cached_passes(config, items: list) -> int:
    """``pytest_collection_modifyitems``: drop the tests recorded under this key."""
    state = getattr(config, "_vtsearch_pass_cache", None)
    if state is None:
        return 0
    if state.deselected_done:
        return state.reused
    state.deselected_done = True
    if not _usable(config, state) or os.environ.get(FRESH_ENV):
        return 0
    entry = _read(config)["keys"].get(state.key)
    if not isinstance(entry, dict):
        return 0
    cached = set(entry.get("nodeids") or ())
    removed = [item for item in items if item.nodeid in cached]
    if not removed:
        return 0
    items[:] = [item for item in items if item.nodeid not in cached]
    config.hook.pytest_deselected(items=removed)
    state.reused = len(removed)
    state.previous_when = entry.get("when")
    workeroutput = getattr(config, "workeroutput", None)
    if workeroutput is not None:
        workeroutput[WORKEROUTPUT_KEY] = {"reused": state.reused, "when": state.previous_when}
    return state.reused


def record_pass_report(report) -> None:
    """``pytest_runtest_logreport``: note a phase's outcome, on the controller."""
    state = _state
    if state is None or state.is_worker:
        return
    phases = state.reports.setdefault(report.nodeid, {})
    phases[report.when] = bool(report.passed) and not hasattr(report, "wasxfail")


def note_worker_reused(node) -> None:
    """``pytest_testnodedown``: take a finished xdist worker's count."""
    state = _state
    if state is None:
        return
    info = (getattr(node, "workeroutput", None) or {}).get(WORKEROUTPUT_KEY)
    if not isinstance(info, dict):
        return
    reused = int(info.get("reused") or 0)
    if reused > state.reused:
        state.reused = reused
        state.previous_when = info.get("when")


def finish_pass_cache(session) -> None:
    """``pytest_sessionfinish``: write the entry and settle the exit status, on the controller.

    Must run before the conftest stashes ``session.exitstatus`` for the
    force-exit in ``pytest_unconfigure``: pytest reads ``session.exitstatus``
    back after this hook, so the override below is what the process exits with.
    """
    config = session.config
    state = getattr(config, "_vtsearch_pass_cache", None)
    if state is None or state.finished:
        return
    state.finished = True
    if state.is_worker or not _usable(config, state):
        return
    newly = passed_nodeids(state.reports)
    data = _read(config)
    entry = data["keys"].get(state.key)
    known = set(entry.get("nodeids") or ()) if isinstance(entry, dict) else set()
    state.newly_recorded = len(newly)
    data["keys"][state.key] = {"when": _now(), "nodeids": sorted(known | newly)}
    data["keys"] = prune(data["keys"])
    config.cache.set(CACHE_KEY, data)
    if state.reused and session.exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED:
        session.exitstatus = pytest.ExitCode.OK


def pass_cache_info(config) -> dict[str, Any]:
    """What the summary printer needs: ``reused``, ``previous_when``, ``newly_recorded``, ``active``."""
    state = getattr(config, "_vtsearch_pass_cache", None)
    if state is None:
        return {"reused": 0, "previous_when": None, "newly_recorded": 0, "active": False}
    return {
        "reused": state.reused,
        "previous_when": state.previous_when,
        "newly_recorded": state.newly_recorded,
        "active": _usable(config, state),
    }
