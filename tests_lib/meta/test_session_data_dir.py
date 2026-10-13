"""Each pytest session runs on a throwaway ``VTSEARCH_DATA_DIR`` (issue #4706).

``tests_shared/data_dir.py`` is what keeps a suite run out of the live app's
data on the GRID, where ``~/.bashrc`` exports the variable, and out of the
developer's own ``data/`` on a laptop.  The first test checks the session this
file is running in, which is what fails if a conftest stops calling the guard,
or calls it after ``vtscore.config`` has been imported.  The next runs a whole
pytest session in a subprocess, because a run ends in ``os._exit()``, which
skips the atexit cleanup.  The rest run the guard alone in a fresh interpreter,
since in this process ``vtscore.config`` was imported long ago.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import vtscore.config as config
from tests_shared.data_dir import MARKER

REPO_ROOT = Path(__file__).resolve().parents[2]

# Prints the guard's dir, whether it exists, and the models dir it left behind.
_PROBE = (
    "import os, sys\n"
    "from tests_shared.data_dir import isolate_data_dir\n"
    "p = isolate_data_dir()\n"
    "assert isolate_data_dir() == p, 'a second call made a second dir'\n"
    "print(p); print(p.is_dir()); print(os.environ.get('VTSEARCH_MODELS_DIR', 'unset'))\n"
)


def _fresh_env(**env: str) -> dict[str, str]:
    """This process's environment minus the session it belongs to, plus ``env``."""
    ours = ("VTSEARCH_DATA_DIR", "VTSEARCH_MODELS_DIR", MARKER, "PYTEST_CURRENT_TEST")
    base = {k: v for k, v in os.environ.items() if k not in ours and not k.startswith("PYTEST_XDIST_")}
    return {**base, **env}


def _probe(env: dict[str, str], code: str = _PROBE) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603  # fixed argv, no shell
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=_fresh_env(**env),
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_this_session_runs_on_its_own_data_dir() -> None:
    isolated = Path(os.environ[MARKER])
    assert os.environ["VTSEARCH_DATA_DIR"] == str(isolated)
    assert config.DATA_DIR == isolated
    assert isolated.name.startswith("vtsearch-test-data-")
    assert config.DATA_DIR != REPO_ROOT / "data"


def _listing(d: Path) -> list[tuple[str, int]]:
    """What ``ls -la`` of ``d`` shows that a write would change: names and mtimes."""
    return [(".", d.stat().st_mtime_ns)] + sorted((p.name, p.stat().st_mtime_ns) for p in d.iterdir())


def test_a_pytest_run_leaves_no_data_dir_behind(tmp_path: Path) -> None:
    """End to end, through the conftest and the summary's force-exit.

    The inherited dir has a model cache, as the live one on the GRID does, so
    the session reads models from it; nothing in its listing may change.
    """
    live = tmp_path / "live-data"
    (live / "models").mkdir(parents=True)
    before = _listing(live)
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    r = subprocess.run(  # noqa: S603  # fixed argv, no shell
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            f"{Path(__file__).relative_to(REPO_ROOT)}::test_this_session_runs_on_its_own_data_dir",
        ],
        cwd=REPO_ROOT,
        env=_fresh_env(VTSEARCH_DATA_DIR=str(live), TMPDIR=str(scratch)),
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "1 passed" in r.stdout
    assert _listing(live) == before, "the run wrote into the inherited data dir"
    assert not list(scratch.glob("vtsearch-test-data-*")), "the session's data dir outlived the run"


@pytest.mark.parametrize("has_models", [True, False])
def test_an_inherited_data_dir_is_replaced_and_left_untouched(tmp_path: Path, has_models: bool) -> None:
    live = tmp_path / "live-data"
    live.mkdir()
    if has_models:
        (live / "models").mkdir()
    before = _listing(live)
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    r = _probe({"VTSEARCH_DATA_DIR": str(live), "TMPDIR": str(scratch)})
    assert r.returncode == 0, r.stderr
    used, existed, models = r.stdout.splitlines()
    assert Path(used).parent == scratch
    assert existed == "True"
    # Models are a download cache: read from where the app would, if it has one.
    assert models == (str(live / "models") if has_models else "unset")
    assert not Path(used).exists(), "the session's data dir outlived the session"
    assert _listing(live) == before


def test_with_no_data_dir_set_the_checkout_is_what_is_replaced(tmp_path: Path) -> None:
    r = _probe({"TMPDIR": str(tmp_path)})
    assert r.returncode == 0, r.stderr
    used, _, models = r.stdout.splitlines()
    assert Path(used).parent == tmp_path
    checkout_models = REPO_ROOT / "data" / "models"
    assert models == (str(checkout_models) if checkout_models.is_dir() else "unset")


def test_an_explicit_models_dir_wins(tmp_path: Path) -> None:
    r = _probe({"VTSEARCH_MODELS_DIR": str(tmp_path / "shared-models"), "TMPDIR": str(tmp_path)})
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines()[2] == str(tmp_path / "shared-models")


def test_an_inheriting_process_reuses_the_dir_and_does_not_remove_it(tmp_path: Path) -> None:
    """An xdist worker, or a subprocess a test starts: same dir, and its exit leaves it."""
    session = tmp_path / "vtsearch-test-data-x"
    session.mkdir()
    # The creating process exported the models dir too; an inheritor has it.
    r = _probe({"VTSEARCH_DATA_DIR": str(session), MARKER: str(session), "VTSEARCH_MODELS_DIR": "m"})
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines()[0] == str(session)
    assert session.is_dir()


@pytest.mark.parametrize("marker", ["gone", "other"])
def test_a_stale_or_foreign_marker_gets_a_fresh_dir(tmp_path: Path, marker: str) -> None:
    current = tmp_path / "data"
    if marker == "other":
        current.mkdir()
    env = {"VTSEARCH_DATA_DIR": str(current), "TMPDIR": str(tmp_path)}
    env[MARKER] = str(current) if marker == "gone" else str(tmp_path / "elsewhere")
    r = _probe(env)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines()[0] != str(current)


def test_too_late_a_call_fails_loudly(tmp_path: Path) -> None:
    """A conftest that imports vtscore.config first would leave the redirect unseen."""
    code = (
        "import sys, types\n"
        f"sys.modules['vtscore.config.paths'] = types.SimpleNamespace(DATA_DIR={str(tmp_path)!r})\n"
        "from tests_shared.data_dir import isolate_data_dir\n"
        "isolate_data_dir()\n"
    )
    r = _probe({"VTSEARCH_DATA_DIR": str(tmp_path), "TMPDIR": str(tmp_path)}, code)
    assert r.returncode != 0
    assert "imported before isolate_data_dir()" in r.stderr
