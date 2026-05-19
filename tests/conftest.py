from unittest.mock import patch

import numpy as np
import os

import pytest

import vtscore.config as config

# ---------------------------------------------------------------------------
# Auto-assign test group markers based on the test file's parent directory,
# so tests can be run by area: pytest -m core, pytest -m sorting, etc.
#
# Layout: tests/<group>/test_*.py — the folder name IS the group. New test
# files automatically inherit their group from where they live, so the old
# registry-drift bug class (a file added without an entry in _TEST_GROUPS
# was silently excluded from `./run-tests.sh <group>`) is gone.
# ---------------------------------------------------------------------------

# Folders that are not test groups (no marker should be added for items in them).
_NON_GROUP_DIRS = {"tests", "fixtures", "__pycache__"}


def pytest_collection_modifyitems(items, config):
    """Auto-assign group markers based on the test file's parent directory."""
    for item in items:
        parent = item.fspath.dirpath().basename
        if parent and parent not in _NON_GROUP_DIRS and not parent.startswith("_"):
            item.add_marker(getattr(pytest.mark, parent))


# Reduce training epochs for faster tests (default is 200; 30 is sufficient
# for the tiny MLP to converge on the small test dataset).
config.TRAIN_EPOCHS = 30

# ---------------------------------------------------------------------------
# Stub out heavy embedding models BEFORE importing the app.
#
# Tests don't need semantically meaningful embeddings — they just need
# deterministic vectors of the correct dimension (512 for CLAP audio).
# By patching embed_audio_file and the AudioMediaType's embed_text, we
# avoid loading the ~600 MB CLAP model, the ~100-200 MB librosa/numba
# stack, and the CLAP processor.  This cuts ~700-800 MB of RSS.
# ---------------------------------------------------------------------------

_EMBEDDING_DIM = 512


def _fake_embed_audio(arg):
    """Deterministic fake audio embedding derived from the file contents.

    Accepts either a path (from the legacy ``embed_audio_file`` wrapper) or a
    media dict (from ``MediaEmbedder.embed_media``).  Uses the first 1000
    bytes of the resolved file as a seed so that different audio files
    (even when written to the same temp path) produce distinct vectors.
    """
    import hashlib

    path = arg["media_path"] if isinstance(arg, dict) else arg
    try:
        with open(path, "rb") as f:
            data = f.read(1000)
        seed = int(hashlib.md5(data).hexdigest(), 16) % 2**31
    except Exception:
        seed = hash(str(path)) % 2**31
    rng = np.random.RandomState(seed)
    return rng.randn(_EMBEDDING_DIM).astype(np.float32)


def _fake_embed_text(text):
    """Deterministic fake text embedding derived from the query string."""
    import hashlib as _hl

    seed = int(_hl.md5(text.encode()).hexdigest(), 16) % 2**31
    rng = np.random.RandomState(seed)
    return rng.randn(_EMBEDDING_DIM).astype(np.float32)


# Patch embed_audio_file so init_medias() never triggers CLAP model loading.
_patch_embed_audio = patch("tests.fixtures.medias.embed_audio_file", side_effect=_fake_embed_audio)
_patch_embed_audio.start()

# Create a default dataset context so init_medias() has somewhere to write,
# and a default detector context so vote proxies have somewhere to delegate.
import vtscore.state.core as _state_core

_startup_ctx = _state_core.DatasetContext("_startup")
_state_core.register_context(_startup_ctx)
_state_core.set_thread_dataset_context(_startup_ctx)
_startup_det = _state_core.DetectorContext("_startup_det")
_state_core.register_detector_context(_startup_det)
_state_core.set_thread_detector_context(_startup_det)

import app as app_module

# Import refactored modules and make them accessible through app_module
from vtscore.media.audio.audio_generator import GENERATOR_SAMPLE_RATE
from tests.fixtures.medias import NUM_MEDIAS, init_medias
from vtscore.media.audio.audio_generator import generate_wav
from vtscore.embedding import initialize_models
from vtscore.detectors.training import train_and_score
from vtscore.detectors.labeling_progress import clear_progress_cache
from vtsearch.state import (
    bad_votes,
    medias,
    good_votes,
)

# Attach to app_module for backward compatibility with existing tests
app_module.NUM_MEDIAS = NUM_MEDIAS
app_module.SAMPLE_RATE = GENERATOR_SAMPLE_RATE
app_module.generate_wav = generate_wav
app_module.train_and_score = train_and_score
app_module.medias = medias
app_module.good_votes = good_votes
app_module.bad_votes = bad_votes
app_module.init_medias = init_medias  # legacy attribute used by some tests

# Initialize models and medias
initialize_models()
init_medias()

# Save the test medias so we can replay them into each test's fresh context.
_test_medias_snapshot = {k: dict(v) for k, v in medias.items()}

# Stop the module-level patch (init_medias is done); the per-test autouse
# fixture below re-applies the patches for every test so that /api/sort and
# other routes that call embed_text don't trigger CLAP loading either.
_patch_embed_audio.stop()

# Grab the audio media-type singleton and the audio embedder so the per-test
# fixture can patch embed_text/embed_media/load_models on both, preventing
# CLAP from loading during /api/sort and similar calls.
from vtscore.media import (
    all_embedders as _all_embedders,
    all_types as _all_types,
    get as _media_get,
    embedders_for_type as _embedders_for_type,
)

_audio_mt = _media_get("audio")
_audio_emb = _embedders_for_type("audio")[0]

# Every registered media type and embedder gets stubbed below — not just
# audio.  Tests that accidentally touch image/video/text/document embedders
# (e.g. via ``/api/sort`` on an image dataset) would otherwise try to
# download real CLIP / X-CLIP / E5 / SigLIP weights.
_ALL_EMBEDDERS = _all_embedders()
_ALL_MEDIA_TYPES = _all_types()


@pytest.fixture(autouse=True)
def _allow_test_tmp_paths(monkeypatch):
    """Allow tests to use system temp dirs with server file-path validation.

    In production, ``validate_server_filepath`` restricts paths to
    ``Path.cwd()``.  During tests, temp files live in the system temp
    directory, so we widen the check to also accept that tree.
    """
    import tempfile
    from pathlib import Path

    import vtscore.security.path_validation as paths_mod

    _original = paths_mod.validate_server_filepath

    def _permissive(filepath_str, base_dir=None):
        try:
            return _original(filepath_str, base_dir)
        except ValueError:
            # Also allow the system temp directory (where pytest tmp_path lives),
            # but only when base_dir was not explicitly set (i.e. only for the
            # default CWD fallback).  When a specific base_dir is given (e.g. in
            # multi-user mode) we must honour that restriction.
            if base_dir is not None:
                raise
            return _original(filepath_str, Path(tempfile.gettempdir()))

    monkeypatch.setattr(paths_mod, "validate_server_filepath", _permissive)


@pytest.fixture(scope="session", autouse=True)
def _stub_embedding_models():
    """Prevent any embedder from loading real model weights during tests.

    Session-scoped: the 40 patches are applied once and held for the entire
    run instead of being torn down and re-applied for each of the ~2900 tests.
    Tests that need different stub behavior can layer their own ``patch.object``
    on top — it will override the session-level patch and restore it on exit.
    """
    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(patch("tests.fixtures.medias.embed_audio_file", side_effect=_fake_embed_audio))
    for mt in _ALL_MEDIA_TYPES:
        stack.enter_context(patch.object(mt, "embed_text", side_effect=_fake_embed_text))
        stack.enter_context(patch.object(mt, "load_models"))
    for emb in _ALL_EMBEDDERS:
        stack.enter_context(patch.object(emb, "embed_media", side_effect=_fake_embed_audio))
        stack.enter_context(patch.object(emb, "embed_text", side_effect=_fake_embed_text))
        stack.enter_context(patch.object(emb, "load_models"))
    yield
    stack.close()


import vtscore.state.core as _core
from vtscore.concurrency.progress import (
    dataset_progress as _dataset_progress,
    eval_progress as _eval_progress,
    find_progress as _find_progress,
    loading_tasks as _loading_tasks,
    detector_loading_tasks as _model_loading_tasks,
    sort_progress as _sort_progress,
)
from vtsearch.auth import DefaultLoginProvider as _DefaultLoginProvider, set_login_provider as _set_login_provider
from vtscore.datasets.registry import reset_for_tests as _reset_ds_reg
from vtscore.detectors.registry import reset_for_tests as _reset_model_reg


@pytest.fixture(autouse=True)
def reset_state():
    """Reset all mutable global state before each test."""
    _core.clear_all_contexts()
    default_ctx = _core.DatasetContext("_test_default")
    _core.register_context(default_ctx)
    _core.set_thread_dataset_context(default_ctx)

    _core.clear_all_detector_contexts()
    default_det = _core.DetectorContext("_test_default_det")
    _core.register_detector_context(default_det)
    _core.set_thread_detector_context(default_det)

    medias.update({k: dict(v) for k, v in _test_medias_snapshot.items()})

    from vtsearch.autorun_processors import clear_all_autorun

    clear_all_autorun()
    clear_progress_cache()

    from vtscore.embedding.helpers import clear_text_query_cache as _clear_query_cache

    _clear_query_cache()

    _dataset_progress.reset_cancel()
    _find_progress.update("idle", "", 0, 0, step=None, total_steps=None, error=None)
    _sort_progress.update("idle", "", 0, 0, step=None, total_steps=None, error=None)
    _eval_progress.update("idle", "", 0, 0, step=None, total_steps=None, error=None)
    _loading_tasks.reset_for_tests()
    _model_loading_tasks.reset_for_tests()

    from vtscore.concurrency.async_jobs import reset_all_async_jobs_for_tests

    reset_all_async_jobs_for_tests()

    # Cancel any debounced labelset-source push left over from the
    # previous test so its captured contexts don't fire after this
    # test's reset_state has dropped them.
    from vtscore.labels.sync import reset_label_sync_for_tests

    reset_label_sync_for_tests()

    # Reset CLI progress format so a test that flips it to "json" can't
    # leak the choice into the next test.
    from vtscore import cli_progress

    cli_progress.set_format("text")

    _set_login_provider(_DefaultLoginProvider())

    _reset_ds_reg()
    _reset_model_reg()

    # ``test_torch_config.py`` reloads ``vtscore.config`` to test env-var
    # behaviour, which wipes the module-level ``_core_config_builder``
    # installed at app startup.  Re-register defensively so any later test
    # that calls ``CoreConfig.from_settings()`` (e.g. via ``get_inclusion()``)
    # still has a backing implementation.
    from vtsearch.shim import register_app_config_builder

    register_app_config_builder()


class _MergedSettingsPath:
    """Path-like helper that bridges the two settings tiers in tests.

    Settings used to live in a single ``data/settings.json`` file, and
    test code reads/writes it via ``isolated_settings.read_text()`` /
    ``.write_text(...)``.  With the two-tier layout, per-user keys live
    in ``users/default/user_settings.json`` while server-tier keys stay
    in ``settings.json``. This wrapper preserves the legacy API by:

    * ``read_text()`` — returns the merged JSON of both files (per-user
      values win over server values on key collisions, which matches the
      runtime behaviour of ``settings.get_all()``).
    * ``write_text(text)`` — parses *text* as JSON, splits the keys by
      tier, and writes each subset to the right file. Non-JSON text is
      treated as a "corrupt the file" probe and goes to the server file
      so the existing corruption tests still see invalid JSON on load.
    * ``exists()`` / ``__fspath__()`` / ``__str__()`` resolve to the
      server-tier file for the rare tests that pass the fixture to APIs
      expecting a real path.
    """

    def __init__(self, server_path, user_default_path) -> None:
        self._server = server_path
        self._user = user_default_path

    def read_text(self, *args, **kwargs) -> str:
        import json as _json

        merged: dict = {}
        for p in (self._server, self._user):
            if p.exists():
                try:
                    data = _json.loads(p.read_text(*args, **kwargs))
                    if isinstance(data, dict):
                        merged.update(data)
                except Exception:
                    pass
        return _json.dumps(merged)

    def write_text(self, content: str, *args, **kwargs) -> int:
        import json as _json
        from vtsearch import settings as _settings

        try:
            data = _json.loads(content)
        except Exception:
            # Non-JSON content: simulate a corrupt server-tier file.
            self._server.parent.mkdir(parents=True, exist_ok=True)
            return self._server.write_text(content, *args, **kwargs)

        if not isinstance(data, dict):
            self._server.parent.mkdir(parents=True, exist_ok=True)
            return self._server.write_text(content, *args, **kwargs)

        server_data = {k: v for k, v in data.items() if k in _settings._SERVER_KEYS}
        user_data = {k: v for k, v in data.items() if k not in _settings._SERVER_KEYS}

        self._server.parent.mkdir(parents=True, exist_ok=True)
        self._server.write_text(_json.dumps(server_data, indent=2) + "\n")
        self._user.parent.mkdir(parents=True, exist_ok=True)
        self._user.write_text(_json.dumps(user_data, indent=2) + "\n")
        # Force the in-memory caches to re-read on next access.
        _settings.reset()
        return len(content)

    def exists(self) -> bool:
        return self._server.exists() or self._user.exists()

    def __fspath__(self) -> str:
        return str(self._server)

    def __str__(self) -> str:
        return str(self._server)


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Redirect both settings tiers to a temp directory for each test.

    Without this, tests that write settings (inclusion, safe_thresholds,
    volume, etc.) would mutate the shared ``data/settings.json`` and the
    per-user files under ``data/<user>/user_settings.json``, leaking
    values into subsequent tests that lazy-load from those files.

    Yields a :class:`_MergedSettingsPath` so legacy tests that call
    ``isolated_settings.read_text()`` / ``.write_text(...)`` continue to
    see a merged view of both tiers.
    """
    from vtsearch import settings as settings_mod

    test_settings_path = tmp_path / "settings.json"
    user_default_path = tmp_path / "users" / "default" / "user_settings.json"
    monkeypatch.setattr(settings_mod, "SETTINGS_PATH", test_settings_path)
    # Redirect per-user settings files under tmp_path/users/<username>/.
    settings_mod.set_user_data_dir_override(tmp_path / "users")
    settings_mod.reset()

    # Also redirect dataset and detector registries to temp paths
    from vtscore.datasets import registry as ds_reg_mod
    from vtscore.detectors import registry as det_reg_mod

    monkeypatch.setattr(ds_reg_mod, "REGISTRY_PATH", tmp_path / "dataset_registry.json")
    monkeypatch.setattr(det_reg_mod, "REGISTRY_PATH", tmp_path / "detector_registry.json")

    # Redirect storage directories to temp paths via settings
    settings_mod.set_saved_datasets_dir(str(tmp_path / "saved_datasets"))
    settings_mod.set_detectors_dir(str(tmp_path / "detectors"))

    ds_reg_mod.reset_for_tests()
    det_reg_mod.reset_for_tests()

    yield _MergedSettingsPath(test_settings_path, user_default_path)
    settings_mod.reset()
    settings_mod.set_user_data_dir_override(None)
    ds_reg_mod.reset_for_tests()
    det_reg_mod.reset_for_tests()


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


def _wait_for_job(job_manager, *, timeout: float = 30.0) -> None:
    """Block until the running job (and any coalesced pending follow-up) finish.

    With the coalescing job manager, a ``start()`` issued while a job is
    running ends up in the pending slot and gets promoted automatically
    when the runner finishes.  Tests that issue several requests need to
    wait for the whole chain to drain, not just the first job.
    """
    import time as _time

    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        job = job_manager.current()
        if job is None:
            return
        if job.status in ("running", "pending"):
            job.done_event.wait(timeout=0.05)
            continue
        # Current is finished.  Yield briefly so any pending promotion
        # spawned from _run can replace current before we re-check.
        _time.sleep(0.01)
        follow = job_manager.current()
        if follow is None or follow.job_id == job.job_id:
            return
    raise TimeoutError(f"Job manager did not drain within {timeout}s")


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    """Force-exit to avoid SIGABRT (exit code 134) from native library cleanup.

    PyTorch, OpenMP, numba (via librosa), and other native libraries spin up
    C++ thread pools that sometimes call ``std::terminate()`` during Python
    interpreter shutdown.  This produces "terminate called without an active
    exception" and exit code 134, even though all tests passed.

    ``os._exit()`` skips the normal interpreter teardown (atexit handlers,
    C++ static destructors) so the problematic cleanup never runs.

    We use ``pytest_unconfigure`` (the very last hook) instead of
    ``pytest_sessionfinish`` so that ``pytest_terminal_summary`` still runs
    first — ensuring failure tracebacks and the short test summary are fully
    printed before we force-exit.

    Prints an additional PASS/FAIL summary right before exiting for clarity.
    """
    import sys

    exitstatus = getattr(config, "_vtsearch_exitstatus", 0)

    reporter = config.pluginmanager.getplugin("terminalreporter")
    print("", flush=True)
    print("=" * 60, flush=True)
    if reporter:
        passed = len(reporter.stats.get("passed", []))
        failed = len(reporter.stats.get("failed", []))
        errors = len(reporter.stats.get("error", []))
        skipped = len(reporter.stats.get("skipped", []))
        xfailed = len(reporter.stats.get("xfailed", []))
        total = passed + failed + errors + skipped

        if failed or errors:
            parts = [f"{failed} failed", f"{errors} errors", f"{passed} passed", f"{skipped} skipped"]
            if xfailed:
                parts.append(f"{xfailed} xfailed")
            print(f"TESTS FAILED: {', '.join(parts)} (total: {total})", flush=True)
        else:
            extra = f"{skipped} skipped"
            if xfailed:
                extra += f", {xfailed} xfailed"
            print(f"ALL {passed} TESTS PASSED ({extra}, total: {total})", flush=True)
    else:
        status = "PASSED" if exitstatus == 0 else "FAILED"
        print(f"TESTS {status} (exit code {exitstatus}; reporter unavailable)", flush=True)
    print("=" * 60, flush=True)

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exitstatus)


def pytest_sessionfinish(session, exitstatus):
    """Stash the exit status so pytest_unconfigure can use it."""
    session.config._vtsearch_exitstatus = exitstatus
