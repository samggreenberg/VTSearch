"""Pytest conftest for the *library-only* test suite under ``tests_lib/``.

These tests exercise the ``vtscore`` candidate subpackages without
booting Flask, ``vtsearch.app``, or ``vtsearch.settings``.  The fixtures
here mirror the behaviour of ``tests/conftest.py`` for the library
seams (context reset, embedding stubs, progress reset, …) and
deliberately omit the app-tier ones (``client``, ``isolated_settings``,
``_set_login_provider``, autorun-processor reset, etc.).

Phase 7 of ``../vtscore/docs/architecture.md``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import os
import pytest

import vtscore.config as config


_NON_GROUP_DIRS = {"tests_lib", "fixtures", "__pycache__"}


def pytest_collection_modifyitems(items, config):
    """Auto-assign group markers based on the test file's parent directory."""
    for item in items:
        parent = item.fspath.dirpath().basename
        if parent and parent not in _NON_GROUP_DIRS and not parent.startswith("_"):
            item.add_marker(getattr(pytest.mark, parent))


config.TRAIN_EPOCHS = 30


# ---------------------------------------------------------------------------
# Library-only CoreConfig builder.
#
# Library code that goes through ``CoreConfig.from_settings()`` (e.g.
# ``vtsearch/datasets/registry.py:get_saved_datasets_dir()``) needs *some*
# builder installed or it raises RuntimeError.  Outside the Flask app we
# install a stable default that points at the repo's ``data/`` tree and
# uses sensible literals for every knob.  Tests that need a different
# value can monkey-patch ``vtscore.config._core_config_builder`` or call
# :func:`register_core_config_builder` themselves.
# ---------------------------------------------------------------------------


def _lib_default_core_config(_settings_path=None):
    data_dir = config.DATA_DIR
    return config.CoreConfig(
        saved_datasets_dir=data_dir / "saved_datasets",
        detectors_dir=data_dir / "detectors",
        max_concurrent_dataset_downloads=1,
        max_concurrent_dataset_embeddings=1,
        autorun_detectors=(),
        safe_thresholds=False,
        calibrate_count=10,
        calibration_fraction=0.1,
        enrich_descriptions=False,
        autopilot_goal_diversity=8,
        inclusion=0,
        data_dir=data_dir,
    )


config.register_core_config_builder(_lib_default_core_config)


# ---------------------------------------------------------------------------
# Stub heavy embedders BEFORE importing anything that touches the registry.
# ---------------------------------------------------------------------------

_EMBEDDING_DIM = 512


def _fake_embed_audio(arg):
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
    import hashlib as _hl

    seed = int(_hl.md5(text.encode()).hexdigest(), 16) % 2**31
    rng = np.random.RandomState(seed)
    return rng.randn(_EMBEDDING_DIM).astype(np.float32)


_patch_embed_audio = patch("tests_lib.fixtures.medias.embed_audio_file", side_effect=_fake_embed_audio)
_patch_embed_audio.start()

import vtscore.state.core as _state_core

_startup_ctx = _state_core.DatasetContext("_startup")
_state_core.register_context(_startup_ctx)
_state_core.set_thread_dataset_context(_startup_ctx)
_startup_det = _state_core.DetectorContext("_startup_det")
_state_core.register_detector_context(_startup_det)
_state_core.set_thread_detector_context(_startup_det)

from vtscore.media.audio.audio_generator import GENERATOR_SAMPLE_RATE  # noqa: F401, E402
from vtscore.media.audio.audio_generator import generate_wav  # noqa: F401, E402
from vtscore.embedding import initialize_models  # noqa: E402
from vtscore.detectors.labeling_progress import clear_progress_cache  # noqa: E402
from vtsearch.state import medias  # noqa: E402

from tests_lib.fixtures.medias import NUM_MEDIAS, init_medias  # noqa: F401, E402


initialize_models()
init_medias()

_test_medias_snapshot = {k: dict(v) for k, v in medias.items()}

_patch_embed_audio.stop()

from vtscore.media import (  # noqa: E402
    all_embedders as _all_embedders,
    all_types as _all_types,
)

_ALL_EMBEDDERS = _all_embedders()
_ALL_MEDIA_TYPES = _all_types()


@pytest.fixture(autouse=True)
def _allow_test_tmp_paths(monkeypatch):
    """Widen ``validate_server_filepath`` to also accept the system temp tree."""
    import tempfile

    import vtscore.security.path_validation as paths_mod

    _original = paths_mod.validate_server_filepath

    def _permissive(filepath_str, base_dir=None):
        try:
            return _original(filepath_str, base_dir)
        except ValueError:
            if base_dir is not None:
                raise
            return _original(filepath_str, Path(tempfile.gettempdir()))

    monkeypatch.setattr(paths_mod, "validate_server_filepath", _permissive)


@pytest.fixture(scope="session", autouse=True)
def _stub_embedding_models():
    """Prevent any embedder from loading real model weights during tests."""
    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(patch("tests_lib.fixtures.medias.embed_audio_file", side_effect=_fake_embed_audio))
    for mt in _ALL_MEDIA_TYPES:
        stack.enter_context(patch.object(mt, "embed_text", side_effect=_fake_embed_text))
        stack.enter_context(patch.object(mt, "load_models"))
    for emb in _ALL_EMBEDDERS:
        stack.enter_context(patch.object(emb, "embed_media", side_effect=_fake_embed_audio))
        stack.enter_context(patch.object(emb, "embed_text", side_effect=_fake_embed_text))
        stack.enter_context(patch.object(emb, "load_models"))
    yield
    stack.close()


import vtscore.state.core as _core  # noqa: E402
from vtscore.concurrency.progress import (  # noqa: E402
    dataset_progress as _dataset_progress,
    eval_progress as _eval_progress,
    find_progress as _find_progress,
    loading_tasks as _loading_tasks,
    detector_loading_tasks as _model_loading_tasks,
    sort_progress as _sort_progress,
)
from vtscore.datasets.registry import reset_for_tests as _reset_ds_reg  # noqa: E402
from vtscore.detectors.registry import reset_for_tests as _reset_model_reg  # noqa: E402


@pytest.fixture(autouse=True)
def reset_contexts(tmp_path, monkeypatch):
    """Reset all library-tier mutable global state before each test.

    Smaller cousin of the app-tier ``reset_state`` fixture in
    ``tests/conftest.py`` — only touches state that lives under
    library-candidate packages.  No login provider, no autorun
    processors, no settings file isolation (the library default
    CoreConfig builder above is stable for every test).
    """
    _core.clear_all_contexts()
    default_ctx = _core.DatasetContext("_test_default")
    _core.register_context(default_ctx)
    _core.set_thread_dataset_context(default_ctx)

    _core.clear_all_detector_contexts()
    default_det = _core.DetectorContext("_test_default_det")
    _core.register_detector_context(default_det)
    _core.set_thread_detector_context(default_det)

    medias.update({k: dict(v) for k, v in _test_medias_snapshot.items()})

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

    from vtscore.labels.sync import reset_label_sync_for_tests

    reset_label_sync_for_tests()

    from vtscore import cli_progress

    cli_progress.set_format("text")

    # Redirect registry storage to tmp_path so tests can't pollute repo data/.
    from vtscore.datasets import registry as ds_reg_mod
    from vtscore.detectors import registry as det_reg_mod

    monkeypatch.setattr(ds_reg_mod, "REGISTRY_PATH", tmp_path / "dataset_registry.json")
    monkeypatch.setattr(det_reg_mod, "REGISTRY_PATH", tmp_path / "detector_registry.json")

    _reset_ds_reg()
    _reset_model_reg()

    # ``test_torch_config.py`` reloads ``vtscore.config`` to test env-var
    # behaviour, which wipes the module-level builder installed at import
    # time.  Re-register defensively so any later test that calls
    # ``CoreConfig.from_settings()`` still has a backing implementation.
    config.register_core_config_builder(_lib_default_core_config)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    """Force-exit to avoid SIGABRT from native library cleanup at shutdown.

    Same rationale as in ``tests/conftest.py`` — PyTorch / OpenMP / numba
    spin up C++ thread pools whose teardown can call ``std::terminate``
    during interpreter shutdown, producing exit code 134 even though all
    tests passed.  ``os._exit`` skips atexit/static-destructor teardown.

    Only runs when ``tests_lib/`` is the *sole* pytest session — when both
    ``tests/`` and ``tests_lib/`` collect together (a single pytest
    invocation), the app-tier conftest's ``pytest_unconfigure`` handles
    the force-exit and prints the summary.
    """
    import sys

    # If the app-tier conftest is loaded (i.e. pytest is collecting both
    # trees), skip — its ``pytest_unconfigure`` already does the work.
    if any("tests/conftest" in str(p) for p in getattr(config, "_inifile", None) and [config._inifile] or []):
        return
    # Heuristic: only one of the two confests should print the summary.
    # When this conftest is loaded standalone (pytest tests_lib/), the
    # app conftest never gets imported.  Check by looking at registered
    # plugin names.
    plugin_names = {getattr(p, "__name__", "") for p in config.pluginmanager.get_plugins()}
    if any("tests.conftest" in n for n in plugin_names):
        return

    exitstatus = getattr(config, "_vtsearch_lib_exitstatus", 0)
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
    session.config._vtsearch_lib_exitstatus = exitstatus
