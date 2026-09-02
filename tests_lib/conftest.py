"""Pytest conftest for the *library-only* test suite under ``tests_lib/``.

These tests exercise the ``vtscore`` candidate subpackages without booting
Flask and without importing ``vtsearch`` at all — see ``tests_lib/__init__.py``
for the two gates that hold that line.  Everything the two suites
share — the group-marker hook, the fake embedders, the tmp-path widener, the
embedder stub fixture, the bulk of the reset fixture and the end-of-run summary
printer — lives in ``tests_shared`` and is imported by both conftests, so the
two can no longer drift (issue #3424).  What stays here is library-tier only:
the Flask blocker, the native-thread caps, and the library-only ``CoreConfig``
builder.  The app-tier fixtures (``client``, ``isolated_settings``,
``_set_login_provider``, autorun-processor reset) are deliberately absent.

Phase 7 of ``../vtscore/docs/architecture.md``.
"""

from __future__ import annotations

# Must run before ANY other import in this file: under ``-n auto`` the xdist
# workers are separate processes that inherit the environment but not the
# controller's ``sys.meta_path``, and this conftest is the first thing a
# worker imports.  Installing the Flask blocker here is what makes
# ``./run-tests.sh vtscore-clean`` actually cover the library code (the
# controller-side install in ``scripts/check-vtscore-clean.py`` would
# otherwise only see this file's own imports).  No-op unless the gate set
# ``VTSEARCH_BLOCK_FLASK``.
from tests_lib.flask_blocker import install_if_requested as _install_flask_blocker

_install_flask_blocker()

# Cap native math threads BEFORE numpy/torch are imported, mirroring the top of
# ``app.py``.  The app tier gets this for free (``tests/conftest.py`` imports
# ``app``), but a ``tests_lib``-only run never does, and the fallback
# (``ensure_torch_configured``) only fires from embedder paths the suite stubs
# out — so torch fell back to one intra-op thread *per core*.  Under ``-n auto``
# that is workers x cores native threads fighting over the same cores, and it
# is not a small effect: ``pytest tests_lib/detectors`` alone went 130s -> 51s
# on a 4-vCPU box when these were set.  Resolved the same way ``app.py`` does
# so an explicit ``VTSEARCH_TORCH_THREADS`` override still wins.
import os  # noqa: E402

_torch_threads = str(max(1, int(os.environ.get("VTSEARCH_TORCH_THREADS", "1"))))
os.environ.setdefault("OMP_NUM_THREADS", _torch_threads)
os.environ.setdefault("MKL_NUM_THREADS", _torch_threads)

from unittest.mock import patch  # noqa: E402

import pytest  # noqa: E402

import vtscore.config as config  # noqa: E402
from tests_shared.embedding_stubs import (  # noqa: E402
    fake_embed_audio as _fake_embed_audio,
    make_stub_embedding_models_fixture,
)
from tests_shared.pytest_plumbing import add_group_markers, print_summary_and_exit  # noqa: E402
from tests_shared.state_reset import (  # noqa: E402
    TEST_TRAIN_EPOCHS,  # noqa: F401  (re-exported: tests_lib/core/test_training_budget_isolation.py)
    allow_test_tmp_paths as _allow_test_tmp_paths,  # noqa: F401  (autouse fixture)
    freeze_startup_heap,
    install_startup_contexts,
    pin_training_budget,
    reset_shared_state,
)


def pytest_collection_modifyitems(items, config):
    """Auto-assign group markers based on the test file's parent directory."""
    add_group_markers(items, root_dir_name="tests_lib")


pin_training_budget()


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
        autofind_detectors=(),
        dataset_max_age_days=None,
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

_EMBED_AUDIO_FILE_TARGET = "tests_lib.fixtures.medias.embed_audio_file"
_patch_embed_audio = patch(_EMBED_AUDIO_FILE_TARGET, side_effect=_fake_embed_audio)
_patch_embed_audio.start()

install_startup_contexts()

from vtscore.media.audio.audio_generator import GENERATOR_SAMPLE_RATE  # noqa: F401, E402
from vtscore.media.audio.audio_generator import generate_wav  # noqa: F401, E402
from vtscore.embedding import initialize_models  # noqa: E402
from vtscore.state.core import get_active_context  # noqa: E402

from tests_lib.fixtures.medias import NUM_MEDIAS, init_medias  # noqa: F401, E402


initialize_models()
init_medias()

_test_medias_snapshot = {k: dict(v) for k, v in get_active_context().medias.items()}

_patch_embed_audio.stop()

freeze_startup_heap()

# Stub every registered media type and embedder for the whole session.
_stub_embedding_models = make_stub_embedding_models_fixture(_EMBED_AUDIO_FILE_TARGET)


@pytest.fixture(scope="session")
def aac_bytes():
    """A short AAC-in-MP4 (``.m4a``) clip, encoded once per session by ffmpeg.

    AAC is the codec ``libsndfile`` cannot parse at all, so anything decoding
    these bytes is exercising the ffmpeg arm of
    :func:`vtscore.media.audio.decode.decode_audio` — the one that replaced
    librosa's removed ``audioread`` fallback.
    """
    import subprocess

    from vtscore.media.audio.ffmpeg import get_ffmpeg_exe

    try:
        ffmpeg = get_ffmpeg_exe()
    except FileNotFoundError:
        pytest.skip("ffmpeg not available")
    result = subprocess.run(  # noqa: S603
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-c:a",
            "aac",
            "-f",
            "mp4",
            "-movflags",
            "frag_keyframe+empty_moov",
            "pipe:1",
        ],
        input=generate_wav(440.0, 1.0),
        capture_output=True,
        timeout=60,
    )
    if result.returncode != 0 or not result.stdout:
        pytest.skip(f"ffmpeg has no AAC encoder: {result.stderr[:200].decode(errors='replace')}")
    return result.stdout


@pytest.fixture(autouse=True)
def reset_contexts(tmp_path, monkeypatch):
    """Reset all library-tier mutable global state before each test.

    The bulk of it lives in :func:`tests_shared.state_reset.reset_shared_state`,
    which the app-tier ``reset_state`` calls too.  Only the library-tier extras
    are spelled out here: no login provider, no autorun processors, no settings
    file isolation (the library default ``CoreConfig`` builder above is stable
    for every test), and registry storage redirected to ``tmp_path`` so tests
    can't pollute the repo's ``data/`` tree.
    """
    from vtscore.datasets import registry as ds_reg_mod
    from vtscore.detectors import registry as det_reg_mod

    monkeypatch.setattr(ds_reg_mod, "REGISTRY_PATH", tmp_path / "dataset_registry.json")
    monkeypatch.setattr(det_reg_mod, "REGISTRY_PATH", tmp_path / "detector_registry.json")

    reset_shared_state(_test_medias_snapshot)

    # ``test_torch_config.py`` reloads ``vtscore.config`` to test env-var
    # behaviour, which wipes *every* module-level value this conftest installed
    # at import time.  That file restores its own snapshot now (issue #3101),
    # but re-assert the builder defensively, so any future reload - or any test
    # that writes to ``vtscore.config`` and forgets to restore - cannot silently
    # leave ``CoreConfig.from_settings()`` raising for the rest of the session.
    # (``reset_shared_state`` re-asserts the training budget for the same
    # reason.)
    config.register_core_config_builder(_lib_default_core_config)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    """Print the run summary and force-exit (see ``tests_shared``).

    Only runs when ``tests_lib/`` is the *sole* pytest session.  When both
    ``tests/`` and ``tests_lib/`` collect together (a single pytest
    invocation), the app-tier conftest's ``pytest_unconfigure`` handles the
    force-exit and prints the summary; exactly one of the two must.  Detected
    by looking for the app conftest among the registered plugins — when this
    conftest is loaded standalone (``pytest tests_lib/``) the app one is never
    imported.
    """
    plugin_names = {getattr(p, "__name__", "") for p in config.pluginmanager.get_plugins()}
    if any("tests.conftest" in n for n in plugin_names):
        return

    print_summary_and_exit(config, getattr(config, "_vtsearch_lib_exitstatus", 0))


def pytest_sessionfinish(session, exitstatus):
    """Stash the exit status so pytest_unconfigure can use it."""
    session.config._vtsearch_lib_exitstatus = exitstatus
