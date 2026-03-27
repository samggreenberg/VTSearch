from unittest.mock import patch

import numpy as np
import os

import pytest

import vtsearch.config as config

# ---------------------------------------------------------------------------
# Auto-assign test group markers based on filename so tests can be run by
# area: pytest -m core, pytest -m sorting, pytest -m datasets, etc.
# ---------------------------------------------------------------------------

_TEST_GROUPS = {
    "core": [
        "test_audio",
        "test_medias",
        "test_votes",
        "test_inclusion",
        "test_settings",
        "test_frontend",
    ],
    "api": [
        "test_api_contracts",
        "test_error_recovery",
        "test_dashboard",
        "test_file_browser",
        "test_path_validation",
        "test_multi_user_security",
        "test_multi_user_dataset_access",
        "test_ssrf_validation",
    ],
    "sorting": [
        "test_sorting",
        "test_label_sorting",
        "test_safe_thresholds",
        "test_enrich_descriptions",
        "test_diversity_tree",
        "test_diversity_tree_integration",
    ],
    "datasets": [
        "test_datasets",
        "test_dataset_split",
        "test_combine_datasets",
        "test_creation_info",
        "test_duplicates",
        "test_origin_labelset",
        "test_thin_loading",
        "test_chunked_loading",
        "test_memory_errors",
        "test_pickle_safety",
        "test_media_sources",
        "test_multi_dataset",
        "test_request_context",
        "test_parallel_loading",
    ],
    "io": [
        "test_exporters",
        "test_csv_webhook_exporters",
        "test_export_options",
        "test_importers",
        "test_label_importers",
        "test_label_import_endpoint",
        "test_label_import_ingestion",
        "test_labels",
        "test_processor_importers",
        "test_pdf_import",
        "test_corrections_export",
    ],
    "models": [
        "test_detectors",
        "test_extractors",
        "test_processors",
        "test_trainable_models",
        "test_multi_detector",
        "test_clippers",
        "test_eval",
        "test_eval_visualize",
        "test_eval_voting_iterations",
        "test_resolver",
        "test_new_embedders",
    ],
    "downloads": [
        "test_ag_news_download",
        "test_bbc_news_download",
        "test_gtzan_download",
        "test_image_sources_download",
        "test_imdb_download",
        "test_ucsf_documents_download",
        "test_download_and_extract",
    ],
    "integration": [
        "test_integration",
        "test_slow_integration",
        "test_thread_safety",
    ],
    "cli": [
        "test_cli_autodetect",
        "test_load_sort_window",
        "test_preload_progress",
        "test_tqdm_progress",
    ],
    "converters": [
        "test_document_and_converters",
        "test_converter_selection",
    ],
}

# Build reverse map: filename -> group name
_FILE_TO_GROUP = {}
for group, files in _TEST_GROUPS.items():
    for fname in files:
        _FILE_TO_GROUP[fname] = group


def pytest_collection_modifyitems(items, config):
    """Auto-assign group markers to tests based on their filename."""
    for item in items:
        # Extract test_xxx from the file path
        fname = item.fspath.purebasename  # e.g. "test_sorting"
        group = _FILE_TO_GROUP.get(fname)
        if group:
            item.add_marker(getattr(pytest.mark, group))


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


def _fake_embed_audio(path):
    """Deterministic fake audio embedding derived from the file contents.

    Uses the first 1000 bytes of the file as a seed so that different audio
    files (even when written to the same temp path) produce distinct vectors.
    """
    import hashlib

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
_patch_embed_audio = patch("vtsearch.medias.embed_audio_file", side_effect=_fake_embed_audio)
_patch_embed_audio.start()

# Create a default dataset context so init_medias() has somewhere to write,
# and a default detector context so vote proxies have somewhere to delegate.
import vtsearch.utils.state_core as _state_core
_startup_ctx = _state_core.DatasetContext("_startup")
_state_core.register_context(_startup_ctx)
_state_core.set_thread_dataset_context(_startup_ctx)
_startup_det = _state_core.DetectorContext("_startup_det")
_state_core.register_detector_context(_startup_det)
_state_core.set_thread_detector_context(_startup_det)

import app as app_module

# Import refactored modules and make them accessible through app_module
from vtsearch.audio.generator import GENERATOR_SAMPLE_RATE
from vtsearch.medias import NUM_MEDIAS
from vtsearch.audio import generate_wav
from vtsearch.models import initialize_models, train_and_score
from vtsearch.models.progress import clear_progress_cache
from vtsearch.utils import (
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

# Initialize models and medias
initialize_models()
app_module.init_medias()

# Save the test medias so we can replay them into each test's fresh context.
_test_medias_snapshot = {k: dict(v) for k, v in medias.items()}

# Stop the module-level patch (init_medias is done); the per-test autouse
# fixture below re-applies the patches for every test so that /api/sort and
# other routes that call embed_text don't trigger CLAP loading either.
_patch_embed_audio.stop()

# Grab the audio media-type singleton and the audio embedder so the per-test
# fixture can patch embed_text/embed_media/load_models on both, preventing
# CLAP from loading during /api/sort and similar calls.
from vtsearch.media import get as _media_get, embedders_for_type as _embedders_for_type

_audio_mt = _media_get("audio")
_audio_emb = _embedders_for_type("audio")[0]


@pytest.fixture(autouse=True)
def _allow_test_tmp_paths(monkeypatch):
    """Allow tests to use system temp dirs with server file-path validation.

    In production, ``validate_server_filepath`` restricts paths to
    ``Path.cwd()``.  During tests, temp files live in the system temp
    directory, so we widen the check to also accept that tree.
    """
    import tempfile
    from pathlib import Path

    import vtsearch.utils.paths as paths_mod

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


@pytest.fixture(autouse=True)
def _stub_embedding_models():
    """Prevent CLAP (and librosa) from loading during individual tests.

    Patches ``embed_audio_file``, the audio media-type's ``embed_text``,
    ``embed_media``, and ``load_models`` so that any test-time calls
    (e.g. via ``/api/sort``) return cheap deterministic fake vectors
    instead of loading a ~600 MB model.
    """
    with (
        patch("vtsearch.medias.embed_audio_file", side_effect=_fake_embed_audio),
        patch.object(_audio_mt, "embed_text", side_effect=_fake_embed_text),
        patch.object(_audio_mt, "embed_media", side_effect=_fake_embed_audio),
        patch.object(_audio_mt, "load_models"),
        patch.object(_audio_emb, "embed_media", side_effect=_fake_embed_audio),
        patch.object(_audio_emb, "embed_text", side_effect=_fake_embed_text),
        patch.object(_audio_emb, "load_models"),
    ):
        yield


@pytest.fixture(autouse=True)
def reset_state():
    """Reset all mutable global state before each test.

    This fixture prevents cross-test contamination by clearing votes,
    autorun entries, and all other mutable state that lives in
    ``vtsearch.utils.state``.  It runs automatically before every test.
    """
    import vtsearch.utils.state_core as _core

    # Clear all dataset contexts and create a fresh default context so that
    # tests that just write to ``medias`` etc. still work.
    _core.clear_all_contexts()
    default_ctx = _core.DatasetContext("_test_default")
    _core.register_context(default_ctx)
    _core.set_thread_dataset_context(default_ctx)

    # Clear all detector contexts and create a fresh default context so that
    # tests that just write to ``good_votes`` / ``bad_votes`` etc. still work.
    _core.clear_all_detector_contexts()
    default_det = _core.DetectorContext("_test_default_det")
    _core.register_detector_context(default_det)
    _core.set_thread_detector_context(default_det)

    # Replay the test medias into the fresh context (medias is intentionally
    # NOT reset between tests to avoid expensive re-generation).
    medias.update({k: dict(v) for k, v in _test_medias_snapshot.items()})

    # Clear global (non-per-dataset) state.
    _core.autorun_detectors.clear()
    _core.autorun_extractors.clear()
    _core.autorun_localizers.clear()
    clear_progress_cache()

    # Reset progress trackers
    from vtsearch.utils.progress import dataset_progress, eval_progress, find_progress, loading_tasks, model_loading_tasks, sort_progress

    dataset_progress.reset_cancel()
    find_progress.update("idle", "", 0, 0, step=None, total_steps=None, error=None)
    sort_progress.update("idle", "", 0, 0)
    eval_progress.update("idle", "", 0, 0)
    loading_tasks.reset_for_tests()
    model_loading_tasks.reset_for_tests()

    # Reset the login provider to DefaultLoginProvider
    from vtsearch.auth import DefaultLoginProvider, set_login_provider

    set_login_provider(DefaultLoginProvider())

    # Reset the dataset and model registries
    from vtsearch.datasets.registry import reset_for_tests as _reset_ds_reg
    from vtsearch.models.registry import reset_for_tests as _reset_model_reg

    _reset_ds_reg()
    _reset_model_reg()


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Redirect the settings file to a temp directory for each test.

    Without this, tests that write settings (inclusion, safe_thresholds,
    volume, etc.) would mutate the shared ``data/settings.json`` on disk,
    leaking values into subsequent tests that lazy-load from that file.
    """
    from vtsearch import settings as settings_mod

    test_settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(settings_mod, "SETTINGS_PATH", test_settings_path)
    settings_mod.reset()

    # Also redirect dataset and model registries to temp paths
    from vtsearch.datasets import registry as ds_reg_mod
    from vtsearch.models import registry as model_reg_mod

    monkeypatch.setattr(ds_reg_mod, "REGISTRY_PATH", tmp_path / "dataset_registry.json")
    monkeypatch.setattr(model_reg_mod, "REGISTRY_PATH", tmp_path / "model_registry.json")

    # Redirect storage directories to temp paths via settings
    settings_mod.set_saved_datasets_dir(str(tmp_path / "saved_datasets"))
    settings_mod.set_detectors_dir(str(tmp_path / "detectors"))
    settings_mod.set_trainable_models_dir(str(tmp_path / "trainable_models"))

    ds_reg_mod.reset_for_tests()
    model_reg_mod.reset_for_tests()

    yield test_settings_path
    settings_mod.reset()
    ds_reg_mod.reset_for_tests()
    model_reg_mod.reset_for_tests()


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """Force-exit to avoid SIGABRT (exit code 134) from native library cleanup.

    PyTorch, OpenMP, numba (via librosa), and other native libraries spin up
    C++ thread pools that sometimes call ``std::terminate()`` during Python
    interpreter shutdown.  This produces "terminate called without an active
    exception" and exit code 134, even though all tests passed.

    ``os._exit()`` skips the normal interpreter teardown (atexit handlers,
    C++ static destructors) so the problematic cleanup never runs.

    Prints a clear PASS/FAIL summary right before exiting, since os._exit()
    prevents the normal pytest summary from being flushed.
    """
    import sys

    # Grab stats from the terminal reporter (if available)
    reporter = session.config.pluginmanager.getplugin("terminalreporter")
    if reporter:
        passed = len(reporter.stats.get("passed", []))
        failed = len(reporter.stats.get("failed", []))
        errors = len(reporter.stats.get("error", []))
        skipped = len(reporter.stats.get("skipped", []))
        total = passed + failed + errors + skipped

        print("", flush=True)
        print("=" * 60, flush=True)
        if failed or errors:
            print(
                f"TESTS FAILED: {failed} failed, {errors} errors, {passed} passed, {skipped} skipped (total: {total})",
                flush=True,
            )
        else:
            print(
                f"ALL {passed} TESTS PASSED ({skipped} skipped, total: {total})",
                flush=True,
            )
        print("=" * 60, flush=True)

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exitstatus)
