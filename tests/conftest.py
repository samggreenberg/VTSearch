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
        "test_settings_api_routes",
        "test_settings_directories",
        "test_frontend",
        "test_resize_handle_centering",
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
        "test_extension_scaffolds",
        "test_synthetic_importer",
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
        "test_importer_loading",
        "test_importer_symlinks",
        "test_dataset_importer_media",
        "test_label_importers",
        "test_label_import_endpoint",
        "test_label_import_ingestion",
        "test_labels",
        "test_processor_importers",
        "test_pdf_import",
        "test_corrections_export",
        "test_settings_io",
        "test_sync_sources",
        "test_bulk_embedding",
    ],
    "models": [
        "test_detectors",
        "test_detector_find",
        "test_detector_export",
        "test_extractors",
        "test_processors",
        "test_trainable_models",
        "test_multi_detector",
        "test_clippers",
        "test_clipper_workflow",
        "test_eval",
        "test_eval_visualize",
        "test_eval_voting_iterations",
        "test_resolver",
        "test_new_embedders",
        "test_labelset_elements_api",
    ],
    "downloads": [
        "test_ag_news_download",
        "test_bbc_news_download",
        "test_gtzan_download",
        "test_image_sources_download",
        "test_imdb_download",
        "test_ucsf_documents_download",
        "test_download_and_extract",
        "test_video_datasets_download",
    ],
    "integration": [
        "test_integration",
        "test_slow_integration",
        "test_thread_safety",
        "test_multi_media_coverage",
    ],
    "cli": [
        "test_cli_autodetect",
        "test_cli_trainable_models",
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
from vtsearch.utils.audio_generator import GENERATOR_SAMPLE_RATE
from vtsearch.medias import NUM_MEDIAS
from vtsearch.utils.audio_generator import generate_wav
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
from vtsearch.media import (
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
    stack.enter_context(patch("vtsearch.medias.embed_audio_file", side_effect=_fake_embed_audio))
    for mt in _ALL_MEDIA_TYPES:
        stack.enter_context(patch.object(mt, "embed_text", side_effect=_fake_embed_text))
        stack.enter_context(patch.object(mt, "load_models"))
    for emb in _ALL_EMBEDDERS:
        stack.enter_context(patch.object(emb, "embed_media", side_effect=_fake_embed_audio))
        stack.enter_context(patch.object(emb, "embed_text", side_effect=_fake_embed_text))
        stack.enter_context(patch.object(emb, "load_models"))
    yield
    stack.close()


import vtsearch.utils.state_core as _core
from vtsearch.utils.progress import (
    dataset_progress as _dataset_progress,
    eval_progress as _eval_progress,
    find_progress as _find_progress,
    loading_tasks as _loading_tasks,
    model_loading_tasks as _model_loading_tasks,
    sort_progress as _sort_progress,
)
from vtsearch.auth import DefaultLoginProvider as _DefaultLoginProvider, set_login_provider as _set_login_provider
from vtsearch.datasets.registry import reset_for_tests as _reset_ds_reg
from vtsearch.models.registry import reset_for_tests as _reset_model_reg


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

    _core.autorun_detectors.clear()
    _core.autorun_extractors.clear()
    _core.autorun_localizers.clear()
    clear_progress_cache()

    _dataset_progress.reset_cancel()
    _find_progress.update("idle", "", 0, 0, step=None, total_steps=None, error=None)
    _sort_progress.update("idle", "", 0, 0)
    _eval_progress.update("idle", "", 0, 0)
    _loading_tasks.reset_for_tests()
    _model_loading_tasks.reset_for_tests()

    _set_login_provider(_DefaultLoginProvider())

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
