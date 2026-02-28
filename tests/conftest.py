from unittest.mock import patch

import numpy as np
import os

import pytest

import vtsearch.config as config

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
        data = open(path, "rb").read(1000)
        seed = int(hashlib.md5(data).hexdigest(), 16) % 2**31
    except Exception:
        seed = hash(str(path)) % 2**31
    rng = np.random.RandomState(seed)
    return rng.randn(_EMBEDDING_DIM).astype(np.float32)


def _fake_embed_text(text):
    """Deterministic fake text embedding derived from the query string."""
    rng = np.random.RandomState(hash(text) % 2**31)
    return rng.randn(_EMBEDDING_DIM).astype(np.float32)


# Patch embed_audio_file so init_medias() never triggers CLAP model loading.
_patch_embed_audio = patch("vtsearch.medias.embed_audio_file", side_effect=_fake_embed_audio)
_patch_embed_audio.start()

import app as app_module

# Import refactored modules and make them accessible through app_module
from vtsearch.config import NUM_MEDIAS, SAMPLE_RATE
from vtsearch.audio import generate_wav
from vtsearch.models import initialize_models, train_and_score
from vtsearch.models.progress import clear_progress_cache
from vtsearch.utils import (
    bad_votes,
    medias,
    good_votes,
    label_history,
    last_learned_scores,
    textsort_suggestions,
    vote_click_times,
)

# Attach to app_module for backward compatibility with existing tests
app_module.NUM_MEDIAS = NUM_MEDIAS
app_module.SAMPLE_RATE = SAMPLE_RATE
app_module.generate_wav = generate_wav
app_module.train_and_score = train_and_score
app_module.medias = medias
app_module.good_votes = good_votes
app_module.bad_votes = bad_votes

# Initialize models and medias
initialize_models()
app_module.init_medias()

# Stop the module-level patch (init_medias is done); the per-test autouse
# fixture below re-applies the patches for every test so that /api/sort and
# other routes that call embed_text don't trigger CLAP loading either.
_patch_embed_audio.stop()

# Grab the audio media-type singleton so the per-test fixture can patch
# embed_text and load_models on it, preventing CLAP from loading during
# /api/sort and similar calls.
from vtsearch.media import get as _media_get

_audio_mt = _media_get("audio")


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
    ):
        yield


@pytest.fixture(autouse=True)
def reset_state():
    """Reset all mutable global state before each test.

    This fixture prevents cross-test contamination by clearing votes,
    favorites, and all other mutable state that lives in
    ``vtsearch.utils.state``.  It runs automatically before every test.
    """
    import vtsearch.utils.state as _state

    good_votes.clear()
    bad_votes.clear()
    label_history.clear()
    textsort_suggestions.clear()
    vote_click_times.clear()
    last_learned_scores.clear()
    _state._click_counter = 0
    _state.inclusion = None  # reset to "not loaded" so it re-reads from settings
    _state._dataset_display_name = None
    _state._diversity_tree = None
    _state.autorun_detectors.clear()
    _state.favorite_extractors.clear()
    _state.favorite_localizers.clear()
    clear_progress_cache()


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
    yield test_settings_path
    settings_mod.reset()


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


def pytest_sessionfinish(session, exitstatus):
    """Force-exit to avoid SIGABRT (exit code 134) from native library cleanup.

    PyTorch, OpenMP, numba (via librosa), and other native libraries spin up
    C++ thread pools that sometimes call ``std::terminate()`` during Python
    interpreter shutdown.  This produces "terminate called without an active
    exception" and exit code 134, even though all tests passed.

    ``os._exit()`` skips the normal interpreter teardown (atexit handlers,
    C++ static destructors) so the problematic cleanup never runs.
    """
    os._exit(exitstatus)
