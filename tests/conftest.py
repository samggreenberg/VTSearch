import pytest

import vtsearch.config as config

# Reduce training epochs for faster tests (default is 200; 30 is sufficient
# for the tiny MLP to converge on the small test dataset).
config.TRAIN_EPOCHS = 30

import app as app_module

# Import refactored modules and make them accessible through app_module
from vtsearch.config import NUM_MEDIAS, SAMPLE_RATE
from vtsearch.audio import generate_wav
from vtsearch.models import initialize_models, train_and_score
from vtsearch.models.progress import clear_progress_cache
from vtsearch.utils import bad_votes, medias, good_votes, label_history, last_learned_scores, textsort_suggestions, vote_click_times

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
    _state._diversity_tree = None
    _state.favorite_detectors.clear()
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
