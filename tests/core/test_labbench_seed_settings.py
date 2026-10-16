"""The settings ``scripts/seed_labbench_settings.py`` seeds are real settings (issue #4958).

The LabBench image seeds its data volume with a server ``settings.json`` and
the default user's ``user_settings.json``. A key either model does not declare
is dropped on load without a word, so the container would serve unstreamlined
and nothing would say why. The library-tier half of these checks (plugin
names, the embedder lock) lives in ``tests_lib/meta/test_dockerfile_labbench.py``;
this half needs the app's settings models.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from vtsearch.settings import USER_SETTINGS_FILENAME
from vtsearch.settings_models import ServerSettings, UserSettings

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "seed_labbench_settings.py"


def _load_seed():
    spec = importlib.util.spec_from_file_location("seed_labbench_settings", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SEED = _load_seed()


@pytest.mark.parametrize("media_type", sorted(_SEED.HIDDEN_BY_TYPE))
def test_server_seed_is_server_settings(media_type: str) -> None:
    seeded = _SEED.server_settings(media_type)
    assert set(seeded) <= set(ServerSettings.model_fields)
    parsed = ServerSettings.model_validate(seeded)
    assert parsed.hidden_plugins == seeded["hidden_plugins"]
    assert parsed.dataset_max_age_days == seeded["dataset_max_age_days"]


@pytest.mark.parametrize("media_type", sorted(_SEED.HIDDEN_BY_TYPE))
def test_user_seed_is_user_settings(media_type: str) -> None:
    seeded = _SEED.user_settings(media_type, "some_embedder")
    assert set(seeded) <= set(UserSettings.model_fields)
    parsed = UserSettings.model_validate(seeded)
    assert parsed.solo_embedder_per_media_type == {media_type: "some_embedder"}


def test_user_seed_has_the_app_s_filename() -> None:
    assert _SEED.USER_SETTINGS_FILENAME == USER_SETTINGS_FILENAME
