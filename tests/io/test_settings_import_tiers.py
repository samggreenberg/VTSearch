"""A settings import or sync writes only the user's own settings (#4965).

``PUT /api/settings`` drops every server-tier key (its schema omits them), but
a settings import and a settings-source sync used to apply every key the file
held, server tier included: any user could repoint ``detectors_dir`` for the
whole server or flip an operator lock with a file of their choosing. Both now
go through :func:`vtsearch.settings.plan_user_write`, the rule ``PUT`` shares.

Covers, on the import route and on both sync pulls (manual and automatic):

- no server-tier key is written, and the response reports what was skipped;
- the bespoke checks ``PUT`` runs (a registry check, the achievement wipe)
  run on an import too;
- a training-setting change reaches loaded detectors the way ``PUT`` makes it
  (heads dropped, thresholds re-cut), without making an import a balance pick
  on the active detector;
- the automatic re-sync carries that change only after it releases the
  per-user sync lock, which ``vtscore.state.set_beta`` waits on while holding
  the detector state lock.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import pytest

from vtsearch import settings
from vtsearch.settings.models import ServerSettings


def _server_key_values(tmp_path: Path) -> dict[str, Any]:
    """A valid, non-default value for every server-tier key."""
    source = {"source_name": "server_json_file", "field_values": {"filepath": str(tmp_path / "evil.json")}}
    return {
        "saved_datasets_dir": str(tmp_path / "evil-datasets"),
        "detectors_dir": str(tmp_path / "evil-detectors"),
        "max_concurrent_dataset_downloads": 3,
        "max_concurrent_dataset_embeddings": 3,
        "hidden_plugins": {"settings_importers": ["server_json_file"]},
        "dataset_max_age_days": 1,
        "support_email": "evil@example.com",
        "docs_links": [{"label": "Evil", "url": "https://example.com/evil"}],
        "semantic_only": True,
        "autopilot_only": True,
        "solo_media_type": "image",
        "projection_n_neighbors": 31,
        "projection_min_dist": 0.42,
        "browse_signpost_vocab": {"image": ["evil"]},
        "default_settings_source": source,
        "default_labelset_source": source,
        "labelset_source_push_interval_seconds": 99.0,
    }


def _server_values() -> dict[str, Any]:
    """Every server-tier key's current value, as ``GET /api/settings`` reads it."""
    current = settings.get_all()
    return {key: current.get(key) for key in settings._SERVER_KEYS}


def _import(client, tmp_path: Path, data: dict[str, Any], **kwargs):
    path = tmp_path / "import.json"
    path.write_text(json.dumps(data))
    return client.post("/api/settings-importers/import/server_json_file", json={"filepath": str(path)}, **kwargs)


def _loaded_ctx(detector_id: str = "det-4965"):
    """A registered detector context holding a cached head and fold orderings.

    The sentinel object stands in for a trained head. The fold orderings let a
    balance change re-cut the threshold without a model to re-score with.
    """
    from vtscore.training.thresholds import CalibrationFolds
    from vtsearch.state import DetectorContext, register_detector_context

    ctx = DetectorContext(detector_id, name="m4965")
    ctx.model = object()
    ctx.threshold = 0.73
    ctx.calibration_cache = ("k", CalibrationFolds([([0.9, 0.8, 0.2, 0.1], [1.0, 1.0, 0.0, 0.0])], None, []))
    register_detector_context(ctx)
    return ctx


def _recut_threshold() -> float:
    from vtscore.training.thresholds import threshold_from_fold_orderings

    return threshold_from_fold_orderings([([0.9, 0.8, 0.2, 0.1], [1.0, 1.0, 0.0, 0.0])], 0)


def _bump(source_file: Path, data: dict[str, Any]) -> None:
    """Rewrite the source and make the next settings read re-sync from it."""
    source_file.write_text(json.dumps(data))
    mtime = source_file.stat().st_mtime_ns + 10_000_000_000
    os.utime(source_file, ns=(mtime, mtime))
    settings._sync_state["default"].last_check_monotonic -= 5.0


class TestFixture:
    def test_server_key_values_cover_the_tier_with_real_changes(self, tmp_path):
        """Every server key is exercised with a value its setter would accept.

        A value the model rejects (or the current one) would make the tests
        below pass whether or not the tier is enforced.
        """
        values = _server_key_values(tmp_path)
        assert set(values) == settings._SERVER_KEYS
        current = _server_values()
        for key, value in values.items():
            assert settings._validate_field(ServerSettings, key, value) == value, key
            assert current[key] != value, key


class TestImportRoute:
    def test_server_tier_keys_are_skipped(self, client, tmp_path):
        before = _server_values()
        res = _import(client, tmp_path, {**_server_key_values(tmp_path), "volume": 0.31})

        assert res.status_code == 200
        data = res.get_json()
        assert data["keys"] == ["volume"]
        assert set(data["skipped"]) == settings._SERVER_KEYS
        assert "Imported 1 setting(s)" in data["message"]
        assert "Skipped 17" in data["message"]
        assert _server_values() == before
        assert settings.get_volume() == 0.31

    def test_server_file_is_not_written(self, client, tmp_path):
        before = settings.SETTINGS_PATH.read_text()
        _import(client, tmp_path, _server_key_values(tmp_path))
        assert settings.SETTINGS_PATH.read_text() == before

    def test_registry_check_runs_on_import(self, client, tmp_path):
        """An exporter ``PUT`` would 400 on is skipped, not stored."""
        res = _import(client, tmp_path, {"autofind_exporter": "no_such_exporter", "theme": "light"})

        data = res.get_json()
        assert data["keys"] == ["theme"]
        assert data["skipped"] == ["autofind_exporter"]
        assert settings.get_autofind_exporter() == ""

    def test_turning_achievements_off_wipes_counters(self, client, tmp_path):
        from vtsearch import achievements
        from vtsearch.settings import _user_caches

        achievements.record_vote("det-1", media_type="audio")
        assert "achievement_state" in _user_caches["default"]

        _import(client, tmp_path, {"enable_achievements": False})

        assert settings.get_enable_achievements() is False
        assert "achievement_state" not in _user_caches["default"]

    def test_string_boolean_goes_through_the_field_validator(self, client, tmp_path):
        """No marshmallow layer sits in front of an import, so ``"false"`` must
        not read as truthy."""
        _import(client, tmp_path, {"enable_achievements": "false"})
        assert settings.get_enable_achievements() is False

    def test_calibrate_count_drops_loaded_heads(self, client, tmp_path):
        ctx = _loaded_ctx()
        _import(client, tmp_path, {"calibrate_count": settings.get_calibrate_count() + 1})
        assert ctx.model is None
        assert ctx.threshold == 0.5

    def test_calibration_fraction_drops_loaded_heads(self, client, tmp_path):
        ctx = _loaded_ctx()
        _import(client, tmp_path, {"calibration_fraction": 0.25})
        assert ctx.model is None

    def test_beta_recuts_loaded_detectors_without_a_balance_pick(self, client, tmp_path):
        """The user's balance re-cuts a detector that takes it, and keeps the head.

        Unlike ``PUT``, the import is not a balance pick: the active detector
        (named by the header) is not given the value as its own balance.
        """
        ctx = _loaded_ctx()
        model = ctx.model

        res = _import(client, tmp_path, {"beta": 2.0}, headers={"X-Detector-Id": ctx.detector_id})

        assert res.status_code == 200
        assert settings.get_beta() == 2.0
        assert ctx.model is model
        assert ctx.threshold == _recut_threshold()
        assert not ctx.beta_seeded

    def test_unchanged_training_setting_leaves_loaded_heads(self, client, tmp_path):
        ctx = _loaded_ctx()
        _import(client, tmp_path, {"calibrate_count": settings.get_calibrate_count()})
        assert ctx.model is not None
        assert ctx.threshold == 0.73


@pytest.fixture
def source_file(tmp_path):
    path = tmp_path / "remote.json"
    settings.set_settings_source_config({"source_name": "server_json_file", "field_values": {"filepath": str(path)}})
    return path


class TestManualSync:
    def test_server_tier_keys_are_skipped(self, source_file, tmp_path):
        before = _server_values()
        source_file.write_text(json.dumps({**_server_key_values(tmp_path), "volume": 0.27}))

        applied = settings.sync_from_settings_source()

        assert applied == {"volume": 0.27}
        assert _server_values() == before

    def test_calibrate_count_drops_loaded_heads(self, source_file):
        ctx = _loaded_ctx()
        source_file.write_text(json.dumps({"calibrate_count": settings.get_calibrate_count() + 1}))

        settings.sync_from_settings_source()

        assert ctx.model is None


class TestAutoResync:
    def test_server_tier_keys_are_skipped(self, source_file, tmp_path):
        before = _server_values()
        _bump(source_file, {**_server_key_values(tmp_path), "volume": 0.61})

        assert settings.get_volume() == 0.61
        assert _server_values() == before

    def test_training_change_reaches_detectors_outside_the_sync_lock(self, source_file, monkeypatch):
        """The hook must not run under the per-user sync lock.

        ``vtscore.state.set_beta`` persists the setting while holding the
        detector state lock, and that write can wait on this sync lock; taking
        the state lock underneath it would be the reverse order.
        """
        ctx = _loaded_ctx()
        lock = settings._store.per_user_sync_lock("default")
        free: list[bool] = []
        real_after_apply = settings._store._after_apply

        def probing_after_apply(applied):
            got: list[bool] = []

            def probe():
                ok = lock.acquire(timeout=1.0)
                got.append(ok)
                if ok:
                    lock.release()

            t = threading.Thread(target=probe, daemon=True)
            t.start()
            t.join(5.0)
            free.append(got == [True])
            real_after_apply(applied)

        monkeypatch.setattr(settings._store, "_after_apply", probing_after_apply)
        new_count = settings.get_calibrate_count() + 1
        _bump(source_file, {"calibrate_count": new_count})

        assert settings.get_calibrate_count() == new_count
        assert free == [True]
        assert ctx.model is None
