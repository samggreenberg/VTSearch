"""A detector's balance, asked for at creation and kept on the detector (#4665).

Autopilot has no Threshold control, so the New Detector form asks for the
balance, the create routes keep it in the detector's JSON, and the detector's
line - Autopilot's included - is drawn at it once the detector is loaded.  A
Threshold pick moves it on the detector too.  Each pick (at creation or on a
Threshold) is also the user's balance, their last pick.
"""

from __future__ import annotations

import json

import pytest

from tests import load_detector_and_wait
from tests.helpers import setup_trainable_model_in_registry
from vtscore.detectors.balance import BETA_KEY
from vtscore.detectors.store import _detector_path, _read_detector, _write_detector
from vtsearch import settings
from vtsearch.state import medias, snapshot_medias


def _file(name: str) -> dict:
    data = _read_detector(_detector_path(name))
    assert data is not None
    return data


def _create(client, name: str, **extra):
    return client.post(
        "/api/detectors/registry",
        json={"name": name, "media_type": "audio", "text_query": "dog barking", **extra},
    )


class TestCreatingADetector:
    def test_keeps_the_balance_on_the_detector_and_as_the_users(self, client):
        res = _create(client, "lean-recall", beta=4)
        assert res.status_code == 201, res.get_json()
        assert _file("lean-recall")[BETA_KEY] == 4.0
        assert settings.get_beta() == 4.0

    def test_with_no_balance_keeps_none(self, client):
        settings.set_beta(0.25)
        assert _create(client, "no-lean").status_code == 201
        assert _create(client, "null-lean", beta=None).status_code == 201
        assert BETA_KEY not in _file("no-lean") and BETA_KEY not in _file("null-lean")
        assert settings.get_beta() == 0.25

    def test_a_balance_outside_the_range_is_clamped(self, client):
        assert _create(client, "too-far", beta=10).status_code == 201
        assert _file("too-far")[BETA_KEY] == 4.0

    @pytest.mark.parametrize("bad", ["lean", True, [1]])
    def test_a_non_number_is_refused(self, client, bad):
        assert _create(client, "bad-lean", beta=bad).status_code in (400, 422)
        assert _read_detector(_detector_path("bad-lean")) is None


class TestTheDetectorLine:
    def test_a_loaded_detector_draws_at_its_own_balance_not_the_users(self, client):
        detector_id = setup_trainable_model_in_registry(
            "kept-balance", good_ids=[1, 2, 3, 4, 5, 6], bad_ids=[7, 8, 9, 10, 11, 12], snap=snapshot_medias()
        )
        data = _file("kept-balance")
        data[BETA_KEY] = 4.0
        _write_detector(_detector_path("kept-balance"), data)
        settings.set_beta(0.25)
        load_detector_and_wait(client, detector_id)
        assert client.get("/api/balance").get_json()["beta"] == 4.0

    def test_a_threshold_pick_keeps_it_on_the_detector(self, client):
        detector_id = setup_trainable_model_in_registry(
            "picked-balance", good_ids=[1, 2, 3, 4, 5, 6], bad_ids=[7, 8, 9, 10, 11, 12], snap=snapshot_medias()
        )
        load_detector_and_wait(client, detector_id)
        assert BETA_KEY not in _file("picked-balance")
        res = client.post("/api/balance", json={"beta": 0.25})
        assert res.status_code == 200 and res.get_json()["beta"] == 0.25
        assert _file("picked-balance")[BETA_KEY] == 0.25
        assert settings.get_beta() == 0.25


class TestImportingADetector:
    def _labels(self, tmp_path) -> str:
        origin = {"importer": "server_folder", "params": {"path": "/tmp/audio_clips", "media_type": "audio"}}
        path = tmp_path / "labels.json"
        path.write_text(
            json.dumps(
                {
                    "labels": [
                        {"md5": medias[1]["md5"], "label": "good", "origin": origin, "origin_name": "a.wav"},
                        {"md5": medias[2]["md5"], "label": "bad", "origin": origin, "origin_name": "b.wav"},
                    ]
                }
            )
        )
        return str(path)

    def test_keeps_the_balance_it_was_sent(self, client, tmp_path):
        res = client.post(
            "/api/detectors/registry/from-labelset/server_json_file",
            json={"name": "imported-lean", "filepath": self._labels(tmp_path), "beta": "0.25"},
        )
        assert res.status_code == 201, res.get_json()
        assert _file("imported-lean")[BETA_KEY] == 0.25
        assert settings.get_beta() == 0.25

    def test_an_empty_balance_keeps_none(self, client, tmp_path):
        res = client.post(
            "/api/detectors/registry/from-labelset/server_json_file",
            json={"name": "imported-plain", "filepath": self._labels(tmp_path), "beta": ""},
        )
        assert res.status_code == 201, res.get_json()
        assert BETA_KEY not in _file("imported-plain")

    def test_a_non_number_is_refused_before_the_import_runs(self, client, tmp_path):
        res = client.post(
            "/api/detectors/registry/from-labelset/server_json_file",
            json={"name": "imported-bad", "filepath": self._labels(tmp_path), "beta": "lean"},
        )
        assert res.status_code == 400
        assert _read_detector(_detector_path("imported-bad")) is None


class TestCombining:
    def _source(self, client, name: str, beta: float | None) -> None:
        assert _create(client, name, beta=beta).status_code == 201
        data = _file(name)
        origin = {"importer": "server_folder", "params": {"path": "/data"}}
        data["labelset"] = {
            "labels": [{"md5": "aa", "label": "good", "origin": origin, "origin_name": "aa"}],
        }
        _write_detector(_detector_path(name), data)

    def test_keeps_the_balance_every_source_keeps(self, client):
        self._source(client, "src-a", 4.0)
        self._source(client, "src-b", 4.0)
        res = client.post("/api/detectors/combine", json={"names": ["src-a", "src-b"], "new_name": "both"})
        assert res.status_code == 201, res.get_json()
        assert _file("both")[BETA_KEY] == 4.0

    def test_keeps_none_when_the_sources_disagree(self, client):
        self._source(client, "src-c", 4.0)
        self._source(client, "src-d", None)
        res = client.post("/api/detectors/combine", json={"names": ["src-c", "src-d"], "new_name": "mixed"})
        assert res.status_code == 201, res.get_json()
        assert BETA_KEY not in _file("mixed")
