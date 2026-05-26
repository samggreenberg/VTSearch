"""Happy-path tests for the processor scoring endpoints.

The error paths (no medias, bad type, mismatched media type) live in
``tests/detectors/test_extractors.py`` and
``tests/integration/test_multi_media_coverage.py``.  This file exercises
the *successful* execution branches in
:mod:`vtsearch.routes.processors.scoring`:

* ``/api/extract`` happy path - extractor produces hits, response has
  the expected per-media shape.
* ``/api/localize`` happy path - localizer produces bounding boxes.
* ``/api/auto-extract`` and ``/api/auto-localize`` parallel runner
  (``_auto_run_processors``) - uses the worker-cap path.

A stub extractor / localizer is patched into the route module so the
tests do not depend on YOLO weights.
"""

from __future__ import annotations

from typing import Any

import pytest

from vtsearch.autorun_processors import (
    autorun_extractors,
    autorun_localizers,
)
from vtscore.media.processors import Extractor, Localizer
from vtsearch.routes.processors import scoring as scoring_mod


class StubAudioExtractor(Extractor):
    """Audio extractor that yields one fake hit per media."""

    def __init__(self, name: str, *, hits_per_media: int = 1):
        self._name = name
        self._hits_per_media = hits_per_media

    @property
    def name(self) -> str:
        return self._name

    @property
    def media_type(self) -> str:
        return "audio"

    def extract(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        return [{"confidence": 0.9, "label": "found", "idx": i} for i in range(self._hits_per_media)]


class StubAudioLocalizer(Localizer):
    """Audio localizer that yields one bounding-box per media."""

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def media_type(self) -> str:
        return "audio"

    def localize(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        return [{"confidence": 0.8, "bbox": [0.0, 1.0]}]


@pytest.fixture
def stub_extractor_factory(monkeypatch):
    """Make ``_build_extractor`` return a :class:`StubAudioExtractor`."""

    def _factory(name: str, extractor_type: str, config: dict):
        return StubAudioExtractor(name)

    monkeypatch.setattr(scoring_mod, "_build_extractor", _factory)


@pytest.fixture
def stub_localizer_factory(monkeypatch):
    """Make ``_build_localizer`` return a :class:`StubAudioLocalizer`."""

    def _factory(name: str, localizer_type: str, config: dict):
        return StubAudioLocalizer(name)

    monkeypatch.setattr(scoring_mod, "_build_localizer", _factory)


# ---------------------------------------------------------------------------
# /api/extract
# ---------------------------------------------------------------------------


class TestExtractHappyPath:
    def test_runs_extractor_on_each_media(self, client, stub_extractor_factory):
        from vtsearch.state import medias

        resp = client.post(
            "/api/extract",
            json={"name": "stub-ext", "extractor_type": "any", "config": {}},
        )
        assert resp.status_code == 200, resp.get_data(as_text=True)
        body = resp.get_json()
        assert body["extractor_name"] == "stub-ext"
        assert body["media_type"] == "audio"
        # Every media produces one hit → all medias appear in results.
        assert body["total_medias_with_hits"] == len(medias)
        assert len(body["results"]) == len(medias)

    def test_result_dict_omits_heavyweight_keys(self, client, stub_extractor_factory):
        resp = client.post(
            "/api/extract",
            json={"name": "stub-ext", "extractor_type": "any", "config": {}},
        )
        body = resp.get_json()
        for result in body["results"]:
            for heavy in ("embedding", "media_bytes", "media_string", "thumbnail_bytes"):
                assert heavy not in result
            assert "extractions" in result
            assert result["extractions"][0]["label"] == "found"


class TestLocalizeHappyPath:
    def test_runs_localizer_on_each_media(self, client, stub_localizer_factory):
        from vtsearch.state import medias

        resp = client.post(
            "/api/localize",
            json={"name": "stub-loc", "localizer_type": "any", "config": {}},
        )
        assert resp.status_code == 200, resp.get_data(as_text=True)
        body = resp.get_json()
        assert body["localizer_name"] == "stub-loc"
        assert body["media_type"] == "audio"
        assert body["total_medias_with_hits"] == len(medias)
        for result in body["results"]:
            assert "localizations" in result
            assert result["localizations"][0]["confidence"] == 0.8


# ---------------------------------------------------------------------------
# /api/auto-extract  &  /api/auto-localize
# ---------------------------------------------------------------------------


class TestAutoExtractHappyPath:
    def test_runs_registered_autorun_extractors(self, client, stub_extractor_factory):
        # Register two autorun extractors for audio so the worker fan-out
        # actually has multiple jobs to dispatch.
        autorun_extractors["a"] = {
            "name": "a",
            "extractor_type": "any",
            "media_type": "audio",
            "config": {},
            "created_at": 0,
        }
        autorun_extractors["b"] = {
            "name": "b",
            "extractor_type": "any",
            "media_type": "audio",
            "config": {},
            "created_at": 0,
        }

        resp = client.post("/api/auto-extract")
        assert resp.status_code == 200, resp.get_data(as_text=True)
        body = resp.get_json()
        assert body["media_type"] == "audio"
        assert body["extractors_run"] == 2
        assert set(body["results"].keys()) == {"a", "b"}
        for entry in body["results"].values():
            assert entry["total_medias_with_hits"] > 0
            assert entry["results"]

    def test_skips_extractors_whose_build_fails(self, client, monkeypatch):
        """A failing factory call yields no entry for that extractor."""

        def _flaky(name: str, extractor_type: str, config: dict):
            if name == "broken":
                raise RuntimeError("simulated config error")
            return StubAudioExtractor(name)

        monkeypatch.setattr(scoring_mod, "_build_extractor", _flaky)

        autorun_extractors["ok"] = {
            "name": "ok",
            "extractor_type": "any",
            "media_type": "audio",
            "config": {},
            "created_at": 0,
        }
        autorun_extractors["broken"] = {
            "name": "broken",
            "extractor_type": "any",
            "media_type": "audio",
            "config": {},
            "created_at": 0,
        }

        resp = client.post("/api/auto-extract")
        assert resp.status_code == 200
        body = resp.get_json()
        assert "ok" in body["results"]
        assert "broken" not in body["results"]


class TestAutoLocalizeHappyPath:
    def test_runs_registered_autorun_localizers(self, client, stub_localizer_factory):
        autorun_localizers["loc-x"] = {
            "name": "loc-x",
            "localizer_type": "any",
            "media_type": "audio",
            "config": {},
            "created_at": 0,
        }

        resp = client.post("/api/auto-localize")
        assert resp.status_code == 200, resp.get_data(as_text=True)
        body = resp.get_json()
        assert body["media_type"] == "audio"
        assert body["localizers_run"] == 1
        assert "loc-x" in body["results"]
        entry = body["results"]["loc-x"]
        assert entry["total_medias_with_hits"] > 0
