"""A pre-computed vector nested in ``custom_metadata`` never reaches a response.

``custom_metadata_map`` is the documented plugin channel for shipping a
pre-computed vector alongside a file, and ``load_dataset_from_folder`` reads
that vector *without popping it* — so a media loaded that way carries
``custom_metadata["embedding"]``.  Every route that strips a media by
top-level key (``_HEAVYWEIGHT_KEYS``) is blind to it, and the response
schemas' free-form ``fields.Dict`` waves it straight through, so the numpy
array reaches Flask's JSON encoder and 500s the whole request.

:func:`vtscore.utils.hits.hit_custom_metadata` is the single filter every one
of those surfaces now runs the dict through.  One test per surface, asserting
the observable outcome: the caller's own keys survive, the vector doesn't.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.helpers import setup_trainable_model_in_registry
from vtscore.media.processors import Extractor
from vtsearch.state import snapshot_medias


@pytest.fixture
def media_with_nested_vector():
    """Stamp a ``custom_metadata_map``-shaped vector onto one loaded media.

    Returns the media id.  ``reset_state`` re-copies every media dict before
    the next test, so the new top-level key does not leak out of this one.
    """
    from vtsearch.state import medias

    cid = sorted(medias.keys())[0]
    medias[cid]["custom_metadata"] = {
        "asset_id": "XY-7",
        "embedding": np.zeros(4, dtype=np.float32),
    }
    return cid


def _assert_sanitised(custom: dict, where: str) -> None:
    assert custom.get("asset_id") == "XY-7", f"{where} dropped the importer's own metadata"
    assert "embedding" not in custom, f"a pre-computed vector leaked into {where}"


class _StubExtractor(Extractor):
    """Audio extractor that fires on every media, so no dataset is needed."""

    @property
    def name(self) -> str:
        return "stub-ext"

    @property
    def media_type(self) -> str:
        return "audio"

    def extract(self, media: dict) -> list[dict]:
        return [{"confidence": 0.9, "label": "found"}]


class TestAutoDetectHits:
    """``POST /api/auto-detect`` — ``media_info_for_response`` builds each hit."""

    def test_hit_custom_metadata_keeps_asset_id_and_drops_vector(self, client, media_with_nested_vector):
        from vtsearch.settings import add_autofind_detector

        setup_trainable_model_in_registry(
            "auto-detect-nested-vector",
            good_ids=[1, 2, 3],
            bad_ids=[18, 19, 20],
            snap=snapshot_medias(),
        )
        add_autofind_detector("auto-detect-nested-vector")

        resp = client.post("/api/auto-detect", json={})
        assert resp.status_code == 200, resp.get_data(as_text=True)

        result = resp.get_json()["results"]["auto-detect-nested-vector"]
        hits = [h for h in result["hits"] + result["negative_hits"] if h["id"] == media_with_nested_vector]
        assert hits, "the seeded media should be scored into one of the two hit lists"
        _assert_sanitised(hits[0]["custom_metadata"], "an auto-detect hit")


class TestProcessorScoringHits:
    """``POST /api/extract`` — the same helper, via the processor routes."""

    def test_extract_result_keeps_asset_id_and_drops_vector(self, client, monkeypatch, media_with_nested_vector):
        from vtsearch.routes.processors import scoring as scoring_mod

        monkeypatch.setattr(scoring_mod, "_build_extractor", lambda name, extractor_type, config: _StubExtractor())

        resp = client.post(
            "/api/extract",
            json={"name": "stub-ext", "extractor_type": "any", "config": {}},
        )
        assert resp.status_code == 200, resp.get_data(as_text=True)
        rows = [r for r in resp.get_json()["results"] if r["id"] == media_with_nested_vector]
        assert rows, "the seeded media should have produced an extraction hit"
        _assert_sanitised(rows[0]["custom_metadata"], "an extract result")


class TestMediaBatch:
    """``POST /api/medias/batch`` layers importer metadata over display metadata."""

    def test_batch_keeps_asset_id_and_drops_vector(self, client, media_with_nested_vector):
        resp = client.post("/api/medias/batch", json={"ids": [media_with_nested_vector]})
        assert resp.status_code == 200, resp.get_data(as_text=True)

        rows = resp.get_json()
        assert len(rows) == 1
        _assert_sanitised(rows[0]["custom_metadata"], "a /api/medias/batch row")

    def test_display_metadata_still_present(self, client, media_with_nested_vector):
        """The importer layer is filtered, not replaced: the media type's own
        display metadata still sits underneath it."""
        resp = client.post("/api/medias/batch", json={"ids": [media_with_nested_vector]})
        custom = resp.get_json()[0]["custom_metadata"]
        assert len(custom) > 1, "display_metadata keys were lost along with the vector"


class TestEnrichedLabelExport:
    """``GET /api/labels/export?enrich=1`` — the metadata blob becomes an export column."""

    def test_export_entry_keeps_asset_id_and_drops_vector(self, client, media_with_nested_vector):
        from vtsearch.state import good_votes

        good_votes[media_with_nested_vector] = None

        resp = client.get("/api/labels/export?enrich=1")
        assert resp.status_code == 200, resp.get_data(as_text=True)

        body = resp.get_json()
        entries = [e for e in body["labels"] if e.get("custom_metadata")]
        assert entries, "the voted media should have produced an enriched label entry"
        _assert_sanitised(entries[0]["custom_metadata"], "an enriched label export entry")
        assert "embedding" not in body["available_columns"]


class TestMediaInfoForResponse:
    """The shared helper, directly."""

    def test_strips_both_layers(self):
        from vtsearch.routes._shared import media_info_for_response

        media = {
            "id": 1,
            "filename": "a.wav",
            "embeddings": {"clap": np.zeros(4)},
            "media_bytes": b"\x00",
            "custom_metadata": {"asset_id": "XY-7", "embedding": np.zeros(4)},
        }
        info = media_info_for_response(media)
        assert info == {"id": 1, "filename": "a.wav", "custom_metadata": {"asset_id": "XY-7"}}

    def test_custom_metadata_is_a_copy(self):
        from vtsearch.routes._shared import media_info_for_response

        media = {"id": 1, "custom_metadata": {"asset_id": "XY-7"}}
        info = media_info_for_response(media)
        info["custom_metadata"]["asset_id"] = "mutated"
        assert media["custom_metadata"] == {"asset_id": "XY-7"}

    def test_leaves_a_media_without_custom_metadata_alone(self):
        from vtsearch.routes._shared import media_info_for_response

        assert media_info_for_response({"id": 1, "custom_metadata": None}) == {"id": 1, "custom_metadata": None}
        assert media_info_for_response({"id": 1}) == {"id": 1}
