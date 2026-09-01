"""Tests for VTSearch extension support: LabeledElement metadata and media_url lazy-fetch.

Covers:
- LabeledElement metadata field: serialisation, round-trip, from_clips_and_votes
- media_url lazy-fetch: _resolve_media_bytes and _resolve_media_string fallback
- Enriched label export surfaces ``origin.params`` as columns

These are the framework hooks an out-of-tree service importer leans on: a
per-media origin whose params round-trip into exports, arbitrary per-element
metadata, and media whose bytes live behind a URL rather than on disk.
"""

from __future__ import annotations

import numpy as np
from unittest.mock import patch

from vtscore.datasets.labelset import LabelSet, LabeledElement


# ---------------------------------------------------------------------------
# LabeledElement metadata
# ---------------------------------------------------------------------------


class TestLabeledElementMetadata:
    def test_to_dict_without_metadata(self):
        e = LabeledElement(md5="abc", label="good")
        d = e.to_dict()
        assert "metadata" not in d

    def test_to_dict_with_metadata(self):
        meta = {"contentID": "C1", "mediaID": "M1", "media_url": "http://pw/M1"}
        e = LabeledElement(md5="abc", label="good", metadata=meta)
        d = e.to_dict()
        assert d["metadata"] == meta

    def test_from_dict_with_metadata(self):
        meta = {"contentID": "C1", "mediaID": "M1"}
        d = {"md5": "abc", "label": "good", "metadata": meta}
        e = LabeledElement.from_dict(d)
        assert e.metadata == meta

    def test_from_dict_without_metadata(self):
        d = {"md5": "abc", "label": "good"}
        e = LabeledElement.from_dict(d)
        assert e.metadata is None

    def test_roundtrip(self):
        meta = {"contentID": "C99", "extra": 42}
        original = LabeledElement(
            md5="hash1",
            label="bad",
            origin={"importer": "svc_importer", "params": {"contentID": "C99"}},
            origin_name="C99",
            metadata=meta,
        )
        d = original.to_dict()
        restored = LabeledElement.from_dict(d)
        assert restored.metadata == meta
        assert restored.md5 == "hash1"
        assert restored.origin_name == "C99"

    def test_labelset_roundtrip_with_metadata(self):
        elements = [
            LabeledElement(md5="a", label="good", metadata={"contentID": "C1"}),
            LabeledElement(md5="b", label="bad"),  # no metadata
        ]
        ls = LabelSet(elements)
        d = ls.to_dict()
        restored = LabelSet.from_dict(d)
        assert len(restored) == 2
        assert restored.elements[0].metadata == {"contentID": "C1"}
        assert restored.elements[1].metadata is None


class TestClipToElementsMetadata:
    """Test that custom_metadata from media flows into LabeledElement.metadata."""

    def test_custom_metadata_carried_through(self):
        medias = {
            1: {
                "id": 1,
                "md5": "hash1",
                "origin": {"importer": "svc_importer", "params": {"contentID": "C1"}},
                "origin_name": "C1",
                "filename": "C1",
                "category": "",
                "custom_metadata": {"contentID": "C1", "mediaID": "M1"},
            }
        }
        good_votes = {1: None}
        bad_votes = {}
        ls = LabelSet.from_clips_and_votes(medias, good_votes, bad_votes)
        assert len(ls) == 1
        assert ls.elements[0].metadata == {"contentID": "C1", "mediaID": "M1"}

    def test_no_custom_metadata_yields_none(self):
        medias = {
            1: {
                "id": 1,
                "md5": "hash1",
                "origin": None,
                "origin_name": "file.wav",
                "filename": "file.wav",
                "category": "",
            }
        }
        good_votes = {1: None}
        bad_votes = {}
        ls = LabelSet.from_clips_and_votes(medias, good_votes, bad_votes)
        assert ls.elements[0].metadata is None

    def test_empty_custom_metadata_yields_none(self):
        """Empty dict should be treated as None (no metadata)."""
        medias = {
            1: {
                "id": 1,
                "md5": "hash1",
                "origin": None,
                "origin_name": "file.wav",
                "filename": "file.wav",
                "category": "",
                "custom_metadata": {},
            }
        }
        good_votes = {1: None}
        bad_votes = {}
        ls = LabelSet.from_clips_and_votes(medias, good_votes, bad_votes)
        assert ls.elements[0].metadata is None


# ---------------------------------------------------------------------------
# media_url lazy-fetch
# ---------------------------------------------------------------------------


class TestMediaUrlLazyFetch:
    """Test that _resolve_media_bytes/string fall back to media_url."""

    def test_resolve_bytes_from_media_url(self):
        from vtscore.media.audio.media_type import AudioMediaType

        mt = AudioMediaType()
        media = {"media_bytes": None, "media_path": None, "media_url": "http://pw/media/123"}

        with patch("vtscore.media.base._fetch_media_url", return_value=b"fetched") as mock:
            result = mt._resolve_media_bytes(media)

        assert result == b"fetched"
        mock.assert_called_once_with("http://pw/media/123")

    def test_resolve_bytes_prefers_bytes_over_url(self):
        from vtscore.media.audio.media_type import AudioMediaType

        mt = AudioMediaType()
        media = {"media_bytes": b"in-memory", "media_path": None, "media_url": "http://pw/123"}

        with patch("vtscore.media.base._fetch_media_url") as mock:
            result = mt._resolve_media_bytes(media)

        assert result == b"in-memory"
        mock.assert_not_called()

    def test_resolve_bytes_prefers_path_over_url(self, tmp_path):
        from vtscore.media.audio.media_type import AudioMediaType

        p = tmp_path / "test.wav"
        p.write_bytes(b"from-disk")

        mt = AudioMediaType()
        media = {"media_bytes": None, "media_path": str(p), "media_url": "http://pw/123"}

        with patch("vtscore.media.base._fetch_media_url") as mock:
            result = mt._resolve_media_bytes(media)

        assert result == b"from-disk"
        mock.assert_not_called()

    def test_resolve_bytes_url_returns_none_on_failure(self):
        from vtscore.media.audio.media_type import AudioMediaType

        mt = AudioMediaType()
        media = {"media_bytes": None, "media_path": None, "media_url": "http://pw/bad"}

        with patch("vtscore.media.base._fetch_media_url", return_value=None):
            result = mt._resolve_media_bytes(media)

        assert result is None

    def test_resolve_bytes_no_url_returns_none(self):
        from vtscore.media.audio.media_type import AudioMediaType

        mt = AudioMediaType()
        media = {"media_bytes": None, "media_path": None}
        assert mt._resolve_media_bytes(media) is None

    def test_resolve_string_from_media_url(self):
        from vtscore.media.text.media_type import TextMediaType

        mt = TextMediaType()
        media = {"media_string": None, "media_path": None, "media_url": "http://pw/text/1"}

        with patch("vtscore.media.base._fetch_media_url", return_value=b"hello text") as mock:
            result = mt._resolve_media_string(media)

        assert result == "hello text"
        mock.assert_called_once_with("http://pw/text/1")

    def test_resolve_string_prefers_string_over_url(self):
        from vtscore.media.text.media_type import TextMediaType

        mt = TextMediaType()
        media = {"media_string": "cached", "media_path": None, "media_url": "http://pw/text/1"}

        with patch("vtscore.media.base._fetch_media_url") as mock:
            result = mt._resolve_media_string(media)

        assert result == "cached"
        mock.assert_not_called()

    def test_resolve_string_url_returns_empty_on_failure(self):
        from vtscore.media.text.media_type import TextMediaType

        mt = TextMediaType()
        media = {"media_string": None, "media_path": None, "media_url": "http://pw/bad"}

        with patch("vtscore.media.base._fetch_media_url", return_value=None):
            result = mt._resolve_media_string(media)

        assert result == ""


# ---------------------------------------------------------------------------
# Enriched label export with origin.params flattened
# ---------------------------------------------------------------------------


class TestEnrichedExportOriginParams:
    """Test that origin.params are surfaced as columns in enriched label export."""

    def test_origin_params_in_enriched_export(self, client):
        """Origin params like contentID appear in custom_metadata and available_columns."""
        from vtsearch.state import (
            medias,
            good_votes,
        )

        saved = dict(medias)
        medias.clear()
        try:
            rng = np.random.default_rng(42)
            medias[1] = {
                "id": 1,
                "media_type": "audio",
                "md5": "enrich_test_hash",
                "filename": "C1.wav",
                "origin": {
                    "importer": "svc_importer",
                    "params": {
                        "contentID": "C1",
                        "mediaID": "M1",
                        "media_url": "http://pw/M1",
                        "media_type": "audio",
                    },
                },
                "origin_name": "C1",
                "category": "",
                "embedding": rng.standard_normal(512).astype(np.float32),
                "embedder": "clap",
                "media_bytes": b"\x00",
                "media_path": None,
                "file_size": 1,
                "duration": 0,
            }
            good_votes[1] = None

            resp = client.get("/api/labels/export?enrich=true")
            assert resp.status_code == 200
            data = resp.get_json()

            labels = data["labels"]
            assert len(labels) == 1
            cm = labels[0].get("custom_metadata", {})
            assert cm["contentID"] == "C1"
            assert cm["mediaID"] == "M1"
            assert cm["media_url"] == "http://pw/M1"

            avail = data.get("available_columns", [])
            assert "contentID" in avail
            assert "mediaID" in avail
        finally:
            medias.clear()
            medias.update(saved)
            good_votes.pop(1, None)

    def test_custom_metadata_overrides_origin_params(self, client):
        """custom_metadata values take precedence over same-named origin params."""
        from vtsearch.state import (
            medias,
            good_votes,
        )

        saved = dict(medias)
        medias.clear()
        try:
            rng = np.random.default_rng(42)
            medias[1] = {
                "id": 1,
                "media_type": "audio",
                "md5": "override_test_hash",
                "filename": "C1.wav",
                "origin": {
                    "importer": "svc_importer",
                    "params": {"contentID": "C1_from_origin"},
                },
                "origin_name": "C1",
                "category": "",
                "embedding": rng.standard_normal(512).astype(np.float32),
                "embedder": "clap",
                "media_bytes": b"\x00",
                "media_path": None,
                "file_size": 1,
                "duration": 0,
                "custom_metadata": {"contentID": "C1_from_custom"},
            }
            good_votes[1] = None

            resp = client.get("/api/labels/export?enrich=true")
            data = resp.get_json()
            cm = data["labels"][0]["custom_metadata"]
            # custom_metadata wins because it's applied after origin params
            assert cm["contentID"] == "C1_from_custom"
        finally:
            medias.clear()
            medias.update(saved)
            good_votes.pop(1, None)
