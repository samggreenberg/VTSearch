"""Tests for VTSearch extension support: LabeledElement metadata and media_url lazy-fetch.

Covers:
- LabeledElement metadata field: serialisation, round-trip, from_clips_and_votes
- media_url lazy-fetch: _resolve_media_bytes and _resolve_media_string fallback
- Plugin discovery: ReCaller importer, Holder exporter, Holder label importer,
  PullWrest media source are registered
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
            origin={"importer": "recaller", "params": {"contentID": "C99"}},
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
                "origin": {"importer": "recaller", "params": {"contentID": "C1"}},
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
# Plugin discovery
# ---------------------------------------------------------------------------


class TestPluginDiscovery:
    """Verify that scaffolded plugins are discoverable by their registries."""

    def test_recaller_importer_registered(self):
        from vtscore.datasets.importers import get_importer

        imp = get_importer("recaller")
        assert imp is not None
        assert imp.name == "recaller"
        assert imp.display_name == "ReCaller Query"

    def test_holder_exporter_registered(self):
        from vtscore.exporters import get_exporter

        exp = get_exporter("holder")
        assert exp is not None
        assert exp.name == "holder"
        assert exp.display_name == "Holder Package"

    def test_holder_label_importer_registered(self):
        from vtscore.labels.importers import get_label_importer

        imp = get_label_importer("holder")
        assert imp is not None
        assert imp.name == "holder"
        assert imp.display_name == "Holder Package"

    def test_pullwrest_source_registered(self):
        from vtscore.datasets.sources import get_source_for_origin

        origin = {
            "importer": "recaller",
            "params": {
                "contentID": "C1",
                "mediaID": "M1",
                "media_url": "http://pw/M1",
                "media_type": "audio",
            },
        }
        source = get_source_for_origin(origin)
        assert source is not None
        assert source.name == "pullwrest"

    def test_pullwrest_source_none_without_url(self):
        from vtscore.datasets.sources import get_source_for_origin

        origin = {"importer": "recaller", "params": {"contentID": "C1"}}
        source = get_source_for_origin(origin)
        assert source is None


# ---------------------------------------------------------------------------
# Holder exporter helpers
# ---------------------------------------------------------------------------


class TestHolderExporterHelpers:
    """Test _extract_content_id and _extract_entry_metadata."""

    def test_extract_content_id_from_metadata(self):
        from vtscore.exporters.holder import _extract_content_id

        entry = {"metadata": {"contentID": "C1"}, "custom_metadata": {"contentID": "C2"}}
        assert _extract_content_id(entry) == "C1"  # metadata wins

    def test_extract_content_id_from_custom_metadata(self):
        from vtscore.exporters.holder import _extract_content_id

        entry = {"custom_metadata": {"contentID": "C2"}}
        assert _extract_content_id(entry) == "C2"

    def test_extract_content_id_from_origin(self):
        from vtscore.exporters.holder import _extract_content_id

        entry = {"origin": {"importer": "recaller", "params": {"contentID": "C3"}}}
        assert _extract_content_id(entry) == "C3"

    def test_extract_content_id_missing(self):
        from vtscore.exporters.holder import _extract_content_id

        assert _extract_content_id({}) is None
        assert _extract_content_id({"origin": {"importer": "server_folder", "params": {}}}) is None

    def test_extract_entry_metadata(self):
        from vtscore.exporters.holder import _extract_entry_metadata

        entry = {
            "md5": "hash1",
            "custom_metadata": {"mediaID": "M1", "media_url": "http://pw/M1"},
            "origin": {"params": {"media_type": "audio"}},
        }
        meta = _extract_entry_metadata(entry)
        assert meta["mediaID"] == "M1"
        assert meta["md5"] == "hash1"
        assert meta["media_url"] == "http://pw/M1"
        assert meta["media_type"] == "audio"


# ---------------------------------------------------------------------------
# Holder label importer entry conversion
# ---------------------------------------------------------------------------


class TestHolderLabelImporterEntry:
    """Test _entry_to_label conversion."""

    def test_entry_to_label_good(self):
        from vtscore.labels.importers.holder import _entry_to_label

        entry = {
            "contentID": "C1",
            "mediaID": "M1",
            "md5": "hash1",
            "media_url": "http://pw/M1",
            "media_type": "audio",
        }
        label = _entry_to_label(entry, "good")
        assert label["md5"] == "hash1"
        assert label["label"] == "good"
        assert label["origin"]["importer"] == "recaller"
        assert label["origin"]["params"]["contentID"] == "C1"
        assert label["origin_name"] == "C1"
        assert label["metadata"]["contentID"] == "C1"
        assert label["metadata"]["mediaID"] == "M1"

    def test_entry_to_label_bad(self):
        from vtscore.labels.importers.holder import _entry_to_label

        entry = {"contentID": "C2", "mediaID": "M2", "md5": "hash2", "media_url": "", "media_type": "image"}
        label = _entry_to_label(entry, "bad")
        assert label["label"] == "bad"
        assert label["origin"]["params"]["media_type"] == "image"


# ---------------------------------------------------------------------------
# ReCaller importer structure
# ---------------------------------------------------------------------------


class TestReCallerImporterStructure:
    """Test ReCaller importer properties (without calling real APIs)."""

    def test_build_origin_is_empty(self):
        from vtscore.datasets.importers.recaller import ReCallerDatasetImporter

        imp = ReCallerDatasetImporter()
        origin = imp.build_origin({"query_id": "Q123", "media_type": "audio"})
        assert origin["importer"] == "recaller"
        assert origin["params"] == {}

    def test_origin_display(self):
        from vtscore.datasets.importers.recaller import ReCallerDatasetImporter

        imp = ReCallerDatasetImporter()
        origin = {"importer": "recaller", "params": {"contentID": "C42"}}
        assert imp.origin_display(origin) == "recaller:C42"

    def test_origin_display_empty(self):
        from vtscore.datasets.importers.recaller import ReCallerDatasetImporter

        imp = ReCallerDatasetImporter()
        origin = {"importer": "recaller", "params": {}}
        assert imp.origin_display(origin) == "recaller"

    def test_fields(self):
        from vtscore.datasets.importers.recaller import ReCallerDatasetImporter

        imp = ReCallerDatasetImporter()
        keys = [f.key for f in imp.fields]
        assert "query_id" in keys
        assert "media_type" in keys

    def test_run_raises_without_query_id(self):
        from vtscore.datasets.importers.recaller import ReCallerDatasetImporter

        imp = ReCallerDatasetImporter()
        medias: dict = {}
        import pytest

        with pytest.raises(ValueError, match="query ID"):
            imp.run({"query_id": "", "media_type": "audio"}, medias)


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
                "type": "audio",
                "md5": "enrich_test_hash",
                "filename": "C1.wav",
                "origin": {
                    "importer": "recaller",
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
                "type": "audio",
                "md5": "override_test_hash",
                "filename": "C1.wav",
                "origin": {
                    "importer": "recaller",
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
