"""Tests for role-typed embedder slot binding (text / patch / structural).

Covers the Phase-1 backend foundation of the three-slot dataset model:
capability-matching migration, the capability getters, meta.json
round-tripping, validation, and the export-side persistence of slot keys.
"""

from __future__ import annotations

import numpy as np

from vtscore.datasets.container import read_meta
from vtscore.datasets.embedder_slots import (
    PATCH_KEY,
    STRUCTURAL_KEY,
    TEXT_KEY,
    EmbedderSlots,
    legacy_embedder_role,
)
from vtscore.datasets.loader import export_dataset_to_file
from vtscore.datasets.loader_pickle import read_pkl_slots


class TestLegacyMigration:
    """A single legacy embedder maps into its capability-matching slot."""

    def test_text_capable_embedder_fills_text_slot(self):
        slots = EmbedderSlots.from_legacy("siglip")
        assert slots.text == "siglip"
        assert slots.patch is None
        assert slots.structural is None

    def test_patch_capable_embedder_fills_patch_slot(self):
        slots = EmbedderSlots.from_legacy("dinov3_patch")
        assert slots.patch == "dinov3_patch"
        assert slots.text is None
        assert slots.structural is None

    def test_structural_embedder_fills_structural_slot(self):
        slots = EmbedderSlots.from_legacy("sift_vlad")
        assert slots.structural == "sift_vlad"
        assert slots.text is None
        assert slots.patch is None

    def test_plain_single_vector_embedder_fills_no_slot(self):
        # dinov2_single matches none of text/patch/structural -> not slottable.
        slots = EmbedderSlots.from_legacy("dinov2_single")
        assert slots.is_empty
        assert slots.bound() == []

    def test_empty_and_unknown_names_fill_no_slot(self):
        assert EmbedderSlots.from_legacy("").is_empty
        assert EmbedderSlots.from_legacy("no_such_embedder").is_empty

    def test_legacy_embedder_role_helper(self):
        assert legacy_embedder_role("siglip") == "text"
        assert legacy_embedder_role("dinov3_patch") == "patch"
        assert legacy_embedder_role("sift_vlad") == "structural"
        assert legacy_embedder_role("dinov2_single") is None
        assert legacy_embedder_role("") is None


class TestCapabilityGetters:
    def test_getters_follow_bound_slots(self):
        slots = EmbedderSlots(text="siglip", structural="sift_vlad")
        assert slots.supports_text is True
        assert slots.supports_geometric_verification is True
        assert slots.supports_patch_regions is False
        assert not slots.is_empty

    def test_empty_binding_reports_no_capabilities(self):
        slots = EmbedderSlots()
        assert slots.supports_text is False
        assert slots.supports_patch_regions is False
        assert slots.supports_geometric_verification is False
        assert slots.is_empty

    def test_bound_dedups_and_orders(self):
        # Same embedder in two roles (hypothetical) is reported once, in role order.
        slots = EmbedderSlots(text="a", patch="b", structural="a")
        assert slots.bound() == ["a", "b"]


class TestMetaRoundTrip:
    def test_to_meta_omits_empty_slots(self):
        assert EmbedderSlots().to_meta() == {}
        assert EmbedderSlots(text="siglip").to_meta() == {TEXT_KEY: "siglip"}

    def test_to_meta_then_from_meta_round_trips(self):
        slots = EmbedderSlots(text="siglip", patch="dinov3_patch", structural="sift_vlad")
        meta = {"embedder": "siglip", **slots.to_meta()}
        restored = EmbedderSlots.from_meta(meta)
        assert restored == slots

    def test_explicit_slot_keys_take_priority_over_legacy(self):
        # Slot keys present -> legacy embedder is ignored for slot resolution.
        meta = {"embedder": "dinov2_single", PATCH_KEY: "dinov3_patch"}
        slots = EmbedderSlots.from_meta(meta)
        assert slots.patch == "dinov3_patch"
        assert slots.text is None

    def test_legacy_only_meta_is_migrated(self):
        # No slot keys -> migrate the legacy embedder by capability.
        assert EmbedderSlots.from_meta({"embedder": "siglip"}).text == "siglip"
        assert EmbedderSlots.from_meta({"embedder": "sift_vlad"}).structural == "sift_vlad"
        assert EmbedderSlots.from_meta({"embedder": "dinov2_single"}).is_empty
        assert EmbedderSlots.from_meta({}).is_empty


class TestValidate:
    def test_valid_bindings_pass(self):
        EmbedderSlots().validate()
        EmbedderSlots(text="siglip", patch="dinov3_patch", structural="sift_vlad").validate()

    def test_capability_mismatch_raises(self):
        # A patch embedder in the text slot is not eligible.
        import pytest

        with pytest.raises(ValueError, match="not eligible for the 'text' slot"):
            EmbedderSlots(text="dinov3_patch").validate()

    def test_structural_embedder_in_patch_slot_raises(self):
        import pytest

        with pytest.raises(ValueError, match="not eligible for the 'patch' slot"):
            EmbedderSlots(patch="sift_vlad").validate()


class TestExportPersistsSlots:
    """``export_dataset_to_file`` writes slot keys derived from the embedder."""

    @staticmethod
    def _media(i: int) -> dict:
        rng = np.random.default_rng(i)
        return {
            "id": i,
            "media_type": "image",
            "duration": 0.0,
            "file_size": 100,
            "md5": f"md5_{i}",
            "embedding": rng.standard_normal(8).astype(np.float32),
            "filename": f"img_{i}.jpg",
        }

    def _export(self, tmp_path, embedder: str):
        medias = {i: self._media(i) for i in range(3)}
        blob = export_dataset_to_file(medias, embedder=embedder, media_type="image", name="t")
        path = tmp_path / "ds.pkl"
        path.write_bytes(blob)
        return path

    def test_text_embedder_slot_persisted(self, tmp_path):
        path = self._export(tmp_path, "siglip")
        meta = read_meta(path)
        assert meta[TEXT_KEY] == "siglip"
        assert PATCH_KEY not in meta
        assert STRUCTURAL_KEY not in meta
        assert read_pkl_slots(path) == EmbedderSlots(text="siglip")

    def test_plain_single_vector_writes_no_slot_keys(self, tmp_path):
        path = self._export(tmp_path, "dinov2_single")
        meta = read_meta(path)
        assert TEXT_KEY not in meta
        assert PATCH_KEY not in meta
        assert STRUCTURAL_KEY not in meta
        # Legacy field still carries the embedder; slots migrate to empty.
        assert meta["embedder"] == "dinov2_single"
        assert read_pkl_slots(path).is_empty

    def test_structural_embedder_slot_persisted(self, tmp_path):
        path = self._export(tmp_path, "sift_vlad")
        assert read_meta(path)[STRUCTURAL_KEY] == "sift_vlad"
        assert read_pkl_slots(path) == EmbedderSlots(structural="sift_vlad")
