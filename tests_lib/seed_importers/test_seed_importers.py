"""Seed importers: the plugin contract, and what an unlabeled seed means.

A seed importer contributes media that are "close but not quite" what the
user is hunting for.  The whole point of the family is that those items are
*queries*, not verdicts, so the assertions here are mostly about what a
seed does **not** do: it must not become a ``good`` label and must not cast
a vote, while a hand-picked exemplar still does both.
"""

from __future__ import annotations

import pytest

import vtscore.security.path_validation as paths_mod
from vtscore.detectors.media_seeding import (
    is_labeled_example,
    labeled_elements_from_examples,
    merge_examples_into_labelset,
)
from vtscore.datasets.labelset import LabelSet
from vtscore.plugins import PluginField
from vtscore.seed_importers import get_seed_importer, list_seed_importers
from vtscore.seed_importers.base import SeedImporter, SeedMediaItem


@pytest.fixture
def example_media_dir(tmp_path, monkeypatch):
    """Redirect ``example_media/`` to a per-test directory and return it."""
    media_dir = tmp_path / "example_media"
    media_dir.mkdir()
    monkeypatch.setattr(paths_mod, "example_media_dir", lambda: media_dir)
    return media_dir


class TestSeedImporterContract:
    def test_metadata_is_auto_derived_from_the_class(self):
        class NeighborhoodSeedImporter(SeedImporter):
            """Seed from a saved cluster of near-miss media."""

            fields: list[PluginField] = []

        imp = NeighborhoodSeedImporter()
        assert imp.name == "neighborhood"
        assert imp.display_name == "Neighborhood"
        assert imp.description == "Seed from a saved cluster of near-miss media."
        # The family's stock seedling is treated as "no icon chosen", so the
        # concrete plugin gets a distinguishing letter glyph instead.
        assert imp.icon == "N"

    def test_to_dict_carries_the_batch_cap(self):
        class CappedSeedImporter(SeedImporter):
            max_items = 7
            fields: list[PluginField] = []

        d = CappedSeedImporter().to_dict()
        assert d["max_items"] == 7
        assert d["name"] == "capped"

    def test_run_is_abstract(self):
        class BareSeedImporter(SeedImporter):
            fields: list[PluginField] = []

        with pytest.raises(NotImplementedError):
            BareSeedImporter().run({})

    def test_registry_is_empty_but_functional(self):
        # No seed importer ships in-tree: the family exists for third-party
        # entry points, so an unknown name must resolve to None rather than
        # raising, and the roster must be listable.
        assert list_seed_importers() == []
        assert get_seed_importer("nope") is None


class TestIsLabeledExample:
    def test_absent_key_means_labeled(self):
        assert is_labeled_example({"type": "media", "value": "a.wav"})

    def test_explicit_false_is_a_seed(self):
        assert not is_labeled_example({"type": "media", "value": "a.wav", "labeled": False})

    def test_explicit_true_is_labeled(self):
        assert is_labeled_example({"type": "media", "value": "a.wav", "labeled": True})


class TestSeedsAreNotLabels:
    def test_seed_examples_produce_no_labeled_elements(self, example_media_dir):
        (example_media_dir / "seed.wav").write_bytes(b"RIFFseedWAVE")

        elements = labeled_elements_from_examples([{"type": "media", "value": "seed.wav", "labeled": False}])

        assert elements == []

    def test_hand_picked_examples_still_produce_good_labels(self, example_media_dir):
        (example_media_dir / "pick.wav").write_bytes(b"RIFFpickWAVE")

        elements = labeled_elements_from_examples([{"type": "media", "value": "pick.wav"}])

        assert [el.label for el in elements] == ["good"]
        assert elements[0].filename == "pick.wav"

    def test_a_mixed_stack_keeps_only_the_hand_picked_ones(self, example_media_dir):
        (example_media_dir / "pick.wav").write_bytes(b"RIFFpickWAVE")
        (example_media_dir / "seed.wav").write_bytes(b"RIFFseedWAVE")

        elements = labeled_elements_from_examples(
            [
                {"type": "media", "value": "seed.wav", "labeled": False},
                {"type": "media", "value": "pick.wav"},
                {"type": "text", "value": "a dog barking"},
            ]
        )

        assert [el.filename for el in elements] == ["pick.wav"]

    def test_merge_into_labelset_ignores_seeds(self, example_media_dir):
        (example_media_dir / "seed.wav").write_bytes(b"RIFFseedWAVE")
        existing = LabelSet([])

        merged = merge_examples_into_labelset(existing, [{"type": "media", "value": "seed.wav", "labeled": False}])

        # Nothing to add, so the very same object comes back.
        assert merged is existing
        assert len(merged) == 0


class TestSeedMediaItem:
    def test_origin_defaults_to_none(self):
        item = SeedMediaItem(data=b"x", filename="a.wav")
        assert item.origin is None

    def test_origin_round_trips(self):
        origin = {"importer": "url_download", "params": {"url": "https://example.com/a.wav"}}
        item = SeedMediaItem(data=b"x", filename="a.wav", origin=origin)
        assert item.origin == origin
