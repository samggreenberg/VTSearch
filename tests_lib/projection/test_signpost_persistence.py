"""Library-tier tests for region-label persistence in the dataset container.

Covers ``append_region_labels`` / ``read_region_labels`` /
``remove_region_labels`` (``vtscore.datasets.container``): the JSON
round-trip, replacement semantics, and the rule that discarding a persisted
projection discards its labels too (anchors are meaningless off their layout).
"""

from __future__ import annotations

import zipfile

import numpy as np
import pytest

from vtscore.datasets.container import (
    append_projection,
    append_region_labels,
    read_projection,
    read_region_labels,
    remove_projections,
    remove_region_labels,
)
from vtscore.projection import build_pyramid
from vtscore.projection.labels import RegionLabel, make_label_set
from vtscore.projection.umap_projection import Projection


@pytest.fixture
def container(tmp_path):
    path = tmp_path / "dataset.pkl"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("placeholder", b"")
    return path


def _label_set(projection_id="proj-1"):
    return make_label_set(
        projection_id,
        [
            RegionLabel(level=0.0, x=1.5, y=-2.25, text="animal sounds", score=40.0, source="keyphrase"),
            RegionLabel(level=1.8, x=0.5, y=0.5, text="dog barking", score=12.0, source="keyphrase"),
        ],
    )


class TestRoundTrip:
    def test_append_and_read(self, container):
        append_region_labels(container, _label_set(), "sig-v1")
        loaded = read_region_labels(container)
        assert loaded is not None
        label_set, signature = loaded
        assert signature == "sig-v1"
        assert label_set.projection_id == "proj-1"
        assert label_set.labels == _label_set().labels

    def test_append_replaces_previous(self, container):
        append_region_labels(container, _label_set("old"), "sig-old")
        append_region_labels(container, _label_set("new"), "sig-new")
        loaded = read_region_labels(container)
        assert loaded is not None
        label_set, signature = loaded
        assert (label_set.projection_id, signature) == ("new", "sig-new")

    def test_terminal_flags_round_trip(self, container):
        # A leaf/root sign's terminal flags must survive the JSON round-trip so a
        # reloaded layout letters its islands the same way it did when built.
        label_set = make_label_set(
            "proj-1",
            [RegionLabel(level=3.6, x=0.0, y=0.0, text="island", has_coarser=False, has_finer=False)],
        )
        append_region_labels(container, label_set, "sig")
        loaded, _ = read_region_labels(container)
        (sign,) = loaded.labels
        assert (sign.has_coarser, sign.has_finer) == (False, False)

    def test_read_missing_entry_returns_none(self, container):
        assert read_region_labels(container) is None

    def test_read_unreadable_container_returns_none(self, tmp_path):
        assert read_region_labels(tmp_path / "nope.pkl") is None

    def test_remove(self, container):
        append_region_labels(container, _label_set(), "sig")
        remove_region_labels(container)
        assert read_region_labels(container) is None
        remove_region_labels(container)  # no-op on absence


class TestRemoveProjectionsDropsLabels:
    def test_labels_go_with_the_layout(self, container):
        rng = np.random.default_rng(5)
        coords = rng.standard_normal((6, 2)).astype(np.float32)
        proj = Projection("proj-1", list(range(1, 7)), coords, "pca")
        pyr = build_pyramid(proj, n_levels=2)
        append_projection(container, proj, pyr)
        append_region_labels(container, _label_set("proj-1"), "sig")

        remove_projections(container)
        assert read_projection(container, pyr.bin_shape) is None
        assert read_region_labels(container) is None
