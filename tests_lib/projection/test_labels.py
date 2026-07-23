"""Region signpost label containers (``vtscore.projection.labels``).

Library-tier: the data contract between the labeling pipeline and the browse
canvas — no Flask, no routes. The serving/staleness rules live in the app-tier
``tests/api/test_projection.py``.
"""

from __future__ import annotations

import pytest

from vtscore.projection import RegionLabel, make_label_set


class TestRegionLabelSet:
    def test_payload_is_json_ready(self):
        label_set = make_label_set(
            "proj-1",
            [
                RegionLabel(level=0, x=1.5, y=-2.0, text="speech", score=0.8, source="llm"),
                RegionLabel(level=2.5, x=0.0, y=0.0, text="dog barking"),
            ],
        )
        payload = label_set.payload()
        assert payload == [
            {"level": 0, "x": 1.5, "y": -2.0, "text": "speech", "score": 0.8, "source": "llm",
             "has_coarser": True, "has_finer": True},
            {"level": 2.5, "x": 0.0, "y": 0.0, "text": "dog barking", "score": 1.0, "source": "",
             "has_coarser": True, "has_finer": True},
        ]

    def test_make_label_set_normalises_to_tuple(self):
        label_set = make_label_set("proj-1", iter([RegionLabel(level=0, x=0, y=0, text="a")]))
        assert isinstance(label_set.labels, tuple)
        assert label_set.projection_id == "proj-1"

    def test_labels_are_immutable(self):
        label = RegionLabel(level=0, x=0, y=0, text="a")
        with pytest.raises(AttributeError):
            label.text = "b"  # type: ignore[misc]
