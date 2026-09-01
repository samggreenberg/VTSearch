"""A pre-computed vector never reaches a ``LabeledElement``'s metadata.

``LabelSet`` carries a media's importer ``custom_metadata`` onto every element
it emits, and a detector's labelset is written to disk as JSON
(``vtscore.detectors.store``).  ``custom_metadata_map`` lets an importer nest
a pre-computed vector inside that dict, so reading it verbatim would both
crash the detector write (``json.dump`` has no numpy encoder) and persist a
vector - the one thing the no-persisted-vectors rule forbids outright.
"""

from __future__ import annotations

import json

import numpy as np

from vtscore.datasets.labelset import LabelSet


def _media(cid: int, custom: dict) -> dict:
    return {
        "id": cid,
        "md5": f"md5-{cid}",
        "filename": f"a{cid}.wav",
        "category": "audio",
        "origin_name": f"a{cid}.wav",
        "custom_metadata": custom,
    }


class TestLabelSetCustomMetadata:
    def test_importer_metadata_survives_without_the_vector(self):
        medias = {1: _media(1, {"asset_id": "XY-7", "embedding": np.zeros(4, dtype=np.float32)})}
        labelset = LabelSet.from_clips_and_votes(medias, {1: None}, {}, expand_dupes=False)

        (element,) = labelset.elements
        assert element.metadata == {"asset_id": "XY-7"}

    def test_labelset_stays_json_writable(self):
        """``_write_detector`` calls ``json.dump`` on exactly this dict."""
        medias = {1: _media(1, {"asset_id": "XY-7", "embedding": np.zeros(4, dtype=np.float32)})}
        labelset = LabelSet.from_clips_and_votes(medias, {1: None}, {}, expand_dupes=False)

        round_tripped = json.loads(json.dumps(labelset.to_dict()))
        assert round_tripped["labels"][0]["metadata"] == {"asset_id": "XY-7"}

    def test_vector_only_metadata_becomes_none(self):
        medias = {1: _media(1, {"embedding": np.zeros(4, dtype=np.float32)})}
        labelset = LabelSet.from_clips_and_votes(medias, {1: None}, {}, expand_dupes=False)

        (element,) = labelset.elements
        assert element.metadata is None

    def test_dupe_set_members_share_the_sanitised_metadata(self):
        """The dupe-expansion branch clones the metadata onto every member."""
        media = _media(1, {"asset_id": "XY-7", "embedding": np.zeros(4, dtype=np.float32)})
        media["origin"] = {
            "importer": "dupe_set",
            "members": [
                {"md5": "AAA", "filename": "a.wav", "origin_name": "a.wav"},
                {"md5": "BBB", "filename": "b.wav", "origin_name": "b.wav"},
            ],
        }
        labelset = LabelSet.from_clips_and_votes({1: media}, {1: None}, {})

        assert len(labelset.elements) == 2
        for element in labelset.elements:
            assert element.metadata == {"asset_id": "XY-7"}
