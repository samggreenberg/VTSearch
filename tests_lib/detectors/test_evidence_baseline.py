"""#5089: the Evidence verdict's training baseline, stamped onto the detector and refused on any mismatch.

The baseline is two distances in the embedder's geometry, recorded where the
detector is tested on its training dataset and read back wherever it travels.
Pinned here: what the stamp covers (any vote, the embedder, its width, ``k`` and
``alpha``), that a reader refuses a stale stamp, and that recording one is a
read-modify-write that keeps every other key of the detector file.
"""

from __future__ import annotations

from vtscore.datasets.labelset import LabeledElement, LabelSet
from vtscore.detectors.evidence_baseline import (
    EVIDENCE_BASELINE_KEY,
    baseline_stamp,
    record_baseline,
    stored_baseline,
)
from vtscore.detectors.store import _read_detector, _write_detector


def _labels(*marks: str) -> LabelSet:
    return LabelSet([LabeledElement(md5=f"m{i}", label=label) for i, label in enumerate(marks)])


def _baseline(stamp: dict) -> dict:
    return {**stamp, "dataset_id": "ds", "n_pos": 30, "n_neg": 500, "cut_pos": 0.41, "cut_neg": 0.52}


class TestTheStamp:
    def test_the_same_labels_in_the_same_embedder_stamp_alike(self):
        assert baseline_stamp(_labels("good", "bad"), "siglip", 768) == baseline_stamp(
            _labels("good", "bad"), "siglip", 768
        )

    def test_a_vote_another_embedder_or_width_restamps(self):
        stamp = baseline_stamp(_labels("good", "bad"), "siglip", 768)
        assert baseline_stamp(_labels("good", "bad", "bad"), "siglip", 768) != stamp
        assert baseline_stamp(_labels("good", "good"), "siglip", 768) != stamp
        assert baseline_stamp(_labels("good", "bad"), "clip", 768) != stamp
        assert baseline_stamp(_labels("good", "bad"), "siglip", 1152) != stamp
        assert baseline_stamp(_labels("good", "bad"), "siglip", 768, alpha=0.01) != stamp

    def test_no_embedder_name_no_stamp(self):
        assert baseline_stamp(_labels("good", "bad"), "", 768) is None


class TestReadingItBack:
    def test_a_matching_stamp_reads_the_baseline(self):
        stamp = baseline_stamp(_labels("good", "bad"), "siglip", 768)
        assert stored_baseline({EVIDENCE_BASELINE_KEY: _baseline(stamp)}, stamp) == _baseline(stamp)

    def test_a_stale_stamp_reads_nothing(self):
        old = baseline_stamp(_labels("good", "bad"), "siglip", 768)
        new = baseline_stamp(_labels("good", "bad", "good"), "siglip", 768)
        assert stored_baseline({EVIDENCE_BASELINE_KEY: _baseline(old)}, new) is None

    def test_no_baseline_or_no_stamp_reads_nothing(self):
        stamp = baseline_stamp(_labels("good", "bad"), "siglip", 768)
        assert stored_baseline({}, stamp) is None
        assert stored_baseline(None, stamp) is None
        assert stored_baseline({EVIDENCE_BASELINE_KEY: "junk"}, stamp) is None
        assert stored_baseline({EVIDENCE_BASELINE_KEY: _baseline(stamp)}, None) is None


class TestRecordingIt:
    def test_it_keeps_every_other_key_and_writes_once(self, tmp_path):
        path = tmp_path / "birds.json"
        _write_detector(path, {"name": "birds", "labelset": {"labels": []}, "text_query": "a bird"})
        baseline = _baseline(baseline_stamp(_labels("good", "bad"), "siglip", 768))
        assert record_baseline(path, baseline) is True
        data = _read_detector(path)
        assert data == {
            "name": "birds",
            "labelset": {"labels": []},
            "text_query": "a bird",
            EVIDENCE_BASELINE_KEY: baseline,
        }
        assert record_baseline(path, baseline) is False

    def test_a_new_baseline_replaces_the_old(self, tmp_path):
        path = tmp_path / "birds.json"
        _write_detector(path, {"name": "birds"})
        record_baseline(path, _baseline(baseline_stamp(_labels("good", "bad"), "siglip", 768)))
        newer = _baseline(baseline_stamp(_labels("good", "bad", "bad"), "siglip", 768))
        assert record_baseline(path, newer) is True
        assert _read_detector(path)[EVIDENCE_BASELINE_KEY] == newer

    def test_no_detector_file_writes_nothing(self, tmp_path):
        assert record_baseline(tmp_path / "gone.json", {"cut_pos": 0.1}) is False
        assert not (tmp_path / "gone.json").exists()
