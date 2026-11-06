"""Unit tests for ``vtscore.detectors.training_domain``: which loaded dataset a detector's labels were cast on (#4961).

The training-domain overlap check needs the dataset a detector was trained on,
and ``votes_dataset_id`` cannot name it (it is restamped on every dataset switch
and detector load).  These pin the replacement: the dataset holding the most of
the labelset's Good / Bad examples, matched by origin or MD5 and never by file
name alone, and not counting the media a label import copied in (#5572).
"""

from __future__ import annotations

from vtscore.datasets.labelset import LabeledElement, LabelSet
from vtscore.detectors.training_domain import (
    INGESTED_LABELS_KEY,
    TRAINING_DATASET_KEY,
    count_labels_held,
    infer_training_dataset,
    labels_held_by_dataset,
    record_ingested_labels,
    record_ingested_labels_on_file,
    recorded_ingests,
    recorded_training_dataset,
    training_dataset_record,
)
from vtscore.state.core import DatasetContext

_ORIGIN = {"importer": "folder", "params": {"path": "/photos"}}


def _media(cid: int, name: str, *, md5: str = "", origin: dict | None = None) -> dict:
    return {"id": cid, "md5": md5 or f"md5-{name}", "origin": origin, "origin_name": name, "filename": name}


def _dataset(dataset_id: str, names: list[str], *, origin: dict | None = _ORIGIN) -> DatasetContext:
    ctx = DatasetContext(dataset_id)
    for cid, name in enumerate(names):
        ctx.medias[cid] = _media(cid, name, origin=origin)
    return ctx


def _labelset(*names: str, label: str = "good") -> LabelSet:
    return LabelSet([LabeledElement(md5=f"md5-{n}", label=label, origin=_ORIGIN, origin_name=n) for n in names])


class TestCountLabelsHeld:
    def test_counts_elements_naming_a_media(self):
        ctx = _dataset("photos", ["a.jpg", "b.jpg", "c.jpg"])
        assert count_labels_held(_labelset("a.jpg", "b.jpg", "z.jpg"), ctx) == 2

    def test_matches_by_md5_when_the_origin_differs(self):
        # The same file imported through another importer: content hash alone.
        ctx = _dataset("copy", ["a.jpg"], origin={"importer": "zip", "params": {}})
        assert count_labels_held(_labelset("a.jpg"), ctx) == 1

    def test_a_shared_file_name_is_not_a_match(self):
        """Two datasets with an ``img_001.jpg`` each say nothing about where a label was cast."""
        ctx = DatasetContext("screens")
        ctx.medias[0] = _media(0, "a.jpg", md5="other-content", origin={"importer": "zip", "params": {}})
        assert count_labels_held(_labelset("a.jpg"), ctx) == 0

    def test_only_good_and_bad_count(self):
        ctx = _dataset("photos", ["a.jpg", "b.jpg"])
        labelset = _labelset("a.jpg")
        labelset.elements.append(LabeledElement(md5="md5-b.jpg", label="skip", origin=_ORIGIN, origin_name="b.jpg"))
        assert count_labels_held(labelset, ctx) == 1

    def test_an_empty_dataset_holds_nothing(self):
        assert count_labels_held(_labelset("a.jpg"), DatasetContext("empty")) == 0

    def test_counts_per_dataset(self):
        photos = _dataset("photos", ["a.jpg", "b.jpg"])
        screens = _dataset("screens", ["s1.png"])
        counts = labels_held_by_dataset(_labelset("a.jpg", "b.jpg", "s1.png"), [photos, screens])
        assert counts == {"photos": 2, "screens": 1}


class TestInferTrainingDataset:
    def test_most_labels_wins(self):
        assert infer_training_dataset({"photos": 40, "screens": 3}, prefer="screens") == "photos"

    def test_the_preferred_dataset_wins_a_tie(self):
        """A detector used where it holds as many examples as anywhere has nothing to warn about."""
        assert infer_training_dataset({"photos": 5, "screens": 5}, prefer="screens") == "screens"

    def test_other_ties_go_to_the_smallest_id(self):
        assert infer_training_dataset({"b": 5, "a": 5, "c": 1}, prefer="c") == "a"

    def test_nothing_held_is_no_answer(self):
        assert infer_training_dataset({"photos": 0, "screens": 0}, prefer="screens") == ""
        assert infer_training_dataset({}) == ""


class TestRecordedTrainingDataset:
    """#5501: the pointer a detector carries to its training dataset, for when that dataset is not loaded."""

    def test_round_trips(self):
        record = training_dataset_record("ds-1", "Photos", 39)
        assert recorded_training_dataset({TRAINING_DATASET_KEY: record}) == {
            "dataset_id": "ds-1",
            "dataset_name": "Photos",
            "labels_held": 39,
        }

    def test_absent_or_malformed_reads_nothing(self):
        assert recorded_training_dataset(None) is None
        assert recorded_training_dataset({}) is None
        assert recorded_training_dataset({TRAINING_DATASET_KEY: "ds-1"}) is None
        assert recorded_training_dataset({TRAINING_DATASET_KEY: {"dataset_name": "Photos", "labels_held": 3}}) is None
        assert recorded_training_dataset({TRAINING_DATASET_KEY: {"dataset_id": "ds-1", "labels_held": "many"}}) is None

    def test_a_dataset_holding_no_label_reads_nothing(self):
        assert recorded_training_dataset({TRAINING_DATASET_KEY: training_dataset_record("ds-1", "Photos", 0)}) is None


def _entry(name: str) -> dict:
    """A label entry, as an import hands it over, naming media *name* of the photos origin."""
    return {"md5": f"md5-{name}", "label": "good", "origin": _ORIGIN, "origin_name": name, "filename": name}


class TestIngestedLabels:
    """#5572: a label on a media an import copied into a dataset says nothing about where it was cast."""

    def test_a_label_on_a_copied_in_media_is_not_counted(self):
        screens = _dataset("screens", ["s1.png", "s2.png"], origin={"importer": "zip", "params": {}})
        # What the ingest added: the photos the labels name, under their own origin.
        for cid, name in enumerate(["a.jpg", "b.jpg"], start=10):
            screens.medias[cid] = _media(cid, name, origin=_ORIGIN)
        labelset = _labelset("a.jpg", "b.jpg")

        assert count_labels_held(labelset, screens) == 2
        assert count_labels_held(labelset, screens, ingested=[_entry("a.jpg"), _entry("b.jpg")]) == 0
        assert count_labels_held(labelset, screens, ingested=[_entry("a.jpg")]) == 1

    def test_labels_on_the_datasets_own_media_still_count(self):
        """The receiver's own votes there are evidence, beside the copied-in ones."""
        screens = _dataset("screens", ["s1.png"], origin={"importer": "zip", "params": {}})
        screens.medias[10] = _media(10, "a.jpg", origin=_ORIGIN)
        labelset = _labelset("a.jpg")
        labelset.elements.append(
            LabeledElement(
                md5="md5-s1.png", label="bad", origin={"importer": "zip", "params": {}}, origin_name="s1.png"
            )
        )

        assert count_labels_held(labelset, screens, ingested=[_entry("a.jpg")]) == 1

    def test_only_the_dataset_it_was_copied_into_discounts_it(self):
        photos = _dataset("photos", ["a.jpg"])
        screens = _dataset("screens", ["a.jpg"])
        counts = labels_held_by_dataset(_labelset("a.jpg"), [photos, screens], ingested={"screens": [_entry("a.jpg")]})
        assert counts == {"photos": 1, "screens": 0}

    def test_the_record_names_media_and_keeps_each_once(self):
        data: dict = {}
        assert record_ingested_labels(data, "screens", [_entry("a.jpg"), _entry("b.jpg")]) is True
        assert data[INGESTED_LABELS_KEY] == {
            "screens": [
                {"md5": "md5-a.jpg", "origin": _ORIGIN, "origin_name": "a.jpg", "filename": "a.jpg"},
                {"md5": "md5-b.jpg", "origin": _ORIGIN, "origin_name": "b.jpg", "filename": "b.jpg"},
            ]
        }
        # The label itself is not part of it: a re-import that flips one changes nothing.
        assert record_ingested_labels(data, "screens", [{**_entry("a.jpg"), "label": "bad"}]) is False
        assert record_ingested_labels(data, "other", [_entry("c.jpg")]) is True
        assert set(recorded_ingests(data)) == {"screens", "other"}

    def test_nothing_to_name_records_nothing(self):
        data: dict = {}
        assert record_ingested_labels(data, "", [_entry("a.jpg")]) is False
        assert record_ingested_labels(data, "screens", []) is False
        assert record_ingested_labels(data, "screens", [{"label": "good", "filename": "a.jpg"}]) is False
        assert data == {}

    def test_absent_or_malformed_reads_nothing(self):
        assert recorded_ingests(None) == {}
        assert recorded_ingests({INGESTED_LABELS_KEY: "screens"}) == {}
        assert recorded_ingests({INGESTED_LABELS_KEY: {"screens": "a.jpg", "": [_entry("a.jpg")]}}) == {}
        assert recorded_ingests({INGESTED_LABELS_KEY: {"screens": [7, {"filename": "x"}, _entry("a.jpg")]}}) == {
            "screens": [{"md5": "md5-a.jpg", "origin": _ORIGIN, "origin_name": "a.jpg", "filename": "a.jpg"}]
        }

    def test_on_file_it_keeps_the_labels_and_writes_once(self, tmp_path):
        from vtscore.detectors.store import _read_detector, _write_detector

        path = tmp_path / "copy.json"
        _write_detector(path, {"name": "copy", "labelset": {"labels": [_entry("a.jpg")]}})

        assert record_ingested_labels_on_file(path, "screens", [_entry("a.jpg")]) is True
        on_disk = _read_detector(path) or {}
        assert on_disk["labelset"] == {"labels": [_entry("a.jpg")]}
        assert list(recorded_ingests(on_disk)) == ["screens"]
        assert record_ingested_labels_on_file(path, "screens", [_entry("a.jpg")]) is False
        assert record_ingested_labels_on_file(tmp_path / "gone.json", "screens", [_entry("a.jpg")]) is False
