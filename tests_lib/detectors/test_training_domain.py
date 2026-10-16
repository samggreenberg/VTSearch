"""Unit tests for ``vtscore.detectors.training_domain``: which loaded dataset a detector's labels were cast on (#4961).

The training-domain overlap check needs the dataset a detector was trained on,
and ``votes_dataset_id`` cannot name it (it is restamped on every dataset switch
and detector load).  These pin the replacement: the dataset holding the most of
the labelset's Good / Bad examples, matched by origin or MD5 and never by file
name alone.
"""

from __future__ import annotations

from vtscore.datasets.labelset import LabeledElement, LabelSet
from vtscore.detectors.training_domain import count_labels_held, infer_training_dataset, labels_held_by_dataset
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
