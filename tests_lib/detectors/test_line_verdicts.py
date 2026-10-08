"""A finished test's verdict, kept on the detector (#4526; ``vtscore.detectors.line_verdicts``).

Pinned here: the record a finished test leaves (ids, labels, bands and the
ranges, nothing else) and its round trip through the detector JSON; one
verdict per tested dataset, newest first, malformed entries skipped; the
stale mark, which follows the training labels' signature and nothing else,
and a verdict born stale when the test set was already folded in; resuming,
which takes the kept picks back only on the ranking and line they were drawn
from; and the write to the active detector's file, which keeps the rest of
the file and re-points the cached labelset at it.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from vtscore import config
from vtscore.datasets.labelset import LabeledElement, LabelSet
from vtscore.detectors.line_verdicts import (
    TEST_VERDICTS_KEY,
    LineTestVerdict,
    current_labels_digest,
    drop_verdict,
    forget_verdict,
    keep_verdict,
    kept_verdict,
    labels_digest,
    put_verdict,
    ranking_digest,
    read_verdicts,
    verdict_for,
    verdict_summaries,
)
from vtscore.detectors.store import _detector_path, _read_detector, _write_detector
from vtscore.state.core import DetectorContext
from vtscore.training.thresholds import LineTest, found_words

RANKING = tuple(range(1, 41))
LINE = 10


def _finished(ranking=RANKING, line=LINE, beta: float = 1.0) -> LineTest:
    """A census of the ranking (the top half of the kept set right, nothing below the line), so the test is done."""
    labels = {cid: (i < line // 2) for i, cid in enumerate(ranking)}
    test = LineTest.start(ranking, line, beta, labels=labels)
    assert test.phase().done and not test.nothing_to_test
    return test


def _element(n: int, label: str = "good", **kw) -> LabeledElement:
    return LabeledElement(md5=f"md5-{n}", label=label, origin_name=f"clip_{n}.wav", **kw)


def _labelset(**kw) -> LabelSet:
    return LabelSet([_element(1), _element(2, "bad"), _element(3, **kw)])


def _verdict(dataset_id: str = "ds-1", tested_at: float = 100.0, digest: str | None = "abc") -> LineTestVerdict:
    return LineTestVerdict.from_test(
        _finished(), dataset_id=dataset_id, dataset_name=f"name-{dataset_id}", labels_digest=digest, tested_at=tested_at
    )


class TestTheRecord:
    def test_keeps_ids_labels_bands_and_the_ranges(self):
        test = _finished()
        v = LineTestVerdict.from_test(test, dataset_id="ds-1", dataset_name="drawings-new", labels_digest="abc")
        est = test.estimates()
        assert v.labelled == len(RANKING)
        assert {cid for cid, _, _ in v.picks} == set(RANKING)
        assert all(band == test.pick_band[cid] and right == test.labels[cid] for cid, right, band in v.picks)
        assert (v.precision, v.recall, v.fbeta) == (est.precision, est.recall, est.fbeta)
        assert (v.beta, v.line_count, v.size) == (1.0, LINE, len(RANKING))
        assert v.found == found_words(est.recall)

    def test_round_trips_through_json_and_holds_no_vector(self):
        v = _verdict()
        raw = json.loads(json.dumps(v.to_dict()))
        back = LineTestVerdict.from_dict(raw)
        assert back is not None
        assert back.to_dict() == v.to_dict()
        assert back.picks == v.picks
        # Ids, labels and numbers only (CLAUDE.md, No Persisted Vectors).
        assert set(raw) == {
            "dataset_id",
            "dataset_name",
            "tested_at",
            "beta",
            "line_count",
            "size",
            "ranking_digest",
            "labels_digest",
            "picks",
            "precision",
            "recall",
            "fbeta",
        }
        assert set(raw["picks"][0]) == {"id", "label", "band"}

    def test_a_line_with_nothing_to_test_has_no_verdict(self):
        test = LineTest.start(RANKING, 2, 1.0)
        assert test.nothing_to_test
        with pytest.raises(ValueError, match="nothing to test"):
            LineTestVerdict.from_test(test, dataset_id="ds", dataset_name="", labels_digest="abc")

    @pytest.mark.parametrize(
        "bad",
        [None, 3, "x", {}, {"dataset_id": "ds"}, {**_verdict().to_dict(), "picks": [{"id": "x"}]}],
    )
    def test_an_entry_that_does_not_parse_is_none(self, bad):
        assert LineTestVerdict.from_dict(bad) is None


class TestTheDetectorJson:
    def test_one_verdict_per_dataset_newest_first(self):
        data: dict = {}
        put_verdict(data, _verdict("ds-1", tested_at=100.0))
        put_verdict(data, _verdict("ds-2", tested_at=200.0))
        assert [v.dataset_id for v in read_verdicts(data)] == ["ds-2", "ds-1"]
        put_verdict(data, _verdict("ds-1", tested_at=300.0))
        kept = read_verdicts(data)
        assert [(v.dataset_id, v.tested_at) for v in kept] == [("ds-1", 300.0), ("ds-2", 200.0)]
        second = verdict_for(data, "ds-2")
        assert second is not None and second.tested_at == 200.0
        assert verdict_for(data, "ds-3") is None

    def test_a_malformed_entry_is_skipped_and_dropped_on_the_next_write(self):
        data = {TEST_VERDICTS_KEY: [{"dataset_id": "broken"}, _verdict("ds-1").to_dict()]}
        assert [v.dataset_id for v in read_verdicts(data)] == ["ds-1"]
        put_verdict(data, _verdict("ds-2", tested_at=200.0))
        assert [e["dataset_id"] for e in data[TEST_VERDICTS_KEY]] == ["ds-2", "ds-1"]

    def test_dropping_a_dataset_keeps_the_others(self):
        data: dict = {}
        put_verdict(data, _verdict("ds-1", tested_at=100.0))
        put_verdict(data, _verdict("ds-2", tested_at=200.0))
        assert drop_verdict(data, "ds-1") is True
        assert [v.dataset_id for v in read_verdicts(data)] == ["ds-2"]
        assert drop_verdict(data, "ds-1") is False

    @pytest.mark.parametrize("data", [None, {}, {TEST_VERDICTS_KEY: "nope"}, {TEST_VERDICTS_KEY: None}])
    def test_no_verdicts(self, data):
        assert read_verdicts(data) == [] and verdict_summaries(data) == []


class TestStale:
    def test_the_digest_follows_what_the_head_trains_from(self):
        base = labels_digest(_labelset())
        assert labels_digest(_labelset()) == base
        assert labels_digest(LabelSet.from_dict(json.loads(json.dumps(_labelset().to_dict())))) == base
        # A flipped label, a new element and a redrawn region each make another head.
        flipped = LabelSet([_element(1, "bad"), _element(2, "bad"), _element(3)])
        assert labels_digest(flipped) != base
        assert labels_digest(LabelSet([*_labelset().elements, _element(4)])) != base
        assert labels_digest(_labelset(region_box=(0.1, 0.1, 0.5, 0.5))) != base
        # Metadata and provenance are not training input.
        assert labels_digest(_labelset(metadata={"vt:provenance": {"flow": "test"}})) == base

    def test_stale_once_the_labels_change(self):
        v = _verdict(digest=labels_digest(_labelset()))
        assert not v.stale(labels_digest(_labelset()))
        assert v.stale(labels_digest(LabelSet([*_labelset().elements, _element(9)])))

    def test_a_verdict_born_stale_stays_stale(self):
        assert _verdict(digest=None).stale(labels_digest(_labelset()))

    def test_summaries_carry_the_stale_mark_against_the_file_labelset(self):
        data = {"labelset": _labelset().to_dict()}
        put_verdict(data, _verdict("ds-1", tested_at=100.0, digest=current_labels_digest(data)))
        put_verdict(data, _verdict("ds-2", tested_at=200.0, digest="other"))
        rows = verdict_summaries(data)
        assert [(r["dataset_id"], r["stale"]) for r in rows] == [("ds-2", True), ("ds-1", False)]
        assert rows[0]["labelled"] == len(RANKING) and rows[0]["found"]
        assert "picks" not in rows[0]
        assert [r["dataset_id"] for r in verdict_summaries(data, latest_only=True)] == ["ds-2"]

    def test_a_summary_names_the_dataset_as_it_is_registered_now(self):
        from vtscore.datasets.registry import register_dataset

        entry = register_dataset(name="renamed-since", media_type="audio", num_items=1, pkl_path="unused.pkl")
        data = {"labelset": _labelset().to_dict()}
        put_verdict(data, _verdict(entry["id"], tested_at=100.0))
        put_verdict(data, _verdict("gone", tested_at=50.0))
        names = {r["dataset_id"]: r["dataset_name"] for r in verdict_summaries(data)}
        assert names == {entry["id"]: "renamed-since", "gone": "name-gone"}


class TestResume:
    def test_the_kept_picks_come_back_on_their_own_ranking_and_line(self):
        v = _verdict()
        labels = v.kept_labels(RANKING, LINE)
        assert labels == {cid: right for cid, right, _ in v.picks}
        resumed = LineTest.start(RANKING, LINE, 1.0, labels=labels)
        assert resumed.phase().done
        assert resumed.estimates().precision == v.precision

    @pytest.mark.parametrize(
        ("ranking", "line"),
        [(RANKING, LINE + 1), (tuple(reversed(RANKING)), LINE), (RANKING[:-1], LINE)],
    )
    def test_another_ranking_or_line_resumes_nothing(self, ranking, line):
        assert _verdict().kept_labels(ranking, line) is None

    def test_the_ranking_digest_is_order_and_line_sensitive(self):
        assert ranking_digest([1, 2, 3], 1) == ranking_digest((1, 2, 3), 1)
        assert ranking_digest([1, 2, 3], 1) != ranking_digest([1, 3, 2], 1)
        assert ranking_digest([1, 2, 3], 1) != ranking_digest([1, 2, 3], 2)


@pytest.fixture
def detector_file(tmp_path):
    """A registered detector with a file in a scratch detectors dir; returns its context and path."""
    from vtscore.detectors.registry import register_detector

    # ``reset_contexts`` re-registers the default builder before the next test.
    scratch = dataclasses.replace(config.CoreConfig.from_settings(), detectors_dir=tmp_path / "detectors")
    config.register_core_config_builder(lambda path=None: scratch)
    entry = register_detector(name="verdict-det", media_type="audio")
    path = _detector_path("verdict-det")
    _write_detector(
        path, {"name": "verdict-det", "media_type": "audio", "text_query": "kept", "labelset": _labelset().to_dict()}
    )
    return DetectorContext(entry["id"], name="verdict-det"), path


def _file(path) -> dict:
    data = _read_detector(path)
    assert data is not None
    return data


def _stale(ctx: DetectorContext, dataset_id: str) -> bool:
    found = kept_verdict(ctx, dataset_id)
    assert found is not None
    return found[1]


class TestKeepingOnTheDetector:
    def test_writes_the_verdict_and_keeps_the_rest_of_the_file(self, detector_file):
        ctx, path = detector_file
        verdict = keep_verdict(ctx, _finished(), dataset_id="ds-1", dataset_name="drawings-new")
        assert verdict is not None
        data = _file(path)
        assert data["text_query"] == "kept" and data["labelset"] == _labelset().to_dict()
        assert [v.to_dict() for v in read_verdicts(data)] == [verdict.to_dict()]
        found = kept_verdict(ctx, "ds-1")
        assert found is not None
        assert found[0].to_dict() == verdict.to_dict() and found[1] is False
        assert kept_verdict(ctx, "ds-2") is None
        # The write re-pointed the cached labelset, so it does not read as an outside edit.
        assert ctx.cached_labelset is not None and ctx.cached_labelset_mtime > 0

    def test_born_stale_after_corrections_folded_the_test_set_in(self, detector_file):
        ctx, _ = detector_file
        ctx.find_eval_stale = True
        verdict = keep_verdict(ctx, _finished(), dataset_id="ds-1", dataset_name="")
        assert verdict is not None and verdict.labels_digest is None
        assert _stale(ctx, "ds-1") is True

    def test_stale_once_the_detector_file_labels_change(self, detector_file):
        ctx, path = detector_file
        keep_verdict(ctx, _finished(), dataset_id="ds-1", dataset_name="")
        data = _file(path)
        data["labelset"] = LabelSet([*_labelset().elements, _element(7)]).to_dict()
        _write_detector(path, data)
        assert _stale(ctx, "ds-1") is True
        assert verdict_summaries(_file(path))[0]["stale"] is True

    def test_nothing_is_kept_without_a_finished_test_or_a_file(self, detector_file):
        ctx, path = detector_file
        running = LineTest.start(RANKING, LINE, 1.0)
        assert keep_verdict(ctx, running, dataset_id="ds-1", dataset_name="") is None
        assert keep_verdict(ctx, _finished(), dataset_id="", dataset_name="") is None
        assert keep_verdict(DetectorContext("unregistered"), _finished(), dataset_id="ds-1", dataset_name="") is None
        assert TEST_VERDICTS_KEY not in _file(path)

    def test_forgetting_drops_one_dataset_and_leaves_the_file_alone_with_nothing_to_drop(self, detector_file):
        ctx, path = detector_file
        keep_verdict(ctx, _finished(), dataset_id="ds-1", dataset_name="")
        keep_verdict(ctx, _finished(), dataset_id="ds-2", dataset_name="")
        assert forget_verdict(ctx, "ds-1") is True
        assert kept_verdict(ctx, "ds-1") is None and kept_verdict(ctx, "ds-2") is not None
        before = path.stat().st_mtime_ns
        assert forget_verdict(ctx, "ds-1") is False
        assert path.stat().st_mtime_ns == before
        assert forget_verdict(DetectorContext("unregistered"), "ds-2") is False
