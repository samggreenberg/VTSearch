"""Tests for ``vtscore.cli._top_records``, the bounded heap behind ``--max-results`` (#5279).

A ``--stream-results`` run writes each chunk's hits as it scores them, so its
output is chunk-ordered.  ``--max-results N`` keeps each detector's N best
records instead and writes them sorted once the last chunk is scored.  The
point of the mode is that it still runs over a source larger than RAM, so
besides the ordering these tests pin the memory: no list of every hit is ever
held.
"""

from __future__ import annotations

import json
import weakref
from collections.abc import Iterator

import pytest

from vtscore import cli_progress
from vtscore.cli import _check_max_results, _top_records


class _Hit(dict):
    """A hit dict that can be weakly referenced, so a test can count the live ones."""


def _hit(name: str, score: float, label: str = "good") -> _Hit:
    return _Hit(filename=name, score=score, label=label)


def _names(records) -> list[tuple[str, str]]:
    return [(det, hit["filename"]) for det, hit in records]


class TestOrdering:
    def test_keeps_the_best_n_across_chunks_highest_first(self):
        # Chunk order, as _stream_hit_records yields it: the best hit arrives last.
        records = [("dogs", _hit(f"h{i}", score)) for i, score in enumerate([0.6, 0.9, 0.7, 0.5, 0.95, 0.8])]
        assert _names(_top_records(iter(records), 3, ["dogs"])) == [("dogs", "h4"), ("dogs", "h1"), ("dogs", "h5")]

    def test_fewer_records_than_n_are_all_written_sorted(self):
        records = [("dogs", _hit("a", 0.6)), ("dogs", _hit("b", 0.9))]
        assert _names(_top_records(iter(records), 10, ["dogs"])) == [("dogs", "b"), ("dogs", "a")]

    def test_ties_keep_the_order_they_arrived_in(self):
        records = [("dogs", _hit(name, 0.7)) for name in "abcd"]
        assert _names(_top_records(iter(records), 3, ["dogs"])) == [("dogs", "a"), ("dogs", "b"), ("dogs", "c")]

    def test_each_detector_keeps_its_own_n_in_the_header_order(self):
        records = [
            ("cats", _hit("c1", 0.5)),
            ("dogs", _hit("d1", 0.6)),
            ("cats", _hit("c2", 0.9)),
            ("dogs", _hit("d2", 0.8)),
            ("dogs", _hit("d3", 0.7)),
        ]
        assert _names(_top_records(iter(records), 2, ["dogs", "cats"])) == [
            ("dogs", "d2"),
            ("dogs", "d3"),
            ("cats", "c2"),
            ("cats", "c1"),
        ]

    def test_bad_records_rank_below_every_good_one(self):
        """With --keep-negatives the bads only fill the places the hits leave.

        Scores are rounded to 4 places, so a bad can tie a good at the cut; the
        good one still comes first.
        """
        records = [
            ("dogs", _hit("bad-high", 0.5, "bad")),
            ("dogs", _hit("good", 0.5)),
            ("dogs", _hit("bad-low", 0.1, "bad")),
        ]
        assert _names(_top_records(iter(records), 2, ["dogs"])) == [("dogs", "good"), ("dogs", "bad-high")]

    def test_a_detector_with_no_records_writes_none(self):
        assert _names(_top_records(iter([("dogs", _hit("a", 0.6))]), 5, ["dogs", "cats"])) == [("dogs", "a")]


class TestStreaming:
    def test_nothing_is_written_before_the_last_record_is_read(self):
        """The last chunk can hold the best hit, so the exporter must wait for it."""
        exhausted = []

        def records() -> Iterator[tuple[str, dict]]:
            yield "dogs", _hit("a", 0.6)
            yield "dogs", _hit("b", 0.9)
            exhausted.append(True)

        top = _top_records(records(), 1, ["dogs"])
        assert _names([next(top)]) == [("dogs", "b")]
        assert exhausted == [True]

    def test_peak_memory_is_n_hits_per_detector_not_the_run(self):
        """However many records stream through, only the kept ones stay alive.

        Every hit is registered in a WeakSet as it is made; a hit the heap
        drops (or never takes) is freed at once, so the set's size is the
        number of hits alive at that moment.  Ten thousand records through a
        heap of five per detector must never have more than the heaps' ten
        alive, plus the new one and the previous one, which the consuming
        loop's variables still name until they are rebound.
        """
        alive: weakref.WeakSet[_Hit] = weakref.WeakSet()
        peak = 0

        def records() -> Iterator[tuple[str, dict]]:
            nonlocal peak
            for i in range(10_000):
                hit = _hit(f"h{i}", (i * 7919 % 10_007) / 10_007)
                alive.add(hit)
                peak = max(peak, len(alive))
                yield ("dogs" if i % 2 else "cats"), hit
                del hit

        out = list(_top_records(records(), 5, ["dogs", "cats"]))
        assert len(out) == 10
        assert peak <= 2 * 5 + 2

    def test_reports_how_many_each_detector_scored_and_kept(self, capsys):
        cli_progress.set_format("json")
        records = [("dogs", _hit(f"h{i}", i / 10)) for i in range(4)]
        list(_top_records(iter(records), 3, ["dogs", "cats"]))
        events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        assert [{k: e[k] for k in ("event", "detector", "kept", "seen", "max_results")} for e in events] == [
            {"event": "top_results", "detector": "dogs", "kept": 3, "seen": 4, "max_results": 3},
            {"event": "top_results", "detector": "cats", "kept": 0, "seen": 0, "max_results": 3},
        ]


class TestRefusals:
    def test_no_max_results_is_always_fine(self):
        _check_max_results(None, stream_results=False, resume=True)

    def test_requires_stream_results(self):
        with pytest.raises(ValueError, match="requires --stream-results"):
            _check_max_results(5, stream_results=False, resume=False)

    def test_cannot_be_resumed(self):
        with pytest.raises(ValueError, match="cannot be combined with --resume"):
            _check_max_results(5, stream_results=True, resume=True)

    @pytest.mark.parametrize("value", [0, -2, True, 2.5, "10"])
    def test_must_be_a_positive_integer(self, value):
        with pytest.raises(ValueError, match="must be a positive integer"):
            _check_max_results(value, stream_results=True, resume=False)
