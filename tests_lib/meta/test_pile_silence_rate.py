"""VG's silence rate, and the four things that decide what it means (#3696).

`silence_rate.py` counts how often VG failed to name a class a human then found.
The count itself is arithmetic; everything that makes it an *upper bound on the
uniform off-COCO rate* rather than a number is in the bookkeeping around it, and
none of that is visible in the rate it prints:

* **the denominator is the pairs VG was silent on** -- the queue's images minus,
  per class, the ones it designated. Counting the designations in would measure
  something else entirely and would only ever make the rate look smaller;
* **an unfinished slate is not a low rate.** Above-cut candidates nobody has
  voted on are neither positives nor negatives, so they enter the bound at their
  worst case and keep the class out of the pooled figure;
* **screening loss is allowed for, not assumed away.** What sits below a class's
  cut was never shown to anyone, and #3768's random below-cut samples are the
  only thing that sizes it;
* **a verdict answers the rule its reviewer was shown.** Three rulings have moved
  a rule's name since the slates they cover were voted, and folding those in
  would answer today's question with yesterday's boundary (#3814).

The last two are what the issue means by "the care is all in stating what the
number is conditioned on", so they get tests rather than comments.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PILE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "pile"


@pytest.fixture(scope="module")
def sr():
    """``silence_rate``, which defers ``setup_env`` into ``main``."""
    if str(_PILE_DIR) not in sys.path:
        sys.path.insert(0, str(_PILE_DIR))
    import silence_rate

    return silence_rate


def _queue(rows: list[tuple[int, list[str]]]) -> list[dict]:
    """The worklist as `annotation_queue.py` writes it: one row per image."""
    return [
        {"image_id": iid, "filename": f"{iid}.jpg", "cells": [f"{c}@large" for c in classes], "classes": classes}
        for iid, classes in rows
    ]


def _dets(rows: dict[int, dict[str, float]]) -> list[dict]:
    """OWLv2's output: one row an image, one detection a class."""
    return [
        {"image_id": iid, "dets": [{"cls": c, "score": s, "box": [0, 0, 1, 1]} for c, s in scores.items()]}
        for iid, scores in rows.items()
    ]


def _labelset(path: Path, cls: str, kind: str, rule: str, good: list[int], bad: list[int], digest: str = "") -> None:
    doc = {
        "class": cls,
        "kind": kind,
        "rule": rule,
        "good": [{"filename": f"{i}.jpg", "label": "good"} for i in good],
        "bad": [{"filename": f"{i}.jpg", "label": "bad"} for i in bad],
    }
    if digest:
        doc["rule_digest"] = digest
    path.write_text(json.dumps(doc))


class TestTheDenominatorIsSilence:
    """The population is the pairs VG did not name, not the queue."""

    def test_a_designated_class_leaves_its_own_denominator(self, sr):
        queue = _queue([(1, ["car"]), (2, ["car"]), (3, ["dog"])])
        silent = sr.silent_pairs(queue, ("car", "dog"))
        assert silent["car"] == {3}
        assert silent["dog"] == {1, 2}

    def test_a_class_the_queue_never_designates_still_has_a_denominator(self, sr):
        """Absent is not zero: `bus` is silent on every image here, not on none."""
        silent = sr.silent_pairs(_queue([(1, ["car"]), (2, ["car"])]), ("car", "bus"))
        assert silent["bus"] == {1, 2}

    def test_two_bands_of_one_class_are_one_image(self, sr):
        """A class counted once per image however many of its cells name it."""
        queue = _queue([(1, ["car"])])
        queue[0]["cells"] = ["car@small", "car@large"]
        assert sr.silent_pairs(queue, ("car", "dog"))["car"] == set()


class TestTheCutIsTheScreensOwnDecision:
    def test_only_scores_at_or_above_the_cut_are_candidates(self, sr):
        above = sr.above_cut(_dets({1: {"car": 0.30}, 2: {"car": 0.10}, 3: {"car": 0.05}}), {"car": 0.10})
        assert above["car"] == {1, 2}

    def test_a_class_with_no_cut_is_not_invented(self, sr):
        """The cuts come from the slates that were actually built; a class with
        no slate has no screen, and reading its detections anyway would give it
        candidates nobody ever cut."""
        assert "dog" not in sr.above_cut(_dets({1: {"car": 0.9, "dog": 0.9}}), {"car": 0.1})


class TestAnUnfinishedSlateIsNotALowRate:
    def test_unreviewed_candidates_enter_the_bound_at_their_worst_case(self, sr):
        silent = {"car": set(range(100))}
        above = {"car": set(range(10))}
        answers = {"car": {0: True, 1: False}}
        prov = {"car": {"rule_in_force": True, "labelsets": []}}
        rows, pooled, _ = sr.measure(silent, answers, {}, above, prov)
        (row,) = rows
        assert row["found_present"] == 1
        assert row["unreviewed_candidates"] == 8
        assert row["slate_complete"] is False
        # 8 unreviewed candidates of 100 silent pairs is 8 points of slack on
        # top of the Wilson upper end -- the class reads as thin, not as clean.
        assert row["bound"] >= row["wilson95"][1] + 0.08
        assert pooled["classes"] == []

    def test_a_finished_slate_pools(self, sr):
        silent = {"car": set(range(100))}
        above = {"car": {0, 1}}
        answers = {"car": {0: True, 1: False}}
        prov = {"car": {"rule_in_force": True, "labelsets": []}}
        rows, pooled, _ = sr.measure(silent, answers, {}, above, prov)
        assert rows[0]["slate_complete"] is True
        assert pooled["classes"] == ["car"]
        assert pooled["found_present"] == 1
        assert pooled["silent_pairs"] == 100

    def test_a_class_nobody_has_started_never_reads_as_zero(self, sr):
        """Its whole candidate set is unreviewed, so its bound is ~its share of
        the queue -- the one reading that cannot be mistaken for a clean class."""
        rows, pooled, _ = sr.measure({"car": set(range(100))}, {}, {}, {"car": set(range(40))}, {})
        assert rows[0]["rate"] == 0.0
        assert rows[0]["bound"] >= 0.40
        assert pooled["classes"] == []


class TestScreeningLossIsAllowedFor:
    def test_the_unseen_below_cut_mass_carries_a_measured_ceiling(self, sr):
        """#3768 drew below-cut images and found none; the allowance is the upper
        end of that, not zero."""
        silent = {"car": set(range(1000))}
        above = {"car": {0}}
        below_seen = {"car": {i: False for i in range(500, 600)}}
        prov = {"car": {"rule_in_force": True, "labelsets": []}}
        rows, _, _ = sr.measure(silent, {"car": {0: False}}, below_seen, above, prov)
        (row,) = rows
        assert row["below_cut_sample"] == 100
        assert row["below_cut_rate_ub"] > 0
        # 899 below-cut pairs are still unseen, and they carry that ceiling.
        assert row["unreviewed_below_cut"] == 899
        assert row["bound"] == pytest.approx(row["wilson95"][1] + row["below_cut_rate_ub"] * 899 / 1000, abs=1e-9)

    def test_measuring_a_class_never_makes_its_own_allowance_worse(self, sr):
        """The incoherence this replaced: 0 of 100 bounds at 3.70% and 0 of 600 at
        0.64%, so taking a class's own empty sample where it existed and the
        pooled one elsewhere gave the *sampled* class the looser ceiling. Every
        class that found nothing gets the pooled one."""
        silent = {c: set(range(1000)) for c in ("car", "dog", "bus")}
        above = {c: {0} for c in silent}
        below_seen = {c: {i: False for i in range(500, 600)} for c in ("car", "dog")}
        prov = {c: {"rule_in_force": True, "labelsets": []} for c in silent}
        rows, _, _ = sr.measure(silent, {c: {0: False} for c in silent}, below_seen, above, prov)
        by_class = {r["class"]: r for r in rows}
        assert by_class["car"]["below_cut_sample"] == 100
        assert by_class["bus"]["below_cut_sample"] == 0  # never sampled
        assert by_class["car"]["below_cut_rate_ub"] == by_class["bus"]["below_cut_rate_ub"]
        # ...and the pooled ceiling is genuinely tighter than the class's own.
        assert by_class["car"]["below_cut_rate_ub"] < by_class["car"]["below_cut_rate_ub_own"]

    def test_a_class_whose_own_sample_found_something_keeps_its_own_ceiling(self, sr):
        """Hits are evidence the pooled rate does not describe it, and pooling
        would dilute them across the classes that saw none."""
        silent = {c: set(range(1000)) for c in ("car", "dog")}
        above = {c: {0} for c in silent}
        below_seen = {
            "car": {i: (i < 505) for i in range(500, 600)},  # 5 of 100 found
            "dog": {i: False for i in range(500, 600)},
        }
        prov = {c: {"rule_in_force": True, "labelsets": []} for c in silent}
        rows, _, _ = sr.measure(silent, {c: {0: False} for c in silent}, below_seen, above, prov)
        by_class = {r["class"]: r for r in rows}
        assert by_class["car"]["below_cut_rate_ub"] == by_class["car"]["below_cut_rate_ub_own"]
        assert by_class["car"]["below_cut_rate_ub"] > by_class["dog"]["below_cut_rate_ub"]

    def test_the_price_of_the_pooling_assumption_is_reported_not_hidden(self, sr):
        """`bound_unpooled` holds every class to the loosest single sample, so a
        reader can price the exchangeability rather than inherit it."""
        silent = {c: set(range(1000)) for c in ("car", "dog", "bus")}
        above = {c: {0} for c in silent}
        below_seen = {c: {i: False for i in range(500, 600)} for c in ("car", "dog")}
        prov = {c: {"rule_in_force": True, "labelsets": []} for c in silent}
        rows, pooled, _ = sr.measure(silent, {c: {0: False} for c in silent}, below_seen, above, prov)
        assert all(r["bound_unpooled"] >= r["bound"] for r in rows)
        assert pooled["bound_unpooled"] > pooled["bound"]
        assert pooled["worst_below_cut_rate_ub"] > pooled["pooled_below_cut_rate_ub"]

    def test_a_below_cut_positive_is_a_silence_error_like_any_other(self, sr):
        """The wider question proves presence as well as the box question does."""
        prov = {"car": {"rule_in_force": True, "labelsets": []}}
        rows, _, _ = sr.measure({"car": {1, 2}}, {}, {"car": {1: True, 2: False}}, {"car": set()}, prov)
        assert rows[0]["found_present"] == 1

    def test_the_wider_question_wins_where_both_were_asked(self, sr):
        """ "There is none anywhere in this image" settles the pair; "this box is
        not one" does not, so a box-Bad never overrides an image-Good."""
        prov = {"car": {"rule_in_force": True, "labelsets": []}}
        rows, _, _ = sr.measure({"car": {1}}, {"car": {1: False}}, {"car": {1: True}}, {"car": {1}}, prov)
        assert rows[0]["found_present"] == 1


class TestAVerdictAnswersItsOwnRule:
    def test_a_superseded_rule_name_keeps_the_class_out_of_the_pool(self, sr, tmp_path):
        _labelset(tmp_path / "LABELSETS__slate__vase.json", "vase", "slate", "vase incl pots and planters", [1], [2])
        answers, below, prov, problems = sr.read_answers(tmp_path, {"vase": "vase not planters"})
        assert answers["vase"] == {1: True, 2: False}
        assert prov["vase"]["rule_in_force"] is False
        assert any("superseded" in p for p in problems)

        rows, pooled, _ = sr.measure({"vase": {1, 2}}, answers, below, {"vase": {1, 2}}, prov)
        assert rows[0]["found_present"] == 1  # reported, not discarded
        assert pooled["classes"] == []

    def test_an_edited_rule_body_under_an_unchanged_name_is_caught_too(self, sr, tmp_path):
        """#3756 rewrote `bench`'s Bad list and kept its name. A reviewer who
        voted before that edit answered a different boundary under identical
        words, which only the digest can see."""
        _labelset(tmp_path / "LABELSETS__slate__bench.json", "bench", "slate", "bench not chairs", [1], [], "old00")
        _, _, prov, problems = sr.read_answers(tmp_path, {"bench": "bench not chairs"}, {"bench": "new11"})
        assert prov["bench"]["rule_in_force"] is False
        assert any("EDITED" in p for p in problems)

    def test_the_rule_in_force_pools(self, sr, tmp_path):
        _labelset(tmp_path / "LABELSETS__slate__bench.json", "bench", "slate", "bench not chairs", [1], [2], "d0")
        _, _, prov, problems = sr.read_answers(tmp_path, {"bench": "bench not chairs"}, {"bench": "d0"})
        assert prov["bench"]["rule_in_force"] is True
        assert problems == []


class TestTheLabelsetsAreReadTheWayTheyWereWritten:
    def test_a_pass25_labelset_carries_no_kind_and_is_read_by_its_prefix(self, sr, tmp_path):
        """`bank_verdicts.py` wrote no ``kind`` until the slates existed. The
        prefix is an exact match and never a substring: keying off a substring
        is what silently clobbered two banks."""
        path = tmp_path / "LABELSETS__pass25__bicycle.json"
        path.write_text(json.dumps({"class": "bicycle", "rule": "b", "good": [], "bad": []}))
        assert sr.labelset_kind(path, json.loads(path.read_text())) == "pass25"

    def test_a_labelset_with_no_rule_field_is_read_off_its_detector_name(self, sr, tmp_path):
        """The `pass25` labelsets predate the ``rule`` field. Their detector name
        IS the wording the reviewer was shown, so reading it back is what keeps
        them from reporting as answered under a superseded rule."""
        (tmp_path / "LABELSETS__pass25__bench.json").write_text(
            json.dumps({"class": "bench", "detector_name": "bench not chairs", "good": [], "bad": [{"filename": "1"}]})
        )
        _, _, prov, problems = sr.read_answers(tmp_path, {"bench": "bench not chairs"})
        assert prov["bench"]["rule_in_force"] is True
        assert problems == []

    def test_the_store_holds_files_that_are_not_labelsets(self, sr, tmp_path):
        """``LABELSETS__`` names the root a copy came from, not a shape: bare
        verdict lists and whole-dashboard exports share the prefix."""
        (tmp_path / "LABELSETS__verdicts_final_20260824.json").write_text(json.dumps([{"filename": "1.jpg"}]))
        assert sr.read_answers(tmp_path, {"dog": "dog"}) == ({}, {}, {}, [])

    def test_below_cut_verdicts_stay_out_of_the_pass_answers(self, sr, tmp_path):
        _labelset(tmp_path / "LABELSETS__slate__dog.json", "dog", "slate", "dog", [1], [])
        _labelset(tmp_path / "LABELSETS__belowcut__dog.json", "dog", "belowcut", "dog", [], [9])
        answers, below, _, _ = sr.read_answers(tmp_path, {"dog": "dog"})
        assert answers["dog"] == {1: True}
        assert below["dog"] == {9: False}

    def test_one_image_answered_both_ways_is_a_problem_not_a_coin_flip(self, sr, tmp_path):
        _labelset(tmp_path / "LABELSETS__slate__dog.json", "dog", "slate", "dog", [1], [1])
        _, _, _, problems = sr.read_answers(tmp_path, {"dog": "dog"})
        assert any("both good and bad" in p for p in problems)

    def test_an_answer_outside_the_queue_is_drift_and_is_named(self, sr):
        """The shape `apply_recheck` refuses: a verdict with no pair to land on
        means the slate and the population have moved apart."""
        prov = {"car": {"rule_in_force": True, "labelsets": []}}
        _, _, problems = sr.measure({"car": {1}}, {"car": {1: True, 77: True}}, {}, {"car": {1}}, prov)
        assert any("drifted" in p and "77" in p for p in problems)


class TestNotEveryExcludedVerdictIsDrift:
    """The `pass25` pass reviewed one dataset, so its reviewer voted on images
    the silence question does not apply to. Reporting those as drift is an alarm
    that fires on the designed behaviour, which is an alarm nobody reads."""

    def test_the_three_buckets_are_told_apart(self, sr):
        known_a, control, unexplained = sr.classify_excluded({1, 2, 3}, designated={1}, controls={2})
        assert (known_a, control, unexplained) == ({1}, {2}, {3})

    def test_a_known_a_is_not_a_silence_measurement_and_is_not_an_alarm(self, sr):
        """VG *did* name this class in this image, so it answers a different
        question -- and it can never be a silence error."""
        prov = {"car": {"rule_in_force": True, "labelsets": []}}
        rows, _, problems = sr.measure(
            {"car": {1}}, {"car": {1: True, 9: True}}, {}, {"car": {1}}, prov, designated={"car": {9}}
        )
        assert rows[0]["excluded_known_a"] == 1
        assert rows[0]["found_present"] == 1  # the known A did not inflate the count
        assert problems == []

    def test_an_anchored_control_is_not_an_alarm_either(self, sr):
        prov = {"car": {"rule_in_force": True, "labelsets": []}}
        rows, _, problems = sr.measure(
            {"car": {1}}, {"car": {1: True, 500: True}}, {}, {"car": {1}}, prov, controls={500}
        )
        assert rows[0]["excluded_control"] == 1
        assert problems == []

    def test_without_the_control_key_a_control_reads_as_drift(self, sr):
        """Stated so the caller knows what it is buying: the key is the only
        thing that separates a mixed-in control from an orphaned review."""
        prov = {"car": {"rule_in_force": True, "labelsets": []}}
        rows, _, problems = sr.measure({"car": {1}}, {"car": {1: True, 500: True}}, {}, {"car": {1}}, prov)
        assert rows[0]["excluded_unexplained"] == 1
        assert any("drifted" in p for p in problems)


class TestWilson:
    def test_zero_of_six_hundred_bounds_below_one_percent(self, sr):
        lo, hi = sr.wilson(0, 600)
        assert lo == 0.0
        assert hi == pytest.approx(0.0064, abs=5e-4)

    def test_an_empty_denominator_is_not_a_division(self, sr):
        assert sr.wilson(0, 0) == (0.0, 0.0)
