"""Tests for the blur-rule evaluator (``vtscore.datasets.blur_rules``, #5691).

A blur rule names an ingest metadata field and how to read it; an image starts
blurred when any rule matches its raw ``custom_metadata``. These pin every
operator both ways, the lenient field matching and value coercion, the
no-match cases, and which rules the normalizer keeps.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.datasets.blur_rules import (
    BLUR_RULE_OPS,
    blur_rule_problem,
    describe_blur_rule,
    field_key,
    matches_blur_rules,
    partition_blur_rules,
    rule_matches,
)


def _rule(op: str, value=None, field: str = "Score") -> dict:
    """A well-formed rule on *field*, through the normalizer as the settings accessor passes it."""
    entry = {"field": field, "op": op} if value is None else {"field": field, "op": op, "value": value}
    kept, rejected = partition_blur_rules([entry])
    assert not rejected, rejected
    return kept[0]


def _matches(op: str, value, media_value, field: str = "Score") -> bool:
    return rule_matches(_rule(op, value, field), {field: media_value})


class TestOperators:
    """Every operator, matching and not matching."""

    @pytest.mark.parametrize(
        ("op", "value", "hit", "miss"),
        [
            ("truthy", None, True, False),
            ("==", "R", "R", "PG"),
            ("!=", "safe", "nsfw", "safe"),
            (">", 0.8, 0.81, 0.8),
            (">=", 0.8, 0.8, 0.79),
            ("<", 0.2, 0.19, 0.2),
            ("<=", 0.2, 0.2, 0.21),
            ("in", ["R", "X"], "X", "PG"),
        ],
    )
    def test_each_operator_both_ways(self, op, value, hit, miss):
        assert _matches(op, value, hit) is True
        assert _matches(op, value, miss) is False

    def test_the_table_above_covers_every_operator(self):
        # A new operator must get a row in ``test_each_operator_both_ways``.
        assert BLUR_RULE_OPS == ("truthy", "==", "!=", ">", ">=", "<", "<=", "in")

    def test_equality_against_a_number(self):
        assert _matches("==", 3, 3.0)
        assert not _matches("==", 3, 4)
        assert _matches("!=", 3, 4)

    def test_equality_against_a_boolean(self):
        assert _matches("==", True, True)
        assert _matches("==", True, "yes")
        assert _matches("==", False, "0")
        assert not _matches("==", True, False)
        assert _matches("!=", True, False)

    def test_string_equality_ignores_case_and_whitespace(self):
        assert _matches("==", "Explicit", "  explicit ")
        assert _matches("in", ["r", "x"], "X")

    def test_a_string_rule_reads_a_numeric_or_boolean_value_its_way(self):
        assert _matches("==", "5", 5)
        assert _matches("==", "yes", True)
        assert _matches("in", ["1", "2"], 2)


class TestFieldMatching:
    @pytest.mark.parametrize("key", ["Porn", "porn", "PORN", "  Porn  ", "Porn:", "porn :", " PORN: "])
    def test_case_whitespace_and_a_trailing_colon(self, key):
        rule = _rule(">=", 0.8, field="Porn")
        assert rule_matches(rule, {key: 0.9})

    def test_the_rule_side_is_normalized_the_same_way(self):
        assert rule_matches(_rule("truthy", field=" GORE: "), {"gore": "yes"})

    def test_field_key(self):
        assert field_key("  Porn:: ") == "porn"
        assert field_key("Porn") == field_key("porn:")

    def test_a_different_field_does_not_match(self):
        assert not rule_matches(_rule("truthy", field="Gore"), {"Gory": True})

    def test_doubled_fields_match_when_either_does(self):
        rule = _rule(">=", 0.8, field="Porn")
        assert rule_matches(rule, {"Porn": 0.1, "porn:": 0.95})
        assert not rule_matches(rule, {"Porn": 0.1, "porn:": 0.2})


class TestCoercion:
    @pytest.mark.parametrize("value", ["0.93", " 0.93 ", "1", "9.3e-1"])
    def test_numeric_strings_against_a_numeric_rule(self, value):
        assert _matches(">=", 0.8, value)

    def test_a_numeric_string_rule_value_is_stored_as_a_number(self):
        assert _rule(">=", "0.8")["value"] == 0.8
        assert _rule(">", "2")["value"] == 2

    def test_numpy_values_read_as_numbers(self):
        assert _matches(">=", 0.8, np.float32(0.9))
        assert _matches("truthy", None, np.int64(1))
        assert not _matches("truthy", None, np.bool_(False))

    @pytest.mark.parametrize(
        "value", ["false", "FALSE", "no", " No ", "off", "none", "null", "0", "0.0", "", "   ", 0, 0.0, False, None]
    )
    def test_truthy_reads_these_as_false(self, value):
        assert not _matches("truthy", None, value)

    @pytest.mark.parametrize("value", ["true", "yes", "1", "explicit", 1, 0.3, True, ["nsfw"], {"k": 1}])
    def test_truthy_reads_these_as_true(self, value):
        assert _matches("truthy", None, value)

    def test_truthy_reads_empty_containers_and_nan_as_false(self):
        assert not _matches("truthy", None, [])
        assert not _matches("truthy", None, {})
        assert not _matches("truthy", None, float("nan"))
        assert not _matches("truthy", None, "nan")


class TestNoMatch:
    """A missing field, or a value the rule cannot compare, never matches."""

    @pytest.mark.parametrize(
        ("op", "value"),
        [("truthy", None), ("==", "R"), ("!=", "R"), (">", 0.5), (">=", 0.5), ("<", 0.5), ("<=", 0.5), ("in", ["R"])],
    )
    def test_a_missing_field(self, op, value):
        assert not rule_matches(_rule(op, value, field="Porn"), {"Gore": 1})

    @pytest.mark.parametrize("metadata", [None, {}, "Porn", ["Porn"]])
    def test_no_metadata(self, metadata):
        assert not rule_matches(_rule("truthy", field="Porn"), metadata)
        assert not matches_blur_rules([_rule("truthy", field="Porn")], metadata)

    @pytest.mark.parametrize("op", [">", ">=", "<", "<="])
    @pytest.mark.parametrize("media_value", ["high", "", None, True, [0.9], float("nan"), "inf"])
    def test_an_incomparable_value_against_a_threshold(self, op, media_value):
        assert not _matches(op, 0.5, media_value)

    @pytest.mark.parametrize("op", ["==", "!="])
    @pytest.mark.parametrize(
        ("value", "media_value"),
        [(3, "three"), (3, None), (3, [3]), (True, "maybe"), (True, 0.93), ("R", None), ("R", {"r": 1}), ("R", 5)],
    )
    def test_an_incomparable_value_against_equality(self, op, value, media_value):
        # ``!=`` included: "cannot compare" is not the same as "differs".
        assert not _matches(op, value, media_value)

    def test_an_incomparable_value_against_in(self):
        assert not _matches("in", [1, 2], "two")
        assert not _matches("in", ["R"], ["R"])


class TestAnyRule:
    RULES = [
        {"field": "Porn", "op": ">=", "value": 0.8},
        {"field": "Gore", "op": "truthy"},
        {"field": "Rating", "op": "in", "value": ["R", "X"]},
    ]

    def test_any_rule_matching_is_enough(self):
        rules, _ = partition_blur_rules(self.RULES)
        assert matches_blur_rules(rules, {"Porn": "0.1", "Gore": "no", "Rating": "x"})
        assert matches_blur_rules(rules, {"Porn:": 0.93})
        assert matches_blur_rules(rules, {"Gore": "yes"})

    def test_no_rule_matching_is_not_blurred(self):
        rules, _ = partition_blur_rules(self.RULES)
        assert not matches_blur_rules(rules, {"Porn": 0.1, "Gore": "false", "Rating": "PG"})

    def test_no_rules_blur_nothing(self):
        assert not matches_blur_rules([], {"Porn": 1.0})


class TestPartition:
    GOOD = [
        {"field": "Porn", "op": ">=", "value": 0.8},
        {"field": "Gore", "op": "truthy"},
        {"field": "Rating", "op": "in", "value": ["R", "X"]},
        {"field": "Flag", "op": "==", "value": True},
        {"field": "Kind", "op": "!=", "value": "safe"},
    ]

    @pytest.mark.parametrize(
        ("entry", "reason"),
        [
            ("Porn >= 0.8", "a rule must be an object"),
            (None, "a rule must be an object"),
            (["Porn", ">=", 0.8], "a rule must be an object"),
            ({"op": "truthy"}, "field must be"),
            ({"field": "", "op": "truthy"}, "field must be"),
            ({"field": " : ", "op": "truthy"}, "field must be"),
            ({"field": 3, "op": "truthy"}, "field must be"),
            ({"field": "Porn"}, "op must be one of"),
            ({"field": "Porn", "op": "~="}, "op must be one of"),
            ({"field": "Porn", "op": 3}, "op must be one of"),
            ({"field": "Porn", "op": ">="}, "needs a value"),
            ({"field": "Porn", "op": ">=", "value": "high"}, "needs a number"),
            ({"field": "Porn", "op": ">=", "value": True}, "needs a number"),
            ({"field": "Porn", "op": ">=", "value": None}, "needs a number"),
            ({"field": "Porn", "op": ">=", "value": float("nan")}, "needs a number"),
            ({"field": "Porn", "op": "==", "value": None}, "needs a string, number or boolean"),
            ({"field": "Porn", "op": "==", "value": ["R"]}, "needs a string, number or boolean"),
            ({"field": "Porn", "op": "in", "value": "R"}, "needs a non-empty list"),
            ({"field": "Porn", "op": "in", "value": []}, "needs a non-empty list"),
            ({"field": "Porn", "op": "in", "value": ["R", None]}, "needs a list of strings"),
            ({"field": "Porn", "op": "in", "value": [["R"]]}, "needs a list of strings"),
        ],
    )
    def test_a_malformed_rule_is_dropped_with_its_reason(self, entry, reason):
        kept, rejected = partition_blur_rules([self.GOOD[0], entry, self.GOOD[1]])
        assert kept == self.GOOD[:2]
        assert len(rejected) == 1
        assert rejected[0][0] == entry
        assert reason in rejected[0][1]
        assert blur_rule_problem(entry) == rejected[0][1]

    def test_good_rules_survive_in_order_unchanged(self):
        kept, rejected = partition_blur_rules(self.GOOD)
        assert kept == self.GOOD
        assert rejected == []

    def test_normalizes_op_field_and_extra_keys(self):
        kept, _ = partition_blur_rules(
            [
                {"field": "  Porn: ", "op": " TRUTHY ", "value": 1, "note": "x"},
                {"field": "R", "op": "IN", "value": ("a",)},
            ]
        )
        # A tuple is not JSON; ``in`` needs a list, so only the first survives.
        assert kept == [{"field": "Porn:", "op": "truthy"}]


class TestDescribe:
    def test_one_phrase_per_rule(self):
        rules, _ = partition_blur_rules(TestAnyRule.RULES)
        assert [describe_blur_rule(rule) for rule in rules] == ["Porn >= 0.8", "Gore is truthy", "Rating in R, X"]
