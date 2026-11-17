"""Blur rules: which ingest metadata marks an image as worth blurring.

An outside ingest service rules some images sensitive (NSFW, gore, …) and says
so in per-item metadata fields, which the importers land in
``media["custom_metadata"]``. A *blur rule* names one such field and how to
read it; an image starts blurred when **any** rule matches its metadata. The
rules are an instance-wide setting (``blur_rules`` in the app's server
settings); this module is the pure half: what makes a rule well formed, and
whether a list of rules matches one media's metadata.

A rule is a ``{"field": ..., "op": ..., "value": ...}`` object:

* ``op`` is one of :data:`BLUR_RULE_OPS`. ``truthy`` takes no ``value``; the
  four orderings take a number; ``==`` and ``!=`` a string, number or boolean;
  ``in`` a non-empty list of those.
* ``field`` names a metadata key, matched after trimming surrounding
  whitespace and any trailing colon, case-insensitively (see
  :func:`field_key`), so an ingest service's ``"Porn:"`` is not silently
  missed by a rule naming ``"porn"``.
* Values are read leniently: an ordering reads the metadata value ``"0.93"``
  as a number, and ``truthy`` reads ``"false"``, ``"no"``, ``"off"``,
  ``"none"``, ``"null"``, a string spelling zero, and the empty string as
  false.
* A missing field, or a value the rule cannot compare (a word against a
  numeric threshold, a list against ``==``), does **not** match, under every
  operator, ``!=`` included.

Nothing here is stored: the rules are re-applied to the raw metadata whenever
a caller asks, so changing a rule re-decides every image at once.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Mapping
from typing import Any

__all__ = [
    "BLUR_RULE_OPS",
    "blur_rule_problem",
    "describe_blur_rule",
    "field_key",
    "matches_blur_rules",
    "partition_blur_rules",
    "rule_matches",
]

#: Every operator a blur rule may use, in the order the docs list them.
BLUR_RULE_OPS: tuple[str, ...] = ("truthy", "==", "!=", ">", ">=", "<", "<=", "in")

_ORDERING_OPS = frozenset({">", ">=", "<", "<="})
_EQUALITY_OPS = frozenset({"==", "!="})

#: Strings ``truthy`` reads as false (compared after trimming and casefolding),
#: besides the empty string and any string that reads as the number zero.
_FALSE_WORDS = frozenset({"false", "no", "off", "none", "null"})

#: Strings ``==`` / ``!=`` / ``in`` read as a boolean when the rule's value is
#: one (``"1"`` and ``"0"`` count too, read as numbers).
_TRUE_WORDS = frozenset({"true", "yes", "on"})
_BOOL_FALSE_WORDS = frozenset({"false", "no", "off"})


def field_key(name: str) -> str:
    """The form a metadata key and a rule's ``field`` are compared in.

    Surrounding whitespace and any trailing colons are trimmed and the rest is
    casefolded, so ``" Porn: "``, ``"porn"`` and ``"PORN"`` are one field.
    """
    return name.strip().rstrip(":").strip().casefold()


def _unwrap(value: Any) -> Any:
    """A NumPy scalar or 0-d array as the plain Python value it holds.

    Read off the ``ndim`` / ``item`` duck type, so this module needs no NumPy.
    Anything else is returned unchanged.
    """
    if getattr(value, "ndim", None) == 0 and callable(getattr(value, "item", None)):
        return value.item()
    return value


def _as_number(value: Any) -> float | None:
    """*value* as a finite number, or ``None`` when it is not one.

    A real number counts, and so does a string that spells one (``"0.93"``,
    ``" 7 "``). A boolean does not: ``True`` is not a score. NaN and the
    infinities are refused, since no threshold reads them sensibly.
    """
    value = _unwrap(value)
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Real):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def _as_bool(value: Any) -> bool | None:
    """*value* read strictly as a boolean, or ``None`` when it is not one.

    A boolean is itself; the numbers 0 and 1 are false and true; a string is
    true when it reads ``true`` / ``yes`` / ``on`` / ``1`` and false when it
    reads ``false`` / ``no`` / ``off`` / ``0`` (any case, any surrounding
    whitespace). Anything else, a score of 0.93 included, is not a boolean.
    """
    value = _unwrap(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        word = value.strip().casefold()
        if word in _TRUE_WORDS:
            return True
        if word in _BOOL_FALSE_WORDS:
            return False
    number = _as_number(value)
    if number == 1:
        return True
    if number == 0:
        return False
    return None


def _truthy(value: Any) -> bool:
    """Whether a ``truthy`` rule reads *value* as set.

    ``None`` is false; a boolean is itself; a number is true when it is
    non-zero (NaN is false); a string is false when it is empty, one of
    :data:`_FALSE_WORDS`, or spells zero, and true otherwise; a list, tuple,
    set or dict is true when it is non-empty. Anything else cannot be read
    and is false.
    """
    value = _unwrap(value)
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Real):
        number = float(value)
        return not math.isnan(number) and number != 0
    if isinstance(value, str):
        word = value.strip().casefold()
        if not word or word in _FALSE_WORDS:
            return False
        try:
            number = float(word)
        except ValueError:
            return True
        return not math.isnan(number) and number != 0
    if isinstance(value, (list, tuple, set, frozenset, dict)):
        return len(value) > 0
    return False


def _equals(media_value: Any, rule_value: Any) -> bool | None:
    """Whether *media_value* equals one of a rule's scalars; ``None`` if they cannot be compared.

    The rule's scalar decides how the metadata value is read: a boolean
    strictly as a boolean (:func:`_as_bool`), a number as a number
    (:func:`_as_number`), and a string as a string, trimmed and casefolded. A
    string rule against a numeric or boolean metadata value is read the other
    way round, so ``"5"`` equals the number 5 and ``"yes"`` equals ``True``.
    """
    media_value = _unwrap(media_value)
    if isinstance(rule_value, bool):
        as_bool = _as_bool(media_value)
        return None if as_bool is None else as_bool == rule_value
    if isinstance(rule_value, numbers.Real):
        number = _as_number(media_value)
        return None if number is None else number == float(rule_value)
    if isinstance(rule_value, str):
        if isinstance(media_value, str):
            return media_value.strip().casefold() == rule_value.strip().casefold()
        if isinstance(media_value, bool):
            as_bool = _as_bool(rule_value)
            return None if as_bool is None else as_bool == media_value
        media_number = _as_number(media_value)
        rule_number = _as_number(rule_value)
        if media_number is None or rule_number is None:
            return None
        return media_number == rule_number
    return None


def _compare(op: str, media_value: Any, threshold: float) -> bool:
    """An ordering rule against one metadata value; an unreadable value does not match."""
    number = _as_number(media_value)
    if number is None:
        return False
    if op == ">":
        return number > threshold
    if op == ">=":
        return number >= threshold
    if op == "<":
        return number < threshold
    return number <= threshold


def _value_matches(rule: Mapping[str, Any], media_value: Any) -> bool:
    """Whether a well-formed *rule* matches one metadata value."""
    op = rule["op"]
    if op == "truthy":
        return _truthy(media_value)
    if op in _ORDERING_OPS:
        return _compare(op, media_value, float(rule["value"]))
    if op == "==":
        return _equals(media_value, rule["value"]) is True
    if op == "!=":
        return _equals(media_value, rule["value"]) is False
    # ``in``
    return any(_equals(media_value, choice) is True for choice in rule["value"])


def _values_by_field(metadata: Mapping[Any, Any]) -> dict[str, list[Any]]:
    """The metadata's values grouped by :func:`field_key`.

    Two keys that differ only in case or a trailing colon land in one group,
    and a rule naming that field matches when any of them does: the rules
    shield users, so a doubled field errs towards blurring.
    """
    grouped: dict[str, list[Any]] = {}
    for key, value in metadata.items():
        if isinstance(key, str):
            grouped.setdefault(field_key(key), []).append(value)
    return grouped


def rule_matches(rule: Mapping[str, Any], custom_metadata: Mapping[Any, Any] | None) -> bool:
    """Whether one well-formed *rule* matches a media's raw ``custom_metadata``.

    *rule* must have come through :func:`partition_blur_rules` (the app's
    settings accessor already has). A missing field does not match.
    """
    if not isinstance(custom_metadata, Mapping):
        return False
    values = _values_by_field(custom_metadata).get(field_key(rule["field"]), [])
    return any(_value_matches(rule, value) for value in values)


def matches_blur_rules(rules: Iterable[Mapping[str, Any]], custom_metadata: Mapping[Any, Any] | None) -> bool:
    """Whether any of *rules* matches a media's raw ``custom_metadata``.

    The question an image's starting blur state turns on: it starts blurred
    exactly when this is true. *rules* must be well formed (see
    :func:`partition_blur_rules`); no rules, or no metadata, is never a match.
    Pass the media's own ``custom_metadata`` dict, not the display-formatted
    metadata the API returns.
    """
    if not isinstance(custom_metadata, Mapping) or not custom_metadata:
        return False
    values = _values_by_field(custom_metadata)
    return any(_value_matches(rule, value) for rule in rules for value in values.get(field_key(rule["field"]), []))


def _scalar_ok(value: Any) -> bool:
    """A value ``==`` / ``!=`` / ``in`` can compare: a string, boolean or finite number."""
    if isinstance(value, (str, bool)):
        return True
    return isinstance(value, numbers.Real) and math.isfinite(float(value))


def blur_rule_problem(entry: Any) -> str | None:
    """What is wrong with one raw ``blur_rules`` entry, or ``None`` when it is usable.

    The reason is a short phrase for the startup log (``"op must be one of
    …"``), naming the first problem found.
    """
    if not isinstance(entry, dict):
        return "a rule must be an object with field and op"
    field = entry.get("field")
    if not isinstance(field, str) or not field_key(field):
        return "field must be a non-empty string"
    op = entry.get("op")
    op = op.strip().lower() if isinstance(op, str) else op
    if op not in BLUR_RULE_OPS:
        return f"op must be one of {', '.join(BLUR_RULE_OPS)}"
    if op == "truthy":
        return None
    if "value" not in entry:
        return f"op {op} needs a value"
    value = entry["value"]
    if op in _ORDERING_OPS:
        return None if _as_number(value) is not None else f"op {op} needs a number"
    if op in _EQUALITY_OPS:
        return None if _scalar_ok(value) else f"op {op} needs a string, number or boolean"
    if not isinstance(value, list) or not value:
        return "op in needs a non-empty list"
    if not all(_scalar_ok(choice) for choice in value):
        return "op in needs a list of strings, numbers or booleans"
    return None


def _normalized(entry: dict[str, Any]) -> dict[str, Any]:
    """A usable *entry* in its stored form.

    ``field`` loses its surrounding whitespace (its spelling is otherwise the
    operator's, for display), ``op`` is lower-cased and trimmed, a ``truthy``
    rule loses any ``value``, an ordering's numeric string becomes a number,
    and any other key is dropped.
    """
    op = entry["op"].strip().lower()
    rule: dict[str, Any] = {"field": entry["field"].strip(), "op": op}
    if op == "truthy":
        return rule
    value = entry["value"]
    if op in _ORDERING_OPS and isinstance(value, str):
        number = _as_number(value)
        value = int(number) if number is not None and number.is_integer() else number
    elif op == "in":
        value = list(value)
    rule["value"] = value
    return rule


def partition_blur_rules(entries: list[Any]) -> tuple[list[dict[str, Any]], list[tuple[Any, str]]]:
    """Split a raw ``blur_rules`` list into ``(usable rules, rejected entries)``.

    Usable rules come back normalized (see :func:`_normalized`), in the order
    the operator wrote them. Each rejected entry comes back as ``(entry,
    reason)``, the reason from :func:`blur_rule_problem`, so a startup log can
    say why it was dropped. A bad rule never takes the good ones beside it
    down with it.
    """
    kept: list[dict[str, Any]] = []
    rejected: list[tuple[Any, str]] = []
    for entry in entries:
        problem = blur_rule_problem(entry)
        if problem is None:
            kept.append(_normalized(entry))
        else:
            rejected.append((entry, problem))
    return kept, rejected


def describe_blur_rule(rule: Mapping[str, Any]) -> str:
    """One well-formed rule as a short phrase, such as ``Porn >= 0.8`` or ``Gore is truthy``."""
    op = rule["op"]
    if op == "truthy":
        return f"{rule['field']} is truthy"
    if op == "in":
        return f"{rule['field']} in {', '.join(str(choice) for choice in rule['value'])}"
    return f"{rule['field']} {op} {rule['value']}"
