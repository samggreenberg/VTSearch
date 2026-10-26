"""imagekit.face.compat: the index-compat guard."""

from __future__ import annotations

import pytest

from imagekit.face.compat import COMPAT_KEY, CompatibilityError, check_compat, single_compat, stamp_compat


def test_stamp_then_check_passes_the_same_space():
    meta = stamp_compat({}, "oss-magface-ir100:1")
    assert meta[COMPAT_KEY] == "oss-magface-ir100:1"
    check_compat(meta, "oss-magface-ir100:1")
    stamp_compat(meta, "oss-magface-ir100:1")  # idempotent


def test_a_different_space_is_refused_on_add_or_query():
    meta = stamp_compat({"other": 1}, "oss-magface-ir100:1")
    with pytest.raises(CompatibilityError, match="vistaface:3"):
        check_compat(meta, "vistaface:3")
    with pytest.raises(CompatibilityError):
        stamp_compat(meta, "vistaface:3")


def test_an_unstamped_index_is_refused_rather_than_passed():
    with pytest.raises(CompatibilityError, match="no 'compat' stamp"):
        check_compat({}, "oss-magface-ir100:1")


def test_empty_stamp_is_refused():
    with pytest.raises(ValueError):
        stamp_compat({}, "")


def test_single_compat():
    assert single_compat(["a:1", "a:1"]) == "a:1"
    with pytest.raises(CompatibilityError, match="mix"):
        single_compat(["a:1", "b:2"])
    with pytest.raises(CompatibilityError):
        single_compat([])
