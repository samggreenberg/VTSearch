"""`vg_scale` must score a class against images holding a *different* class (#3667).

The rule these pin down is asymmetric on purpose: an image holding `bus` at one
size is still excluded from `bus`'s other bands, because scoring it there would
penalise a detector for finding a real bus. It is *not* excluded from `book`,
because it genuinely has no book.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "experiments" / "pile"))

import pile_config as pc  # noqa: E402
from pilebuild.loaders.vg_scale import _evaluable  # noqa: E402

CELLS: list[str] = [pc.scale_cell(c, b) for c in pc.SCALE_CLASSES for b in pc.BOX_BANDS]
#: pyright reads `[[0, 0, 1, 1]]` as list[list[int]], which is not
#: list[list[float]] because list is invariant. Name the value once, typed.
BOX: list[list[float]] = [[0.0, 0.0, 1.0, 1.0]]
BUS_MED = pc.scale_cell("bus", "medium")
BUS_SMALL = pc.scale_cell("bus", "small")
BOOK_MED = pc.scale_cell("book", "medium")


def test_shared_negative_is_evaluable_everywhere():
    assert _evaluable(1, [], CELLS, {1}, {}, set()) == CELLS


def test_image_that_is_neither_is_evaluable_nowhere():
    assert _evaluable(1, [], CELLS, set(), {}, set()) == []


def test_positive_still_excluded_from_its_own_other_bands():
    """The exclusion #3156 exists for, which #3667 must not undo."""
    out = _evaluable(1, [BUS_MED], CELLS, set(), {1: {"bus": BOX}}, {1})
    assert BUS_MED in out
    assert BUS_SMALL not in out, "a large-bus image must not be a small-bus negative"


def test_positive_becomes_a_negative_for_classes_it_does_not_hold():
    out = _evaluable(1, [BUS_MED], CELLS, set(), {1: {"bus": BOX}}, {1})
    for band in pc.BOX_BANDS:
        assert pc.scale_cell("book", band) in out
    assert len(out) > 1, "#3667: a bus image is a perfectly good book negative"


def test_a_second_held_class_is_also_excluded():
    labels: dict[int, dict[str, list[list[float]]]] = {1: {"bus": BOX, "book": BOX}}
    out = _evaluable(1, [BUS_MED], CELLS, set(), labels, {1})
    assert BOOK_MED not in out, "it holds a book; it cannot be a book negative"
    assert pc.scale_cell("dog", "small") in out


def test_off_coco_images_are_left_alone():
    """VG's silence is not a fact. Absence is only free on the exhaustive half."""
    out = _evaluable(1, [BUS_MED], CELLS, set(), {1: {"bus": BOX}}, set())
    assert out == [BUS_MED]


def test_the_knob_turns_it_off(monkeypatch):
    monkeypatch.setattr(pc, "SCALE_CROSS_CLASS_NEGATIVES", False)
    out = _evaluable(1, [BUS_MED], CELLS, set(), {1: {"bus": BOX}}, {1})
    assert out == [BUS_MED]


@pytest.mark.parametrize("held", [(), ("bus",), ("bus", "book"), tuple(pc.SCALE_CLASSES)])
def test_never_evaluable_in_a_cell_of_a_class_it_holds(held):
    labels: dict[int, dict[str, list[list[float]]]] = {1: {c: BOX for c in held}}
    out = set(_evaluable(1, [BUS_MED], CELLS, set(), labels, {1}))
    for c in held:
        if c == "bus":
            continue
        assert not (out & {pc.scale_cell(c, b) for b in pc.BOX_BANDS})
