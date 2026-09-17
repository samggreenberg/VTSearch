"""pass_verdicts: a stored OWLv2 box is clipped to the image, never passed through past its edge (#3926)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PILE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "pile"


@pytest.fixture(scope="module")
def pv():
    if str(_PILE_DIR) not in sys.path:
        sys.path.insert(0, str(_PILE_DIR))
    import pass_verdicts

    return pass_verdicts


def test_a_box_past_the_frame_is_clipped_to_it(pv):
    # chair 12: the first application wrote this unclipped, and the build refused it
    assert pv._clipped([0.00041, 0.80741, 0.21859, 1.00209]) == [0.00041, 0.80741, 0.21859, 1.0]


def test_an_in_frame_box_is_returned_unchanged(pv):
    box = [0.1, 0.2, 0.3, 0.4]
    assert pv._clipped(box) is box


def test_a_box_wholly_outside_the_frame_is_refused(pv):
    with pytest.raises(SystemExit):
        pv._clipped([1.01, 0.2, 1.2, 0.4])


def test_no_box_stays_no_box(pv):
    assert pv._clipped(None) is None
