"""Every class in *C* carries a measured composition note (#3983).

`SCALE_CLASS_RULES` records what a reviewer should count as Good.
`SCALE_CLASS_CONTENTS` records what COCO's annotators *did* count, measured
against LVIS's defined synsets by ``coco_class_purity.py``. The two are
different facts, and under a pure-COCO build the second is the one that governs,
because nobody reviews COCO's labels: a result on `cup` is a result on whatever
COCO put in `cup`, which is a drinking glass as often as a cup.

The note exists to be quoted beside a published number. A class added to *C*
without one is a class whose results cannot be read safely, and the gap is
invisible -- which is the same failure mode `SCALE_VG_NAMES_AUDITED` was added
for: "no note is written" and "there is nothing to note" are the same absence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PILE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "pile"
sys.path.insert(0, str(_PILE_DIR))

pc = pytest.importorskip("pile_config")


def test_every_class_in_c_has_a_composition_note():
    missing = sorted(set(pc.SCALE_CLASSES) - set(pc.SCALE_CLASS_CONTENTS))
    assert not missing, (
        f"no measured composition for {missing}; regenerate with `coco_class_purity.py --out` and add the row"
    )


def test_no_note_for_a_class_not_in_c():
    """A stale row outlives the class it describes and is quoted by accident."""
    extra = sorted(set(pc.SCALE_CLASS_CONTENTS) - set(pc.SCALE_CLASSES))
    assert not extra, f"composition note for a class not in C: {extra}"


def test_notes_name_a_share_so_they_cannot_become_assertions():
    """Each note is measured, so each carries at least one percentage.

    A row that reads `mostly cups` is an opinion wearing a measurement's clothes.
    """
    for cls, note in pc.SCALE_CLASS_CONTENTS.items():
        assert "%" in note, f"{cls}'s note quotes no share: {note!r}"


def test_the_classes_the_measurement_singled_out_say_so():
    """The rows a reader is most likely to misread must carry their warning.

    Three were singled out originally: `cup` was not predominantly cups, `truck`
    overlapped `car` asymmetrically while both were in *C*, and `stop sign` is a
    fifth generic street signs. **Two of the three were merged away in #4056 for
    exactly that reason**, so what has to be pinned now is that their successors
    say what they contain -- a merged class whose note reads like its name would
    reintroduce the defect the merge was for.
    """
    assert "street_sign 20%" in pc.SCALE_CLASS_CONTENTS["stop sign"]

    veh = pc.SCALE_CLASS_CONTENTS["enclosed road vehicle"]
    assert "car_(automobile)" in veh and "truck" in veh, "the union must name both halves"
    assert "minivan" in veh, "minivan is the type COCO splits, and the reason for the merge"

    vessel = pc.SCALE_CLASS_CONTENTS["single serving drinking vessel"]
    assert "glass_(drink_container)" in vessel, "the stemless glass is the type COCO cannot place"
    assert "wineglass" in vessel and "mug" in vessel, "stem and no stem both live here now"
    # `cup` proper is a fifth of its own former class; a note that hid that is
    # what made `cup` misleading in the first place.
    assert "cup 20%" in vessel


def test_contents_and_rules_cover_the_same_classes():
    """The two tables answer different questions about the same list.

    A class with a reviewer brief and no composition note (or the reverse) is
    half-documented in a way neither table reveals on its own.
    """
    assert set(pc.SCALE_CLASS_CONTENTS) == set(pc.SCALE_CLASS_RULES)
