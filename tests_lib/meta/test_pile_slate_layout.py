"""One slate layout rule, for the two callers that need it in different shapes.

A campaign writes `slates/<class>/manifest.csv`; #3729's record flattens the same
file to `<CAMPAIGN>__slates__<class>__manifest.csv`. `verdicts_to_corrections.py`
wants all of them and `make_contact_sheets.py` wants one, which is how a layout
rule gets spelled twice and drifts (#4006).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PILE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "pile"


@pytest.fixture(scope="module")
def slates():
    if str(_PILE_DIR) not in sys.path:
        sys.path.insert(0, str(_PILE_DIR))
    from pilebuild import slates as mod

    return mod


def _nested(root: Path, name: str) -> Path:
    d = root / name
    d.mkdir(parents=True)
    p = d / "manifest.csv"
    p.write_text("image_id,class,cell\n1,bus,bus@small\n")
    return p


def _flat(root: Path, campaign: str, name: str) -> Path:
    p = root / f"{campaign}__slates__{name}__manifest.csv"
    p.write_text("image_id,class,cell\n2,bus,bus@large\n")
    return p


def test_nested_layout(slates, tmp_path: Path):
    nested = _nested(tmp_path, "Backpack")
    assert slates.slate_manifests(tmp_path) == [nested]


def test_flat_record_layout(slates, tmp_path: Path):
    flat = _flat(tmp_path, "WORK3588", "Bench")
    assert slates.slate_manifests(tmp_path) == [flat]
    assert slates.slate_manifest(tmp_path, "Bench") == flat, "the campaign prefix is not part of the name"


def test_a_working_directory_beats_the_record(slates, tmp_path: Path):
    """A campaign mid-flight is the live answer; the committed copy is a snapshot."""
    nested = _nested(tmp_path, "Bench")
    _flat(tmp_path, "WORK3588", "Bench")
    assert slates.slate_manifest(tmp_path, "Bench") == nested


def test_two_campaigns_of_the_same_class_are_both_kept(slates, tmp_path: Path):
    """The bug this API shape exists to prevent: keying on the name drops one.

    `WORK3588__slates__Bench` and `WORK__slates_pos2__Bench` are different slates
    of the same class. On the real record, name-keying collapsed 64 manifests to
    41 -- and the corrections recipe reads all of them.
    """
    a = _flat(tmp_path, "WORK3588", "Bench")
    b = tmp_path / "WORK__slates_pos2__Bench__manifest.csv"
    b.write_text("image_id\n3\n")
    assert slates.slate_manifests(tmp_path) == sorted([a, b])
    assert slates.slate_manifest(tmp_path, "Bench") == a, "the primary __slates__ set wins"


def test_both_layouts_at_once(slates, tmp_path: Path):
    nested = _nested(tmp_path, "Backpack")
    flat = _flat(tmp_path, "WORK", "Bench")
    assert slates.slate_manifests(tmp_path) == sorted([nested, flat])


def test_flat_without_the_slates_infix(slates, tmp_path: Path):
    p = tmp_path / "WORK__Table_Objects__manifest.csv"
    p.write_text("image_id\n1\n")
    assert slates.slate_manifest(tmp_path, "Table_Objects") == p


def test_one_slate_by_name(slates, tmp_path: Path):
    _flat(tmp_path, "WORK3588", "Vehicles")
    assert slates.slate_manifest(tmp_path, "Vehicles") is not None
    assert slates.slate_manifest(tmp_path, "Nothing") is None


def test_the_committed_record_resolves(slates):
    """The real record: 64 slate manifests live there flattened."""
    found = slates.slate_manifests(_PILE_DIR / "human_record")
    assert len(found) >= 60, f"expected the record's slate manifests, got {len(found)}"
    assert all(p.name.endswith("manifest.csv") for p in found)
    assert slates.slate_manifest(_PILE_DIR / "human_record", "Bench") is not None
