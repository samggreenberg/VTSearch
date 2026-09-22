"""The negative pass, restored from the committed CSV (#4012).

#4011 guarded three readers whose input was deleted (#4001). Two of them did not
need a guard: the judgements survive committed, so the capability survives. What
follows pins that claim the only way it can be pinned -- by reproducing the
numbers #3666 published, from the repository alone.

The acceptance test is `test_the_shipped_pool_headline_reproduces_exactly`. If it
fails, the restoration is not a restoration.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PILE = ROOT / "scripts" / "experiments" / "pile"
RECORD = PILE / "human_record"
STUDY = ROOT / "docs" / "experiments" / "2026-09-06-shipped-pool-3666"

#: What #3666 published, and what this restoration has to give back.
SHIPPED_FIVE = ("Clock", "Book", "Backpack", "Umbrella", "Stop Sign")
SHIPPED_HEADLINE = (7, 500, 1.40, 0.68, 2.86)
CANDIDATE_HEADLINE = (19, 910, 2.09, 1.34, 3.24)


def _npv():
    spec = importlib.util.spec_from_file_location("_npv", PILE / "negative_pass_verdicts.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # `@dataclass` resolves its own module out of `sys.modules`, so a
    # module_from_spec that is never registered there fails in dataclasses
    # itself rather than anywhere that reads like the cause.
    sys.modules[spec.name] = module
    sys.path.insert(0, str(PILE))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(PILE))
    return module


@pytest.fixture(scope="module")
def npv():
    return _npv()


@pytest.fixture(scope="module")
def rows(npv):
    return npv.load()


def test_the_committed_verdicts_are_where_the_reader_says(npv) -> None:
    assert npv.VERDICTS == STUDY / "verdicts.csv"
    assert npv.VERDICTS.is_file(), "the restoration rests on this file being committed"


def test_every_pass_judged_the_same_two_hundred_images(rows, npv) -> None:
    """The union rate is only meaningful because the passes share a frame."""
    by_pass: dict[str, set[int]] = {}
    for v in rows:
        by_pass.setdefault(v.pass_name, set()).add(v.image_id)
    assert len(rows) == 2400
    assert len(by_pass) == 12
    assert {len(ids) for ids in by_pass.values()} == {200}
    assert len(set(map(frozenset, by_pass.values()))) == 1, "the twelve passes no longer share one frame"


def test_a_stratum_belongs_to_an_image_not_to_a_pass(rows, npv) -> None:
    seen: dict[int, set[str]] = {}
    for v in rows:
        seen.setdefault(v.image_id, set()).add(v.stratum)
    assert all(len(s) == 1 for s in seen.values())
    counts = {st: sum(1 for s in npv.strata(rows).values() if s == st) for st in ("random", "boundary")}
    assert counts == {"random": 100, "boundary": 100}


def test_the_csv_strata_agree_with_the_committed_manifest(rows, npv) -> None:
    """`negative_pass_strata.py` dropped the manifest as a second input for one fact.

    The fact still gets checked -- here, once, rather than on every run.
    """
    manifest = {
        int(r["image_id"]): r["stratum"]
        for r in csv.DictReader((RECORD / "WORK3588__slates__Table_Objects__manifest.csv").open())
    }
    assert npv.strata(rows) == manifest


def test_the_shipped_pool_headline_reproduces_exactly(rows, npv) -> None:
    """#3666's 1.40% [0.68, 2.86], from the repository alone. The acceptance test."""
    k, n = npv.rate(rows, stratum="random", passes=SHIPPED_FIVE)
    lo, hi = npv.wilson(k, n)
    assert (k, n, round(100 * k / n, 2), round(lo, 2), round(hi, 2)) == SHIPPED_HEADLINE


@pytest.mark.parametrize(
    ("pass_name", "finds"),
    [("Clock", 3), ("Book", 2), ("Backpack", 2), ("Umbrella", 0), ("Stop Sign", 0)],
)
def test_each_published_per_class_rate_reproduces(rows, npv, pass_name: str, finds: int) -> None:
    """The report's per-class column, row by row: 3/100, 2/100, 2/100, 0/100, 0/100."""
    k, n = npv.rate(rows, stratum="random", passes=(pass_name,))
    assert (k, n) == (finds, 100)


def test_the_candidates_column_reproduces_from_the_record(rows) -> None:
    """The other half of the comparison: 19/910 = 2.09% [1.34, 3.24].

    `candidate_pool_error()` used to read this from the deleted study dir and
    return {} when it was missing, which dropped the column silently.
    """
    src = RECORD / "WORK3588__verdicts_20260904.json"
    assert src.is_file()
    random = [v for v in json.loads(src.read_text()) if v.get("stratum") == "random"]
    k = sum(1 for v in random if v["human"] == "present")
    assert (k, len(random)) == CANDIDATE_HEADLINE[:2]
    assert "WORK3588__verdicts_20260904.json" in (PILE / "shipped_pool_error.py").read_text()


def test_seeded_rows_are_the_reference_and_are_dropped_by_default(rows, npv) -> None:
    """Ten rows were written from COCO. Scoring them against COCO is circular."""
    seeded = [v for v in rows if v.seeded]
    assert len(seeded) == 10
    assert all(v.present for v in seeded), "a seeded row is a COCO positive by construction"
    default = npv.found(rows)
    with_seeded = npv.found(rows, include_seeded=True)
    assert sum(len(p) for p in default.values()) == 33
    assert sum(len(p) for p in with_seeded.values()) == 43


def test_the_union_rate_reconciles_with_what_was_published(rows, npv) -> None:
    """15% now; 14% as published. The difference is the two later passes and the seeds."""
    strata = npv.strata(rows)
    random_ids = [i for i, s in strata.items() if s == "random"]

    now = npv.found(rows)
    assert sum(1 for i in random_ids if i in now) == 15

    published = npv.found(rows, include_seeded=True, only=npv.NEGBANK_PASSES)
    assert sum(1 for i in random_ids if i in published) == 14, "#3666's published union no longer reproduces"

    assert set(npv.NEGBANK_PASSES) == {v.pass_name for v in rows} - {"Backpack", "Umbrella"}


def test_wilson_matches_the_intervals_the_report_prints(npv) -> None:
    for k, n, _pct, lo, hi in (SHIPPED_HEADLINE, CANDIDATE_HEADLINE):
        got = npv.wilson(k, n)
        assert (round(got[0], 2), round(got[1], 2)) == (lo, hi)


def test_the_retired_audit_pass_names_the_pass_that_survives() -> None:
    """Half 2: a guard that says where the measurement lives, not only what is gone."""
    text = (PILE / "make_audit_pass.py").read_text()
    assert "LABELSETS__verdicts_audit_20260825.json" in text
    assert "460 rows" in text and "260" in text and "200" in text
    assert "triage" in text and "definite" in text, "the lost qualifier has to be recorded"
    survivor = RECORD / "LABELSETS__verdicts_audit_20260825.json"
    assert survivor.is_file()
    verdicts = json.loads(survivor.read_text())
    strata = [v.get("stratum") for v in verdicts]
    assert len(verdicts) == 460
    assert (strata.count("flag"), strata.count("audit")) == (260, 200)
    assert all(v.get("triage") is None for v in verdicts), "if the kind came back, the guard's limit is stale"
