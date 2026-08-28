"""Tests for the declared opening and the guards that enforce it (#3278).

Autopilot has two real starts -- a text sort of a typed query, and three random
known-good examples -- and which one a cell takes is decided silently, by
whether its (dataset, category) has a query and whether its embedder has a text
tower.  Since #3269 the harness takes the first wherever it can, so a grid
holding SigLIP and DINOv3 arms opens two different ways along one axis with
nothing anywhere recording it.

``CALIB_REQUIRE_OPENING`` is the study saying which opening it means, and it is
enforced at three places that do not overlap:

* ``run_cells.check_declared_opening`` -- per cell, which is the only guard that
  runs when the array is submitted from a grid job with no preflight;
* ``preflight.sh`` check 14 -- per grid, before thousands of cells are queued
  (not exercised here; it is bash);
* ``_cells_io.assert_one_opening`` -- at analysis time, which is what catches a
  RESUME that left cells from before the fix beside cells from after it.

The calibration modules are loose scripts rather than package members, so they
are loaded by path.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

_CALIB = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "calibration"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cells_io():
    return _load("_calib_cells_io", _CALIB / "_cells_io.py")


def _config(monkeypatch, declared: str | None):
    """``experiment_config`` re-imported with ``CALIB_REQUIRE_OPENING`` set.

    The declaration is read at import time, like every other knob in that module,
    so a test that wants a different value has to load the module again.
    """
    if declared is None:
        monkeypatch.delenv("CALIB_REQUIRE_OPENING", raising=False)
    else:
        monkeypatch.setenv("CALIB_REQUIRE_OPENING", declared)
    return _load("_calib_experiment_config", _CALIB / "experiment_config.py")


def _run_cells(cfg):
    """``run_cells`` bound to *cfg*, with a stub ``common`` so import is inert."""
    stub: Any = types.ModuleType("common")
    stub.setup_env = lambda: None
    stub.log = lambda _msg: None
    stub.Path = Path
    stub.RESULTS = Path(".")
    saved = {k: sys.modules.get(k) for k in ("common", "experiment_config")}
    sys.modules["common"] = stub
    sys.modules["experiment_config"] = cfg
    try:
        return _load("_calib_run_cells", _CALIB / "run_cells.py")
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


# --- The declaration itself --------------------------------------------------


@pytest.mark.parametrize("declared", ["text", "known_good", "mixed"])
def test_every_documented_value_is_accepted(monkeypatch, declared):
    assert _config(monkeypatch, declared).REQUIRE_OPENING == declared


def test_unset_declares_nothing(monkeypatch):
    assert _config(monkeypatch, None).REQUIRE_OPENING == ""


def test_a_misspelt_declaration_fails_at_import(monkeypatch):
    """`known-good` and `text_seed` are the plausible near-misses.

    A declaration nobody validates is worse than none: the launcher reads as
    though the opening were pinned while every cell asserts nothing.
    """
    with pytest.raises(ValueError, match="CALIB_REQUIRE_OPENING"):
        _config(monkeypatch, "known-good")


# --- The per-cell guard ------------------------------------------------------


@pytest.mark.parametrize("declared,got", [("text", "text"), ("known_good", "known_good")])
def test_a_cell_that_got_what_was_declared_runs(monkeypatch, declared, got):
    cfg = _config(monkeypatch, declared)
    _run_cells(cfg).check_declared_opening("vg_scale", "siglip", "boat@small", got)


def test_a_text_study_refuses_a_known_good_cell(monkeypatch):
    """The #3278 case: a DINOv3 arm silently taking the app's other start."""
    cfg = _config(monkeypatch, "text")
    with pytest.raises(RuntimeError, match="opened on 'known_good'"):
        _run_cells(cfg).check_declared_opening("vg_scale", "dinov3_patch", "boat@small", "known_good")


def test_a_pinned_known_good_study_refuses_a_text_cell(monkeypatch):
    """The mirror: a pin is only a pin if it fails when it stops holding."""
    cfg = _config(monkeypatch, "known_good")
    with pytest.raises(RuntimeError, match="opened on 'text'"):
        _run_cells(cfg).check_declared_opening("vg_scale", "siglip", "boat@small", "text")


@pytest.mark.parametrize("declared", [None, "mixed"])
def test_mixed_and_unset_assert_nothing(monkeypatch, declared):
    """A re-runner mirroring a completed grid legitimately holds both openings."""
    run_cells = _run_cells(_config(monkeypatch, declared))
    run_cells.check_declared_opening("vg_scale", "siglip", "boat@small", "text")
    run_cells.check_declared_opening("vg_scale", "dinov3_patch", "boat@small", "known_good")


def test_the_message_names_the_arm_and_its_text_half(monkeypatch):
    """A paired arm's opening lives in a different space than its name implies."""
    cfg = _config(monkeypatch, "text")
    with pytest.raises(RuntimeError, match="text half=siglip"):
        _run_cells(cfg).check_declared_opening("vg_scale", "siglip+dinov3_patch", "boat@small", "known_good")


# --- The analysis-time guard -------------------------------------------------


def _rows(*specs: tuple[str, str, str, str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"dataset": d, "embedder": e, "category": c, "seed_mode": m, "seed_embedder": s} for d, e, c, m, s in specs]
    )


def test_one_opening_per_environment_pools(cells_io):
    frame = _rows(
        ("vg_scale", "siglip", "boat", "text", "siglip"),
        ("vg_scale", "siglip", "boat", "text", "siglip"),
        ("vg_scale", "siglip+dinov3_patch", "boat", "text", "siglip"),
    )
    cells_io.assert_one_opening(frame)


def test_two_openings_inside_one_environment_are_refused(cells_io):
    """The resume case: cells from before the seeding fix beside cells from after."""
    frame = _rows(
        ("vg_scale", "dinov3_patch", "boat", "known_good", ""),
        ("vg_scale", "dinov3_patch", "boat", "text", "siglip"),
    )
    with pytest.raises(ValueError, match="vg_scale x dinov3_patch x boat"):
        cells_io.assert_one_opening(frame)


def test_a_category_with_no_query_shifts_every_arm_together(cells_io):
    """Not the analyzer's business: a shared shift cancels in every contrast.

    `bus` has no typed query here, so it takes the known-good start on the SigLIP
    arm and the DINOv3 one alike.  That is a legitimate grid -- the study that
    wants it gone sets `CALIB_REQUIRE_SEED_QUERY=1` -- and refusing it would fire
    on most Visual Genome runs while catching nothing.
    """
    frame = _rows(
        ("visual_genome_m", "siglip", "boat", "text", "siglip"),
        ("visual_genome_m", "siglip", "bus", "known_good", ""),
        ("visual_genome_m", "siglip+dinov3_patch", "boat", "text", "siglip"),
        ("visual_genome_m", "siglip+dinov3_patch", "bus", "known_good", ""),
    )
    cells_io.assert_one_opening(frame)


def test_the_same_environment_opening_in_two_spaces_is_refused(cells_io):
    """`seed_mode` alone would call these identical; the space is the difference."""
    frame = _rows(
        ("vg_scale", "siglip+dinov3_patch", "boat", "text", "siglip"),
        ("vg_scale", "siglip+dinov3_patch", "boat", "text", "siglip2_l"),
    )
    with pytest.raises(ValueError, match="siglip2_l"):
        cells_io.assert_one_opening(frame)


def test_cells_written_before_the_columns_existed_read_as_unrecorded(cells_io):
    """A pre-#3269 cell has no value at all, and mixing it in is the same fault."""
    frame = _rows(("vg_scale", "dinov3_patch", "boat", "known_good", ""))
    frame = pd.concat(
        [frame, pd.DataFrame([{"dataset": "vg_scale", "embedder": "dinov3_patch", "category": "boat"}])],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="unrecorded"):
        cells_io.assert_one_opening(frame)


def test_a_frame_with_no_opening_columns_is_entirely_old(cells_io):
    """Every cell predates #3269, so they agree; there is nothing to compare."""
    frame = pd.DataFrame([{"dataset": "vg_scale", "embedder": "siglip", "category": "boat", "t": 1}])
    cells_io.assert_one_opening(frame)


def test_an_empty_frame_is_not_an_error(cells_io):
    cells_io.assert_one_opening(pd.DataFrame())
    cells_io.assert_one_opening(None)


def test_the_message_says_where_it_came_from(cells_io):
    frame = _rows(
        ("vg_scale", "dinov3_patch", "boat", "known_good", ""),
        ("vg_scale", "dinov3_patch", "boat", "text", "siglip"),
    )
    with pytest.raises(ValueError, match="analyze_cut.py"):
        cells_io.assert_one_opening(frame, "analyze_cut.py")
