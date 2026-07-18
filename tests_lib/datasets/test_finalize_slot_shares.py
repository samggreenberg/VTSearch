"""Device/media-aware finalize sub-slot shares.

``FinalizeProgress`` spreads the finalize phase (step 4) across ordered
sub-stages. The shares were static ballpark guesses; this wires them to a
measured per-``(device, media)`` cost model (``FINALIZE_SLOT_SHARES``) with the
static ``_SLOTS`` as the fallback when no calibrated row exists. See issue #2624
and ``scripts/profiling/fit_load_weights.py`` (which fits the table).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from vtscore.datasets.stages import _load_cost_model as _cm
from vtscore.datasets.stages._common import FinalizeProgress, _finalize_slots


# --- the cost-model accessor ---------------------------------------------------


def test_finalize_slot_shares_returns_none_for_uncalibrated_cell(monkeypatch):
    monkeypatch.setattr(_cm, "FINALIZE_SLOT_SHARES", {})
    assert _cm.finalize_slot_shares("cpu", "audio") is None
    assert _cm.finalize_slot_shares("cuda", "image") is None


def test_finalize_slot_shares_returns_measured_row(monkeypatch):
    row = (("dedup", 0.4), ("coverage", 0.6))
    monkeypatch.setattr(_cm, "FINALIZE_SLOT_SHARES", {("cuda", "image"): row})
    assert _cm.finalize_slot_shares("cuda", "image") == row
    # Wrong media / device -> no row.
    assert _cm.finalize_slot_shares("cuda", "audio") is None
    assert _cm.finalize_slot_shares("cpu", "image") is None


def test_finalize_slot_shares_normalizes_device(monkeypatch):
    row = (("registry", 1.0),)
    monkeypatch.setattr(_cm, "FINALIZE_SLOT_SHARES", {("cuda", "image"): row})
    # "cuda:1" collapses to the "cuda" key.
    assert _cm.finalize_slot_shares("cuda:1", "image") == row


def test_shipped_finalize_slot_shares_table_is_well_formed():
    # Every checked-in row is an ordered, positively-weighted, non-empty tuple;
    # the empty pre-calibration table trivially passes.
    for key, row in _cm.FINALIZE_SLOT_SHARES.items():
        device, media = key
        assert device in ("cpu", "cuda")
        assert isinstance(media, str)
        assert len(row) >= 1
        assert all(w > 0 for _, w in row)


# --- _finalize_slots resolution ------------------------------------------------


def test_finalize_slots_falls_back_to_static_when_uncalibrated(monkeypatch):
    monkeypatch.setattr(_cm, "FINALIZE_SLOT_SHARES", {})
    assert _finalize_slots("audio") == FinalizeProgress._SLOTS


def test_finalize_slots_uses_measured_row(monkeypatch):
    monkeypatch.setattr("vtscore.config.resolve_device", lambda: "cuda")
    row = (("coverage", 0.7), ("registry", 0.3))
    monkeypatch.setattr(_cm, "FINALIZE_SLOT_SHARES", {("cuda", "image"): row})
    assert _finalize_slots("image") == row
    # A cell with no measured row still falls back to static.
    assert _finalize_slots("audio") == FinalizeProgress._SLOTS


def test_finalize_slots_never_raises_when_device_resolution_fails(monkeypatch):
    def _boom():
        raise RuntimeError("no torch")

    monkeypatch.setattr("vtscore.config.resolve_device", _boom)
    assert _finalize_slots("image") == FinalizeProgress._SLOTS


# --- FinalizeProgress consumes the resolved shares -----------------------------


class _RecTracker:
    """Minimal tracker stand-in: FinalizeProgress only calls update/check_cancelled."""

    def check_cancelled(self):
        pass

    def update(self, status, message="", current=0, total=0, **kw):
        pass


def test_finalize_progress_maps_slots_from_measured_shares(monkeypatch):
    monkeypatch.setattr("vtscore.config.resolve_device", lambda: "cuda")
    # coverage owns the first 80% of the finalize slice, registry the last 20%.
    row = (("coverage", 0.8), ("registry", 0.2))
    monkeypatch.setattr(_cm, "FINALIZE_SLOT_SHARES", {("cuda", "image"): row})

    fin = FinalizeProgress(_RecTracker(), "image")
    assert fin._ranges == {"coverage": (0.0, 0.8), "registry": (0.8, 0.2)}
    # A slot outside the measured row is unknown -> begin() KeyErrors, which is
    # the correct signal that the cost model and the pipeline slots disagree.
    with pytest.raises(KeyError):
        fin.begin("dedup")


def test_finalize_progress_static_fallback_keeps_all_pipeline_slots(monkeypatch):
    monkeypatch.setattr(_cm, "FINALIZE_SLOT_SHARES", {})
    fin = FinalizeProgress(_RecTracker(), "image")
    # All six pipeline sub-stages are mapped from the static ballpark.
    assert set(fin._ranges) == {name for name, _ in FinalizeProgress._SLOTS}


# --- the fit-script aggregation ------------------------------------------------


def _load_fit_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "profiling" / "fit_load_weights.py"
    spec = importlib.util.spec_from_file_location("_fit_load_weights_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fit_finalize_slots_medians_and_normalizes():
    fit = _load_fit_module()
    fin_slots = {
        ("cuda", "image"): {
            "coverage": [8.0, 8.0, 8.0],
            "registry": [2.0, 2.0, 2.0],
            "cleanup": [0.0],  # zero-only -> median 0 -> dropped from the mix
        }
    }
    shares, summary = fit._fit_finalize_slots(fin_slots)
    # Zeros drop out; the rest normalize; canonical order (cleanup < coverage <
    # registry) is preserved for the slots that survive.
    assert shares == {("cuda", "image"): (("coverage", 0.8), ("registry", 0.2))}
    assert len(summary) == 1


def test_fit_finalize_slots_orders_canonically_then_appends_unknown():
    fit = _load_fit_module()
    fin_slots = {
        ("cpu", "audio"): {
            "registry": [5.0],
            "dedup": [3.0],
            "mystery": [2.0],  # unrecognized slot -> appended after known ones
        }
    }
    shares, _ = fit._fit_finalize_slots(fin_slots)
    slot_names = [s for s, _ in shares[("cpu", "audio")]]
    assert slot_names == ["dedup", "registry", "mystery"]
    # Shares normalize to 1.
    assert sum(w for _, w in shares[("cpu", "audio")]) == pytest.approx(1.0, abs=1e-3)


def test_fit_finalize_slots_skips_empty_cell():
    fit = _load_fit_module()
    shares, summary = fit._fit_finalize_slots({("cpu", "audio"): {"dedup": [0.0]}})
    assert shares == {}
    assert summary == []
