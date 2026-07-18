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


def test_finalize_slots_overrides_measured_slots_keeps_ballpark_for_rest(monkeypatch):
    monkeypatch.setattr("vtscore.config.resolve_device", lambda: "cuda")
    # Calibration measured only coverage + registry (the two big always-run
    # slots here); the opt-in projection/signpost slots weren't observed.
    row = (("coverage", 0.7), ("registry", 0.3))
    monkeypatch.setattr(_cm, "FINALIZE_SLOT_SHARES", {("cuda", "image"): row})
    slots = dict(_finalize_slots("image"))
    # Every canonical pipeline slot survives, in canonical order.
    assert [name for name, _ in _finalize_slots("image")] == [name for name, _ in FinalizeProgress._SLOTS]
    # Measured slots take the measured share; unobserved slots keep the ballpark.
    static = dict(FinalizeProgress._SLOTS)
    assert slots["coverage"] == 0.7
    assert slots["registry"] == 0.3
    assert slots["projection"] == static["projection"]
    assert slots["signpost_texts"] == static["signpost_texts"]
    # A cell with no measured row still falls back to the full static ballpark.
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
    # Every canonical slot is mapped; the measured coverage/registry shares
    # dominate while the unobserved slots keep their small ballpark slices.
    assert set(fin._ranges) == {name for name, _ in FinalizeProgress._SLOTS}
    cov_base, cov_span = fin._ranges["coverage"]
    reg_base, reg_span = fin._ranges["registry"]
    # coverage's measured 0.8 vs registry's 0.2 -> coverage gets ~4x the span,
    # both scaled down by the ballpark filler for the opt-in slots.
    assert cov_span > reg_span
    assert cov_span / reg_span == pytest.approx(4.0, rel=1e-6)
    # begin() resolves every pipeline slot without raising (no KeyError).
    for name, _ in FinalizeProgress._SLOTS:
        fin.begin(name)


def test_finalize_progress_static_fallback_keeps_all_pipeline_slots(monkeypatch):
    monkeypatch.setattr(_cm, "FINALIZE_SLOT_SHARES", {})
    fin = FinalizeProgress(_RecTracker(), "image")
    # All six pipeline sub-stages are mapped from the static ballpark.
    assert set(fin._ranges) == {name for name, _ in FinalizeProgress._SLOTS}


# --- the fit-script aggregation ------------------------------------------------


def _load_fit_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "profiling" / "fit_load_weights.py"
    spec = importlib.util.spec_from_file_location("_fit_load_weights_under_test", path)
    assert spec is not None and spec.loader is not None
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
