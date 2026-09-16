"""Tests for the diagnostic preset and the startup settings line (#3853).

The preset exists because two sessions in that issue produced inconclusive
logs for configuration reasons: one ran at the shipped 1000 ms request bar so
sub-second votes were invisible, and one got lower bars only through
uncommitted local edits. The properties pinned here are the ones that make a
session's configuration hard to get wrong and impossible to misread
afterwards.
"""

from __future__ import annotations

import logging

import pytest

from vtsearch import diagnose

LOGGER = "vtsearch.diagnose"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Start from an unset environment; each test opts into what it needs."""
    monkeypatch.delenv(diagnose.DIAGNOSE_ENV, raising=False)
    for name in diagnose.PRESET:
        monkeypatch.delenv(name, raising=False)
    yield


class TestPreset:
    def test_off_by_default(self, monkeypatch):
        """A normal deployment must be untouched by this module."""
        assert diagnose.apply_diagnostic_preset() == {}
        assert not diagnose.diagnose_enabled()

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_truthy_spellings(self, monkeypatch, value):
        monkeypatch.setenv(diagnose.DIAGNOSE_ENV, value)
        assert diagnose.diagnose_enabled()

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
    def test_falsey_spellings(self, monkeypatch, value):
        monkeypatch.setenv(diagnose.DIAGNOSE_ENV, value)
        assert not diagnose.diagnose_enabled()

    def test_sets_the_whole_set_together(self, monkeypatch):
        """The point of the switch: one action, no bar left behind."""
        monkeypatch.setenv(diagnose.DIAGNOSE_ENV, "1")
        applied = diagnose.apply_diagnostic_preset()
        assert applied == diagnose.PRESET
        import os

        for name, value in diagnose.PRESET.items():
            assert os.environ[name] == value

    def test_explicit_env_wins_and_is_reported_as_not_ours(self, monkeypatch):
        """``setdefault`` semantics: overriding one bar is not a conflict, and
        the return value says which bars the preset is responsible for."""
        monkeypatch.setenv(diagnose.DIAGNOSE_ENV, "1")
        monkeypatch.setenv("VTSEARCH_SLOW_PHASE_MS", "50")
        applied = diagnose.apply_diagnostic_preset()
        assert "VTSEARCH_SLOW_PHASE_MS" not in applied
        assert applied["VTSEARCH_SLOW_REQUEST_MS"] == "400"

        import os

        assert os.environ["VTSEARCH_SLOW_PHASE_MS"] == "50"

    def test_preset_never_pins_the_gc_bar(self):
        """Pinning it would re-create the bug the coupling fixes: a collection
        under the GC bar is invisible while still inflating the phase it lands
        in. Unset, it tracks the phase bar instead."""
        assert "VTSEARCH_GC_WARN_MS" not in diagnose.PRESET


class TestEffectiveSettings:
    def test_reads_through_the_instruments_own_accessors(self, monkeypatch):
        """A summary that re-parsed the environment could report a bar the
        code does not use, which is worse than no summary."""
        monkeypatch.setenv("VTSEARCH_SLOW_PHASE_MS", "321")
        monkeypatch.setenv("VTSEARCH_SLOW_REQUEST_MS", "654")
        settings = diagnose.effective_settings()
        assert settings["slow_phase_ms"] == 321.0
        assert settings["slow_request_ms"] == 654.0

    def test_reports_the_coupled_gc_bar_not_the_raw_default(self, monkeypatch):
        """The whole reason to print the GC bar: it is derived, so a reader
        cannot work it out from the environment alone."""
        monkeypatch.delenv("VTSEARCH_GC_WARN_MS", raising=False)
        monkeypatch.setenv("VTSEARCH_SLOW_PHASE_MS", "150")
        assert diagnose.effective_settings()["gc_warn_ms"] == 75.0

    def test_covers_every_knob_that_decides_what_a_stall_leaves_behind(self):
        settings = diagnose.effective_settings()
        assert set(settings) == {
            "diagnose",
            "log_level",
            "slow_request_ms",
            "slow_phase_ms",
            "gc_warn_ms",
            "watchdog_ms",
            "gc_freeze",
            "detector_write",
            "log_file",
        }


class TestStartupLine:
    def test_logged_at_warning_so_a_stock_deployment_records_it(self, caplog):
        """The failure this prevents: a log whose 'no slow requests' cannot be
        told from 'the bar was a second'. At INFO it would be absent from
        exactly the deployments that need it."""
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            diagnose.log_effective_settings()
        records = [r for r in caplog.records if "diagnostics config" in r.getMessage()]
        assert len(records) == 1
        assert records[0].levelno == logging.WARNING

    def test_line_carries_every_threshold(self, caplog, monkeypatch):
        monkeypatch.setenv("VTSEARCH_SLOW_REQUEST_MS", "654")
        monkeypatch.setenv("VTSEARCH_SLOW_PHASE_MS", "321")
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            diagnose.log_effective_settings()
        msg = [r for r in caplog.records if "diagnostics config" in r.getMessage()][0].getMessage()
        assert "slow_request=654ms" in msg
        assert "slow_phase=321ms" in msg
        for field in ("diagnose=", "log_level=", "gc_warn=", "watchdog=", "gc_freeze=", "detector_write=", "log_file="):
            assert field in msg, f"{field} missing from {msg!r}"
