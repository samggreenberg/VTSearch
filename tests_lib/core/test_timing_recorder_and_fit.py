"""Tests for the timing recorder and the fit that turns its rows into a profile.

The recorder and the fitter are the two halves of the tuning loop, so they are
tested together and end-to-end: rows recorded from a simulated task must fit into
a profile that the reader then loads and paces with. If that round trip works,
an admin's sweep works.
"""

import json

import math

import pytest

from vtscore import timing
from vtscore.concurrency.progress import PROGRESS_COMMON_EXTRAS, ProgressTracker
from vtscore.timing import profile as timing_profile
from vtscore.timing import recorder as timing_recorder
from vtscore.timing.fit import (
    affine_fit,
    coverage_report,
    device_key,
    fit_profile,
    fit_step,
    load_rows,
    normalize_row,
)
from vtscore.timing.profile import StepCoeffs
from vtscore.timing.recorder import RECORD_ENV_VAR, record_task, recording_enabled


def _row(raw: dict) -> dict:
    """``normalize_row`` narrowed to non-``None`` (its reject return)."""
    row = normalize_row(raw)
    assert row is not None, f"expected {raw} to normalize"
    return row


def _fit(samples: list[dict], *, byte_scaled: bool) -> StepCoeffs:
    """``fit_step`` narrowed to non-``None`` (its no-samples return)."""
    coeffs = fit_step(samples, byte_scaled)
    assert coeffs is not None
    return coeffs


def _weights(task: str, **kwargs) -> list[float]:
    """``step_weights`` narrowed to non-``None`` (its no-coverage return)."""
    weights = timing.step_weights(task, **kwargs)
    assert weights is not None, f"expected weights for {task}"
    return weights


def _tracker() -> ProgressTracker:
    return ProgressTracker(extra_fields=dict(PROGRESS_COMMON_EXTRAS))


@pytest.fixture
def sink(tmp_path, monkeypatch):
    """Arm the recorder at a temp JSONL and yield a reader for what it wrote.

    Reading goes through :func:`load_rows`, so every test that inspects recorded
    output also exercises the path the tuning script takes.
    """
    path = tmp_path / "timings.jsonl"
    monkeypatch.setenv(RECORD_ENV_VAR, str(path))

    def read():
        return load_rows([str(path)]) if path.exists() else []

    read.path = str(path)
    return read


class TestRecorderArming:
    def test_disarmed_by_default(self, monkeypatch):
        monkeypatch.delenv(RECORD_ENV_VAR, raising=False)
        assert not recording_enabled()
        # The no-op stand-in must accept the whole surface without a tracker
        # ever being subscribed, so call sites need no branching.
        tracker = _tracker()
        with record_task(tracker, "text_sort") as rec:
            rec.set_scale(n=10)
        rec.finish(ok=True)
        assert tracker._subscribers == []

    def test_armed_recorder_unsubscribes_on_finish(self, sink, monkeypatch):
        tracker = _tracker()
        rec = record_task(tracker, "text_sort")
        rec.start()
        assert len(tracker._subscribers) == 1
        rec.finish()
        assert tracker._subscribers == []


class TestRecordedRows:
    def _run_text_sort(self, tracker, rec, *, skip_model_load=False):
        if not skip_model_load:
            tracker.update("sorting", "loading", 0, 0, step=1, total_steps=3)
        tracker.update("sorting", "embedding", 0, 0, step=2, total_steps=3)
        tracker.update("sorting", "scoring", 0, 0, step=3, total_steps=3)
        rec.set_scale(n=1234)

    def test_one_row_per_declared_step(self, sink):
        tracker = _tracker()
        with record_task(tracker, "text_sort", media_type="image", embedder="siglip") as rec:
            self._run_text_sort(tracker, rec)
        rows = sink()
        assert [r["step"] for r in rows] == ["load_model", "embed_query", "score"]
        assert all(r["task"] == "text_sort" and r["n"] == 1234 for r in rows)
        assert all(r["ok"] and r["complete"] for r in rows)

    def test_a_skipped_step_is_recorded_as_zero_not_dropped(self, sink):
        # A warm text sort never enters load_model because the encoder is
        # already resident. That is a real measurement of zero, and dropping it
        # would leave the fit permanently over-budgeting the phase.
        tracker = _tracker()
        with record_task(tracker, "text_sort") as rec:
            self._run_text_sort(tracker, rec, skip_model_load=True)
        rows = {r["step"]: r for r in sink()}
        assert set(rows) == {"load_model", "embed_query", "score"}
        assert rows["load_model"]["seconds"] == 0.0
        assert rows["load_model"]["complete"] is True

    def test_a_run_that_stops_early_is_marked_incomplete(self, sink):
        tracker = _tracker()
        rec = record_task(tracker, "text_sort")
        rec.start()
        tracker.update("sorting", "loading", 0, 0, step=1, total_steps=3)
        rec.finish(ok=False)
        rows = sink()
        assert rows and all(not r["complete"] and not r["ok"] for r in rows)
        assert [r["step"] for r in rows] == ["load_model"]

    def test_exception_inside_the_context_marks_the_run_bad(self, sink):
        tracker = _tracker()
        with pytest.raises(RuntimeError):
            with record_task(tracker, "find") as rec:
                tracker.update("running", "prepare", 0, 0, step=1, total_steps=3)
                rec.set_scale(n=5)
                raise RuntimeError("boom")
        assert all(not r["ok"] for r in sink())

    def test_auto_finish_closes_on_a_terminal_status(self, sink):
        # Singleton trackers (sort_progress, find_progress) end by parking at
        # "idle" on every exit path, which is a far more reliable end-of-task
        # signal than a finally on a route handler full of aborts.
        tracker = _tracker()
        rec = record_task(tracker, "find", auto_finish=True)
        rec.start()
        for step in (1, 2, 3):
            tracker.update("running", "x", 0, 0, step=step, total_steps=3)
        rec.set_scale(n=99)
        tracker.update("idle", "", 0, 0, step=None, total_steps=None)
        assert tracker._subscribers == []  # closed itself
        rows = sink()
        assert [r["step"] for r in rows] == ["prepare", "load", "score"]
        assert all(r["ok"] and r["complete"] for r in rows)

    def test_auto_finish_marks_an_errored_end_bad(self, sink):
        tracker = _tracker()
        rec = record_task(tracker, "find", auto_finish=True)
        rec.start()
        tracker.update("running", "x", 0, 0, step=1, total_steps=3)
        tracker.update("idle", "", 0, 0, step=None, total_steps=None, error="Cancelled")
        assert all(not r["ok"] for r in sink())

    def test_status_phases_disambiguate_a_shared_step(self, sink):
        # dataset_load's step 1 is both the transfer and the unpack; only the
        # status tells them apart.
        tracker = _tracker()
        with record_task(tracker, "dataset_load", status_phases={"extracting": "extract"}) as rec:
            tracker.update("downloading", "", 0, 0, step=1, total_steps=4)
            tracker.update("extracting", "", 0, 0, step=1, total_steps=4)
            tracker.update("loading", "", 0, 0, step=2, total_steps=4)
            tracker.update("embedding", "", 0, 0, step=3, total_steps=4)
            tracker.update("finalizing", "", 0, 0, step=4, total_steps=4)
            rec.set_scale(n=500, size_mb=120.0)
        rows = sink()
        assert [r["step"] for r in rows] == ["download", "extract", "load", "embed", "finalize"]
        assert all(r["size_mb"] == 120.0 for r in rows)

    def test_a_run_with_no_steps_writes_nothing(self, sink):
        tracker = _tracker()
        with record_task(tracker, "text_sort"):
            tracker.update("idle", "", 0, 0)
        assert sink() == []


class TestNormalizeRow:
    def test_generic_row_round_trips(self):
        row = _row(
            {
                "task": "find",
                "device": "cuda:0",
                "cuml": True,
                "media_type": "image",
                "embedder": "siglip",
                "n": 10,
                "size_mb": 0,
                "step": "score",
                "seconds": 1.5,
                "ok": True,
                "complete": True,
            }
        )
        assert row["device"] == "cuda+cuml"
        assert row["step"] == "score"
        assert row["seconds"] == 1.5

    def test_legacy_load_profiler_row_is_understood(self):
        # An existing dataset-load calibration sweep folds into a new profile
        # rather than having to be re-measured.
        row = _row(
            {
                "device": "cuda",
                "media_type": "image",
                "embedder": "siglip",
                "n": 500,
                "download_size_mb": 120.0,
                "phase": "model_load",
                "seconds": 0.5,
                "cuml": False,
            }
        )
        assert row["task"] == "dataset_load"
        assert row["step"] == "load"  # phase name mapped onto the registry's
        assert row["device"] == "cuda"
        assert row["size_mb"] == 120.0

    def test_legacy_sub_slot_row_becomes_a_slot_sample(self):
        row = _row(
            {
                "device": "cpu",
                "media_type": "image",
                "embedder": "",
                "n": 5,
                "phase": "finalize:coverage",
                "seconds": 2.0,
            }
        )
        assert (row["step"], row["slot"]) == ("finalize", "coverage")

    @pytest.mark.parametrize(
        "raw",
        [
            {"task": "text_sort", "step": "score", "seconds": 1.0, "ok": False, "complete": True},
            {"task": "text_sort", "step": "score", "seconds": 1.0, "ok": True, "complete": False},
            {"task": "text_sort", "step": "not_a_step", "seconds": 1.0},
            {"task": "who_knows", "step": "score", "seconds": 1.0},
            {"phase": "embed", "seconds": 1.0, "n": 0},  # legacy failed load
            {"task": "text_sort", "step": "score"},  # no duration
            {"seconds": 1.0},  # neither shape
        ],
    )
    def test_unusable_rows_are_rejected(self, raw):
        assert normalize_row(raw) is None

    def test_device_key_splits_cuda_on_cuml(self):
        assert device_key("cuda:0", True) == "cuda+cuml"
        assert device_key("cuda", False) == "cuda"
        assert device_key("cpu", True) == "cpu"


class TestFitting:
    def test_affine_fit_recovers_a_known_line(self):
        xs = [0.0, 10.0, 20.0, 30.0]
        ys = [2.0 + 0.5 * x for x in xs]
        a, b, r2 = affine_fit(xs, ys)
        assert a == pytest.approx(2.0)
        assert b == pytest.approx(0.5)
        assert r2 == pytest.approx(1.0)

    def test_no_spread_in_n_yields_no_slope(self):
        a, b, _ = affine_fit([100.0, 100.0, 100.0], [3.0, 4.0, 5.0])
        assert b == 0.0
        assert a == pytest.approx(4.0)

    def test_a_negative_slope_collapses_to_the_median(self):
        # Noise beating signal on a short step must not ship a coefficient that
        # extrapolates to "this gets faster the bigger it gets".
        samples = [
            {"n": 100.0, "size_mb": 0.0, "seconds": 5.0},
            {"n": 200.0, "size_mb": 0.0, "seconds": 3.0},
            {"n": 300.0, "size_mb": 0.0, "seconds": 1.0},
        ]
        coeffs = _fit(samples, byte_scaled=False)
        assert coeffs.b == 0.0
        assert coeffs.a == pytest.approx(3.0)

    def test_the_fits_r2_is_kept_not_discarded(self):
        # `affine_fit` has always computed an r2 and `fit_step` threw it away at
        # the call site, which made this the one place in the tree that measured
        # a fit's quality and discarded it (#3329). A clean line must arrive
        # with r2 ~ 1, and a noisy one materially below it, or the number is
        # being carried without meaning anything.
        clean = _fit(
            [{"n": float(x), "size_mb": 0.0, "seconds": 1.0 + 0.01 * x} for x in (100, 200, 300, 400)],
            byte_scaled=False,
        )
        assert clean.b > 0
        assert clean.r2 == pytest.approx(1.0)

        noisy = _fit(
            [
                {"n": 100.0, "size_mb": 0.0, "seconds": 2.0},
                {"n": 200.0, "size_mb": 0.0, "seconds": 9.0},
                {"n": 300.0, "size_mb": 0.0, "seconds": 4.0},
                {"n": 400.0, "size_mb": 0.0, "seconds": 12.0},
            ],
            byte_scaled=False,
        )
        assert noisy.b > 0
        assert noisy.r2 < 0.9

    def test_a_step_that_was_not_fitted_as_a_line_has_no_r2(self):
        # NaN here means "not fitted this way", which is a different statement
        # from a bad fit: the median fallback and the byte-scaled path never
        # drew a line, so attaching a goodness score to them would be a lie.
        median_fallback = _fit(
            [
                {"n": 100.0, "size_mb": 0.0, "seconds": 5.0},
                {"n": 200.0, "size_mb": 0.0, "seconds": 3.0},
                {"n": 300.0, "size_mb": 0.0, "seconds": 1.0},
            ],
            byte_scaled=False,
        )
        assert math.isnan(median_fallback.r2)
        byte_scaled = _fit([{"n": 500.0, "size_mb": 100.0, "seconds": 10.0}], byte_scaled=True)
        assert math.isnan(byte_scaled.r2)
        assert "r2" not in median_fallback.to_json()

    def test_r2_survives_a_json_round_trip(self):
        coeffs = StepCoeffs(a=1.0, b=2.0, r2=0.9876)
        assert coeffs.to_json()["r2"] == pytest.approx(0.9876)
        # `from_json` is Optional-returning (it parses untrusted profile JSON),
        # so narrow before reading the field rather than chaining through it.
        parsed = StepCoeffs.from_json(coeffs.to_json())
        assert parsed is not None
        assert parsed.r2 == pytest.approx(0.9876)
        bare = StepCoeffs.from_json({"a": 1.0})
        assert bare is not None
        assert math.isnan(bare.r2)

    def test_byte_scaled_step_fits_a_per_mb_rate(self):
        samples = [
            {"n": 500.0, "size_mb": 100.0, "seconds": 10.0},
            {"n": 5000.0, "size_mb": 400.0, "seconds": 40.0},
        ]
        coeffs = _fit(samples, byte_scaled=True)
        assert coeffs.per_mb == pytest.approx(0.1)
        assert (coeffs.a, coeffs.b) == (0.0, 0.0)

    def test_byte_scaled_step_with_no_archive_is_a_real_zero(self):
        # Every measured load hit a warm cache: this deployment genuinely pays
        # nothing to acquire, which is a measurement, not a missing one.
        coeffs = _fit([{"n": 5.0, "size_mb": 0.0, "seconds": 0.0}], byte_scaled=True)
        assert coeffs.per_mb == 0.0

    def test_thin_cells_are_dropped(self):
        rows = [
            {
                "task": "find",
                "device": "cpu",
                "media_type": "image",
                "embedder": "siglip",
                "n": 10,
                "size_mb": 0,
                "step": "score",
                "seconds": 1.0,
                "ok": True,
                "complete": True,
            }
        ]
        assert fit_profile(rows, min_samples=2)["tasks"] == {}
        assert fit_profile(rows, min_samples=1)["tasks"]["find"]["cells"]

    def test_rollup_cells_are_emitted_alongside_the_exact_one(self):
        rows = []
        for n in (100, 200):
            rows.append(
                {
                    "task": "find",
                    "device": "cpu",
                    "media_type": "image",
                    "embedder": "siglip",
                    "n": n,
                    "size_mb": 0,
                    "step": "score",
                    "seconds": 0.01 * n,
                    "ok": True,
                    "complete": True,
                }
            )
        cells = fit_profile(rows, min_samples=2)["tasks"]["find"]["cells"]
        assert set(cells) == {"cpu|image|siglip", "cpu|image|", "cpu||"}


class TestRoundTrip:
    def test_recorded_rows_fit_into_a_profile_that_paces_the_task(self, sink, tmp_path, monkeypatch):
        # The whole tuning loop in one test: record a task at two sizes, fit,
        # load the result, and check the pacing now reflects what was measured.
        monkeypatch.setattr(timing_profile, "resolve_device_name", lambda: "cpu")
        monkeypatch.setattr(timing_recorder, "resolve_device_name", lambda: "cpu")
        monkeypatch.setattr(timing_profile, "cuml_active", lambda: False)
        monkeypatch.setattr(timing_recorder, "cuml_active", lambda: False)

        clock = {"t": 0.0}
        monkeypatch.setattr(timing_recorder.time, "monotonic", lambda: clock["t"])

        for n, score_secs in ((1_000, 1.0), (100_000, 100.0)):
            tracker = _tracker()
            with record_task(tracker, "text_sort", media_type="image", embedder="siglip") as rec:
                tracker.update("sorting", "", 0, 0, step=1, total_steps=3)
                clock["t"] += 8.0  # a fixed 8s encoder load, whatever the size
                tracker.update("sorting", "", 0, 0, step=2, total_steps=3)
                clock["t"] += 0.05
                tracker.update("sorting", "", 0, 0, step=3, total_steps=3)
                clock["t"] += score_secs  # scoring scales with n
                rec.set_scale(n=n)

        profile = fit_profile(sink(), min_samples=2)
        path = tmp_path / "fitted.json"
        path.write_text(json.dumps(profile), encoding="utf-8")

        try:
            timing.reload_profile(str(path))
            small = _weights("text_sort", device="cpu", media_type="image", embedder="siglip", n=1_000)
            large = _weights("text_sort", device="cpu", media_type="image", embedder="siglip", n=100_000)
            # 8s load vs 1s score at n=1k; 8s load vs 100s score at n=100k.
            assert small[0] > small[2]
            assert large[2] > large[0]
        finally:
            timing.reload_profile("")

    def test_coverage_report_names_the_unmeasured_families(self):
        rows = [
            {
                "task": "find",
                "device": "cpu",
                "media_type": "",
                "embedder": "",
                "n": n,
                "size_mb": 0,
                "step": "score",
                "seconds": 1.0,
                "ok": True,
                "complete": True,
            }
            for n in (10, 20)
        ]
        lines = "\n".join(coverage_report(rows, fit_profile(rows, min_samples=2)))
        assert "find" in lines
        assert "NOT MEASURED" in lines
        assert "text_sort" in lines
