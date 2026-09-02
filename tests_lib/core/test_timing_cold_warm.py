"""Cold vs warm runs: the marker the recorder stamps and the fit that reads it.

A once-per-process cost — downloading and instantiating an encoder — is paid by
the first run in a process and by no other. Without a marker saying which run
that was, its rows are unfittable: `dataset_load`'s model load measured
``54.905, 0.000, 0.000, 0.000`` on rack7n06 and fitted to ``{"a": 0.0}``, giving
0 % of a 412-image import's bar to a step that took most of a minute inside the
very run it was fitted from (#3520, measured in
``docs/experiments/2026-09-02-timing-r2-3345/REPORT.md``).

These tests pin both halves: the recorder must label the branch, and the fitter
must fit the two populations apart rather than medianing them together.
"""

import pytest

from vtscore.concurrency.progress import PROGRESS_COMMON_EXTRAS, ProgressTracker
from vtscore.timing import recorder as timing_recorder
from vtscore.timing.fit import fit_profile, fit_step, load_rows, normalize_row
from vtscore.timing.profile import StepCoeffs
from vtscore.timing.recorder import RECORD_ENV_VAR, record_task
from vtscore.timing.tasks import TASKS


def _tracker() -> ProgressTracker:
    return ProgressTracker(extra_fields=dict(PROGRESS_COMMON_EXTRAS))


def _fit(samples: list[dict], *, byte_scaled: bool = False) -> StepCoeffs:
    """``fit_step`` narrowed to non-``None`` (its no-samples return)."""
    coeffs = fit_step(samples, byte_scaled)
    assert coeffs is not None
    return coeffs


def _sample(n: float, seconds: float, *, cold: bool = False) -> dict:
    return {"n": n, "size_mb": 0.0, "seconds": seconds, "cold": cold}


@pytest.fixture
def sink(tmp_path, monkeypatch):
    """Arm the recorder at a temp JSONL and yield a reader for what it wrote."""
    path = tmp_path / "timings.jsonl"
    monkeypatch.setenv(RECORD_ENV_VAR, str(path))
    timing_recorder.reset_seen_models_for_tests()

    def read():
        return load_rows([str(path)]) if path.exists() else []

    return read


def _run_text_sort(tracker, *, skip_model_load: bool = False) -> None:
    if not skip_model_load:
        tracker.update("sorting", "loading", 0, 0, step=1, total_steps=3)
    tracker.update("sorting", "embedding", 0, 0, step=2, total_steps=3)
    tracker.update("sorting", "scoring", 0, 0, step=3, total_steps=3)


class TestRecorderMarksTheBranch:
    def test_first_run_for_an_encoder_is_cold_and_the_next_is_warm(self, sink):
        for skip in (False, True):
            tracker = _tracker()
            with record_task(tracker, "text_sort", media_type="image", embedder="siglip") as rec:
                _run_text_sort(tracker, skip_model_load=skip)
                rec.set_scale(n=1234)
        rows = sink()
        first = [r for r in rows if r["cold_model"]]
        rest = [r for r in rows if not r["cold_model"]]
        # One whole run per branch, not one row: the cold run's *other* steps
        # carry the process warmup too, which is why the flag is per-run.
        assert len(first) == 3 and len(rest) == 3

    def test_the_ledger_is_keyed_by_media_type_and_embedder(self, sink):
        # Keying on the embedder alone lets two blanks from different media
        # types collide, and the second one's genuinely cold load is then
        # written warm — the mislabel #3345 measured on the other recorder.
        for media_type in ("image", "audio"):
            tracker = _tracker()
            with record_task(tracker, "text_sort", media_type=media_type) as rec:
                _run_text_sort(tracker)
                rec.set_scale(n=10)
        rows = sink()
        assert {r["media_type"] for r in rows if r["cold_model"]} == {"image", "audio"}

    def test_a_task_that_loads_no_encoder_neither_claims_nor_carries_the_flag(self, sink):
        # dataset_open reads a pkl. If it claimed (image, siglip), the genuinely
        # cold text sort behind it would be stamped warm.
        tracker = _tracker()
        with record_task(tracker, "dataset_open", media_type="image", embedder="siglip") as rec:
            tracker.update("loading", "items", 0, 0, step=1, total_steps=2)
            tracker.update("loading", "coverage", 0, 0, step=2, total_steps=2)
            rec.set_scale(n=500)
        assert all("cold_model" not in r for r in sink())

        tracker = _tracker()
        with record_task(tracker, "text_sort", media_type="image", embedder="siglip") as rec:
            _run_text_sort(tracker)
            rec.set_scale(n=500)
        assert any(r["cold_model"] for r in sink() if r["task"] == "text_sort")

    def test_a_run_that_recorded_nothing_does_not_claim_the_key(self, sink):
        # The key is claimed where the rows are written, not at construction: a
        # run whose tracker never reached a step writes nothing, and claiming on
        # its behalf would leave the next run — the first one anybody can fit —
        # stamped warm.
        tracker = _tracker()
        with record_task(tracker, "text_sort", media_type="image", embedder="siglip"):
            tracker.update("idle", "", 0, 0)
        assert sink() == []

        tracker = _tracker()
        with record_task(tracker, "text_sort", media_type="image", embedder="siglip") as rec:
            _run_text_sort(tracker)
            rec.set_scale(n=10)
        assert all(r["cold_model"] for r in sink())

    def test_an_embedder_learned_late_still_decides_the_key(self, sink):
        # A dataset load that lets the media type's default stand only resolves
        # the encoder deep in the embed stage (#3345), so the key has to be read
        # where the rows are written rather than at construction — otherwise the
        # late-named run claims the blank key and the next one is stamped warm.
        tracker = _tracker()
        with record_task(tracker, "text_sort", media_type="image") as rec:
            _run_text_sort(tracker)
            rec.set_scale(n=10, embedder="siglip")
        assert all(r["embedder"] == "siglip" and r["cold_model"] for r in sink())

        tracker = _tracker()
        with record_task(tracker, "text_sort", media_type="image", embedder="siglip") as rec:
            _run_text_sort(tracker)
            rec.set_scale(n=20)
        assert not any(r["cold_model"] for r in sink() if r["n"] == 20)

    def test_every_registered_task_declares_whether_it_loads_an_encoder(self):
        # A new task defaults to False, which fits exactly as today; this pins
        # the ones that were classified so a rename cannot silently drop one.
        marked = {name for name, spec in TASKS.items() if spec.loads_encoder}
        assert marked == {
            "dataset_load",
            "dataset_stage",
            "detector_load",
            "find",
            "text_sort",
            "train_and_score",
        }


class TestNormalizeCarriesTheFlag:
    def test_generic_row(self):
        row = normalize_row({"task": "text_sort", "step": "load_model", "seconds": 15.4, "cold_model": True})
        assert row is not None and row["cold"] is True

    def test_legacy_profiler_row(self):
        # `_load_profiler` has stamped `cold_model` since long before the generic
        # recorder did; the fitter simply never read it.
        row = normalize_row({"phase": "model_load", "n": 412, "seconds": 54.9, "cold_model": True})
        assert row is not None and row["task"] == "dataset_load" and row["cold"] is True

    def test_a_row_without_the_marker_is_warm(self):
        row = normalize_row({"task": "text_sort", "step": "score", "seconds": 1.8})
        assert row is not None and row["cold"] is False


class TestFittingTheTwoPopulationsApart:
    def test_a_cold_row_among_warm_zeros_no_longer_fits_to_free(self):
        # The measured rack7n06 rows for `dataset_load` · `load`: one cold load
        # at 54.9 s and three warm runs that skipped the step outright.
        measured = [_sample(412, 54.905, cold=True)] + [_sample(n, 0.0) for n in (245, 1000, 2954)]

        unmarked = _fit([{**s, "cold": False} for s in measured])
        assert unmarked.a == 0.0, "the bug: one population, median zero, a step priced free"

        marked = _fit(measured)
        assert marked.a > unmarked.a, "the coefficient must MOVE, not merely be finite"
        assert marked.a == pytest.approx(0.5)

    def test_a_step_the_warm_runs_really_do_pay_keeps_its_measured_cost(self):
        # The floor is a guard against a confident zero, never a replacement for
        # a measurement: a warm population with a real median keeps it.
        coeffs = _fit([_sample(412, 9.0, cold=True)] + [_sample(n, 2.0) for n in (245, 1000, 2954)])
        assert coeffs.a == pytest.approx(2.0)

    def test_a_step_that_is_free_on_both_branches_stays_free(self):
        coeffs = _fit([_sample(412, 0.0, cold=True)] + [_sample(n, 0.0) for n in (245, 1000)])
        assert coeffs.a == 0.0

    def test_the_cold_run_is_held_out_of_a_slope(self):
        # #3062: one cold row at the smallest n pulled a finalize slope to less
        # than half its warm value and collapsed the fit's r2. The cold row here
        # is 20 s of process warmup on top of the same 0.01 s/item line.
        warm = [_sample(float(n), 0.01 * n) for n in (200, 400, 600, 800)]
        coeffs = _fit([_sample(100.0, 21.0, cold=True), *warm])
        assert coeffs.b == pytest.approx(0.01)
        assert coeffs.r2 == pytest.approx(1.0)

        # Pooled, the same rows lose the line altogether: OLS comes back with a
        # negative slope, `fit_step` refuses it, and the step ships as a flat
        # 6 s median that is wrong at every size measured.
        poisoned = _fit([{**s, "cold": False} for s in [_sample(100.0, 21.0, cold=True), *warm]])
        assert poisoned.b == 0.0
        assert poisoned.a == pytest.approx(6.0)

    def test_the_holdout_never_costs_a_cell_its_only_line(self):
        # A two-run sweep is the minimum `fit_profile` accepts, and its first
        # run is always the cold one. Holding that row out would leave a single
        # warm point, from which no slope is estimable — so the step would ship
        # as a flat median and the profile would lose the scaling it measured.
        two_run = [_sample(1_000.0, 1.0, cold=True), _sample(100_000.0, 100.0)]
        assert _fit(two_run).b == pytest.approx(0.001)

        # One more warm size and the holdout is affordable, so it applies.
        three_run = [*two_run, _sample(50_000.0, 50.0)]
        assert _fit(three_run).b == pytest.approx(0.001)
        assert _fit([*three_run[:1], _sample(50_000.0, 5.0), _sample(100_000.0, 10.0)]).b == pytest.approx(0.0001)

    def test_a_cell_that_only_ever_ran_cold_still_fits_its_cold_rows(self):
        # One measurement of the expensive branch beats none — and it is all the
        # legacy profiler ever writes, since a warm load emits no model row.
        coeffs = _fit([_sample(float(n), 0.02 * n, cold=True) for n in (200, 400, 600)])
        assert coeffs.b == pytest.approx(0.02)

    def test_byte_scaled_steps_ignore_the_marker(self):
        # `cold_model` says nothing about whether the archive was cached; the
        # per-MB branch has its own filter for that.
        rows = [
            {"n": 400.0, "size_mb": 100.0, "seconds": 10.0, "cold": True},
            {"n": 400.0, "size_mb": 100.0, "seconds": 10.0, "cold": False},
        ]
        assert _fit(rows, byte_scaled=True).per_mb == pytest.approx(0.1)


class TestEndToEnd:
    def test_a_sweep_that_saw_one_cold_load_prices_the_step_above_zero(self):
        rows = []
        for i, n in enumerate((412, 245, 1000, 2954)):
            for step, seconds in (("load_model", 15.4 if i == 0 else 0.0), ("embed_query", 0.04), ("score", 0.001 * n)):
                rows.append(
                    {
                        "task": "text_sort",
                        "device": "cuda",
                        "cuml": True,
                        "media_type": "image",
                        "embedder": "siglip",
                        "n": n,
                        "size_mb": 0.0,
                        "step": step,
                        "seconds": seconds,
                        "ok": True,
                        "complete": True,
                        "cold_model": i == 0,
                    }
                )
        cell = fit_profile(rows)["tasks"]["text_sort"]["cells"]["cuda+cuml|image|siglip"]
        assert cell["steps"]["load_model"]["a"] > 0
        # The step the sweep measured properly is untouched by the split.
        assert cell["steps"]["score"]["b"] == pytest.approx(0.001)
