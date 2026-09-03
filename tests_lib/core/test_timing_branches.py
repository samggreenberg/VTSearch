"""Cache branches: which path a step took, and what the fitter may do with it.

A step's cost often forks on a cache that its duration cannot reveal. #3345's
sweep opened 16 datasets and every one of them **restored** the coverage atlas
cached in its pickle, recording 0.008-0.016 s at every ``n`` from 245 to 2954;
the same sweep's ``dataset_stage`` leg read the embeddings pkl the
``dataset_load`` leg had just written and recorded 0.000-0.002 s of embedding
across all four image tiers, in a separate interpreter. Every one of those
numbers is correct. None of them is a cost model, because the branch a user
waits on — the minutes-long hierarchical-k-means rebuild, the real embed — was
never run (#3521, measured in
``docs/experiments/2026-09-02-timing-r2-3345/REPORT.md``).

These tests pin the three halves of the answer: the recorder carries the branch
name, the fitter prices a forked step only from the runs that did the work, and
a step that only ever read a cache withholds its whole cell rather than
shipping a confident millisecond.
"""

import pytest

from vtscore.concurrency.progress import PROGRESS_COMMON_EXTRAS, ProgressTracker
from vtscore.timing import recorder as timing_recorder
from vtscore.timing.fit import cheap_branch_only, coverage_report, fit_profile, fit_step, load_rows
from vtscore.timing.recorder import RECORD_ENV_VAR, note_branch, note_no_encoder_load, record_task


def _tracker() -> ProgressTracker:
    return ProgressTracker(extra_fields=dict(PROGRESS_COMMON_EXTRAS))


@pytest.fixture
def sink(tmp_path, monkeypatch):
    """Arm the recorder at a temp JSONL and yield a reader for what it wrote."""
    path = tmp_path / "timings.jsonl"
    monkeypatch.setenv(RECORD_ENV_VAR, str(path))
    timing_recorder.reset_seen_models_for_tests()

    def read():
        return load_rows([str(path)]) if path.exists() else []

    return read


def _open_run(tracker) -> None:
    """The two tracker steps a dataset open reports."""
    tracker.update("loading", "items", 0, 0, step=1, total_steps=2)
    tracker.update("loading", "coverage", 0, 0, step=2, total_steps=2)


def _sample(n: float, seconds: float, branch: str = "") -> dict:
    return {"n": n, "size_mb": 0.0, "seconds": seconds, "cold": False, "branch": branch}


class TestRecorderCarriesTheBranch:
    def test_marked_step_carries_its_branch(self, sink):
        tracker = _tracker()
        with record_task(tracker, "dataset_open", media_type="image") as rec:
            rec.mark_branch("coverage", "restored")
            _open_run(tracker)
            rec.set_scale(n=980)
        rows = {r["step"]: r for r in sink()}
        assert rows["coverage"]["branch"] == "restored"
        assert "branch" not in rows["items"], "an unforked step must not claim one"

    def test_note_branch_reaches_the_thread_bound_recorder(self, sink):
        """The deep call sites name the branch without holding the recorder."""
        tracker = _tracker()
        with record_task(tracker, "dataset_open", media_type="image") as rec:
            rec.bind_thread()
            note_branch("coverage", "rebuilt")  # as the route's atlas branch does
            _open_run(tracker)
        assert {r["step"]: r.get("branch") for r in sink()}["coverage"] == "rebuilt"

    def test_note_branch_is_a_no_op_with_nothing_recording(self):
        """Product code calls this unconditionally; it must never raise."""
        timing_recorder._active.recorder = None
        note_branch("embed", "cached")
        note_no_encoder_load()

    def test_a_cached_run_does_not_claim_the_encoder_key(self, sink):
        """The run that reads a pkl must leave the residency key for the one
        that really loads the model — otherwise the genuinely cold load that
        follows is written ``cold_model: false`` (the #3345 mislabel, reached
        by a different route)."""
        cached_tracker = _tracker()
        with record_task(cached_tracker, "dataset_stage", media_type="image", embedder="siglip") as rec:
            rec.bind_thread()
            note_branch("embed", "cached")
            note_no_encoder_load()
            for step in (1, 2, 3):
                cached_tracker.update("loading", "x", 0, 0, step=step, total_steps=3)
        fresh_tracker = _tracker()
        with record_task(fresh_tracker, "dataset_stage", media_type="image", embedder="siglip") as rec:
            rec.bind_thread()
            note_branch("embed", "fresh")
            for step in (1, 2, 3):
                fresh_tracker.update("loading", "x", 0, 0, step=step, total_steps=3)
        embeds = [r for r in sink() if r["step"] == "embed"]
        assert [r["branch"] for r in embeds] == ["cached", "fresh"]
        assert "cold_model" not in embeds[0], "a run that loaded no encoder claims nothing"
        assert embeds[1]["cold_model"] is True, "the real load is the cold one"

    def test_only_phases_records_a_partial_run(self, sink):
        """The on-demand atlas rebuild is a dataset_open's second step alone.

        Without the narrowing it would write ``items: 0.0`` — indistinguishable
        from a measurement that opening a dataset's pickle is free.
        """
        tracker = _tracker()
        rec = record_task(
            tracker,
            "dataset_open",
            media_type="image",
            status_phases={"loading": "coverage"},
            only_phases=("coverage",),
        )
        rec.start()
        rec.mark_branch("coverage", "rebuilt")
        tracker.update("loading", "Building coverage atlas…", 0, 0, step=1, total_steps=1)
        rec.finish(n=980)
        assert [r["step"] for r in sink()] == ["coverage"]


class TestFittingAForkedStep:
    def test_priced_from_the_runs_that_did_the_work(self):
        """A restore is not a cheap sample of a rebuild; it is a different path."""
        cheap = [_sample(n, 0.01, "restored") for n in (245, 980, 2954)]
        dear = [_sample(245, 61.0, "rebuilt"), _sample(980, 240.0, "rebuilt")]
        coeffs = fit_step(cheap + dear, byte_scaled=False)
        assert coeffs is not None
        # The two rebuilds alone: 61 s at 245 items, 240 s at 980.
        assert coeffs.b == pytest.approx(0.2435, abs=1e-3)
        assert coeffs.seconds(n=980) == pytest.approx(240.0, rel=0.02)

    def test_unmarked_rows_fit_exactly_as_they_did(self):
        """Absent markers are not evidence of a fork, and must change nothing."""
        plain = [_sample(100, 0.1), _sample(200, 0.2), _sample(300, 0.3)]
        assert fit_step(plain, byte_scaled=False) == fit_step(
            [{k: v for k, v in s.items() if k != "branch"} for s in plain], byte_scaled=False
        )

    def test_cheap_only_is_not_the_same_as_cheap(self):
        assert cheap_branch_only([_sample(1, 0.01, "restored")]) is True
        assert cheap_branch_only([_sample(1, 0.01, "restored"), _sample(1, 60.0, "rebuilt")]) is False
        assert cheap_branch_only([_sample(1, 0.01)]) is False, "unmarked is not cheap-only"
        assert cheap_branch_only([]) is False


def _row(step: str, seconds: float, n: float, branch: str = "") -> dict:
    row = {
        "task": "dataset_open",
        "device": "cuda",
        "cuml": False,
        "media_type": "image",
        "embedder": "siglip",
        "n": n,
        "size_mb": 0.0,
        "ok": True,
        "complete": True,
        "step": step,
        "seconds": seconds,
    }
    if branch:
        row["branch"] = branch
    return row


#: The three opens #3345 measured, at the sizes it measured them.
_RESTORED_ONLY = [
    row
    for n, items, cov in ((245, 0.4, 0.010), (980, 1.2, 0.012), (2954, 3.0, 0.016))
    for row in (_row("items", items, n), _row("coverage", cov, n, "restored"))
]


class TestACheapOnlyStepWithholdsItsCell:
    def test_no_cell_is_emitted(self):
        """Not just the coverage step: the whole cell.

        ``step_terms`` fills a missing step from ``TaskSpec.default_terms``,
        which are *pseudo-seconds*. Emitting a measured ``items`` in real
        seconds beside a 0.85 pseudo-second ``coverage`` would produce a weight
        vector in no units at all.
        """
        profile = fit_profile(_RESTORED_ONLY, min_samples=2)
        assert profile["tasks"].get("dataset_open", {}).get("cells", {}) == {}

    def test_one_real_rebuild_unlocks_the_cell(self):
        rows = _RESTORED_ONLY + [
            _row("coverage", 61.0, 245, "rebuilt"),
            _row("coverage", 240.0, 980, "rebuilt"),
            _row("items", 0.4, 245),
            _row("items", 1.2, 980),
        ]
        steps = fit_profile(rows, min_samples=2)["tasks"]["dataset_open"]["cells"]["cuda|image|siglip"]["steps"]
        assert set(steps) == {"items", "coverage"}
        assert steps["coverage"]["b"] > 0.2, "priced from the rebuilds, not the restores"

    def test_the_report_says_which_branch_it_saw(self):
        """The line a sample count cannot carry — the ask in #3521."""
        profile = fit_profile(_RESTORED_ONLY, min_samples=2)
        report = "\n".join(coverage_report(_RESTORED_ONLY, profile))
        assert "coverage: 3 runs, all 'restored'" in report
        assert "cached path only" in report

    def test_the_report_counts_runs_not_cell_buckets(self):
        """Every row lands in three cells at three specificities; counting the
        buckets would report each of the three opens nine times."""
        report = "\n".join(coverage_report(_RESTORED_ONLY, fit_profile(_RESTORED_ONLY, min_samples=2)))
        assert "9 runs" not in report
