"""Tests for the shipped per-step timing defaults (:mod:`vtscore.timing`).

Every long-running task paces its bar from the default terms it declares in
:mod:`vtscore.timing.tasks`. The admin-measured per-environment profile that
could once override them was retired (#4667); its public names survive as
no-ops, and the last class here pins that they stay inert.
"""

import json
import math

import pytest

from vtscore import timing
from vtscore.timing import profile as timing_profile
from vtscore.timing.profile import EMPTY_PROFILE, StepCoeffs, parse_profile
from vtscore.timing.tasks import TASKS, TaskSpec


def _weights(task: str, **kwargs) -> list[float]:
    """``step_weights`` narrowed to non-``None`` (its no-coverage return)."""
    weights = timing.step_weights(task, **kwargs)
    assert weights is not None, f"expected weights for {task}"
    return weights


class TestShippedDefaults:
    def test_every_task_declares_a_consistent_spec(self):
        for name, spec in TASKS.items():
            assert spec.name == name
            assert len(spec.step_index) == len(spec.steps)
            assert max(spec.step_index) == spec.tracker_steps
            assert set(spec.byte_scaled) <= set(spec.steps)

    def test_defaults_reproduce_the_hand_tuned_vectors(self):
        # These are the literal vectors the tasks shipped with before the
        # terms were centralised in vtscore.timing.tasks.
        assert _weights("text_sort", device="cpu") == pytest.approx([0.75, 0.05, 0.20])
        assert _weights("find", device="cpu") == pytest.approx([0.10, 0.30, 0.60])
        assert _weights("train_and_score", device="cpu") == pytest.approx([0.10, 0.45, 0.40, 0.05])
        assert _weights("detector_load", device="cpu") == pytest.approx([0.15, 0.15, 0.70])
        assert _weights("dataset_open", device="cpu") == pytest.approx([0.15, 0.85])
        assert _weights("dataset_promote", device="cpu") == pytest.approx([0.6, 0.35, 0.05])

    def test_unknown_task_returns_the_callers_fallback(self):
        assert timing.step_weights("not_a_task", fallback=[1.0, 2.0]) == [1.0, 2.0]
        assert timing.step_weights("not_a_task") is None

    def test_dataset_load_defers_to_its_own_cost_model(self):
        # dataset_load deliberately ships no flat default terms: its default is
        # the measured affine table in _load_cost_model, which is n-aware.
        assert TASKS["dataset_load"].default_terms == ()
        assert timing.step_terms("dataset_load", device="cpu", n=100) is None


class TestPerMediaDefaults:
    """A task's shipped default may differ by media type (#4105)."""

    def test_audio_open_gives_the_read_a_larger_slice(self):
        # #3595 measured a rebuilding audio open spending 0.52-0.63 of its time
        # in the coverage step, against 0.81-0.94 for image: an audio pickle's
        # read is a much larger part of the open.
        assert _weights("dataset_open", device="cpu", media_type="audio") == pytest.approx([0.40, 0.60])
        assert _weights("dataset_open", device="cpu", media_type="image") == pytest.approx([0.15, 0.85])

    @pytest.mark.parametrize("media_type", ["", "video", "text", "not_a_media_type"])
    def test_a_media_type_without_an_override_keeps_the_task_wide_vector(self, media_type):
        assert _weights("dataset_open", device="cpu", media_type=media_type) == pytest.approx([0.15, 0.85])

    def test_overrides_name_registered_media_types(self):
        # A misspelt key ("audios") would never match a lookup and would leave
        # that media type silently on the task-wide vector.
        from vtscore import media

        known = set(media.all_type_ids())
        for spec in TASKS.values():
            assert set(spec.media_default_terms) <= known, spec.name

    def test_defaults_for_resolves_override_then_task_wide(self):
        spec = TASKS["dataset_open"]
        assert spec.defaults_for("audio") == (0.40, 0.60)
        assert spec.defaults_for("image") == spec.default_terms
        assert spec.defaults_for() == spec.default_terms

    def test_an_override_must_be_parallel_to_steps(self):
        with pytest.raises(ValueError, match="parallel to steps"):
            TaskSpec(
                name="t",
                steps=("a", "b"),
                step_index=(1, 2),
                tracker_steps=2,
                scale="n",
                default_terms=(1.0, 1.0),
                media_default_terms={"audio": (1.0,)},
            )

    def test_an_override_needs_a_task_wide_vector_to_override(self):
        # A task with no flat default has its own richer model (dataset_load);
        # a per-media vector must not quietly replace it for one media type.
        with pytest.raises(ValueError, match="requires default_terms"):
            TaskSpec(
                name="t",
                steps=("a",),
                step_index=(1,),
                tracker_steps=1,
                scale="n",
                media_default_terms={"audio": (1.0,)},
            )

    def test_specs_stay_hashable(self):
        # TaskSpec was hashable before the mapping field existed; keep it so.
        assert len({spec for spec in TASKS.values()}) == len(TASKS)


class TestCellLookup:
    def test_keys_run_specific_to_general(self):
        keys = timing.cell_keys("cpu", "image", "siglip")
        assert keys == ("cpu|image|siglip", "cpu|image|", "cpu||")

    def test_cuda_tries_both_cuml_variants_before_generalizing(self, monkeypatch):
        monkeypatch.setattr(timing_profile, "cuml_active", lambda: True)
        keys = timing.cell_keys("cuda:0", "image", "siglip")
        # cuML-on first (it matches this host), but the cuML-off row for the
        # exact media+embedder still beats a media-agnostic row.
        assert keys[:2] == ("cuda+cuml|image|siglip", "cuda|image|siglip")
        assert keys[-1] == "cuda||"


class TestStepCoeffs:
    def test_byte_scaled_terms_track_archive_size(self):
        coeffs = StepCoeffs(per_mb=0.1)
        assert coeffs.seconds(n=999_999, size_mb=100.0) == pytest.approx(10.0)
        assert coeffs.seconds(n=999_999, size_mb=0.0) == pytest.approx(0.0)

    def test_negative_intercept_is_clamped(self):
        # A steep least-squares slope can overshoot into a negative intercept;
        # a step must never be handed a negative slice of the bar.
        assert StepCoeffs(a=-5.0, b=0.001).seconds(n=100) == 0.0

    def test_bare_number_is_shorthand_for_a_fixed_cost(self):
        # Field by field: r2 defaults to NaN, and since Python 3.13 a dataclass
        # __eq__ compares fields with ``==``, so two unfitted StepCoeffs never
        # compare equal as wholes.
        coeffs = StepCoeffs.from_json(4)
        assert coeffs is not None
        assert (coeffs.a, coeffs.b, coeffs.per_mb) == (4.0, 0.0, 0.0)
        assert math.isnan(coeffs.r2)
        assert StepCoeffs.from_json("nope") is None
        assert StepCoeffs.from_json({"a": "nope"}) is None
        assert StepCoeffs.from_json(True) is None


class TestSkippedSteps:
    """A step this run will not enter is priced at zero, not merely cheap.

    #3596: `text_sort`'s `load_model` is seconds on a process's first sort and
    exactly zero on the next 47, so no single term paces both branches. The
    caller can tell the branches apart before it starts; `skip_steps` is how it
    says so.
    """

    def test_a_skipped_step_gives_its_whole_slice_to_the_others(self):
        # Shipped defaults: (0.75, 0.05, 0.20). Warm, the load never happens.
        warm = _weights("text_sort", device="cpu", skip_steps=("load_model",))
        assert warm[0] == 0.0
        assert warm[1] == pytest.approx(0.2)
        assert warm[2] == pytest.approx(0.8)
        assert sum(warm) == pytest.approx(1.0)

    def test_skipping_everything_falls_back_rather_than_dividing_by_zero(self):
        assert timing.step_weights("text_sort", device="cpu", skip_steps=TASKS["text_sort"].steps) is None
        assert timing.step_weights(
            "text_sort", device="cpu", skip_steps=TASKS["text_sort"].steps, fallback=[1.0, 0.0, 0.0]
        ) == [1.0, 0.0, 0.0]

    def test_an_unknown_step_name_is_inert(self):
        # Skipping is a claim about this run, not about the registry, so a name
        # that no longer exists must not blank a task's pacing.
        assert _weights("text_sort", device="cpu", skip_steps=("no_such_step",)) == pytest.approx(
            _weights("text_sort", device="cpu")
        )


class TestRetiredProfile:
    """The per-environment profile is gone (#4667); its names are inert shims."""

    _DOC = {
        "schema": "vtsearch-timing-profile",
        "version": 1,
        "tasks": {"text_sort": {"cells": {"cpu||": {"steps": {"load_model": 1.0, "score": 99.0}}}}},
    }

    def test_the_env_var_no_longer_changes_pacing(self, tmp_path, monkeypatch):
        path = tmp_path / "profile.json"
        path.write_text(json.dumps(self._DOC), encoding="utf-8")
        monkeypatch.setenv(timing_profile.PROFILE_ENV_VAR, str(path))
        assert timing.reload_profile() is EMPTY_PROFILE
        assert timing.reload_profile(str(path)) is EMPTY_PROFILE
        assert timing.active_profile() is EMPTY_PROFILE
        assert not timing.active_profile()
        assert _weights("text_sort", device="cpu") == pytest.approx([0.75, 0.05, 0.20])

    def test_a_valid_document_parses_to_nothing(self):
        assert parse_profile(self._DOC, source="test") is EMPTY_PROFILE

    def test_lookups_report_no_coverage(self):
        assert timing.profile_covers("text_sort") is False
        assert timing.slot_shares("dataset_load", "finalize", device="cpu", media_type="image") is None

    def test_profile_only_arguments_are_ignored(self):
        # device / embedder / n / size_mb / branch selected and scaled profile
        # cells; the flat defaults have no such axes.
        plain = _weights("dataset_open", media_type="image")
        assert _weights(
            "dataset_open",
            device="cuda",
            media_type="image",
            embedder="siglip",
            n=50_000,
            size_mb=900.0,
            branch="restored",
        ) == pytest.approx(plain)

    def test_the_recorder_is_a_no_op(self, tmp_path, monkeypatch):
        from vtscore.timing import recorder

        sink = tmp_path / "rows.jsonl"
        monkeypatch.setenv(recorder.RECORD_ENV_VAR, str(sink))
        assert timing.recording_enabled() is False
        with timing.record_task(object(), "text_sort", media_type="image") as rec:
            rec.start()
            rec.bind_thread()
            rec.mark_branch("coverage", "rebuilt")
            rec.set_scale(n=10)
            rec.finish(ok=True)
        timing.note_branch("embed", "cached")
        timing.note_no_encoder_load()
        assert not sink.exists()
