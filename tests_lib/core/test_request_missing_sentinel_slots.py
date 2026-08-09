"""The request-missing sentinels must expose the *full* slot set of the
contexts they stand in for.

The sentinels are documented (and relied on) to read as empty contexts so
non-mutating endpoints keep working when a request didn't identify a
dataset/detector.  Both subclasses declare ``__slots__ = ()`` and define no
``__getattr__``, so any slot the sentinel fails to initialise is not "empty"
— it's an ``AttributeError`` (a 500 on the exact dropped-header path the
sentinel exists to serve).  The hand-maintained slot lists drifted twice
(issue #2933); these tests pin the parity so they can't drift again.
"""

import pytest

from vtscore.state.core import (
    DatasetContext,
    DetectorContext,
    RequestMissingContextError,
    _iter_slots,
    _request_missing_dataset_context,
    _request_missing_detector_context,
    is_request_missing_dataset_context,
    is_request_missing_detector_context,
)


class TestSentinelSlotParity:
    """Every slot a real context initialises exists on the sentinel too."""

    @pytest.mark.parametrize(
        ("sentinel", "cls"),
        [
            (_request_missing_dataset_context, DatasetContext),
            (_request_missing_detector_context, DetectorContext),
        ],
    )
    def test_sentinel_exposes_every_slot(self, sentinel, cls):
        real = cls("")
        missing = [name for name in _iter_slots(cls) if hasattr(real, name) and not hasattr(sentinel, name)]
        assert missing == [], f"{type(sentinel).__name__} is missing slots: {missing}"

    def test_dataset_sentinel_reads_projection_slots_as_empty(self):
        """The concrete #2933 crash path: ``GET /api/projection/meta`` with no
        ``X-Dataset-Id`` reads these and must see empty, not AttributeError."""
        ctx = _request_missing_dataset_context
        assert ctx._pyramids == {}
        assert ctx._subset_pyramids == {}
        assert ctx._projection is None
        assert ctx._subset_projection is None
        assert ctx._full_job_id is None
        assert ctx._subset_job_id is None
        assert ctx._subset_ids is None
        assert ctx._subset_content_version == 0
        assert ctx._region_labels is None
        assert ctx._subset_region_labels is None
        assert ctx._relabel_job_id is None
        assert ctx._emb_sidecar_disabled is False

    def test_detector_sentinel_reads_find_mode_as_false(self):
        """``is_find_mode()`` reads this on the sentinel; it must be False."""
        assert _request_missing_detector_context.find_mode is False

    def test_identity_predicates_still_hold(self):
        assert is_request_missing_dataset_context(_request_missing_dataset_context)
        assert is_request_missing_detector_context(_request_missing_detector_context)
        assert not is_request_missing_dataset_context(DatasetContext(""))
        assert not is_request_missing_detector_context(DetectorContext(""))


class TestSentinelStaysFrozen:
    """Copying the slots across must not soften the mutation guard."""

    def test_dataset_containers_are_frozen(self):
        ctx = _request_missing_dataset_context
        with pytest.raises(RequestMissingContextError):
            ctx.medias[1] = {}
        with pytest.raises(RequestMissingContextError):
            ctx._pyramids["hex"] = object()
        with pytest.raises(RequestMissingContextError):
            ctx._subset_pyramids["square"] = object()
        with pytest.raises(RequestMissingContextError):
            ctx.bump_media_revision()

    def test_dataset_attribute_writes_raise(self):
        ctx = _request_missing_dataset_context
        with pytest.raises(RequestMissingContextError):
            ctx._projection = object()
        with pytest.raises(RequestMissingContextError):
            ctx._subset_content_version = 7

    def test_detector_containers_are_frozen(self):
        ctx = _request_missing_detector_context
        with pytest.raises(RequestMissingContextError):
            ctx.good_votes[1] = None
        with pytest.raises(RequestMissingContextError):
            ctx.label_history.append((1, "good", 0.0))
        with pytest.raises(RequestMissingContextError):
            ctx.textsort_suggestions.append("x")
        with pytest.raises(RequestMissingContextError):
            ctx.find_mode = True

    def test_sentinel_ids_are_the_placeholders(self):
        assert _request_missing_dataset_context.dataset_id == "__request_missing__"
        assert _request_missing_detector_context.detector_id == "__request_missing__"


class TestSetFindModeIgnoresSentinel:
    """``set_find_mode`` documents itself as a no-op for the sentinel; its
    ``if det_ctx.detector_id:`` guard treated the truthy placeholder id as a
    real detector (issue #2933)."""

    def test_set_find_mode_is_a_noop_on_the_sentinel(self):
        import vtscore.state.core as core
        from vtscore.detectors.registry import set_find_mode

        prev_detector = getattr(core._thread_local, "detector_context", None)
        prev_forced = getattr(core._thread_local, "forced_detector_context", None)
        prev_predicate = core._request_context_predicate
        core._thread_local.detector_context = None
        core._thread_local.forced_detector_context = None
        core.register_request_context_predicate(lambda: True)
        try:
            assert core.get_active_detector_context() is _request_missing_detector_context
            set_find_mode(True)  # must not raise, must not stick
            assert _request_missing_detector_context.find_mode is False
        finally:
            core.register_request_context_predicate(prev_predicate)
            core._thread_local.detector_context = prev_detector
            core._thread_local.forced_detector_context = prev_forced
