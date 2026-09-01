"""``DatasetContext.reset_derived_caches`` is the single derived-cache drop.

Four call sites used to hand-write their own list of private slots to clear,
and all four had drifted from ``__slots__`` (issue #3377).  Two invariants
keep that from happening again:

1. The derived-cache families and ``_NON_DERIVED_SLOTS`` **partition**
   ``__slots__``, so a newly added slot has to be filed as one or the other
   before this suite passes.
2. ``reset_derived_caches`` actually restores every slot its families name to
   the value a freshly constructed context has — the tables can't claim
   coverage the method doesn't deliver.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.state import clear_medias
from vtscore.state.core import DatasetContext, _iter_slots, thread_dataset_context

_FAMILIES = DatasetContext._DERIVED_CACHE_SLOTS


def _dirty(ctx: DatasetContext, names) -> None:
    """Write a distinguishable non-default value into each of *names*."""
    for i, name in enumerate(sorted(names)):
        # Ints for the counter/revision slots, sentinel objects elsewhere; the
        # assertions only care that the value differs from a fresh context's.
        setattr(ctx, name, i + 1 if name.endswith(("_revision", "_version")) else object())


class TestSlotPartition:
    """The tables beside ``__slots__`` must account for every slot, once."""

    def test_families_and_non_derived_partition_slots(self):
        slots = set(_iter_slots(DatasetContext))
        derived = set().union(*_FAMILIES.values())
        assert derived & DatasetContext._NON_DERIVED_SLOTS == set(), (
            "a slot is filed as both a derived cache and as state"
        )
        assert derived | DatasetContext._NON_DERIVED_SLOTS == slots, (
            "unfiled slots (add them to a _DERIVED_CACHE_SLOTS family or to "
            f"_NON_DERIVED_SLOTS): {slots ^ (derived | DatasetContext._NON_DERIVED_SLOTS)}"
        )

    def test_families_are_pairwise_disjoint(self):
        seen: set[str] = set()
        for name, slots in _FAMILIES.items():
            assert not (seen & slots), f"{name} repeats slots from an earlier family"
            seen |= slots


class TestResetCoversItsFamilies:
    """Every slot a family names is genuinely restored by the reset."""

    @pytest.mark.parametrize("family", sorted(_FAMILIES))
    def test_family_slots_return_to_fresh_values(self, family):
        fresh = DatasetContext("")
        ctx = DatasetContext("dirty")
        _dirty(ctx, _FAMILIES[family])

        ctx.reset_derived_caches(**{f: f == family for f in _FAMILIES})

        for name in sorted(_FAMILIES[family]):
            assert getattr(ctx, name) == getattr(fresh, name), f"{name} not reset"

    @pytest.mark.parametrize("family", sorted(_FAMILIES))
    def test_resetting_one_family_leaves_the_others_alone(self, family):
        ctx = DatasetContext("dirty")
        others = set().union(*(s for f, s in _FAMILIES.items() if f != family))
        _dirty(ctx, others)
        before = {name: getattr(ctx, name) for name in others}

        ctx.reset_derived_caches(**{f: f == family for f in _FAMILIES})

        for name, value in before.items():
            assert getattr(ctx, name) is value, f"{name} was dropped by the {family} reset"

    def test_default_resets_every_family(self):
        fresh = DatasetContext("")
        ctx = DatasetContext("dirty")
        _dirty(ctx, set().union(*_FAMILIES.values()))

        ctx.reset_derived_caches()

        for name in sorted(set().union(*_FAMILIES.values())):
            assert getattr(ctx, name) == getattr(fresh, name), f"{name} not reset"


class TestResetLeavesStateAlone:
    """Non-derived slots are state, not cache; a reset must not touch them."""

    def test_state_slots_survive_a_full_reset(self):
        ctx = DatasetContext("keepme")
        ctx.medias[1] = {"id": 1}
        ctx.coverage_atlas = object()
        ctx.dataset_display_name = "Keep Me"
        ctx.merge_near_duplicates = True
        ctx._emb_sidecar_disabled = True  # a latch, not a cache
        ctx._binding_explicit = True
        revision = ctx.media_revision

        ctx.reset_derived_caches()

        assert ctx.dataset_id == "keepme"
        assert dict(ctx.medias) == {1: {"id": 1}}
        assert ctx.coverage_atlas is not None
        assert ctx.dataset_display_name == "Keep Me"
        assert ctx.merge_near_duplicates is True
        assert ctx._emb_sidecar_disabled is True
        assert ctx._binding_explicit is True
        # Dropping a cache is not a change to the medias.
        assert ctx.media_revision == revision


class TestClearMediasDropsTheSubsetLayout:
    """The concrete #3377 bug: ``clear_medias`` missed eleven slots.

    A subset layout surviving a reload is served verbatim by
    ``POST /api/projection/subset`` whenever the new dataset's requested id set
    matches the stale ``_subset_ids`` — the same stale-pyramid failure the
    ``clear_medias`` docstring already guarded against for the full layout.
    """

    def test_subset_lookup_and_job_slots_are_cleared(self):
        ctx = DatasetContext("test_clear_subset")
        with thread_dataset_context(ctx):
            ctx.medias[1] = {"id": 1, "embedding": np.ones(4, dtype=np.float32)}
            ctx._subset_projection = object()
            ctx._subset_pyramids = {"hex": object()}
            ctx._subset_ids = [1]
            ctx._subset_job_id = "job-abc"
            ctx._subset_content_version = 3
            ctx._subset_region_labels = object()
            ctx._origin_key_index = {"k": [1]}
            ctx._md5_index = {"m": [1]}
            ctx._name_index = {"n": [1]}
            ctx._lookup_index_revision = 1
            ctx._full_job_id = "job-def"
            ctx._relabel_job_id = "job-ghi"

            clear_medias()

            assert ctx._subset_projection is None
            assert ctx._subset_pyramids == {}
            assert ctx._subset_ids is None
            assert ctx._subset_job_id is None
            assert ctx._subset_content_version == 0
            assert ctx._subset_region_labels is None
            assert ctx._origin_key_index is None
            assert ctx._md5_index is None
            assert ctx._name_index is None
            assert ctx._lookup_index_revision is None
            assert ctx._full_job_id is None
            assert ctx._relabel_job_id is None
