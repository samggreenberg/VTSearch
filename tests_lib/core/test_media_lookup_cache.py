"""Tests for the revision-keyed secondary media-lookup cache (S14).

``cached_media_lookups`` / ``cached_md5_lookup`` mirror the embedding-matrix
cache: they build the ``(origin, md5, name)`` lookup triple once and reuse it
until the dataset's ``media_revision`` advances (add / remove / reload), so the
many routes that resolve label entries against the active dataset stop paying
an O(N) ``json.dumps``-per-origin rebuild on every request.

See ``tests_lib/core/test_embedding_matrix.py`` for the sibling matrix cache
these tests are patterned on.
"""

from __future__ import annotations

from vtscore.state.core import (
    DatasetContext,
    _request_missing_dataset_context,
    set_thread_dataset_context,
)
from vtscore.state.media_lookup import (
    _origin_key,
    build_media_lookup,
    cached_md5_lookup,
    cached_media_lookups,
)


def _ctx(name: str = "test_lookup_cache") -> DatasetContext:
    ctx = DatasetContext(name)
    ctx.medias[1] = {"id": 1, "md5": "aaa", "origin": {"importer": "x"}, "origin_name": "one.wav"}
    ctx.medias[2] = {"id": 2, "md5": "bbb", "origin": {"importer": "x"}, "origin_name": "two.wav"}
    return ctx


class TestLookupContents:
    """The cached triple matches a fresh ``build_media_lookup``."""

    def test_matches_build_media_lookup(self):
        ctx = _ctx()
        origin, md5, name = cached_media_lookups(ctx)
        exp_origin, exp_md5, exp_name = build_media_lookup(dict(ctx.medias))
        assert origin == exp_origin
        assert md5 == exp_md5
        assert name == exp_name
        # Spot-check the actual mappings.
        assert md5["aaa"] == [1]
        assert md5["bbb"] == [2]
        assert name["one.wav"] == [1]
        assert origin[_origin_key({"importer": "x"}, "two.wav")] == [2]

    def test_md5_helper_returns_md5_element(self):
        ctx = _ctx()
        assert cached_md5_lookup(ctx) == cached_media_lookups(ctx)[1]

    def test_shared_md5_maps_to_both_ids(self):
        ctx = DatasetContext("test_shared_md5")
        ctx.medias[1] = {"id": 1, "md5": "dup", "origin_name": "a"}
        ctx.medias[2] = {"id": 2, "md5": "dup", "origin_name": "b"}
        assert cached_md5_lookup(ctx)["dup"] == [1, 2]


class TestRevisionKeying:
    """Root-cause Pattern #4: the lookup cache keys on ``media_revision``."""

    def test_cache_reused_at_same_revision(self):
        ctx = _ctx()
        first = cached_media_lookups(ctx)
        second = cached_media_lookups(ctx)
        # Same underlying dict objects → served from cache, not rebuilt.
        for a, b in zip(first, second):
            assert a is b
        assert ctx._lookup_index_revision == ctx.media_revision

    def test_add_media_invalidates(self):
        ctx = _ctx()
        _, md5_before, _ = cached_media_lookups(ctx)
        assert "ccc" not in md5_before

        ctx.medias[3] = {"id": 3, "md5": "ccc", "origin_name": "three.wav"}
        _, md5_after, name_after = cached_media_lookups(ctx)
        assert md5_after["ccc"] == [3]
        assert name_after["three.wav"] == [3]
        # A rebuild produced a fresh dict, not the stale cached one.
        assert md5_after is not md5_before

    def test_remove_media_invalidates(self):
        ctx = _ctx()
        assert "bbb" in cached_md5_lookup(ctx)
        del ctx.medias[2]
        assert "bbb" not in cached_md5_lookup(ctx)

    def test_reload_wholesale_reassignment_invalidates(self):
        ctx = _ctx()
        assert cached_md5_lookup(ctx)["aaa"] == [1]
        # Reload: reassign medias to a fresh mapping (even reusing ids).
        ctx.medias = {
            1: {"id": 1, "md5": "zzz", "origin_name": "renamed.wav"},
        }
        md5 = cached_md5_lookup(ctx)
        assert "aaa" not in md5
        assert md5["zzz"] == [1]


class TestEmptyAndSentinel:
    """Empty contexts never populate the cache (the frozen sentinel refuses
    attribute writes, so an attempted store would raise)."""

    def test_empty_context_returns_empties_without_caching(self):
        ctx = DatasetContext("test_empty_lookup")
        origin, md5, name = cached_media_lookups(ctx)
        assert origin == {} and md5 == {} and name == {}
        # Nothing cached: a genuinely-empty dataset rebuilds three empty dicts.
        assert ctx._md5_index is None
        assert ctx._lookup_index_revision is None

    def test_request_missing_sentinel_does_not_raise(self):
        # The sentinel's medias is a frozen empty dict AND it refuses attribute
        # assignment; the empty-guard must return before any store is attempted.
        origin, md5, name = cached_media_lookups(_request_missing_dataset_context)
        assert origin == {} and md5 == {} and name == {}


class TestActiveContextDefault:
    """With no ``ctx`` argument the accessor resolves the active context."""

    def test_defaults_to_active_context(self):
        ctx = _ctx("test_active_default")
        set_thread_dataset_context(ctx)
        try:
            assert cached_md5_lookup()["aaa"] == [1]
            # Serves the same cached objects as the explicit-ctx call.
            assert cached_media_lookups()[1] is cached_media_lookups(ctx)[1]
        finally:
            set_thread_dataset_context(None)
