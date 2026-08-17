"""Tests for :class:`vtscore.datasets.sources.local_archive_member.LocalArchiveMemberSource`.

The manifest-backed media source re-supplies precomputed vectors straight out
of an NPZ manifest (no byte re-derivation), so "Find from origin" works on
archive members that have no on-disk path.  The importer twin is covered in
``test_archive_member_import.py``; here we pin the *source* side: window-keyed
lookup, the bare-member fallback, extension filtering, ``origin_name`` /
``filename`` resolution, and the auto-discovery factory.

No real archive is needed — the source reads only the manifest's vectors, never
the members' bytes — so ``archives`` points at a synthetic path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vtscore.datasets.sources.local_archive_member import (
    SOURCE,
    LocalArchiveMemberSource,
)
from vtscore.embedding.binding import expected_dim_for_embedder

#: Width used when a manifest declares no embedder (nothing to agree with).
DIM = 8


def _dim_for(embedder_name: str | None) -> int:
    """Width a manifest naming *embedder_name* must use to be self-consistent.

    Manifest reads reject an archive whose vectors contradict the embedder it
    names, so a fixture cannot pair a real name with an arbitrary toy width.
    Deriving the width keeps every fixture a *valid* manifest.
    """
    return expected_dim_for_embedder(embedder_name) or DIM


def _make_manifest(
    tmp_path: Path,
    *,
    members,
    clip_starts=None,
    embedder_name: str | None = "xclip",
    filenames=None,
) -> Path:
    rng = np.random.default_rng(3)
    vectors = rng.standard_normal((len(members), _dim_for(embedder_name))).astype(np.float32)
    archive = tmp_path / "shard.tar"
    arrays = {
        "vectors": vectors,
        "members": np.array(members),
        "archives": np.array(str(archive)),
    }
    if clip_starts is not None:
        arrays["clip_start"] = np.array(clip_starts, dtype=np.float32)
    if filenames is not None:
        arrays["filenames"] = np.array(filenames)
    if embedder_name is not None:
        arrays["embedder_name"] = np.array(embedder_name)
    manifest = tmp_path / "manifest.npz"
    np.savez(manifest, **arrays)
    return manifest


# ---------------------------------------------------------------------------
# list_items
# ---------------------------------------------------------------------------


class TestListItems:
    def test_yields_one_item_per_member(self, tmp_path):
        manifest = _make_manifest(tmp_path, members=["a.mp4", "sub/b.mp4"])
        src = LocalArchiveMemberSource(manifest)
        items = list(src.list_items())
        assert {it.key for it in items} == {"a.mp4", "sub/b.mp4"}
        assert all(it.source_name == "local_archive_member" for it in items)
        assert {it.filename for it in items} == {"a.mp4", "b.mp4"}

    def test_windowed_members_key_by_clip_start(self, tmp_path):
        manifest = _make_manifest(
            tmp_path,
            members=["a.mp4", "a.mp4", "b.mp4"],
            clip_starts=[0.0, 10.0, float("nan")],
        )
        src = LocalArchiveMemberSource(manifest)
        keys = {it.key for it in src.list_items()}
        # Two windows of a.mp4 (@0, @10) plus whole-member b.mp4.
        assert keys == {"a.mp4@0", "a.mp4@10", "b.mp4"}

    def test_extension_filter(self, tmp_path):
        manifest = _make_manifest(tmp_path, members=["a.mp4", "b.wav"])
        src = LocalArchiveMemberSource(manifest)
        keys = {it.key for it in src.list_items(extensions=[".mp4"])}
        assert keys == {"a.mp4"}

    def test_extension_filter_is_case_insensitive(self, tmp_path):
        manifest = _make_manifest(tmp_path, members=["A.MP4"])
        src = LocalArchiveMemberSource(manifest)
        keys = {it.key for it in src.list_items(extensions=[".mp4"])}
        assert keys == {"A.MP4"}


# ---------------------------------------------------------------------------
# fetch_item
# ---------------------------------------------------------------------------


class TestFetchItem:
    def test_returns_embedding_never_path(self, tmp_path):
        manifest = _make_manifest(tmp_path, members=["a.mp4"])
        src = LocalArchiveMemberSource(manifest)
        fetched = src.fetch_item("a.mp4")
        assert fetched.path is None
        assert fetched.embedding is not None
        assert fetched.embedding.shape == (_dim_for("xclip"),)
        assert fetched.embedder_name == "xclip"

    def test_unknown_key_returns_empty_fetch(self, tmp_path):
        manifest = _make_manifest(tmp_path, members=["a.mp4"])
        src = LocalArchiveMemberSource(manifest)
        fetched = src.fetch_item("nope.mp4")
        assert fetched.path is None
        assert fetched.embedding is None

    def test_blank_key_returns_empty_fetch(self, tmp_path):
        manifest = _make_manifest(tmp_path, members=["a.mp4"])
        src = LocalArchiveMemberSource(manifest)
        assert src.fetch_item("   ").embedding is None

    def test_bare_member_falls_back_to_first_window(self, tmp_path):
        manifest = _make_manifest(
            tmp_path,
            members=["a.mp4", "a.mp4"],
            clip_starts=[0.0, 10.0],
        )
        src = LocalArchiveMemberSource(manifest)
        # The windowed key resolves to that specific window...
        w0 = src.fetch_item("a.mp4@0").embedding
        w10 = src.fetch_item("a.mp4@10").embedding
        bare = src.fetch_item("a.mp4").embedding
        assert w0 is not None and w10 is not None and bare is not None
        assert not np.array_equal(w0, w10)
        # ...and the bare member resolves to the *first* window's vector.
        assert np.array_equal(bare, w0)

    def test_archive_prefixed_origin_name_resolves(self, tmp_path):
        manifest = _make_manifest(tmp_path, members=["a.mp4"])
        src = LocalArchiveMemberSource(manifest)
        fetched = src.fetch_item("shard.tar::a.mp4")
        assert fetched.embedding is not None


# ---------------------------------------------------------------------------
# resolve_path
# ---------------------------------------------------------------------------


class TestResolvePath:
    def test_resolves_by_origin_name(self, tmp_path):
        manifest = _make_manifest(tmp_path, members=["a.mp4"])
        src = LocalArchiveMemberSource(manifest)
        fetched = src.resolve_path(origin_name="a.mp4")
        assert fetched.embedding is not None
        assert fetched.path is None

    def test_falls_back_to_filename_when_origin_missing(self, tmp_path):
        manifest = _make_manifest(
            tmp_path,
            members=["sub/a.mp4"],
            filenames=["display_name.mp4"],
        )
        src = LocalArchiveMemberSource(manifest)
        # origin_name won't match, but the filename column will.
        fetched = src.resolve_path(origin_name="unmatched", filename="display_name.mp4")
        assert fetched.embedding is not None

    def test_unresolvable_returns_empty(self, tmp_path):
        manifest = _make_manifest(tmp_path, members=["a.mp4"])
        src = LocalArchiveMemberSource(manifest)
        fetched = src.resolve_path(origin_name="x", filename="y.mp4")
        assert fetched.embedding is None
        assert fetched.path is None


# ---------------------------------------------------------------------------
# embedder-name fallback + factory
# ---------------------------------------------------------------------------


class TestEmbedderNameAndFactory:
    def test_blank_embedder_name_falls_back_to_manifest(self, tmp_path):
        manifest = _make_manifest(tmp_path, members=["a.mp4"], embedder_name="siglip")
        src = LocalArchiveMemberSource(manifest, embedder_name="")
        assert src.fetch_item("a.mp4").embedder_name == "siglip"

    def test_explicit_embedder_name_wins(self, tmp_path):
        manifest = _make_manifest(tmp_path, members=["a.mp4"], embedder_name="siglip")
        src = LocalArchiveMemberSource(manifest, embedder_name="override")
        assert src.fetch_item("a.mp4").embedder_name == "override"

    def test_factory_creates_source_from_origin(self, tmp_path):
        manifest = _make_manifest(tmp_path, members=["a.mp4"])
        origin = {"params": {"manifest": str(manifest), "embedder_name": "xclip"}}
        src = SOURCE.create_from_origin(origin)
        assert isinstance(src, LocalArchiveMemberSource)
        assert src.fetch_item("a.mp4").embedding is not None

    def test_factory_returns_none_without_manifest(self, tmp_path):
        assert SOURCE.create_from_origin({"params": {}}) is None
        assert SOURCE.create_from_origin({}) is None

    def test_factory_name(self):
        assert SOURCE.name == "local_archive_member"
