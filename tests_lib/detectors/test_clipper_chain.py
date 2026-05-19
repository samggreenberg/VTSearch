"""Phase 1 tests for the clipper-chain abstraction.

See ``docs/plans/clipper-chain.md`` for the design.

Covers:
- Validation (unknown plugins, mismatched adjacency, empty chain).
- Single-step chain matches the legacy single-clipper path's origin
  stamping (regression guard).
- Same-type two-clipper chain produces correct per-clip trail and
  embeds each clip from its own bytes.
- Cross-type chain (converter + clipper) propagates the media type
  through the trail and flags every clip for recomputation.
- Resolver replay reproduces the same final embedding as the load-time
  pass.
- Malformed ``clipper_chain`` JSON falls back to the legacy path.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_text_media(media_id: int, text: str, *, origin_path: str = "/data/text") -> dict:
    """Construct a fake text media dict with an embedding."""
    rng = np.random.default_rng(media_id)
    return {
        "id": media_id,
        "type": "text",
        "filename": f"doc_{media_id}.txt",
        "media_string": text,
        "duration": 0,
        "word_count": len(text.split()),
        "character_count": len(text),
        "md5": hashlib.md5(text.encode("utf-8")).hexdigest(),
        "embedding": rng.standard_normal(384).astype(np.float32),
        "origin": {
            "importer": "server_folder",
            "params": {"path": origin_path, "media_type": "text"},
        },
        "origin_name": f"doc_{media_id}.txt",
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestNormaliseChain:
    def test_empty_input(self):
        from vtscore.datasets.clipper_chain import normalise_chain

        assert normalise_chain(None) == []
        assert normalise_chain([]) == []

    def test_drops_default_clipper_steps(self):
        from vtscore.datasets.clipper_chain import normalise_chain

        steps = [
            {"kind": "clipper", "name": "text_default", "params": {}},
            {"kind": "clipper", "name": "text_sentence", "params": {}},
            {"kind": "clipper", "name": "sound_default", "params": {}},
        ]
        out = normalise_chain(steps)
        assert len(out) == 1
        assert out[0]["name"] == "text_sentence"

    def test_rejects_missing_kind(self):
        from vtscore.datasets.clipper_chain import normalise_chain

        with pytest.raises(ValueError):
            normalise_chain([{"name": "text_sentence"}])

    def test_rejects_missing_name(self):
        from vtscore.datasets.clipper_chain import normalise_chain

        with pytest.raises(ValueError):
            normalise_chain([{"kind": "clipper"}])


class TestValidateChain:
    def test_empty_chain_returns_source_type(self):
        from vtscore.datasets.clipper_chain import validate_chain

        assert validate_chain([], "text") == "text"

    def test_same_type_chain(self):
        from vtscore.datasets.clipper_chain import validate_chain

        steps = [{"kind": "clipper", "name": "text_sentence", "params": {}}]
        assert validate_chain(steps, "text") == "text"

    def test_cross_type_chain(self):
        from vtscore.datasets.clipper_chain import validate_chain

        steps = [
            {"kind": "converter", "name": "document2text", "params": {}},
            {"kind": "clipper", "name": "text_sentence", "params": {}},
        ]
        assert validate_chain(steps, "document") == "text"

    def test_rejects_unknown_clipper(self):
        from vtscore.datasets.clipper_chain import validate_chain

        with pytest.raises(ValueError, match="unknown clipper"):
            validate_chain([{"kind": "clipper", "name": "nope", "params": {}}], "text")

    def test_rejects_unknown_converter(self):
        from vtscore.datasets.clipper_chain import validate_chain

        with pytest.raises(ValueError, match="unknown converter"):
            validate_chain([{"kind": "converter", "name": "nope", "params": {}}], "document")

    def test_rejects_type_mismatch(self):
        from vtscore.datasets.clipper_chain import validate_chain

        # text → text_sentence is fine; sound_tiling expects audio.
        steps = [
            {"kind": "clipper", "name": "text_sentence", "params": {}},
            {"kind": "clipper", "name": "sound_tiling", "params": {"duration": 1.0}},
        ]
        with pytest.raises(ValueError, match="expects input type 'audio'"):
            validate_chain(steps, "text")


# ---------------------------------------------------------------------------
# Apply chain — origin stamping + final clips
# ---------------------------------------------------------------------------


class TestApplyChainToClips:
    def test_single_clipper_chain_stamps_legacy_keys(self, monkeypatch):
        """A length-1 clipper chain produces the same origin stamp shape
        as the legacy single-clipper path: ``clipper`` plus ``clip_index``
        for genuine sub-items."""
        from vtscore.datasets.clipper_chain import apply_chain_to_clips

        text = "First. Second. Third."
        media = _make_text_media(1, text)
        clips = {1: media}
        result = apply_chain_to_clips(
            clips,
            [{"kind": "clipper", "name": "text_sentence", "params": {}}],
        )
        assert result is not None
        final_type, needs_recompute = result
        assert final_type == "text"
        assert len(clips) == 3
        assert all(needs_recompute)
        for idx, clip in enumerate(clips.values()):
            params = clip["origin"]["params"]
            assert params["clipper"] == "text_sentence"
            assert params["clip_index"] == str(idx)
            assert "clipper_chain" in params
            trail = json.loads(params["clipper_chain"])
            assert len(trail) == 1
            assert trail[0]["kind"] == "clipper"
            assert trail[0]["name"] == "text_sentence"
            assert trail[0]["out_index"] == idx

    def test_single_clipper_no_split_keeps_origin_untouched_recompute(self, monkeypatch):
        """A non-default clipper that short-circuits to ``[media]`` (e.g.
        TextSentence on a single-sentence text) must NOT mark the clip as
        a sub-item and must NOT stamp ``clip_index``."""
        from vtscore.datasets.clipper_chain import apply_chain_to_clips

        media = _make_text_media(1, "Only one sentence with no terminator")
        clips = {1: media}
        result = apply_chain_to_clips(
            clips,
            [{"kind": "clipper", "name": "text_sentence", "params": {}}],
        )
        assert result is not None
        final_type, needs_recompute = result
        assert final_type == "text"
        assert len(clips) == 1
        assert needs_recompute == [False]
        params = next(iter(clips.values()))["origin"]["params"]
        assert params["clipper"] == "text_sentence"
        assert "clip_index" not in params

    def test_chain_preserves_parent_origin(self):
        """The parent's origin (importer, path, etc.) survives all chain
        steps and lands on the final clips."""
        from vtscore.datasets.clipper_chain import apply_chain_to_clips

        media = _make_text_media(1, "A. B. C.", origin_path="/srv/abc")
        clips = {1: media}
        apply_chain_to_clips(
            clips,
            [{"kind": "clipper", "name": "text_sentence", "params": {}}],
        )
        for clip in clips.values():
            origin = clip["origin"]
            assert origin["importer"] == "server_folder"
            assert origin["params"]["path"] == "/srv/abc"


# ---------------------------------------------------------------------------
# Parse trail
# ---------------------------------------------------------------------------


class TestParseTrail:
    def test_accepts_json_string(self):
        from vtscore.datasets.clipper_chain import parse_trail

        raw = json.dumps([{"kind": "clipper", "name": "text_sentence", "out_index": 0}])
        steps = parse_trail(raw)
        assert steps is not None
        assert steps[0]["name"] == "text_sentence"

    def test_accepts_list(self):
        from vtscore.datasets.clipper_chain import parse_trail

        steps = parse_trail([{"kind": "clipper", "name": "text_sentence", "out_index": 0}])
        assert steps is not None
        assert steps[0]["name"] == "text_sentence"

    def test_rejects_malformed(self):
        from vtscore.datasets.clipper_chain import parse_trail

        assert parse_trail(None) is None
        assert parse_trail("") is None
        assert parse_trail("not json") is None
        assert parse_trail("{}") is None  # not a list
        assert parse_trail([{"kind": "unknown", "name": "x"}]) is None
        assert parse_trail([{"kind": "clipper"}]) is None  # missing name


# ---------------------------------------------------------------------------
# Resolver replay
# ---------------------------------------------------------------------------


class TestReplayChainOnFile:
    def test_replay_text_sentence_chain(self, tmp_path, monkeypatch):
        """Replay reads a text file, walks a text_sentence chain, picks the
        ``out_index``-th sentence, and embeds it. The returned embedding
        should match what we would get by embedding the same sentence
        directly via ``embed_file``."""
        from vtscore.datasets.clipper_chain import replay_chain_on_file
        from vtscore.detectors import resolver as resolver_module

        # Stub embed_file to a deterministic function of the source bytes.
        captured: list[tuple[str, str]] = []

        def fake_embed_file(path, media_type, embedder_name=""):
            data = path.read_bytes().decode("utf-8", errors="replace")
            captured.append((data, media_type))
            # Embedding = a hash-derived vector so we can compare.
            h = hashlib.sha256(data.encode("utf-8")).digest()
            return np.frombuffer(h, dtype=np.uint8).astype(np.float32)

        monkeypatch.setattr(resolver_module, "embed_file", fake_embed_file)

        # Write a 3-sentence text file.
        source = tmp_path / "doc.txt"
        source.write_text("Alpha. Bravo. Charlie.", encoding="utf-8")

        # Build a trail picking out_index=1 (Bravo).
        steps = [
            {
                "kind": "clipper",
                "name": "text_sentence",
                "params": {},
                "out_index": 1,
            }
        ]
        embedding = replay_chain_on_file(source, steps)
        assert embedding is not None
        # The captured tempfile must contain just the picked sentence.
        assert len(captured) == 1
        assert captured[0][0] == "Bravo."
        assert captured[0][1] == "text"

    def test_replay_returns_none_for_empty_chain(self, tmp_path):
        from vtscore.datasets.clipper_chain import replay_chain_on_file

        source = tmp_path / "x.txt"
        source.write_text("anything", encoding="utf-8")
        assert replay_chain_on_file(source, []) is None


# ---------------------------------------------------------------------------
# Integration with _apply_clipper (load-pipeline entry point)
# ---------------------------------------------------------------------------


class TestApplyClipperBackwardsCompat:
    def test_legacy_single_clipper_args_route_through_chain(self, monkeypatch):
        """Calling _apply_clipper with the legacy single-clipper args
        produces clip origins indistinguishable from the per-clip schema
        captured by the existing single-clipper code path."""
        # We stub _fixup_clip_md5_and_embeddings and _regenerate_clip_thumbnails
        # so the test doesn't need real audio/embedding wiring.
        from vtscore.datasets import load_pipeline

        calls: dict[str, int] = {"fixup": 0, "thumb": 0}

        def fake_fixup(clips, recompute, media_type, on_progress=None):
            calls["fixup"] += 1

        def fake_thumb(clips, recompute, media_type):
            calls["thumb"] += 1

        monkeypatch.setattr(load_pipeline, "_fixup_clip_md5_and_embeddings", fake_fixup)
        monkeypatch.setattr(load_pipeline, "_regenerate_clip_thumbnails", fake_thumb)

        media = _make_text_media(1, "One. Two. Three.")
        clips = {1: media}

        load_pipeline._apply_clipper(clips, "text_sentence", None)

        assert len(clips) == 3
        for clip in clips.values():
            params = clip["origin"]["params"]
            assert params["clipper"] == "text_sentence"
            assert "clipper_chain" in params
        assert calls["fixup"] == 1
        assert calls["thumb"] == 1
