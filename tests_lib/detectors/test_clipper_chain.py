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
        "media_type": "text",
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
# Apply chain: origin stamping + final clips
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
        result = replay_chain_on_file(source, steps)
        assert result is not None
        embedding, content = result
        assert embedding is not None
        # The clip bytes returned match the picked sentence.
        assert content == b"Bravo."
        # The captured tempfile must contain just the picked sentence.
        assert len(captured) == 1
        assert captured[0][0] == "Bravo."
        assert captured[0][1] == "text"

    def test_replay_returns_none_for_empty_chain(self, tmp_path):
        from vtscore.datasets.clipper_chain import replay_chain_on_file

        source = tmp_path / "x.txt"
        source.write_text("anything", encoding="utf-8")
        assert replay_chain_on_file(source, []) is None

    def test_replay_fails_closed_when_out_index_out_of_range(self, tmp_path, monkeypatch, caplog):
        """If the source file produces fewer outputs than recorded, replay
        must NOT silently embed outputs[0]; it must return None so the
        resolver records an embed failure rather than training on the
        wrong sub-clip's embedding."""
        import logging

        from vtscore.datasets.clipper_chain import replay_chain_on_file
        from vtscore.detectors import resolver as resolver_module

        called = {"n": 0}

        def fake_embed_file(path, media_type, embedder_name=""):
            called["n"] += 1
            return np.zeros(8, dtype=np.float32)

        monkeypatch.setattr(resolver_module, "embed_file", fake_embed_file)

        # Source has only 2 sentences ...
        source = tmp_path / "doc.txt"
        source.write_text("Alpha. Bravo.", encoding="utf-8")

        # ... but the trail recorded out_index=5 with no disambiguators.
        # (Mimics an older trail predating the new clip_index/content_hash
        # fields, where the source file has since been edited shorter.)
        steps = [
            {
                "kind": "clipper",
                "name": "text_sentence",
                "params": {},
                "out_index": 5,
            }
        ]
        with caplog.at_level(logging.WARNING, logger="vtscore.datasets.clipper_chain"):
            result = replay_chain_on_file(source, steps)
        assert result is None
        assert called["n"] == 0
        assert any("clipper_chain" in r.message for r in caplog.records)

    def test_replay_detects_n_out_drift_via_content_hash(self, tmp_path, monkeypatch, caplog):
        """When n_out drifts, the selector should fall back to content
        matching and pick the correctly-recorded sub-clip, not the one
        at the recorded positional index."""
        import logging

        from vtscore.datasets.clipper_chain import replay_chain_on_file
        from vtscore.detectors import resolver as resolver_module

        captured: list[str] = []

        def fake_embed_file(path, media_type, embedder_name=""):
            captured.append(path.read_bytes().decode("utf-8", errors="replace"))
            return np.zeros(8, dtype=np.float32)

        monkeypatch.setattr(resolver_module, "embed_file", fake_embed_file)

        # Replay produces 3 sentences. Trail recorded n_out=4 with
        # content_hash matching "Charlie."; i.e. the original load had
        # one extra sentence (now removed) but Charlie still exists.
        source = tmp_path / "doc.txt"
        source.write_text("Alpha. Bravo. Charlie.", encoding="utf-8")

        target_hash = hashlib.md5(b"Charlie.").hexdigest()[:12]
        steps = [
            {
                "kind": "clipper",
                "name": "text_sentence",
                "params": {},
                "out_index": 3,
                "n_out": 4,
                "content_hash": target_hash,
            }
        ]
        with caplog.at_level(logging.WARNING, logger="vtscore.datasets.clipper_chain"):
            result = replay_chain_on_file(source, steps)
        assert result is not None
        embedding, content = result
        assert embedding is not None
        assert content == b"Charlie."
        assert captured == ["Charlie."]
        assert any("drift" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Trail enrichment (n_out / clip_index / content_hash)
# ---------------------------------------------------------------------------


class TestTrailEnrichment:
    def test_clipper_trail_records_n_out_and_clip_index_and_hash(self):
        """Each clipper trail entry carries n_out, clip_index (when the
        clipper stamps it), and a short content_hash."""
        from vtscore.datasets.clipper_chain import _run_clipper_step

        media = _make_text_media(1, "First. Second. Third.")
        outputs, trail = _run_clipper_step(
            media,
            {"kind": "clipper", "name": "text_sentence", "params": {}},
        )
        assert len(outputs) == 3
        assert len(trail) == 3
        for idx, entry in enumerate(trail):
            assert entry["out_index"] == idx
            assert entry["n_out"] == 3
            assert entry["clip_index"] == str(idx)
            assert "content_hash" in entry
            assert len(entry["content_hash"]) == 12

    def test_select_chain_output_returns_none_on_ambiguous_match(self):
        """If multiple replay outputs match the recorded disambiguators,
        the selector refuses to guess and returns None."""
        from vtscore.datasets.clipper_chain import _content_hash, _select_chain_output

        # Two outputs with identical payload naturally produce the same
        # content_hash. The recorded entry's only disambiguator is that
        # hash → both outputs match → genuine ambiguity → None.
        outputs = [
            {"media_string": "duplicate sentence."},
            {"media_string": "duplicate sentence."},
        ]
        h = _content_hash(outputs[0])
        # n_out drift forces the selector past the indexed pick into the
        # content-matching loop, where both outputs match the recorded hash.
        entry = {
            "kind": "clipper",
            "name": "text_sentence",
            "out_index": 0,
            "n_out": 3,
            "content_hash": h,
        }
        assert _select_chain_output(outputs, entry) is None


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
        from vtscore.datasets.stages import clipper as clipper_stage

        calls: dict[str, int] = {"fixup": 0, "thumb": 0}

        def fake_fixup(clips, recompute, media_type, on_progress=None, embedder=None):
            calls["fixup"] += 1

        def fake_thumb(clips, recompute, media_type):
            calls["thumb"] += 1

        monkeypatch.setattr(clipper_stage, "_fixup_clip_md5_and_embeddings", fake_fixup)
        monkeypatch.setattr(clipper_stage, "_regenerate_clip_thumbnails", fake_thumb)

        media = _make_text_media(1, "One. Two. Three.")
        clips = {1: media}

        clipper_stage._apply_clipper(clips, "text_sentence", None)

        assert len(clips) == 3
        for clip in clips.values():
            params = clip["origin"]["params"]
            assert params["clipper"] == "text_sentence"
            assert "clipper_chain" in params
        assert calls["fixup"] == 1
        assert calls["thumb"] == 1


# ---------------------------------------------------------------------------
# Phase 2b: flat (run_converters_on_folder) converter origin replay
# ---------------------------------------------------------------------------


class _FakeDoc2Image:
    """Deterministic converter stand-in: one image output per page."""

    name = "doc2img"
    source_type = "document"
    target_type = "image"

    def __init__(self, n_pages: int = 3) -> None:
        self.n_pages = n_pages

    def convert_normalized(self, media, params):
        src = media.get("media_bytes") or b""
        n = int(params.get("pages", self.n_pages))
        return [
            {"filename": f"page_{i}.png", "media_bytes": src + f"|page{i}".encode(), "duration": 0} for i in range(n)
        ]


class TestFlatConverterOriginToChain:
    def test_builds_one_step_chain_with_disambiguators(self):
        from vtscore.detectors.resolver import _converter_origin_to_chain

        params = {
            "converter": "doc2img",
            "converter_param_pages": "3",
            "converter_out_index": "2",
            "converter_n_out": "3",
            "converter_content_hash": "abc123def456",
        }
        chain = _converter_origin_to_chain(params)
        assert chain == [
            {
                "kind": "converter",
                "name": "doc2img",
                "params": {"pages": "3"},
                "out_index": 2,
                "n_out": 3,
                "content_hash": "abc123def456",
            }
        ]

    def test_returns_none_without_converter(self):
        from vtscore.detectors.resolver import _converter_origin_to_chain

        assert _converter_origin_to_chain({"path": "/data"}) is None


class TestEmbedResolvedLabelFlatConverter:
    def test_replays_converter_and_embeds_right_page(self, tmp_path, monkeypatch):
        """A flat converter origin re-runs the converter on the resolved
        source file and embeds the recorded sub-output (the rendered page),
        NOT the raw source file."""
        from vtscore.datasets.clipper_chain import _content_hash
        from vtscore.detectors import resolver as resolver_module
        from vtscore.detectors.resolver import _embed_resolved_label

        fake = _FakeDoc2Image(n_pages=3)
        monkeypatch.setattr("vtscore.converters.get_converter", lambda name: fake if name == "doc2img" else None)

        captured: dict = {}

        def fake_embed_file(path, media_type, embedder_name=""):
            captured["bytes"] = path.read_bytes()
            captured["media_type"] = media_type
            return np.ones(4, dtype=np.float32)

        monkeypatch.setattr(resolver_module, "embed_file", fake_embed_file)

        src = tmp_path / "doc.pdf"
        src.write_bytes(b"PDF")
        ch = _content_hash({"media_bytes": b"PDF|page2"})
        origin = {
            "importer": "converter",
            "params": {
                "converter": "doc2img",
                "converter_out_index": "2",
                "converter_n_out": "3",
                "converter_content_hash": ch,
                "converter_param_pages": "3",
            },
        }

        embedding = _embed_resolved_label(src, "image", origin, "")
        assert embedding is not None
        # Embedded the rendered page, not the source PDF.
        assert captured["bytes"] == b"PDF|page2"
        assert captured["media_type"] == "image"

    def test_falls_back_to_whole_file_when_replay_fails(self, tmp_path, monkeypatch):
        """If the converter is gone, _embed_resolved_label falls back to a
        whole-file embed rather than dropping the label."""
        from vtscore.detectors import resolver as resolver_module
        from vtscore.detectors.resolver import _embed_resolved_label

        monkeypatch.setattr("vtscore.converters.get_converter", lambda name: None)

        captured: dict = {}

        def fake_embed_file(path, media_type, embedder_name=""):
            captured["bytes"] = path.read_bytes()
            return np.zeros(4, dtype=np.float32)

        monkeypatch.setattr(resolver_module, "embed_file", fake_embed_file)

        src = tmp_path / "doc.pdf"
        src.write_bytes(b"RAWPDF")
        origin = {
            "importer": "converter",
            "params": {"converter": "doc2img", "converter_out_index": "0", "converter_n_out": "1"},
        }
        embedding = _embed_resolved_label(src, "image", origin, "")
        assert embedding is not None
        assert captured["bytes"] == b"RAWPDF"  # whole-file fallback
