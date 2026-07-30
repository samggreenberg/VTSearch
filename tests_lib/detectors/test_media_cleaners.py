"""Tests for MediaCleaner: the optional 1→1 cleanup gates run before embedding.

See ``docs/plans/media-cleaners.md`` for the design.

Covers:
- The cleaner registry is separate from the clipper registry.
- ``cleaners`` field decoding and its always-last placement in the chain.
- Chain validation / apply / replay for ``kind: "cleaner"`` steps.
- The dual payload: ``original_*`` is stamped only on a real change, only
  once across several cleaners, and survives the pickle round-trip.
- A cleaned unit is flagged for MD5 / embedding / thumbnail recomputation.
- Provenance hides cleaner steps that no-opped on the item.
- The shipped ``image_exif_orient`` cleaner.
"""

from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
import pytest
from vtscore.media.cleaner import MediaCleaner
from vtscore.utils.hashing import content_md5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _UpperTextCleaner(MediaCleaner):
    """Test cleaner: upper-cases text that has any lowercase to remove.

    Deliberately a no-op on already-upper-case text so a single dataset
    exercises both the "changed" and "unchanged" branches of the runner.
    """

    @property
    def name(self) -> str:
        return "text_test_upper"

    @property
    def media_type(self) -> str:
        return "text"

    @property
    def description(self) -> str:
        return "Upper-case the text."

    def clean(self, media: dict[str, Any]) -> dict[str, Any]:
        text = media.get("media_string")
        if not isinstance(text, str) or text == text.upper():
            return media
        cleaned = dict(media)
        cleaned["media_string"] = text.upper()
        cleaned["character_count"] = len(text)
        return cleaned


class _SuffixTextCleaner(MediaCleaner):
    """Second test cleaner, so ordering / single-snapshot behaviour is testable."""

    @property
    def name(self) -> str:
        return "text_test_suffix"

    @property
    def media_type(self) -> str:
        return "text"

    @property
    def default_enabled(self) -> bool:
        return True

    def clean(self, media: dict[str, Any]) -> dict[str, Any]:
        text = media.get("media_string")
        if not isinstance(text, str):
            return media
        cleaned = dict(media)
        cleaned["media_string"] = text + "!"
        return cleaned


@pytest.fixture
def registered_cleaners():
    """Register the test cleaners for one test, then restore the registry."""
    import vtscore.media as media_registry

    before = dict(media_registry._cleaner_registry)
    media_registry.register_cleaner(_UpperTextCleaner())
    media_registry.register_cleaner(_SuffixTextCleaner())
    try:
        yield
    finally:
        media_registry._cleaner_registry.clear()
        media_registry._cleaner_registry.update(before)


def _make_text_media(media_id: int, text: str) -> dict:
    rng = np.random.default_rng(media_id)
    return {
        "id": media_id,
        "media_type": "text",
        "filename": f"doc_{media_id}.txt",
        "media_string": text,
        "duration": 0,
        "file_size": len(text),
        "word_count": len(text.split()),
        "character_count": len(text),
        "md5": content_md5(text.encode("utf-8")),
        "embeddings": {"e5": rng.standard_normal(384).astype(np.float32)},
        "embedder": "e5",
        "thumbnail_bytes": b"stale-thumbnail",
        "origin": {"importer": "server_folder", "params": {"path": "/data/text", "media_type": "text"}},
        "origin_name": f"doc_{media_id}.txt",
    }


def _exif_jpeg(orientation: int, size: tuple[int, int] = (40, 20)) -> bytes:
    """Encode a solid JPEG carrying *orientation* in its EXIF block."""
    from PIL import Image

    img = Image.new("RGB", size, "red")
    exif = img.getexif()
    exif[274] = orientation
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestCleanerRegistry:
    def test_cleaners_are_not_clippers(self):
        """A cleaner must never surface in a clipper chooser."""
        from vtscore.media import all_clippers, get_cleaner, get_clipper

        assert get_cleaner("image_exif_orient").name == "image_exif_orient"
        assert "image_exif_orient" not in {c.name for c in all_clippers()}
        with pytest.raises(KeyError):
            get_clipper("image_exif_orient")

    def test_unknown_cleaner_raises(self):
        from vtscore.media import get_cleaner

        with pytest.raises(KeyError, match="Unknown cleaner"):
            get_cleaner("nope")

    def test_cleaners_for_type_filters(self, registered_cleaners):
        from vtscore.media import cleaners_for_type

        assert [c.name for c in cleaners_for_type("image")] == ["image_exif_orient"]
        assert [c.name for c in cleaners_for_type("text")] == ["text_test_upper", "text_test_suffix"]

    def test_to_dict_carries_default_enabled(self, registered_cleaners):
        from vtscore.media import get_cleaner

        assert get_cleaner("image_exif_orient").to_dict()["default_enabled"] is True
        assert get_cleaner("text_test_upper").to_dict()["default_enabled"] is False

    def test_clip_wraps_clean_as_one_output(self, registered_cleaners):
        from vtscore.media import get_cleaner

        out = get_cleaner("text_test_upper").clip(_make_text_media(1, "abc"))
        assert len(out) == 1
        assert out[0]["media_string"] == "ABC"


# ---------------------------------------------------------------------------
# Field decoding and chain placement
# ---------------------------------------------------------------------------


class TestParseCleanerField:
    def test_none_and_empty(self):
        from vtscore.datasets.clipper_chain import parse_cleaner_field

        assert parse_cleaner_field(None) == []
        assert parse_cleaner_field("") == []
        assert parse_cleaner_field([]) == []

    def test_name_list(self):
        from vtscore.datasets.clipper_chain import parse_cleaner_field

        assert parse_cleaner_field(["image_exif_orient"]) == [
            {"kind": "cleaner", "name": "image_exif_orient", "params": {}}
        ]

    def test_dict_list_with_params(self):
        from vtscore.datasets.clipper_chain import parse_cleaner_field

        assert parse_cleaner_field([{"name": "a", "params": {"tol": 8}}]) == [
            {"kind": "cleaner", "name": "a", "params": {"tol": 8}}
        ]

    def test_json_string(self):
        from vtscore.datasets.clipper_chain import parse_cleaner_field

        raw = json.dumps([{"name": "a", "params": {}}, "b"])
        assert [s["name"] for s in parse_cleaner_field(raw)] == ["a", "b"]

    def test_deduplicates_and_skips_blanks(self):
        from vtscore.datasets.clipper_chain import parse_cleaner_field

        assert [s["name"] for s in parse_cleaner_field(["a", "a", "", {"name": ""}, 7, "b"])] == ["a", "b"]

    def test_malformed_json_is_empty_not_fatal(self):
        from vtscore.datasets.clipper_chain import parse_cleaner_field

        assert parse_cleaner_field("{not json") == []


class TestAppendCleanerSteps:
    def test_no_cleaners_leaves_chain_identical(self):
        from vtscore.datasets.clipper_chain import append_cleaner_steps

        assert append_cleaner_steps(None, None) is None
        steps = [{"kind": "clipper", "name": "text_sentence", "params": {}}]
        assert append_cleaner_steps(steps, None) is steps

    def test_cleaners_go_last(self):
        from vtscore.datasets.clipper_chain import append_cleaner_steps

        steps = [
            {"kind": "converter", "name": "document2text", "params": {}},
            {"kind": "clipper", "name": "text_sentence", "params": {}},
        ]
        out = append_cleaner_steps(steps, ["a"])
        assert out is not None
        assert [s["kind"] for s in out] == ["converter", "clipper", "cleaner"]

    def test_cleaner_only_chain_from_empty(self):
        from vtscore.datasets.clipper_chain import append_cleaner_steps

        out = append_cleaner_steps(None, ["a"])
        assert out == [{"kind": "cleaner", "name": "a", "params": {}}]


class TestChainValidationWithCleaners:
    def test_normalise_keeps_cleaner_steps(self, registered_cleaners):
        from vtscore.datasets.clipper_chain import normalise_chain

        out = normalise_chain(
            [
                {"kind": "clipper", "name": "text_default", "params": {}},
                {"kind": "cleaner", "name": "text_test_upper", "params": {}},
            ]
        )
        assert out == [{"kind": "cleaner", "name": "text_test_upper", "params": {}}]

    def test_cleaner_only_chain_keeps_type(self, registered_cleaners):
        from vtscore.datasets.clipper_chain import validate_chain

        steps = [{"kind": "cleaner", "name": "text_test_upper", "params": {}}]
        assert validate_chain(steps, "text") == "text"

    def test_rejects_unknown_cleaner(self):
        from vtscore.datasets.clipper_chain import validate_chain

        with pytest.raises(ValueError, match="unknown cleaner"):
            validate_chain([{"kind": "cleaner", "name": "nope", "params": {}}], "text")

    def test_rejects_type_mismatch(self, registered_cleaners):
        from vtscore.datasets.clipper_chain import validate_chain

        steps = [{"kind": "cleaner", "name": "text_test_upper", "params": {}}]
        with pytest.raises(ValueError, match="expects input type 'text'"):
            validate_chain(steps, "image")

    def test_parse_trail_accepts_cleaner_kind(self):
        from vtscore.datasets.clipper_chain import parse_trail

        raw = json.dumps([{"kind": "cleaner", "name": "a", "params": {}}])
        assert parse_trail(raw) == [{"kind": "cleaner", "name": "a", "params": {}}]

    def test_parse_trail_rejects_unknown_kind(self):
        from vtscore.datasets.clipper_chain import parse_trail

        assert parse_trail(json.dumps([{"kind": "scrubber", "name": "a"}])) is None


# ---------------------------------------------------------------------------
# Apply chain: dual payload + recompute flags
# ---------------------------------------------------------------------------


class TestApplyChainWithCleaner:
    def test_changed_item_snapshots_original_unchanged_does_not(self, registered_cleaners):
        from vtscore.datasets.clipper_chain import apply_chain_to_clips, has_original_payload

        clips = {1: _make_text_media(1, "hello world"), 2: _make_text_media(2, "ALREADY UPPER")}
        result = apply_chain_to_clips(clips, [{"kind": "cleaner", "name": "text_test_upper", "params": {}}])
        assert result is not None
        final_type, needs_recompute = result
        assert final_type == "text"
        # One output per input: a cleaner never splits.
        assert len(clips) == 2

        changed, unchanged = clips[1], clips[2]
        assert changed["media_string"] == "HELLO WORLD"
        assert changed["original_media_string"] == "hello world"
        assert has_original_payload(changed)

        assert unchanged["media_string"] == "ALREADY UPPER"
        assert not has_original_payload(unchanged)
        assert "original_media_string" not in unchanged

        # Only the rewritten item needs its MD5 / embedding / thumbnail redone.
        assert needs_recompute == [True, False]

    def test_changed_item_drops_stale_thumbnail(self, registered_cleaners):
        from vtscore.datasets.clipper_chain import apply_chain_to_clips

        clips = {1: _make_text_media(1, "hello"), 2: _make_text_media(2, "UPPER")}
        apply_chain_to_clips(clips, [{"kind": "cleaner", "name": "text_test_upper", "params": {}}])
        assert "thumbnail_bytes" not in clips[1]
        assert clips[2]["thumbnail_bytes"] == b"stale-thumbnail"

    def test_trail_records_name_params_change_and_hash(self, registered_cleaners):
        from vtscore.datasets.clipper_chain import _content_hash, apply_chain_to_clips

        clips = {1: _make_text_media(1, "hello")}
        apply_chain_to_clips(clips, [{"kind": "cleaner", "name": "text_test_upper", "params": {}}])
        trail = json.loads(clips[1]["origin"]["params"]["clipper_chain"])
        assert len(trail) == 1
        entry = trail[0]
        assert entry["kind"] == "cleaner"
        assert entry["name"] == "text_test_upper"
        assert entry["n_out"] == 1
        assert entry["out_index"] == 0
        assert entry["changed"] is True
        assert entry["content_hash"] == _content_hash(clips[1])

    def test_cleaner_stamps_no_legacy_clipper_keys(self, registered_cleaners):
        """A cleaner has one output, so there is no sibling to disambiguate."""
        from vtscore.datasets.clipper_chain import apply_chain_to_clips

        clips = {1: _make_text_media(1, "hello")}
        apply_chain_to_clips(clips, [{"kind": "cleaner", "name": "text_test_upper", "params": {}}])
        params = clips[1]["origin"]["params"]
        assert "clipper" not in params
        assert "clip_index" not in params

    def test_snapshot_taken_once_across_two_cleaners(self, registered_cleaners):
        """``original_*`` is the pre-*any*-clean payload, not the previous gate's."""
        from vtscore.datasets.clipper_chain import apply_chain_to_clips

        clips = {1: _make_text_media(1, "hello")}
        apply_chain_to_clips(
            clips,
            [
                {"kind": "cleaner", "name": "text_test_upper", "params": {}},
                {"kind": "cleaner", "name": "text_test_suffix", "params": {}},
            ],
        )
        assert clips[1]["media_string"] == "HELLO!"
        assert clips[1]["original_media_string"] == "hello"

    def test_snapshot_survives_a_leading_no_op_cleaner(self, registered_cleaners):
        """The first *mutating* cleaner takes the snapshot, not the first step."""
        from vtscore.datasets.clipper_chain import apply_chain_to_clips

        clips = {1: _make_text_media(1, "UPPER")}
        apply_chain_to_clips(
            clips,
            [
                {"kind": "cleaner", "name": "text_test_upper", "params": {}},
                {"kind": "cleaner", "name": "text_test_suffix", "params": {}},
            ],
        )
        assert clips[1]["media_string"] == "UPPER!"
        assert clips[1]["original_media_string"] == "UPPER"

    def test_clipper_then_cleaner_cleans_each_sub_clip(self, registered_cleaners):
        """Cleaners run on the finished units, so each sub-clip gets its own
        Original rather than sharing the parent's."""
        from vtscore.datasets.clipper_chain import apply_chain_to_clips

        clips = {1: _make_text_media(1, "first one. SECOND ONE.")}
        result = apply_chain_to_clips(
            clips,
            [
                {"kind": "clipper", "name": "text_sentence", "params": {}},
                {"kind": "cleaner", "name": "text_test_upper", "params": {}},
            ],
        )
        assert result is not None
        assert len(clips) == 2
        first, second = clips[1], clips[2]
        assert first["media_string"] == "FIRST ONE."
        assert first["original_media_string"] == "first one."
        # The second sentence was already upper-case, so nothing was snapshotted.
        assert "original_media_string" not in second
        # Both are still sub-items of a multi-output clipper, so both recompute.
        assert result[1] == [True, True]

    def test_progress_reports_a_cleaning_phase(self, registered_cleaners):
        from vtscore.datasets.clipper_chain import apply_chain_to_clips

        phases: list[str] = []
        apply_chain_to_clips(
            {1: _make_text_media(1, "hello")},
            [{"kind": "cleaner", "name": "text_test_upper", "params": {}}],
            on_progress=lambda cur, total, phase: phases.append(phase),
        )
        assert phases == ["cleaning"]


class TestResolvedParamsBaseKeys:
    def test_descriptor_metadata_is_not_a_parameter(self, registered_cleaners):
        """``summary_template`` / ``default_enabled`` describe the plugin; they
        must not leak into a step's effective parameters (and from there into
        ``origin.params`` as bogus ``clipper_<key>`` entries)."""
        from vtscore.datasets.clipper_chain import _resolved_clipper_params
        from vtscore.media import get_cleaner, get_clipper

        assert _resolved_clipper_params(get_cleaner("text_test_upper")) == {}
        video_params = _resolved_clipper_params(get_clipper("video_tiling"))
        assert "summary_template" not in video_params
        assert "duration" in video_params


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class TestCleanerProvenance:
    def test_changed_cleaner_appears_in_derived_via(self, registered_cleaners):
        from vtscore.datasets.clipper_chain import apply_chain_to_clips
        from vtscore.media.provenance import DERIVED_VIA_LABEL, provenance_metadata

        clips = {1: _make_text_media(1, "hello")}
        apply_chain_to_clips(clips, [{"kind": "cleaner", "name": "text_test_upper", "params": {}}])
        assert provenance_metadata(clips[1])[DERIVED_VIA_LABEL] == "Test Upper"

    def test_no_op_cleaner_is_hidden(self, registered_cleaners):
        """A gate that left the item alone is not how the item was made."""
        from vtscore.datasets.clipper_chain import apply_chain_to_clips
        from vtscore.media.provenance import DERIVED_VIA_LABEL, SOURCE_LABEL, provenance_metadata

        clips = {1: _make_text_media(1, "UPPER")}
        apply_chain_to_clips(clips, [{"kind": "cleaner", "name": "text_test_upper", "params": {}}])
        meta = provenance_metadata(clips[1])
        assert DERIVED_VIA_LABEL not in meta
        assert SOURCE_LABEL not in meta


# ---------------------------------------------------------------------------
# Replay (cross-dataset resolver path)
# ---------------------------------------------------------------------------


class TestCleanerReplay:
    def test_replay_runs_the_cleaner_on_the_source_file(self, registered_cleaners, tmp_path, monkeypatch):
        """A label's replay must embed the *cleaned* bytes, matching what the
        dataset embedded — not the untouched source file."""
        from vtscore.datasets import clipper_chain

        src = tmp_path / "doc.txt"
        src.write_text("hello world", encoding="utf-8")

        embedded: list[str] = []

        def fake_embed_file(path, media_type, embedder_name=""):
            embedded.append(path.read_text(encoding="utf-8"))
            return np.ones(4, dtype=np.float32)

        monkeypatch.setattr("vtscore.detectors.resolver.embed_file", fake_embed_file)

        clips = {1: _make_text_media(1, "hello world")}
        clipper_chain.apply_chain_to_clips(clips, [{"kind": "cleaner", "name": "text_test_upper", "params": {}}])
        trail = clipper_chain.parse_trail(clips[1]["origin"]["params"]["clipper_chain"])
        assert trail is not None

        result = clipper_chain.replay_chain_on_file(src, trail)
        assert result is not None
        _embedding, content = result
        assert embedded == ["HELLO WORLD"]
        assert content == b"HELLO WORLD"

    def test_replay_survives_source_drift(self, registered_cleaners, tmp_path, monkeypatch):
        """A cleaner has exactly one output, so a content-hash mismatch is no
        reason to refuse: re-running the gate beats embedding the raw file."""
        from vtscore.datasets import clipper_chain

        monkeypatch.setattr(
            "vtscore.detectors.resolver.embed_file",
            lambda path, media_type, embedder_name="": np.ones(4, dtype=np.float32),
        )
        clips = {1: _make_text_media(1, "hello world")}
        clipper_chain.apply_chain_to_clips(clips, [{"kind": "cleaner", "name": "text_test_upper", "params": {}}])
        trail = clipper_chain.parse_trail(clips[1]["origin"]["params"]["clipper_chain"])
        assert trail is not None

        drifted = tmp_path / "doc.txt"
        drifted.write_text("hello world, edited", encoding="utf-8")
        result = clipper_chain.replay_chain_on_file(drifted, trail)
        assert result is not None
        assert result[1] == b"HELLO WORLD, EDITED"


# ---------------------------------------------------------------------------
# Load stage: legacy clipper folding + re-lazify guard
# ---------------------------------------------------------------------------


class TestApplyClipperStageWithCleaners:
    def test_legacy_clipper_leads_a_cleaner_only_chain(self, registered_cleaners, monkeypatch):
        """An import that picks a real clipper *and* cleanup gates must run
        both — the clipper first, then the gates on its output."""
        from vtscore.datasets.stages.clipper import _apply_clipper

        monkeypatch.setattr(
            "vtscore.datasets.stages.clipper._fixup_clip_md5_and_embeddings",
            lambda *a, **k: None,
        )
        clips = {1: _make_text_media(1, "first one. second one.")}
        _apply_clipper(
            clips,
            "text_sentence",
            None,
            chain_steps=[{"kind": "cleaner", "name": "text_test_upper", "params": {}}],
        )
        assert [c["media_string"] for c in clips.values()] == ["FIRST ONE.", "SECOND ONE."]
        trail = json.loads(clips[1]["origin"]["params"]["clipper_chain"])
        assert [s["kind"] for s in trail] == ["clipper", "cleaner"]

    def test_default_clipper_still_stamped_alongside_cleaners(self, registered_cleaners, monkeypatch):
        from vtscore.datasets.stages.clipper import _apply_clipper

        monkeypatch.setattr(
            "vtscore.datasets.stages.clipper._fixup_clip_md5_and_embeddings",
            lambda *a, **k: None,
        )
        clips = {1: _make_text_media(1, "hello")}
        _apply_clipper(
            clips,
            "text_default",
            None,
            chain_steps=[{"kind": "cleaner", "name": "text_test_upper", "params": {}}],
        )
        params = clips[1]["origin"]["params"]
        assert params["clipper"] == "text_default"
        assert clips[1]["media_string"] == "HELLO"

    def test_cleaned_reference_media_keeps_its_bytes(self, registered_cleaners):
        """Re-lazifying a cleaned item would serve and re-embed the *uncleaned*
        source: lazy_clip has no recipe that reproduces a cleaner's output."""
        from vtscore.datasets.stages.clipper import _relazify_reference_clips_stage
        from vtscore.state import DatasetContext

        ctx = DatasetContext("cleaner-relazify-test")
        cleaned = _make_text_media(1, "HELLO")
        cleaned["original_media_string"] = "hello"
        cleaned["_lazy_source"] = "/data/text/doc_1.txt"
        plain = _make_text_media(2, "world")
        plain["_lazy_source"] = "/data/text/doc_2.txt"
        ctx.medias.update({1: cleaned, 2: plain})

        _relazify_reference_clips_stage(ctx)

        assert cleaned["media_string"] == "HELLO"
        assert "_lazy_source" not in cleaned
        assert plain["media_string"] is None
        assert plain["media_path"] == "/data/text/doc_2.txt"


# ---------------------------------------------------------------------------
# Pickle round-trip
# ---------------------------------------------------------------------------


class TestOriginalPayloadPersistence:
    def test_original_payload_survives_pickle_round_trip(self, registered_cleaners, tmp_path):
        """The pre-clean payload is dataset content, not a cache: it is the only
        copy of what the user imported."""
        from vtscore.datasets.clipper_chain import apply_chain_to_clips, has_original_payload
        from vtscore.datasets.loader import export_dataset_to_file
        from vtscore.datasets.loader_pickle import load_dataset_from_pickle

        clips = {1: _make_text_media(1, "hello world"), 2: _make_text_media(2, "UPPER")}
        apply_chain_to_clips(clips, [{"kind": "cleaner", "name": "text_test_upper", "params": {}}])

        out = tmp_path / "ds.pkl"
        out.write_bytes(export_dataset_to_file(clips, embedder="e5", media_type="text"))

        reloaded: dict[int, dict] = {}
        load_dataset_from_pickle(out, reloaded)
        by_text = {m["media_string"]: m for m in reloaded.values()}

        assert by_text["HELLO WORLD"]["original_media_string"] == "hello world"
        assert has_original_payload(by_text["HELLO WORLD"])
        assert not has_original_payload(by_text["UPPER"])
        assert "original_media_string" not in by_text["UPPER"]


# ---------------------------------------------------------------------------
# The shipped image cleaner
# ---------------------------------------------------------------------------


class TestImageExifOrientCleaner:
    def test_defaults_on(self):
        from vtscore.media import get_cleaner

        assert get_cleaner("image_exif_orient").default_enabled is True

    def test_rotates_and_clears_the_tag(self):
        from PIL import Image
        from vtscore.media import get_cleaner

        data = _exif_jpeg(6)
        media = {"media_type": "image", "media_bytes": data, "width": 40, "height": 20, "file_size": len(data)}
        out = get_cleaner("image_exif_orient").clean(media)

        assert out is not media
        # 90-degree rotation swaps the reported dimensions.
        assert (out["width"], out["height"]) == (20, 40)
        assert out["file_size"] == len(out["media_bytes"])
        with Image.open(io.BytesIO(out["media_bytes"])) as img:
            assert img.size == (20, 40)
            # Re-encoding must not leave the tag behind, or a viewer that honours
            # it would rotate the already-rotated pixels a second time.
            assert img.getexif().get(274) is None

    def test_upright_photo_is_a_no_op(self):
        from vtscore.media import get_cleaner

        media = {"media_type": "image", "media_bytes": _exif_jpeg(1)}
        assert get_cleaner("image_exif_orient").clean(media) is media

    def test_no_exif_is_a_no_op(self):
        from PIL import Image
        from vtscore.media import get_cleaner

        buf = io.BytesIO()
        Image.new("RGB", (8, 8), "blue").save(buf, format="PNG")
        media = {"media_type": "image", "media_bytes": buf.getvalue()}
        assert get_cleaner("image_exif_orient").clean(media) is media

    def test_undecodable_payload_is_a_no_op(self):
        from vtscore.media import get_cleaner

        cleaner = get_cleaner("image_exif_orient")
        assert cleaner.clean({"media_type": "image", "media_bytes": b"<svg/>"})["media_bytes"] == b"<svg/>"
        assert cleaner.clean({"media_type": "image"}) == {"media_type": "image"}
        assert cleaner.clean({"media_type": "image", "media_bytes": b""})["media_bytes"] == b""
