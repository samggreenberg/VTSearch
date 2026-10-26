"""Tests for MediaCleaner: the optional 1→1 cleanup gates run before embedding.

See ``docs/plans/media-cleaners.md`` for the design.

Covers:
- The cleaner registry is separate from the clipper registry.
- ``cleaners`` field decoding and its always-last placement in the chain.
- Chain validation / apply / replay for ``kind: "cleaner"`` steps.
- A re-parameterised cleaner's values are recorded and replayed, not dropped.
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
from vtscore.media.embedder import MediaEmbedder
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


class _WindowTrimVideoCleaner(MediaCleaner):
    """Third test cleaner: cleans by **metadata**, never by rewriting a payload.

    Stands in for the shipped video gates (``video_blank_trim`` /
    ``video_letterbox_crop``) without needing a decodable video.  What the
    runner has to get right for those is that narrowing a unit's window / box
    counts as a change - it changes what gets embedded - while snapshotting
    nothing, because there is no second payload to keep.

    A unit that already carries the trimmed window is a no-op, so one dataset
    exercises both branches.
    """

    @property
    def name(self) -> str:
        return "video_test_window"

    @property
    def media_type(self) -> str:
        return "video"

    @property
    def description(self) -> str:
        return "Narrow the unit's window and pixel box."

    def clean(self, media: dict[str, Any]) -> dict[str, Any]:
        if media.get("clip_start") == 0.5:
            return media
        cleaned = dict(media)
        cleaned["clip_start"] = 0.5
        cleaned["clip_end"] = 1.5
        cleaned["duration"] = 1.0
        cleaned["clip_box"] = [0, 10, 100, 90]
        return cleaned


class _BoxVideoCleaner(MediaCleaner):
    """Fourth test cleaner: records a tighter pixel box and leaves the window alone.

    Run after :class:`_WindowTrimVideoCleaner`, the unit's window comes from one
    step and its box from another, so an embed that reads either off the wrong
    step shows up.
    """

    @property
    def name(self) -> str:
        return "video_test_box"

    @property
    def media_type(self) -> str:
        return "video"

    def clean(self, media: dict[str, Any]) -> dict[str, Any]:
        cleaned = dict(media)
        cleaned["clip_box"] = [4, 12, 96, 88]
        return cleaned


class _RecordingVideoEmbedder(MediaEmbedder):
    """A video embedder that keeps every media dict it is handed."""

    def __init__(self) -> None:
        self._model = True
        self.seen: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "video_test_recorder"

    @property
    def media_type_id(self) -> str:
        return "video"

    def _load_models_impl(self) -> None:  # pragma: no cover - model is pre-set
        self._model = True

    def _embed_media_impl(self, media: dict) -> np.ndarray:
        self.seen.append(dict(media))
        return np.ones(4, dtype=np.float32)


@pytest.fixture
def registered_cleaners():
    """Register the test cleaners for one test, then restore the registry."""
    import vtscore.media as media_registry

    before = dict(media_registry._cleaner_registry)
    media_registry.register_cleaner(_UpperTextCleaner())
    media_registry.register_cleaner(_SuffixTextCleaner())
    media_registry.register_cleaner(_WindowTrimVideoCleaner())
    media_registry.register_cleaner(_BoxVideoCleaner())
    try:
        yield
    finally:
        media_registry._cleaner_registry.clear()
        media_registry._cleaner_registry.update(before)


@pytest.fixture
def recording_video_embedder():
    """Register a :class:`_RecordingVideoEmbedder` for one test, then restore the registry."""
    import vtscore.media as media_registry

    embedder = _RecordingVideoEmbedder()
    before = dict(media_registry._embedder_registry)
    media_registry.register_embedder(embedder)
    try:
        yield embedder
    finally:
        media_registry._embedder_registry.clear()
        media_registry._embedder_registry.update(before)


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


def _make_video_media(media_id: int, **extra) -> dict:
    rng = np.random.default_rng(media_id)
    media = {
        "id": media_id,
        "media_type": "video",
        "filename": f"clip_{media_id}.mp4",
        "media_bytes": b"video-payload",
        "duration": 2.0,
        "file_size": 13,
        "md5": content_md5(b"video-payload"),
        "embeddings": {"xclip": rng.standard_normal(512).astype(np.float32)},
        "embedder": "xclip",
        "thumbnail_bytes": b"stale-thumbnail",
        "origin": {"importer": "server_folder", "params": {"path": "/data/video", "media_type": "video"}},
        "origin_name": f"clip_{media_id}.mp4",
    }
    media.update(extra)
    return media


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

        assert [c.name for c in cleaners_for_type("image")] == ["image_exif_orient", "image_edge_trim"]
        assert [c.name for c in cleaners_for_type("text")] == [
            "text_markup_strip",
            "text_whitespace",
            "text_test_upper",
            "text_test_suffix",
        ]

    def test_to_dict_carries_default_enabled(self, registered_cleaners):
        from vtscore.media import get_cleaner

        assert get_cleaner("image_edge_trim").to_dict()["default_enabled"] is False
        assert get_cleaner("text_test_suffix").to_dict()["default_enabled"] is True

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


class TestMetadataOnlyClean:
    """A cleaner may clean by narrowing metadata instead of rewriting a payload.

    Video units are metadata-only - every clip of a parent shares its bytes -
    so the video gates trim the time window and record a pixel box.  The runner
    has to count that as a change (it changes what gets embedded and what the
    thumbnail should show) while snapshotting nothing.
    """

    def test_counts_as_changed_and_needs_recompute(self, registered_cleaners):
        from vtscore.datasets.clipper_chain import apply_chain_to_clips

        clips = {1: _make_video_media(1), 2: _make_video_media(2, clip_start=0.5, clip_end=1.5)}
        result = apply_chain_to_clips(clips, [{"kind": "cleaner", "name": "video_test_window", "params": {}}])
        assert result is not None
        assert result[0] == "video"
        assert clips[1]["clip_start"] == 0.5
        assert clips[1]["clip_box"] == [0, 10, 100, 90]
        # The trimmed unit's MD5 / embedding / thumbnail all describe the old
        # window, so only it needs the fixup.
        assert result[1] == [True, False]

    def test_snapshots_nothing(self, registered_cleaners):
        """There is no second payload: the served file is untouched, and the
        player already loops within ``[clip_start, clip_end]``."""
        from vtscore.datasets.clipper_chain import ORIGINAL_PAYLOAD_KEYS, apply_chain_to_clips, has_original_payload

        clips = {1: _make_video_media(1)}
        apply_chain_to_clips(clips, [{"kind": "cleaner", "name": "video_test_window", "params": {}}])
        assert clips[1]["media_bytes"] == b"video-payload"
        assert not has_original_payload(clips[1])
        for key in ORIGINAL_PAYLOAD_KEYS:
            assert key not in clips[1]

    def test_drops_the_stale_thumbnail(self, registered_cleaners):
        """The parent's poster frame shows the old window at the old framing."""
        from vtscore.datasets.clipper_chain import apply_chain_to_clips

        clips = {1: _make_video_media(1), 2: _make_video_media(2, clip_start=0.5, clip_end=1.5)}
        apply_chain_to_clips(clips, [{"kind": "cleaner", "name": "video_test_window", "params": {}}])
        assert "thumbnail_bytes" not in clips[1]
        assert clips[2]["thumbnail_bytes"] == b"stale-thumbnail"

    def test_the_trail_records_the_new_window_and_box(self, registered_cleaners):
        from vtscore.datasets.clipper_chain import apply_chain_to_clips

        clips = {1: _make_video_media(1), 2: _make_video_media(2, clip_start=0.5, clip_end=1.5)}
        apply_chain_to_clips(clips, [{"kind": "cleaner", "name": "video_test_window", "params": {}}])

        entry = json.loads(clips[1]["origin"]["params"]["clipper_chain"])[0]
        assert entry["changed"] is True
        assert (entry["clip_start"], entry["clip_end"]) == ("0.5", "1.5")
        assert entry["clip_box"] == "0,10,100,90"

        # A gate that no-opped records no window: there is nothing it did.
        no_op = json.loads(clips[2]["origin"]["params"]["clipper_chain"])[0]
        assert no_op["changed"] is False
        assert "clip_start" not in no_op

    def test_a_metadata_clean_appears_in_derived_via(self, registered_cleaners):
        from vtscore.datasets.clipper_chain import apply_chain_to_clips
        from vtscore.media.provenance import _describe_derivation

        clips = {1: _make_video_media(1)}
        apply_chain_to_clips(clips, [{"kind": "cleaner", "name": "video_test_window", "params": {}}])
        # Rendered by display name, like every other step in the chain.
        assert _describe_derivation(clips[1]["origin"]["params"]) == "Test Window"

    def test_a_metadata_cleaned_reference_clip_stays_lazy(self, registered_cleaners):
        """Unlike a payload clean, this one costs a thin import nothing.

        ``lazy_clip`` reproduces the unit by reading the source file and
        honouring the window / box, which is exactly what the cleaner recorded -
        so there is no reason to hold the bytes.
        """
        from vtscore.datasets.clipper_chain import apply_chain_to_clips
        from vtscore.datasets.stages.clipper import _relazify_reference_clips_stage
        from vtscore.state import DatasetContext

        clips = {1: _make_video_media(1, _lazy_source="/data/video/clip_1.mp4")}
        apply_chain_to_clips(clips, [{"kind": "cleaner", "name": "video_test_window", "params": {}}])
        ctx = DatasetContext("metadata-clean-relazify-test")
        ctx.medias.update(clips)

        _relazify_reference_clips_stage(ctx)

        assert clips[1]["media_bytes"] is None
        assert clips[1]["media_path"] == "/data/video/clip_1.mp4"
        assert clips[1]["clip_start"] == 0.5


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

    def test_replay_embeds_a_video_unit_with_its_window_and_box(
        self, registered_cleaners, recording_video_embedder, tmp_path, monkeypatch
    ):
        """A video label re-embeds the slice the dataset embedded (#4994).

        Video cleaners clean by metadata, so the unit is the source file plus a
        window and a box.  The replay used to copy the file to a tempfile and
        embed it bare, which sampled the whole video, uncropped.
        """
        from vtscore.datasets import clipper_chain
        from vtscore.datasets.stages.clipper import _build_clip_embed_input

        monkeypatch.setattr("vtscore.media.video.decode.probe", lambda path: None)
        src = tmp_path / "clip_1.mp4"
        src.write_bytes(b"video-payload")

        clips = {1: _make_video_media(1)}
        steps = [
            {"kind": "cleaner", "name": "video_test_window", "params": {}},
            {"kind": "cleaner", "name": "video_test_box", "params": {}},
        ]
        clipper_chain.apply_chain_to_clips(clips, steps)
        dataset_input = _build_clip_embed_input(clips[1], "video")
        trail = clipper_chain.parse_trail(clips[1]["origin"]["params"]["clipper_chain"])
        assert trail is not None

        result = clipper_chain.replay_chain_on_file(src, trail, recording_video_embedder.name)
        assert result is not None
        assert result[1] is None  # still metadata-only: no clip bytes to hash

        [seen] = recording_video_embedder.seen
        assert seen["media_path"] == str(src.resolve())
        assert (seen["clip_start"], seen["clip_end"], seen["clip_box"]) == (0.5, 1.5, [4, 12, 96, 88])
        unit_keys = ("clip_start", "clip_end", "clip_box")
        assert {k: seen[k] for k in unit_keys} == {k: dataset_input[k] for k in unit_keys}


#: Every shipped cleaner that takes parameters, with a value for each one that
#: differs from its default.  At the defaults the round trip below would pass
#: even if the params were dropped, so only off-default values prove anything.
_NON_DEFAULT_CLEANER_PARAMS: dict[str, dict[str, Any]] = {
    "audio_loudness_normalize": {"mode": "peak", "rms_db": -16.0, "peak_db": -3.0},
    "audio_silence_trim": {"top_db": 25.0, "pad": 0.2},
    "image_edge_trim": {"edge_tol": 24.0, "max_edge_trim": 0.3, "min_edge_trim": 0.05},
    "video_blank_trim": {"blank_ratio": 0.95, "max_trim": 0.1, "step": 0.5},
    "video_letterbox_crop": {"samples": 3, "edge_tol": 24.0, "max_edge_trim": 0.3, "min_edge_trim": 0.05},
}


class TestCleanerParamsRoundTrip:
    """A cleaner the user re-parameterised replays at *their* values (#4963).

    The chain records a step's params from ``to_dict()``, so a cleaner whose
    ``to_dict()`` emitted none recorded ``{}`` and every replay ran it at its
    defaults.
    """

    def test_every_declared_parameter_is_recorded(self):
        """The contract behind the bug, checked across both registries."""
        from vtscore.datasets.clipper_chain import _resolved_clipper_params
        from vtscore.media import all_cleaners, all_clippers

        unrecorded: dict[str, list[str]] = {}
        for plugin in [*all_clippers(), *all_cleaners()]:
            declared = {p["key"] for p in plugin.parameters}
            missing = declared - set(_resolved_clipper_params(plugin))
            if missing:
                unrecorded[plugin.name] = sorted(missing)
        assert unrecorded == {}

    def test_the_table_covers_every_parameterised_cleaner(self):
        from vtscore.media import all_cleaners

        assert {c.name for c in all_cleaners() if c.parameters} == set(_NON_DEFAULT_CLEANER_PARAMS)

    @pytest.mark.parametrize("name", sorted(_NON_DEFAULT_CLEANER_PARAMS))
    def test_user_params_survive_record_and_replay(self, name, tmp_path, monkeypatch):
        from vtscore.datasets import clipper_chain
        from vtscore.media import get_cleaner

        cleaner = get_cleaner(name)
        chosen = _NON_DEFAULT_CLEANER_PARAMS[name]
        assert all(getattr(cleaner, k) != v for k, v in chosen.items()), "pick off-default values"

        # Note what each run of the gate was configured with, read from its
        # own properties rather than from ``to_dict()``, and leave the payload
        # alone so no decodable media is needed.
        ran_with: list[dict[str, Any]] = []

        def spy_clean(self, media):
            ran_with.append({k: getattr(self, k) for k in chosen})
            return media

        monkeypatch.setattr(type(cleaner), "clean", spy_clean)
        monkeypatch.setattr(
            "vtscore.detectors.resolver.embed_file",
            lambda path, media_type, embedder_name="", **_: np.ones(4, dtype=np.float32),
        )

        src = tmp_path / "source.bin"
        src.write_bytes(b"payload")
        clips = {
            1: {
                "id": 1,
                "media_type": cleaner.media_type,
                "filename": src.name,
                "media_bytes": b"payload",
                "origin": {"importer": "server_folder", "params": {}},
                "origin_name": src.name,
            }
        }
        clipper_chain.apply_chain_to_clips(clips, [{"kind": "cleaner", "name": name, "params": chosen}])
        trail = clipper_chain.parse_trail(clips[1]["origin"]["params"]["clipper_chain"])
        assert trail is not None
        assert trail[0]["params"] == chosen

        assert clipper_chain.replay_chain_on_file(src, trail) is not None
        assert ran_with == [chosen, chosen]  # the import, then the replay


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

    def test_cleaned_reference_media_keeps_only_a_recipe(self, registered_cleaners, tmp_path):
        """A cleaned item goes back to a reference once its trail, replayed from
        the source, rebuilds both the cleaned payload and the Original; it then
        serves the *cleaned* text, never the source file it points at."""
        import vtscore.media as media_registry
        from vtscore.datasets.clipper_chain import apply_chain_to_clips, has_original_payload
        from vtscore.datasets.stages.clipper import _relazify_reference_clips_stage
        from vtscore.state import DatasetContext

        src = tmp_path / "doc_1.txt"
        src.write_text("hello", encoding="utf-8")
        clips = {1: _make_text_media(1, "hello")}
        apply_chain_to_clips(clips, [{"kind": "cleaner", "name": "text_test_upper", "params": {}}])
        cleaned = clips[1]
        cleaned["_lazy_source"] = str(src)
        ctx = DatasetContext("cleaner-relazify-test")
        ctx.medias.update(clips)

        _relazify_reference_clips_stage(ctx)

        assert "_lazy_source" not in cleaned
        assert cleaned["media_string"] is None
        assert "original_media_string" not in cleaned
        assert has_original_payload(cleaned)
        assert media_registry.get("text")._resolve_media_string(cleaned) == "HELLO"

    def test_cleaned_reference_media_keeps_its_bytes_when_the_replay_cannot_run(self, registered_cleaners):
        """With no readable source the dry run cannot rebuild the payload, and
        dropping it would lose the only copy - so the item keeps both."""
        from vtscore.datasets.clipper_chain import apply_chain_to_clips
        from vtscore.datasets.stages.clipper import _relazify_reference_clips_stage
        from vtscore.state import DatasetContext

        clips = {1: _make_text_media(1, "hello")}
        apply_chain_to_clips(clips, [{"kind": "cleaner", "name": "text_test_upper", "params": {}}])
        cleaned = clips[1]
        cleaned["_lazy_source"] = "/data/text/doc_1.txt"
        ctx = DatasetContext("cleaner-relazify-test")
        ctx.medias.update(clips)

        _relazify_reference_clips_stage(ctx)

        assert "_lazy_source" not in cleaned
        assert cleaned["media_string"] == "HELLO"
        assert cleaned["original_media_string"] == "hello"


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
    def test_defaults_off(self):
        """Opt-in: every decode already applies the tag, so baking it into the
        stored bytes buys nothing inside VTSearch and costs a lossy re-encode."""
        from vtscore.media import get_cleaner

        assert get_cleaner("image_exif_orient").default_enabled is False

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
