"""Tests for the shipped roster of :class:`~vtscore.media.cleaner.MediaCleaner`s.

The framework itself (registry, chain placement, dual payload, replay) is
covered by ``test_media_cleaners.py``; this file exercises what each concrete
gate actually does to a payload.  Every cleaner shares one contract, so every
class below checks the same three things:

- it removes what it claims to remove,
- it returns the media object **itself** when there is nothing to remove (that
  identity is what tells the chain runner to skip the ``original_*`` snapshot),
- a degenerate or undecodable payload is a no-op, never an error.

The two *video* gates add a fourth thing to check: they clean by narrowing
metadata (the unit's time window and pixel box) rather than by rewriting a
payload, so every test also asserts the payload came through untouched.

Covered: ``image_edge_trim``, ``audio_silence_trim``, ``text_whitespace``,
``text_markup_strip``, ``video_letterbox_crop``, ``video_blank_trim``.
"""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from vtscore.media import get_cleaner
from vtscore.media.video import decode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _padded(w: int, h: int, box: tuple[int, int, int, int], bg, fg) -> Image.Image:
    """A *bg*-filled frame with an *fg* rectangle painted at *box*."""
    img = Image.new("RGB", (w, h), color=bg)
    x0, y0, x1, y1 = box
    img.paste(Image.new("RGB", (x1 - x0, y1 - y0), color=fg), (x0, y0))
    return img


def _image_media(img: Image.Image, fmt: str = "PNG") -> dict:
    data = _encode(img, fmt)
    return {
        "id": 1,
        "media_type": "image",
        "media_bytes": data,
        "width": img.size[0],
        "height": img.size[1],
        "file_size": len(data),
    }


def _concat_wavs(*wavs: bytes) -> bytes:
    """Concatenate WAV byte strings sharing a sample rate/width/channel count."""
    frames: list[bytes] = []
    params = None
    for wb in wavs:
        with wave.open(io.BytesIO(wb), "rb") as wf:
            if params is None:
                params = wf.getparams()
            frames.append(wf.readframes(wf.getnframes()))
    assert params is not None
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setparams(params)
        out.writeframes(b"".join(frames))
    return buf.getvalue()


def _tone(duration: float, frequency: float = 440.0) -> bytes:
    from vtscore.media.audio.audio_generator import generate_wav

    return generate_wav(frequency, duration)


def _audio_media(wav: bytes, duration: float) -> dict:
    return {
        "id": 1,
        "media_type": "audio",
        "media_bytes": wav,
        "duration": duration,
        "file_size": len(wav),
    }


def _text_media(text: str) -> dict:
    return {
        "id": 1,
        "media_type": "text",
        "media_string": text,
        "word_count": len(text.split()),
        "character_count": len(text),
        "file_size": len(text.encode("utf-8")),
    }


def _wav_duration(wav: bytes) -> float:
    with wave.open(io.BytesIO(wav), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


# ---------------------------------------------------------------------------
# image_edge_trim
# ---------------------------------------------------------------------------


class TestImageEdgeTrimCleaner:
    def test_identity(self):
        cleaner = get_cleaner("image_edge_trim")
        assert cleaner.media_type == "image"
        assert cleaner.display_name == "Edge Trim"
        # Cropping is a judgment call about what counts as padding, so the box
        # is unchecked until the user asks for it.
        assert cleaner.default_enabled is False

    def test_crops_black_letterbox_bars(self):
        # 100 px black bars top and bottom, red content across the middle 200 px.
        media = _image_media(_padded(400, 400, (0, 100, 400, 300), (0, 0, 0), (200, 30, 30)))
        out = get_cleaner("image_edge_trim").clean(media)

        assert out is not media
        assert out["width"] == 400  # no side bars to lose
        assert abs(out["height"] - 200) <= 6
        assert out["file_size"] == len(out["media_bytes"])
        with Image.open(io.BytesIO(out["media_bytes"])) as img:
            assert img.size == (out["width"], out["height"])

    def test_crops_a_single_padded_edge(self):
        # White margin only on the left; the other three edges are content.
        media = _image_media(_padded(400, 200, (120, 0, 400, 200), (255, 255, 255), (10, 120, 40)))
        out = get_cleaner("image_edge_trim").clean(media)

        assert out["height"] == 200
        assert abs(out["width"] - 280) <= 6

    def test_content_filled_frame_is_a_no_op(self):
        media = _image_media(Image.new("RGB", (300, 200), (120, 60, 180)), fmt="JPEG")
        # The *same* dict back: an unchanged item must not be re-encoded, or
        # every JPEG in the dataset would pay a generation loss and a snapshot.
        assert get_cleaner("image_edge_trim").clean(media) is media

    def test_wholly_solid_frame_is_a_no_op(self):
        cleaner = get_cleaner("image_edge_trim")
        for tone in ((255, 255, 255), (0, 0, 0)):
            media = _image_media(Image.new("RGB", (300, 200), tone))
            assert cleaner.clean(media) is media

    def test_saturated_border_is_content_not_padding(self):
        media = _image_media(_padded(400, 200, (100, 0, 300, 200), (0, 200, 0), (0, 0, 200)))
        assert get_cleaner("image_edge_trim").clean(media) is media

    def test_trim_is_capped_so_a_tiny_subject_cannot_explode(self):
        # A 10 px dot centred in a 400 px white field: uncapped this would crop
        # to ~10 px; the per-side cap keeps the middle 10% each way.
        media = _image_media(_padded(400, 400, (195, 195, 205, 205), (255, 255, 255), (200, 0, 0)))
        out = get_cleaner("image_edge_trim").clean(media)
        assert (out["width"], out["height"]) == (40, 40)

    def test_undecodable_and_empty_payloads_are_no_ops(self):
        cleaner = get_cleaner("image_edge_trim")
        for media in (
            {"media_type": "image", "media_bytes": b"<svg/>"},
            {"media_type": "image", "media_bytes": b""},
            {"media_type": "image"},
        ):
            assert cleaner.clean(media) is media

    def test_params_override_the_thresholds(self):
        from vtscore.media.image.cleaner import ImageEdgeTrimCleaner

        cleaner = get_cleaner("image_edge_trim")
        assert isinstance(cleaner, ImageEdgeTrimCleaner)
        assert [p["key"] for p in cleaner.parameters] == ["edge_tol", "max_edge_trim", "min_edge_trim"]

        tighter = cleaner.with_params({"max_edge_trim": 0.1})
        assert tighter is not cleaner
        assert tighter.max_edge_trim == 0.1
        assert tighter.edge_tol == cleaner.edge_tol  # unnamed params carry over

        # The same tiny-dot frame: a 0.1 cap leaves 80% of each axis standing.
        media = _image_media(_padded(400, 400, (195, 195, 205, 205), (255, 255, 255), (200, 0, 0)))
        out = tighter.clean(media)
        assert (out["width"], out["height"]) == (320, 320)

    def test_a_margin_thinner_than_min_edge_trim_is_left_alone(self):
        # A 4 px white margin on a 400 px frame is 1%, under the 2% floor.
        media = _image_media(_padded(400, 400, (4, 4, 396, 396), (255, 255, 255), (30, 30, 200)))
        assert get_cleaner("image_edge_trim").clean(media) is media

    def test_trims_the_margins_the_viewer_sees_not_the_sensor_ones(self):
        """A rotated photo is trimmed in *display* space.

        The stored bitmap here has a black letterbox above and below a red band,
        but its EXIF orientation says to rotate 90 degrees — so what anyone
        actually sees is a *pillarbox*, black to the left and right.  Trimming
        the sensor bitmap would shave the wrong two edges and leave the visible
        bars untouched, and would also report dimensions transposed from the
        media's stored ``width``/``height``.

        The re-encoded copy must not carry the orientation tag onward either:
        the pixels are upright now, so a viewer honouring the tag would rotate
        an already-rotated image a second time.
        """
        img = _padded(400, 400, (0, 100, 400, 300), (0, 0, 0), (200, 30, 30))
        exif = Image.Exif()
        exif[274] = 6
        buf = io.BytesIO()
        img.save(buf, format="JPEG", exif=exif)
        media = {"media_type": "image", "media_bytes": buf.getvalue(), "width": 400, "height": 400}

        out = get_cleaner("image_edge_trim").clean(media)
        assert out is not media
        # Upright, the 400x200 content band stands on end: the *horizontal*
        # axis is the one that loses its bars.  (JPEG ringing smears the
        # boundary a few pixels, so the width is checked as a band.)
        assert out["height"] == 400
        assert 190 <= out["width"] <= 215
        with Image.open(io.BytesIO(out["media_bytes"])) as trimmed:
            assert trimmed.size == (out["width"], out["height"])
            assert trimmed.getexif().get(274) is None


class TestEdgeTrimIsSharedWithThumbnails:
    def test_one_detector_backs_both_callers(self):
        """The cleaner and the thumbnail agree on where the content starts."""
        from vtscore.media.image.edge_trim import solid_edge_box, trim_solid_edges

        img = _padded(400, 400, (0, 100, 400, 300), (0, 0, 0), (200, 30, 30))
        box = solid_edge_box(img)
        assert box is not None
        assert trim_solid_edges(img).size == (box[2] - box[0], box[3] - box[1])

        media = _image_media(img)
        out = get_cleaner("image_edge_trim").clean(media)
        assert (out["width"], out["height"]) == (box[2] - box[0], box[3] - box[1])


# ---------------------------------------------------------------------------
# audio_silence_trim
# ---------------------------------------------------------------------------


class TestAudioSilenceTrimCleaner:
    def test_identity(self):
        from vtscore.media.audio.cleaner import AudioSilenceTrimCleaner

        cleaner = get_cleaner("audio_silence_trim")
        assert isinstance(cleaner, AudioSilenceTrimCleaner)
        assert cleaner.media_type == "audio"
        assert cleaner.display_name == "Silence Trim"
        assert cleaner.default_enabled is False
        assert cleaner.top_db == 40.0
        assert cleaner.pad == 0.05

    def test_rejects_invalid_construction(self):
        from vtscore.media.audio.cleaner import AudioSilenceTrimCleaner

        with pytest.raises(ValueError):
            AudioSilenceTrimCleaner(top_db=0)
        with pytest.raises(ValueError):
            AudioSilenceTrimCleaner(pad=-0.1)

    def test_trims_lead_in_and_tail(self):
        wav = _concat_wavs(_tone(0.7, 0), _tone(1.0), _tone(0.7, 0))
        media = _audio_media(wav, 2.4)
        out = get_cleaner("audio_silence_trim").clean(media)

        assert out is not media
        assert out["duration"] == pytest.approx(1.1, abs=0.2)
        assert out["file_size"] == len(out["media_bytes"])
        assert _wav_duration(out["media_bytes"]) == pytest.approx(out["duration"], abs=0.01)

    def test_keeps_the_pause_in_the_middle(self):
        """Internal silence is content (rhythm); only the ends are wasted."""
        wav = _concat_wavs(_tone(0.7, 0), _tone(0.5), _tone(1.0, 0), _tone(0.5), _tone(0.7, 0))
        out = get_cleaner("audio_silence_trim").clean(_audio_media(wav, 3.4))

        # ~2.0 s of audible span (0.5 + 1.0 gap + 0.5) survives, not 1.0 s.
        assert out["duration"] == pytest.approx(2.1, abs=0.25)

    def test_pure_tone_is_a_no_op(self):
        media = _audio_media(_tone(1.0), 1.0)
        assert get_cleaner("audio_silence_trim").clean(media) is media

    def test_pure_silence_is_a_no_op(self):
        # No audible content to anchor a span: leave the clip alone rather than
        # trim it to nothing.
        media = _audio_media(_tone(1.0, 0), 1.0)
        assert get_cleaner("audio_silence_trim").clean(media) is media

    def test_negligible_trim_is_a_no_op(self):
        # 30 ms of silence at each end is under the trim floor; re-encoding to
        # shave it would cost a full ``original_*`` snapshot for nothing.
        wav = _concat_wavs(_tone(0.03, 0), _tone(1.0), _tone(0.03, 0))
        media = _audio_media(wav, 1.06)
        assert get_cleaner("audio_silence_trim").clean(media) is media

    def test_undecodable_and_empty_payloads_are_no_ops(self):
        cleaner = get_cleaner("audio_silence_trim")
        for media in (
            {"media_type": "audio", "media_bytes": b"not a wav"},
            {"media_type": "audio", "media_bytes": b""},
            {"media_type": "audio", "duration": 3.0},
        ):
            assert cleaner.clean(media) is media

    def test_params_override_the_thresholds(self):
        from vtscore.media.audio.cleaner import AudioSilenceTrimCleaner

        cleaner = get_cleaner("audio_silence_trim")
        assert isinstance(cleaner, AudioSilenceTrimCleaner)
        assert [p["key"] for p in cleaner.parameters] == ["top_db", "pad"]

        padded = cleaner.with_params({"pad": 0.4})
        assert padded is not cleaner
        assert (padded.pad, padded.top_db) == (0.4, cleaner.top_db)

        wav = _concat_wavs(_tone(0.7, 0), _tone(1.0), _tone(0.7, 0))
        tight = cleaner.clean(_audio_media(wav, 2.4))["duration"]
        loose = padded.clean(_audio_media(wav, 2.4))["duration"]
        assert loose > tight  # more padding kept => a longer surviving span

    def test_shares_the_detector_with_the_silence_clipper(self):
        """One detector, two policies: N clips vs one trimmed span."""
        from vtscore.media.audio.clipper import SoundSilenceClipper
        from vtscore.media.audio.silence import detect_nonsilent_segments

        wav = _concat_wavs(_tone(0.7, 0), _tone(0.5), _tone(1.0, 0), _tone(0.5), _tone(0.7, 0))
        segments = detect_nonsilent_segments(wav, top_db=40.0, pad=0.05)
        assert segments is not None and len(segments) == 2

        clips = SoundSilenceClipper(top_db=40.0, min_clip_duration=0.0, pad=0.05).clip(_audio_media(wav, 3.4))
        assert len(clips) == 2  # the clipper emits one clip per segment...

        trimmed = get_cleaner("audio_silence_trim").clean(_audio_media(wav, 3.4))
        # ...while the cleaner keeps the single span that covers both.
        assert trimmed["duration"] == pytest.approx(segments[-1][1] - segments[0][0], abs=0.01)


class TestCleanedAudioWaveform:
    def test_waveform_is_not_re_sliced_for_materialized_bytes(self):
        """A trimmed clip's waveform must render its own bytes, whole.

        The cleaner drops ``thumbnail_bytes`` (the old one describes the
        pre-clean payload), so the waveform is regenerated on demand.  The
        item's ``clip_start`` / ``clip_end`` are relative to the *source* file,
        so applying them to the already-trimmed bytes would render the wrong
        stretch of audio.
        """
        from vtscore.media import get as get_media_type

        wav = _concat_wavs(_tone(0.7, 0), _tone(1.0), _tone(0.7, 0))
        trimmed = get_cleaner("audio_silence_trim").clean(_audio_media(wav, 2.4))
        trimmed.pop("thumbnail_bytes", None)
        # A window from the parent clipper that no longer describes these bytes.
        trimmed["clip_start"], trimmed["clip_end"] = 8.0, 10.4

        audio_type = get_media_type("audio")
        windowed = audio_type.ensure_thumbnail_bytes(dict(trimmed))
        bare = {k: v for k, v in trimmed.items() if k not in ("clip_start", "clip_end")}
        assert windowed is not None
        assert windowed == audio_type.ensure_thumbnail_bytes(bare)


# ---------------------------------------------------------------------------
# text_whitespace
# ---------------------------------------------------------------------------


class TestTextWhitespaceCleaner:
    def test_identity(self):
        cleaner = get_cleaner("text_whitespace")
        assert cleaner.media_type == "text"
        assert cleaner.default_enabled is False

    def test_rejoins_a_word_hyphen_broken_across_lines(self):
        media = _text_media("the repre-\nsentation of data")
        out = get_cleaner("text_whitespace").clean(media)
        assert out["media_string"] == "the representation of data"
        assert out["word_count"] == 4
        assert out["character_count"] == len(out["media_string"])
        assert out["file_size"] == len(out["media_string"].encode("utf-8"))

    def test_keeps_a_genuine_hyphenated_compound(self):
        # No line break after the hyphen, so it is a real compound word.
        media = _text_media("a state-of-the-art result")
        assert get_cleaner("text_whitespace").clean(media) is media

    def test_collapses_horizontal_runs_and_blank_lines(self):
        media = _text_media("one   two\t\tthree\n\n\n\nfour   \n")
        out = get_cleaner("text_whitespace").clean(media)
        assert out["media_string"] == "one two three\n\nfour"

    def test_normalises_line_endings_and_exotic_spaces(self):
        media = _text_media("a\r\nb c d　e")
        out = get_cleaner("text_whitespace").clean(media)
        assert out["media_string"] == "a\nb\nc d e"

    def test_strips_control_and_zero_width_characters(self):
        media = _text_media("clean​word\x07 and­more")
        out = get_cleaner("text_whitespace").clean(media)
        assert out["media_string"] == "cleanword andmore"

    def test_already_clean_text_is_a_no_op(self):
        media = _text_media("A tidy paragraph.\n\nAnd a second one.")
        assert get_cleaner("text_whitespace").clean(media) is media

    def test_paragraph_breaks_survive(self):
        media = _text_media("First para.\n\nSecond para.")
        assert get_cleaner("text_whitespace").clean(media) is media

    def test_empty_and_non_text_payloads_are_no_ops(self):
        cleaner = get_cleaner("text_whitespace")
        for media in ({"media_type": "text", "media_string": ""}, {"media_type": "text"}):
            assert cleaner.clean(media) is media


# ---------------------------------------------------------------------------
# text_markup_strip
# ---------------------------------------------------------------------------


class TestTextMarkupStripCleaner:
    def test_identity(self):
        cleaner = get_cleaner("text_markup_strip")
        assert cleaner.media_type == "text"
        assert cleaner.display_name == "Markup Strip"
        assert cleaner.default_enabled is False

    def test_strips_html_tags_and_unescapes_entities(self):
        media = _text_media("<p>Hello <b>world</b> &amp; friends.</p>")
        out = get_cleaner("text_markup_strip").clean(media)
        assert out["media_string"] == "Hello world & friends."

    def test_drops_script_and_style_bodies(self):
        media = _text_media("<style>.a{color:red}</style>Keep me<script>var x = 1;</script>")
        out = get_cleaner("text_markup_strip").clean(media)
        assert out["media_string"] == "Keep me"

    def test_drops_html_comments(self):
        media = _text_media("before<!-- nav chrome -->after")
        assert get_cleaner("text_markup_strip").clean(media)["media_string"] == "beforeafter"

    def test_block_tags_keep_words_apart(self):
        media = _text_media("<div>alpha</div><div>beta</div>")
        out = get_cleaner("text_markup_strip").clean(media)
        assert "alphabeta" not in out["media_string"]
        assert out["media_string"].split() == ["alpha", "beta"]

    def test_collapses_markdown_links_and_images_to_their_labels(self):
        media = _text_media("see [the docs](http://x.y/a?b=1) and ![a cat](cat.png)")
        out = get_cleaner("text_markup_strip").clean(media)
        assert out["media_string"] == "see the docs and a cat"

    def test_strips_headings_bullets_quotes_and_emphasis(self):
        media = _text_media("## Title\n\n- a **bold** point\n- an _italic_ one\n\n> quoted\n\n---\n")
        out = get_cleaner("text_markup_strip").clean(media)
        assert out["media_string"] == "Title\n\na bold point\nan italic one\n\nquoted"

    def test_leaves_snake_case_identifiers_alone(self):
        media = _text_media("call load_media_bytes then media_string")
        assert get_cleaner("text_markup_strip").clean(media) is media

    def test_leaves_comparison_operators_alone(self):
        # Not a tag: no element name follows the "<".
        media = _text_media("holds when a < b and b > c")
        assert get_cleaner("text_markup_strip").clean(media) is media

    def test_plain_prose_is_a_no_op(self):
        media = _text_media("An ordinary sentence, with punctuation.")
        assert get_cleaner("text_markup_strip").clean(media) is media

    def test_empty_and_non_text_payloads_are_no_ops(self):
        cleaner = get_cleaner("text_markup_strip")
        for media in ({"media_type": "text", "media_string": ""}, {"media_type": "text"}):
            assert cleaner.clean(media) is media

    def test_runs_before_whitespace_in_registration_order(self):
        """The strip leaves gaps; the whitespace gate mops them up afterwards."""
        from vtscore.media import cleaners_for_type

        order = [c.name for c in cleaners_for_type("text")]
        assert order.index("text_markup_strip") < order.index("text_whitespace")


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

#: Frame rate of every fabricated video below, so a frame index and a timestamp
#: convert cleanly (frame 6 of a 12fps clip is at 0.5s).
_VIDEO_FPS = 12.0


def _solid_frame(width: int, height: int, color: tuple[int, int, int]) -> np.ndarray:
    return np.full((height, width, 3), color, dtype=np.uint8)


def _barred_frame(width: int, height: int, bar: int, color: tuple[int, int, int]) -> np.ndarray:
    """A *color* frame with black bars *bar* px deep along the top and bottom."""
    frame = _solid_frame(width, height, color)
    frame[:bar] = 0
    frame[height - bar :] = 0
    return frame


class _FakeVideo:
    """Decode-layer stand-in serving a fixed frame list, recording every seek."""

    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = frames
        self.times: list[float] = []

    def probe(self, _path):
        height, width, _ = self.frames[0].shape
        return decode.VideoInfo(
            duration=len(self.frames) / _VIDEO_FPS,
            fps=_VIDEO_FPS,
            width=width,
            height=height,
        )

    def frame_at(self, _path, time_seconds: float):
        self.times.append(round(float(time_seconds), 6))
        index = int(round(float(time_seconds) * _VIDEO_FPS))
        if index < 0 or index >= len(self.frames):
            return None
        return self.frames[index]

    def frames_at(self, path, times):
        got = [self.frame_at(path, t) for t in times]
        return [f for f in got if f is not None]


@pytest.fixture
def fake_video(monkeypatch):
    """Return an installer swapping the video decode layer for a frame list."""

    def _install(frames: list[np.ndarray]) -> _FakeVideo:
        fake = _FakeVideo(frames)
        monkeypatch.setattr(decode, "probe", fake.probe)
        monkeypatch.setattr(decode, "frame_at", fake.frame_at)
        monkeypatch.setattr(decode, "frames_at", fake.frames_at)
        return fake

    return _install


def _video_media(tmp_path: Path, **extra) -> dict:
    """A video media dict backed by a real (if bogus) file the cleaners can open."""
    path = tmp_path / "clip.mp4"
    if not path.exists():
        path.write_bytes(b"not really a video; the decode layer is faked")
    media = {
        "id": 1,
        "media_type": "video",
        "filename": "clip.mp4",
        "media_path": str(path),
        "thumbnail_bytes": b"stale-thumb",
    }
    media.update(extra)
    return media


def _assert_payload_untouched(before: dict, after: dict) -> None:
    """A video gate must clean by metadata alone, never by rewriting bytes."""
    assert after.get("media_bytes") == before.get("media_bytes")
    assert after.get("media_string") == before.get("media_string")
    assert "original_media_bytes" not in after
    assert "original_media_string" not in after


# ---------------------------------------------------------------------------
# video_letterbox_crop
# ---------------------------------------------------------------------------


class TestVideoLetterboxCropCleaner:
    def test_identity(self):
        from vtscore.media.video.cleaner import VideoLetterboxCropCleaner

        cleaner = get_cleaner("video_letterbox_crop")
        assert isinstance(cleaner, VideoLetterboxCropCleaner)
        assert cleaner.media_type == "video"
        assert cleaner.display_name == "Letterbox Crop"
        # Cropping is a judgment call about what counts as padding.
        assert cleaner.default_enabled is False

    def test_records_the_bars_as_a_clip_box(self, fake_video, tmp_path):
        fake_video([_barred_frame(200, 100, 25, (200, 40, 40)) for _ in range(24)])
        media = _video_media(tmp_path)
        out = get_cleaner("video_letterbox_crop").clean(media)

        assert out is not media
        assert out["clip_box"] == [0, 25, 200, 75]
        _assert_payload_untouched(media, out)

    def test_the_stale_parent_thumbnail_is_not_carried_forward_by_the_runner(self, fake_video, tmp_path):
        """The gate itself copies the dict; the chain runner drops the thumbnail.

        Asserted here so the pair stays honest: a cropped unit whose preview
        still showed the bars would contradict the crop.
        """
        from vtscore.datasets.clipper_chain import _run_cleaner_step

        fake_video([_barred_frame(200, 100, 25, (200, 40, 40)) for _ in range(24)])
        media = _video_media(tmp_path)
        (out,), (entry,) = _run_cleaner_step(media, {"kind": "cleaner", "name": "video_letterbox_crop", "params": {}})
        assert entry["changed"] is True
        assert "thumbnail_bytes" not in out

    def test_the_union_of_the_sampled_frames_wins(self, fake_video, tmp_path):
        """A margin is cropped only where *every* sampled frame agrees it is bar.

        The subject drifts into the upper bar half way through the clip; the
        union keeps that room, an intersection would have cut it off.
        """
        deep = [_barred_frame(200, 100, 25, (200, 40, 40)) for _ in range(12)]
        shallow = [_barred_frame(200, 100, 10, (200, 40, 40)) for _ in range(12)]
        fake_video(deep + shallow)
        out = get_cleaner("video_letterbox_crop").clean(_video_media(tmp_path))
        assert out["clip_box"] == [0, 10, 200, 90]

    def test_content_to_the_edges_is_a_no_op(self, fake_video, tmp_path):
        fake_video([_solid_frame(200, 100, (90, 140, 60)) for _ in range(24)])
        media = _video_media(tmp_path)
        assert get_cleaner("video_letterbox_crop").clean(media) is media

    def test_one_unpadded_frame_vetoes_the_crop(self, fake_video, tmp_path):
        """Bars for most of the clip, but one frame fills the frame: no crop."""
        frames = [_barred_frame(200, 100, 25, (200, 40, 40)) for _ in range(24)]
        frames[12] = _solid_frame(200, 100, (90, 140, 60))
        fake_video(frames)
        media = _video_media(tmp_path)
        assert get_cleaner("video_letterbox_crop").clean(media) is media

    def test_a_wholly_blank_frame_does_not_veto_the_crop(self, fake_video, tmp_path):
        """A fade-to-black frame says nothing about where the bars are."""
        frames = [_barred_frame(200, 100, 25, (200, 40, 40)) for _ in range(24)]
        frames[12] = _solid_frame(200, 100, (0, 0, 0))
        fake_video(frames)
        out = get_cleaner("video_letterbox_crop").clean(_video_media(tmp_path))
        assert out["clip_box"] == [0, 25, 200, 75]

    def test_an_all_blank_clip_is_a_no_op(self, fake_video, tmp_path):
        fake_video([_solid_frame(200, 100, (0, 0, 0)) for _ in range(24)])
        media = _video_media(tmp_path)
        assert get_cleaner("video_letterbox_crop").clean(media) is media

    def test_a_margin_thinner_than_min_edge_trim_is_left_alone(self, fake_video, tmp_path):
        # A 1 px bar on a 100 px frame is 1%, under the 2% floor.
        fake_video([_barred_frame(200, 100, 1, (200, 40, 40)) for _ in range(24)])
        media = _video_media(tmp_path)
        assert get_cleaner("video_letterbox_crop").clean(media) is media

    def test_only_the_units_own_window_is_sampled(self, fake_video, tmp_path):
        """A tile must measure its own frames, not the whole parent video's."""
        fake = fake_video([_barred_frame(200, 100, 25, (200, 40, 40)) for _ in range(24)])
        media = _video_media(tmp_path, clip_start=1.0, clip_end=1.5)
        get_cleaner("video_letterbox_crop").clean(media)
        assert fake.times
        assert all(1.0 <= t <= 1.5 for t in fake.times)

    def test_composes_with_an_earlier_crop(self, fake_video, tmp_path):
        """The box is stored in source-frame coordinates, so crops compose.

        The unit already covers rows 10-90; inside *that* region the bars are
        another 15 px deep, which lands back at the true content box.
        """
        fake_video([_barred_frame(200, 100, 25, (200, 40, 40)) for _ in range(24)])
        out = get_cleaner("video_letterbox_crop").clean(_video_media(tmp_path, clip_box=[0, 10, 200, 90]))
        assert out["clip_box"] == [0, 25, 200, 75]

    def test_an_already_cropped_unit_is_a_no_op(self, fake_video, tmp_path):
        fake_video([_barred_frame(200, 100, 25, (200, 40, 40)) for _ in range(24)])
        media = _video_media(tmp_path, clip_box=[0, 25, 200, 75])
        assert get_cleaner("video_letterbox_crop").clean(media) is media

    def test_undecodable_and_pathless_payloads_are_no_ops(self, tmp_path):
        cleaner = get_cleaner("video_letterbox_crop")
        for media in (
            {"media_type": "video", "media_path": str(tmp_path / "nope.mp4")},
            {"media_type": "video", "media_bytes": b"not a video at all"},
            {"media_type": "video"},
        ):
            assert cleaner.clean(media) is media

    def test_params_override_the_thresholds(self, fake_video, tmp_path):
        from vtscore.media.video.cleaner import VideoLetterboxCropCleaner

        cleaner = get_cleaner("video_letterbox_crop")
        assert [p["key"] for p in cleaner.parameters] == [
            "samples",
            "edge_tol",
            "max_edge_trim",
            "min_edge_trim",
        ]

        looser = cleaner.with_params({"min_edge_trim": 0.005, "samples": 2})
        assert isinstance(cleaner, VideoLetterboxCropCleaner)
        assert isinstance(looser, VideoLetterboxCropCleaner)
        assert looser is not cleaner
        assert looser.samples == 2
        assert looser.edge_tol == cleaner.edge_tol  # unnamed params carry over

        # The 1 px bar the default floor ignores is now worth cropping.
        fake = fake_video([_barred_frame(200, 100, 1, (200, 40, 40)) for _ in range(24)])
        out = looser.clean(_video_media(tmp_path))
        assert out["clip_box"] == [0, 1, 200, 99]
        assert len(fake.times) == 2

    def test_the_crop_is_capped_per_side(self, fake_video, tmp_path):
        # A 10 px dot in a 200x200 black field: uncapped this would crop to the
        # dot; the 0.45 cap keeps the middle 10% of each axis.
        frame = _solid_frame(200, 200, (0, 0, 0))
        frame[95:105, 95:105] = (220, 30, 30)
        fake_video([frame for _ in range(24)])
        out = get_cleaner("video_letterbox_crop").clean(_video_media(tmp_path))
        assert out["clip_box"] == [90, 90, 110, 110]


# ---------------------------------------------------------------------------
# video_blank_trim
# ---------------------------------------------------------------------------


def _blank_led_frames(lead: int, content: int, tail: int) -> list[np.ndarray]:
    """``lead`` black frames, ``content`` coloured ones, ``tail`` black ones."""
    black = _solid_frame(160, 90, (0, 0, 0))
    lit = _solid_frame(160, 90, (180, 90, 40))
    return [black] * lead + [lit] * content + [black] * tail


class TestVideoBlankTrimCleaner:
    def test_identity(self):
        from vtscore.media.video.cleaner import VideoBlankTrimCleaner

        cleaner = get_cleaner("video_blank_trim")
        assert isinstance(cleaner, VideoBlankTrimCleaner)
        assert cleaner.media_type == "video"
        assert cleaner.display_name == "Blank Frame Trim"
        assert cleaner.default_enabled is False

    def test_narrows_the_window_past_the_blank_head_and_tail(self, fake_video, tmp_path):
        # 24 frames at 12fps: 6 black, 12 lit, 6 black.
        fake_video(_blank_led_frames(6, 12, 6))
        media = _video_media(tmp_path)
        out = get_cleaner("video_blank_trim").clean(media)

        assert out is not media
        # The cut lands on a scan step and never crosses the first frame with
        # content in it, so one 0.25s step of black survives at each end.
        assert (out["clip_start"], out["clip_end"]) == (0.25, 1.75)
        assert out["duration"] == 1.5
        _assert_payload_untouched(media, out)

    def test_the_cut_never_crosses_into_content(self, fake_video, tmp_path):
        """Trimming to the first content probe would eat up to a step of picture."""
        fake_video(_blank_led_frames(6, 12, 6))
        out = get_cleaner("video_blank_trim").clean(_video_media(tmp_path))
        first_content = 6 / _VIDEO_FPS
        last_content = 18 / _VIDEO_FPS
        assert out["clip_start"] <= first_content
        assert out["clip_end"] >= last_content

    def test_content_at_both_ends_costs_two_probes(self, fake_video, tmp_path):
        fake = fake_video(_blank_led_frames(0, 24, 0))
        media = _video_media(tmp_path)
        assert get_cleaner("video_blank_trim").clean(media) is media
        assert len(fake.times) == 2

    def test_blank_frames_inside_the_clip_are_kept(self, fake_video, tmp_path):
        """A mid-clip cut to black is content rhythm, not waste."""
        frames = _blank_led_frames(0, 24, 0)
        frames[10:14] = [_solid_frame(160, 90, (0, 0, 0))] * 4
        fake_video(frames)
        media = _video_media(tmp_path)
        assert get_cleaner("video_blank_trim").clean(media) is media

    def test_a_white_card_counts_as_blank(self, fake_video, tmp_path):
        frames = [_solid_frame(160, 90, (255, 255, 255))] * 6 + [_solid_frame(160, 90, (180, 90, 40))] * 18
        fake_video(frames)
        out = get_cleaner("video_blank_trim").clean(_video_media(tmp_path))
        assert out["clip_start"] == 0.25

    def test_a_trim_shorter_than_the_floor_is_a_no_op(self, fake_video, tmp_path):
        # One black frame (~0.083s) at the head, under the 0.1s floor.
        fake_video(_blank_led_frames(1, 23, 0))
        media = _video_media(tmp_path)
        assert get_cleaner("video_blank_trim").clean(media) is media

    def test_the_trim_is_capped_per_end(self, fake_video, tmp_path):
        """An almost entirely blank clip loses at most max_trim off each end."""
        fake_video(_blank_led_frames(11, 2, 11))
        out = get_cleaner("video_blank_trim").clean(_video_media(tmp_path))
        # 2s clip, 0.25 cap: 0.5s off each end and no more.
        assert (out["clip_start"], out["clip_end"]) == (0.5, 1.5)

    def test_trims_within_a_tiles_own_window(self, fake_video, tmp_path):
        """A tile's blank head is trimmed relative to the tile, not the parent."""
        fake_video(_blank_led_frames(12, 12, 0))
        media = _video_media(tmp_path, clip_start=0.5, clip_end=1.5, duration=1.0)
        out = get_cleaner("video_blank_trim").clean(media)
        assert (out["clip_start"], out["clip_end"]) == (0.75, 1.5)
        assert out["duration"] == 0.75

    def test_high_contrast_content_is_not_blank(self, fake_video, tmp_path):
        """Black-on-white content is two flat tones, but neither dominates."""
        frame = _solid_frame(160, 90, (255, 255, 255))
        frame[:, :80] = 0
        fake_video([frame] * 24)
        media = _video_media(tmp_path)
        assert get_cleaner("video_blank_trim").clean(media) is media

    def test_undecodable_and_pathless_payloads_are_no_ops(self, tmp_path):
        cleaner = get_cleaner("video_blank_trim")
        for media in (
            {"media_type": "video", "media_path": str(tmp_path / "nope.mp4")},
            {"media_type": "video", "media_bytes": b"not a video at all"},
            {"media_type": "video"},
        ):
            assert cleaner.clean(media) is media

    def test_params_override_the_thresholds(self, fake_video, tmp_path):
        from vtscore.media.video.cleaner import VideoBlankTrimCleaner

        cleaner = get_cleaner("video_blank_trim")
        assert [p["key"] for p in cleaner.parameters] == ["blank_ratio", "max_trim", "step"]

        tighter = cleaner.with_params({"max_trim": 0.1})
        assert isinstance(cleaner, VideoBlankTrimCleaner)
        assert isinstance(tighter, VideoBlankTrimCleaner)
        assert tighter is not cleaner
        assert tighter.max_trim == 0.1
        assert tighter.step == cleaner.step  # unnamed params carry over

        fake_video(_blank_led_frames(6, 12, 6))
        out = tighter.clean(_video_media(tmp_path))
        assert (out["clip_start"], out["clip_end"]) == (0.2, 1.8)

    def test_the_probe_count_stays_bounded_on_a_long_clip(self, fake_video, tmp_path):
        """The scan step coarsens with duration instead of probing every 0.25s."""
        from vtscore.media.video.cleaner import _MAX_PROBES_PER_END

        # 20 minutes of black: a fixed 0.25s step would be 1200 probes per end.
        fake = fake_video([_solid_frame(64, 36, (0, 0, 0))] * int(_VIDEO_FPS * 1200))
        get_cleaner("video_blank_trim").clean(_video_media(tmp_path))
        assert len(fake.times) <= 2 * (_MAX_PROBES_PER_END + 1)

    def test_the_pair_runs_crop_first(self):
        """Cropping before the blank scan keeps the two gates order-insensitive.

        Bars are near-black, so an uncropped frame reads as far blanker than
        the picture inside it does.
        """
        from vtscore.media import cleaners_for_type

        order = [c.name for c in cleaners_for_type("video")]
        assert order.index("video_letterbox_crop") < order.index("video_blank_trim")


class TestVideoGatesOnRealVideo:
    """End-to-end over ffmpeg-encoded files, no faked decode layer."""

    def _encode(self, frames: list[np.ndarray], tmp_path: Path, name: str) -> Path:
        from vtscore.utils.synthetic.video import _encode_frames

        path = tmp_path / name
        _encode_frames(path, frames, fps=int(_VIDEO_FPS))
        return path

    def test_real_letterboxed_video_is_cropped_to_its_content(self, tmp_path):
        path = self._encode([_barred_frame(160, 96, 24, (210, 40, 40)) for _ in range(24)], tmp_path, "bars.mp4")
        out = get_cleaner("video_letterbox_crop").clean({"media_type": "video", "media_path": str(path)})
        x0, y0, x1, y1 = out["clip_box"]
        assert (x0, x1) == (0, 160)  # no side bars to lose
        assert abs(y0 - 24) <= 3
        assert abs(y1 - 72) <= 3

    def test_real_unpadded_video_is_a_no_op(self, tmp_path):
        frames = []
        rng = np.random.default_rng(0)
        for _ in range(24):
            frames.append(rng.integers(40, 210, size=(96, 160, 3), dtype=np.uint8))
        path = self._encode(frames, tmp_path, "noise.mp4")
        media = {"media_type": "video", "media_path": str(path)}
        assert get_cleaner("video_letterbox_crop").clean(media) is media

    def test_real_black_led_video_is_trimmed(self, tmp_path):
        path = self._encode(_blank_led_frames(6, 12, 6), tmp_path, "leader.mp4")
        media = {"media_type": "video", "media_path": str(path)}
        out = get_cleaner("video_blank_trim").clean(media)
        assert out is not media
        assert out["clip_start"] >= 0.25
        assert out["clip_end"] <= 1.75
        assert out["duration"] == round(out["clip_end"] - out["clip_start"], 6)
