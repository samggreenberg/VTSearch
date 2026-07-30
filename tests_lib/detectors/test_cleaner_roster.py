"""Tests for the shipped roster of :class:`~vtscore.media.cleaner.MediaCleaner`s.

The framework itself (registry, chain placement, dual payload, replay) is
covered by ``test_media_cleaners.py``; this file exercises what each concrete
gate actually does to a payload.  Every cleaner shares one contract, so every
class below checks the same three things:

- it removes what it claims to remove,
- it returns the media object **itself** when there is nothing to remove (that
  identity is what tells the chain runner to skip the ``original_*`` snapshot),
- a degenerate or undecodable payload is a no-op, never an error.

Covered: ``image_edge_trim``, ``audio_silence_trim``, ``text_whitespace``,
``text_markup_strip``.
"""

from __future__ import annotations

import io
import wave

import pytest
from PIL import Image
from vtscore.media import get_cleaner

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

    def test_crop_preserves_an_unbaked_exif_orientation(self):
        """Trimming must not silently un-rotate a photo.

        ``image_edge_trim`` can run with ``image_exif_orient`` switched off, in
        which case the payload still carries an orientation tag that viewers
        honour.  Dropping it during the re-encode would leave the trimmed copy
        displayed sideways.
        """
        img = _padded(400, 400, (0, 100, 400, 300), (0, 0, 0), (200, 30, 30))
        exif = Image.Exif()
        exif[274] = 6
        buf = io.BytesIO()
        img.save(buf, format="JPEG", exif=exif)
        media = {"media_type": "image", "media_bytes": buf.getvalue(), "width": 400, "height": 400}

        out = get_cleaner("image_edge_trim").clean(media)
        assert out is not media
        with Image.open(io.BytesIO(out["media_bytes"])) as trimmed:
            assert trimmed.getexif().get(274) == 6


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
