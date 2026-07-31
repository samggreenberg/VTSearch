"""Tests for :mod:`vtscore.datasets.file_types`.

The histogram the Stats window shows used to come from each item's
``filename`` alone, so a service-style importer that names items after an
opaque content id put every item in one useless bucket.  These tests pin the
fallback chain (name → path → URL → magic numbers) that fixes that.
"""

from __future__ import annotations

from vtscore.datasets.file_types import (
    UNKNOWN_FILE_TYPE,
    count_file_types,
    counts_are_uninformative,
    media_file_type,
    sniff_file_type,
)


# Minimal but genuine headers for the formats an image/audio/video import hits.
JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01"
PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
GIF = b"GIF89a\x10\x00\x10\x00"
WEBP = b"RIFF\x24\x00\x00\x00WEBPVP8 "
WAV = b"RIFF\x24\x00\x00\x00WAVEfmt "
AVI = b"RIFF\x24\x00\x00\x00AVI LIST"
MP4 = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00"
MOV = b"\x00\x00\x00\x14ftypqt  \x00\x00\x02\x00"
HEIC = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00"
MKV = b"\x1a\x45\xdf\xa3\x01\x00\x00\x00\x00\x00\x00\x1f" + b"matroska" + b"\x00" * 8
WEBM = b"\x1a\x45\xdf\xa3\x01\x00\x00\x00\x00\x00\x00\x1f" + b"webm" + b"\x00" * 8
PDF = b"%PDF-1.7\n%\xc7\xec\x8f\xa2"
MP3_TAGGED = b"ID3\x03\x00\x00\x00\x00\x00\x00"
MP3_BARE = b"\xff\xfb\x90\x64\x00\x00\x00\x00"
ZIP = b"PK\x03\x04\x14\x00\x00\x00\x08\x00"


class TestSniffFileType:
    """Magic-number recognition for the containers VTSearch ingests."""

    def test_recognises_image_formats(self):
        assert sniff_file_type(JPEG) == "jpg"
        assert sniff_file_type(PNG) == "png"
        assert sniff_file_type(GIF) == "gif"
        assert sniff_file_type(WEBP) == "webp"
        assert sniff_file_type(b"BM\x8a\x00\x00\x00\x00\x00") == "bmp"
        assert sniff_file_type(b"II*\x00\x08\x00\x00\x00") == "tif"
        assert sniff_file_type(HEIC) == "heic"

    def test_recognises_audio_formats(self):
        assert sniff_file_type(WAV) == "wav"
        assert sniff_file_type(b"fLaC\x00\x00\x00\x22") == "flac"
        assert sniff_file_type(b"OggS\x00\x02\x00\x00") == "ogg"
        assert sniff_file_type(MP3_TAGGED) == "mp3"
        assert sniff_file_type(MP3_BARE) == "mp3"

    def test_recognises_video_formats(self):
        assert sniff_file_type(MP4) == "mp4"
        assert sniff_file_type(MOV) == "mov"
        assert sniff_file_type(AVI) == "avi"
        assert sniff_file_type(MKV) == "mkv"
        # Matroska and WebM share a magic number; the DocType breaks the tie.
        assert sniff_file_type(WEBM) == "webm"

    def test_recognises_documents_and_archives(self):
        assert sniff_file_type(PDF) == "pdf"
        assert sniff_file_type(ZIP) == "zip"
        assert sniff_file_type(b"\x1f\x8b\x08\x00\x00\x00\x00\x00") == "gz"

    def test_unrecognised_and_truncated_data_yield_nothing(self):
        assert sniff_file_type(b"hello, world, this is not a container") == ""
        assert sniff_file_type(b"\xff\xd8") == ""
        assert sniff_file_type(b"") == ""

    def test_accepts_any_bytes_like_payload(self):
        assert sniff_file_type(bytearray(PNG)) == "png"
        assert sniff_file_type(memoryview(JPEG)) == "jpg"


class TestMediaFileType:
    """The fallback chain, in the order :func:`media_file_type` walks it."""

    def test_filename_extension_wins(self):
        assert media_file_type({"filename": "kitten.JPG", "media_bytes": PNG}) == "jpg"

    def test_only_the_final_path_segment_counts(self):
        assert media_file_type({"filename": "run.v2/audio"}) == UNKNOWN_FILE_TYPE
        assert media_file_type({"filename": "run.v2/audio.wav"}) == "wav"
        assert media_file_type({"filename": r"C:\clips\take.wav"}) == "wav"

    def test_dotted_ids_and_dates_are_not_extensions(self):
        # A trailing dotted chunk that no filesystem would call an extension
        # must not become its own one-item bucket in the histogram.
        assert media_file_type({"filename": "photo.2024-06-01-final"}) == UNKNOWN_FILE_TYPE
        assert media_file_type({"filename": "id.9f3ca77e1b4d"}) == UNKNOWN_FILE_TYPE
        assert media_file_type({"filename": "clip.12345"}) == UNKNOWN_FILE_TYPE

    def test_falls_back_to_origin_name_then_path(self):
        assert media_file_type({"filename": "a7f3c9e2", "origin_name": "kitten.png"}) == "png"
        assert media_file_type({"filename": "a7f3c9e2", "media_path": "/data/kitten.gif"}) == "gif"

    def test_falls_back_to_url_ignoring_query_and_fragment(self):
        media = {"filename": "a7f3c9e2", "media_url": "https://host/media/kitten.webp?token=abc#x"}
        assert media_file_type(media) == "webp"

    def test_falls_back_to_sniffing_the_bytes(self):
        # The reported shape: a service importer names items after a content
        # id and the URL carries no extension either, so only the bytes know.
        media = {
            "filename": "a7f3c9e2",
            "origin_name": "a7f3c9e2",
            "media_url": "https://host/api/v1/media/a7f3c9e2",
            "media_bytes": JPEG,
        }
        assert media_file_type(media) == "jpg"

    def test_inline_text_with_no_file_counts_as_txt(self):
        assert media_file_type({"filename": "row_5", "media_string": "some text"}) == "txt"
        assert media_file_type({"filename": "row_6", "media_string": ""}) == "txt"

    def test_unknowable_media_lands_in_the_sentinel_bucket(self):
        assert media_file_type({}) == UNKNOWN_FILE_TYPE
        assert media_file_type({"filename": "a7f3c9e2", "media_bytes": b"not a container"}) == UNKNOWN_FILE_TYPE

    def test_sentinel_is_distinguishable_from_a_real_extension(self):
        # The frontend keys off the leading paren to skip its leading dot.
        assert UNKNOWN_FILE_TYPE.startswith("(")


class TestCountFileTypes:
    def test_tallies_most_common_first(self):
        medias = [
            {"filename": "a.wav"},
            {"filename": "b.mp3"},
            {"filename": "c.wav"},
            {"filename": "d.wav"},
        ]
        assert count_file_types(medias) == {"wav": 3, "mp3": 1}
        assert list(count_file_types(medias)) == ["wav", "mp3"]

    def test_extensionless_image_import_reports_the_real_format(self):
        # Regression: 437 sniffable images used to collapse into one bucket.
        medias = [{"filename": f"id{i}", "media_bytes": JPEG} for i in range(437)]
        assert count_file_types(medias) == {"jpg": 437}

    def test_mixes_named_and_sniffed_items(self):
        medias = [
            {"filename": "a.png"},
            {"filename": "b", "media_bytes": PNG},
            {"filename": "c", "media_bytes": b"???"},
        ]
        assert count_file_types(medias) == {"png": 2, UNKNOWN_FILE_TYPE: 1}


class TestCountsAreUninformative:
    def test_empty_and_all_unknown_counts_are_uninformative(self):
        assert counts_are_uninformative({})
        assert counts_are_uninformative({UNKNOWN_FILE_TYPE: 437})
        # The bucket name entries stamped before this module existed used.
        assert counts_are_uninformative({"(no extension)": 437})

    def test_any_real_type_makes_counts_informative(self):
        assert not counts_are_uninformative({"jpg": 1})
        assert not counts_are_uninformative({"jpg": 400, UNKNOWN_FILE_TYPE: 37})
