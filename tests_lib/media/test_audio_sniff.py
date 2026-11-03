"""Audio responses are typed by their container, not assumed WAV (#4981).

An import keeps its file verbatim, so the bytes an audio media serves can be
MP3, FLAC, OGG, an MP4 container or a raw ADTS stream; only clip slices are WAV.
:func:`sniff_audio_format` reads the leading bytes, and ``media_response``
labels the payload with what it finds.
"""

from __future__ import annotations

import pytest

from vtscore.media.audio.audio_generator import generate_wav
from vtscore.media.audio.media_type import AudioMediaType
from vtscore.media.audio.sniff import sniff_audio_format

# Minimal headers for each container, padded past the 12 bytes the sniffer reads.
_PAD = b"\x00" * 16
_MP3_ID3 = b"ID3\x04\x00\x00\x00\x00\x00\x00" + _PAD
_MP3_FRAME = b"\xff\xfb\x90\x64" + _PAD  # MPEG-1 Layer III, no ID3 tag
_FLAC = b"fLaC\x00\x00\x00\x22" + _PAD
_OGG = b"OggS\x00\x02" + _PAD
_M4A = b"\x00\x00\x00\x20ftypM4A " + _PAD
_ADTS = b"\xff\xf1\x50\x80" + _PAD  # MPEG-4 AAC, no CRC


class TestSniffAudioFormat:
    def test_wav(self):
        assert sniff_audio_format(generate_wav(440.0, 0.1)) == ("audio/wav", ".wav")

    def test_flac(self):
        assert sniff_audio_format(_FLAC) == ("audio/flac", ".flac")

    def test_ogg(self):
        assert sniff_audio_format(_OGG) == ("audio/ogg", ".ogg")

    @pytest.mark.parametrize("brand", [b"M4A ", b"isom", b"mp42", b"qt  "])
    def test_iso_bmff_is_m4a(self, brand):
        # An MP4 that also holds video is still served for an <audio> element.
        data = b"\x00\x00\x00\x18ftyp" + brand + _PAD
        assert sniff_audio_format(data) == ("audio/mp4", ".m4a")

    def test_id3_tagged_mp3(self):
        assert sniff_audio_format(_MP3_ID3) == ("audio/mpeg", ".mp3")

    @pytest.mark.parametrize(
        "second_byte",
        [
            0xFB,  # MPEG-1 Layer III
            0xF3,  # MPEG-2 Layer III
            0xE3,  # MPEG-2.5 Layer III
            0xFD,  # MPEG-1 Layer II
        ],
    )
    def test_bare_mpeg_frame_sync(self, second_byte):
        assert sniff_audio_format(bytes([0xFF, second_byte]) + _PAD) == ("audio/mpeg", ".mp3")

    @pytest.mark.parametrize("second_byte", [0xF0, 0xF1, 0xF8, 0xF9])
    def test_adts_is_aac_not_mp3(self, second_byte):
        # ADTS shares MPEG audio's sync word; its layer bits (00) tell it apart.
        assert sniff_audio_format(bytes([0xFF, second_byte]) + _PAD) == ("audio/aac", ".aac")

    @pytest.mark.parametrize(
        "data",
        [
            None,
            b"",
            b"\xff",  # a lone sync byte
            b"\xff\xe1" + _PAD,  # 11-bit sync with the reserved layer: neither MP3 nor ADTS
            b"not audio at all",
            b"FORM\x00\x00\x00\x00AIFF",  # a container the sniffer doesn't name
        ],
    )
    def test_unrecognised_defaults_to_wav(self, data):
        assert sniff_audio_format(data) == ("audio/wav", ".wav")


class TestMediaResponseType:
    @pytest.mark.parametrize(
        ("payload", "mimetype", "download_name"),
        [
            (_MP3_ID3, "audio/mpeg", "media_7.mp3"),
            (_MP3_FRAME, "audio/mpeg", "media_7.mp3"),
            (_FLAC, "audio/flac", "media_7.flac"),
        ],
    )
    def test_payload_decides_type_and_extension(self, payload, mimetype, download_name):
        # The filename says .wav; the bytes win, since they are what is served.
        resp = AudioMediaType().media_response({"id": 7, "media_bytes": payload, "filename": "clip.wav"})
        assert resp.data == payload
        assert resp.mimetype == mimetype
        assert resp.download_name == download_name

    def test_no_bytes_takes_the_default(self):
        resp = AudioMediaType().media_response({"id": 3, "media_bytes": None, "media_path": None})
        assert resp.data == b""
        assert resp.mimetype == "audio/wav"
        assert resp.download_name == "media_3.wav"
