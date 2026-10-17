"""Identify an audio payload's container from its leading bytes.

Audio media are served as whatever bytes the resolver yields: the original file
verbatim for an MP3 / FLAC / OGG / M4A import, a WAV slice for a clip the
clippers cut, a raw ADTS stream or an MP4 container for an archive member.  A
response labelled ``audio/wav`` regardless of which, with a ``.wav`` download
name, misleads any client that trusts ``Content-Type`` (#4981).  The magic
bytes say which it really is; this module reads them.

Nothing here decodes: it only inspects the first dozen bytes.
"""

from __future__ import annotations

__all__ = ["sniff_audio_format"]

# What an unrecognised (or empty) payload is served as: WAV is the format every
# clip slice and every on-the-fly encode produces, so it is the likeliest
# answer when the signature is unknown.
_DEFAULT = ("audio/wav", ".wav")


def sniff_audio_format(data: bytes | None) -> tuple[str, str]:
    """Return ``(mimetype, extension)`` for an audio payload, by its magic bytes.

    ======================================  ===============  ==========
    Leading bytes                           Mimetype         Extension
    ======================================  ===============  ==========
    ``RIFF``                                ``audio/wav``    ``.wav``
    ``fLaC``                                ``audio/flac``   ``.flac``
    ``OggS``                                ``audio/ogg``    ``.ogg``
    ``ftyp`` at offset 4 (ISO BMFF / MP4)   ``audio/mp4``    ``.m4a``
    ``ID3``, or an MPEG audio frame sync    ``audio/mpeg``   ``.mp3``
    an ADTS frame sync                      ``audio/aac``    ``.aac``
    anything else, or no data               ``audio/wav``    ``.wav``
    ======================================  ===============  ==========

    MPEG audio and ADTS share a sync word, so they are told apart by the two
    *layer* bits that follow it: ADTS fixes them at ``00``, which MPEG audio
    reserves, so ``00`` means AAC and any other value means MP3.  An MP4 is
    labelled ``audio/mp4`` even when it also carries video, because this is the
    type an ``<audio>`` element needs to play its audio track.
    """
    head = data[:12] if data else b""
    magic = head[:4]
    if magic == b"RIFF":
        return _DEFAULT
    if magic == b"fLaC":
        return ("audio/flac", ".flac")
    if magic == b"OggS":
        return ("audio/ogg", ".ogg")
    if head[4:8] == b"ftyp":
        return ("audio/mp4", ".m4a")
    if head[:3] == b"ID3":
        return ("audio/mpeg", ".mp3")
    if len(head) >= 2 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0:
        if head[1] & 0xF6 == 0xF0:  # 12-bit sync, layer 00: an ADTS header
            return ("audio/aac", ".aac")
        if head[1] & 0x06:  # 11-bit sync, a real layer: an MPEG audio frame
            return ("audio/mpeg", ".mp3")
    return _DEFAULT
