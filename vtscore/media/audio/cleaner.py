"""Audio cleaners - 1→1 cleanup gates run on each clip before embedding."""

from __future__ import annotations

import logging
from typing import Any

from vtscore.media.cleaner import MediaCleaner

log = logging.getLogger(__name__)

#: Don't rewrite the payload for a trim this small (seconds, summed over both
#: ends).  Re-encoding a clip to shave a few milliseconds costs a second copy of
#: its bytes (the ``original_*`` snapshot) and buys the embedder nothing.
_MIN_TRIM_SECONDS = 0.1


class AudioSilenceTrimCleaner(MediaCleaner):
    """Drop the leading and trailing silence from a clip.

    A recording that opens with two seconds of room tone and ends with a
    three-second tail spends a large fraction of the embedder's fixed-length
    window on nothing.  This gate keeps the single span
    ``[first_start, last_end]`` of the audible material and discards only what
    lies outside it - internal pauses are left exactly as they are, because they
    are part of the content's rhythm.

    Silence detection is the same machinery
    (:func:`~vtscore.media.audio.silence.detect_nonsilent_segments`) that backs
    :class:`~vtscore.media.audio.clipper.SoundSilenceClipper`, so the two agree
    on what counts as quiet.  Running both is harmless: the clipper's output
    already starts and ends on audible content, so this gate no-ops on it.

    Parameters mirror the clipper's:

    top_db:
        Audio quieter than this many dB below the clip's reference level counts
        as silence.  Defaults to 40.0.
    pad:
        Seconds of the surrounding silence kept on each side, so an attack or
        decay isn't clipped.  Defaults to 0.05.
    """

    def __init__(self, top_db: float = 40.0, pad: float = 0.05) -> None:
        if top_db <= 0:
            raise ValueError("top_db must be positive")
        if pad < 0:
            raise ValueError("pad must be non-negative")
        self._top_db = top_db
        self._pad = pad

    @property
    def name(self) -> str:
        return "audio_silence_trim"

    @property
    def media_type(self) -> str:
        return "audio"

    @property
    def display_name(self) -> str:
        return "Silence Trim"

    @property
    def description(self) -> str:
        return (
            "Trim the silence off the head and tail of each clip so the embedder's window is spent "
            "on audible content. Internal pauses are kept."
        )

    @property
    def summary_template(self) -> str:
        return "Trim head/tail audio quieter than {top_db}dB, keeping {pad}s of padding."

    @property
    def top_db(self) -> float:
        return self._top_db

    @property
    def pad(self) -> float:
        return self._pad

    @property
    def parameters(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "top_db",
                "label": "Silence threshold (dB)",
                "description": (
                    "Audio quieter than this many dB below the reference is treated as silence. "
                    "Lower values are more aggressive - more gets trimmed."
                ),
                "type": "number",
                "default": self._top_db,
                "min": 5,
                "max": 80,
                "step": 1,
            },
            {
                "key": "pad",
                "label": "Padding (seconds)",
                "description": "Silence kept on each side of the audible span so attack/decay isn't trimmed.",
                "type": "number",
                "default": self._pad,
                "min": 0,
                "max": 5,
                "step": 0.05,
            },
        ]

    def with_params(self, params: dict[str, Any]) -> "AudioSilenceTrimCleaner":
        return AudioSilenceTrimCleaner(
            top_db=float(params.get("top_db", self._top_db)),
            pad=float(params.get("pad", self._pad)),
        )

    def clean(self, media: dict[str, Any]) -> dict[str, Any]:
        """Return *media* with its head/tail silence removed, or unchanged.

        Unchanged when there are no bytes, the audio can't be decoded or sliced,
        no silence structure is detectable (a clip that is uniformly loud, or
        uniformly silent), or the two ends together account for less than
        :data:`_MIN_TRIM_SECONDS`.
        """
        from vtscore.media.audio.clipper import _wav_duration, _wav_slice  # noqa: PLC0415
        from vtscore.media.audio.silence import detect_nonsilent_segments  # noqa: PLC0415

        payload = media.get("media_bytes")
        if not isinstance(payload, (bytes, bytearray)) or not payload:
            return media
        media_bytes = bytes(payload)

        segments = detect_nonsilent_segments(media_bytes, top_db=self._top_db, pad=self._pad)
        if not segments:
            return media

        t0, t1 = segments[0][0], segments[-1][1]
        if t1 <= t0:
            return media

        try:
            total = _wav_duration(media_bytes)
            if t0 < _MIN_TRIM_SECONDS and (total - t1) < _MIN_TRIM_SECONDS:
                return media  # nothing meaningful at either end
            trimmed = _wav_slice(media_bytes, t0, t1)
        except Exception:
            log.debug("audio_silence_trim: undecodable payload, leaving unchanged", exc_info=True)
            return media

        if not trimmed:
            return media

        cleaned = dict(media)
        cleaned["media_bytes"] = trimmed
        cleaned["file_size"] = len(trimmed)
        cleaned["duration"] = round(t1 - t0, 6)
        return cleaned
