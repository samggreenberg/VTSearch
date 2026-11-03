"""Audio cleaners - 1→1 cleanup gates run on each clip before embedding."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from vtscore.media.clipping.cleaner import MediaCleaner

if TYPE_CHECKING:
    from vtscore.media.audio.loudness import PcmAudio

log = logging.getLogger(__name__)

#: Don't rewrite the payload for a trim this small (seconds, summed over both
#: ends).  Re-encoding a clip to shave a few milliseconds costs a second copy of
#: its bytes (the ``original_*`` snapshot) and buys the embedder nothing.
_MIN_TRIM_SECONDS = 0.1

#: Don't rewrite the payload for a gain this small (dB): about the smallest
#: loudness step a listener can hear, and for the same reason as
#: :data:`_MIN_TRIM_SECONDS` not worth a second copy of the clip.
_MIN_GAIN_DB = 1.0

#: A clip whose loudest sample is below this (dBFS) is treated as silence and
#: left alone.  Boosting it would only turn its noise floor into loud hiss.
_SILENCE_FLOOR_DBFS = -60.0

#: The two ``mode`` values :class:`AudioLoudnessNormalizeCleaner` accepts.
_LOUDNESS_MODES = ("rms", "peak")


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

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["top_db"] = self._top_db
        d["pad"] = self._pad
        return d

    def clean(self, media: dict[str, Any]) -> dict[str, Any]:
        """Return *media* with its head/tail silence removed, or unchanged.

        Unchanged when there are no bytes, the audio can't be decoded or sliced,
        no silence structure is detectable (a clip that is uniformly loud, or
        uniformly silent), or the two ends together account for less than
        :data:`_MIN_TRIM_SECONDS`.
        """
        from vtscore.media.audio.wav import wav_duration, wav_slice  # noqa: PLC0415
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
            total = wav_duration(media_bytes)
            if t0 < _MIN_TRIM_SECONDS and (total - t1) < _MIN_TRIM_SECONDS:
                return media  # nothing meaningful at either end
            trimmed = wav_slice(media_bytes, t0, t1)
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


class AudioLoudnessNormalizeCleaner(MediaCleaner):
    """Scale every clip to one loudness, so a quiet recording embeds like a loud one.

    CLAP, like every log-mel audio embedder, is not level-invariant.  A gain
    shifts every bin of the log-mel spectrogram by the same number of dB, the
    model's input batch-norm passes that shift straight through, and so a
    recording made 20 dB quieter embeds somewhere else than the same sound at a
    typical level.  This gate applies one gain per clip so every clip arrives at
    the same level.  It is a gain and nothing else: no compression, no limiting,
    no EQ, and never a clipped sample.

    ``mode`` says what "the same level" means:

    ``"rms"`` (the default)
        Scale so the clip's gated RMS is ``rms_db``, but never so far that its
        loudest sample would pass ``peak_db``.
    ``"peak"``
        Scale so the clip's loudest sample sits at ``peak_db``.

    **Why RMS is the default.**  RMS measures what the embedder sees.  A log-mel
    spectrogram is a map of energy, and RMS is energy; the peak is a single
    sample.  Two clips peaking at the same level can differ by 15-20 dB in
    energy (a steady drone against a lone drum hit), so peak normalization
    leaves exactly the spread this gate exists to remove, while gated-RMS
    clips land at one energy whatever their shape.  ``peak_db`` doubles as RMS
    mode's ceiling, so the choice costs no safety: a spiky clip that would clip
    before reaching ``rms_db`` stops at ``peak_db``, which is what peak mode
    would have done with it.  The -20 dBFS target leaves 19 dB under that
    ceiling, enough for the crest factor of speech and most music, so the
    ceiling binds only on genuinely spiky clips.

    The RMS is *gated*
    (:meth:`~vtscore.media.audio.loudness.PcmAudio.gated_rms_dbfs`): windows
    more than :data:`~vtscore.media.audio.loudness.GATE_DB` below the loudest
    are left out, so a short sound is measured at its own level rather than
    diluted by the quiet around it.

    **Order.**  The result does not depend on whether
    :class:`AudioSilenceTrimCleaner` runs before or after this gate.  The trim
    only removes audio far below the clip's loudest frame, which holds neither
    the peak nor any window the gated RMS counts, so it never changes the gain
    chosen here; and its threshold is relative to that loudest frame, so a gain
    never changes what it cuts.

    A clip is left unchanged when it can't be decoded as WAV (the silence trim's
    own limit, via :func:`~vtscore.media.audio.wav.open_wav`), when its peak is
    below :data:`_SILENCE_FLOOR_DBFS` (silence, including all-zero audio), or
    when the gain would move its level by at most :data:`_MIN_GAIN_DB`.  A rewritten clip is
    PCM WAV in the source's channel count, sample rate and sample width; an MP3
    or FLAC payload comes out as 16-bit WAV, as it does from the silence trim.

    Parameters:

    mode:
        ``"rms"`` (default) or ``"peak"``, case-insensitive.
    rms_db:
        RMS mode's target level, in dBFS.  Defaults to -20.0.
    peak_db:
        Peak mode's target for the loudest sample, and RMS mode's ceiling, in
        dBFS.  Defaults to -1.0.
    """

    def __init__(self, mode: str = "rms", rms_db: float = -20.0, peak_db: float = -1.0) -> None:
        mode = mode.strip().lower()
        if mode not in _LOUDNESS_MODES:
            raise ValueError(f"mode must be one of {', '.join(_LOUDNESS_MODES)}, not {mode!r}")
        if rms_db > 0:
            raise ValueError("rms_db must be at most 0 dBFS")
        if peak_db > 0:
            raise ValueError("peak_db must be at most 0 dBFS")
        self._mode = mode
        self._rms_db = rms_db
        self._peak_db = peak_db

    @property
    def name(self) -> str:
        return "audio_loudness_normalize"

    @property
    def media_type(self) -> str:
        return "audio"

    @property
    def display_name(self) -> str:
        return "Loudness Normalize"

    @property
    def description(self) -> str:
        return (
            "Scale each clip to one loudness so quiet recordings embed like loud ones. A gain only: nothing is "
            "compressed, and nothing clips."
        )

    @property
    def summary_template(self) -> str:
        return "Scale each clip to one loudness ({mode}): RMS {rms_db} dBFS, peaks at most {peak_db} dBFS."

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def rms_db(self) -> float:
        return self._rms_db

    @property
    def peak_db(self) -> float:
        return self._peak_db

    @property
    def parameters(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "mode",
                "label": "Mode (rms or peak)",
                "description": (
                    "rms brings each clip's loudness to the target RMS, never letting a peak pass the peak level; "
                    "peak brings each clip's loudest sample to the peak level."
                ),
                "type": "string",
                "default": self._mode,
            },
            {
                "key": "rms_db",
                "label": "Target RMS (dBFS)",
                "description": "The loudness every clip is scaled to in rms mode.",
                "type": "number",
                "default": self._rms_db,
                "min": -60,
                "max": 0,
                "step": 1,
            },
            {
                "key": "peak_db",
                "label": "Peak level (dBFS)",
                "description": (
                    "In peak mode, where each clip's loudest sample ends up. In rms mode, a ceiling: no clip is "
                    "boosted so far that a sample passes it."
                ),
                "type": "number",
                "default": self._peak_db,
                "min": -60,
                "max": 0,
                "step": 0.5,
            },
        ]

    def with_params(self, params: dict[str, Any]) -> "AudioLoudnessNormalizeCleaner":
        return AudioLoudnessNormalizeCleaner(
            mode=str(params.get("mode", self._mode)),
            rms_db=float(params.get("rms_db", self._rms_db)),
            peak_db=float(params.get("peak_db", self._peak_db)),
        )

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["mode"] = self._mode
        d["rms_db"] = self._rms_db
        d["peak_db"] = self._peak_db
        return d

    def _gain_db(self, audio: PcmAudio) -> float | None:
        """Return the gain (dB) this gate would apply to *audio*, or ``None`` to leave it.

        Rounded to 0.01 dB, so float noise in the measurement can't change the
        output bytes from one run to the next.
        """
        peak = audio.peak_dbfs
        if peak < _SILENCE_FLOOR_DBFS:
            return None
        if self._mode == "peak":
            gain = self._peak_db - peak
        else:
            gain = self._rms_db - audio.gated_rms_dbfs()
            if gain > 0:
                # Boost only as far as the ceiling allows; a clip already past
                # it is not boosted, but never attenuated on its account either.
                gain = min(gain, max(self._peak_db - peak, 0.0))
        gain = round(gain, 2)
        return None if abs(gain) <= _MIN_GAIN_DB else gain

    def clean(self, media: dict[str, Any]) -> dict[str, Any]:
        """Return *media* scaled to the target level, or unchanged (see the class docstring)."""
        from vtscore.media.audio.loudness import PcmAudio  # noqa: PLC0415

        payload = media.get("media_bytes")
        if not isinstance(payload, (bytes, bytearray)) or not payload:
            return media

        try:
            audio = PcmAudio.from_wav(bytes(payload))
        except Exception:
            log.debug("audio_loudness_normalize: undecodable payload, leaving unchanged", exc_info=True)
            return media

        gain = self._gain_db(audio)
        if gain is None:
            return media

        normalized = audio.with_gain(gain).to_wav()
        cleaned = dict(media)
        cleaned["media_bytes"] = normalized
        cleaned["file_size"] = len(normalized)
        return cleaned
