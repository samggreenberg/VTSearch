"""Audio clippers — tile or pass-through audio media."""

from __future__ import annotations

import io
import math
import os
import wave
from pathlib import Path
from typing import Any

from vtscore.media.clipper import MediaClipper


def _wav_duration(wav_bytes: bytes) -> float:
    """Return the duration in seconds of a WAV byte string."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def _wav_slice(wav_bytes: bytes, start: float, end: float) -> bytes:
    """Extract a [start, end) slice from a WAV byte string.

    Returns a new WAV byte string containing only the requested segment.
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        start_frame = int(start * sr)
        end_frame = int(end * sr)
        total_frames = wf.getnframes()
        start_frame = max(0, min(start_frame, total_frames))
        end_frame = max(start_frame, min(end_frame, total_frames))
        wf.setpos(start_frame)
        frames = wf.readframes(end_frame - start_frame)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(n_channels)
        out.setsampwidth(sampwidth)
        out.setframerate(sr)
        out.writeframes(frames)
    return buf.getvalue()


class SoundDefaultClipper(MediaClipper):
    """Returns the audio media unchanged."""

    @property
    def name(self) -> str:
        return "sound_default"

    @property
    def display_name(self) -> str:
        return "None"

    @property
    def media_type(self) -> str:
        return "audio"

    @property
    def description(self) -> str:
        return "Import each audio file as-is, without splitting."

    def clip(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        return [media]


class SoundTilingClipper(MediaClipper):
    """Tile audio into equally-spaced segments of a given duration.

    ``SoundTilingClipper(2)`` tiles a 9.5 s clip into five 2 s segments
    whose start times are equally spaced so the first starts at 0 and the
    last ends at 9.5 s (with a little overlap between neighbours when the
    total duration is not an exact multiple of the segment size).

    If *min_overlap* is set, the clipper ensures that consecutive segments
    overlap by at least that many seconds (producing more tiles when needed).

    If the audio is shorter than or equal to *duration*, a single segment
    covering the full audio is returned.
    """

    def __init__(self, duration: float, min_overlap: float = 0.0) -> None:
        if duration <= 0:
            raise ValueError("duration must be positive")
        if min_overlap < 0:
            raise ValueError("min_overlap must be non-negative")
        if min_overlap >= duration:
            raise ValueError("min_overlap must be less than duration")
        self._duration = duration
        self._min_overlap = min_overlap

    @property
    def name(self) -> str:
        return "sound_tiling"

    @property
    def media_type(self) -> str:
        return "audio"

    @property
    def description(self) -> str:
        return "Split each audio file into fixed-length overlapping segments."

    @property
    def summary_template(self) -> str:
        return "Cut each audio file into {duration}s tiles (min overlap {min_overlap}s)."

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def min_overlap(self) -> float:
        return self._min_overlap

    def clip(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        wav_bytes = media.get("media_bytes")
        if wav_bytes is None:
            return [media]

        total = _wav_duration(wav_bytes)
        seg = self._duration

        if total <= seg:
            return [media]

        max_stride = seg - self._min_overlap
        n_tiles = max(1, math.ceil((total - seg) / max_stride) + 1)
        # Space n_tiles segments so that the first starts at 0 and the last
        # ends at *total*.  When n_tiles == 1 this degenerates to [0, seg).
        if n_tiles == 1:
            starts = [0.0]
        else:
            starts = [i * (total - seg) / (n_tiles - 1) for i in range(n_tiles)]

        results: list[dict[str, Any]] = []
        for idx, t0 in enumerate(starts):
            t1 = t0 + seg
            sliced = _wav_slice(wav_bytes, t0, t1)
            tile = dict(media)
            tile["media_bytes"] = sliced
            tile["duration"] = round(t1 - t0, 6)
            tile["file_size"] = len(sliced)
            tile["clip_index"] = idx
            tile["clip_start"] = round(t0, 6)
            tile["clip_end"] = round(t1, 6)
            results.append(tile)
        return results

    @property
    def parameters(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "duration",
                "label": "Clip length (seconds)",
                "description": "Duration of each audio segment in seconds.",
                "type": "number",
                "default": self._duration,
                "min": 0.1,
                "max": 300,
                "step": 0.1,
            },
            {
                "key": "min_overlap",
                "label": "Minimum overlap (seconds)",
                "description": "Minimum overlap between consecutive segments. Higher values produce more tiles.",
                "type": "number",
                "default": self._min_overlap,
                "min": 0,
                "max": 299.9,
                "step": 0.1,
            },
        ]

    def with_params(self, params: dict[str, Any]) -> "SoundTilingClipper":
        duration = float(params.get("duration", self._duration))
        min_overlap = float(params.get("min_overlap", self._min_overlap))
        return SoundTilingClipper(duration, min_overlap)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["duration"] = self._duration
        d["min_overlap"] = self._min_overlap
        return d


class SoundAutoClipper(MediaClipper):
    """Pass-through for short audio, tile longer audio.

    Designed as the recommended default in the importer picker. The
    decision is made per item: if a media's duration exceeds
    *threshold*, that item is clipped by :class:`SoundTilingClipper`
    with the configured *tile_duration*; otherwise it passes through
    via :class:`SoundDefaultClipper`. Different items in the same
    dataset can take different branches.

    The chosen concrete clipper is what gets recorded in each clip's
    origin, so cross-dataset replay is deterministic.
    """

    def __init__(self, threshold: float = 30.0, tile_duration: float = 10.0) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        if tile_duration <= 0:
            raise ValueError("tile_duration must be positive")
        self._threshold = threshold
        self._tile_duration = tile_duration

    @property
    def name(self) -> str:
        return "sound_auto"

    @property
    def media_type(self) -> str:
        return "audio"

    @property
    def display_name(self) -> str:
        return "Auto (recommended)"

    @property
    def description(self) -> str:
        return (
            f"Pass short audio through unchanged; tile audio longer than "
            f"{self._threshold:g}s into {self._tile_duration:g}s segments."
        )

    @property
    def summary_template(self) -> str:
        return "Cut into {tile_duration}s tiles when audio is over {threshold}s."

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def tile_duration(self) -> float:
        return self._tile_duration

    def resolve_for_media(self, media: dict[str, Any]) -> "MediaClipper":
        duration = float(media.get("duration", 0) or 0)
        if duration <= 0:
            wav_bytes = media.get("media_bytes")
            if wav_bytes is not None:
                try:
                    duration = _wav_duration(wav_bytes)
                except Exception:
                    duration = 0.0
        if duration > self._threshold:
            return SoundTilingClipper(self._tile_duration)
        return SoundDefaultClipper()

    def clip(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        # Direct-call path (outside the load pipeline).  The pipeline
        # uses resolve_for_media() then calls clip() on the concrete
        # result, so this method only runs when someone uses
        # SoundAutoClipper().clip(media) directly.
        return self.resolve_for_media(media).clip(media)

    @property
    def parameters(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "threshold",
                "label": "Auto-tile threshold (seconds)",
                "description": "Audio longer than this is automatically tiled into segments.",
                "type": "number",
                "default": self._threshold,
                "min": 1,
                "max": 600,
                "step": 1,
            },
            {
                "key": "tile_duration",
                "label": "Tile length when tiling (seconds)",
                "description": "Segment length used when auto-tiling is triggered.",
                "type": "number",
                "default": self._tile_duration,
                "min": 0.5,
                "max": 300,
                "step": 0.5,
            },
        ]

    def with_params(self, params: dict[str, Any]) -> "SoundAutoClipper":
        threshold = float(params.get("threshold", self._threshold))
        tile_duration = float(params.get("tile_duration", self._tile_duration))
        return SoundAutoClipper(threshold=threshold, tile_duration=tile_duration)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["threshold"] = self._threshold
        d["tile_duration"] = self._tile_duration
        return d


class SoundSilenceClipper(MediaClipper):
    """Split audio into non-silent segments via :func:`librosa.effects.split`.

    Detects silence using an amplitude threshold (*top_db* dB below the
    reference) and returns one clip per non-silent interval — intro/outro
    silence is dropped automatically.  Suitable for podcasts, voice
    recordings, and sound-event datasets where each "thing" is separated
    by quiet.

    Parameters
    ----------
    top_db : float
        Threshold in dB below the reference to consider as silence.
        Lower values are more aggressive (more, shorter clips).
        Defaults to 40.
    min_clip_duration : float
        Non-silent intervals shorter than this (in seconds) are discarded,
        suppressing noise-spike micro-clips.  Defaults to 0.3.
    pad : float
        Padding (seconds) added on each side of every non-silent interval
        so the attack/decay of the audible content isn't trimmed.
        Defaults to 0.05.

    If ``librosa`` is unavailable, the audio cannot be decoded, or no
    non-silent intervals survive the *min_clip_duration* filter, the media
    is returned unchanged (single-element list).
    """

    def __init__(
        self,
        top_db: float = 40.0,
        min_clip_duration: float = 0.3,
        pad: float = 0.05,
    ) -> None:
        if top_db <= 0:
            raise ValueError("top_db must be positive")
        if min_clip_duration < 0:
            raise ValueError("min_clip_duration must be non-negative")
        if pad < 0:
            raise ValueError("pad must be non-negative")
        self._top_db = top_db
        self._min_clip_duration = min_clip_duration
        self._pad = pad

    @property
    def name(self) -> str:
        return "sound_silence"

    @property
    def media_type(self) -> str:
        return "audio"

    @property
    def description(self) -> str:
        return "Split each audio file into non-silent segments. Drops intro/outro silence."

    @property
    def summary_template(self) -> str:
        return (
            "Split each audio file at silences quieter than {top_db}dB; drop clips shorter than {min_clip_duration}s."
        )

    @property
    def top_db(self) -> float:
        return self._top_db

    @property
    def min_clip_duration(self) -> float:
        return self._min_clip_duration

    @property
    def pad(self) -> float:
        return self._pad

    def _detect_segments(self, media_bytes: bytes) -> list[tuple[float, float]] | None:
        """Detect non-silent ``(start, end)`` ranges (seconds) in *media_bytes*.

        Returns ``None`` if the audio cannot be decoded or librosa is
        unavailable.  Returns an empty list if no intervals survive the
        ``min_clip_duration`` filter.
        """
        try:
            import librosa  # noqa: PLC0415
        except ImportError:
            return None

        try:
            audio_data, sr = librosa.load(io.BytesIO(media_bytes), sr=None, mono=True)
        except Exception:
            return None

        if audio_data.size == 0 or sr <= 0:
            return None

        try:
            intervals = librosa.effects.split(audio_data, top_db=self._top_db)
        except Exception:
            return None

        if len(intervals) == 0:
            return []

        total_samples = len(audio_data)
        # librosa's effects.split returns a single full-coverage interval on
        # degenerate input (e.g. pure silence — ref amplitude is zero, so the
        # dB threshold is meaningless).  Treat that as "no segmentation".
        if len(intervals) == 1 and int(intervals[0][0]) <= 0 and int(intervals[0][1]) >= total_samples:
            return []

        total = total_samples / sr
        segments: list[tuple[float, float]] = []
        for s0, s1 in intervals:
            t0 = max(0.0, float(s0) / sr - self._pad)
            t1 = min(total, float(s1) / sr + self._pad)
            if t1 - t0 >= self._min_clip_duration:
                segments.append((t0, t1))
        return segments

    def clip(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        media_bytes = media.get("media_bytes")
        if media_bytes is None:
            return [media]

        segments = self._detect_segments(media_bytes)
        if not segments:
            return [media]

        try:
            results: list[dict[str, Any]] = []
            for idx, (t0, t1) in enumerate(segments):
                sliced = _wav_slice(media_bytes, t0, t1)
                clip = dict(media)
                clip["media_bytes"] = sliced
                clip["duration"] = round(t1 - t0, 6)
                clip["file_size"] = len(sliced)
                clip["clip_index"] = idx
                clip["clip_start"] = round(t0, 6)
                clip["clip_end"] = round(t1, 6)
                results.append(clip)
            return results
        except Exception:
            return [media]

    @property
    def parameters(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "top_db",
                "label": "Silence threshold (dB)",
                "description": (
                    "Audio quieter than this many dB below the reference is treated as silence. "
                    "Lower values are more aggressive — more, shorter clips."
                ),
                "type": "number",
                "default": self._top_db,
                "min": 5,
                "max": 80,
                "step": 1,
            },
            {
                "key": "min_clip_duration",
                "label": "Minimum clip length (seconds)",
                "description": "Non-silent intervals shorter than this are discarded.",
                "type": "number",
                "default": self._min_clip_duration,
                "min": 0,
                "max": 60,
                "step": 0.1,
            },
            {
                "key": "pad",
                "label": "Padding (seconds)",
                "description": "Extra audio kept on each side of every non-silent interval so attack/decay isn't trimmed.",
                "type": "number",
                "default": self._pad,
                "min": 0,
                "max": 5,
                "step": 0.05,
            },
        ]

    def with_params(self, params: dict[str, Any]) -> "SoundSilenceClipper":
        top_db = float(params.get("top_db", self._top_db))
        min_clip_duration = float(params.get("min_clip_duration", self._min_clip_duration))
        pad = float(params.get("pad", self._pad))
        return SoundSilenceClipper(top_db=top_db, min_clip_duration=min_clip_duration, pad=pad)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["top_db"] = self._top_db
        d["min_clip_duration"] = self._min_clip_duration
        d["pad"] = self._pad
        return d


class SoundClipClipper(MediaClipper):
    """Extract a single user-specified ``[start, end)`` slice from an audio media.

    Unlike :class:`SoundTilingClipper`, which auto-tiles a clip into many
    equally-spaced segments, ``SoundClipClipper`` returns exactly one tile
    bounded by the start and end times the caller provides.  The intended
    use is user-driven cropping — e.g. picking a sub-region of an example
    sound to drive a similarity search or a training example.

    The returned media dict carries the same ``clip_start`` / ``clip_end``
    fields as a tiling clip, so downstream code (embedding, learning, label
    export) treats the cropped result as a first-class clip.
    """

    def __init__(self, start: float, end: float) -> None:
        if start < 0:
            raise ValueError("start must be non-negative")
        if end <= start:
            raise ValueError("end must be greater than start")
        self._start = start
        self._end = end

    @property
    def name(self) -> str:
        return "sound_clip"

    @property
    def media_type(self) -> str:
        return "audio"

    @property
    def description(self) -> str:
        return "Extract a single user-specified [start, end) range from the audio."

    @property
    def summary_template(self) -> str:
        return "Extract audio from {start}s to {end}s."

    @property
    def start(self) -> float:
        return self._start

    @property
    def end(self) -> float:
        return self._end

    def clip(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        wav_bytes = media.get("media_bytes")
        if wav_bytes is None:
            return [media]

        total = _wav_duration(wav_bytes)
        t0 = max(0.0, min(self._start, total))
        t1 = max(t0, min(self._end, total))
        if t1 <= t0:
            return [media]

        sliced = _wav_slice(wav_bytes, t0, t1)
        clip = dict(media)
        clip["media_bytes"] = sliced
        clip["duration"] = round(t1 - t0, 6)
        clip["file_size"] = len(sliced)
        clip["clip_index"] = 0
        clip["clip_start"] = round(t0, 6)
        clip["clip_end"] = round(t1, 6)
        return [clip]

    @property
    def parameters(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "start",
                "label": "Start (seconds)",
                "description": "Start time of the clip in seconds.",
                "type": "number",
                "default": self._start,
                "min": 0,
                "step": 0.01,
            },
            {
                "key": "end",
                "label": "End (seconds)",
                "description": "End time of the clip in seconds.",
                "type": "number",
                "default": self._end,
                "min": 0,
                "step": 0.01,
            },
        ]

    def with_params(self, params: dict[str, Any]) -> "SoundClipClipper":
        start = float(params.get("start", self._start))
        end = float(params.get("end", self._end))
        return SoundClipClipper(start, end)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["start"] = self._start
        d["end"] = self._end
        return d


class SoundSpeechActivityClipper(MediaClipper):
    """Split audio into one clip per detected speech turn using Silero VAD.

    Loads the `snakers4/silero-vad` model via :func:`torch.hub.load` and runs
    voice-activity detection on a 16 kHz mono downmix of the input.  Returned
    speech intervals are sliced out of the **original** WAV bytes (preserving
    the source sample rate / channels) and emitted as separate clips, the
    same way :class:`SoundSilenceClipper` works — non-speech gaps and
    intro/outro silence are dropped automatically.  Designed for podcasts,
    interviews, voice memos, and lecture recordings where each unit of
    interest is a contiguous speech turn.

    Parameters
    ----------
    threshold : float
        Silero VAD confidence threshold in ``[0, 1]``.  Higher values are
        more conservative (only louder/clearer speech survives).
        Defaults to 0.5.
    min_clip_duration : float
        Speech intervals shorter than this (in seconds, **after** padding)
        are discarded, suppressing single-syllable false positives.
        Defaults to 0.3.
    pad : float
        Padding (seconds) added on each side of every speech interval so
        word onsets and tails aren't trimmed.  Defaults to 0.05.

    If ``torch`` / Silero is unavailable, the audio cannot be decoded, or
    no speech intervals survive the ``min_clip_duration`` filter, the media
    is returned unchanged (single-element list).
    """

    _SILERO_SR = 16000

    def __init__(
        self,
        threshold: float = 0.5,
        min_clip_duration: float = 0.3,
        pad: float = 0.05,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if min_clip_duration < 0:
            raise ValueError("min_clip_duration must be non-negative")
        if pad < 0:
            raise ValueError("pad must be non-negative")
        self._threshold = float(threshold)
        self._min_clip_duration = float(min_clip_duration)
        self._pad = float(pad)
        self._model: Any = None
        self._get_speech_timestamps: Any = None

    @property
    def name(self) -> str:
        return "sound_speech_activity"

    @property
    def media_type(self) -> str:
        return "audio"

    @property
    def description(self) -> str:
        return "Split each audio file into one clip per speech turn using Silero VAD. Good for podcasts."

    @property
    def summary_template(self) -> str:
        return (
            "Split each audio file at detected speech turns (VAD threshold {threshold}, min clip {min_clip_duration}s)."
        )

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def min_clip_duration(self) -> float:
        return self._min_clip_duration

    @property
    def pad(self) -> float:
        return self._pad

    def _load_model(self) -> bool:
        """Lazy-load the Silero VAD model. Returns False if unavailable."""
        if self._model is not None and self._get_speech_timestamps is not None:
            return True
        try:
            import torch  # noqa: PLC0415
            import torch.hub  # noqa: F401, PLC0415

            from vtscore.config import MODELS_CACHE_DIR  # noqa: PLC0415
        except ImportError:
            return False

        os.environ.setdefault("TORCH_HOME", str(Path(MODELS_CACHE_DIR).expanduser()))
        try:
            # ``torch.hub.load`` is typed to return ``object``; cast to Any so
            # we can unpack the (model, utils) tuple and index into utils.
            loaded: Any = torch.hub.load(
                "snakers4/silero-vad",
                "silero_vad",
                source="github",
                trust_repo=True,  # pyright: ignore[reportArgumentType]
            )
            model, utils = loaded
            get_speech_timestamps = utils[0]
        except Exception:
            return False

        self._model = model
        self._get_speech_timestamps = get_speech_timestamps
        return True

    def _detect_speech_intervals(self, media_bytes: bytes) -> list[tuple[float, float]] | None:
        """Detect speech ``(start, end)`` ranges (seconds) in *media_bytes*.

        Returns ``None`` if the audio cannot be decoded or Silero is
        unavailable.  Returns an empty list if no intervals survive the
        ``min_clip_duration`` filter.
        """
        try:
            import librosa  # noqa: PLC0415
        except ImportError:
            return None

        if not self._load_model():
            return None

        try:
            audio_data, _ = librosa.load(io.BytesIO(media_bytes), sr=self._SILERO_SR, mono=True)
        except Exception:
            return None
        if audio_data.size == 0:
            return None

        try:
            import torch  # noqa: PLC0415

            wav = torch.from_numpy(audio_data)
            raw = self._get_speech_timestamps(
                wav,
                self._model,
                sampling_rate=self._SILERO_SR,
                threshold=self._threshold,
                return_seconds=True,
            )
        except Exception:
            return None

        if not raw:
            return []

        total = audio_data.size / self._SILERO_SR
        segments: list[tuple[float, float]] = []
        for ts in raw:
            t0 = max(0.0, float(ts["start"]) - self._pad)
            t1 = min(total, float(ts["end"]) + self._pad)
            if t1 - t0 >= self._min_clip_duration:
                segments.append((t0, t1))
        return segments

    def clip(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        media_bytes = media.get("media_bytes")
        if media_bytes is None:
            return [media]

        segments = self._detect_speech_intervals(media_bytes)
        if not segments:
            return [media]

        try:
            results: list[dict[str, Any]] = []
            for idx, (t0, t1) in enumerate(segments):
                sliced = _wav_slice(media_bytes, t0, t1)
                clip = dict(media)
                clip["media_bytes"] = sliced
                clip["duration"] = round(t1 - t0, 6)
                clip["file_size"] = len(sliced)
                clip["clip_index"] = idx
                clip["clip_start"] = round(t0, 6)
                clip["clip_end"] = round(t1, 6)
                results.append(clip)
            return results
        except Exception:
            return [media]

    @property
    def parameters(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "threshold",
                "label": "VAD threshold",
                "description": (
                    "Silero VAD confidence threshold (0–1). Higher values are more conservative — "
                    "only louder, clearer speech is kept."
                ),
                "type": "number",
                "default": self._threshold,
                "min": 0.05,
                "max": 1.0,
                "step": 0.05,
            },
            {
                "key": "min_clip_duration",
                "label": "Minimum clip length (seconds)",
                "description": "Speech intervals shorter than this are discarded.",
                "type": "number",
                "default": self._min_clip_duration,
                "min": 0,
                "max": 60,
                "step": 0.1,
            },
            {
                "key": "pad",
                "label": "Padding (seconds)",
                "description": "Extra audio kept on each side of every speech interval so word onsets/tails aren't trimmed.",
                "type": "number",
                "default": self._pad,
                "min": 0,
                "max": 5,
                "step": 0.05,
            },
        ]

    def with_params(self, params: dict[str, Any]) -> "SoundSpeechActivityClipper":
        threshold = float(params.get("threshold", self._threshold))
        min_clip_duration = float(params.get("min_clip_duration", self._min_clip_duration))
        pad = float(params.get("pad", self._pad))
        return SoundSpeechActivityClipper(threshold=threshold, min_clip_duration=min_clip_duration, pad=pad)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["threshold"] = self._threshold
        d["min_clip_duration"] = self._min_clip_duration
        d["pad"] = self._pad
        return d
