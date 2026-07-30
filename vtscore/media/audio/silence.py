"""Non-silent interval detection, shared by the silence clipper and cleaner.

Two consumers ask the same question - *where is this recording actually
audible?* - and answer it differently:

* :class:`~vtscore.media.audio.clipper.SoundSilenceClipper` emits **one clip per
  interval**, dropping the gaps between them.
* :class:`~vtscore.media.audio.cleaner.AudioSilenceTrimCleaner` keeps the single
  span ``[first_start, last_end]``, dropping only the lead-in and tail.

The detector itself lives here once so the two can never drift apart on what
counts as silence.
"""

from __future__ import annotations


def detect_nonsilent_segments(
    media_bytes: bytes,
    *,
    top_db: float,
    pad: float = 0.0,
    min_duration: float = 0.0,
) -> list[tuple[float, float]] | None:
    """Detect non-silent ``(start, end)`` ranges (seconds) in *media_bytes*.

    Audio quieter than *top_db* below the clip's reference level counts as
    silence.  Each surviving interval is widened by *pad* seconds on both sides
    (clamped to the clip) so attack and decay aren't shaved off, and intervals
    shorter than *min_duration* after padding are discarded.

    Returns ``None`` if the audio cannot be decoded or ``librosa`` is
    unavailable, and an empty list when nothing is left to report - either the
    clip has no detectable silence structure at all, or no interval survived the
    *min_duration* filter.  Callers treat both as "leave the media alone".
    """
    try:
        import librosa  # noqa: PLC0415

        from vtscore.media.audio.decode import decode_audio  # noqa: PLC0415
    except ImportError:
        return None

    try:
        audio_data, sr = decode_audio(media_bytes, sr=None, mono=True)
    except Exception:
        return None

    if audio_data.size == 0 or sr <= 0:
        return None

    try:
        intervals = librosa.effects.split(audio_data, top_db=top_db)
    except Exception:
        return None

    if len(intervals) == 0:
        return []

    total_samples = len(audio_data)
    # librosa's effects.split returns a single full-coverage interval on
    # degenerate input (e.g. pure silence - ref amplitude is zero, so the
    # dB threshold is meaningless).  Treat that as "no segmentation".
    if len(intervals) == 1 and int(intervals[0][0]) <= 0 and int(intervals[0][1]) >= total_samples:
        return []

    total = total_samples / sr
    segments: list[tuple[float, float]] = []
    for s0, s1 in intervals:
        t0 = max(0.0, float(s0) / sr - pad)
        t1 = min(total, float(s1) / sr + pad)
        if t1 - t0 >= min_duration:
            segments.append((t0, t1))
    return segments
