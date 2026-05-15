"""Extract audio track from a video."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

from vtsearch.converters.base import MediaConverter
from vtsearch.media.audio.ffmpeg import get_ffmpeg_exe
from vtsearch.plugins import PluginField


class Video2AudioMediaConverter(MediaConverter):
    """Extract the audio track from a video file as a WAV.

    Uses ``ffmpeg`` (must be on ``$PATH``) to demux and transcode the
    audio stream to 16-bit PCM WAV at the video's native sample rate.
    Falls back to ``ffprobe`` for duration detection.

    Returns a single-element list with the WAV bytes and duration, or
    an empty list if the video has no audio track or ffmpeg is not
    available.

    User-configurable parameters
    ----------------------------
    ``ffmpeg_timeout``
        Seconds to wait for the ffmpeg subprocess before aborting.
        Defaults to 600 (10 minutes) \u2014 independent of the gunicorn
        worker timeout so long videos don't race the request budget.
    """

    display_name = "Video \u2192 Audio"
    converter_description = "Extract audio tracks from video files"
    fields = [
        PluginField(
            key="ffmpeg_timeout",
            label="ffmpeg timeout (seconds)",
            field_type="text",
            description="Maximum seconds to wait for ffmpeg to extract the audio track.",
            default="600",
            required=False,
        ),
    ]

    @property
    def source_type(self) -> str:
        return "video"

    @property
    def target_type(self) -> str:
        return "audio"

    def convert(self, media: dict[str, Any], params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        try:
            ffmpeg_timeout = float(self.get_param(params, "ffmpeg_timeout") or 600)
        except (TypeError, ValueError):
            ffmpeg_timeout = 600.0
        if ffmpeg_timeout <= 0:
            ffmpeg_timeout = 600.0

        media_bytes = media.get("media_bytes")
        media_path = media.get("media_path")
        filename = media.get("filename", "video.mp4")
        stem = Path(filename).stem

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write source video to a temp file if we only have bytes
            if media_path and Path(media_path).exists():
                src_path = media_path
            elif media_bytes:
                ext = Path(filename).suffix or ".mp4"
                src_path = str(Path(tmpdir) / f"input{ext}")
                with open(src_path, "wb") as f:
                    f.write(media_bytes)
            else:
                return []

            wav_path = str(Path(tmpdir) / "output.wav")

            try:
                ffmpeg = get_ffmpeg_exe()
                subprocess.run(
                    [
                        ffmpeg,
                        "-y",
                        "-i",
                        str(src_path),
                        "-vn",
                        "-acodec",
                        "pcm_s16le",
                        wav_path,
                    ],
                    capture_output=True,
                    timeout=ffmpeg_timeout,
                    check=True,
                )
            except FileNotFoundError:
                print("Video2AudioMediaConverter requires ffmpeg — install it or 'pip install imageio-ffmpeg'")
                return []
            except subprocess.CalledProcessError as e:
                stderr_text = (
                    e.stderr[:500].decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")[:500]
                )
                print(f"ffmpeg failed for {filename}: {stderr_text}")
                return []
            except subprocess.TimeoutExpired:
                print(f"ffmpeg timed out for {filename}")
                return []

            wav_file = Path(wav_path)
            if not wav_file.exists() or wav_file.stat().st_size == 0:
                return []

            wav_bytes = wav_file.read_bytes()

            # Compute duration from WAV header or via librosa
            duration = 0.0
            try:
                import librosa  # noqa: PLC0415

                audio_data, sr = librosa.load(wav_path, sr=None, mono=True)
                duration = len(audio_data) / sr
            except Exception:
                pass

        return [
            {
                "filename": f"{stem}.wav",
                "media_bytes": wav_bytes,
                "duration": duration,
            }
        ]


CONVERTER = Video2AudioMediaConverter()
