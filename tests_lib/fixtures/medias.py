"""Test media generation."""

import hashlib
import io
import os

from vtscore.media.audio.audio_generator import generate_wav
from vtscore.config import DATA_DIR

NUM_MEDIAS = 20
from vtscore.embedding import embed_audio_file
from vtsearch.state import medias


def _worker_suffix():
    """Return a suffix unique to the current xdist worker (or empty if not running under xdist).

    Avoids races when pytest-xdist spawns multiple worker processes that all
    call :func:`init_medias` simultaneously.
    """
    worker = os.environ.get("PYTEST_XDIST_WORKER", "")
    return f".{worker}" if worker else ""


def _fake_waveform_thumbnail() -> bytes:
    """A tiny valid PNG standing in for a real waveform thumbnail.

    The real ``generate_waveform_thumbnail()`` imports librosa (and its numba
    stack) and decodes every generated WAV — seconds of startup per xdist
    worker per run, spent on pixels no test inspects beyond "non-empty PNG
    bytes".  A solid-colour PIL PNG satisfies every consumer; routes that
    exercise real on-the-fly thumbnailing pop ``thumbnail_bytes`` first and
    still hit the real code path.
    """
    from PIL import Image  # noqa: PLC0415

    buf = io.BytesIO()
    Image.new("RGB", (80, 80), (32, 32, 32)).save(buf, format="PNG")
    return buf.getvalue()


def init_medias():
    """Generate test medias with (conftest-stubbed) embeddings.

    ``embed_audio_file`` is patched with a deterministic fake by conftest
    before this runs, so embedding is cheap; each WAV is still written to a
    temp file because the fake derives its seed from the file's leading bytes
    (distinct audio → distinct, worker-independent vectors).
    """
    DATA_DIR.mkdir(exist_ok=True)
    temp_path = DATA_DIR / f"temp_embed{_worker_suffix()}.wav"
    thumbnail = _fake_waveform_thumbnail()

    for i in range(1, NUM_MEDIAS + 1):
        freq = 200 + (i - 1) * 50  # 200 Hz .. 1150 Hz
        duration = round(1.0 + (i % 5) * 0.5, 1)  # 1.0 – 3.0 s
        wav_bytes = generate_wav(freq, duration)
        temp_path.write_bytes(wav_bytes)
        embedding = embed_audio_file(temp_path)

        fname = f"test_media_{i}.wav"
        medias[i] = {
            "id": i,
            "media_type": "audio",
            "embedder": "clap",
            "frequency": freq,
            "duration": duration,
            "file_size": len(wav_bytes),
            "md5": hashlib.md5(wav_bytes).hexdigest(),
            "embeddings": {"clap": embedding},
            "media_bytes": wav_bytes,
            "thumbnail_bytes": thumbnail,
            "filename": fname,
            "category": "test",
            "origin": {"importer": "test", "params": {}},
            "origin_name": fname,
        }

    # Clean up temp file
    if temp_path.exists():
        temp_path.unlink()
