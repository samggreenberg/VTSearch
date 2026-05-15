"""Shared test helpers for the VTSearch test suite."""

from __future__ import annotations

import io
import json
import struct
import wave
from pathlib import Path


# ---------------------------------------------------------------------------
# WAV generation helpers
# ---------------------------------------------------------------------------


def make_wav_bytes(frequency: float = 440.0, duration: float = 0.1) -> bytes:
    """Generate a WAV file using the application's audio generator.

    Supports variable ``frequency`` and ``duration`` — useful when tests
    need multiple distinct WAV files (e.g. different frequencies per media).
    """
    from vtsearch.media.audio.audio_generator import generate_wav  # noqa: PLC0415

    return generate_wav(frequency, duration)


def make_raw_wav_bytes() -> bytes:
    """Create a minimal valid WAV file (100 zero-samples) in memory.

    Lighter weight than :func:`make_wav_bytes` — does not depend on the
    ``vtsearch.media.audio.audio_generator`` module.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        samples = struct.pack("<" + "h" * 100, *([0] * 100))
        wf.writeframes(samples)
    return buf.getvalue()


def make_wav_file(tmp_dir: Path, name: str, frequency: float = 440.0) -> Path:
    """Write a WAV file to ``tmp_dir / name`` and return its path."""
    p = tmp_dir / name
    p.write_bytes(make_wav_bytes(frequency))
    return p


# ---------------------------------------------------------------------------
# Dataset pickle helpers
# ---------------------------------------------------------------------------


def make_dataset_file(tmp_path, clips_dict, name: str = "dataset.pkl") -> Path:
    """Export a medias dict to a pickle file and return the path."""
    from vtsearch.datasets.loader import export_dataset_to_file  # noqa: PLC0415

    pkl_bytes = export_dataset_to_file(clips_dict)
    dataset_path = tmp_path / name
    dataset_path.write_bytes(pkl_bytes)
    return dataset_path


# ---------------------------------------------------------------------------
# Non-audio media helpers (for multi-media test fixtures)
# ---------------------------------------------------------------------------


def make_png_bytes(width: int = 16, height: int = 16, color: tuple = (128, 64, 200)) -> bytes:
    """Produce a valid PNG byte string for a solid-color image.

    Falls back to a minimal handcrafted 1×1 PNG if Pillow is unavailable so
    the helpers are still usable in slim test environments.
    """
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        # 1×1 magenta PNG — fixed bytes
        return bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000d49444154789c63f8cf00000003000100" + "5cdd3a7d" + "0000000049454e44ae426082"
        )
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def make_minimal_mp4_bytes() -> bytes:
    """Return a tiny (not a real video) MP4 byte string suitable for testing.

    Has the ftyp header so basic probes recognise the container.  This is NOT a
    playable video — it exists so code that inspects file headers works.
    """
    # ftyp box: size=32, type='ftyp', major_brand='isom', minor_version=0x200,
    # compat='isom','iso2','avc1','mp41'
    return (
        b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2avc1mp41" + b"\x00\x00\x00\x08free"  # padding
    )


def make_minimal_pdf_bytes() -> bytes:
    """Return a minimal valid PDF byte string.

    Attempts to use PyMuPDF (fitz) for a real, renderable single-page PDF; if
    that is unavailable, falls back to a handcrafted PDF that is parseable by
    PyPDF-style readers.
    """
    try:
        import fitz  # type: ignore  # noqa: PLC0415
    except ImportError:
        # Minimal valid PDF 1.4 with one empty page
        return (
            b"%PDF-1.4\n"
            b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
            b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n"
            b"xref\n0 4\n"
            b"0000000000 65535 f \n"
            b"0000000009 00000 n \n"
            b"0000000053 00000 n \n"
            b"0000000098 00000 n \n"
            b"trailer<</Size 4/Root 1 0 R>>\n"
            b"startxref\n152\n%%EOF\n"
        )
    doc = fitz.open()
    doc.new_page()
    data = doc.tobytes()
    doc.close()
    return data


def make_image_media(media_id: int, embedding_dim: int = 512) -> dict:
    """Build a test image media dict with a solid-color PNG and fake embedding."""
    import hashlib  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    color = ((media_id * 37) % 256, (media_id * 53) % 256, (media_id * 71) % 256)
    img = make_png_bytes(color=color)
    rng = np.random.RandomState(media_id)
    return {
        "id": media_id,
        "type": "image",
        "embedder": "siglip",
        "width": 16,
        "height": 16,
        "file_size": len(img),
        "md5": hashlib.md5(img).hexdigest(),
        "embedding": rng.randn(embedding_dim).astype("float32"),
        "media_bytes": img,
        "filename": f"image_{media_id}.png",
        "category": "test-image",
        "origin": {"importer": "test", "params": {}},
        "origin_name": f"image_{media_id}.png",
    }


def make_text_media(media_id: int, embedding_dim: int = 512) -> dict:
    """Build a test text media dict with a short string and fake embedding."""
    import hashlib  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    content = f"test text document number {media_id}"
    rng = np.random.RandomState(media_id + 1000)
    return {
        "id": media_id,
        "type": "text",
        "embedder": "e5",
        "word_count": len(content.split()),
        "character_count": len(content),
        "file_size": len(content.encode()),
        "md5": hashlib.md5(content.encode()).hexdigest(),
        "embedding": rng.randn(embedding_dim).astype("float32"),
        "media_string": content,
        "filename": f"text_{media_id}.txt",
        "category": "test-text",
        "origin": {"importer": "test", "params": {}},
        "origin_name": f"text_{media_id}.txt",
    }


def make_video_media(media_id: int, embedding_dim: int = 512) -> dict:
    """Build a test video media dict with a minimal MP4 header and fake embedding."""
    import hashlib  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    mp4 = make_minimal_mp4_bytes()
    rng = np.random.RandomState(media_id + 2000)
    return {
        "id": media_id,
        "type": "video",
        "embedder": "xclip",
        "duration": 1.0,
        "file_size": len(mp4),
        "md5": hashlib.md5(mp4).hexdigest(),
        "embedding": rng.randn(embedding_dim).astype("float32"),
        "media_bytes": mp4,
        "filename": f"video_{media_id}.mp4",
        "category": "test-video",
        "origin": {"importer": "test", "params": {}},
        "origin_name": f"video_{media_id}.mp4",
    }


def make_document_media(media_id: int, embedding_dim: int = 512) -> dict:
    """Build a test document media dict with a minimal PDF and fake embedding."""
    import hashlib  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    pdf = make_minimal_pdf_bytes()
    rng = np.random.RandomState(media_id + 3000)
    return {
        "id": media_id,
        "type": "document",
        "embedder": "siglip",  # via document→image converter
        "file_size": len(pdf),
        "md5": hashlib.md5(pdf).hexdigest(),
        "embedding": rng.randn(embedding_dim).astype("float32"),
        "media_bytes": pdf,
        "filename": f"doc_{media_id}.pdf",
        "category": "test-doc",
        "origin": {"importer": "test", "params": {}},
        "origin_name": f"doc_{media_id}.pdf",
    }


# ---------------------------------------------------------------------------
# Trainable model / results helpers
# ---------------------------------------------------------------------------


def build_results_dict(hits, model_name, media_type="unknown"):
    """Build a single-model results dict (same shape as exporter input)."""
    return {
        "media_type": media_type,
        "detectors_run": 1,
        "results": {
            model_name: {
                "detector_name": model_name,
                "threshold": 0.5,
                "total_hits": len(hits),
                "hits": hits,
            }
        },
    }


def make_trainable_model_file(tmp_path, name, good_ids, bad_ids, snap, media_type="audio"):
    """Write a detector JSON file populated from *snap* and votes.

    Returns the path to the written JSON file.
    """
    from vtsearch.datasets.labelset import LabelSet  # noqa: PLC0415

    good_votes_dict = {k: None for k in good_ids}
    bad_votes_dict = {k: None for k in bad_ids}
    labelset = LabelSet.from_clips_and_votes(snap, good_votes_dict, bad_votes_dict, expand_dupes=False)

    data = {
        "name": name,
        "text_query": "",
        "media_example": "",
        "media_type": media_type,
        "examples": [],
        "labelset": labelset.to_dict(),
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(data))
    return path


def setup_trainable_model_in_registry(name, good_ids, bad_ids, snap, media_type="audio"):
    """Build a detector on disk + register it.  Returns its registry id.

    Writes ``<get_detectors_dir()>/<name>.json`` with a labelset built
    from *good_ids* and *bad_ids* using *snap* as the medias source, and
    registers the model so ``/api/detectors/registry`` lists it.
    """
    from vtsearch.datasets.labelset import LabelSet  # noqa: PLC0415
    from vtsearch.models.detector_registry import register_detector  # noqa: PLC0415
    from vtsearch.models.detector_store import _detector_path, _write_detector  # noqa: PLC0415

    good_votes_dict = {k: None for k in good_ids}
    bad_votes_dict = {k: None for k in bad_ids}
    labelset = LabelSet.from_clips_and_votes(snap, good_votes_dict, bad_votes_dict, expand_dupes=False)

    data = {
        "name": name,
        "text_query": "",
        "media_example": "",
        "media_type": media_type,
        "examples": [],
        "labelset": labelset.to_dict(),
    }
    _write_detector(_detector_path(name), data)

    entry = register_detector(name=name, media_type=media_type, num_training=len(labelset))
    return entry["id"]
