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
    from vtsearch.audio import generate_wav  # noqa: PLC0415

    return generate_wav(frequency, duration)


def make_raw_wav_bytes() -> bytes:
    """Create a minimal valid WAV file (100 zero-samples) in memory.

    Lighter weight than :func:`make_wav_bytes` — does not depend on the
    ``vtsearch.audio`` module.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        samples = struct.pack("<" + "h" * 100, *([0] * 100))
        wf.writeframes(samples)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Detector / results helpers
# ---------------------------------------------------------------------------


def build_results_dict(hits, detector_path, media_type="unknown"):
    """Build a single-detector results dict (same shape as exporter input)."""
    detector_data = json.loads(Path(detector_path).read_text())
    detector_name = detector_data.get("name", Path(detector_path).stem)
    threshold = detector_data.get("threshold", 0.5)
    return {
        "media_type": media_type,
        "detectors_run": 1,
        "results": {
            detector_name: {
                "detector_name": detector_name,
                "threshold": threshold,
                "total_hits": len(hits),
                "hits": hits,
            }
        },
    }


def make_detector_file(tmp_path, good_ids, bad_ids, name="detector.json"):
    """Train a detector from given vote IDs and write its JSON to a file.

    Returns ``(detector_path, detector_dict)``.
    """
    import app as app_module  # noqa: PLC0415

    app_module.good_votes.update({k: None for k in good_ids})
    app_module.bad_votes.update({k: None for k in bad_ids})
    detector = train_detector_from_votes()
    app_module.good_votes.clear()
    app_module.bad_votes.clear()

    detector_path = tmp_path / name
    detector_path.write_text(json.dumps(detector))
    return detector_path, detector


def train_detector_from_votes():
    """Train a detector from current good/bad votes and return the payload.

    Replacement for the removed ``POST /api/detector/export`` endpoint.
    Returns a dict with ``weights``, ``threshold``, ``good_origins``,
    ``bad_origins``, ``inclusion``, and ``media_type``.
    """
    from vtsearch.models import collect_media_origins
    from vtsearch.models.detector_training import serialize_weights, train_and_threshold
    from vtsearch.utils import bad_votes, get_inclusion, good_votes, snapshot_medias

    if not good_votes or not bad_votes:
        raise ValueError("Need at least one good and one bad vote")

    snap = snapshot_medias()

    good_origins = collect_media_origins(good_votes, snap)
    bad_origins = collect_media_origins(bad_votes, snap)

    X_list, y_list = [], []
    for cid in good_votes:
        if cid in snap and "embedding" in snap[cid]:
            X_list.append(snap[cid]["embedding"])
            y_list.append(1.0)
    for cid in bad_votes:
        if cid in snap and "embedding" in snap[cid]:
            X_list.append(snap[cid]["embedding"])
            y_list.append(0.0)

    model, threshold = train_and_threshold(X_list, y_list, snap)
    weights = serialize_weights(model)

    media_type = "audio"
    if snap:
        media_type = next(iter(snap.values())).get("type", "audio")

    return {
        "weights": weights,
        "threshold": threshold,
        "good_origins": good_origins,
        "bad_origins": bad_origins,
        "inclusion": get_inclusion(),
        "media_type": media_type,
    }
