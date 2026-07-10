"""Library-tier tests: the coverage atlas survives a dataset pickle round-trip.

The hierarchical k-means atlas is expensive to rebuild, so a dataset pickle
caches its structure under the ``"coverage_atlas"`` key.  ``export_dataset_to_file``
writes it via ``extra_pickle_keys`` and ``load_dataset_from_pickle`` hands it
back so the reload path can skip the rebuild (see
``vtscore.state.coverage.restore_coverage_atlas_from_cache``).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

from vtscore.datasets.loader import export_dataset_to_file
from vtscore.datasets.loader_pickle import load_dataset_from_pickle
from vtscore.state.core import DatasetContext
from vtscore.embedding.media_vectors import media_embedding
from vtscore.state.coverage import restore_coverage_atlas_from_cache
from vtscore.state.coverage_atlas import CoverageAtlas


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _image_dataset(n: int) -> dict[int, dict]:
    """Build *n* image medias with deterministic, distinct embeddings."""
    rng = np.random.default_rng(11)
    medias: dict[int, dict] = {}
    for i in range(n):
        src = _png_bytes((i % 255, (2 * i) % 255, (3 * i) % 255))
        medias[i] = {
            "id": i,
            "media_type": "image",
            "duration": 0,
            "file_size": len(src),
            "md5": f"md5-{i}",
            "embedder": "siglip",
            "embeddings": {"siglip": rng.standard_normal(64).astype(np.float32)},
            "filename": f"{i}.png",
            "category": "test",
            "media_bytes": src,
            "width": 16,
            "height": 16,
        }
    return medias


def test_atlas_round_trips_through_pickle(tmp_path: Path):
    medias = _image_dataset(120)
    vectors = {cid: media_embedding(m) for cid, m in medias.items()}
    atlas = CoverageAtlas(vectors, k=3, min_node_size=10)

    container = export_dataset_to_file(
        medias,
        embedder="siglip",
        media_type="image",
        extra_pickle_keys={"coverage_atlas": atlas.to_serializable()},
    )
    pkl = tmp_path / "ds.pkl"
    pkl.write_bytes(container)

    loaded: dict = {}
    cached = load_dataset_from_pickle(pkl, loaded)

    assert cached is not None, "load_dataset_from_pickle should surface the cached atlas"
    assert cached["format"] == "coverage-atlas/1"
    assert loaded.keys() == medias.keys()

    # The returned snapshot restores onto a context whose medias match.
    ctx = DatasetContext("roundtrip")
    ctx.medias = loaded
    assert restore_coverage_atlas_from_cache(ctx, cached) is True
    assert ctx.coverage_atlas.vector_to_leaf.keys() == loaded.keys()
    assert ctx.coverage_atlas.next_sample() == atlas.next_sample()


def test_pickle_without_cached_atlas_returns_none(tmp_path: Path):
    """Older pickles (no cache key) load fine and report no cached atlas."""
    medias = _image_dataset(10)
    container = export_dataset_to_file(medias, embedder="siglip", media_type="image")
    pkl = tmp_path / "ds.pkl"
    pkl.write_bytes(container)

    loaded: dict = {}
    assert load_dataset_from_pickle(pkl, loaded) is None
