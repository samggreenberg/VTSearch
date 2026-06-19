"""Library-tier tests: the diversity tree survives a dataset pickle round-trip.

The hierarchical k-means tree is expensive to rebuild, so a dataset pickle
caches its structure under the ``"diversity_tree"`` key.  ``export_dataset_to_file``
writes it via ``extra_pickle_keys`` and ``load_dataset_from_pickle`` hands it
back so the reload path can skip the rebuild (see
``vtscore.state.diversity.restore_diversity_tree_from_cache``).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

from vtscore.datasets.loader import export_dataset_to_file
from vtscore.datasets.loader_pickle import load_dataset_from_pickle
from vtscore.state.core import DatasetContext
from vtscore.state.diversity import restore_diversity_tree_from_cache
from vtscore.state.diversity_tree import DiversityTree


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
            "embedding": rng.standard_normal(64).astype(np.float32),
            "filename": f"{i}.png",
            "category": "test",
            "media_bytes": src,
            "width": 16,
            "height": 16,
        }
    return medias


def test_tree_round_trips_through_pickle(tmp_path: Path):
    medias = _image_dataset(120)
    vectors = {cid: m["embedding"] for cid, m in medias.items()}
    tree = DiversityTree(vectors, k=3, min_node_size=10)

    container = export_dataset_to_file(
        medias,
        embedder="siglip",
        media_type="image",
        extra_pickle_keys={"diversity_tree": tree.to_serializable()},
    )
    pkl = tmp_path / "ds.pkl"
    pkl.write_bytes(container)

    loaded: dict = {}
    cached = load_dataset_from_pickle(pkl, loaded)

    assert cached is not None, "load_dataset_from_pickle should surface the cached tree"
    assert cached["format"] == 1
    assert loaded.keys() == medias.keys()

    # The returned snapshot restores onto a context whose medias match.
    ctx = DatasetContext("roundtrip")
    ctx.medias = loaded
    assert restore_diversity_tree_from_cache(ctx, cached) is True
    assert ctx.diversity_tree.vector_to_leaf.keys() == loaded.keys()
    assert ctx.diversity_tree.next_sample() == tree.next_sample()


def test_pickle_without_cached_tree_returns_none(tmp_path: Path):
    """Older pickles (no cache key) load fine and report no cached tree."""
    medias = _image_dataset(10)
    container = export_dataset_to_file(medias, embedder="siglip", media_type="image")
    pkl = tmp_path / "ds.pkl"
    pkl.write_bytes(container)

    loaded: dict = {}
    assert load_dataset_from_pickle(pkl, loaded) is None
