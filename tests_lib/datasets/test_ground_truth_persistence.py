"""Library-tier tests: eval ground truth survives the pickle round-trip (#4117).

Visual Genome's demo source stamps every clip with a multi-label
``categories`` list and the ``regions`` boxes the eval harness drags.  The demo
cache pickle *stored* both, but ``load_dataset_from_pickle`` rebuilds each media
from a fixed field list, so a warm cache came back with both gone on all 4,193
``visual_genome_m`` medias -- with the right count, vectors and ids, and no
error.  ``vtscore.eval.labels.media_is_positive`` then reads the missing
``categories`` as "single-label" and scores against the primary ``category``,
and a region arm finds no box to drag.  A cold build and a warm one were two
different datasets under one name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from vtscore.datasets import loader_demo as _loader_demo
from vtscore.datasets.loader import export_dataset_to_file
from vtscore.datasets.loader_common import GROUND_TRUTH_FIELDS
from vtscore.datasets.loader_pickle import load_dataset_from_pickle
from vtscore.eval.labels import media_is_positive, region_box_for_category
from vtscore.media.image.media_type import ImageMediaType

DEMO_ID = "visual_genome_m"

REGIONS = [
    {"box": [0.1, 0.2, 0.5, 0.6], "label": "dog"},
    {"box": [0.6, 0.1, 0.9, 0.4], "label": "frisbee"},
]


def _media(cid: int = 1, **extra: Any) -> dict[str, Any]:
    rng = np.random.default_rng(cid)
    return {
        "id": cid,
        "media_type": "image",
        "duration": 0,
        "file_size": 64,
        "md5": f"md5-{cid}",
        "embedder": "siglip",
        "embeddings": {"siglip": rng.standard_normal(16).astype(np.float32)},
        "filename": f"{cid}.jpg",
        "category": "dog",
        "media_bytes": b"\x00" * 64,
        "media_string": None,
        "media_path": None,
        "width": 500,
        "height": 375,
        **extra,
    }


def _vg_media(cid: int = 1) -> dict[str, Any]:
    """The shape ``_emit_image_clip`` gives a multi-label source's clip."""
    return _media(cid, categories=["dog", "frisbee"], regions=list(REGIONS))


def _round_trip(media: dict[str, Any], tmp_path: Path, thin: bool = False) -> dict[str, Any]:
    pkl = tmp_path / "ds.pkl"
    pkl.write_bytes(export_dataset_to_file({1: media}, embedder="siglip", media_type="image"))
    loaded: dict[int, dict[str, Any]] = {}
    load_dataset_from_pickle(pkl, loaded, thin=thin)
    return loaded[1]


class TestGroundTruthSurvivesPickleRoundTrip:
    def test_categories_and_regions_persist(self, tmp_path: Path):
        out = _round_trip(_vg_media(), tmp_path)

        assert out["categories"] == ["dog", "frisbee"]
        assert out["regions"] == REGIONS

    def test_they_persist_through_a_thin_load(self, tmp_path: Path):
        out = _round_trip(_vg_media(), tmp_path, thin=True)

        assert out["categories"] == ["dog", "frisbee"]
        assert out["regions"] == REGIONS

    def test_an_empty_categories_list_is_kept(self, tmp_path: Path):
        """``[]`` is a multi-label negative; dropping it makes the media single-label.

        A single-label media is positive for its ``category``, so the negative
        would turn positive for "dog" on reload.
        """
        out = _round_trip(_media(categories=[], regions=[]), tmp_path)

        assert out["categories"] == []
        assert out["regions"] == []
        assert not media_is_positive(out, "dog")

    def test_evaluable_categories_persist(self, tmp_path: Path):
        out = _round_trip(_media(categories=[], regions=[], evaluable_categories=["dog@small"]), tmp_path)

        assert out["evaluable_categories"] == ["dog@small"]

    def test_single_label_media_gains_no_ground_truth_fields(self, tmp_path: Path):
        out = _round_trip(_media(), tmp_path)

        assert not [field for field in GROUND_TRUTH_FIELDS if field in out]


class TestWarmDemoCacheMatchesColdBuild:
    """The #4117 path itself: ``_write_demo_cache`` then a cache hit."""

    def _plant_cache(self, tmp_path: Path, monkeypatch, medias: dict[int, dict]) -> Path:
        monkeypatch.setattr(_loader_demo, "EMBEDDINGS_DIR", tmp_path)
        pkl = tmp_path / f"{DEMO_ID}.pkl"
        _loader_demo._write_demo_cache(
            pkl_file=pkl,
            dataset_name=DEMO_ID,
            media_type_id="image",
            medias=medias,
            mt=ImageMediaType(),
            embedder=None,
            external_dir=None,
            converter_name="",
            clipper_name="",
            clipper_params=None,
            clipper_applied=False,
        )
        return pkl

    def test_a_cache_hit_keeps_every_label(self, tmp_path: Path, monkeypatch):
        cold = {1: _vg_media(1), 2: _media(2, category="frisbee", categories=[], regions=[])}
        self._plant_cache(tmp_path, monkeypatch, cold)

        warm: dict[int, dict] = {}
        _loader_demo.load_demo_dataset(DEMO_ID, warm, on_progress=lambda *a, **k: None)

        assert {cid: m["categories"] for cid, m in warm.items()} == {1: ["dog", "frisbee"], 2: []}
        assert warm[1]["regions"] == REGIONS
        # What the harness reads off them: the box a region arm drags, and the
        # multi-label membership rule rather than the single-label fallback.
        assert region_box_for_category(warm[1], "frisbee") == (0.6, 0.1, 0.9, 0.4)
        assert media_is_positive(warm[1], "frisbee")
        assert not media_is_positive(warm[2], "frisbee")

    def test_use_cache_false_neither_reads_nor_writes_the_cache(self, tmp_path: Path, monkeypatch):
        """The pile's path: build from the source even when a cache is sitting there.

        A hit hands back vectors whichever earlier build embedded, under a
        provenance sidecar that records this one; and rewriting the cache would
        change what every other reader of that directory gets.
        """
        pkl = self._plant_cache(tmp_path, monkeypatch, {1: _vg_media(1)})
        planted = pkl.read_bytes()

        fresh = _vg_media(7)
        sources: list[str] = []

        class _FakeImageType:
            dir_key = "image_dir"

            def load_demo_source(self, *, source, clips, **_kw):
                sources.append(source)
                clips[7] = dict(fresh)
                return None

        monkeypatch.setattr("vtscore.media.get", lambda _type_id: _FakeImageType())
        monkeypatch.setattr(_loader_demo, "_resolve_demo_embedder", lambda *_a: None)

        out: dict[int, dict] = {}
        _loader_demo.load_demo_dataset(DEMO_ID, out, on_progress=lambda *a, **k: None, use_cache=False)

        assert len(sources) == 1, "use_cache=False must build from the demo source"
        assert list(out) == [7]
        assert out[7]["categories"] == ["dog", "frisbee"]
        assert pkl.read_bytes() == planted, "use_cache=False must not rewrite the cache"
