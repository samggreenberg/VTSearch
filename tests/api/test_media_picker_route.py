"""``GET /api/medias/picker``: what New Detector's *Pick from Dataset* grid lists (#5465).

A Face dataset's ids come in similarity order (the 1-D order fit at build, or
here on first ask); any other dataset does once its Browse layout brought it
one (#5464), and lists in import order until then.  ``q`` keeps
the items whose name holds every word of it, so a photo's name finds the faces
cropped from it.  The fit is faked: what is under test is the route's use of
the order, not the arrangement.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np

from vtscore.projection import similarity_order as so
from vtscore.projection.similarity_order import SimilarityOrder
from vtscore.state.core import DatasetContext, register_context


def _media(mid: int, media_type: str, name: str) -> dict[str, Any]:
    return {
        "id": mid,
        "media_type": media_type,
        "md5": f"md5-{mid}",
        "filename": name,
        "origin_name": name,
        "embedder": "face",
        "embeddings": {"face": np.random.default_rng(mid).standard_normal(8).astype(np.float32)},
    }


def _load(dataset_id: str, media_type: str, names: list[str]) -> DatasetContext:
    ctx = DatasetContext(dataset_id)
    for i, name in enumerate(names, start=1):
        ctx.medias[i] = _media(i, media_type, name)
    register_context(ctx)
    return ctx


FACES = [
    "beach.jpg→beach_face_1.jpg",
    "party.jpg→party_face_1.jpg",
    "beach.jpg→beach_face_2.jpg",
    "Beach Party.jpg→Beach Party_face_1.jpg",
]


def _reversed(matrix, ids, **kwargs):
    return SimilarityOrder(list(reversed(ids)), "umap")


def _get(client, dataset_id: str, q: str | None = None):
    query = {"q": q} if q is not None else {}
    return client.get("/api/medias/picker", query_string=query, headers={"X-Dataset-Id": dataset_id})


class TestPickerListing:
    def test_a_face_dataset_lists_in_similarity_order(self, client):
        _load("faces", "face", FACES)
        with (
            patch("vtscore.projection.store.pkl_path_for", return_value=None),
            patch.object(so, "fit_similarity_order", side_effect=_reversed),
        ):
            res = _get(client, "faces")
        assert res.status_code == 200
        assert res.get_json() == {"ids": [4, 3, 2, 1], "total": 4, "order": "similarity"}

    def test_other_types_list_in_import_order(self, client):
        _load("photos", "image", ["c.jpg", "a.jpg", "b.jpg"])
        with patch.object(so, "fit_similarity_order") as fit:
            res = _get(client, "photos")
        fit.assert_not_called()
        assert res.get_json() == {"ids": [1, 2, 3], "total": 3, "order": "dataset"}

    def test_other_types_list_in_an_order_their_layout_brought(self, client):
        """Any type's order comes with its Browse layout (#5464); none is fit here."""
        ctx = _load("photos", "image", ["c.jpg", "a.jpg", "b.jpg"])
        ctx._similarity_order = SimilarityOrder([2, 3, 1], "umap")
        with patch.object(so, "fit_similarity_order") as fit:
            res = _get(client, "photos")
        fit.assert_not_called()
        assert res.get_json() == {"ids": [2, 3, 1], "total": 3, "order": "similarity"}

    def test_search_keeps_items_naming_every_word(self, client):
        _load("faces", "face", FACES)
        with (
            patch("vtscore.projection.store.pkl_path_for", return_value=None),
            patch.object(so, "fit_similarity_order", side_effect=_reversed),
        ):
            beach = _get(client, "faces", "BEACH").get_json()
            both = _get(client, "faces", "  beach   party ").get_json()
            none = _get(client, "faces", "nobody").get_json()
        # Case-insensitive, in the similarity order, with the full count beside.
        assert beach == {"ids": [4, 3, 1], "total": 4, "order": "similarity"}
        assert both["ids"] == [4]
        assert none == {"ids": [], "total": 4, "order": "similarity"}

    def test_a_failed_fit_falls_back_to_import_order(self, client):
        _load("faces", "face", FACES)
        with (
            patch("vtscore.projection.store.pkl_path_for", return_value=None),
            patch.object(so, "fit_similarity_order", side_effect=RuntimeError("umap exploded")),
        ):
            res = _get(client, "faces")
        assert res.get_json() == {"ids": [1, 2, 3, 4], "total": 4, "order": "dataset"}

    def test_requires_a_dataset(self, client):
        # The test client sends the active dataset by default; blank it.
        assert client.get("/api/medias/picker", headers={"X-Dataset-Id": ""}).status_code == 400
