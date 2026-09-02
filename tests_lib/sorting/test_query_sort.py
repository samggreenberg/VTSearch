"""Library-tier tests for the external-query sorts (issue #3419).

These four entry points used to be private helpers inside
``vtsearch/routes/sorting.py``, reachable only through a Flask test client
(and, for three of them, only via a cross-blueprint import of another route
module's underscore names).  They now live in
:mod:`vtscore.training.query_sort`, so this file exercises them with nothing
but an active dataset context — no app, no request, no client.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest

from vtscore.state.core import get_active_context
from vtscore.training.query_sort import (
    apply_crop_or_keep,
    cosine_sort_active,
    embed_external_labels,
    parse_label_file,
)


def _fill_active_medias(dim: int = 8, n: int = 6) -> np.ndarray:
    """Populate the active context with medias whose vectors fan out from a target.

    Media 1 is the target; each later media is rotated further away, so the
    expected cosine ranking is exactly the id order.  Returns the query vector.
    """
    medias = get_active_context().medias
    medias.clear()
    base = np.zeros(dim, dtype=np.float32)
    base[0] = 1.0
    off = np.zeros(dim, dtype=np.float32)
    off[1] = 1.0
    for i in range(n):
        vec = base + (i * 0.4) * off
        medias[i + 1] = {
            "id": i + 1,
            "embedding": (vec / np.linalg.norm(vec)).astype(np.float32),
            "media_type": "image",
        }
    return base


class TestParseLabelFile:
    """The label-file reader raises ValueError; the route owns the HTTP mapping."""

    def test_returns_the_labels_list(self):
        payload = {"labels": [{"path": "a.png", "label": "good"}]}
        assert parse_label_file(io.BytesIO(json.dumps(payload).encode())) == payload["labels"]

    def test_non_json_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid label file format"):
            parse_label_file(io.BytesIO(b"not json at all"))

    def test_non_object_document_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid label file format"):
            parse_label_file(io.BytesIO(b'["a", "b"]'))

    def test_empty_labels_raises_value_error(self):
        with pytest.raises(ValueError, match="No labels found in file"):
            parse_label_file(io.BytesIO(b'{"labels": []}'))


class TestEmbedExternalLabels:
    """Malformed / unreachable entries are skipped and counted, never raised on."""

    class _Emb:
        def embed_media(self, media):
            return np.ones(4, dtype=np.float32)

    def test_skips_malformed_missing_and_unembeddable_entries(self, tmp_path):
        real = tmp_path / "real.png"
        real.write_bytes(b"\x89PNG\r\n\x1a\n")

        labels = [
            {"path": str(real), "label": "good"},
            {"path": str(real), "label": "bad"},
            {"path": str(real), "label": "maybe"},  # not good/bad
            {"label": "good"},  # no path at all
            {"path": str(tmp_path / "missing.png"), "label": "bad"},
        ]
        X, y, loaded, skipped = embed_external_labels(labels, self._Emb())

        assert loaded == 2
        assert skipped == 3
        assert len(X) == len(y) == 2
        assert y == [1.0, 0.0]

    def test_none_embedding_counts_as_skipped(self, tmp_path):
        real = tmp_path / "real.png"
        real.write_bytes(b"\x89PNG\r\n\x1a\n")

        class _NullEmb:
            def embed_media(self, media):
                return None

        X, y, loaded, skipped = embed_external_labels(
            [{"path": str(real), "label": "good"}], _NullEmb()
        )
        assert (X, y, loaded, skipped) == ([], [], 0, 1)


class TestCosineSortActive:
    """The whole-dataset cosine sort runs against the active context alone."""

    def test_ranks_every_media_by_similarity_to_the_query(self):
        query = _fill_active_medias()
        results, threshold = cosine_sort_active(query)

        assert [r["id"] for r in results] == [1, 2, 3, 4, 5, 6]
        sims = [r["similarity"] for r in results]
        assert sims == sorted(sims, reverse=True)
        assert threshold == round(threshold, 4)

    def test_returns_a_row_per_loaded_media(self):
        query = _fill_active_medias(n=4)
        results, _ = cosine_sort_active(query)
        assert len(results) == 4


class TestApplyCropOrKeep:
    """A falsy crop spec is a no-op that leaves the file byte-for-byte intact."""

    @pytest.mark.parametrize("crop", [None, {}])
    def test_no_crop_params_keeps_the_file_untouched(self, tmp_path, crop):
        path = tmp_path / "example.bin"
        path.write_bytes(b"original bytes")

        assert apply_crop_or_keep(path, crop) is path
        assert path.read_bytes() == b"original bytes"
