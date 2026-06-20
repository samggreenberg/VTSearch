"""Loader multi-embed (v3 Phase 2b.3).

A dataset may bind up to one text-capable and one patch-capable embedder.
At ingest the embed stage runs each in turn so ``media["embeddings"]``
carries one vector per bound embedder, and the per-embedder vectors survive
a pickle save/reload round-trip.  Single-embedder datasets are unchanged.
"""

from __future__ import annotations

import unittest.mock as mock

import numpy as np

from vtscore.datasets.loader import export_dataset_to_file, load_dataset_from_pickle
from vtscore.datasets.stages.embedding import _ordered_load_embedders, embed_missing
from vtscore.embedding.media_vectors import media_embedder_names, media_embedding


def _fake_embedder(name: str, dim: int = 3):
    """A MagicMock embedder with a native bulk path and no patch support."""
    emb = mock.MagicMock()
    emb.name = name
    emb.media_type_id = "audio"
    emb._model = True
    emb._on_progress = lambda *a, **kw: None
    emb.supports_patch_regions = False
    emb.supports_geometric_verification = False

    def _bulk(medias):
        return [np.full(dim, float(i + 1), dtype=np.float32) for i, _ in enumerate(medias)]

    emb.embed_media_bulk.side_effect = _bulk
    return emb


# ---------------------------------------------------------------------------
# _ordered_load_embedders: which embedders the stage runs, in what order
# ---------------------------------------------------------------------------


class TestOrderedLoadEmbedders:
    def test_fresh_create_uses_requested_only(self):
        medias = {1: {"media_type": "audio", "embedding": None}}
        assert _ordered_load_embedders(medias, "siglip") == ["siglip"]

    def test_fresh_create_no_pick_falls_back_to_blank(self):
        # No embedder anywhere → [""], which lets embed_missing resolve the
        # media-type default (single-embedder create path, unchanged).
        medias = {1: {"media_type": "audio", "embedding": None}}
        assert _ordered_load_embedders(medias, "") == [""]

    def test_reload_runs_present_embedders(self):
        # A reloaded two-embedder pickle: no requested pick, run both present.
        medias = {
            1: {
                "embedder": "siglip",
                "embeddings": {"siglip": np.ones(3), "dinov3_patch": np.ones(3)},
            }
        }
        assert _ordered_load_embedders(medias, "") == ["siglip", "dinov3_patch"]

    def test_requested_leads_then_present(self):
        medias = {1: {"embeddings": {"dinov3_patch": np.ones(3)}}}
        assert _ordered_load_embedders(medias, "siglip") == ["siglip", "dinov3_patch"]


# ---------------------------------------------------------------------------
# embed_missing: a second bound embedder embeds items the first already covered
# ---------------------------------------------------------------------------


class TestSecondEmbedderPass:
    def test_second_embedder_writes_own_key_without_disturbing_first(self):
        text_emb = _fake_embedder("emb_text")
        patch_emb = _fake_embedder("emb_patch")
        by_name = {"emb_text": text_emb, "emb_patch": patch_emb}

        medias = {i: {"media_type": "audio", "embedding": None, "media_path": f"/tmp/{i}.wav"} for i in range(1, 4)}

        with (
            mock.patch("vtscore.media.get_embedder", side_effect=lambda n: by_name[n]),
            mock.patch("vtscore.media.embedders_for_type", return_value=[text_emb]),
        ):
            embed_missing(medias, "emb_text")
            # Singular mirror + first key now belong to emb_text.
            for m in medias.values():
                assert m["embedder"] == "emb_text"
                assert "emb_text" in m["embeddings"]
            first_vecs = {i: m["embeddings"]["emb_text"] for i, m in medias.items()}

            embed_missing(medias, "emb_patch")

        # The second embedder ran over every item even though the singular
        # mirror was already set (its missing-detection is per-embedder).
        assert patch_emb.embed_media_bulk.call_count == 1
        assert len(patch_emb.embed_media_bulk.call_args.args[0]) == 3

        for i, m in medias.items():
            assert set(m["embeddings"]) == {"emb_text", "emb_patch"}
            # First embedder's vectors untouched; mirror still on the primary.
            np.testing.assert_array_equal(m["embeddings"]["emb_text"], first_vecs[i])
            assert m["embedder"] == "emb_text"
            np.testing.assert_array_equal(media_embedding(m, "emb_text"), m["embeddings"]["emb_text"])
            np.testing.assert_array_equal(media_embedding(m, "emb_patch"), m["embeddings"]["emb_patch"])

    def test_single_embedder_path_unchanged(self):
        emb = _fake_embedder("solo")
        medias = {i: {"media_type": "audio", "embedding": None, "media_path": f"/tmp/{i}.wav"} for i in range(1, 4)}

        with mock.patch("vtscore.media.embedders_for_type", return_value=[emb]):
            embed_missing(medias)

        assert emb.embed_media_bulk.call_count == 1
        for m in medias.values():
            assert m["embedding"] is not None
            assert m["embeddings"] == {"solo": m["embedding"]}


# ---------------------------------------------------------------------------
# Persistence: per-embedder vectors round-trip through save/reload
# ---------------------------------------------------------------------------


def _two_embedder_text_media(mid: int) -> dict:
    rng = np.random.default_rng(mid)
    a = rng.standard_normal(4).astype(np.float32)
    b = rng.standard_normal(4).astype(np.float32)
    return {
        "id": mid,
        "media_type": "text",
        "duration": 0,
        "file_size": 10,
        "md5": f"md5{mid}",
        "embedder": "emb_text",
        "embedding": a,
        "embeddings": {"emb_text": a, "emb_patch": b},
        "media_string": f"document {mid}",
        "filename": f"doc{mid}.txt",
        "category": "unknown",
    }


class TestPersistenceRoundTrip:
    def test_embeddings_dict_survives_save_reload(self, tmp_path):
        medias = {i: _two_embedder_text_media(i) for i in range(1, 4)}
        data = export_dataset_to_file(medias, embedder="emb_text", media_type="text", name="rt")
        pkl = tmp_path / "rt.pkl"
        pkl.write_bytes(data)

        loaded: dict[int, dict] = {}
        load_dataset_from_pickle(pkl, loaded)

        assert len(loaded) == 3
        for m in loaded.values():
            assert set(m["embeddings"]) == {"emb_text", "emb_patch"}
            assert set(media_embedder_names(m)) == {"emb_text", "emb_patch"}
            # Stored normalised on load; vectors differ between the two slots.
            ta = m["embeddings"]["emb_text"]
            pa = m["embeddings"]["emb_patch"]
            assert not np.allclose(ta, pa)
            np.testing.assert_allclose(np.linalg.norm(ta), 1.0, atol=1e-5)

    def test_legacy_single_vector_pickle_has_no_dict(self, tmp_path):
        # A media with only a singular vector writes no "embeddings" key; the
        # embed stage's ensure_embeddings_dict materialises it later.
        rng = np.random.default_rng(7)
        v = rng.standard_normal(4).astype(np.float32)
        media = {
            "id": 1,
            "media_type": "text",
            "duration": 0,
            "file_size": 10,
            "md5": "m",
            "embedder": "solo",
            "embedding": v,
            "media_string": "doc",
            "filename": "doc.txt",
            "category": "unknown",
        }
        data = export_dataset_to_file({1: media}, embedder="solo", media_type="text", name="legacy")
        pkl = tmp_path / "legacy.pkl"
        pkl.write_bytes(data)

        loaded: dict[int, dict] = {}
        load_dataset_from_pickle(pkl, loaded)
        assert loaded[1].get("embeddings") is None
        assert media_embedder_names(loaded[1]) == ["solo"]


def test_meta_records_binding_slots(tmp_path):
    """export writes the role-typed slots to meta (informational)."""
    from vtscore.datasets.container import read_meta

    medias = {i: _two_embedder_text_media(i) for i in range(1, 3)}
    data = export_dataset_to_file(medias, embedder="emb_text", media_type="text", name="rt")
    pkl = tmp_path / "rt.pkl"
    pkl.write_bytes(data)

    meta = read_meta(pkl)
    # emb_text / emb_patch aren't registered, so capability lookup yields no
    # role match; the slots are present but None.  The keys themselves must
    # exist so external readers can rely on the schema.
    assert "text_embedder" in meta
    assert "patch_embedder" in meta
