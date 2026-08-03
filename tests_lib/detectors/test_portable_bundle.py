"""Library-tier tests for the portable detector bundle builder.

Covers the pure builder in :mod:`vtscore.detectors.portable_bundle`: ONNX graph
correctness (numerically matches the trained torch model), manifest/README
contents, and - critically - that the bundle carries *only* the classifier and
never leaks embeddings or raw media (the scoring-only contract that makes this
the sanctioned exception to the no-persisted-vectors rule).
"""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest
import torch
from vtscore.detectors import portable_bundle as pb
from vtscore.detectors.training import serialize_weights
from vtscore.training.mlp import LINEAR_HEAD, build_model

# The exporter has to handle both heads: the linear (logistic) head production
# trains (#2790) and the MLP the eval harness still builds.  Parameterising on
# ``hidden`` runs every graph test over both shapes.
HEADS = pytest.mark.parametrize("hidden", [LINEAR_HEAD, 8], ids=["linear", "mlp"])


def _trained_weights(input_dim: int = 512, hidden: int = LINEAR_HEAD) -> dict:
    """A small, deterministic detector head serialized to the export format.

    Defaults to the production linear head so the head-agnostic manifest/bundle
    tests below package what the app actually exports.
    """
    gen = torch.Generator().manual_seed(0)
    model = build_model(input_dim, hidden_dim=hidden, dropout=0.5, generator=gen)
    model.eval()
    return serialize_weights(model)


def _sample_manifest(weights: dict) -> dict:
    return pb.build_manifest(
        detector_name="Cats",
        media_type="image",
        embedder="siglip",
        embedder_display_name="SigLIP (general images)",
        embedder_model_id="google/siglip-base-patch16-224",
        embedder_type="semantic",
        embedding_dim=pb.embedding_dim_from_weights(weights),
        threshold=0.6123456,
        good_count=7,
        bad_count=4,
        exported_by="vtsearch test",
        exported_at="2026-06-30T00:00:00Z",
    )


class TestOnnxGraph:
    @HEADS
    def test_embedding_dim_from_weights(self, hidden):
        assert pb.embedding_dim_from_weights(_trained_weights(input_dim=768, hidden=hidden)) == 768

    def test_split_rejects_non_two_layer(self):
        # ``_split_linear_weights`` backs the MLP branch only; the linear head
        # takes the 1-layer path in ``mlp_weights_to_onnx`` and never reaches it.
        with pytest.raises(ValueError, match="2-layer MLP"):
            pb._split_linear_weights({"0.weight": [[1.0]], "0.bias": [0.0]})

    @HEADS
    def test_onnx_is_valid(self, hidden):
        import onnx  # noqa: PLC0415

        model = onnx.load_from_string(pb.mlp_weights_to_onnx(_trained_weights(hidden=hidden)))
        onnx.checker.check_model(model)
        assert {i.name for i in model.graph.input} == {pb.ONNX_INPUT_NAME}
        assert {o.name for o in model.graph.output} == {pb.ONNX_OUTPUT_NAME}
        # The MLP's hidden activation is a Relu; the linear head has none.
        assert ("Relu" in {node.op_type for node in model.graph.node}) is (hidden != LINEAR_HEAD)

    @HEADS
    def test_onnx_scores_match_torch(self, hidden):
        """The emitted graph must reproduce sigmoid(model(x)) of the trained head."""
        ort = pytest.importorskip("onnxruntime")

        gen = torch.Generator().manual_seed(1)
        model = build_model(512, hidden_dim=hidden, dropout=0.5, generator=gen)
        model.eval()
        weights = serialize_weights(model)

        rng = np.random.default_rng(2)
        x = rng.standard_normal((6, 512)).astype(np.float32)
        with torch.no_grad():
            expected = torch.sigmoid(model(torch.from_numpy(x))).numpy().ravel()

        session = ort.InferenceSession(pb.mlp_weights_to_onnx(weights))
        got = session.run([pb.ONNX_OUTPUT_NAME], {pb.ONNX_INPUT_NAME: x})[0].ravel()
        np.testing.assert_allclose(got, expected, atol=1e-5)


class TestManifest:
    def test_manifest_shape(self):
        weights = _trained_weights(input_dim=768)
        manifest = _sample_manifest(weights)
        assert manifest["format"] == pb.BUNDLE_FORMAT
        assert manifest["format_version"] == pb.BUNDLE_FORMAT_VERSION
        assert manifest["embedder"]["name"] == "siglip"
        assert manifest["embedder"]["model_id"] == "google/siglip-base-patch16-224"
        assert manifest["embedder"]["embedding_dim"] == 768
        assert manifest["scoring"]["activation"] == "sigmoid"
        # Threshold is rounded but preserved.
        assert manifest["scoring"]["threshold"] == pytest.approx(0.612346, abs=1e-6)
        assert manifest["training_labels"] == {"good": 7, "bad": 4}
        assert manifest["contains_media_data"] is False

    def test_readme_mentions_key_facts(self):
        weights = _trained_weights(input_dim=768)
        readme = pb.render_readme(_sample_manifest(weights))
        assert "SigLIP (general images)" in readme
        assert "siglip" in readme
        assert "google/siglip-base-patch16-224" in readme  # exact HF model id
        assert "768" in readme  # embedding dim
        assert "0.612346" in readme  # threshold
        assert "onnxruntime" in readme  # inference snippet

    def test_readme_omits_model_id_when_absent(self):
        """Embedders with no model id (e.g. SIFT/VLAD) still render cleanly."""
        weights = _trained_weights(input_dim=768)
        manifest = pb.build_manifest(
            detector_name="Cats",
            media_type="image",
            embedder="sift_vlad",
            embedder_display_name="SIFT/VLAD (instances)",
            embedder_model_id=None,
            embedder_type="structural",
            embedding_dim=pb.embedding_dim_from_weights(weights),
            threshold=0.5,
            good_count=3,
            bad_count=2,
            exported_by="vtsearch test",
            exported_at="2026-06-30T00:00:00Z",
        )
        assert manifest["embedder"]["model_id"] is None
        readme = pb.render_readme(manifest)
        assert "SIFT/VLAD (instances)" in readme
        assert "exact pretrained model" not in readme

    def test_manifest_caveats_default_empty(self):
        weights = _trained_weights(input_dim=768)
        assert _sample_manifest(weights)["caveats"] == []

    def test_manifest_carries_given_caveats(self):
        weights = _trained_weights(input_dim=768)
        manifest = pb.build_manifest(
            detector_name="Cats",
            media_type="image",
            embedder="dinov2",
            embedder_display_name="DINOv2 (patch)",
            embedder_model_id="facebook/dinov2-base",
            embedder_type="patch_semantic",
            embedding_dim=pb.embedding_dim_from_weights(weights),
            threshold=0.5,
            good_count=3,
            bad_count=2,
            exported_by="vtsearch test",
            exported_at="2026-06-30T00:00:00Z",
            caveats=pb.caveats_for_embedder_type("patch_semantic"),
        )
        assert manifest["caveats"] == pb.caveats_for_embedder_type("patch_semantic")
        readme = pb.render_readme(manifest)
        assert "WHOLE item" in readme


class TestExportability:
    def test_structural_is_blocked(self):
        with pytest.raises(ValueError, match="structural"):
            pb.check_exportable("structural")

    def test_semantic_and_patch_and_legacy_are_exportable(self):
        for embedder_type in ("semantic", "patch_semantic", ""):
            pb.check_exportable(embedder_type)  # must not raise

    def test_patch_semantic_has_one_caveat(self):
        caveats = pb.caveats_for_embedder_type("patch_semantic")
        assert len(caveats) == 1
        assert "WHOLE item" in caveats[0]

    def test_semantic_and_legacy_have_no_caveats(self):
        assert pb.caveats_for_embedder_type("semantic") == []
        assert pb.caveats_for_embedder_type("") == []


class TestBundle:
    def test_bundle_members(self):
        weights = _trained_weights()
        bundle = pb.build_bundle(weights=weights, manifest=_sample_manifest(weights))
        with zipfile.ZipFile(io.BytesIO(bundle)) as zf:
            assert sorted(zf.namelist()) == ["README.md", "detector.onnx", "manifest.json"]

    def test_bundle_is_byte_stable(self):
        """Same inputs produce identical bytes (fixed zip timestamps)."""
        weights = _trained_weights()
        manifest = _sample_manifest(weights)
        a = pb.build_bundle(weights=weights, manifest=manifest)
        b = pb.build_bundle(weights=weights, manifest=manifest)
        assert a == b

    def test_bundle_carries_no_media_vectors(self):
        """Scoring-only contract: nothing in the bundle leaks embeddings/media.

        The only place floats legitimately appear is inside the ONNX weight
        tensors (``detector.onnx``).  The manifest and README must not embed any
        media-derived vector data - only counts, names, and the threshold.
        """
        weights = _trained_weights()
        bundle = pb.build_bundle(weights=weights, manifest=_sample_manifest(weights))
        with zipfile.ZipFile(io.BytesIO(bundle)) as zf:
            manifest = json.loads(zf.read("manifest.json"))
            readme = zf.read("README.md").decode()

        # No embeddings/labels payloads smuggled into the manifest.
        forbidden = {"embeddings", "vectors", "labels", "labelset", "elements", "media"}
        assert forbidden.isdisjoint(manifest.keys())
        # Nothing under "embedder" or "training_labels" is a raw vector either.
        assert isinstance(manifest["training_labels"]["good"], int)
        assert all(not isinstance(v, list) for v in manifest["embedder"].values())
        # The README documents new-media scoring, not stored vectors.
        assert "no raw media" in readme.lower()
