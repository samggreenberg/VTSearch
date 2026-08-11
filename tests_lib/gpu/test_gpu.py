"""GPU-specific tests for VTSearch.

These tests exercise the same code paths as the CPU test suite but on CUDA
devices.  They are guarded by the ``gpu`` pytest marker **and** a runtime
``torch.cuda.is_available()`` check, so they are never collected during
regular CI runs (which use ``-m "not gpu"`` or simply omit the marker).

Run them on a machine with a CUDA GPU::

    python -m pytest tests/test_gpu.py -v

Coverage areas
--------------
1. MLP training (train_model) on GPU
2. Cross-calibration threshold computation on GPU
3. Full train_and_score pipeline on GPU
4. Detector export → reconstruct → score on GPU
5. Embedding models (CLAP, CLIP, X-CLIP, E5) on GPU
6. CPU ↔ GPU numerical equivalence for training
7. GPU memory cleanup after inference
"""

import gc

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Marker applied to every test in this module
# ---------------------------------------------------------------------------
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]


# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def device():
    """Return the first available CUDA device."""
    return torch.device("cuda", 0)


def _make_embeddings(n: int, dim: int = 512, seed: int = 42) -> np.ndarray:
    """Create deterministic random embeddings for *n* medias."""
    rng = np.random.RandomState(seed)
    return rng.randn(n, dim).astype(np.float32)


def _make_clips_dict(n: int = 20, dim: int = 512, seed: int = 42) -> dict:
    """Build a minimal medias dict similar to the real application."""
    embs = _make_embeddings(n, dim, seed)
    return {i + 1: {"id": i + 1, "embedding": embs[i], "media_type": "audio"} for i in range(n)}


def _make_votes(good_ids: list[int], bad_ids: list[int]):
    good = {k: None for k in good_ids}
    bad = {k: None for k in bad_ids}
    return good, bad


# ---------------------------------------------------------------------------
# 0. Device-aware embedding plumbing
# ---------------------------------------------------------------------------


class TestDeviceAwareEmbedding:
    """The device-resolution machinery every embedder now loads through.

    These exercise the mechanism the device-aware embedder refactor hinges on
    (``resolve_device`` -> ``to_compute_device``) without touching the heavy
    HF model downloads or the embedder registry (which the library-tier test
    suite stubs out), so they stay fast and deterministic on any CUDA box.
    """

    def test_resolve_device_picks_cuda_when_usable(self):
        from vtscore.config import resolve_device

        # On a host where the marker's torch.cuda.is_available() guard let this
        # test run, the smoke-test in resolve_device() should also pass.
        assert resolve_device().startswith("cuda")

    def test_to_compute_device_moves_model_to_gpu(self):
        import torch.nn as nn

        from vtscore.media.embedder import to_compute_device

        model = to_compute_device(nn.Linear(8, 4))
        assert next(model.parameters()).device.type == "cuda"

    def test_concurrent_embeddings_default_is_one_on_gpu(self, monkeypatch):
        # With a usable GPU resolved, the embed default collapses to 1 to avoid
        # stacking model copies on a single shared device (env override aside).
        from vtscore.embedding import loader

        monkeypatch.delenv("VTSEARCH_MAX_CONCURRENT_EMBEDDINGS", raising=False)
        assert loader.default_concurrent_embeddings() == 1


# ---------------------------------------------------------------------------
# 0b. cuML-accelerated UMAP + k-means backends
# ---------------------------------------------------------------------------


def _cuml_installed() -> bool:
    """True when the optional cuML/RAPIDS stack is importable on this host."""
    try:
        import cuml  # noqa: F401  # pyright: ignore[reportMissingImports]
    except ImportError:
        return False
    return True


class TestCuMLBackends:
    """The cuML backend selectors that accelerate UMAP + the coverage atlas.

    cuML is an *optional* GPU dependency, so these tests exercise the factory
    contract on any CUDA host: the estimators must work (and return numpy)
    whether they resolve to cuML or fall back to the CPU libraries.  The few
    cuML-specific assertions are guarded by ``_cuml_installed()`` so a GPU box
    without RAPIDS still passes.
    """

    def test_cuml_enabled_matches_install(self):
        # On a CUDA host (guarded by the module marker) the device resolves to
        # cuda, so cuml_enabled() tracks purely whether cuML is importable.
        from vtscore.gpu_backends import cuml_enabled

        assert cuml_enabled() == _cuml_installed()

    def test_umap_fit_transform_produces_2d_numpy_layout(self):
        from vtscore.gpu_backends import umap_fit_transform

        rng = np.random.default_rng(0)
        mat = rng.standard_normal((60, 16)).astype(np.float32)
        coords = umap_fit_transform(
            mat, n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42
        )
        assert isinstance(coords, np.ndarray)
        assert coords.shape == (60, 2)
        assert np.isfinite(coords).all()

    def test_kmeans_fit_predict_clusters_and_reports_inertia(self):
        from vtscore.gpu_backends import kmeans_fit_predict

        rng = np.random.default_rng(1)
        vecs = np.vstack([rng.standard_normal((30, 8)) + 5.0, rng.standard_normal((30, 8)) - 5.0]).astype(np.float32)
        labels, inertia = kmeans_fit_predict(vecs, n_clusters=2, random_state=42, n_init=1)
        assert isinstance(labels, np.ndarray)
        assert labels.shape == (60,)
        assert set(np.unique(labels).tolist()) <= {0, 1}
        assert inertia is not None

    def test_fit_projection_umap_path_on_gpu(self):
        from vtscore.projection.umap_projection import fit_projection

        rng = np.random.default_rng(2)
        matrix = rng.standard_normal((60, 16)).astype(np.float32)
        ids = list(range(60))
        proj = fit_projection(matrix, ids, random_state=42)
        assert proj.method == "umap"
        assert proj.coords.shape == (60, 2)
        assert proj.coords.dtype == np.float32
        assert np.isfinite(proj.coords).all()

    def test_coverage_atlas_builds_on_gpu(self):
        from vtscore.state.coverage_atlas import CoverageAtlas

        rng = np.random.default_rng(3)
        vectors = {i: rng.standard_normal(8).astype(np.float32) for i in range(60)}
        atlas = CoverageAtlas(vectors, k=2, max_depth=4, min_node_size=20)
        # Every vector lands in a leaf and the atlas actually split the root.
        assert len(atlas.vector_to_leaf) == 60
        assert atlas.total_nodes > 1
        for vid in vectors:
            assert atlas.lookup(vid) in atlas.nodes


# ---------------------------------------------------------------------------
# 1. train_model on GPU
# ---------------------------------------------------------------------------


class TestTrainModelGPU:
    """Verify ``train_model`` works when tensors and model live on CUDA."""

    def test_model_trains_on_gpu(self, device):
        from vtscore.training.mlp import train_model

        dim = 64
        X = torch.randn(10, dim, device=device)
        y = torch.cat([torch.ones(5, 1), torch.zeros(5, 1)]).to(device)

        model = train_model(X, y, dim)
        # train_model creates its own model on CPU; verify it can evaluate GPU data
        # after moving the model to GPU
        model = model.to(device)
        with torch.no_grad():
            scores = torch.sigmoid(model(X))
        assert scores.shape == (10, 1)
        assert scores.device.type == "cuda"

    def test_gpu_trained_scores_between_zero_and_one(self, device):
        from vtscore.training.mlp import train_model

        dim = 64
        X = torch.randn(10, dim, device=device)
        y = torch.cat([torch.ones(5, 1), torch.zeros(5, 1)]).to(device)

        model = train_model(X, y, dim).to(device)
        with torch.no_grad():
            scores = torch.sigmoid(model(X)).squeeze(1).cpu().numpy()
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_gpu_model_separates_classes(self, device):
        """Good examples should score higher than bad examples on average."""
        import vtscore.config as config

        saved = config.TRAIN_EPOCHS
        config.TRAIN_EPOCHS = 100
        try:
            from vtscore.training.mlp import train_model

            rng = np.random.RandomState(0)
            dim = 32
            # Linearly separable data
            good_embs = rng.randn(20, dim).astype(np.float32) + 2.0
            bad_embs = rng.randn(20, dim).astype(np.float32) - 2.0
            X = torch.tensor(np.vstack([good_embs, bad_embs]), device=device)
            y = torch.cat([torch.ones(20, 1), torch.zeros(20, 1)]).to(device)

            model = train_model(X, y, dim).to(device)
            with torch.no_grad():
                scores = torch.sigmoid(model(X)).squeeze(1).cpu().numpy()
            avg_good = scores[:20].mean()
            avg_bad = scores[20:].mean()
            assert avg_good > avg_bad
        finally:
            config.TRAIN_EPOCHS = saved

    def test_inclusion_lowers_threshold_not_model(self, device):
        """Inclusion is a pure threshold knob: it leaves the trained model
        (and its scores) untouched, but a higher inclusion yields a lower
        (more inclusive) decision threshold via ``conformal_threshold``."""
        import vtscore.config as config

        saved = config.TRAIN_EPOCHS
        config.TRAIN_EPOCHS = 100
        try:
            from vtscore.training.mlp import train_model
            from vtscore.training.thresholds import conformal_threshold

            dim = 32
            rng = np.random.RandomState(1)
            good_embs = rng.randn(10, dim).astype(np.float32) + 1.0
            bad_embs = rng.randn(10, dim).astype(np.float32) - 1.0
            X = torch.tensor(np.vstack([good_embs, bad_embs]), device=device)
            y = torch.cat([torch.ones(10, 1), torch.zeros(10, 1)]).to(device)

            # The model is inclusion-independent and deterministic (seed=42),
            # so two trainings yield identical scores.
            model_a = train_model(X, y, dim).to(device)
            model_b = train_model(X, y, dim).to(device)
            with torch.no_grad():
                scores_a = torch.sigmoid(model_a(X)).squeeze(1).cpu().numpy()
                scores_b = torch.sigmoid(model_b(X)).squeeze(1).cpu().numpy()
            np.testing.assert_allclose(scores_a, scores_b, rtol=1e-5, atol=1e-5)

            labels = y.squeeze(1).cpu().tolist()
            scores = scores_a.tolist()
            t_neutral = conformal_threshold(scores, labels, 0)
            t_inclusive = conformal_threshold(scores, labels, 5)
            # Higher inclusion => prefer recall => lower (more inclusive) threshold.
            assert t_inclusive <= t_neutral
        finally:
            config.TRAIN_EPOCHS = saved


# ---------------------------------------------------------------------------
# 2. Cross-calibration threshold on GPU
# ---------------------------------------------------------------------------


class TestCrossCalibrationGPU:
    """Verify ``calculate_cross_calibration_threshold`` produces a valid
    threshold when the underlying training happens on a GPU-capable system."""

    def test_threshold_is_valid_float(self):
        import vtscore.config as config

        saved = config.TRAIN_EPOCHS
        config.TRAIN_EPOCHS = 30
        try:
            from vtscore.training.thresholds import calculate_cross_calibration_threshold

            dim = 64
            rng = np.random.RandomState(7)
            X_list = list(rng.randn(20, dim).astype(np.float32))
            y_list = [1.0] * 10 + [0.0] * 10
            threshold = calculate_cross_calibration_threshold(X_list, y_list, dim)
            assert isinstance(threshold, float)
            assert 0.0 <= threshold <= 1.0
        finally:
            config.TRAIN_EPOCHS = saved

    def test_threshold_with_inclusion(self):
        import vtscore.config as config

        saved = config.TRAIN_EPOCHS
        config.TRAIN_EPOCHS = 30
        try:
            from vtscore.training.thresholds import calculate_cross_calibration_threshold

            dim = 64
            rng = np.random.RandomState(8)
            X_list = list(rng.randn(20, dim).astype(np.float32))
            y_list = [1.0] * 10 + [0.0] * 10
            t_neg = calculate_cross_calibration_threshold(X_list, y_list, dim, inclusion_value=-5)
            t_pos = calculate_cross_calibration_threshold(X_list, y_list, dim, inclusion_value=5)
            # Both should be valid
            assert isinstance(t_neg, float)
            assert isinstance(t_pos, float)
        finally:
            config.TRAIN_EPOCHS = saved


# ---------------------------------------------------------------------------
# 3. Full train_and_score pipeline on GPU
# ---------------------------------------------------------------------------


class TestTrainAndScoreGPU:
    """Verify that the full ``train_and_score`` pipeline (used by the
    learned-sort endpoint) works on a system with a GPU."""

    def test_returns_all_clips_scored(self):
        import vtscore.config as config

        saved = config.TRAIN_EPOCHS
        config.TRAIN_EPOCHS = 30
        try:
            from vtscore.detectors.training import train_and_score

            clips_dict = _make_clips_dict(20, dim=64)
            good, bad = _make_votes([1, 2, 3], [18, 19, 20])
            results, threshold, _model = train_and_score(clips_dict, good, bad)

            assert len(results) == 20
            assert isinstance(threshold, float)
            for entry in results:
                assert "id" in entry
                assert "score" in entry
        finally:
            config.TRAIN_EPOCHS = saved

    def test_scores_between_zero_and_one(self):
        import vtscore.config as config

        saved = config.TRAIN_EPOCHS
        config.TRAIN_EPOCHS = 30
        try:
            from vtscore.detectors.training import train_and_score

            clips_dict = _make_clips_dict(20, dim=64)
            good, bad = _make_votes([1, 2, 3], [18, 19, 20])
            results, _, _m = train_and_score(clips_dict, good, bad)
            for entry in results:
                assert 0.0 <= entry["score"] <= 1.0
        finally:
            config.TRAIN_EPOCHS = saved

    def test_results_sorted_descending(self):
        import vtscore.config as config

        saved = config.TRAIN_EPOCHS
        config.TRAIN_EPOCHS = 30
        try:
            from vtscore.detectors.training import train_and_score

            clips_dict = _make_clips_dict(20, dim=64)
            good, bad = _make_votes([1, 2], [3, 4])
            results, _, _m = train_and_score(clips_dict, good, bad)
            scores = [e["score"] for e in results]
            assert scores == sorted(scores, reverse=True)
        finally:
            config.TRAIN_EPOCHS = saved

    def test_good_clips_scored_higher_than_bad(self):
        import vtscore.config as config

        saved = config.TRAIN_EPOCHS
        config.TRAIN_EPOCHS = 50
        try:
            from vtscore.detectors.training import train_and_score

            # Use separable embeddings so the model can learn
            dim = 64
            rng = np.random.RandomState(99)
            clips_dict = {}
            for i in range(1, 21):
                emb = rng.randn(dim).astype(np.float32) + (2.0 if i <= 5 else -2.0 if i > 15 else 0.0)
                clips_dict[i] = {"id": i, "embedding": emb, "media_type": "audio"}

            good, bad = _make_votes([1, 2, 3, 4, 5], [16, 17, 18, 19, 20])
            results, _, _m = train_and_score(clips_dict, good, bad)
            score_map = {e["id"]: e["score"] for e in results}
            avg_good = np.mean([score_map[i] for i in good])
            avg_bad = np.mean([score_map[i] for i in bad])
            assert avg_good > avg_bad
        finally:
            config.TRAIN_EPOCHS = saved

    def test_with_inclusion_value(self):
        import vtscore.config as config

        saved = config.TRAIN_EPOCHS
        config.TRAIN_EPOCHS = 30
        try:
            from vtscore.detectors.training import train_and_score

            clips_dict = _make_clips_dict(20, dim=64)
            good, bad = _make_votes([1, 2, 3], [18, 19, 20])
            results, threshold, _model = train_and_score(clips_dict, good, bad, inclusion_value=5)
            assert len(results) == 20
            assert isinstance(threshold, float)
        finally:
            config.TRAIN_EPOCHS = saved


# ---------------------------------------------------------------------------
# 4. Detector export → reconstruct → score on GPU
# ---------------------------------------------------------------------------


class TestDetectorGPU:
    """Verify detector model reconstruction and scoring on GPU."""

    def test_reconstruct_and_score_on_gpu(self, device):
        import vtscore.config as config

        saved = config.TRAIN_EPOCHS
        config.TRAIN_EPOCHS = 30
        try:
            from vtscore.training.mlp import train_model

            dim = 64
            clips_dict = _make_clips_dict(20, dim)
            good, bad = _make_votes([1, 2, 3], [18, 19, 20])

            # Build training data
            X_list, y_list = [], []
            for cid in good:
                X_list.append(clips_dict[cid]["embedding"])
                y_list.append(1.0)
            for cid in bad:
                X_list.append(clips_dict[cid]["embedding"])
                y_list.append(0.0)

            X = torch.tensor(np.array(X_list), dtype=torch.float32)
            y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)

            # Train on CPU (as the app does)
            model = train_model(X, y, dim)

            # Export weights (serialise model state_dict to JSON-safe lists)
            state_dict = model.state_dict()
            weights = {k: v.tolist() for k, v in state_dict.items()}

            # Reconstruct on GPU (as a GPU-aware detector-sort would)
            from vtscore.training.mlp import build_model_from_weights

            gpu_model = build_model_from_weights(weights).to(device)

            # Score all medias on GPU
            all_embs = np.array([clips_dict[cid]["embedding"] for cid in sorted(clips_dict.keys())])
            X_all = torch.tensor(all_embs, dtype=torch.float32, device=device)
            with torch.no_grad():
                scores = torch.sigmoid(gpu_model(X_all)).squeeze(1).cpu().numpy()

            assert len(scores) == 20
            assert np.all(scores >= 0.0)
            assert np.all(scores <= 1.0)
        finally:
            config.TRAIN_EPOCHS = saved

    def test_gpu_cpu_scores_match(self, device):
        """Scores from GPU and CPU model should be numerically close."""
        import vtscore.config as config

        saved = config.TRAIN_EPOCHS
        config.TRAIN_EPOCHS = 30
        try:
            from vtscore.training.mlp import train_model

            dim = 64
            clips_dict = _make_clips_dict(20, dim, seed=123)
            good, bad = _make_votes([1, 2, 3], [18, 19, 20])

            X_list, y_list = [], []
            for cid in good:
                X_list.append(clips_dict[cid]["embedding"])
                y_list.append(1.0)
            for cid in bad:
                X_list.append(clips_dict[cid]["embedding"])
                y_list.append(0.0)

            X = torch.tensor(np.array(X_list), dtype=torch.float32)
            y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)

            model = train_model(X, y, dim)

            all_embs = np.array([clips_dict[cid]["embedding"] for cid in sorted(clips_dict.keys())])
            X_all = torch.tensor(all_embs, dtype=torch.float32)

            # CPU scores
            model.eval()
            with torch.no_grad():
                cpu_scores = torch.sigmoid(model(X_all)).squeeze(1).numpy()

            # GPU scores
            gpu_model = model.to(device)
            X_all_gpu = X_all.to(device)
            with torch.no_grad():
                gpu_scores = torch.sigmoid(gpu_model(X_all_gpu)).squeeze(1).cpu().numpy()

            np.testing.assert_allclose(cpu_scores, gpu_scores, atol=1e-5)
        finally:
            config.TRAIN_EPOCHS = saved

    def test_multiple_detectors_on_gpu(self, device):
        """Simulate auto-detect: run multiple detectors on GPU sequentially."""
        import vtscore.config as config

        saved = config.TRAIN_EPOCHS
        config.TRAIN_EPOCHS = 30
        try:
            from vtscore.training.mlp import train_model

            dim = 64
            clips_dict = _make_clips_dict(20, dim)
            all_embs = np.array([clips_dict[cid]["embedding"] for cid in sorted(clips_dict.keys())])
            X_all = torch.tensor(all_embs, dtype=torch.float32, device=device)

            detector_results = {}
            for det_idx, (good_ids, bad_ids) in enumerate([([1, 2, 3], [18, 19, 20]), ([5, 6, 7], [14, 15, 16])]):
                good, bad = _make_votes(good_ids, bad_ids)
                X_list, y_list = [], []
                for cid in good:
                    X_list.append(clips_dict[cid]["embedding"])
                    y_list.append(1.0)
                for cid in bad:
                    X_list.append(clips_dict[cid]["embedding"])
                    y_list.append(0.0)

                X = torch.tensor(np.array(X_list), dtype=torch.float32)
                y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
                model = train_model(X, y, dim).to(device)

                with torch.no_grad():
                    scores = torch.sigmoid(model(X_all)).squeeze(1).cpu().numpy()

                detector_results[f"det-{det_idx}"] = scores

            assert len(detector_results) == 2
            for name, scores in detector_results.items():
                assert len(scores) == 20
                assert np.all(scores >= 0.0)
                assert np.all(scores <= 1.0)
        finally:
            config.TRAIN_EPOCHS = saved


# ---------------------------------------------------------------------------
# 5. Embedding models on GPU
# ---------------------------------------------------------------------------


class TestCLAPEmbeddingGPU:
    """Exercise the CLAP audio embedder's own wrapper (load + forward) on GPU.

    These route the *real* VTSearch pre/post-processing - librosa decode,
    deterministic 10 s truncation, the ClapProcessor call, the audio / text
    projection heads, and the base-class L2-normalisation - through CUDA,
    rather than re-implementing the forward pass against raw ``transformers``
    as this file used to.  A *fresh* :class:`AudioClapEmbedder` instance is
    used so the session-wide embedder stub (which patches the registered
    singletons) does not intercept the call.  Downloads the CLAP model on
    first run and may be slow.
    """

    def test_clap_model_loads_on_gpu(self, device):
        from vtscore.media.audio.embedder_clap import AudioClapEmbedder

        emb = AudioClapEmbedder()
        emb.load_models()
        # to_compute_device() inside the wrapper moves the model to CUDA.
        assert next(emb._model.parameters()).device.type == "cuda"

    def test_clap_text_embedding_on_gpu(self, device):
        from vtscore.media.audio.embedder_clap import AudioClapEmbedder

        emb = AudioClapEmbedder()
        vec = emb.embed_text("a dog barking")
        assert vec is not None
        assert vec.shape == (512,)
        assert np.isfinite(vec).all()
        # The wrapper L2-normalises every text vector.
        np.testing.assert_allclose(np.linalg.norm(vec), 1.0, atol=1e-4)

    def test_clap_audio_embedding_on_gpu(self, device, tmp_path):
        import soundfile as sf

        from vtscore.config import CLAP_SAMPLE_RATE
        from vtscore.media.audio.embedder_clap import AudioClapEmbedder
        from vtscore.media.embedder import media_from_path

        # Generate a short sine wave
        duration = 1.0
        t = np.linspace(0, duration, int(CLAP_SAMPLE_RATE * duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        wav_path = tmp_path / "test.wav"
        sf.write(str(wav_path), audio, CLAP_SAMPLE_RATE)

        emb = AudioClapEmbedder()
        vec = emb.embed_media(media_from_path(wav_path))
        assert vec is not None
        assert vec.shape == (512,)
        assert np.isfinite(vec).all()
        np.testing.assert_allclose(np.linalg.norm(vec), 1.0, atol=1e-4)


class TestXCLIPEmbeddingGPU:
    """Exercise the X-CLIP video embedder's own wrapper on GPU.

    Routes the real wrapper (processor call, ``get_video_features`` /
    ``get_text_features``, ``extract_tensor``, L2-normalisation) through CUDA
    via a fresh :class:`VideoXClipEmbedder`, rather than raw ``transformers``.
    """

    def test_xclip_model_loads_on_gpu(self, device):
        from vtscore.media.video.embedder_xclip import VideoXClipEmbedder

        emb = VideoXClipEmbedder()
        emb.load_models()
        assert next(emb._model.parameters()).device.type == "cuda"

    def test_xclip_text_embedding_on_gpu(self, device):
        from vtscore.media.video.embedder_xclip import VideoXClipEmbedder

        emb = VideoXClipEmbedder()
        vec = emb.embed_text("a person walking")
        assert vec is not None
        assert vec.ndim == 1
        assert np.isfinite(vec).all()

    def test_xclip_video_embedding_on_gpu(self, device, monkeypatch):
        from PIL import Image

        from vtscore.media.video import embedder_xclip

        emb = embedder_xclip.VideoXClipEmbedder()
        # Feed deterministic frames straight into the wrapper's forward path
        # so we exercise the model plumbing without needing a real video file.
        frames = [Image.new("RGB", (224, 224), color=(i * 30, 100, 200)) for i in range(8)]
        monkeypatch.setattr(embedder_xclip, "sample_video_frames", lambda media, n: frames)
        vec = emb.embed_media({"media_path": "/fake.mp4"})
        assert vec is not None
        assert vec.ndim == 1
        assert np.isfinite(vec).all()


class TestE5EmbeddingGPU:
    """Exercise the E5 text embedder's own wrapper on GPU.

    Routes the real wrapper (``query:`` / ``passage:`` prefixing, the
    sentence-transformers ``encode`` call, L2-normalisation) through CUDA via
    a fresh :class:`TextE5Embedder`, rather than calling ``SentenceTransformer``
    directly.
    """

    def test_e5_model_loads_on_gpu(self, device):
        from vtscore.media.text.embedder_e5 import TextE5Embedder

        emb = TextE5Embedder()
        vec = emb.embed_text("test sentence")
        assert vec is not None
        assert vec.ndim == 1
        assert np.isfinite(vec).all()

    def test_e5_passage_embedding_on_gpu(self, device):
        from vtscore.media.text.embedder_e5 import TextE5Embedder

        emb = TextE5Embedder()
        # embed_media applies the ``passage:`` prefix under the hood.
        vec = emb.embed_media({"media_string": "The quick brown fox jumps over the lazy dog."})
        assert vec is not None
        assert vec.ndim == 1
        assert np.isfinite(vec).all()

    def test_e5_query_passage_same_space(self, device):
        """Query and passage embeddings should have the same dimensionality."""
        from vtscore.media.text.embedder_e5 import TextE5Embedder

        emb = TextE5Embedder()
        q_vec = emb.embed_text("animals")
        p_vec = emb.embed_media({"media_string": "Dogs are loyal companions."})
        assert q_vec is not None and p_vec is not None
        assert q_vec.shape == p_vec.shape
        # Cosine similarity should be defined (both are unit vectors)
        sim = float(np.dot(q_vec, p_vec))
        assert -1.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# 6. CPU ↔ GPU numerical equivalence for embeddings
# ---------------------------------------------------------------------------


class TestEmbeddingEquivalence:
    """Verify that GPU embeddings match CPU embeddings within tolerance."""

    def test_e5_cpu_gpu_match(self, device):
        from sentence_transformers import SentenceTransformer

        from vtscore.config import E5_MODEL_ID, MODELS_CACHE_DIR

        text = "query: machine learning algorithms"

        cpu_model = SentenceTransformer(E5_MODEL_ID, cache_folder=str(MODELS_CACHE_DIR), device="cpu")
        cpu_vec = cpu_model.encode(text, normalize_embeddings=True)

        gpu_model = SentenceTransformer(E5_MODEL_ID, cache_folder=str(MODELS_CACHE_DIR), device=str(device))
        gpu_vec = gpu_model.encode(text, normalize_embeddings=True)

        np.testing.assert_allclose(cpu_vec, gpu_vec, atol=1e-4)


# ---------------------------------------------------------------------------
# 7. GPU memory cleanup
# ---------------------------------------------------------------------------


class TestGPUMemoryCleanup:
    """Verify that GPU memory is freed after model use."""

    def test_training_frees_gpu_memory(self, device):
        import vtscore.config as config

        saved = config.TRAIN_EPOCHS
        config.TRAIN_EPOCHS = 30
        try:
            from vtscore.training.mlp import train_model

            torch.cuda.reset_peak_memory_stats(device)
            initial_mem = torch.cuda.memory_allocated(device)

            dim = 128
            X = torch.randn(50, dim, device=device)
            y = torch.cat([torch.ones(25, 1), torch.zeros(25, 1)]).to(device)
            model = train_model(X, y, dim).to(device)

            with torch.no_grad():
                _ = model(X)

            # Clean up
            del model, X, y
            gc.collect()
            torch.cuda.empty_cache()

            final_mem = torch.cuda.memory_allocated(device)
            # Memory should return close to initial (within 1 MB tolerance)
            assert final_mem - initial_mem < 1_000_000
        finally:
            config.TRAIN_EPOCHS = saved

    def test_embedding_model_frees_gpu_memory(self, device):
        """Loading and unloading an embedding model should free GPU memory."""
        torch.cuda.reset_peak_memory_stats(device)
        initial_mem = torch.cuda.memory_allocated(device)

        from sentence_transformers import SentenceTransformer

        from vtscore.config import E5_MODEL_ID, MODELS_CACHE_DIR

        model = SentenceTransformer(E5_MODEL_ID, cache_folder=str(MODELS_CACHE_DIR), device=str(device))
        _ = model.encode("query: test", normalize_embeddings=True)

        # Should have allocated significant memory
        peak_mem = torch.cuda.max_memory_allocated(device)
        assert peak_mem > initial_mem

        # Clean up
        del model
        gc.collect()
        torch.cuda.empty_cache()

        final_mem = torch.cuda.memory_allocated(device)
        # Memory should return close to initial (within 5 MB tolerance for E5)
        assert final_mem - initial_mem < 5_000_000
