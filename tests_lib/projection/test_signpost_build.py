"""Library-tier tests for the Toponymy → RegionLabelSet extraction glue.

Covers ``vtscore.projection.signpost_build`` with the library fit stubbed at
the ``_fit_topic_layers`` seam (and the clusterable UMAP stubbed too), so no
test here imports toponymy or compiles numba: level mapping, medoid anchors,
noise/empty-name handling, size guards, and the text-encoder adapter.  The
real library is exercised by the ``slow``-marked smoke test.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.projection import signpost_build as sb
from vtscore.projection.labels import medoid
from vtscore.projection.umap_projection import Projection


def _proj(n: int, projection_id: str = "proj-1") -> Projection:
    rng = np.random.default_rng(42)
    coords = rng.standard_normal((n, 2)).astype(np.float32)
    return Projection(projection_id, list(range(1, n + 1)), coords, "pca")


class FakeEmbedder:
    name = "fake_clap"
    supports_text = True

    def embed_text(self, text: str):
        import hashlib

        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % 2**32
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(8).astype(np.float32)
        return vec / np.linalg.norm(vec)


@pytest.fixture
def stub_fit(monkeypatch):
    """Stub the clusterable UMAP + the toponymy fit; capture the fit inputs."""
    captured: dict = {}

    def fake_clusterable(matrix, on_progress=None):
        captured["clusterable_input"] = matrix
        return np.asarray(matrix[:, :2], dtype=np.float32)

    def fake_fit(texts, embedding_vectors, clusterable_vectors, text_encoder, **kwargs):
        captured["texts"] = texts
        captured["embedding_vectors"] = embedding_vectors
        captured["kwargs"] = kwargs
        return captured["layers"]

    monkeypatch.setattr(sb, "_clusterable_vectors", fake_clusterable)
    monkeypatch.setattr(sb, "_fit_topic_layers", fake_fit)
    return captured


def _build(n=60, layers=None, projection_id="proj-1", captured=None, **kwargs):
    proj = _proj(n, projection_id)
    matrix = np.tile(np.eye(4, dtype=np.float32), (n // 4 + 1, 2))[:n]
    texts = [f"text {i}" for i in range(n)]
    if captured is not None:
        captured["layers"] = layers or []
    return proj, sb.build_region_labels(
        proj,
        matrix,
        matrix,
        texts,
        FakeEmbedder(),
        object_description="things",
        corpus_description="a test corpus",
        **kwargs,
    )


class TestBuildRegionLabels:
    def test_flattens_layers_into_pinned_labels(self, stub_fit):
        n = 60
        fine = (["fine-a", "fine-b"], np.array([0, 1] * (n // 2)))
        coarse = (["coarse"], np.zeros(n, dtype=int))
        proj, label_set = _build(n=n, layers=[fine, coarse], captured=stub_fit)

        assert label_set.projection_id == proj.projection_id
        texts = [lab.text for lab in label_set.labels]
        assert texts == ["coarse", "fine-a", "fine-b"]  # coarse (level 0) first

        # Toponymy's layer 0 is the finest: it gets the deepest zoom band.
        by_text = {lab.text: lab for lab in label_set.labels}
        assert by_text["coarse"].level == 0.0
        assert by_text["fine-a"].level == pytest.approx(sb._LEVEL_STEP)
        assert all(lab.source == "keyphrase" for lab in label_set.labels)

    def test_anchor_is_layout_medoid_and_score_is_member_count(self, stub_fit):
        n = 60
        labels_arr = np.array([0] * 20 + [1] * 40)
        proj, label_set = _build(n=n, layers=[(["a", "b"], labels_arr)], captured=stub_fit)

        by_text = {lab.text: lab for lab in label_set.labels}
        expected = medoid(np.asarray(proj.coords)[:20])
        assert (by_text["a"].x, by_text["a"].y) == (pytest.approx(expected[0]), pytest.approx(expected[1]))
        assert by_text["a"].score == 20.0
        assert by_text["b"].score == 40.0

    def test_noise_and_empty_names_are_skipped(self, stub_fit):
        n = 60
        # Topic 0 has no members (all noise or topic 1); topic 2's name is blank.
        labels_arr = np.array([-1] * 30 + [1] * 20 + [2] * 10)
        proj, label_set = _build(n=n, layers=[(["orphan", "kept", "  "], labels_arr)], captured=stub_fit)
        assert [lab.text for lab in label_set.labels] == ["kept"]

    def test_too_small_corpus_returns_empty_without_fitting(self, monkeypatch):
        def boom(*args, **kwargs):
            raise AssertionError("fit must not run for tiny corpora")

        monkeypatch.setattr(sb, "_clusterable_vectors", boom)
        monkeypatch.setattr(sb, "_fit_topic_layers", boom)
        proj, label_set = _build(n=sb._MIN_POINTS - 1, layers=None, captured=None)
        assert label_set.projection_id == proj.projection_id
        assert label_set.labels == ()

    def test_misaligned_inputs_return_empty(self, stub_fit):
        proj = _proj(60)
        matrix = np.zeros((59, 4), dtype=np.float32)  # one row short
        label_set = sb.build_region_labels(
            proj,
            matrix,
            matrix,
            ["t"] * 60,
            FakeEmbedder(),
            object_description="things",
            corpus_description="c",
        )
        assert label_set.labels == ()

    def test_empty_layer_list_returns_empty(self, stub_fit):
        _, label_set = _build(n=60, layers=[], captured=stub_fit)
        assert label_set.labels == ()


class TestEmbedderTextEncoder:
    def test_encodes_into_media_space(self):
        encoder = sb.EmbedderTextEncoder(FakeEmbedder(), dim=8)
        out = encoder.encode(["dog", "rain"])
        assert out.shape == (2, 8)
        assert not np.allclose(out[0], out[1])

    def test_blank_or_failing_strings_become_zero_vectors(self):
        class Broken:
            def embed_text(self, text):
                raise RuntimeError("no model")

        encoder = sb.EmbedderTextEncoder(Broken(), dim=8)
        out = encoder.encode(["", "boom"])
        assert out.shape == (2, 8)
        assert np.allclose(out, 0.0)

    def test_empty_batch(self):
        encoder = sb.EmbedderTextEncoder(FakeEmbedder(), dim=8)
        assert encoder.encode([]).shape == (0, 8)


class TestScaling:
    def test_base_min_cluster_size_tracks_corpus_size(self):
        assert sb._base_min_cluster_size(500) == 10  # library default floor
        assert sb._base_min_cluster_size(3_000) == 10
        assert sb._base_min_cluster_size(21_000) == 70  # the study's 50–100 regime
        assert sb._base_min_cluster_size(50_000) == 167


class TestKeyphraseNamerParsing:
    """The no-LLM namer's prompt parsing (issue #2558).

    Exercised without importing toponymy — these are the module-level parse
    helpers the ``KeyphraseNamer`` delegates to. The disambiguation test
    reproduces the ``combined`` prompt layout the real library builds and the
    contract its ``default_extract_topic_names`` enforces: exactly one mapping
    entry per topic, keyed ``f"{i}. {name}"``.
    """

    def test_topic_name_uses_top_keyphrase(self):
        prompt = "Here is the group.\n - Keywords for this group include: dog barking, puppy, howl\n"
        assert sb._keyphrase_topic_name(prompt) == "dog barking"

    def test_topic_name_defaults_when_no_keywords(self):
        assert sb._keyphrase_topic_name("nothing useful here") == "unnamed"

    def test_disambiguation_maps_every_topic(self):
        # The header layout Toponymy's `combined` disambiguation prompt uses:
        # bare `"N. name":` lines at column 0, keyword lines indented below.
        prompt = (
            "You are an expert in images.\n\n"
            '"1. man boy lily shirt":\n'
            " - Keywords for this group include: man, boy, lily, shirt\n"
            '"2. man boy jacket lily":\n'
            " - Keywords for this group include: man, boy, jacket, lily\n\n"
            "The response should be formatted as JSON in the format\n"
            '    {"new_topic_name_mapping": {<1. OLD_NAME1>: <NEW_NAME1>, ... }, ...}\n'
        )
        result = sb._keyphrase_disambiguation(prompt)
        mapping = result["new_topic_name_mapping"]
        # One entry per topic (the length Toponymy requires), keyed so the
        # library's default_extract_topic_names maps each back to its topic.
        assert mapping == {
            "1. man boy lily shirt": "man boy lily shirt",
            "2. man boy jacket lily": "man boy jacket lily",
        }
        assert result["topic_specificities"] == [0.5, 0.5]

    def test_disambiguation_ignores_output_format_example(self):
        # The JSON output-format example (indented, `<1. …>` placeholders,
        # continuing past the colon) must not be miscounted as a topic header.
        prompt = (
            '"1. only real topic":\n'
            " - Keywords for this group include: alpha, beta\n\n"
            '    {"new_topic_name_mapping": {"1. OLD_NAME1": "NEW_NAME1"}, "topic_specificities": [0.5]}\n'
        )
        mapping = sb._keyphrase_disambiguation(prompt)["new_topic_name_mapping"]
        assert mapping == {"1. only real topic": "only real topic"}

    def test_disambiguation_round_trips_through_library_extractor(self):
        # Mirror toponymy.templates.default_extract_topic_names: given a mapping
        # of the right length, it recovers one name per topic in order without
        # raising. This is the exact contract issue #2558 was violating.
        old_names = ["man boy lily shirt", "man boy jacket lily"]
        prompt = (
            '"1. man boy lily shirt":\n'
            " - Keywords for this group include: man, boy\n"
            '"2. man boy jacket lily":\n'
            " - Keywords for this group include: man, jacket\n"
        )
        mapping = sb._keyphrase_disambiguation(prompt)["new_topic_name_mapping"]
        assert len(mapping) == len(old_names)
        recovered = []
        for i, old in enumerate(old_names, start=1):
            recovered.append(mapping.get(f"{i}. {old}", old))
        assert recovered == old_names


class TestAvailability:
    def test_availability_probe_does_not_import(self, monkeypatch):
        import importlib.util as util

        monkeypatch.setattr(util, "find_spec", lambda name: None)
        assert sb.signposting_available() is False

    def test_require_signposting_true_when_present(self, monkeypatch):
        monkeypatch.setattr(sb, "signposting_available", lambda: True)
        assert sb.require_signposting() is True

    def test_require_signposting_logs_loudly_when_missing(self, monkeypatch, caplog):
        import logging

        monkeypatch.setattr(sb, "signposting_available", lambda: False)
        # Reset the one-time latch so the error is emitted for this test.
        monkeypatch.setattr(sb, "_missing_toponymy_warned", False)
        with caplog.at_level(logging.ERROR, logger=sb.logger.name):
            assert sb.require_signposting() is False
        assert any("toponymy" in r.message.lower() for r in caplog.records)
        assert any("install.sh" in r.message for r in caplog.records)
        assert all(r.levelno == logging.ERROR for r in caplog.records)

    def test_require_signposting_logs_only_once(self, monkeypatch, caplog):
        import logging

        monkeypatch.setattr(sb, "signposting_available", lambda: False)
        monkeypatch.setattr(sb, "_missing_toponymy_warned", False)
        with caplog.at_level(logging.ERROR, logger=sb.logger.name):
            assert sb.require_signposting() is False
            assert sb.require_signposting() is False
        assert sum("toponymy" in r.message.lower() for r in caplog.records) == 1

    def test_version_string_when_missing(self, monkeypatch):
        from importlib import metadata

        def raise_missing(name):
            raise metadata.PackageNotFoundError(name)

        monkeypatch.setattr(sb.metadata, "version", raise_missing)
        assert sb.toponymy_version() == ""
