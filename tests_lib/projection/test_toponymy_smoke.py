"""Slow smoke test: the real toponymy library, end-to-end through our glue.

toponymy is installed ``--no-deps`` (its ``transformers<5`` pin is bypassed —
see ``docs/plans/vtsbrowse-toponymy.md``), so the resolver never checks that a
future environment still satisfies it.  This test is that check: import the
library for real, run :func:`build_region_labels` (clusterable UMAP →
multiresolution clustering → contrastive keyphrases → KeyphraseNamer) on a
tiny seeded corpus, and assert signs come out.  Numba compilation makes this
a ~1 minute test → ``slow`` marker; run with ``-m slow``.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

pytest.importorskip("toponymy")

from vtscore.projection.signpost_build import build_region_labels  # noqa: E402
from vtscore.projection.umap_projection import Projection  # noqa: E402

pytestmark = pytest.mark.slow

_DIM = 16
_WORDS = {
    0: ["dog barking", "dog growl", "puppy bark", "dog howl"],
    1: ["rain falling", "thunder storm", "rain drops", "storm wind"],
    2: ["car engine", "engine idle", "car horn", "traffic noise"],
}


class SeededEmbedder:
    name = "seeded"
    supports_text = True

    def embed_text(self, text: str):
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % 2**32
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(_DIM).astype(np.float32)
        return vec / np.linalg.norm(vec)


def _seeded_corpus(per_cluster: int = 100):
    """A tiny 3-cluster seeded corpus + its 2-D layout, for the real fit."""
    rng = np.random.default_rng(42)
    centers = rng.standard_normal((3, _DIM)).astype(np.float32) * 4
    rows, texts = [], []
    for c in range(3):
        rows.append(centers[c] + rng.standard_normal((per_cluster, _DIM)).astype(np.float32))
        texts.extend(_WORDS[c][i % len(_WORDS[c])] for i in range(per_cluster))
    matrix = np.vstack(rows)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    n = matrix.shape[0]
    coords = rng.standard_normal((n, 2)).astype(np.float32)
    proj = Projection("smoke-proj", list(range(1, n + 1)), coords, "pca")
    return proj, matrix, texts


def test_real_toponymy_fit_produces_signs():
    proj, matrix, texts = _seeded_corpus()

    label_set = build_region_labels(
        proj,
        matrix,
        matrix,
        texts,
        SeededEmbedder(),
        object_description="sounds",
        corpus_description="a tiny synthetic corpus of animal, weather, and traffic sounds",
    )

    assert label_set.projection_id == "smoke-proj"
    assert label_set.labels, "the real pipeline produced no signs"
    assert all(lab.source == "keyphrase" for lab in label_set.labels)
    assert all(lab.text.strip() for lab in label_set.labels)
    # The coarsest layer sits at zoom band 0 and every anchor is a real
    # layout point (a medoid, not a centroid off in empty space).
    assert min(lab.level for lab in label_set.labels) == 0.0
    coord_set = {(round(float(x), 4), round(float(y), 4)) for x, y in np.asarray(proj.coords)}
    assert all((round(lab.x, 4), round(lab.y, 4)) in coord_set for lab in label_set.labels)
    # The keyphrase namer names from the corpus vocabulary.
    corpus_vocab = {w for words in _WORDS.values() for phrase in words for w in phrase.split()}
    named_words = {w for lab in label_set.labels for w in lab.text.lower().split()}
    assert named_words & corpus_vocab


def test_real_toponymy_fit_is_quiet_and_reports_layers(capsys):
    """The library's ``Layer N found M clusters`` prints and its per-cluster
    tqdm bars must stay off; the layer/topic breakdown surfaces through the
    ``on_progress`` callback as a determinate, per-layer naming bar instead."""
    proj, matrix, texts = _seeded_corpus()

    progress: list[tuple[int, int, str]] = []
    build_region_labels(
        proj,
        matrix,
        matrix,
        texts,
        SeededEmbedder(),
        object_description="sounds",
        corpus_description="a tiny synthetic corpus of animal, weather, and traffic sounds",
        on_progress=lambda current, total, message: progress.append((current, total, message)),
    )

    captured = capsys.readouterr()
    # The exact stdout flood the user reported ("Layer 0 found 65 clusters").
    assert "found" not in captured.out.lower()
    assert "Layer" not in captured.out
    # tqdm progress bars (stderr) from exemplar selection / keyphrase mining
    # must not flash by.
    for bar_desc in ("exemplars", "informative keyphrases", "Building topic names by layer"):
        assert bar_desc not in captured.err

    # Naming reported a determinate bar broken down by layer (0-based, finest
    # first), the "measuring layers 0,1,2" upgrade.
    naming = [(c, t, m) for c, t, m in progress if "layer" in m.lower()]
    assert naming, "naming never reported per-layer progress"
    assert all(total > 0 for _, total, _ in naming), "naming bar was indeterminate"
    assert all(current <= total for current, total, _ in naming), "naming count overran its total"
    assert any("layer 0" in m.lower() for _, _, m in naming), "finest layer never reported"
