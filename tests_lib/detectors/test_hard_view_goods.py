"""#5413: the app and the harness name a hard-view face's Good to the head's fit, which weighs it less.

A face under its space's Face Quality cut (#5372) among clear starting faces cost a four-photo FHIBE
session F 0.10-0.11 at 50 votes at beta 4 (#5376). The trained head, a linear SVM started from the
Goods' centroid, now weighs such a Good at ``HARD_VIEW_GOOD_WEIGHT``. The rule is
``vtscore.training.svm_prior``'s (``test_svm_prior_head.py``); this file checks that both callers enter it.
"""

from __future__ import annotations

import numpy as np
import pytest

import vtscore.detectors.labelset_training as lt
from vtscore.datasets.labelset import LabeledElement, LabelSet
from vtscore.detectors.labelset_elements import stable_element_id
from vtscore.media.face.imagekit_bridge import COMPAT_METADATA_KEY, QUALITY_METADATA_KEY
from vtscore.state.core import DetectorContext
from vtscore.training import svm_prior

EMBEDDER = "test_embedder"
DIM = 16
COMPAT = "oss-arcface-mbf:1"


def _unit(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x)).astype(np.float32)


def _faces(n: int = 60, hard: tuple[int, ...] = (1,), seed: int = 5413) -> dict[int, dict]:
    """A face corpus: one person (ids 1-12) and other people, each stamped with its Face Quality."""
    rng = np.random.default_rng(seed)
    person = rng.standard_normal(DIM)
    snap = {}
    for cid in range(1, n + 1):
        vec = person + (0.4 if cid <= 12 else 2.0) * rng.standard_normal(DIM)
        snap[cid] = {
            "id": cid,
            "md5": f"m{cid}",
            "media_type": "face",
            "embedder": EMBEDDER,
            "embeddings": {EMBEDDER: _unit(vec)},
            "custom_metadata": {QUALITY_METADATA_KEY: 9.0 if cid in hard else 25.0, COMPAT_METADATA_KEY: COMPAT},
        }
    return snap


def _labelset(goods: list[int], bads: list[int]) -> LabelSet:
    return LabelSet(
        [LabeledElement(md5=f"m{c}", label="good", origin_name=f"m{c}") for c in goods]
        + [LabeledElement(md5=f"m{c}", label="bad", origin_name=f"m{c}") for c in bads]
    )


def _ctx(snap: dict[int, dict], labelset: LabelSet) -> DetectorContext:
    ctx = DetectorContext("", name="faces", media_type="face")
    ctx.embedder = EMBEDDER
    for elem in labelset.elements:
        ctx.label_embeddings[stable_element_id(elem)] = snap[int(elem.md5[1:])]["embeddings"][EMBEDDER]
    return ctx


class TestTheAppNamesItsHardGoods:
    def test_only_resolved_hard_goods(self):
        snap = _faces(hard=(1, 30))
        labelset = _labelset([1, 2, 3], [30, 31])  # 30 is a hard Bad: never named
        ctx = _ctx(snap, labelset)
        got = lt.hard_view_good_vectors(ctx, labelset, snap)
        assert [v.tobytes() for v in got] == [snap[1]["embeddings"][EMBEDDER].tobytes()]
        assert lt.hard_view_good_vectors(ctx, labelset, None) == []

    def test_a_media_with_no_verdict_is_not_named(self):
        snap = _faces(hard=(1,))
        snap[1]["custom_metadata"] = {}
        labelset = _labelset([1, 2, 3], [30])
        assert lt.hard_view_good_vectors(_ctx(snap, labelset), labelset, snap) == []

    @pytest.mark.parametrize("hard,want", [((1,), True), ((), False)])
    def test_a_training_pass_weighs_the_hard_good(self, monkeypatch, hard, want):
        """The learned sort's fit sees the hard Good's weight; with none hard it sees none."""
        snap = _faces(hard=hard)
        labelset = _labelset([1, 2, 3, 4], list(range(30, 42)))
        ctx = _ctx(snap, labelset)
        monkeypatch.setattr(lt, "populate_label_embeddings", lambda *a, **k: None)
        seen: list[bool] = []
        original = svm_prior.hard_view_good_weights

        def spy(X, y):
            out = original(X, y)
            seen.append(out is not None)
            return out

        monkeypatch.setattr(svm_prior, "hard_view_good_weights", spy)
        lt.labelset_train_and_score(ctx, labelset, media_type="face", clips_dict=snap)
        assert seen and any(seen) is want


class TestTheHarnessNamesItsPoolsHardFaces:
    def test_every_hard_face_of_the_pool(self):
        from vtscore.eval.voting_iterations import _fits_weigh_hard_views

        snap = _faces(hard=(1, 30))

        @_fits_weigh_hard_views
        def session(clips_dict, target_category, seed):
            return svm_prior._HARD_VIEW_ROWS.get()

        keys = session(snap, "person", 0)
        assert keys == {snap[c]["embeddings"][EMBEDDER].tobytes() for c in (1, 30)}
        assert session(_faces(hard=()), "person", 0) is None
        assert svm_prior._HARD_VIEW_ROWS.get() is None
