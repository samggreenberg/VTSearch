"""Each AutoFind result carries its cut at every Threshold preset (#5540).

The Find Results dialog shows what the other two notches of the Threshold would
have kept.  A run cannot retrain, but the balance is a pure cutoff knob: the
cut at another balance is re-derived from the estimators the train cached
(:func:`~vtscore.state.core.recut_detector_threshold`), and the dialog splits the
scores the run already holds at it.  Pinned here:

- every result names the balance it was cut at (``beta``) and the cut at each
  preset (``lines``), from false positives to false negatives;
- the detector's own preset keeps the run's ``threshold`` exactly, so its notch
  reproduces the run;
- a cut that cannot be re-derived, or is not a number, keeps the run's
  ``threshold``, so that notch keeps what the run kept.
"""

from __future__ import annotations

import math

import numpy as np

import vtscore.state as state_mod
import vtscore.state.core as core_mod
import vtsearch.runtime.autofind as autofind_mod

DIM = 4
EMB = "test_embedder"


def _media(cid: int, vec: np.ndarray) -> dict:
    return {
        "id": cid,
        "media_type": "image",
        "embedder": EMB,
        "embeddings": {EMB: vec},
        "filename": f"m{cid}.png",
        "md5": f"m{cid}",
        "origin_name": f"m{cid}.png",
        "origin": {"importer": "test", "params": {}},
    }


def _corpus() -> dict[int, dict]:
    """Two hundred media: ids 1-100 lean toward one direction, ids 101-200 toward another.

    Noisy enough that the two overlap, so the balance moves the line: on two
    clean clusters every preset would cut in the same gap.
    """
    rng = np.random.default_rng(1)
    corpus: dict[int, dict] = {}
    for cid in range(1, 201):
        base = np.array([1.0, 0.2, 0.0, 0.0]) if cid <= 100 else np.array([0.0, 0.2, 1.0, 0.0])
        vec = base + 1.0 * rng.standard_normal(DIM)
        corpus[cid] = _media(cid, (vec / np.linalg.norm(vec)).astype(np.float32))
    return corpus


def _activate(corpus: dict[int, dict]) -> None:
    from vtscore.state.core import DatasetContext, register_context, set_thread_dataset_context

    ctx = DatasetContext("ds-autofind-lines")
    ctx.medias.update(corpus)
    register_context(ctx)
    set_thread_dataset_context(ctx)


def _write_detector(name: str, corpus: dict[int, dict], goods: list[int], bads: list[int]) -> None:
    from vtscore.detectors.registry import register_detector
    from vtscore.detectors.store import _detector_path
    from vtscore.detectors.store import _write_detector as write

    labels = [
        {
            "md5": corpus[cid]["md5"],
            "label": label,
            "origin": corpus[cid]["origin"],
            "origin_name": corpus[cid]["origin_name"],
            "filename": corpus[cid]["filename"],
        }
        for label, ids in (("good", goods), ("bad", bads))
        for cid in ids
    ]
    write(_detector_path(name), {"name": name, "media_type": "image", "labelset": {"labels": labels}})
    register_detector(name=name, media_type="image", num_training=len(labels))


class TestBalanceLines:
    def test_the_own_preset_keeps_the_run_cut_and_the_others_are_recut(self, monkeypatch):
        ctx = object()
        recut: list[float] = []

        def _recut(det_ctx, _inclusion, *, beta):
            assert det_ctx is ctx
            recut.append(beta)
            return {4.0: 0.2, 0.25: 0.8}[beta]

        monkeypatch.setattr(state_mod, "detector_beta", lambda _ctx: 1.0)
        monkeypatch.setattr(core_mod, "recut_detector_threshold", _recut)

        beta, lines = autofind_mod._balance_lines(ctx, 0.512345)

        assert beta == 1.0
        assert lines == [
            {"beta": 4.0, "threshold": 0.2},
            {"beta": 1.0, "threshold": 0.5123},
            {"beta": 0.25, "threshold": 0.8},
        ]
        assert sorted(recut) == [0.25, 4.0]

    def test_a_cut_that_cannot_be_rederived_keeps_the_run_cut(self, monkeypatch):
        monkeypatch.setattr(state_mod, "detector_beta", lambda _ctx: 4.0)
        monkeypatch.setattr(
            core_mod,
            "recut_detector_threshold",
            lambda _ctx, _inclusion, *, beta: None if beta == 1.0 else math.inf,
        )

        beta, lines = autofind_mod._balance_lines(object(), 0.3)

        assert beta == 4.0
        assert [line["threshold"] for line in lines] == [0.3, 0.3, 0.3]

    def test_no_context_has_no_balance_and_keeps_the_run_cut(self):
        beta, lines = autofind_mod._balance_lines(None, 0.4)

        assert beta is None
        assert lines == [
            {"beta": 4.0, "threshold": 0.4},
            {"beta": 1.0, "threshold": 0.4},
            {"beta": 0.25, "threshold": 0.4},
        ]


class TestAutoDetectRoute:
    def test_each_result_carries_its_cut_at_every_preset(self, client):
        from vtsearch.settings import add_autofind_detector

        corpus = _corpus()
        _activate(corpus)
        _write_detector("af-lines", corpus, goods=list(range(1, 16)), bads=list(range(101, 131)))
        add_autofind_detector("af-lines")

        resp = client.post("/api/auto-detect", json={})
        assert resp.status_code == 200, resp.get_json()
        result = resp.get_json()["results"]["af-lines"]

        assert result["beta"] == 1.0
        lines = result["lines"]
        assert [line["beta"] for line in lines] == [4.0, 1.0, 0.25]
        # The detector's own notch is the run's cut, so it reproduces the run's split.
        assert lines[1]["threshold"] == result["threshold"]
        # Re-cut, not copied: toward false positives keeps more, toward false negatives fewer.
        assert lines[0]["threshold"] < lines[1]["threshold"] < lines[2]["threshold"]
        # The run scored every item, so any notch's split is a split of what it holds.
        assert len(result["hits"]) + len(result["negative_hits"]) == len(corpus)
