"""CLI autodetect scores a media at the same geometry the GUI does.

Issue #3180: ``_score_direct_all`` / ``_score_one_detector`` forwarded the
**image-level** embedding matrix, while the threshold they compare each score
against is fitted on ``_score_all_media`` - which, on a patch dataset,
max-pools every media's score-row stack (image-level vector + every raw patch).
A detector trained on a small object therefore scored ~0.8 in the GUI and ~0.5
in the CLI on the identical media, so a CLI run could report zero hits where a
GUI Find reported dozens, with no error anywhere.

Both CLI scorers now build their rows through
:func:`~vtscore.detectors.training.scoring_rows_for_snap`, the same builder
``_score_all_media`` uses, so the two cannot diverge again. These tests pin the
per-media score itself (against the max over the media's own score rows), not
just the hit count, so a future change that pools differently fails here rather
than only on a threshold that happens to straddle the gap.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import app as app_module  # noqa: F401  (activates the default dataset context)
from vtscore.cli import _score_direct_all, _score_one_detector
from vtscore.detectors.training import _score_all_media, scoring_rows_for_snap
from vtscore.embedding.matrix import media_score_rows

DIM = 4


def _basis(i: int) -> np.ndarray:
    return np.eye(DIM, dtype=np.float32)[i]


def _patch_medias() -> dict[int, dict]:
    """Two patch media whose *object* lives in a patch, not the image vector.

    Every image-level vector is ``e0`` (which the head below scores low); media
    1 hides one ``e1`` patch (which it scores high) while media 2's grid is all
    ``e0``. Max-pooling is therefore the only geometry that separates them.
    """
    grid_hit = np.stack([np.stack([_basis(0), _basis(1)]), np.stack([_basis(0), _basis(0)])])
    grid_miss = np.stack([np.stack([_basis(0), _basis(0)]), np.stack([_basis(0), _basis(0)])])
    return {
        1: {
            "id": 1,
            "media_type": "image",
            "embedder": "dinov3_patch",
            "embeddings": {"dinov3_patch": _basis(0)},
            "patch_grid": grid_hit.astype(np.float16),
            "filename": "m1.png",
            "md5": "m1",
        },
        2: {
            "id": 2,
            "media_type": "image",
            "embedder": "dinov3_patch",
            "embeddings": {"dinov3_patch": _basis(0)},
            "patch_grid": grid_miss.astype(np.float16),
            "filename": "m2.png",
            "md5": "m2",
        },
    }


def _fires_on_dim1_mlp() -> nn.Sequential:
    linear = nn.Linear(DIM, 1)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[0.0, 10.0, 0.0, 0.0]]))
        linear.bias.copy_(torch.tensor([-5.0]))
    return nn.Sequential(linear).eval()


def _pooled_score(model: nn.Sequential, media: dict) -> float:
    """The max over this media's own score rows - what the GUI reports."""
    rows = torch.from_numpy(np.asarray(media_score_rows(media, "dinov3_patch"), dtype=np.float32))
    with torch.no_grad():
        return float(torch.sigmoid(model(rows)).max())


def _score_by_id(result: dict) -> dict[int, float]:
    return {h["id"]: h["score"] for h in result["hits"] + result["negative_hits"]}


class TestCliScoresAtPoolGeometry:
    """The CLI's per-media score is the max over that media's score rows."""

    def test_direct_path_max_pools_patch_rows(self):
        model = _fires_on_dim1_mlp()
        medias = _patch_medias()
        detector_mlps = {"d": {"mlp": model, "threshold": 0.5}}

        out = _score_direct_all(["d"], detector_mlps, medias, "dinov3_patch")

        scores = _score_by_id(out["d"])
        assert scores[1] == round(_pooled_score(model, medias[1]), 4)
        assert scores[2] == round(_pooled_score(model, medias[2]), 4)
        # Only media 1 carries the object, and only pooling can see it.
        assert {h["id"] for h in out["d"]["hits"]} == {1}

    def test_routed_path_max_pools_patch_rows(self):
        model = _fires_on_dim1_mlp()
        medias = _patch_medias()
        rows = scoring_rows_for_snap(medias, "dinov3_patch")

        out = _score_one_detector(
            "d",
            {"mlp": model, "threshold": 0.5},
            medias,
            rows,
            {cid: cid for cid in medias},
        )

        assert {h["id"] for h in out["hits"]} == {1}
        assert _score_by_id(out)[1] == round(_pooled_score(model, medias[1]), 4)

    def test_agrees_with_the_app_scorer(self):
        """The CLI and ``_score_all_media`` report the same score per media."""
        model = _fires_on_dim1_mlp()
        medias = _patch_medias()

        all_ids, app_scores, _best = _score_all_media(model, medias, "dinov3_patch")
        out = _score_direct_all(["d"], {"d": {"mlp": model, "threshold": 0.5}}, medias, "dinov3_patch")
        cli_scores = _score_by_id(out["d"])

        for cid, app_score in zip(all_ids, app_scores, strict=True):
            assert cli_scores[cid] == round(app_score, 4)

    def test_gridless_dataset_unchanged(self):
        """A dataset with no patch grid still scores one row per media."""
        model = _fires_on_dim1_mlp()
        medias = _patch_medias()
        for media in medias.values():
            media.pop("patch_grid")
        medias[2]["embeddings"]["dinov3_patch"] = _basis(1)

        out = _score_direct_all(["d"], {"d": {"mlp": model, "threshold": 0.5}}, medias, "dinov3_patch")

        # Image-level vectors decide: media 2 is the e1 one now.
        assert {h["id"] for h in out["d"]["hits"]} == {2}
