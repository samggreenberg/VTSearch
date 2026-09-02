"""The acquisition pool must be scored in the space the thresholds are cut in.

Autopilot's Hard pick locates its cutoff with an **absolute** comparison -
``ranking[cid] <= threshold`` in
:func:`~vtscore.eval.al_strategies._hard_pick_by_index` - so the ranking and the
cut have to live in one score space.  On a patch dataset every threshold the
harness reports (and the one it hands the selector) is fitted on the style's
region max-pooled scores, while the pool used to be scored by each media's
single whole-image vector.  A max over ~197 patch rows dominates that one row by
construction, so whole-image pool scores sit systematically *below* the cut,
almost the whole pool falls under it, and the cutoff index slides toward the top
of the ranking - the simulated user then votes on far more positive items than
the app's user, whose learned sort ranks the very same pooled scores its
threshold cuts.  That is issue #2943, and it contaminated #2876's
acquisition-inclusion study, whose entire subject is where the cut sits in the
ranking.

These tests pin the two halves of the fix: the pool scorer's geometry, and the
fact that the ``*_pool_percentile`` diagnostics - added so a misplaced
acquisition cut could not hide again - are now measured in the same space as the
cuts they locate.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vtscore.eval.patch_styles import resolve_style
from vtscore.eval.step_model import StepModel
from vtscore.eval.step_trainers import _score_pool
from vtscore.eval.voting_iterations import simulate_voting_iterations

from .test_max_patch_style import DIM, _linear_scorer, _planted_dataset


def _step(direction) -> StepModel:
    """A hand-built step model - the same module behind both scoring paths.

    ``predict`` is the trainer-agnostic whole-image path and ``torch_model`` the
    one the styles max-pool, so any difference between the two scorers is the
    geometry rather than the weights.
    """
    model = _linear_scorer(direction)

    def predict(embs):
        with torch.no_grad():
            return torch.sigmoid(model(torch.tensor(np.asarray(embs, dtype=np.float32)))).squeeze(1).numpy()

    return StepModel(predict=predict, torch_model=model, backend="torch", device="cpu")


def _step_and_clips(seed=3):
    """A hand-built model over a planted patch dataset - no training involved."""
    medias, target = _planted_dataset(n_per_cat=8, seed=seed)
    return medias, _step(target)


class TestPoolScoreGeometry:
    def test_style_pool_scores_are_the_style_pooled_scores(self):
        """The pooled score, restricted to the pool - not the whole-image score."""
        clips, step = _step_and_clips()
        style = resolve_style("max_patch")
        pool = sorted(clips)[:5]

        got = _score_pool(step, pool, clips, region_aware=True, style_obj=style, sim_clips=clips)

        pooled = style.score_media(step.torch_model, clips)
        assert sorted(got) == pool
        for cid in pool:
            assert got[cid] == pytest.approx(pooled[cid], abs=1e-9)

    def test_the_two_spaces_really_differ(self):
        """Guard against the fix being a no-op on this fixture.

        The whole-image row is row 0 of the max-pool stack, so the pooled score
        dominates it deterministically; the test is only interesting if the
        inequality is strict somewhere.
        """
        clips, step = _step_and_clips()
        style = resolve_style("max_patch")
        pool = sorted(clips)

        pooled = _score_pool(step, pool, clips, region_aware=True, style_obj=style, sim_clips=clips)
        whole_image = _score_pool(step, pool, clips)

        assert all(pooled[cid] >= whole_image[cid] - 1e-9 for cid in pool)
        assert any(pooled[cid] > whole_image[cid] + 1e-6 for cid in pool)

    def test_precomputed_sim_scores_are_reused_verbatim(self):
        """The safe-threshold path already scored the sim set in this geometry.

        Restricting those scores must give exactly what a fresh scoring pass
        would, or the free path and the paid path would be different arms.
        """
        clips, step = _step_and_clips()
        style = resolve_style("max_patch")
        pool = sorted(clips)[::2]
        score_map = style.score_media(step.torch_model, clips)
        ids = list(score_map)

        reused = _score_pool(
            step,
            pool,
            clips,
            region_aware=True,
            style_obj=style,
            sim_clips=clips,
            sim_scored=(ids, [score_map[cid] for cid in ids]),
        )
        rescored = _score_pool(step, pool, clips, region_aware=True, style_obj=style, sim_clips=clips)
        assert reused == rescored

    def test_single_vector_datasets_keep_the_whole_image_path(self):
        """No style, no patch grid: whole-image *is* the threshold's space."""
        rng = np.random.default_rng(0)
        clips = {
            cid: {"id": cid, "category": "c", "embeddings": {"emb": rng.standard_normal(DIM).astype(np.float32)}}
            for cid in range(1, 6)
        }
        step = _step(np.ones(DIM))
        pool = sorted(clips)

        got = _score_pool(step, pool, clips)
        assert sorted(got) == pool
        # Same numbers with the new keyword arguments left at their defaults.
        assert got == _score_pool(step, pool, clips, region_aware=False, style_obj=None, sim_clips=None)

    def test_empty_pool_scores_nothing(self):
        clips, step = _step_and_clips()
        assert _score_pool(step, [], clips, region_aware=True, style_obj=resolve_style("max_patch")) == {}


class TestPoolPercentilesLocateTheCut:
    """The diagnostic columns now measure the cut in the pool's own space.

    With whole-image pool scores against a pooled cut, nearly every pool item
    scored below the threshold, so ``report_pool_percentile`` pinned near 1.0
    step after step - the cutoff index sitting at the very top of the ranking -
    and the column could not reveal the shift it was added to catch.
    """

    def test_the_reporting_cut_does_not_sit_above_the_whole_pool(self):
        medias, target = _planted_dataset(n_per_cat=20, seed=41)
        seed_scores = resolve_style("max_patch").exemplar_sims(medias, target)
        rows = simulate_voting_iterations(
            medias,
            target_category="cat0",
            seed=1,
            dataset_name="synthetic",
            region_voting=True,
            max_steps=12,
            style="max_patch",
            seed_scores=seed_scores,
            atlas_min_node_size=5,
        )
        assert rows

        pcts = [r["report_pool_percentile"] for r in rows if np.isfinite(r["report_pool_percentile"])]
        assert pcts, "no step produced a finite pool percentile"
        assert float(np.median(pcts)) < 0.99, pcts

    def test_the_acquisition_cut_stays_at_or_above_the_reporting_cut(self):
        """Both cuts are located in the pool's space, so the #2876 ordering holds.

        The acquisition cut is taken ``ACQUISITION_INCLUSION_OFFSET`` inclusion
        steps below the reporting one, which *raises* it - so it can only sit
        further up the ranking, i.e. at a percentile at or above the reporting
        cut's.  Measured in mismatched spaces this said nothing.
        """
        medias, target = _planted_dataset(n_per_cat=20, seed=42)
        seed_scores = resolve_style("max_patch").exemplar_sims(medias, target)
        rows = simulate_voting_iterations(
            medias,
            target_category="cat0",
            seed=1,
            dataset_name="synthetic",
            region_voting=True,
            max_steps=12,
            style="max_patch",
            seed_scores=seed_scores,
            atlas_min_node_size=5,
        )
        assert rows
        for r in rows:
            if np.isfinite(r["acq_pool_percentile"]) and np.isfinite(r["report_pool_percentile"]):
                assert r["acq_pool_percentile"] >= r["report_pool_percentile"], r
