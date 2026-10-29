"""Issue #4936's instrument: every fold model's scores ride on the test-score snapshot beside the final model's.

Eval only: ``fold_scale_steps`` adds arrays to the snapshots ``run_cells.py``
keeps for the listed clicks and changes nothing the session does.  What a study
rests on is that the arrays are the step's own: the withheld half under every
model in the snapshot's order, and the sim set under every model, from which
the haystacks the labels line's transfers are fitted on replay exactly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vtscore.eval.voting_iterations import _fold_scale_gate, simulate_voting_iterations
from vtscore.training.thresholds import transfer_fold_orderings


def _separable(n_per_cat: int = 30, dim: int = 16) -> dict[int, dict]:
    rng = np.random.RandomState(0)
    medias: dict[int, dict] = {}
    for i in range(2 * n_per_cat):
        cat = "alpha" if i < n_per_cat else "beta"
        centre = 1.0 if cat == "alpha" else -1.0
        emb = rng.normal(centre, 0.9, dim).astype(np.float32)
        medias[i + 1] = {"id": i + 1, "embeddings": {"emb": emb}, "category": cat}
    return medias


_STEPS = (8, 12, 18)


def _run(fold_scale_steps=_STEPS, **kw):
    snaps: list[dict] = []
    rows = simulate_voting_iterations(
        _separable(),
        "alpha",
        seed=0,
        max_steps=20,
        calibrate_count=2,
        style="whole_image",
        emit_calibration_metrics=True,
        safe_thresholds=True,
        label_quota=False,
        beta=1.0,
        test_score_sink=snaps,
        fold_scale_steps=fold_scale_steps,
        **kw,
    )
    return rows, snaps


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
    return bool(a == b)


def _kept(snaps: list[dict], at: int) -> dict | None:
    """The snapshot run_cells.py keeps as ``t<at>``: the first ordinary one at or past the click."""
    ordinary = [f for f in snaps if f["phase"] not in ("check", "ceiling")]
    return next((f for f in ordinary if int(f["t"]) >= at), None)


class TestFoldScaleInstrument:
    def test_refuses_what_it_cannot_record(self):
        with pytest.raises(ValueError, match="fold_scale_steps needs"):
            simulate_voting_iterations(
                _separable(), "alpha", seed=0, max_steps=2, style="whole_image", fold_scale_steps=(5,)
            )
        with pytest.raises(ValueError, match="fold_scale_steps needs"):
            simulate_voting_iterations(
                _separable(),
                "alpha",
                seed=0,
                max_steps=2,
                style="whole_image",
                emit_calibration_metrics=True,
                safe_thresholds=False,
                test_score_sink=[],
                fold_scale_steps=(5,),
            )

    def test_the_session_is_unchanged(self):
        from vtscore.eval.voting_columns import TIMING_COLUMNS

        with_it, snaps = _run()
        without, plain = _run(fold_scale_steps=None)
        assert len(with_it) == len(without)
        for r, s in zip(with_it, without):
            assert set(r) == set(s)
            assert all(_same(r[k], s[k]) for k in r if k not in TIMING_COLUMNS), r["t"]
        assert len(snaps) == len(plain)
        for a, b in zip(snaps, plain):
            assert all(_same(a[k], b[k]) for k in b), a["t"]

    def test_the_arrays_ride_on_the_snapshots_run_cells_keeps(self):
        _rows, snaps = _run()
        carrying = [f for f in snaps if "fold_test_scores" in f]
        kept = {id(f) for f in (_kept(snaps, at) for at in _STEPS) if f is not None}
        assert kept and {id(f) for f in carrying} == kept

    def test_every_model_on_the_same_images(self):
        _rows, snaps = _run()
        live = [f for f in snaps if "fold_test_scores" in f and f["n_live_folds"] > 0]
        assert live, "no listed click had live calibration folds"
        for f in live:
            k = f["n_live_folds"]
            assert f["fold_test_scores"].shape == (k, f["ids"].size)
            assert f["fold_sim_scores"].shape == (k, f["sim_ids"].size)
            assert f["sim_scores"].shape == f["sim_ids"].shape == f["sim_labels"].shape == f["sim_vote"].shape
            assert set(np.unique(f["fold_index"]).tolist()) == set(range(k))
            # The withheld half is unseen: none of its images is in the sim set, so none is a vote.
            assert not set(f["ids"].tolist()) & set(f["sim_ids"].tolist())
            # A fold model of a separable task ranks the withheld positives above the negatives.
            pos = f["labels"] == 1
            assert all(f["fold_test_scores"][j][pos].mean() > f["fold_test_scores"][j][~pos].mean() for j in range(k))

    def test_each_held_out_score_is_its_fold_model_on_that_vote(self):
        """The calibration's held-out score of a vote is the same fold model's score of it in the sim pass."""
        _rows, snaps = _run()
        live = [f for f in snaps if "fold_test_scores" in f and f["n_live_folds"] > 0]
        assert live
        for f in live:
            assert f["fold_ids"].shape == f["fold_scores"].shape
            assert (f["fold_ids"] >= 0).all(), "the row-wise path names every held-out vote"
            at = {int(cid): i for i, cid in enumerate(f["sim_ids"])}
            idx = np.array([at[int(v)] for v in f["fold_ids"]])
            assert (f["sim_vote"][idx] == f["fold_labels"]).all()
            assert (f["sim_labels"][idx] == f["fold_labels"]).all()  # the simulated user answers from the truth
            on_sim = f["fold_sim_scores"][f["fold_index"], idx]
            np.testing.assert_allclose(on_sim, f["fold_scores"], rtol=1e-5, atol=1e-7)

    def test_the_haystacks_replay(self):
        """The quantile transfer fitted on the replayed haystacks moves the folds exactly as the run's own did."""
        rows, snaps = _run(labels_scale_shadow="quantile")
        by_t = {r["t"]: r for r in rows if r["pool_variant"] == "max" and r["gmm_variant"] == ""}
        checked = 0
        for f in snaps:
            if "fold_test_scores" not in f or f["n_live_folds"] == 0:
                continue
            unvoted = f["sim_vote"] < 0 if f["fold_sim_excluding"] else np.ones(f["sim_vote"].size, dtype=bool)
            hays = [f["fold_sim_scores"][j][unvoted] for j in range(f["n_live_folds"])]
            final = f["sim_scores"][unvoted]
            orderings = [
                (f["fold_scores"][f["fold_index"] == j].tolist(), f["fold_labels"][f["fold_index"] == j].tolist())
                for j in range(f["n_live_folds"])
            ]
            moved = transfer_fold_orderings(orderings, hays, final, tails=False)
            gate = _fold_scale_gate(orderings, moved, hays)
            row = by_t[f["t"]]
            assert gate["xfer_dlogit_med"] == pytest.approx(row["xfer_dlogit_med"], abs=2e-6)
            assert _same(round(gate["xfer_clip_good"], 6), row["xfer_clip_good"])
            checked += 1
        assert checked
