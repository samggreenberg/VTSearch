"""The cached tiled-VLAD Stage 1 (#3928): geometry, projection, scoring, round trip.

No SIFT runs here -- the GRID build is the measurement.  What is checked is the
part that decides whether a stored cell answers the same question the probe did:
the tile geometry, that a sliced projection equals the narrower fit, that a
page's score is the max over its own tiles and no one else's, and that a cell
survives being written and read back.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

_DOCMARKS = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "docmarks"


@pytest.fixture(scope="module")
def s1():
    sys.path.insert(0, str(_DOCMARKS))
    try:
        yield importlib.import_module("stage1_cell")
    finally:
        sys.path.remove(str(_DOCMARKS))


def _fake_aggregate(desc: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Stand-in for VLAD: mean descriptor, tiled to the codebook's flat width."""
    return np.tile(desc.mean(axis=0), codebook.shape[0])


class TestGeometry:
    def test_windows_overlap_by_half_and_stop_within_half_a_tile_of_the_edge(self, s1):
        """The last window starts on the stride grid, so it need not reach 1.0.

        At 0.25 x 0.18 the right edge lands exactly on 1.0 but the bottom stops
        at 0.99: a mark in the last 1% strip is seen only through the window that
        overlaps it.  This is the probe's geometry and #3928's numbers are
        measured under it, so it is pinned rather than corrected here.
        """
        wins = s1.tile_windows(0.25, 0.18)
        xs = sorted({w[0] for w in wins})
        assert xs[0] == 0.0 and pytest.approx(xs[1], abs=1e-9) == 0.125  # stride is half the width
        assert max(w[2] for w in wins) == pytest.approx(1.0, abs=1e-9)
        assert max(w[3] for w in wins) == pytest.approx(0.99, abs=1e-9)
        assert max(w[3] for w in wins) >= 1.0 - 0.18 / 2  # never more than half a tile short

    def test_matches_the_probe_it_replaces(self, s1):
        """#3928's probe owns the published numbers; a divergence here voids them."""
        probe = importlib.import_module("eval_stage1_tiles")
        assert s1.tile_windows(0.25, 0.18) == probe.tile_windows(0.25, 0.18)
        assert (s1.TILE_W, s1.TILE_H) == probe._ALL_LAYOUTS["t4"]
        assert s1.MIN_TILE_KP == probe.MIN_TILE_KP

    def test_only_tiles_with_enough_keypoints_get_a_row(self, s1):
        codebook = np.zeros((2, 4), dtype=np.float32)
        kp = np.concatenate([np.full((25, 2), 0.05), np.full((3, 2), 0.8)])  # one dense corner, one sparse
        desc = np.ones((28, 4), dtype=np.float32)
        rows, boxes = s1.tile_rows(kp, desc, codebook, _fake_aggregate, min_kp=20)
        assert rows.shape[0] == 1 and boxes.shape == (1, 4)
        assert boxes[0][0] == 0.0 and boxes[0][1] == 0.0

    def test_a_page_too_sparse_to_tile_still_gets_one_row(self, s1):
        codebook = np.zeros((2, 4), dtype=np.float32)
        rows, boxes = s1.tile_rows(np.full((3, 2), 0.5), np.ones((3, 4), dtype=np.float32), codebook, _fake_aggregate)
        assert rows.shape[0] == 1
        assert tuple(boxes[0]) == (0.0, 0.0, 1.0, 1.0)  # the whole page, as the probe scored it


class TestProjection:
    def _sample(self, rows=200, dims=64, seed=0):
        rng = np.random.default_rng(seed)
        basis = rng.normal(size=(8, dims))
        return (rng.normal(size=(rows, 8)) @ basis + 0.01 * rng.normal(size=(rows, dims))).astype(np.float32)

    def test_slicing_a_wide_fit_equals_the_narrow_fit(self, s1):
        """One pass over the corpus can write every width only if this holds."""
        sample = self._sample()
        wide = s1.fit_projection(sample, 16)
        narrow = s1.fit_projection(sample, 4)
        assert np.allclose(np.abs(wide.slice(4).components), np.abs(narrow.components), atol=1e-4)

    def test_projected_rows_are_unit_length(self, s1):
        proj = s1.fit_projection(self._sample(), 8)
        out = proj.apply(self._sample(rows=5, seed=1))
        assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)

    def test_whitening_makes_components_comparable(self, s1):
        sample = self._sample()
        proj = s1.fit_projection(sample, 8)
        spread = ((sample - proj.mean) @ proj.components.T).std(axis=0)
        assert np.allclose(spread, spread[0], rtol=0.15)  # no component dominates the dot product

    def test_a_projection_survives_a_round_trip(self, s1, tmp_path):
        proj = s1.fit_projection(self._sample(), 8)
        proj.save(tmp_path / "p.npz")
        back = s1.Projection.load(tmp_path / "p.npz")
        assert np.allclose(back.components, proj.components) and np.allclose(back.mean, proj.mean)

    def test_widening_is_refused(self, s1):
        with pytest.raises(ValueError, match="widen"):
            s1.fit_projection(self._sample(), 8).slice(16)


class TestScoring:
    def test_a_page_scores_the_max_over_its_own_tiles(self, s1):
        tiles = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8], [1.0, 0.0]], dtype=np.float32)
        starts = s1.starts_from_counts([2, 2])  # page A has rows 0-1, page B rows 2-3
        got = s1.page_scores(tiles, starts, np.array([1.0, 0.0], dtype=np.float32))
        assert list(np.round(got, 3)) == [1.0, 1.0]
        got = s1.page_scores(tiles, starts, np.array([0.0, 1.0], dtype=np.float32))
        assert list(np.round(got, 3)) == [1.0, 0.8]

    def test_starts_follow_uneven_tile_counts(self, s1):
        assert list(s1.starts_from_counts([3, 1, 5])) == [0, 3, 4]

    def test_top_k_is_ordered_and_breaks_ties_by_page_id(self, s1):
        ids = ["b", "a", "c"]
        got = s1.top_k(ids, np.array([0.5, 0.5, 0.9], dtype=np.float32), 3)
        assert [p for p, _ in got] == ["c", "a", "b"]

    def test_top_k_caps_at_the_pool_size(self, s1):
        assert len(s1.top_k(["a", "b"], np.array([1.0, 2.0], dtype=np.float32), 10)) == 2


class TestCellRoundTrip:
    def test_a_written_cell_reads_back_and_searches(self, s1, tmp_path):
        rng = np.random.default_rng(3)
        dirs = {0: tmp_path / "raw"}
        dirs[0].mkdir()
        tiles = s1.normalise(rng.normal(size=(5, 16)).astype(np.float32)).astype(np.float16)
        np.savez(
            dirs[0] / "shard-0000.npz",
            page_ids=np.array(["p1", "p2"]).astype("U"),
            counts=np.array([2, 3], dtype=np.int32),
            tiles=tiles,
            boxes=np.zeros((5, 4), dtype=np.float16),
        )
        (dirs[0] / "meta.json").write_text(
            '{"dim": 16, "budget": 8192, "pages": 2, "complete": true}', encoding="utf-8"
        )

        cell = s1.Cell(dirs[0])
        assert cell.page_ids == ["p1", "p2"] and list(cell.starts) == [0, 2]
        hits = cell.search(np.asarray(tiles[3], dtype=np.float32), k=2)
        assert hits[0][0] == "p2"  # row 3 belongs to p2, and it matches itself best
        assert hits[0][1] == pytest.approx(1.0, abs=2e-3)

    def test_a_cell_with_no_shards_is_an_error(self, s1, tmp_path):
        (tmp_path / "meta.json").write_text('{"complete": true, "pages": 0}', encoding="utf-8")
        with pytest.raises(ValueError, match="no shards"):
            s1.Cell(tmp_path)

    def test_a_partial_cell_refuses_to_open(self, s1, tmp_path):
        """A killed build leaves shards behind; searching them would miss pages silently."""
        (tmp_path / "meta.json").write_text('{"pages": 2, "complete": false}', encoding="utf-8")
        with pytest.raises(ValueError, match="PARTIAL"):
            s1.Cell(tmp_path)

    def test_a_cell_whose_shards_do_not_match_its_meta_is_an_error(self, s1, tmp_path):
        np.savez(
            tmp_path / "shard-0000.npz",
            page_ids=np.array(["p1"]).astype("U"),
            counts=np.array([1], dtype=np.int32),
            tiles=np.zeros((1, 4), dtype=np.float16),
            boxes=np.zeros((1, 4), dtype=np.float16),
        )
        (tmp_path / "meta.json").write_text('{"pages": 9, "complete": true}', encoding="utf-8")
        with pytest.raises(ValueError, match="claims 9"):
            s1.Cell(tmp_path)
