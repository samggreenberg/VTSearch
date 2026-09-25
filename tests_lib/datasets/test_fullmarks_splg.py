"""The SuperPoint + LightGlue ranking helpers behind #3911's measurements.

No model runs here -- the GPU runs are the measurement.  What a laptop can check
is the ordering logic every number in the report passes through: how inlier
counts rank a pool, how the verify-then-SigLIP fallback orders what is below the
floor, and that a shortlist only ever re-orders SigLIP's top K.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_FULLMARKS = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "fullmarks"


@pytest.fixture(scope="module")
def mods():
    sys.path.insert(0, str(_FULLMARKS))
    try:
        yield {"rank": importlib.import_module("eval_splg_rank"), "short": importlib.import_module("sim_shortlist")}
    finally:
        sys.path.remove(str(_FULLMARKS))


class TestRankingByInliers:
    def test_inliers_first_then_tentative_matches_then_page_id(self, mods):
        inliers = {"a": (10, 3), "b": (30, 0), "c": (10, 9), "d": (0, 50)}
        assert mods["rank"].rank_splg(inliers, {"a", "b", "c", "d", "e"}) == ["b", "c", "a", "d", "e"]

    def test_the_fallback_puts_only_pages_over_the_floor_ahead_of_siglip(self, mods):
        inliers = {"weak": (20, 0), "strong": (40, 0), "verified": (24, 0)}
        siglip = {"weak": 0.9, "strong": 0.1, "verified": 0.2, "plain": 0.5}
        ranked = mods["rank"].rank_splg_siglip(inliers, siglip, {"weak", "strong", "verified", "plain"}, floor=24)
        # over the floor by inliers, then the rest by SigLIP -- 20 inliers is not a verification
        assert ranked == ["strong", "verified", "weak", "plain"]


class TestShortlist:
    def test_only_the_top_k_is_reordered(self, mods):
        siglip_ranked = ["s1", "s2", "s3", "s4", "s5"]
        inliers = {"s2": [50, 0], "s5": [99, 0]}
        # s5 is below K, so its inliers were never computed in the real design and must not lift it
        assert mods["short"].shortlist_rank(siglip_ranked, inliers, 3) == ["s2", "s1", "s3", "s4", "s5"]

    def test_unverified_shortlist_pages_keep_siglip_order(self, mods):
        assert mods["short"].shortlist_rank(["a", "b", "c"], {}, 3) == ["a", "b", "c"]

    def test_k_covering_the_pool_is_the_exhaustive_ranking(self, mods):
        siglip_ranked = ["a", "b", "c"]
        inliers = {"c": [5, 0], "b": [9, 0]}
        assert mods["short"].shortlist_rank(siglip_ranked, inliers, 3) == ["b", "c", "a"]
