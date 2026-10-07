"""Planted answers for ``handoff_price_4604.py`` (#4604): run it before trusting a pricing.

python selftest_handoff_4604.py
"""

from __future__ import annotations

import argparse
import importlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
price = importlib.import_module("handoff_price_4604")


def _fixture(d: Path) -> argparse.Namespace:
    """Two runs, 12 clicks each. Run A: the typed query's set scores 0.3; a detector from click 4 scores 0.1 at
    click 4 (3 Goods + 1 Bad) and 0.6 from click 5; the Bad phase ends at click 7; Hard at click 12. Run B never
    trains (no rows, 12 Bad clicks): it stays in, at its typed query's 0.2 under every rule (#4631)."""
    steps, picks = [], []
    phases = {4: "bad", 5: "bad", 6: "bad", 7: "more", 8: "more", 9: "more"}
    for t in range(4, 13):
        p, r = (0.1, 0.1) if t == 4 else (0.6, 0.6)
        steps.append({"category": "a@large", "seed": 0, "t": t, "phase": phases.get(t, "hard"),
                      "app_trained": int(t >= 12), "threshold": 0.5 + 0.01 * t, "precision": p, "recall": r,
                      "floor_count": 10, "n_good": 3, "n_bad": 1})  # fmt: skip
    # Picks: the detector's score at pick s is judged against the line of step s-1 (0.5 + 0.01 (s-1)).
    # Click 8: a positive scored 0.565 -> above step 7's 0.57? No (0.565 < 0.57): the detector misses it.
    # Click 9: a positive scored 0.585 -> above step 8's 0.58: kept. Text keeps both (score >= cut 0.0).
    # Click 11: a negative the typed query keeps (score 1.0) and the detector does not.
    for s in range(1, 13):
        lab = 1 if s in (1, 2, 3, 8, 9) else 0
        dscore = {8: 0.565, 9: 0.585}.get(s, np.nan if s < 5 else 0.0)
        picks.append({"category": "a@large", "seed": 0, "t": s, "phase": "x", "picked_label": lab,
                      "picked_seed_score": 1.0 if (lab or s == 11) else -1.0, "picked_detector_score": dscore})  # fmt: skip
        picks.append({"category": "b@large", "seed": 0, "t": s, "phase": "x", "picked_label": 0,
                      "picked_seed_score": -1.0, "picked_detector_score": np.nan})  # fmt: skip
    pd.DataFrame(steps).to_csv(d / "steps.csv", index=False)
    pd.DataFrame(picks).to_csv(d / "picks.csv", index=False)
    pd.DataFrame(
        [{"category": "a@large", "class": "a", "band": "large", "seed": 0, "never_trained": False, "shown_from": 12.0},
         {"category": "b@large", "class": "b", "band": "large", "seed": 0, "never_trained": True, "shown_from": np.inf}]
    ).to_csv(d / "cells.csv", index=False)  # fmt: skip
    pd.DataFrame(
        [{"embedder": "siglip", "category": "a@large", "seed": 0, "text_precision": 0.3, "text_recall": 0.3, "text_gmm_cut": 0.0},
         {"embedder": "siglip", "category": "b@large", "seed": 0, "text_precision": 0.2, "text_recall": 0.2, "text_gmm_cut": 0.0}]
    ).to_csv(d / "base.csv", index=False)  # fmt: skip
    return argparse.Namespace(steps=d / "steps.csv", picks=d / "picks.csv", cells=d / "cells.csv",
                              baseline=d / "base.csv", text_embedder="siglip", beta=1.0)  # fmt: skip


def main() -> int:
    assert price.fbeta(np.array([np.nan]), np.array([0.0]), 1.0)[0] == 0.0, "an empty set scores 0"
    assert np.isclose(price.fbeta(np.array([0.5]), np.array([1.0]), 1.0)[0], 2 / 3)
    assert np.isclose(price.counts_fbeta(np.array([1.0]), np.array([1.0]), np.array([0.0]), 1.0)[0], 2 / 3)
    with tempfile.TemporaryDirectory() as tmp:
        a = _fixture(Path(tmp))
        runs, tq, det, shown, kept, lab, tside, dside = price.load(a)
        assert list(runs["category"]) == ["a@large", "b@large"], "the never-trained run stays in (#4631)"
        assert np.isclose(tq[0], 0.3) and shown[0] == 12 and runs["more_from"][0] == 7
        assert np.isclose(tq[1], 0.2) and np.isinf(shown[1]) and np.isnan(det[1]).all(), "no detector, ever"
        assert np.isnan(det[0, 3]) and np.isclose(det[0, 4], 0.1) and np.isclose(det[0, 150], 0.6), "carried forward"
        assert dside[0, 8] == 0 and dside[0, 9] == 1, "a pick is judged against the PREVIOUS step's line"
        assert np.isnan(dside[0, 4]), "no detector before click 5's pick"
        # Without a cells.csv (#4583's arms) the runs come off the steps and picks, class, band and Hard off the steps.
        derived = price.load(argparse.Namespace(**{**vars(a), "cells": None}))[0]
        assert list(derived["category"]) == ["a@large", "b@large"], "the picks list the never-trained run"
        assert derived["class"][0] == "a" and derived["band"][0] == "large"
        assert derived["shown_from"][0] == 12 and np.isinf(derived["shown_from"][1])
        rs = price.rules(runs, tq, det, shown, kept, lab, tside, dside, 1.0)
        f = {k: v[2][0] for k, v in rs.items()}
        assert np.allclose(f["today"][:12], 0.3) and np.allclose(f["today"][12:], 0.6), "today hands off at Hard"
        assert np.isclose(f["detector_always"][4], 0.1) and np.isclose(f["detector_always"][5], 0.6)
        assert np.allclose(f["after_bad"][:7], 0.3) and np.allclose(f["after_bad"][7:], 0.6), "after the Bad phase"
        assert np.isclose(f["ceiling_opening"][4], 0.3) and np.isclose(f["ceiling_opening"][6], 0.6), "max per click"
        assert np.allclose(f["click_8"][:8], 0.3) and np.isclose(f["click_8"][8], 0.6)
        # On the honest votes: after click 9 the typed query keeps both positives (F1 1), the detector one (2/3);
        # after click 11 the typed query also keeps a negative (4/5). At margin -0.2 the detector qualifies at
        # click 11 (2/3 >= 0.6); at margin 0 never, so Hard (12) decides.
        for rule in ("votes_m-0.2_p1", "after_bad_votes_m-0.2_p1"):
            assert np.allclose(f[rule][:11], 0.3) and np.isclose(f[rule][11], 0.6), rule
        assert np.allclose(f["after_bad_votes_m+0.0_p1"][:12], 0.3) and np.isclose(
            f["after_bad_votes_m+0.0_p1"][12], 0.6
        )
        assert all(np.allclose(v[2][1], 0.2) for v in rs.values()), "the never-trained run: its typed query, always"
        row = price.summarize("today", "today", np.nan, rs["today"][2], rs["today"][2], runs)
        assert row["runs"] == 2 and np.isclose(row["at_150"], (0.6 + 0.2) / 2), "both runs in the mean"
    print("selftest_handoff_4604: all planted answers recovered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
