"""Planted-answer self-test for ``viewer.py`` — the interactive report viewer.

The page is a *lossy* encoding of the run (int16 quantisation, a thinned click
grid for the per-seed lines) wrapped around exact arithmetic (pooling across
datasets and categories).  Both halves can be wrong in ways that look fine on
screen, so both are checked against values that are known by construction:

* the codec must round-trip a value to within its quantisation step, and must
  bring **NaN back as NaN** rather than as the neighbour it was filled with;
* pooling "all categories" must be **weighted by the cells that contributed**,
  not a mean of means — the two differ exactly when one category trained on
  fewer cells, which is the case the coverage strip exists to expose;
* click 0 must carry the **zero-click text sort**, for every arm, including on
  a cell that never trained a detector;
* every metric the frame emits must reach the payload, with the direction
  (``lower``) that :mod:`vtscore.eval.calibration_metrics` declares — a viewer
  that decided for itself would eventually attach "lower is better" to recall;
* the per-seed payload must stay inside its byte budget, and must **say** which
  click grid it landed on rather than thinning in silence;
* the page's opening ``view`` (the metric it opens on, the ones it hides) must
  come out the same whether it was built in or put on by a reskin, must survive
  a later plain reskin, and must refuse a choice the page could only honour by
  quietly opening somewhere else (#4576);
* a run inside a spot check round must **stay in the mean** between rounds, at
  its last scored value, with nothing carried before its first row or after
  its last, and a reskin must get the same carry from the per-seed lines
  (#4624);
* a click with **no trained detector** (before a run's first Good and Bad, or
  any click of a run that never got both) must be in the mean as the empty
  returned set the app gives there, a loss and not a gap, with the runs read
  off the text-sort baseline when there is no cell list, never a group or a
  seed the run never ran.

Run: ``python selftest_viewer.py``
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import viewer as V  # noqa: E402

ARMS = ["ctl", "alt"]
DSS = ["dsA", "dsB"]
EMBS = ["embA", "embB"]
#: ``rich`` trains on every seed; ``lean`` on one.  Pooling them weighted by
#: cells is the whole point of storing ``n`` beside the moments, and it is the
#: only thing that separates the right answer from the mean of means below.
CATS = {"rich": 8, "lean": 1}
N_SEED, T_MAX = 8, 40
TEXT_COST = 0.30
LEVEL = {"ctl": 0.20, "alt": 0.10}

#: The planted oracle cut.  Only the two rates and the split's class counts are
#: emitted, exactly as the harness emits them - precision / recall / F1 at that
#: cut have to be *reconstructed* by the builder, which is the point of the
#: check: a wrong reconstruction is invisible on screen (it draws a plausible
#: dotted line) and would misprice the calibration regret in every report.
ORACLE_COST, ORACLE_FPR, ORACLE_FNR = 0.05, 0.02, 0.10
N_TEST_POS, N_TEST_NEG = 100.0, 900.0
_TP = N_TEST_POS * (1.0 - ORACLE_FNR)
_FP = N_TEST_NEG * ORACLE_FPR
_FN = N_TEST_POS * ORACLE_FNR
ORACLE_PRECISION = _TP / (_TP + _FP)
ORACLE_F1 = 2.0 * _TP / (2.0 * _TP + _FP + _FN)

#: The planted supervised skyline (#3322), one row per cell, differing between
#: the two categories so a wrong pool shows up as a level rather than as noise.
SKY_COST = {"rich": 0.06, "lean": 0.16}

#: A spot check round (#4624): this run has no row at clicks 21..24, and its
#: click-20 row carries a precision of its own and an undefined f1, so the
#: carry can be told from a neighbour's value and from a fill.
GAP_RUN = ("ctl", "dsA", "embA", "rich", 0)
GAP_LO, GAP_HI = 21, 24
GAP_PRECISION = 0.55
#: A run that starts late and ends early: nothing is carried before its first
#: row or after its last.
SHORT_RUN = ("alt", "dsB", "embB", "rich", 7)
SHORT_LO, SHORT_HI = 10, 30


def _frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, cells, base, sky = [], [], [], []
    for arm in ARMS:
        for ds in DSS:
            for emb in EMBS:
                for cat, trained in CATS.items():
                    for seed in range(N_SEED):
                        cells.append({"arm": arm, "dataset": ds, "embedder": emb, "category": cat, "seed": seed})
                        # The skyline is vote-independent, so every attempted
                        # cell has one whether or not the loop ever trained.
                        sky.append(
                            {
                                "arm": arm,
                                "dataset": ds,
                                "embedder": emb,
                                "category": cat,
                                "seed": seed,
                                "t": 0,
                                "gmm_variant": "skyline_train_full",
                                "cost": SKY_COST[cat],
                                "precision": 0.9,
                                "recall": 0.85,
                                "f1": 0.87,
                                "average_precision": 0.95,
                            }
                        )
                        if seed >= trained:
                            continue  # never trained: no metric row at all
                        run = (arm, ds, emb, cat, seed)
                        for t in range(1, T_MAX + 1):
                            if run == GAP_RUN and GAP_LO <= t <= GAP_HI:
                                continue  # inside a spot check round: no row (#4624)
                            if run == SHORT_RUN and not (SHORT_LO <= t <= SHORT_HI):
                                continue  # not started yet, or already over
                            on_gap_edge = run == GAP_RUN and t == GAP_LO - 1
                            rows.append(
                                {
                                    "arm": arm,
                                    "dataset": ds,
                                    "embedder": emb,
                                    "category": cat,
                                    "seed": seed,
                                    "t": t,
                                    # Flat in t and in seed, so any pooling error
                                    # shows up as a level rather than as noise.
                                    "cost": LEVEL[arm] + (0.10 if cat == "lean" else 0.0),
                                    "precision": GAP_PRECISION if on_gap_edge else 0.7,
                                    "recall": 0.6,
                                    "f1": np.nan if on_gap_edge else 0.65,
                                    "average_precision": 0.8,
                                    # The oracle cut, as the harness emits it:
                                    # the cost and the two rates, never the
                                    # confusion-matrix metrics at that cut.
                                    "oracle_cost": ORACLE_COST,
                                    "oracle_fpr": ORACLE_FPR,
                                    "oracle_fnr": ORACLE_FNR,
                                    "n_test_pos": N_TEST_POS,
                                    "n_test_neg": N_TEST_NEG,
                                }
                            )
    for ds in DSS:
        for emb in EMBS:
            for cat in CATS:
                for seed in range(N_SEED):
                    base.append(
                        {
                            "dataset": ds,
                            "embedder": emb,
                            "category": cat,
                            "seed": seed,
                            "supports_text": 1,
                            "text_cost": TEXT_COST,
                            "text_precision": 0.4,
                            "text_recall": 0.3,
                            "text_f1": 0.34,
                            "text_AP": 0.5,
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(cells), pd.DataFrame(base), pd.DataFrame(sky)


def _payload(path: Path) -> dict:
    html = path.read_text(encoding="utf-8")
    m = re.search(r'type="application/json">(.*?)</script>', html, re.S)
    assert m, "no payload script tag"
    return json.loads(m.group(1))


def _decode(enc: dict) -> np.ndarray:
    """Mirror of the page's decoder, so the codec is checked end to end."""
    import base64
    import gzip

    shape = enc["shape"]
    n_t = shape[-1]
    rows = int(np.prod(shape[:-1]))
    deltas = np.frombuffer(gzip.decompress(base64.b64decode(enc["v"])), dtype=np.int16).reshape(rows, n_t)
    mask = np.unpackbits(np.frombuffer(gzip.decompress(base64.b64decode(enc["m"])), dtype=np.uint8))
    vals = np.cumsum(deltas.astype(np.int64), axis=1) / enc["scale"]
    valid = mask[: rows * n_t].reshape(rows, n_t).astype(bool)
    return np.where(valid, vals, np.nan).reshape(shape)


def _check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{(' - ' + detail) if detail and not ok else ''}")
    return ok


def main() -> int:  # noqa: C901
    tmp = Path(tempfile.mkdtemp(prefix="viewer-selftest-"))
    try:
        main_df, cells, base, sky = _frames()
        out = V.build_viewer(
            main_df,
            tmp / "viewer.html",
            arms=ARMS,
            denominator=cells,
            baseline=base,
            skyline=sky,
            runs_budget_mb=0.25,
        )
        P = _payload(out)

        ok = True
        print("planted-answer checks:")

        # --- the axes the page offers ---------------------------------------
        ok &= _check(
            "every dataset, embedder and category reaches the page",
            P["datasets"] == DSS and P["embedders"] == EMBS and set(P["categories"]) == set(CATS),
            f"{P['datasets']} {P['embedders']} {P['categories']}",
        )
        keys = [m["key"] for m in P["metrics"]]
        ok &= _check(
            "every emitted metric is offered, and only those",
            keys == ["cost", "precision", "recall", "fbeta_b025", "f1", "fbeta_b4", "average_precision"],
            str(keys),
        )
        # A viewer that decided direction for itself would eventually attach
        # "lower is better" to recall.
        lower = {m["key"]: m["lower"] for m in P["metrics"]}
        ok &= _check(
            "direction comes from the shared metric table",
            lower["cost"] and not lower["recall"] and not lower["f1"],
            str(lower),
        )
        ok &= _check(
            "groups are (dataset, embedder, category)",
            len(P["groups"]) == len(DSS) * len(EMBS) * len(CATS),
            str(len(P["groups"])),
        )

        # --- the codec -------------------------------------------------------
        mean = _decode(P["agg"]["mean"])
        n = _decode(P["agg"]["n"])
        cellsA = _decode(P["agg"]["cells"])
        gi = {tuple(g): i for i, g in enumerate(P["groups"])}
        ai = {a: i for i, a in enumerate(P["arms"])}
        mi = keys.index("cost")
        ti = P["t"].index(T_MAX)
        got = mean[gi[("dsA", "embA", "rich")], ai["ctl"], mi, ti]
        ok &= _check(
            "a value round-trips through quantise/delta/gzip/mask",
            abs(got - LEVEL["ctl"]) <= 1.0 / P["agg"]["mean"]["scale"],
            f"{got} vs {LEVEL['ctl']}",
        )
        # The gap-fill must not leak: a never-trained cell is NaN, not the value
        # its neighbour happened to carry.
        blank = mean[gi[("dsA", "embA", "rich")], ai["ctl"], mi, P["t"].index(0) + 0]
        ok &= _check(
            "click 0 is the text sort, on every arm",
            abs(blank - TEXT_COST) <= 2.0 / P["agg"]["mean"]["scale"],
            f"{blank} vs {TEXT_COST}",
        )

        # --- pooling ---------------------------------------------------------
        # `rich` trained 8 cells at 0.20, `lean` trained 1 at 0.30.  The
        # cell-weighted pool is (8*0.20 + 1*0.30)/9 = 0.2111; the mean of means
        # is 0.25.  Only the first is right, and only the payload's `n` makes it
        # computable in the page.
        g_rich, g_lean = gi[("dsA", "embA", "rich")], gi[("dsA", "embA", "lean")]
        n_rich = n[g_rich, ai["ctl"], mi, ti]
        n_lean = n[g_lean, ai["ctl"], mi, ti]
        pooled = (n_rich * mean[g_rich, ai["ctl"], mi, ti] + n_lean * mean[g_lean, ai["ctl"], mi, ti]) / (
            n_rich + n_lean
        )
        ok &= _check(
            "n is stored per (group, arm, metric, click), so pooling can be weighted",
            abs(n_rich - CATS["rich"]) < 0.5 and abs(n_lean - CATS["lean"]) < 0.5,
            f"rich {n_rich} lean {n_lean}",
        )
        ok &= _check(
            "the cell-weighted pool is the right answer, not the mean of means",
            abs(pooled - 0.21111) < 2e-3 and abs(pooled - 0.25) > 0.03,
            f"{pooled:.5f}",
        )

        # --- coverage denominator -------------------------------------------
        # `lean` attempted N_SEED cells and trained one: coverage must be 1/8,
        # which is only possible because the caller's cell list was the
        # denominator rather than the rows that happen to exist.
        ok &= _check(
            "cells attempted counts the ones that never trained",
            abs(cellsA[g_lean, ai["ctl"], 0] - N_SEED) < 0.5,
            str(cellsA[g_lean, ai["ctl"], 0]),
        )
        cov = n_lean / cellsA[g_lean, ai["ctl"], 0]
        ok &= _check(
            "...so a starving category reports coverage well below 1", abs(cov - 1.0 / N_SEED) < 1e-6, f"{cov:.3f}"
        )

        # --- the denominator with no cell list ------------------------------
        # The CLI has no cell list, only the text-sort baseline, which scores
        # every cell of the grid; read off the rows, the denominator counts the
        # survivors.  A baseline wider than the run must not read as
        # starvation, so a group the arm trained nothing in and a seed that
        # trained nowhere are both left out.
        wide = pd.concat(
            [base, base.assign(category="ghost"), base[base["category"] == "rich"].assign(seed=N_SEED + 5)],
            ignore_index=True,
        )
        own = V.build_viewer(main_df, tmp / "own.html", arms=ARMS, baseline=wide, runs_budget_mb=0.25)
        PO = _payload(own)
        ok &= _check(
            "with no cell list, the baseline counts the runs that never trained",
            PO["groups"] == P["groups"] and bool(np.array_equal(_decode(PO["agg"]["cells"]), cellsA)),
            f"lean {_decode(PO['agg']['cells'])[g_lean, ai['ctl'], 0]}",
        )
        ok &= _check(
            "...but not a group the run never trained in, nor a seed it never ran",
            "ghost" not in PO["categories"] and PO["n_cells"] == len(cells) == P["n_cells"],
            f"{PO['categories']} {PO['n_cells']}",
        )

        # --- a click with no trained detector -------------------------------
        # Before a run's first Good and Bad, and at every click of a run that
        # never got both, the harness writes no row; a user there has a
        # labelset the app cannot train, and Find returns nothing.  So the
        # click is the empty returned set, a loss in the mean, not a gap that
        # leaves the failing sessions out of it (owner, 2026-10-07).
        step = 2.0 / P["agg"]["mean"]["scale"]
        mi_ap = keys.index("average_precision")
        prev = N_TEST_POS / (N_TEST_POS + N_TEST_NEG)
        lean_p = (CATS["lean"] * 0.7 + (N_SEED - CATS["lean"]) * 0.0) / N_SEED
        lean_ap = (CATS["lean"] * 0.8 + (N_SEED - CATS["lean"]) * prev) / N_SEED
        a_c = ai["ctl"]
        ok &= _check(
            "a run that never trained is in the mean at every click",
            all(abs(n[g_lean, a_c, keys.index(k), ti] - N_SEED) < 0.5 for k in ("precision", "recall", "f1")),
            str([n[g_lean, a_c, keys.index(k), ti] for k in ("precision", "recall", "f1")]),
        )
        ok &= _check(
            "...as the empty set: precision, recall and F1 0",
            abs(mean[g_lean, a_c, keys.index("precision"), ti] - lean_p) <= step
            and abs(mean[g_lean, a_c, keys.index("recall"), ti] - CATS["lean"] * 0.6 / N_SEED) <= step
            and abs(mean[g_lean, a_c, keys.index("f1"), ti] - CATS["lean"] * 0.65 / N_SEED) <= step,
            f"{mean[g_lean, a_c, keys.index('precision'), ti]} vs {lean_p}",
        )
        ok &= _check(
            "...and AP at the test split's prevalence, the chance level of no ranking",
            abs(mean[g_lean, a_c, mi_ap, ti] - lean_ap) <= step,
            f"{mean[g_lean, a_c, mi_ap, ti]} vs {lean_ap}",
        )
        ok &= _check(
            "...but no cost is invented when the rows cannot pin the miss weight",
            abs(n_lean - CATS["lean"]) < 0.5,
            str(n_lean),
        )
        ok &= _check(
            "...the oracle is the same empty set: with no model there is no cut to move",
            P["agg"]["omean"] is not None
            and abs(
                _decode(P["agg"]["omean"])[g_lean, a_c, keys.index("recall"), ti]
                - CATS["lean"] * (1 - ORACLE_FNR) / N_SEED
            )
            <= step,
            str(_decode(P["agg"]["omean"])[g_lean, a_c, keys.index("recall"), ti]),
        )
        ok &= _check(
            "...click 0 stays the text sort",
            abs(mean[g_lean, a_c, mi, P["t"].index(0)] - TEXT_COST) <= step,
        )
        ok &= _check(
            "...and the page is told the clicks without a detector were scored",
            bool(P.get("empty_sets_scored"))
            and "P.empty_sets_scored" in re.sub(r'<script id="payload".*?</script>', "", out.read_text(), flags=re.S),
        )
        own_n = _decode(PO["agg"]["n"])
        ok &= _check(
            "with no cell list, the baseline's runs are the ones scored",
            abs(own_n[g_lean, a_c, keys.index("precision"), ti] - N_SEED) < 0.5,
            str(own_n[g_lean, a_c, keys.index("precision"), ti]),
        )
        bare = V.build_viewer(
            main_df,
            tmp / "bare.html",
            arms=ARMS,
            denominator=cells,
            baseline=base,
            runs_budget_mb=0.25,
            score_empty_sets=False,
        )
        ok &= _check(
            "a build that opts out leaves them out, and says nothing",
            abs(_decode(_payload(bare)["agg"]["n"])[g_lean, a_c, keys.index("precision"), ti] - CATS["lean"]) < 0.5
            and "empty_sets_scored" not in _payload(bare),
        )

        # A detector that trained and flags nothing returns the same empty
        # set: the harness leaves its precision undefined, which would drop the
        # run from the precision mean, so it counts as 0 (owner, 2026-10-07).
        # Its oracle cut flagging nothing (FPR 0, FNR 1) is the same case.
        nil_run = ("alt", "dsB", "embA", "rich", 3)
        nil = main_df.copy()
        at = (
            (nil["arm"] == nil_run[0])
            & (nil["dataset"] == nil_run[1])
            & (nil["embedder"] == nil_run[2])
            & (nil["category"] == nil_run[3])
            & (nil["seed"] == nil_run[4])
            & (nil["t"] == T_MAX)
        )
        nil.loc[at, ["precision", "recall", "f1", "oracle_fpr", "oracle_fnr"]] = [np.nan, 0.0, 0.0, 0.0, 1.0]
        PN = _payload(
            V.build_viewer(nil, tmp / "nil.html", arms=ARMS, denominator=cells, baseline=base, runs_budget_mb=0.25)
        )
        g_nil, a_nil, m_p = gi[nil_run[1:4]], ai[nil_run[0]], keys.index("precision")
        n_nil, mean_nil = _decode(PN["agg"]["n"]), _decode(PN["agg"]["mean"])
        ok &= _check(
            "a detector that flags nothing stays in the precision mean, at 0",
            abs(n_nil[g_nil, a_nil, m_p, ti] - CATS["rich"]) < 0.5
            and abs(mean_nil[g_nil, a_nil, m_p, ti] - (CATS["rich"] - 1) * 0.7 / CATS["rich"]) <= step,
            f"n {n_nil[g_nil, a_nil, m_p, ti]} mean {mean_nil[g_nil, a_nil, m_p, ti]}",
        )
        o_nil = _decode(PN["agg"]["omean"])[g_nil, a_nil, m_p, ti]
        ok &= _check(
            "...and so does an oracle cut that flags nothing",
            abs(_decode(PN["agg"]["on"])[g_nil, a_nil, m_p, ti] - CATS["rich"]) < 0.5
            and abs(o_nil - (CATS["rich"] - 1) * ORACLE_PRECISION / CATS["rich"]) <= step,
            str(o_nil),
        )
        PNo = _payload(
            V.build_viewer(
                nil, tmp / "nil-off.html", arms=ARMS, denominator=cells, baseline=base, runs_budget_mb=0.25,
                score_empty_sets=False,
            )
        )  # fmt: skip
        ok &= _check(
            "...which a build that opts out leaves undefined",
            abs(_decode(PNo["agg"]["n"])[g_nil, a_nil, m_p, ti] - (CATS["rich"] - 1)) < 0.5,
        )

        # --- a spot check's rounds (#4624) -----------------------------------
        # A run inside a prompted check is scored once per round, so between
        # rounds it has no row.  The viewer's mean used to skip it there, and a
        # check prompts where the labels separate weakly, so what it skipped was
        # the weak sessions: a survivor's mean that dipped at the end of every
        # review when the checks ran out of budget and the weak runs came back.
        step = 2.0 / P["agg"]["mean"]["scale"]
        g_gap, a_gap = gi[("dsA", "embA", "rich")], ai[GAP_RUN[0]]
        mi_p, mi_f = keys.index("precision"), keys.index("f1")
        t_in, t_next = P["t"].index(GAP_LO + 1), P["t"].index(GAP_HI + 1)
        ok &= _check(
            "a run inside a spot check round stays in the count between rounds",
            abs(n[g_gap, a_gap, mi_p, t_in] - CATS["rich"]) < 0.5,
            str(n[g_gap, a_gap, mi_p, t_in]),
        )
        want_p = ((CATS["rich"] - 1) * 0.7 + GAP_PRECISION) / CATS["rich"]
        ok &= _check(
            "...at its last scored value, not a neighbour's",
            abs(mean[g_gap, a_gap, mi_p, t_in] - want_p) <= step,
            f"{mean[g_gap, a_gap, mi_p, t_in]} vs {want_p}",
        )
        ok &= _check(
            "...and a metric undefined on that row stays undefined through the gap",
            abs(n[g_gap, a_gap, mi_f, t_in] - (CATS["rich"] - 1)) < 0.5
            and abs(mean[g_gap, a_gap, mi_f, t_in] - 0.65) <= step,
            f"n {n[g_gap, a_gap, mi_f, t_in]} mean {mean[g_gap, a_gap, mi_f, t_in]}",
        )
        ok &= _check(
            "the carry ends where the next round is scored",
            abs(mean[g_gap, a_gap, mi_p, t_next] - 0.7) <= step,
            str(mean[g_gap, a_gap, mi_p, t_next]),
        )
        g_short, a_short = gi[("dsB", "embB", "rich")], ai[SHORT_RUN[0]]
        n_short = [n[g_short, a_short, mi_p, P["t"].index(t)] for t in (SHORT_LO - 5, SHORT_LO + 10, SHORT_HI + 5)]
        p_early = mean[g_short, a_short, mi_p, P["t"].index(SHORT_LO - 5)]
        ok &= _check(
            "nothing is carried before a run's first row or after its last",
            abs(n_short[1] - CATS["rich"]) < 0.5 and abs(n_short[2] - (CATS["rich"] - 1)) < 0.5,
            str(n_short),
        )
        ok &= _check(
            "...before its first row it has no detector, so it is the empty set, not its first value",
            abs(n_short[0] - CATS["rich"]) < 0.5 and abs(p_early - (CATS["rich"] - 1) * 0.7 / CATS["rich"]) <= step,
            f"n {n_short[0]} precision {p_early}",
        )
        per_seed = _decode(P["runs"]["values"]) if P["runs"] else None
        r_gap = P["runs"]["index"].index([g_gap, a_gap, P["seeds"].index(GAP_RUN[4])]) if P["runs"] else -1
        ok &= _check(
            "the per-seed line is carried the same way",
            per_seed is not None
            and abs(per_seed[r_gap, mi_p, P["runs"]["t"].index(GAP_LO + 1)] - GAP_PRECISION) <= 1.0 / V.RUNS_SCALE
            and not np.isfinite(per_seed[r_gap, mi_f, P["runs"]["t"].index(GAP_LO + 1)]),
        )
        ok &= _check("the page is told the gaps were carried", bool(P.get("gaps_filled")))

        # A committed page whose results are gone gets the same carry from its
        # per-seed lines, which hold every run at every click.
        raw = V.build_viewer(
            main_df,
            tmp / "raw.html",
            arms=ARMS,
            denominator=cells,
            baseline=base,
            skyline=sky,
            runs_budget_mb=0.25,
            fill_gaps=False,
        )
        PR = _payload(raw)
        ok &= _check(
            "a build without the carry skips the run between rounds, and says nothing",
            abs(_decode(PR["agg"]["n"])[g_gap, a_gap, mi_p, t_in] - (CATS["rich"] - 1)) < 0.5
            and "gaps_filled" not in PR,
        )
        V.reskin(raw, fill_gaps=True)
        PF = _payload(raw)
        n_f, mean_f = _decode(PF["agg"]["n"]), _decode(PF["agg"]["mean"])
        ok &= _check(
            "--fill-gaps on a reskin carries them from the per-seed payload",
            abs(n_f[g_gap, a_gap, mi_p, t_in] - CATS["rich"]) < 0.5 and bool(PF.get("gaps_filled")),
            str(n_f[g_gap, a_gap, mi_p, t_in]),
        )
        ok &= _check(
            "...re-averaging to within the per-seed quantisation of a build that carried",
            bool(np.array_equal(np.isfinite(mean_f), np.isfinite(mean)))
            and float(np.nanmax(np.abs(mean_f - mean))) <= 1.0 / V.RUNS_SCALE + step,
            f"max |diff| {np.nanmax(np.abs(mean_f - mean))}",
        )
        ok &= _check(
            "...leaving click 0 and the oracle companion as built",
            bool(np.allclose(mean_f[..., P["t"].index(0)], mean[..., P["t"].index(0)], equal_nan=True))
            and PF["agg"]["omean"] == PR["agg"]["omean"],
        )
        ok &= _check("...with the marker where a build puts it", list(PF) == list(P), f"{list(PF)} vs {list(P)}")
        thin = V.build_viewer(
            main_df,
            tmp / "thin.html",
            arms=ARMS,
            denominator=cells,
            baseline=base,
            runs_budget_mb=0.0001,
            fill_gaps=False,
        )
        try:
            V.reskin(thin, fill_gaps=True)
            ok &= _check("a page whose per-seed lines were thinned is refused", False, "filled anyway")
        except SystemExit as exc:
            ok &= _check("a page whose per-seed lines were thinned is refused", "thinned" in str(exc), str(exc))

        # --- the oracle companion -------------------------------------------
        # Reconstructed, not emitted: the harness ships an (FPR, FNR) pair and
        # the split's class counts, and the builder turns that back into a full
        # confusion matrix.  Checked against numbers computed the long way here,
        # because a wrong reconstruction draws a perfectly plausible line.
        oracle_on = {m["key"]: m["oracle"] for m in P["metrics"]}
        ok &= _check(
            "every cut metric gets an oracle, and the ranking metric does not",
            all(oracle_on[k] for k in ("cost", "precision", "recall", "f1")) and not oracle_on["average_precision"],
            str(oracle_on),
        )
        ok &= _check(
            "...and the page is told which metrics those are",
            set(P["oracle_metrics"]) >= {"cost", "precision", "recall", "f1"}
            and "average_precision" not in P["oracle_metrics"],
            str(P["oracle_metrics"]),
        )
        omean = _decode(P["agg"]["omean"])
        on = _decode(P["agg"]["on"])
        g0, a0 = gi[("dsA", "embA", "rich")], ai["ctl"]
        step = 1.0 / P["agg"]["omean"]["scale"]
        want = {
            "cost": ORACLE_COST,
            "recall": 1.0 - ORACLE_FNR,
            "precision": ORACLE_PRECISION,
            "f1": ORACLE_F1,
        }
        for key, expect in want.items():
            got_o = omean[g0, a0, keys.index(key), ti]
            ok &= _check(
                f"oracle {key} is reconstructed exactly ({expect:.4f})",
                abs(got_o - expect) <= 2 * step,
                f"{got_o} vs {expect}",
            )
        ok &= _check(
            "the oracle carries its own n, so a pooled oracle is weighted too",
            abs(on[g0, a0, keys.index("cost"), ti] - CATS["rich"]) < 0.5,
            str(on[g0, a0, keys.index("cost"), ti]),
        )
        ok &= _check(
            "no oracle value is invented for the ranking metric",
            not np.isfinite(omean[g0, a0, keys.index("average_precision"), ti]),
        )
        ok &= _check(
            "the oracle line does not reach back to click 0 (there is no model there)",
            not np.isfinite(omean[g0, a0, keys.index("cost"), P["t"].index(0)]),
        )

        # --- the supervised skyline -----------------------------------------
        ok &= _check(
            "the skyline reaches the page, named by arm",
            P["skyline"] is not None and P["skyline"]["arm"] == "skyline_train_full",
            str(P["skyline"] and P["skyline"]["arm"]),
        )
        sky_v = _decode(P["skyline"]["mean"])
        sky_n = _decode(P["skyline"]["n"])
        mi_cost = keys.index("cost")
        got_s = sky_v[g0, a0, mi_cost, 0]
        ok &= _check(
            "a skyline value round-trips",
            abs(got_s - SKY_COST["rich"]) <= 2.0 / P["skyline"]["mean"]["scale"],
            f"{got_s} vs {SKY_COST['rich']}",
        )
        # Vote-independent: EVERY attempted cell has one, including the seven
        # `lean` seeds that never trained a detector - which is exactly why the
        # skyline can be a floor for a run that produced no curve at all.
        ok &= _check(
            "the skyline counts every attempted cell, trained or not",
            abs(sky_n[gi[("dsA", "embA", "lean")], a0, mi_cost, 0] - N_SEED) < 0.5,
            str(sky_n[gi[("dsA", "embA", "lean")], a0, mi_cost, 0]),
        )
        pooled_sky = (
            sky_n[g0, a0, mi_cost, 0] * sky_v[g0, a0, mi_cost, 0]
            + sky_n[gi[("dsA", "embA", "lean")], a0, mi_cost, 0] * sky_v[gi[("dsA", "embA", "lean")], a0, mi_cost, 0]
        ) / (sky_n[g0, a0, mi_cost, 0] + sky_n[gi[("dsA", "embA", "lean")], a0, mi_cost, 0])
        ok &= _check(
            "a pooled skyline is the cell-weighted mean of the two categories",
            abs(pooled_sky - (SKY_COST["rich"] + SKY_COST["lean"]) / 2) < 3e-3,
            f"{pooled_sky:.5f}",
        )

        # --- the per-seed payload -------------------------------------------
        ok &= _check("a per-seed payload is present", P["runs"] is not None)
        if P["runs"]:
            budget = 0.25 * 1024 * 1024
            size = len(P["runs"]["values"]["v"]) + len(P["runs"]["values"]["m"])
            ok &= _check("it fits the byte budget it was given", size <= budget, f"{size} > {budget}")
            ok &= _check(
                "and it says which click grid it landed on",
                "clicks" in P["runs_note"] and str(len(P["runs"]["t"])) in P["runs_note"],
                P["runs_note"],
            )
            ok &= _check(
                "the per-seed grid keeps click 0 and the horizon",
                P["runs"]["t"][0] == 0 and P["runs"]["t"][-1] == T_MAX,
                str(P["runs"]["t"][:3]),
            )
            # Every attempted cell has a text sort, so a run that never trained
            # is still in the index - as a lone click-0 point.
            want = len(P["groups"]) * len(ARMS) * N_SEED
            ok &= _check(
                "every attempted run is indexed, including the ones that never trained",
                len(P["runs"]["index"]) == want,
                f"{len(P['runs']['index'])} vs {want}",
            )

        # --- the page itself -------------------------------------------------
        html = out.read_text(encoding="utf-8")
        # Check the page SHELL, not the payload: base64 is arbitrary text and
        # will contain "cdn" or "src=" by chance, which made the naive version
        # of this check fail on a page that was perfectly self-contained.
        shell = re.sub(r'<script id="payload".*?</script>', "", html, flags=re.S)
        external = re.findall(r'(?:src|href)\s*=\s*["\'](?!#)([^"\']+)', shell)
        ok &= _check(
            "the page is self-contained: no src/href leaves the file",
            not external,
            str(external),
        )
        ok &= _check(
            "...and nothing fetches at runtime",
            "fetch(" not in shell and "XMLHttpRequest" not in shell and "import(" not in shell,
        )
        ok &= _check("the payload token was substituted exactly once", V.TOKEN not in html)
        ok &= _check("the page reports its own payload budget", bool(P.get("payload_kb")))

        # --- how the page was built (#3326) ----------------------------------
        # A page whose build arguments are nowhere costs the next rebuild a
        # guess, and one wrong guess (which results dir carried which chip)
        # inverts a study silently.  Recorded only when the caller knew them.
        stamped = V.build_viewer(
            main_df,
            tmp / "stamped.html",
            arms=ARMS,
            denominator=cells,
            baseline=base,
            runs_budget_mb=0.25,
            build={"results": "/somewhere", "arms": "d1=ctl,d2=alt"},
        )
        ok &= _check(
            "the build arguments reach the page when the caller knows them",
            _payload(stamped).get("build", {}).get("arms") == "d1=ctl,d2=alt",
            str(_payload(stamped).get("build")),
        )
        ok &= _check(
            "...and no empty `build` key when it does not",
            "build" not in P,
        )

        # --- a floor measured after the fact (#3326) -------------------------
        # `--skyline-results` reads the skyline from a SECOND results root.  It
        # is sound because the skyline is vote-independent: a later, cheaper
        # pass over the same cells measures the same quantity, where re-running
        # the loop to collect one would replace the curves a finished report was
        # read off.  What has to hold is that the rows come from that root and
        # from nowhere else -- a silent fallback to the curve root would show on
        # screen as "this study has no floor", which is indistinguishable from
        # the truth for a study that never measured one.
        curve_root, floor_root = tmp / "curves", tmp / "floor"
        for root, frame in ((curve_root, main_df), (floor_root, sky)):
            (root / "ctl" / "cells").mkdir(parents=True)
            frame[frame["arm"] == "ctl"].to_csv(root / "ctl" / "cells" / "task_0000.csv", index=False)
        ok &= _check(
            "the curve root alone carries no floor",
            V.load_skyline(curve_root, ["ctl"], ["ctl"]).empty,
        )
        picked = V.load_skyline(floor_root, ["ctl"], ["control"])
        ok &= _check(
            "a second results root supplies one",
            len(picked) == int((sky["arm"] == "ctl").sum()) and not picked.empty,
            f"{len(picked)} rows",
        )
        ok &= _check(
            "...carrying the PAGE's arm label, so it lands beside the right curve",
            set(picked["arm"]) == {"control"},
            str(set(picked["arm"])),
        )
        ok &= _check(
            "...and the skyline arm's own name, so `_skyline_arrays` can pick between arms",
            set(picked["gmm_variant"]) == {"skyline_train_full"},
            str(set(picked["gmm_variant"])),
        )

        # --- reskin ----------------------------------------------------------
        # A template improvement has to be pushable onto a committed report
        # whose results directory is long gone, and it must move the SHELL
        # without touching a byte of the numbers.
        V.reskin(out)
        again = out.read_text(encoding="utf-8")
        ok &= _check(
            "reskin rewrites the page and leaves the payload byte-identical",
            _payload(out) == P
            and re.search(r'type="application/json">(.*?)</script>', again, re.S).group(1)
            == re.search(r'type="application/json">(.*?)</script>', html, re.S).group(1),
        )
        ok &= _check("...and the reskinned page is still whole", again == html)

        # --- the opening view (#4576) ---------------------------------------
        # A report that retired a metric must not open its viewer on it.  The
        # choice lives in the payload, so it has to come out identical whether
        # the page was built with it or a reskin put it on afterwards (the
        # committed SotA pages were built before it existed), and a later plain
        # reskin -- the routine template push -- must not undo it.
        def blob(path: Path) -> str:
            return re.search(r'type="application/json">(.*?)</script>', path.read_text(encoding="utf-8"), re.S).group(1)

        retire = {"metric": "average_precision", "hide": ["cost"]}
        ok &= _check("a page built without a choice carries no `view`, so it reads as before", "view" not in P)
        viewed = V.build_viewer(
            main_df,
            tmp / "viewed.html",
            arms=ARMS,
            denominator=cells,
            baseline=base,
            skyline=sky,
            runs_budget_mb=0.25,
            default_metric="average_precision",
            hide_metrics=["cost"],
        )
        PV = _payload(viewed)
        ok &= _check("the build writes the view into the payload", PV.get("view") == retire, str(PV.get("view")))
        ok &= _check(
            "...and hiding is not dropping: the hidden metric's numbers are still carried",
            PV["metrics"] == P["metrics"] and PV["agg"]["mean"]["shape"] == P["agg"]["mean"]["shape"],
            str([m["key"] for m in PV["metrics"]]),
        )
        late = tmp / "late.html"
        shutil.copyfile(out, late)
        V.reskin(late, default_metric="average_precision", hide_metrics=["cost"])
        PL = _payload(late)
        ok &= _check("a reskin puts the same view on a page built without one", PL.get("view") == retire)
        ok &= _check(
            "...touching nothing else in the payload",
            blob(late) == blob(out)[:-1] + ',"view":' + json.dumps(retire, separators=(",", ":")) + "}",
        )
        ok &= _check(
            "...and lands where a build puts it, so the two pages agree key for key",
            list(PL) == list(PV),
            f"{list(PL)[-3:]} vs {list(PV)[-3:]}",
        )
        kept = blob(late)
        V.reskin(late)
        ok &= _check("a later plain reskin keeps the view, byte for byte", blob(late) == kept)
        V.reskin(late, default_metric="precision")
        ok &= _check(
            "a reskin that names one half leaves the other alone",
            _payload(late).get("view") == {"metric": "precision", "hide": ["cost"]},
            str(_payload(late).get("view")),
        )
        V.reskin(late, default_metric="", hide_metrics=[])
        ok &= _check(
            "clearing both halves drops the block and restores the original payload",
            blob(late) == blob(out),
        )
        ok &= _check(
            "the template reads the block the builder writes",
            "P.view" in re.sub(r'<script id="payload".*?</script>', "", html, flags=re.S),
        )
        ok &= _check(
            "...and says when the gaps were carried",
            "P.gaps_filled" in re.sub(r'<script id="payload".*?</script>', "", html, flags=re.S),
        )

        def refuses(label: str, **kw) -> bool:
            try:
                V.build_viewer(main_df, tmp / "refused.html", arms=ARMS, runs_budget_mb=0.25, **kw)
            except SystemExit as exc:
                return _check(f"refuses {label}", True, str(exc))
            return _check(f"refuses {label}", False, "built anyway")

        ok &= refuses("a misspelt metric to hide", hide_metrics=["cots"])
        ok &= refuses("opening on a metric the run never emitted", default_metric="auroc")
        ok &= refuses("opening on a metric it also hides", default_metric="cost", hide_metrics=["cost"])
        ok &= refuses(
            "hiding every metric the page carries",
            hide_metrics=["cost", "precision", "recall", "fbeta_b025", "f1", "fbeta_b4", "average_precision"],
        )
        ok &= _check(
            "...but hiding a known metric the run never emitted is allowed",
            V.opening_view(["cost", "precision"], hide=["auroc"]) == {"hide": ["auroc"]},
        )

        ok &= _beta_checks(tmp)

        print("\n" + ("SELFTEST PASSED" if ok else "SELFTEST FAILED"))
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


#: A review's two session sets (#4636), each with its own returned set at
#: every click, and the text sort's own line at each beta (#4603), all chosen
#: so that no two of the numbers checked below coincide.
SESSION = {0.25: (0.75, 0.35), 4.0: (0.30, 0.80)}
TEXT_LINE = {0.25: (0.80, 0.20, 0.01), 4.0: (0.20, 0.90, 0.40)}
TEXT_BLIND = (0.40, 0.30, 0.05)
B_SEEDS, B_T = 3, 10


def _fb(p: float, r: float, beta: float) -> float:
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def _beta_checks(tmp: Path) -> bool:  # noqa: C901
    """F1/4 and F4 on the menu, and a review's session sets as chips, each anchored at its own line (#4636)."""
    print("a review's session sets (#4636):")
    ok = True
    dirs = {}
    for beta, (p, r) in SESSION.items():
        rows = [
            {"dataset": "dsA", "embedder": "embA", "category": cat, "seed": seed, "t": t, "beta": beta,
             "precision": p, "recall": r, "f1": _fb(p, r, 1.0), "average_precision": 0.8}
            for cat in ("cat@large", "cat@small") for seed in range(B_SEEDS) for t in range(1, B_T + 1)
        ]  # fmt: skip
        d = tmp / f"sessions-{beta:g}"
        (d / "results" / "cells").mkdir(parents=True)
        pd.DataFrame(rows).to_csv(d / "results" / "cells" / "task_0000.csv", index=False)
        dirs[beta] = d
    line_cols = {
        f"text_line_{m}_{tag}": v
        for beta, tag in ((0.25, "b025"), (4.0, "b4"))
        for m, v in zip(("precision", "recall", "fpr"), TEXT_LINE[beta], strict=True)
    }
    blind = dict(zip(("text_precision", "text_recall", "text_fpr"), TEXT_BLIND, strict=True))
    base = pd.DataFrame(
        [
            {
                "dataset": "dsA",
                "embedder": "embA",
                "category": cat,
                "seed": seed,
                "supports_text": 1,
                **blind,
                **line_cols,
                "text_AP": 0.5,
                "text_fbeta_b025": 0.99,
                "text_fbeta_b4": 0.99,
            }
            for cat in ("cat@large", "cat@small")
            for seed in range(B_SEEDS)
        ]  # fmt: skip
    )
    base_csv = tmp / "text_baseline.csv"
    base.to_csv(base_csv, index=False)

    # --- naming the sets ------------------------------------------------------
    ok &= _check(
        "--beta-run pairs are sorted by beta",
        [b for b, _ in V.parse_beta_runs([f"4={dirs[4.0]}", f"0.25={dirs[0.25]}"])] == [0.25, 4.0],
    )
    for bad in ([f"1={dirs[0.25]}", f"1={dirs[4.0]}"], ["x=somewhere"], ["1"], ["-1=somewhere"]):
        try:
            V.parse_beta_runs(bad)
            ok &= _check(f"refuses --beta-run {bad}", False, "parsed anyway")
        except SystemExit as exc:
            ok &= _check(f"refuses --beta-run {bad}", True, str(exc))
    # A swapped pair is invisible on screen and inverts every comparison the
    # page exists for, so a set whose rows carry another beta is refused.
    try:
        V.load_beta_runs([(4.0, dirs[0.25])], skyline=False)
        ok &= _check("refuses a set named for a beta its rows were not drawn at", False, "loaded anyway")
    except SystemExit as exc:
        ok &= _check("refuses a set named for a beta its rows were not drawn at", "0.25" in str(exc), str(exc))
    ok &= _check(
        "a chip reads as the preset's fraction", [V.beta_label(b) for b in (0.25, 1.0, 4.0)] == ["β 1/4", "β 1", "β 4"]
    )

    # --- the page, through the CLI -------------------------------------------
    page = tmp / "betas.html"
    rc = V.main(
        ["--beta-run", f"4={dirs[4.0]}", "--beta-run", f"0.25={dirs[0.25]}", "--baseline", str(base_csv),
         "--out", str(page), "--runs-budget-mb", "0.25", "--no-skyline"]
    )  # fmt: skip
    P = _payload(page)
    ok &= _check("the CLI builds a page from the session sets", rc == 0)
    ok &= _check("one chip per set, in beta order", P["arms"] == ["β 1/4", "β 4"], str(P["arms"]))
    ok &= _check(
        "the arms control says it chooses the sessions' beta",
        P.get("arms_control", {}).get("title") == "Sessions' beta"
        and "P.arms_control" in re.sub(r'<script id="payload".*?</script>', "", page.read_text(), flags=re.S),
        str(P.get("arms_control")),
    )
    ok &= _check(
        "the build records which directory carried which beta",
        [s.split("=")[0] for s in P.get("build", {}).get("beta_runs", [])] == ["0.25", "4"],
        str(P.get("build")),
    )
    labels = {m["key"]: m["label"] for m in P["metrics"]}
    ok &= _check(
        "F1/4 and F4 are offered beside F1, higher is better",
        labels.get("fbeta_b025", "").startswith("F1/4")
        and labels.get("fbeta_b4", "").startswith("F4")
        and not any(m["lower"] for m in P["metrics"] if m["key"].startswith("fbeta")),
        str(labels),
    )

    # --- what each set returned, at each beta ---------------------------------
    keys = [m["key"] for m in P["metrics"]]
    mean = _decode(P["agg"]["mean"])
    gi = {tuple(g): i for i, g in enumerate(P["groups"])}
    g, ai = gi[("dsA", "embA", "cat@large")], {a: i for i, a in enumerate(P["arms"])}
    lo, hi = ai["β 1/4"], ai["β 4"]
    t_end, t0 = P["t"].index(B_T), P["t"].index(0)
    step = 2.0 / P["agg"]["mean"]["scale"]

    def at(arm: int, key: str, t: int) -> float:
        return float(mean[g, arm, keys.index(key), t])

    want_end = {
        (lo, "fbeta_b025"): _fb(*SESSION[0.25], 0.25),
        (lo, "fbeta_b4"): _fb(*SESSION[0.25], 4.0),
        (hi, "fbeta_b4"): _fb(*SESSION[4.0], 4.0),
        (hi, "fbeta"): _fb(*SESSION[4.0], 4.0),
        (lo, "fbeta"): _fb(*SESSION[0.25], 0.25),
    }
    ok &= _check(
        "each set's returned set is scored at every preset, and the objective at its own beta",
        all(abs(at(a, k, t_end) - v) <= step for (a, k), v in want_end.items()),
        str({(P["arms"][a], k): (round(at(a, k, t_end), 4), round(v, 4)) for (a, k), v in want_end.items()}),
    )

    # --- click 0: each set from the line the app shows at its beta -------------
    # Not one shared notch (the beta-blind cut), and never the top-K reading the
    # baseline also carries (0.99 here): one rule on both sides (#4474).
    want0 = {
        (lo, "precision"): TEXT_LINE[0.25][0],
        (hi, "precision"): TEXT_LINE[4.0][0],
        (hi, "recall"): TEXT_LINE[4.0][1],
        (lo, "fbeta"): _fb(*TEXT_LINE[0.25][:2], 0.25),
        (hi, "fbeta"): _fb(*TEXT_LINE[4.0][:2], 4.0),
        (lo, "fbeta_b4"): _fb(*TEXT_LINE[0.25][:2], 4.0),
        (hi, "f1"): _fb(*TEXT_LINE[4.0][:2], 1.0),
    }
    ok &= _check(
        "click 0 is the text sort's own line at each set's beta, every cut metric off that one set",
        all(abs(at(a, k, t0) - v) <= step for (a, k), v in want0.items()),
        str({(P["arms"][a], k): (round(at(a, k, t0), 4), round(v, 4)) for (a, k), v in want0.items()}),
    )
    ok &= _check(
        "...the ranking metric keeps its own column, the same on both",
        abs(at(lo, "average_precision", t0) - 0.5) <= step and abs(at(hi, "average_precision", t0) - 0.5) <= step,
    )
    runs = _decode(P["runs"]["values"])
    r_hi = P["runs"]["index"].index([g, hi, P["seeds"].index(0)])
    ok &= _check(
        "...and the per-seed line starts on the same notch",
        abs(runs[r_hi, keys.index("precision"), P["runs"]["t"].index(0)] - TEXT_LINE[4.0][0]) <= 1.0 / V.RUNS_SCALE,
    )

    # A baseline from before the per-beta line (#4603) anchors every set at the
    # line the app drew then: one notch, scored at each set's beta.
    old = base.drop(columns=list(line_cols))
    frame, _sky, arms = V.load_beta_runs(V.parse_beta_runs([f"0.25={dirs[0.25]}", f"4={dirs[4.0]}"]), skyline=False)
    PO = _payload(V.build_viewer(frame, tmp / "old.html", arms=arms, baseline=old, runs_budget_mb=0.25))
    mo = _decode(PO["agg"]["mean"])
    ko = [m["key"] for m in PO["metrics"]]
    ok &= _check(
        "an older baseline anchors both sets at its beta-blind cut, each scored at its own beta",
        abs(mo[g, lo, ko.index("precision"), t0] - TEXT_BLIND[0]) <= step
        and abs(mo[g, hi, ko.index("precision"), t0] - TEXT_BLIND[0]) <= step
        and abs(mo[g, hi, ko.index("fbeta"), t0] - _fb(*TEXT_BLIND[:2], 4.0)) <= step,
    )
    return ok


if __name__ == "__main__":
    raise SystemExit(main())
