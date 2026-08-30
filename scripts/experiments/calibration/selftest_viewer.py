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
  click grid it landed on rather than thinning in silence.

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
                        for t in range(1, T_MAX + 1):
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
                                    "precision": 0.7,
                                    "recall": 0.6,
                                    "f1": 0.65,
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
            keys == ["cost", "precision", "recall", "f1", "average_precision"],
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

        print("\n" + ("SELFTEST PASSED" if ok else "SELFTEST FAILED"))
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
