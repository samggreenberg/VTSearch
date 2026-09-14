#!/usr/bin/env python
"""Planted answers for ``analyze_3825.py``'s joins, rates and counts.

    python selftest_analyze_3825.py

Every check here is a number a *reader* of the report would take at face value,
built from frames whose answer is known by construction.  The failures being
guarded against are the ones that produce a well-formed wrong number rather than
a crash: a join that silently drops half the cases, a rate computed over the
wrong denominator, a "did not converge" count that quietly includes the folds
that never ran an anchored refit.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_3825 as AZ  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def _cut_row(cell, case, arm, thr, adm, n=1000, seconds=0.01, provenance="fold_anchored[2/2]", unconv=0):
    row = {
        "cell": cell,
        "dataset": "d",
        "embedder": "e",
        "category": "c",
        "seed": 0,
        "style": "whole_image",
        "kind": "fold",
        "case": case,
        "arm": arm,
        "n": n,
        "n_folds": 2,
        "n_anchored": 2,
        "n_unconverged": unconv,
        "provenance": provenance,
        "seconds": seconds,
    }
    for v in AZ.INCLUSIONS:
        row[f"threshold_i{v}"] = thr
        row[f"n_admitted_i{v}"] = adm
    return row


def _fit_row(cell, case, fold, arm, ll_a, ll_f, n_iter, converged, refit=0.01, init=0.002, provenance="anchored"):
    return {
        "cell": cell,
        "dataset": "d",
        "embedder": "e",
        "category": "c",
        "seed": 0,
        "style": "whole_image",
        "kind": "fold",
        "case": case,
        "fold": fold,
        "arm": arm,
        "n": 2000,
        "n_anchors": 10,
        "provenance": provenance,
        "n_iter": n_iter,
        "converged": converged,
        "init_seconds": init,
        "total_seconds": init + refit,
        "refit_seconds": refit,
        "loglik_anchored": ll_a,
        "loglik_free": ll_f,
        "w_lo": 0.5,
        "mu_lo": 0.2,
        "var_lo": 0.01,
        "w_hi": 0.5,
        "mu_hi": 0.8,
        "var_hi": 0.01,
        "midpoint": 0.5,
    }


def test_gate_rate_and_denominator() -> None:
    """4 of 10 cases move the admitted set; the rate is 40%, not 40% of something else."""
    rows = []
    for i in range(10):
        rows.append(_cut_row("c0", f"{i:04d}", "baseline", 0.5, 500))
        moved = i < 4
        rows.append(_cut_row("c0", f"{i:04d}", "cand", 0.51 if moved else 0.5, 520 if moved else 500))
    t = AZ.gate_table(pd.DataFrame(rows))
    r = t[t.arm == "cand"].iloc[0]
    check("gate: paired case count", r.cases == 10, f"got {r.cases}")
    check("gate: admitted-moved rate", abs(r.admitted_moved_pct - 40.0) < 1e-9, f"got {r.admitted_moved_pct}")
    check("gate: rate not reportable under MIN_CASES", not bool(r.rate_reportable))
    # 20 of 1000 medias on the four that moved.
    check(
        "gate: |d admitted| median over the CHANGED cases",
        r.d_adm_median_of_changed == 20,
        f"got {r.d_adm_median_of_changed}",
    )
    check("gate: max fraction of the haystack", abs(r.d_adm_frac_max - 0.02) < 1e-12, f"got {r.d_adm_frac_max}")


def test_gate_pairs_within_a_case_not_across() -> None:
    """Two cells with the same case name must not cross-join."""
    rows = []
    for cell, adm in (("c0", 500), ("c1", 700)):
        rows.append(_cut_row(cell, "0000", "baseline", 0.5, adm))
        rows.append(_cut_row(cell, "0000", "cand", 0.5, adm))  # each arm matches ITS OWN cell
    t = AZ.gate_table(pd.DataFrame(rows))
    r = t[t.arm == "cand"].iloc[0]
    check(
        "gate: no cross-cell join",
        r.cases == 2 and r.admitted_moved_pct == 0.0,
        f"cases={r.cases} moved={r.admitted_moved_pct}",
    )


def test_incumbent_counts_only_the_baseline_and_only_anchored() -> None:
    """The incumbent table is about the SHIPPED rule, on folds that actually anchored."""
    rows = [
        _fit_row("c0", "0000", 0, "baseline", 1.0, 1.0, 200, 0),
        _fit_row("c0", "0000", 1, "baseline", 1.0, 1.0, 50, 1),
        _fit_row("c0", "0001", 0, "baseline", 1.0, 1.0, 60, 1),
        _fit_row("c0", "0001", 1, "baseline", 1.0, 1.0, 70, 1),
        # a fold that fell back: not an anchored refit, must not dilute the rate
        _fit_row("c0", "0002", 0, "baseline", 1.0, 1.0, 0, 0, provenance="inverted_means"),
        # another arm entirely: must not appear
        _fit_row("c0", "0000", 0, "cand", 1.0, 1.0, 5, 1),
    ]
    t = AZ.incumbent_table(pd.DataFrame(rows))
    allrow = t[t.dataset == "ALL"].iloc[0]
    check("incumbent: folds counted", allrow.folds == 4, f"got {allrow.folds}")
    check("incumbent: hit-cap share is 1 of 4", abs(allrow.hit_cap_pct - 25.0) < 1e-9, f"got {allrow.hit_cap_pct}")
    check("incumbent: median iterations", allrow.iter_median == 65, f"got {allrow.iter_median}")
    # refit 0.01 vs init 0.002 -> 0.01/0.012
    check(
        "incumbent: refit share",
        abs(allrow.refit_share_pct - 100 * 0.01 / 0.012) < 1e-6,
        f"got {allrow.refit_share_pct}",
    )


def test_fit_table_objectives_and_lost_fits() -> None:
    """Wins are counted on the anchored objective; a lost anchored fit is named."""
    rows = [
        _fit_row("c0", "0000", 0, "baseline", 1.000, 2.000, 100, 1),
        _fit_row("c0", "0001", 0, "baseline", 1.000, 2.000, 100, 1),
        _fit_row("c0", "0002", 0, "baseline", 1.000, 2.000, 100, 1),
        # better on the anchored objective, worse on the free one - the two must
        # not be read off each other
        _fit_row("c0", "0000", 0, "cand", 1.001, 1.999, 10, 1),
        _fit_row("c0", "0001", 0, "cand", 0.999, 2.001, 10, 1),
        _fit_row("c0", "0002", 0, "cand", 1.000, 2.000, 10, 1, provenance="component_collapse"),
    ]
    t = AZ.fit_table(pd.DataFrame(rows))
    r = t[t.arm == "cand"].iloc[0]
    check("fits: only mutually-anchored folds are compared", r.anchored_both == 2, f"got {r.anchored_both}")
    check("fits: an arm that lost an anchored fit is named", r.lost_anchored_fit == 1, f"got {r.lost_anchored_fit}")
    check("fits: anchored-objective wins", (r.obj_better, r.obj_worse) == (1, 1), f"got {(r.obj_better, r.obj_worse)}")
    check(
        "fits: free-objective wins are counted separately",
        (r.free_better, r.free_worse) == (1, 1),
        f"got {(r.free_better, r.free_worse)}",
    )
    # baseline refit 0.01 x2 vs cand 0.01 x2 over the mutually anchored folds...
    # all three rows are summed, so the speedup is 1.0 by construction here.
    check("fits: refit speedup is a ratio of sums", abs(r.refit_speedup - 1.0) < 1e-9, f"got {r.refit_speedup}")


def test_monotonicity_finds_a_planted_decrease() -> None:
    rows = []
    for k, v in enumerate([1.0, 1.5, 1.7, 1.8], 1):
        rows.append({"cell": "c0", "case": "0000", "fold": 0, "objective": "anchored", "iteration": k, "value": v})
    for k, v in enumerate([1.0, 1.5, 1.4, 1.8], 1):  # one real decrease
        rows.append({"cell": "c0", "case": "0000", "fold": 0, "objective": "free", "iteration": k, "value": v})
    t = AZ.monotonicity_table(pd.DataFrame(rows)).set_index("objective")
    check(
        "monotonicity: a clean trace has none",
        t.loc["anchored", "decreasing_iterations"] == 0,
        f"got {t.loc['anchored', 'decreasing_iterations']}",
    )
    check(
        "monotonicity: a planted decrease is found",
        t.loc["free", "decreasing_iterations"] == 1,
        f"got {t.loc['free', 'decreasing_iterations']}",
    )
    check(
        "monotonicity: the worst step is reported",
        abs(t.loc["free", "worst_single_step"] + 0.1) < 1e-9,
        f"got {t.loc['free', 'worst_single_step']}",
    )


def test_bench_buckets_measured_and_resampled_apart() -> None:
    rows = []
    for arm, secs in (("baseline", 0.02), ("cand", 0.005)):
        for n, resampled in ((2000, 0), (50000, 1)):
            rows.append(
                {
                    "sample": "s0",
                    "dataset": "d",
                    "embedder": "e",
                    "style": "whole_image",
                    "case": "0000",
                    "fold": 0,
                    "n": n,
                    "n_anchors": 10,
                    "resampled": resampled,
                    "arm": arm,
                    "init_seconds": 0.001,
                    "total_seconds": secs + 0.001,
                    "refit_seconds": secs,
                    "n_iter": 10,
                    "converged": 1,
                    "midpoint": 0.5,
                }
            )
    t = AZ.bench_table(pd.DataFrame(rows))
    cand = t[(t.arm == "cand")]
    check("bench: one row per (arm, size)", len(cand) == 2, f"got {len(cand)}")
    check(
        "bench: refit speedup", np.allclose(cand.refit_speedup_median, 4.0), f"got {cand.refit_speedup_median.tolist()}"
    )
    check("bench: the resampled row is flagged", set(cand.resampled) == {0, 1})


def test_main_writes_every_table() -> None:
    """The whole path, on a frame small but complete enough to run end to end."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        cuts, fits = [], []
        for i in range(40):
            for arm, thr, adm in (("baseline", 0.5, 500), ("cand", 0.52, 480)):
                cuts.append(_cut_row("c0", f"{i:04d}", arm, thr, adm))
                fits.append(_fit_row("c0", f"{i:04d}", 0, arm, 1.0, 2.0, 100 if arm == "baseline" else 9, 1))
        pd.DataFrame(cuts).to_csv(d / "gate3825_cuts.csv", index=False)
        pd.DataFrame(fits).to_csv(d / "gate3825_fits.csv", index=False)
        pd.DataFrame(
            [
                {
                    "cell": "c0",
                    "case": "0000",
                    "fold": 0,
                    "objective": "anchored",
                    "iteration": k,
                    "value": 1.0 + 0.1 * k,
                }
                for k in range(1, 5)
            ]
        ).to_csv(d / "gate3825_trace.csv", index=False)
        rc = AZ.main(["--analysis", str(d)])
        agg = d / "agg"
        want = {
            "gate_by_arm.csv",
            "gate_by_inclusion.csv",
            "gate_by_env.csv",
            "incumbent.csv",
            "fits_by_arm.csv",
            "monotonicity.csv",
            "TABLES.md",
            "summary_3825.json",
        }
        have = {p.name for p in agg.glob("*")}
        check("main: exit 0", rc == 0, f"got {rc}")
        check("main: every table written", want <= have, f"missing {sorted(want - have)}")


def main() -> int:
    for fn in (
        test_gate_rate_and_denominator,
        test_gate_pairs_within_a_case_not_across,
        test_incumbent_counts_only_the_baseline_and_only_anchored,
        test_fit_table_objectives_and_lost_fits,
        test_monotonicity_finds_a_planted_decrease,
        test_bench_buckets_measured_and_resampled_apart,
        test_main_writes_every_table,
    ):
        print(f"\n-- {fn.__name__}")
        fn()
    print(f"\n{len(FAILURES)} failure(s)" + (": " + ", ".join(FAILURES) if FAILURES else ""))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
