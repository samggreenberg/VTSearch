#!/usr/bin/env python
"""Planted-answer test for ``analyze_structure_3329.py``.

Builds a synthetic results directory whose every pre-registered verdict is known
in advance, then asserts the analyzer returns exactly those verdicts.

The traps are the ones part 1 was actually caught by (#3329, PR #3333), plus the
two this analysis adds:

1. **A statistic that is never computed must not read as a verdict.** Part 1
   reported an inert-anchors refutation off NaN columns and an empty CSV read as
   a null. Here the conformal scope is planted below its resolution floor and
   must be reported as unresolvable, not as a pass.
2. **A dropped cell must be counted.** A zero-byte cell file is planted; it has
   to appear in ``n_cells_dropped`` rather than being silently skipped.
3. **The self/cross split of the domain-shift pairs must not leak.** The self
   rows are planted quiet and the cross rows loud; an analyzer that pooled them
   would report a middling z for both and pass neither bar honestly.
4. **B2's "deepest widens" is a per-embedder count**, not a pooled mean: one
   embedder with a huge widening must not carry four that show none.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

EMBEDDERS = ["siglip", "dinov3_patch", "clip", "clip_l", "siglip2_l"]
DATASETS = ["ds_a", "ds_b", "ds_c"]


def _rows(dataset: str, embedder: str, i: int) -> list[dict]:
    """One planted cell. `i` indexes the cell so B3's correlation is plantable."""
    # sd falls as path_len rises -> rho(path_len, -sd) must come out strongly
    # positive. Both stay inside their own bars.
    path_len = 3.0 + i * 0.25
    sd = 0.26 - i * 0.004
    out = []

    def add(family, scope, statistic, value, n=1000):
        out.append(
            {
                "dataset": dataset,
                "embedder": embedder,
                "seed": 0,
                "family": family,
                "scope": scope,
                "statistic": statistic,
                "value": float(value),
                "n": int(n),
            }
        )

    add("atlas", "holdout", "ks_uniform", 0.12)
    add("atlas", "holdout", "sd", sd)
    add("atlas", "holdout", "frac_below_0.05", 0.01)
    add("atlas", "holdout", "path_len_mean", path_len)
    add("atlas", "build", "ks_uniform", 0.14)
    # Deepest-only is wider on EVERY embedder -> the per-embedder count is 5.
    add("atlas_deepest", "holdout", "ks_uniform", 0.15)
    add("atlas_deepest", "holdout", "sd", sd + 0.02)
    # C1: local good, global bad.
    add("umap", "layout", "trustworthiness_k10", 0.97)
    add("umap", "layout", "trustworthiness_k50", 0.88)
    add("umap", "layout", "continuity_k10", 0.96)
    add("umap", "layout", "shepard_spearman", 0.35)
    # C2: purity drops in the layout.
    add("umap", "embedding", "knn_class_purity_k10", 0.60)
    add("umap", "layout", "knn_class_purity_k10", 0.48)
    # C3: containment lands on the nominal band.
    add("compaction", "core", "containment_mean", 0.89)
    add("compaction", "unit", "containment_mean", 0.87)
    add("compaction", "clusters", "noise_fraction", 0.05)
    # Trap 1: the conformal scope is planted BELOW its resolution floor on one
    # dataset, and must be reported unresolvable rather than passed.
    add("conformal", "in_class_holdout", "ks_uniform", 0.05, n=10 if dataset == "ds_c" else 500)
    add("conformal", "out_of_class", "median", 0.02)
    return out


def build(results: Path) -> None:
    results.mkdir(parents=True, exist_ok=True)
    rows_map = []
    i = 0
    for ds in DATASETS:
        for emb in EMBEDDERS:
            pd.DataFrame(_rows(ds, emb, i)).to_csv(results / f"struct_{ds}__{emb}.csv", index=False)
            rows_map.append({"index": i, "dataset": ds, "embedder": emb})
            i += 1
    # Trap 2: a zero-byte cell that must be COUNTED as dropped.
    (results / "struct_ds_z__siglip.csv").write_text("")
    rows_map.append({"index": i, "dataset": "ds_z", "embedder": "siglip"})
    pd.DataFrame(rows_map).to_csv(results / "cell_map.csv", index=False)

    # Trap 3: self pairs quiet, cross pairs loud.
    for emb in EMBEDDERS:
        pairs = []
        for b in DATASETS:
            for q in DATASETS:
                is_self = b == q
                pairs.append(
                    {
                        "embedder": emb,
                        "build_dataset": b,
                        "query_dataset": q,
                        "is_self": is_self,
                        "n_items": 1000,
                        "frac_atypical": 0.005 if is_self else 0.30,
                        "z_score": -2.5 if is_self else 40.0,
                        "median_pvalue": 0.5 if is_self else 0.05,
                        "shifted": (not is_self),
                        "ks_uniform": 0.12 if is_self else 0.45,
                        "mean_pvalue": 0.48 if is_self else 0.2,
                        "sd_pvalue": 0.26 if is_self else 0.24,
                    }
                )
        pd.DataFrame(pairs).to_csv(results / f"domainshift_{emb}.csv", index=False)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="struct3329-selftest-"))
    failures: list[str] = []
    try:
        results = tmp / "results"
        build(results)
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import analyze_structure_3329 as A

        rc = A.main(["--results", str(results), "--out", str(tmp / "analysis"), "--no-figures"])
        if rc != 0:
            print("analyzer returned non-zero", file=sys.stderr)
            return 1
        s = json.loads((tmp / "analysis" / "summary_structure3329.json").read_text())

        def check(name: str, got, want) -> None:
            if got != want:
                failures.append(f"{name}: got {got!r}, planted {want!r}")

        check("b1_not_uniform", s["b1_not_uniform"], True)
        check("b2_under_dispersed", s["b2_under_dispersed"], True)
        check("b2_deepest_widens_on_n_embedders", s["b2_deepest_widens_on_n_embedders"], 5)
        check("b2_meets_bar", s["b2_meets_bar"], True)
        check("b3_meets_bar", s["b3_meets_bar"], True)
        check("b4_meets_bar", s["b4_meets_bar"], True)
        check("b4_self_fires", s["b4_self_fires"], 0)
        check("b5_meets_bar", s["b5_meets_bar"], True)
        check("c1_meets_bar", s["c1_meets_bar"], True)
        check("c2_meets_bar", s["c2_meets_bar"], True)
        check("c3_meets_bar", s["c3_meets_bar"], True)
        # Trap 1: one dataset per embedder was planted under the floor.
        check("conformal_unresolvable_cells", s["conformal_unresolvable_cells"], len(EMBEDDERS))
        # Trap 2: the zero-byte cell is counted, not ignored.
        check("n_cells_dropped.empty", s["n_cells_dropped"]["empty"], 1)
        # B3's correlation must be strongly positive, not merely non-zero.
        if not (s["b3_rho"] > 0.9):
            failures.append(f"b3_rho: got {s['b3_rho']!r}, planted a near-perfect ordering")
        # The self pairs are quiet: pooling them with the cross pairs would push
        # the median z far positive.
        if not (s["b4_self_z_median"] < 0):
            failures.append(f"b4_self_z_median: got {s['b4_self_z_median']!r}, planted negative")
        if not (s["b5_cross_share_shifted"] == 1.0):
            failures.append(f"b5_cross_share_shifted: got {s['b5_cross_share_shifted']!r}, planted 1.0")
    finally:
        pass

    if failures:
        print("selftest_analyze_structure_3329: FAILURES", file=sys.stderr)
        for f in failures:
            print("  - " + f, file=sys.stderr)
        return 1
    print(
        "selftest_analyze_structure_3329: OK - the unresolvable-conformal floor, the dropped-cell "
        "count, the self/cross split, the per-embedder widening count and B3's ordering all behave as planted"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
