"""Vector drift and retrieval-order stability across precision arms (issue #3143).

    python analyze_drift.py                 # tables + CSVs + figures
    python analyze_drift.py --no-figures    # tables only

Four questions, in the order they gate the decision:

1. **How far do the vectors move?**  Per-media ``1 - cos(fp32, arm)``, reported
   as a distribution (median / p95 / max), never as a mean alone — the mean of a
   long-tailed drift is the number that hides the tail.
2. **How far do they move *compared to nothing changing*?**  The ``fp32_v100``
   arm is the same math on a different card, so its drift is the irreducible
   floor.  Every treatment number is reported as a **ratio to that floor**: a
   drift of 1e-3 means something only against a floor of 1e-7.
3. **Does the ranking move?**  Cosine drift is not the user-visible quantity;
   the *order* is.  Spearman ρ over the full gallery plus top-k overlap at
   k ∈ {10, 50, 100}, for two ranking sources: the **text query** (cross-modal
   search) and the **exemplar vector** (the startup sort the benchmark actually
   runs).  Both towers shift under half precision, so both are measured.
4. **Which items actually flip, by name?**  Every claim here owes literal
   examples — the specific media, its rank in each arm, and its filename — so a
   reader can open the image and judge whether a flip is a real ordering change
   or two near-tied items swapping.

Everything is paired by media id, so the reported ±SE is the SE of a *paired*
difference and not the much larger between-media spread.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "calibration"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import precision_config as pcfg  # noqa: E402

TOPKS = (10, 50, 100)


def log(msg: str) -> None:
    print(msg, flush=True)


def sig2(x: float) -> str:
    """Two significant digits.  Four decimals are clutter that *invents*
    findings; the report standard is two (docs/experiments/overview-bench)."""
    if x == 0 or not np.isfinite(x):
        return f"{x:.0f}"
    if abs(x) >= 0.01:
        return f"{x:.2g}"
    return f"{x:.1e}"


def pm(mean: float, se: float) -> str:
    """A difference is only ever quoted with its paired standard error."""
    return f"{sig2(mean)} ± {sig2(se)}"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


#: Synthetic arm name for the cell already in the shared pile.  It belongs in the
#: matrix because two of this study's own measurements disagreed about it: the
#: L40S fp32 rebuild reproduced it to 2.7e-12, yet the V100 fp32 arm sits 1.5e-4
#: from that same rebuild — and `sacct` says job 495266 built it on a **V100**.
#: All three cannot be true, and only a direct three-way comparison says which
#: measurement is wrong. Placing it as an arm is how it gets compared to both.
PUBLISHED = "published_pile"


def cell_for(arm: str, embedder: str) -> Path:
    return pcfg.shared_cell(embedder) if arm == PUBLISHED else pcfg.arm_cell(arm, embedder)


def load_arm(arm: str, embedder: str) -> tuple[list[int], np.ndarray, dict]:
    """``(ids, (N, D) unit-norm float64, medias)`` for one arm x embedder cell."""
    from _cells_io import load_medias  # noqa: PLC0415

    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

    medias = load_medias(cell_for(arm, embedder))
    ids, rows = [], []
    for mid in sorted(medias):
        vec = media_embedding(medias[mid])
        if vec is None:
            continue
        ids.append(mid)
        rows.append(np.asarray(vec, dtype=np.float64))
    mat = np.vstack(rows) if rows else np.zeros((0, 0))
    mat = mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12)
    return ids, mat, medias


def load_text_queries(arm: str, embedder: str) -> tuple[list[str], np.ndarray] | None:
    path = pcfg.arm_pile(arm) / "datadir" / "embeddings" / f"{pcfg.DATASET}__{embedder}__textq.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        cats = [str(c) for c in z["categories"]]
        vecs = np.asarray(z["vectors"], dtype=np.float64)
    vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    return cats, vecs


# --------------------------------------------------------------------------
# Drift
# --------------------------------------------------------------------------


def vector_drift(ref: np.ndarray, arm: np.ndarray) -> np.ndarray:
    """Per-row ``1 - cos``, the quantity every similarity is built out of."""
    cos = np.clip((ref * arm).sum(axis=1), -1.0, 1.0)
    return 1.0 - cos


def _rank_of(scores: np.ndarray) -> np.ndarray:
    """Dense 0-based rank, best score first."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(order))
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _rank_of(a).astype(np.float64), _rank_of(b).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def ranking_stability(ref_scores: np.ndarray, arm_scores: np.ndarray) -> dict:
    """Spearman ρ plus top-k overlap, and where the top item went."""
    out: dict = {"spearman": _spearman(ref_scores, arm_scores)}
    ref_order = np.argsort(-ref_scores, kind="stable")
    arm_order = np.argsort(-arm_scores, kind="stable")
    for k in TOPKS:
        kk = min(k, len(ref_scores))
        out[f"top{k}_overlap"] = len(set(ref_order[:kk]) & set(arm_order[:kk])) / kk
    out["top1_same"] = bool(ref_order[0] == arm_order[0])
    # Where the reference's #1 landed in the arm's ordering: the single most
    # legible statement of "did the answer change".
    arm_rank = _rank_of(arm_scores)
    out["ref_top1_rank_in_arm"] = int(arm_rank[ref_order[0]])
    out["max_score_delta"] = float(np.abs(ref_scores - arm_scores).max())
    return out


def biggest_flips(ref_scores, arm_scores, ids, medias, n=3) -> list[dict]:
    """The n items whose rank moved most, with names, so a reader can look.

    Reported alongside each item's *score gap to its neighbour*: a 40-place move
    among items separated by 1e-6 of cosine is a tie being broken differently,
    not a retrieval failure, and the two are indistinguishable from the rank
    alone.
    """
    ref_rank, arm_rank = _rank_of(ref_scores), _rank_of(arm_scores)
    moved = np.abs(ref_rank.astype(int) - arm_rank.astype(int))
    ref_sorted = np.sort(ref_scores)[::-1]
    out = []
    for idx in np.argsort(-moved)[:n]:
        r = int(ref_rank[idx])
        neighbour_gap = float(ref_sorted[r] - ref_sorted[min(r + 1, len(ref_sorted) - 1)])
        media = medias.get(ids[idx], {})
        out.append(
            {
                "media_id": int(ids[idx]),
                "filename": media.get("filename") or media.get("origin_name") or "?",
                "category": media.get("category"),
                "ref_rank": r,
                "arm_rank": int(arm_rank[idx]),
                "rank_moved": int(moved[idx]),
                "ref_score": float(ref_scores[idx]),
                "arm_score": float(arm_scores[idx]),
                "gap_to_next_in_ref": neighbour_gap,
            }
        )
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def analyse(embedders: list[str], arms: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    ref_arm = pcfg.REFERENCE_ARM
    drift_rows: list[dict] = []
    rank_rows: list[dict] = []
    examples: list[dict] = []
    skipped: list[str] = []

    for emb in embedders:
        if not pcfg.arm_cell(ref_arm, emb).exists():
            skipped.append(f"{ref_arm} x {emb} (reference cell missing)")
            continue
        ref_ids, ref_mat, ref_medias = load_arm(ref_arm, emb)
        ref_text = load_text_queries(ref_arm, emb)

        for arm in arms:
            if arm == ref_arm or not pcfg.arm_cell(arm, emb).exists():
                if arm != ref_arm:
                    skipped.append(f"{arm} x {emb} (cell missing)")
                continue
            arm_ids, arm_mat, _ = load_arm(arm, emb)
            if arm_ids != ref_ids:
                skipped.append(f"{arm} x {emb} (id set differs from reference: {len(arm_ids)} vs {len(ref_ids)})")
                continue

            d = vector_drift(ref_mat, arm_mat)
            drift_rows.append(
                {
                    "embedder": emb,
                    "arm": arm,
                    "precision": pcfg.ARMS[arm]["precision"],
                    "gpu": pcfg.ARMS[arm]["gpu"],
                    "n": len(d),
                    "median": float(np.median(d)),
                    "mean": float(d.mean()),
                    "se": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0,
                    "p95": float(np.quantile(d, 0.95)),
                    "max": float(d.max()),
                    "frac_above_1e6": float((d > 1e-6).mean()),
                }
            )

            # --- ranking, from the text query and from an exemplar -----------
            sources: list[tuple[str, list[str], np.ndarray, np.ndarray]] = []
            arm_text = load_text_queries(arm, emb)
            if ref_text and arm_text and ref_text[0] == arm_text[0]:
                sources.append(("text_query", ref_text[0], ref_text[1], arm_text[1]))
            # Exemplar: the whole-image vector of the first positive per
            # category, which is exactly what the benchmark seeds its sort with
            # (the pile drops pixels, so crops degrade to whole images).
            cats = (
                ref_text[0]
                if ref_text
                else sorted({m.get("category") for m in ref_medias.values() if m.get("category")})
            )
            ex_idx = {}
            for i, mid in enumerate(ref_ids):
                cat = ref_medias[mid].get("category")
                if cat and cat not in ex_idx:
                    ex_idx[cat] = i
            ex_cats = [c for c in cats if c in ex_idx]
            if ex_cats:
                rows = [ex_idx[c] for c in ex_cats]
                sources.append(("exemplar", ex_cats, ref_mat[rows], arm_mat[rows]))

            for source, cat_names, ref_q, arm_q in sources:
                for j, cat in enumerate(cat_names):
                    ref_scores = ref_mat @ ref_q[j]
                    arm_scores = arm_mat @ arm_q[j]
                    stats = ranking_stability(ref_scores, arm_scores)
                    rank_rows.append(
                        {
                            "embedder": emb,
                            "arm": arm,
                            "precision": pcfg.ARMS[arm]["precision"],
                            "source": source,
                            "category": cat,
                            **stats,
                        }
                    )
                # Literal examples from the worst category of this source.
                worst = min(
                    (r for r in rank_rows if r["arm"] == arm and r["embedder"] == emb and r["source"] == source),
                    key=lambda r: r["spearman"],
                )
                j = cat_names.index(worst["category"])
                for ex in biggest_flips(ref_mat @ ref_q[j], arm_mat @ arm_q[j], ref_ids, ref_medias):
                    examples.append(
                        {"embedder": emb, "arm": arm, "source": source, "category": worst["category"], **ex}
                    )

    # Count what was dropped rather than quietly analysing a subset.
    if skipped:
        log(f"\nSKIPPED {len(skipped)} arm x embedder combination(s):")
        for s in skipped:
            log(f"  - {s}")
    return pd.DataFrame(drift_rows), pd.DataFrame(rank_rows), examples


def pairwise(embedders: list[str], arms: list[str], outdir: Path) -> "pd.DataFrame":
    """Full arm x arm median drift, per embedder.

    Distance-to-the-reference is not enough once the reference itself is in
    question.  The first six arms showed ``fp32_v100`` sitting 1.5e-4 from
    ``fp32_l40s`` on ``siglip2_l`` — two arms that differ only in the card, both
    labelled fp32.  Which of the two is the outlier cannot be read off a column
    of distances to one of them; it needs the matrix, where a *cluster* is
    visible.  If TF32 is the cause, ``fp32_notf32_l40s`` sits with the V100 and
    apart from ``fp32_l40s``, and no reference-relative table would say so.
    """
    rows = []
    for emb in embedders:
        present = [a for a in arms if cell_for(a, emb).exists()]
        mats = {}
        for a in present:
            _, mat, _ = load_arm(a, emb)
            mats[a] = mat
        for i, a in enumerate(present):
            for b in present[i + 1 :]:
                if mats[a].shape != mats[b].shape:
                    continue
                d = vector_drift(mats[a], mats[b])
                rows.append(
                    {
                        "embedder": emb,
                        "arm_a": a,
                        "arm_b": b,
                        "median": float(np.median(d)),
                        "p95": float(np.quantile(d, 0.95)),
                        "max": float(d.max()),
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        log("no pairwise rows")
        return df

    for emb in sorted(df["embedder"].unique()):
        sub = df[df["embedder"] == emb]
        present = sorted(set(sub["arm_a"]) | set(sub["arm_b"]))
        log("")
        log(f"=== Pairwise median 1-cos — {emb} ===")
        log("      " + " ".join(f"{a[:14]:>14s}" for a in present))
        lookup = {}
        for _, r in sub.iterrows():
            lookup[(r["arm_a"], r["arm_b"])] = r["median"]
            lookup[(r["arm_b"], r["arm_a"])] = r["median"]
        for a in present:
            cells = []
            for b in present:
                cells.append("             -" if a == b else f"{sig2(lookup.get((a, b), float('nan'))):>14s}")
            log(f"{a[:20]:20s} " + " ".join(cells))
    df.to_csv(outdir / "pairwise_drift.csv", index=False)
    log(f"\nwrote {outdir}/pairwise_drift.csv")
    return df


def report(drift: pd.DataFrame, ranks: pd.DataFrame, examples: list[dict]) -> None:
    if drift.empty:
        log("no drift rows — nothing to report")
        return

    # --- the floor, first: every other number is read relative to it --------
    log("\n=== 1. Vector drift, 1 - cos vs %s (per media) ===" % pcfg.REFERENCE_ARM)
    log(
        f"{'embedder':12s} {'arm':16s} {'precision':14s} {'median':>10s} {'p95':>10s} {'max':>10s} {'mean ± SE':>22s} {'>1e-6':>8s}"
    )
    for _, r in drift.sort_values(["embedder", "arm"]).iterrows():
        log(
            f"{r['embedder']:12s} {r['arm']:16s} {r['precision']:14s} "
            f"{sig2(r['median']):>10s} {sig2(r['p95']):>10s} {sig2(r['max']):>10s} "
            f"{pm(r['mean'], r['se']):>22s} {r['frac_above_1e6'] * 100:7.0f}%"
        )

    log("\n=== 2. Drift relative to the cross-GPU floor ===")
    log("The floor arm is the same fp32 math on a different card, so it is the drift")
    log("that is not attributable to precision at all.")
    for emb in sorted(drift["embedder"].unique()):
        sub = drift[drift["embedder"] == emb]
        floor = sub[sub["arm"].isin(pcfg.FLOOR_ARMS)]
        if floor.empty:
            log(f"  {emb}: NO FLOOR ARM PRESENT — treatment drifts have no denominator")
            continue
        f_med = float(floor["median"].iloc[0])
        log(f"  {emb}: floor (median 1-cos) = {sig2(f_med)}")
        for _, r in sub[~sub["arm"].isin(pcfg.FLOOR_ARMS)].iterrows():
            ratio = r["median"] / f_med if f_med > 0 else float("inf")
            log(f"    {r['arm']:16s} {sig2(r['median']):>10s}  = {sig2(ratio):>9s} x floor")

    if ranks.empty:
        log("\nno ranking rows — the text-query and exemplar sources were both unavailable")
        return

    log("\n=== 3. Retrieval-order stability (per category, paired) ===")
    log(
        f"{'embedder':12s} {'arm':16s} {'source':12s} {'n':>4s} {'Spearman ρ':>22s} "
        + " ".join(f"{'top' + str(k):>10s}" for k in TOPKS)
        + f"{'top1 same':>11s}"
    )
    grouped = ranks.groupby(["embedder", "arm", "source"], sort=True)
    for (emb, arm, source), g in grouped:
        rho_se = g["spearman"].std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else 0.0
        overlaps = " ".join(f"{g[f'top{k}_overlap'].mean() * 100:9.0f}%" for k in TOPKS)
        log(
            f"{emb:12s} {arm:16s} {source:12s} {len(g):4d} {pm(g['spearman'].mean(), rho_se):>22s} "
            f"{overlaps} {g['top1_same'].mean() * 100:10.0f}%"
        )

    log("\n=== 4. Literal examples: the biggest rank moves ===")
    log("A large move across a tiny score gap is a tie broken differently, not a")
    log("retrieval failure.  The gap column is there so the two can be told apart.")
    for ex in examples:
        log(
            f"  {ex['embedder']:10s} {ex['arm']:14s} {ex['source']:11s} {str(ex['category'])[:14]:14s} "
            f"media {ex['media_id']:>8} ({str(ex['filename'])[:28]:28s}) "
            f"rank {ex['ref_rank']:>5} -> {ex['arm_rank']:>5} (moved {ex['rank_moved']:>5}), "
            f"score {sig2(ex['ref_score'])} -> {sig2(ex['arm_score'])}, "
            f"gap to next in ref {sig2(ex['gap_to_next_in_ref'])}"
        )


def figures(drift: pd.DataFrame, ranks: pd.DataFrame, outdir: Path) -> None:
    """Averaged *and* per-run: a mean drift curve hides which arm owns the tail."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)

    # (a) drift by arm, log scale, floor marked
    for emb in sorted(drift["embedder"].unique()):
        sub = drift[drift["embedder"] == emb].sort_values("median")
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 3.6))
        y = np.arange(len(sub))
        ax.barh(y, sub["median"], color=["#999" if a in pcfg.FLOOR_ARMS else "#2b6cb0" for a in sub["arm"]])
        ax.errorbar(
            sub["median"],
            y,
            xerr=[sub["median"] - sub["median"], sub["p95"] - sub["median"]],
            fmt="none",
            ecolor="#333",
            capsize=3,
        )
        ax.set_yticks(y, sub["arm"])
        ax.set_xscale("log")
        ax.set_xlabel(f"1 - cos vs {pcfg.REFERENCE_ARM}  (bar = median, whisker = p95)")
        ax.set_title(f"Vector drift — {emb} on {pcfg.DATASET}\ngrey = cross-GPU fp32 floor")
        fig.tight_layout()
        fig.savefig(outdir / f"drift_{emb}.png", dpi=150)
        plt.close(fig)

    # (b) per-category Spearman, every point drawn — the average alone would
    #     hide a single category that fell apart.
    if not ranks.empty:
        for source in sorted(ranks["source"].unique()):
            sub = ranks[ranks["source"] == source]
            fig, ax = plt.subplots(figsize=(8, 4))
            labels = []
            for i, ((emb, arm), g) in enumerate(sub.groupby(["embedder", "arm"])):
                jitter = (np.random.default_rng(i).random(len(g)) - 0.5) * 0.3
                ax.scatter(np.full(len(g), i) + jitter, g["spearman"], s=14, alpha=0.6)
                ax.plot([i - 0.25, i + 0.25], [g["spearman"].mean()] * 2, color="k", lw=2)
                labels.append(f"{emb}\n{arm}")
            ax.set_xticks(range(len(labels)), labels, fontsize=7)
            ax.set_ylabel("Spearman ρ vs fp32 ordering")
            ax.set_title(f"Retrieval-order stability per category — {source}\n(dot = one category, bar = mean)")
            ax.axhline(1.0, color="#999", ls=":", lw=1)
            fig.tight_layout()
            fig.savefig(outdir / f"rank_stability_{source}.png", dpi=150)
            plt.close(fig)

        # (c) top-k overlap vs k
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for (emb, arm, source), g in ranks.groupby(["embedder", "arm", "source"]):
            ax.plot(
                TOPKS, [g[f"top{k}_overlap"].mean() for k in TOPKS], marker="o", label=f"{emb} {arm} {source}", lw=1.2
            )
        ax.set_xlabel("k")
        ax.set_ylabel("mean top-k overlap with fp32")
        ax.set_title("How much of the head survives a precision change")
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(outdir / "topk_overlap.png", dpi=150)
        plt.close(fig)

    log(f"\nwrote figures to {outdir}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embedders", default=",".join(pcfg.EMBEDDERS))
    ap.add_argument("--arms", default=",".join(pcfg.ARMS))
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--pairwise", action="store_true", help="also print the full arm x arm drift matrix")
    ap.add_argument(
        "--include-published",
        action="store_true",
        help="add the shared pile's existing cell to the pairwise matrix as an arm",
    )
    ap.add_argument("--outdir", default=str(pcfg.results_dir()))
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    embedders = [e for e in args.embedders.split(",") if e]
    arms = [a for a in args.arms.split(",") if a]

    drift, ranks, examples = analyse(embedders, arms)
    report(drift, ranks, examples)
    if args.pairwise:
        pairwise(embedders, [*arms, PUBLISHED] if args.include_published else arms, outdir)

    drift.to_csv(outdir / "drift.csv", index=False)
    ranks.to_csv(outdir / "rank_stability.csv", index=False)
    (outdir / "examples.json").write_text(json.dumps(examples, indent=2) + "\n")
    log(f"\nwrote {outdir}/drift.csv, rank_stability.csv, examples.json")

    if not args.no_figures:
        figures(drift, ranks, outdir / "figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
