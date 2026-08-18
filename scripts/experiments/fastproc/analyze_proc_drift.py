"""Vector drift and retrieval-order stability across image-processor arms (#3146).

    python analyze_proc_drift.py                 # tables + CSVs + figures
    python analyze_proc_drift.py --no-figures    # tables only

The measurement machinery is #3143's — ``vector_drift``, ``ranking_stability``
and ``biggest_flips`` are imported from ``../precision/analyze_drift.py`` rather
than reimplemented, so the two studies' drift numbers are the same quantity
computed the same way and can be quoted against each other.  What is different
here is the orchestration, and it is different in two ways that matter:

**The floor is a reproduction, not a displacement.**  #3143 could not hold the
device fixed — its treatment was compute precision, so the cross-card arm was
the only floor available, and it turned out to span twelve orders of magnitude.
Here the node is pinned, so the floor arm (``tv_cpu_rep``) is the *reference arm
run twice*.  Anything it drifts by is pipeline nondeterminism and nothing else,
which makes it a real denominator rather than an upper bound.

**The published pile is an adjudicator, not an arm.**  The premise this study
had to overturn — that the shipped path is PIL — is settled empirically here
rather than by reading a class name: the reference arm is a torchvision rebuild,
so if it reproduces the shared pile's cell then the pile is torchvision-built
and every embedder in the tree has been on the fast path all along.  If instead
the ``pil_cpu`` arm is the one that reproduces it, the premise-check is wrong
and the whole study inverts.  That comparison is run first and reported first,
because nothing below it means anything until it resolves.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "calibration"))
sys.path.insert(0, str(HERE.parent / "precision"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import fastproc_config as fcfg  # noqa: E402

# The math, from the study that established it.  Imported rather than copied:
# a second implementation of "1 - cos" is a second thing to keep in agreement,
# and the whole point of quoting #3143's 2.9e-6 next to these numbers is that
# both were produced by this function.
from analyze_drift import (  # noqa: E402
    biggest_flips,
    pm,
    ranking_stability,
    sig2,
    vector_drift,
)

TOPKS = (10, 50, 100)

#: Synthetic arm name for the cell already in the shared pile.
PUBLISHED = "published_pile"


def log(msg: str) -> None:
    print(msg, flush=True)


def cell_for(arm: str, embedder: str) -> Path:
    return fcfg.shared_cell(embedder) if arm == PUBLISHED else fcfg.arm_cell(arm, embedder)


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
    path = fcfg.arm_pile(arm) / "datadir" / "embeddings" / f"{fcfg.DATASET}__{embedder}__textq.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        cats = [str(c) for c in z["categories"]]
        vecs = np.asarray(z["vectors"], dtype=np.float64)
    vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    return cats, vecs


def adjudicate(embedders: list[str]) -> pd.DataFrame:
    """Which arm reproduces the published pile cell?  Run before anything else.

    This is the premise-check with a measurement behind it.  A class name says
    what transformers *thinks* it loaded; only a bit-level agreement with the
    cell that is actually in the shared pile says what built the pile.
    """
    rows = []
    for emb in embedders:
        pub = cell_for(PUBLISHED, emb)
        if not pub.exists():
            log(f"  {emb}: no published cell at {pub} — cannot adjudicate")
            continue
        pub_ids, pub_mat, _ = load_arm(PUBLISHED, emb)
        for arm in fcfg.ARMS:
            if not fcfg.arm_cell(arm, emb).exists():
                continue
            ids, mat, _ = load_arm(arm, emb)
            if ids != pub_ids:
                rows.append(
                    {
                        "embedder": emb,
                        "arm": arm,
                        "n": len(ids),
                        "n_published": len(pub_ids),
                        "median": float("nan"),
                        "note": "id set differs from the published cell",
                    }
                )
                continue
            d = vector_drift(pub_mat, mat)
            rows.append(
                {
                    "embedder": emb,
                    "arm": arm,
                    "n": len(d),
                    "n_published": len(pub_ids),
                    "median": float(np.median(d)),
                    "max": float(d.max()),
                    "frac_identical": float((d == 0).mean()),
                    "note": "",
                }
            )
    return pd.DataFrame(rows)


def analyse(embedders: list[str], arms: list[str], ref_arm: str) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    drift_rows: list[dict] = []
    rank_rows: list[dict] = []
    examples: list[dict] = []
    skipped: list[str] = []

    for emb in embedders:
        if not fcfg.arm_cell(ref_arm, emb).exists():
            skipped.append(f"{ref_arm} x {emb} (reference cell missing)")
            continue
        ref_ids, ref_mat, ref_medias = load_arm(ref_arm, emb)
        ref_text = load_text_queries(ref_arm, emb)

        for arm in arms:
            if arm == ref_arm:
                continue
            if not fcfg.arm_cell(arm, emb).exists():
                skipped.append(f"{arm} x {emb} (cell missing)")
                continue
            arm_ids, arm_mat, _ = load_arm(arm, emb)
            if arm_ids != ref_ids:
                skipped.append(f"{arm} x {emb} (id set differs: {len(arm_ids)} vs {len(ref_ids)})")
                continue

            d = vector_drift(ref_mat, arm_mat)
            drift_rows.append(
                {
                    "embedder": emb,
                    "arm": arm,
                    "backend": fcfg.ARMS[arm]["backend"],
                    "proc_device": fcfg.ARMS[arm]["device"],
                    "n": len(d),
                    "median": float(np.median(d)),
                    "mean": float(d.mean()),
                    "se": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0,
                    "p95": float(np.quantile(d, 0.95)),
                    "max": float(d.max()),
                    "frac_identical": float((d == 0).mean()),
                    "frac_above_1e6": float((d > 1e-6).mean()),
                }
            )

            # --- ranking, from the text query and from an exemplar -----------
            sources: list[tuple[str, list[str], np.ndarray, np.ndarray]] = []
            arm_text = load_text_queries(arm, emb)
            if ref_text and arm_text and ref_text[0] == arm_text[0]:
                # The text tower is untouched by an image processor, so these
                # should be identical.  Checked rather than assumed: if they are
                # not, the ranking change is not attributable to the gallery.
                tq_delta = float(np.abs(ref_text[1] - arm_text[1]).max())
                drift_rows[-1]["text_query_max_delta"] = tq_delta
                sources.append(("text_query", ref_text[0], ref_text[1], arm_text[1]))
            cats = (
                ref_text[0]
                if ref_text
                else sorted({m.get("category") for m in ref_medias.values() if m.get("category")})
            )
            ex_idx: dict[str, int] = {}
            for i, mid in enumerate(ref_ids):
                cat = ref_medias[mid].get("category")
                if cat and cat not in ex_idx:
                    ex_idx[cat] = i
            ex_cats = [c for c in cats if c in ex_idx]
            if ex_cats:
                rows_i = [ex_idx[c] for c in ex_cats]
                sources.append(("exemplar", ex_cats, ref_mat[rows_i], arm_mat[rows_i]))

            for source, cat_names, ref_q, arm_q in sources:
                for j, cat in enumerate(cat_names):
                    stats = ranking_stability(ref_mat @ ref_q[j], arm_mat @ arm_q[j])
                    rank_rows.append(
                        {
                            "embedder": emb,
                            "arm": arm,
                            "backend": fcfg.ARMS[arm]["backend"],
                            "proc_device": fcfg.ARMS[arm]["device"],
                            "source": source,
                            "category": cat,
                            **stats,
                        }
                    )
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


def report(adj: pd.DataFrame, drift: pd.DataFrame, ranks: pd.DataFrame, examples: list[dict], ref_arm: str) -> None:
    log("\n=== 0. Which arm built the published pile? ===")
    log("Nothing below this table means anything until it resolves: the study exists")
    log("because the shipped path was misidentified from the source, and only agreement")
    log("with the cell actually in the shared pile settles which backend produced it.")
    if adj.empty:
        log("  no published cells available — UNADJUDICATED")
    else:
        log(f"{'embedder':12s} {'arm':14s} {'median 1-cos':>14s} {'max':>12s} {'identical':>11s}  note")
        for _, r in adj.iterrows():
            log(
                f"{r['embedder']:12s} {r['arm']:14s} {sig2(r['median']):>14s} "
                f"{sig2(r.get('max', float('nan'))):>12s} "
                f"{(r.get('frac_identical', float('nan')) * 100):10.0f}%  {r['note']}"
            )

    log(f"\n=== 1. Vector drift vs {ref_arm} (per media, 1 - cos) ===")
    if drift.empty:
        log("  no arms to compare")
        return
    log(
        f"{'embedder':12s} {'arm':14s} {'backend/dev':18s} "
        f"{'median':>10s} {'p95':>10s} {'max':>10s} {'mean ± SE':>22s} {'>1e-6':>8s}"
    )
    for _, r in drift.iterrows():
        log(
            f"{r['embedder']:12s} {r['arm']:14s} {r['backend'] + '/' + r['proc_device']:18s} "
            f"{sig2(r['median']):>10s} {sig2(r['p95']):>10s} {sig2(r['max']):>10s} "
            f"{pm(r['mean'], r['se']):>22s} {r['frac_above_1e6'] * 100:7.0f}%"
        )

    log("\n=== 2. Drift relative to the reproduction floor ===")
    log("The floor arm is the SAME code on the SAME node, run twice, so its drift is")
    log("pipeline nondeterminism and nothing else.  A treatment number without this")
    log("denominator is unreadable: 1e-3 means one thing against a floor of 0 and")
    log("another against a floor of 1e-4.")
    for emb in sorted(drift["embedder"].unique()):
        sub = drift[drift["embedder"] == emb]
        floor = sub[sub["arm"].isin(fcfg.FLOOR_ARMS)]
        if floor.empty:
            log(f"  {emb}: NO REPRODUCTION ARM PRESENT — treatment drifts have no denominator")
            continue
        f_med = float(floor["median"].min())
        f_max = float(floor["max"].max())
        log(f"  {emb}: floor {sig2(f_med)} median, {sig2(f_max)} max ({', '.join(floor['arm'])})")
        for _, r in sub[~sub["arm"].isin(fcfg.FLOOR_ARMS)].iterrows():
            ratio = (r["median"] / f_med) if f_med > 0 else float("inf")
            shown = "exactly 0 — the pipeline is deterministic" if f_med == 0 else f"{sig2(ratio)} x floor"
            log(f"    {r['arm']:14s} {sig2(r['median']):>10s}  {shown}")

    tq = drift.get("text_query_max_delta")
    if tq is not None and tq.notna().any():
        worst = float(tq.max())
        log(
            f"\n  text-tower control: max |delta| between arms' query vectors = {sig2(worst)}"
            + ("  (identical, as expected — the image processor cannot reach it)" if worst == 0 else "  ** NON-ZERO **")
        )

    if ranks.empty:
        log("\nno ranking rows — the text-query and exemplar sources were both unavailable")
        return

    log("\n=== 3. Retrieval-order stability (per category, paired) ===")
    log(
        f"{'embedder':12s} {'arm':14s} {'source':12s} {'n':>4s} {'Spearman ρ':>22s} "
        + " ".join(f"{'top' + str(k):>10s}" for k in TOPKS)
        + f"{'top1 same':>11s}"
    )
    for (emb, arm, source), g in ranks.groupby(["embedder", "arm", "source"], sort=True):
        rho_se = g["spearman"].std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else 0.0
        overlaps = " ".join(f"{g[f'top{k}_overlap'].mean() * 100:9.0f}%" for k in TOPKS)
        log(
            f"{emb:12s} {arm:14s} {source:12s} {len(g):4d} {pm(g['spearman'].mean(), rho_se):>22s} "
            f"{overlaps} {g['top1_same'].mean() * 100:10.0f}%"
        )

    log("\n=== 4. Literal examples: the biggest rank moves ===")
    log("A large move across a tiny score gap is a tie broken differently, not a")
    log("retrieval failure.  The gap column is there so the two can be told apart.")
    for ex in examples:
        log(
            f"  {ex['embedder']:10s} {ex['arm']:12s} {ex['source']:11s} {str(ex['category'])[:14]:14s} "
            f"media {ex['media_id']:>8} ({str(ex['filename'])[:28]:28s}) "
            f"rank {ex['ref_rank']:>5} -> {ex['arm_rank']:>5} (moved {ex['rank_moved']:>5}), "
            f"score {sig2(ex['ref_score'])} -> {sig2(ex['arm_score'])}, "
            f"gap to next in ref {sig2(ex['gap_to_next_in_ref'])}"
        )


def figures(drift: pd.DataFrame, ranks: pd.DataFrame, outdir: Path, ref_arm: str) -> None:
    """Averaged *and* per-run: a mean hides which arm owns the tail."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)

    for emb in sorted(drift["embedder"].unique()):
        sub = drift[drift["embedder"] == emb].sort_values("median")
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 3.4))
        y = np.arange(len(sub))
        # A zero-median floor cannot be drawn on a log axis; substituting the
        # smallest positive drift keeps the bar visible and the caption says so.
        floor_pos = sub["median"][sub["median"] > 0]
        eps = float(floor_pos.min()) / 10 if len(floor_pos) else 1e-16
        vals = sub["median"].where(sub["median"] > 0, eps)
        ax.barh(y, vals, color=["#999" if a in fcfg.FLOOR_ARMS else "#2b6cb0" for a in sub["arm"]])
        ax.set_yticks(
            y, [f"{a}\n{b}/{d}" for a, b, d in zip(sub["arm"], sub["backend"], sub["proc_device"])], fontsize=7
        )
        for yi, (m, p) in enumerate(zip(sub["median"], sub["p95"])):
            if p > 0:
                ax.plot([max(m, eps), p], [yi, yi], color="#333", lw=1.2)
        ax.set_xscale("log")
        ax.set_xlabel(f"1 - cos vs {ref_arm}   (bar = median, line to p95)")
        ax.set_title(
            f"Vector drift — {emb} on {fcfg.DATASET}\ngrey = the SAME code on the SAME node, run twice (the floor)"
        )
        fig.tight_layout()
        fig.savefig(outdir / f"drift_{emb}.png", dpi=130)
        plt.close(fig)

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
            ax.set_ylabel("Spearman ρ vs the reference ordering")
            ax.set_title(f"Retrieval-order stability per category — {source}\n(dot = one category, bar = mean)")
            ax.axhline(1.0, color="#999", ls=":", lw=1)
            fig.tight_layout()
            fig.savefig(outdir / f"rank_stability_{source}.png", dpi=130)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        for (emb, arm, source), g in ranks.groupby(["embedder", "arm", "source"]):
            ax.plot(
                TOPKS, [g[f"top{k}_overlap"].mean() for k in TOPKS], marker="o", label=f"{emb} {arm} {source}", lw=1.2
            )
        ax.set_xlabel("k")
        ax.set_ylabel("mean top-k overlap with the reference")
        ax.set_title("How much of the head survives a processor change")
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(outdir / "topk_overlap.png", dpi=130)
        plt.close(fig)

    log(f"\nwrote figures to {outdir}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embedders", default=",".join(fcfg.EMBEDDERS))
    ap.add_argument("--arms", default=",".join(fcfg.ARMS))
    ap.add_argument("--reference", default=fcfg.REFERENCE_ARM)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--outdir", default=str(fcfg.results_dir()))
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    embedders = [e for e in args.embedders.split(",") if e]
    arms = [a for a in args.arms.split(",") if a]

    adj = adjudicate(embedders)
    drift, ranks, examples = analyse(embedders, arms, ref_arm=args.reference)
    report(adj, drift, ranks, examples, ref_arm=args.reference)

    adj.to_csv(outdir / "adjudication.csv", index=False)
    drift.to_csv(outdir / "drift.csv", index=False)
    ranks.to_csv(outdir / "rank_stability.csv", index=False)
    (outdir / "examples.json").write_text(json.dumps(examples, indent=2) + "\n")
    log(f"\nwrote {outdir}/adjudication.csv, drift.csv, rank_stability.csv, examples.json")

    if not args.no_figures and not drift.empty:
        figures(drift, ranks, outdir / "figures", args.reference)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
