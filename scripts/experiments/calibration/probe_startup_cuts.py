"""Where does an inclusion `k` actually put the cut on a real seed sort?  (#3267)

The Good Mining arms name a **position on the seed sort**, and two families name
it two ways: `q<frac>` names the rank position directly, `k<N>` names an
Inclusion and lets the sort's own fitted GMM decide where that lands.  Only the
`k` family could ship - the app has an Inclusion knob and no rank-position knob -
but how far a given `k` moves the pick is a property of the fitted mixture, not
of `k`.  On a steep sort the whole usable inclusion range can land inside a
couple of rank percent, and an arm grid that looks well spread in `k` is then
nearly a point in the space the picks actually live in.

That is a property of *these* sorts, so it is measurable before the run rather
than discovered in the analysis.  This probe builds each cell's real text seed
sort - the same `embed_text_query` path `run_cells.py` seeds from - and prints
where every candidate cut lands as a **rank percentile**, which is the space the
`hard` select samples in.

    python probe_startup_cuts.py            # every prepared cell
    python probe_startup_cuts.py --json OUT

Read the spread of the `k` columns against the `q` columns.  If `k-10` and `k0`
differ by less than the analyzer's separation floor, the `incl_k` arms have not
been tested and the run should say so rather than report them as a null.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402

import experiment_config as cfg  # noqa: E402

#: The inclusions the arms use, plus enough of the tail to show whether the
#: lever saturates or merely moves slowly.
K_GRID = (0, -1, -2, -4, -6, -10, -20, -50, -100)
#: The rank positions the `q` arms use, for scale.
Q_GRID = (0.02, 0.05, 0.10, 0.25, 0.35)


def _percentile_of(sorted_desc: np.ndarray, cut: float) -> float:
    """Rank percentile (0 = top of the sort) of the first score at or below *cut*."""
    if not np.isfinite(cut):
        return 0.0
    idx = int(np.searchsorted(-sorted_desc, -cut, side="left"))
    return 100.0 * idx / max(1, len(sorted_desc))


def probe_cell(ds: str, emb: str, cat: str, medias: dict) -> dict | None:
    from vtscore.eval.startup_schedule import StartupRound, round_cut  # noqa: PLC0415
    from vtscore.embedding.helpers import embed_text_query  # noqa: PLC0415
    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

    text = cfg.seed_query_text(ds, cat)
    if not text:
        return None
    qvec = embed_text_query(text, "image", enrich=cfg.SEED_ENRICH, embedder_name=emb)
    if qvec is None:
        return None

    def _unit(vec):
        v = np.asarray(vec, dtype=np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-12 else v

    ids = list(medias.keys())
    matrix = np.stack([_unit(media_embedding(medias[c])) for c in ids])
    scores = (matrix @ _unit(qvec)).astype(np.float64)
    order = np.argsort(-scores)
    sorted_desc = scores[order]
    is_pos = np.array(
        [1.0 if cat in (medias[c].get("categories") or [medias[c].get("category")]) else 0.0 for c in ids]
    )[order]

    row: dict = {
        "dataset": ds,
        "embedder": emb,
        "category": cat,
        "query": text,
        "n": len(ids),
        "n_positive": int(is_pos.sum()),
        "prevalence_pct": round(100.0 * float(is_pos.mean()), 2),
        # How good the text sort is at all: without this a flat `k` family is
        # unreadable, because a sort with no separation has no good mass for a
        # cut to sit above.
        "precision_at_20_pct": round(100.0 * float(is_pos[:20].mean()), 1),
        "positives_in_top_1pct": int(is_pos[: max(1, len(ids) // 100)].sum()),
        "cuts": {},
    }
    row["cuts"]["mid"] = round(_percentile_of(sorted_desc, round_cut(scores, StartupRound("clicks", 1, "mid"))), 3)
    for k in K_GRID:
        cut = round_cut(scores, StartupRound("clicks", 1, "rate", k=k))
        row["cuts"][f"k{k}"] = round(_percentile_of(sorted_desc, cut), 3)
    for q in Q_GRID:
        cut = round_cut(scores, StartupRound("clicks", 1, "quantile", q=q))
        row["cuts"][f"q{q}"] = round(_percentile_of(sorted_desc, cut), 3)
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Where a startup cut lands on the real seed sort (#3267).")
    ap.add_argument("--json", default=None, help="Write the rows here as JSON.")
    args = ap.parse_args(argv)

    from _cells_io import load_medias  # noqa: PLC0415
    from vtscore.datasets import loader as _loader  # noqa: PLC0415

    info = json.loads((common.RESULTS / "prepare_info.json").read_text())
    rows: list[dict] = []
    for ds, per_emb in info.get("datasets", {}).items():
        for emb, entry in per_emb.items():
            medias = load_medias(_loader.EMBEDDINGS_DIR / cfg.pickle_name(ds, emb))
            for cat in entry.get("selected_categories") or []:
                r = probe_cell(ds, emb, cat, medias)
                if r is not None:
                    rows.append(r)

    keys = ["mid", *(f"k{k}" for k in K_GRID), *(f"q{q}" for q in Q_GRID)]
    header = f"{'dataset':16s} {'category':12s} {'prev%':>6s} {'P@20':>5s} " + " ".join(f"{k:>8s}" for k in keys)
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['dataset']:16s} {r['category']:12s} {r['prevalence_pct']:6.2f} {r['precision_at_20_pct']:5.0f} "
            + " ".join(f"{r['cuts'][k]:8.3f}" for k in keys)
        )

    # The verdict this probe exists for, stated rather than left to the reader.
    span = [r["cuts"]["k0"] - r["cuts"]["k-10"] for r in rows]
    print(
        f"\nk0 -> k-10 moves the cut by {np.median(span):.2f} rank percent (median over "
        f"{len(rows)} cells; min {min(span):.2f}, max {max(span):.2f})."
    )
    full = [r["cuts"]["k0"] - r["cuts"]["k-100"] for r in rows]
    print(f"k0 -> k-100 (the whole usable range): {np.median(full):.2f} rank percent median.")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
