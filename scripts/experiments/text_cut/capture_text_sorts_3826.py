#!/usr/bin/env python
"""Capture real typed-query (text) sorts **with their labels** for #3826.

    python capture_text_sorts_3826.py --dataset coco_val --embedder siglip --out <corpus>/coco_val__siglip.npz

#3585's sort corpus (``gmm_init/capture_sorts_3585.py``) kept each sort's score
array and nothing else, which is enough to ask "does a fit move" and not enough
to ask "is the line in a good place".  #3826 needs the second question, so this
walks every category of one (dataset, text embedder) pair in the pile and keeps,
per typed query:

* ``scores`` - the whole cosine-sort score array, exactly the list the app hands
  :func:`~vtscore.training.thresholds.calculate_gmm_threshold`;
* ``labels`` - ground-truth membership of each media in the category, aligned
  to ``scores`` (``media_is_positive`` on the ``evaluable_pool``, the harness's
  one definition);
* the query text, and whether it was a curated query or the category's name.

The sort is scored the way ``capture_sorts_3585.py`` scores it - the harness's
``whole_image`` ``exemplar_sims`` with the text vector in place of a crop - and
the first category of every pair is also scored through the app's own
``cosine_sort_with_boxes`` and the two compared, so the capture carries its own
fidelity check rather than inheriting #3585's.

Every category with at least ``--min-positives`` positives is captured, not the
handful a prepare stage selected: a cut rule is a property of the *population*
of queries a user types, and a sample of six per dataset cannot show where it
breaks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "calibration"))

import common  # noqa: E402

common.setup_env()

import experiment_config as cfg  # noqa: E402


def _categories(medias: dict) -> "dict[str, int]":
    """Every category named anywhere in *medias*, with its raw positive count."""
    counts: dict[str, int] = {}
    for m in medias.values():
        cats = m.get("categories")
        for c in cats if cats is not None else [m.get("category")]:
            if c is not None:
                counts[c] = counts.get(c, 0) + 1
    return counts


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--embedder", required=True, help="a text-capable embedder with a pile pickle")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-positives", type=int, default=5)
    args = ap.parse_args(list(argv) if argv is not None else None)

    from vtscore.config import EMBEDDINGS_DIR
    from vtscore.embedding import embed_text_query
    from vtscore.eval.labels import evaluable_pool, media_is_positive
    from vtscore.eval.patch_styles import resolve_style
    from vtscore.training.region_similarity import cosine_sort_with_boxes

    from _cells_io import load_medias  # noqa: PLC0415

    ds, emb = args.dataset, args.embedder
    pkl = EMBEDDINGS_DIR / cfg.text_pickle_name(ds, emb)
    if not pkl.exists():
        print(f"no pickle {pkl}", file=sys.stderr)
        return 2
    if embed_text_query("a photo", "image", enrich=cfg.SEED_ENRICH, embedder_name=emb) is None:
        print(f"{emb} has no text tower", file=sys.stderr)
        return 2
    medias = load_medias(pkl)
    counts = _categories(medias)
    style = resolve_style("whole_image")
    store: dict[str, np.ndarray] = {}
    queries: dict[str, dict] = {}
    skipped: list[str] = []
    fidelity: dict[str, float] = {}
    common.log(f"=== {ds} x {emb}: {len(medias)} medias, {len(counts)} categories")

    for cat in sorted(counts):
        pool = evaluable_pool(medias, cat)
        ids = sorted(pool.keys())
        labels = np.asarray([media_is_positive(pool[i], cat) for i in ids], dtype=bool)
        n_pos = int(labels.sum())
        if n_pos < args.min_positives or n_pos == labels.size:
            skipped.append(f"{cat}: {n_pos} positives of {labels.size}")
            continue
        curated = cfg.seed_query_text(ds, cat)
        query = curated or cat.split("@")[0].replace("_", " ")
        tvec = embed_text_query(query, "image", enrich=cfg.SEED_ENRICH, embedder_name=emb)
        if tvec is None:
            skipped.append(f"{cat}: no vector for {query!r}")
            continue
        qv = np.asarray(tvec, dtype=np.float32)
        sims = style.exemplar_sims(pool, qv)
        scores = np.asarray([float(sims[i]) for i in ids], dtype=np.float64)
        if not fidelity:
            # The app's own scoring call, once per pair: the harness path above
            # is what every row is built from, so it had better be the list
            # the route hands the cut.
            _, app_sims = cosine_sort_with_boxes(pool, qv, emb, region_aware=False)
            a, b = np.sort(np.asarray(app_sims, dtype=np.float64)), np.sort(scores)
            fidelity = {"category": cat, "n": int(a.size), "max_abs_diff": float(np.max(np.abs(a - b)))}
        key = f"{ds}|{emb}|{cat.replace('|', '_')}"
        store[f"{key}|scores"] = scores
        store[f"{key}|labels"] = labels
        queries[key] = {
            "category": cat,
            "query": query,
            "curated": bool(curated),
            "n": int(scores.size),
            "n_pos": n_pos,
        }
        common.log(f"  {cat}: n={scores.size} pos={n_pos} q={query!r}")

    meta = {
        "dataset": ds,
        "embedder": emb,
        "pickle": str(pkl),
        "n_medias": len(medias),
        "min_positives": args.min_positives,
        "queries": queries,
        "skipped": skipped,
        "fidelity_vs_app": fidelity,
        "seed_enrich": cfg.SEED_ENRICH,
    }
    store["_meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **store)
    print(f"wrote {args.out}: {len(queries)} sorts, {len(skipped)} skipped; fidelity {fidelity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
