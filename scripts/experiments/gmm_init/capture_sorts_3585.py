#!/usr/bin/env python
"""Capture real **cosine/text sort** score distributions for the #3585 gate.

    python capture_sorts_3585.py --results <prepare dir> --out <corpus>/sorts.npz

The fold corpus (``capture_folds_3585.py``) covers the calibration path.  It
barely covers the *sort* path, which is where this issue's cost argument
actually lives: ``calculate_gmm_threshold`` runs on the full score distribution
of **every cosine/text sort**, and was measured at 91-95% of one.  A simulated
user types one query per cell, so a whole capture grid yields a handful of sort
haystacks; there is no reason to get them that way when the sort itself is a
text embed and a matmul.

So this walks the same (dataset, embedder, category) grid a prepare stage
selected and dumps the score array of each query's sort - the exact array the
app hands the cut - using the harness's own ``exemplar_sims`` path with the text
vector in place of a crop, which is what ``text_baseline.py`` does.  Sorting the
whole dataset rather than a split is deliberate: the app cuts the haystack it
can see.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402

common.setup_env()

import experiment_config as cfg  # noqa: E402


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, help="a results dir holding prepare_info.json")
    ap.add_argument("--out", required=True, help="npz to write")
    ap.add_argument(
        "--max-categories",
        type=int,
        default=0,
        help="cap the categories taken per (dataset, embedder); 0 means all. A cap takes the FIRST n "
        "in the prepare's own order, so two runs of this script capture the same sorts.",
    )
    ap.add_argument("--embedders", default="", help="comma-separated allowlist; default every embedder in the prepare")
    args = ap.parse_args(list(argv) if argv is not None else None)

    from vtscore.config import EMBEDDINGS_DIR
    from vtscore.embedding import embed_text_query
    from vtscore.eval.labels import evaluable_pool
    from vtscore.eval.patch_styles import resolve_style
    from vtscore.training.region_similarity import cosine_sort_with_boxes

    from _cells_io import load_medias  # noqa: PLC0415

    info = json.loads(Path(args.results, "prepare_info.json").read_text())
    allow = {e for e in args.embedders.split(",") if e}
    store: dict[str, np.ndarray] = {}
    # The rest of a sort, timed on this hardware.  Without it the saving on the
    # fit can only be turned into a user-visible latency by borrowing the
    # issue's own "91-95% of a sort" from a different machine.
    sort_seconds: dict[str, float] = {}
    style = resolve_style("whole_image")
    skipped: list[str] = []

    for ds, per_emb in sorted(info.get("datasets", {}).items()):
        for emb, entry in sorted(per_emb.items()):
            if allow and emb not in allow:
                continue
            cats = entry.get("selected_categories", [])
            if args.max_categories:
                cats = cats[: args.max_categories]
            text_emb = cfg.text_embedder(emb)
            pkl = EMBEDDINGS_DIR / cfg.text_pickle_name(ds, emb)
            if not pkl.exists():
                skipped.append(f"{ds}x{emb}: no pickle {pkl.name}")
                continue
            # Probe once, before loading a multi-GB pickle: a vision-only
            # embedder has no text sort to capture, and that is a fact about the
            # embedder rather than a failure.
            if embed_text_query("a photo", "image", enrich=cfg.SEED_ENRICH, embedder_name=text_emb) is None:
                skipped.append(f"{ds}x{emb}: {text_emb} is vision-only")
                continue
            medias = load_medias(pkl)
            common.log(f"=== {ds} x {emb} === {len(medias)} medias, {len(cats)} categories (space: {text_emb})")
            for cat in cats:
                query = cfg.seed_query_text(ds, cat) or cat
                tvec = embed_text_query(query, "image", enrich=cfg.SEED_ENRICH, embedder_name=text_emb)
                if tvec is None:
                    skipped.append(f"{ds}x{emb}x{cat}: no vector for {query!r}")
                    continue
                pool = evaluable_pool(medias, cat)
                sims = style.exemplar_sims(pool, np.asarray(tvec, dtype=np.float32))
                ids = sorted(pool.keys())
                scores = np.asarray([float(sims[i]) for i in ids], dtype=np.float64)
                key = f"sort|{ds}|{emb}|{cat.replace(' ', '_').replace('|', '_')}"
                store[f"{key}|scores"] = scores
                # `cosine_sort_active` is exactly this call plus the cut, so
                # this is the whole of the sort that is NOT the mixture fit:
                # the scoring pass, the result dicts and the sort itself.
                t0 = time.perf_counter()
                cosine_sort_with_boxes(pool, np.asarray(tvec, dtype=np.float32), text_emb, region_aware=False)
                sort_seconds[key] = time.perf_counter() - t0
                common.log(f"  {cat}: {scores.size} scores, {scores.min():.3f}..{scores.max():.3f}")

    if not store:
        print("captured no sorts", file=sys.stderr)
        for line in skipped:
            print(f"  skipped {line}", file=sys.stderr)
        return 1
    meta = {
        "skipped": skipped,
        "results": args.results,
        "max_categories": args.max_categories,
        "embedders": sorted(allow),
        "sort_seconds": sort_seconds,
    }
    store["_meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **store)
    print(f"wrote {args.out}: {len(store) - 1} sorts, {len(skipped)} skipped")
    for line in skipped:
        print(f"  skipped {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
