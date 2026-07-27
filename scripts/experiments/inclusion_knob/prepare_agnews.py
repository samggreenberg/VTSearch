"""Stage 0: download AG News and embed a subset with the production E5 embedder.

Caches ``CACHE/agnews_e5.npz`` with:

* ``X``   - ``(N, 768)`` float32 normalized E5 passage embeddings
* ``y``   - ``(N,)`` int8 category index into ``categories``
* ``categories`` - the 4 AG News category names

The embeddings are produced exactly the way VTSearch's text pipeline does:
``SentenceTransformer(E5_MODEL_ID).encode(f"passage: {text}",
normalize_embeddings=True)`` (see ``vtscore/media/text/embedder_e5.py``), so
the geometry the MLP sees here matches what a real text dataset produces.

Usage::

    python prepare_agnews.py [--per-category 600]
"""

from __future__ import annotations

import argparse

import common

common.setup_env()

import numpy as np  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Embed an AG News subset with E5 (CPU).")
    parser.add_argument("--per-category", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    out = common.CACHE / "agnews_e5.npz"
    if out.exists():
        common.log(f"already cached: {out}")
        return 0

    from sentence_transformers import SentenceTransformer

    from vtscore.config import E5_MODEL_ID
    from vtscore.datasets.downloader import download_ag_news

    timings: dict[str, float] = {}
    with common.timed("download", timings):
        by_cat = download_ag_news(on_progress=lambda *a, **k: None)
    categories = sorted(by_cat.keys())
    common.log(f"AG News categories: {categories} (counts: {[len(by_cat[c]) for c in categories]})")

    # Deterministic subset: first N per category (the CSV order is fixed).
    texts: list[str] = []
    labels: list[int] = []
    for ci, cat in enumerate(categories):
        take = by_cat[cat][: args.per_category]
        texts.extend(take)
        labels.extend([ci] * len(take))

    with common.timed("load-model", timings):
        model = SentenceTransformer(E5_MODEL_ID)
    with common.timed("embed", timings):
        X = model.encode(
            [f"passage: {t}" for t in texts],
            batch_size=args.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int8)
    np.savez_compressed(out, X=X, y=y, categories=np.array(categories))
    common.log(f"wrote {out}: X={X.shape}, {len(categories)} categories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
