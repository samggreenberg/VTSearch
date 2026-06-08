#!/usr/bin/env python3
"""Compute exact demo-dataset media counts for ``vtscore/datasets/demo_counts.py``.

The dataset picker advertises a ``# Media`` figure *before* a demo dataset is
downloaded.  That figure used to be estimated from a single per-category
average, which is wrong for sources with uneven category sizes (see the module
docstring in :mod:`vtscore.datasets.demo_counts`).  This script measures the
true count so it can be written down and shown accurately.

How the count is obtained, per dataset id:

* If an embedded ``<id>.pkl`` is already cached in ``EMBEDDINGS_DIR``, the
  count is read straight from it — this is the exact number the loader
  produced.
* Otherwise the dataset's source is downloaded and run through the loader's
  *collection* phase (download → list → per-category slice) with embedding
  **stubbed out** (no model weights, no GPU).  The collection phase is what
  determines how many media end up in the dataset, so its length is the count.
  The throwaway pickle is written to a temp dir, never the real cache.

The two paths agree in practice; collection can only over-count relative to a
full embed if some source files fail to embed (corrupt/unsupported), which is
rare.  When in doubt, embed the dataset for real once and re-run with the
cached pkl present.

Usage::

    python scripts/compute_demo_counts.py caltech101_s caltech101_m
    python scripts/compute_demo_counts.py --source caltech101   # all S/M/L/A
    python scripts/compute_demo_counts.py --all                 # every demo

Output is ready-to-paste ``"id": count,`` lines for ``DEMO_MEDIA_COUNTS``.
Sources are downloaded one at a time; large ones (Food-101, Places365, the big
text corpora) take real time and disk, so prefer ``--source`` over ``--all``.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np

# Ensure the repo root is importable when run as a bare script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_FAKE_DIM = 8


def _fake_vector(*_args, **_kwargs) -> np.ndarray:
    return np.zeros(_FAKE_DIM, dtype=np.float32)


def _count_from_cache(dataset_id: str) -> int | None:
    """Return the media count from a cached embedded pkl, or ``None``."""
    from vtscore.config import EMBEDDINGS_DIR
    from vtscore.datasets.container import read_container

    pkl = EMBEDDINGS_DIR / f"{dataset_id}.pkl"
    if not pkl.exists():
        return None
    data, _meta = read_container(pkl)  # already-deserialized (data_dict, meta)
    return len(data["medias"])


def _stub_embedders(stack: ExitStack) -> None:
    """Patch every embedder/media type so loading needs no model weights."""
    from vtscore.media import all_embedders, all_types

    for mt in all_types():
        if hasattr(mt, "load_models"):
            stack.enter_context(patch.object(mt, "load_models"))
        if hasattr(mt, "embed_text"):
            stack.enter_context(patch.object(mt, "embed_text", side_effect=_fake_vector))
    # Scalar methods that return one vector for one item; stub them all so no
    # item is dropped on a None/raising embed regardless of which the loader
    # calls (images use embed_media/embed_pil_image, text uses embed_text_passage).
    scalar_embed_methods = (
        "embed_media",
        "embed_pil_image",
        "embed_text",
        "embed_text_passage",
        "embed_text_enriched",
    )
    for emb in all_embedders():
        stack.enter_context(patch.object(emb, "load_models"))
        for meth in scalar_embed_methods:
            if hasattr(emb, meth):
                stack.enter_context(patch.object(emb, meth, side_effect=_fake_vector))
        if hasattr(emb, "embed_media_bulk"):
            stack.enter_context(
                patch.object(emb, "embed_media_bulk", side_effect=lambda ms: [_fake_vector() for _ in ms])
            )


def _count_via_collection(dataset_id: str) -> int:
    """Download + run the collection phase with embedding stubbed; return count."""
    from vtscore.datasets.loader_demo import load_demo_dataset

    with ExitStack() as stack:
        tmp_dir = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        # The loader resolves EMBEDDINGS_DIR through the parent module at call
        # time; point it at a throwaway dir so we never read a stale real pkl
        # (forcing a fresh collection) nor write a stub-embedded one to the cache.
        stack.enter_context(patch("vtscore.datasets.loader.EMBEDDINGS_DIR", tmp_dir))
        _stub_embedders(stack)
        medias: dict[int, dict] = {}
        load_demo_dataset(dataset_id, medias, on_progress=lambda *a, **k: None)
        return len(medias)


def compute_count(dataset_id: str) -> tuple[int, str]:
    """Return ``(count, source)`` where source is ``"cache"`` or ``"collection"``."""
    cached = _count_from_cache(dataset_id)
    if cached is not None:
        return cached, "cache"
    return _count_via_collection(dataset_id), "collection"


def _resolve_ids(args: argparse.Namespace, all_ids: list[str]) -> list[str]:
    from vtscore.datasets.config import DEMO_DATASETS

    if args.all:
        return all_ids
    if args.source:
        ids = [i for i in all_ids if DEMO_DATASETS[i].get("source", "") == args.source]
        if not ids:
            # Fall back to id-prefix match (e.g. "caltech101" -> caltech101_*).
            ids = [i for i in all_ids if i.startswith(args.source)]
        return ids
    return list(args.ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ids", nargs="*", help="Demo dataset id(s) to measure, e.g. caltech101_s")
    parser.add_argument("--source", help="Measure every demo whose source (or id prefix) matches")
    parser.add_argument("--all", action="store_true", help="Measure every registered demo dataset")
    args = parser.parse_args()

    from vtscore.datasets.config import DEMO_DATASETS
    from vtscore.embedding import initialize_models

    initialize_models()  # register media types + embedders so stubbing can find them
    all_ids = list(DEMO_DATASETS)

    ids = _resolve_ids(args, all_ids)
    if not ids:
        parser.error("no dataset ids selected; pass ids, --source NAME, or --all")

    unknown = [i for i in ids if i not in DEMO_DATASETS]
    if unknown:
        parser.error(f"unknown dataset id(s): {', '.join(unknown)}")

    print(f"Measuring {len(ids)} demo dataset(s)...\n", file=sys.stderr)
    results: dict[str, int] = {}
    for did in ids:
        try:
            count, source = compute_count(did)
        except Exception as exc:  # noqa: BLE001 - report and continue across datasets
            print(f"  {did}: FAILED ({type(exc).__name__}: {exc})", file=sys.stderr)
            continue
        results[did] = count
        print(f"  {did}: {count} (from {source})", file=sys.stderr)

    print("\n# Paste into DEMO_MEDIA_COUNTS (keep sorted by id):")
    for did in sorted(results):
        print(f'    "{did}": {results[did]},')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
