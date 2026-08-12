"""The shared pre-embedded pile: which ``(dataset, embedder)`` cells exist and where.

A *cell* is one ``<dataset>__<embedder>.pkl`` under the pile's
``embeddings/`` dir — the per-pair artifact every study loads instead of
re-embedding. Studies point ``VTSEARCH_DATA_DIR`` at the pile and read the
cells in place; nothing here is study-specific.

**Reproducibility.** The pile lives on scratch, which is treated as purgeable,
so every cell must be rebuildable from sources that are *not* on scratch:

* ``visual_genome_m`` / ``caltech101_m`` are VTSearch demo datasets, downloaded
  into the shared demo cache (``DEMO_CACHE``) and loaded by ``load_demo_dataset``.
* ``coco_val`` is not a demo dataset; it is assembled from the COCO-2017-val
  images and the flattened annotations staged under ``COCO_ROOT``.

Because ``_cells_io.dump_medias`` drops ``media_bytes``, a cell holds vectors
(plus ``patch_grid`` for patch embedders) and no pixels — so the pile is small
relative to its sources and a rebuild always re-reads the staged originals.

**Region voting.** Only patch embedders emit ``patch_grid``; a boxed dataset
paired with a single-vector embedder silently degrades to binary voting. That
mis-specification has burned three studies (#2877, #2897, #2905), so
:func:`region_capable` states it per *cell* rather than per dataset, and
``build_pile.py --verify`` asserts the geometry is actually present.
"""

from __future__ import annotations

import os
from pathlib import Path

USER = os.environ.get("USER", "sgreenberg")

#: Root of the shared pile. Everything below is derived from it.
PILE = Path(os.environ.get("VTS_PILE", f"/expscratch/{USER}/vts-cache"))
DATADIR = PILE / "datadir"
EMBEDDINGS = DATADIR / "embeddings"
MODELS = PILE / "models"

#: Shared, non-scratch sources the pile is rebuilt from.
DEMO_CACHE = Path(os.environ.get("VTS_DEMO_CACHE", "/exp/scale26/datasets/external/vtsearch-demos"))
COCO_ROOT = Path(os.environ.get("VTS_COCO_ROOT", "/exp/scale26/datasets/external/COCO"))
COCO_IMAGES = COCO_ROOT / "images" / "val2017"
COCO_ANNOTATIONS = COCO_ROOT / "derived" / "objects_flat_val2017.jsonl.gz"

#: Datasets in the pile. ``boxed`` means the medias carry ground-truth region
#: boxes, which is what a region-voting arm drags — necessary but not
#: sufficient (the embedder must also be patch-capable; see region_capable).
DATASETS: dict[str, dict] = {
    "visual_genome_m": {"boxed": True, "kind": "demo"},
    "caltech101_m": {"boxed": False, "kind": "demo"},
    "coco_val": {"boxed": True, "kind": "coco"},
}

#: Embedders in the pile. ``patch`` embedders attach ``patch_grid`` and are the
#: only ones that can carry a region-voting arm.
EMBEDDERS: dict[str, dict] = {
    "siglip": {"patch": False},
    "siglip2": {"patch": False},
    "siglip_l": {"patch": False},
    "siglip2_l": {"patch": False},
    "dinov3_patch": {"patch": True, "gated": True},
}


def cells() -> list[tuple[str, str]]:
    """Every ``(dataset, embedder)`` cell in the full grid."""
    return [(ds, emb) for ds in DATASETS for emb in EMBEDDERS]


def pickle_name(dataset: str, embedder: str) -> str:
    return f"{dataset}__{embedder}.pkl"


def cell_path(dataset: str, embedder: str) -> Path:
    return EMBEDDINGS / pickle_name(dataset, embedder)


def is_patch_embedder(embedder: str) -> bool:
    return bool(EMBEDDERS.get(embedder, {}).get("patch"))


def region_capable(dataset: str, embedder: str) -> bool:
    """True when this *cell* can actually region-vote.

    Both halves are required: ground-truth boxes to drag (dataset) and a patch
    grid to pool them over (embedder). Stated per cell precisely because the
    per-dataset flag alone reads as "this arm region-votes" and does not.
    """
    return bool(DATASETS.get(dataset, {}).get("boxed")) and is_patch_embedder(embedder)


def setup_env() -> None:
    """Point vtscore + HF at the pile. Call before importing anything vtscore."""
    import sys

    os.environ.setdefault("VTSEARCH_DATA_DIR", str(DATADIR))
    os.environ.setdefault("VTSEARCH_MODELS_DIR", str(MODELS))
    os.environ.setdefault("HF_HOME", str(MODELS))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for var in ("VTSEARCH_DATA_DIR", "VTSEARCH_MODELS_DIR", "HF_HOME"):
        Path(os.environ[var]).mkdir(parents=True, exist_ok=True)

    repo = os.environ.get("VTS_REPO")
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)
    # Drop the venv's editable-install finder so ``import vtscore`` resolves to
    # this checkout rather than whichever clone the editable install points at.
    keep = []
    for finder in sys.meta_path:
        mod = type(finder).__module__ or ""
        name = f"{mod}.{type(finder).__name__}".lower()
        if "editable" in name and ("vtsearch" in name or "vtscore" in name):
            continue
        keep.append(finder)
    sys.meta_path[:] = keep
