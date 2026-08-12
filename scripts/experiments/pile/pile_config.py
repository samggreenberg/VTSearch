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
#: ``source_dir`` is the demo extraction dir the loader treats as "already
#: downloaded" (vtscore/datasets/downloader/*.py). It must be present in the
#: datadir before a demo cell is built — see :func:`require_demo_source`.
DATASETS: dict[str, dict] = {
    "visual_genome_m": {"boxed": True, "kind": "demo", "source_dir": "visual_genome"},
    "caltech101_m": {"boxed": False, "kind": "demo", "source_dir": "caltech-101"},
    "coco_val": {"boxed": True, "kind": "coco"},
    # Box-size-banded VG, drawn from the WHOLE source (all 108k images, full
    # free-text vocabulary) rather than the demo pipeline's 100 curated
    # categories on a 4% slice.  The `_s`/`_m`/`_l` on `visual_genome_*` is a
    # dataset *size* tier and says nothing about boxes; these are the box bands.
    "vg_box_small": {"boxed": True, "kind": "vg_band", "band": "small"},
    "vg_box_medium": {"boxed": True, "kind": "vg_band", "band": "medium"},
    "vg_box_large": {"boxed": True, "kind": "vg_band", "band": "large"},
}

#: Box-size bands, as a fraction of image area, anchored to the patch
#: embedder's geometry (the same anchors the calibration harness bands on):
#: one DINOv3 patch is 1/196 of the image and the smallest HAC leaf is 1/12.
#: ``small`` is therefore "below what the patch grid can resolve at all".
#: The upper cut mirrors ``MAX_VOTED_AREA``: a box covering >80% of the image
#: is not a region, it is the image.
PATCH_AREA = 1 / 196
LEAF_AREA = 1 / 12
MAX_VOTED_AREA = 0.80
BOX_BANDS: dict[str, tuple[float, float]] = {
    "small": (0.0, PATCH_AREA),
    "medium": (PATCH_AREA, LEAF_AREA),
    "large": (LEAF_AREA, MAX_VOTED_AREA),
}

#: How many categories each banded dataset draws, and the image cap.  Categories
#: are stratified *within* the band so a band is not silently all one size.
BAND_N_CATEGORIES = int(os.environ.get("VTS_BAND_N_CATEGORIES", "40"))
BAND_MAX_IMAGES = int(os.environ.get("VTS_BAND_MAX_IMAGES", "12000"))
#: Categories whose union box is much larger than a single instance are
#: scattered instances, not a region a user would drag.
BAND_MAX_INFLATION = float(os.environ.get("VTS_BAND_MAX_INFLATION", "1.5"))
BAND_MIN_IMAGES = int(os.environ.get("VTS_BAND_MIN_IMAGES", "50"))

#: VG is annotated with free text, so its vocabulary is not a list of objects.
#: A detector asked to find "red" or "front" is measuring nothing, so these are
#: excluded from the banded datasets. The policy is **concrete countable
#: objects only**, which drops three kinds of name:
#:
#: * colours and other attributes -- properties, not things;
#: * frame relations and abstractions -- "front", "group", "object": either a
#:   position in the image or a placeholder the annotator reached for;
#: * mass nouns and unbounded surfaces -- "grass", "sky", "floor": real, but
#:   *stuff* rather than an object with an extent a user would drag a box around.
#:
#: The third group is the aggressive part of the policy and it costs coverage
#: in the large band specifically, because scene-scale stuff is exactly what
#: large boxes are made of. Countable landforms and structures ("tree",
#: "mountain", "building") are deliberately kept.
NON_OBJECT_CATEGORIES: frozenset[str] = frozenset(
    # attributes
    """red blue green yellow orange purple pink brown black white gray grey tan beige
    silver gold golden dark light bright colorful clear blurry shiny""".split()
    # frame relations, abstractions, placeholders
    + """front back side top bottom left right middle center centre corner edge end
    part section area region spot place row line lines stripe stripes pattern design
    shape size distance background foreground surface object objects thing things item
    items stuff group bunch pile set collection image picture photo photograph view
    scene display something other""".split()
    # mass nouns, unbounded surfaces and scene regions
    + """water snow sand dirt mud grass gravel concrete pavement asphalt sky smoke steam
    fog haze shade shadow shadows reflection glare sunlight ice foam liquid air weather
    ground floor flooring wall walls ceiling road roadway street sidewalk pathway path
    field beach ocean sea lake river land terrain lawn grassy lot traffic""".split()
)


def is_object_category(name: str) -> bool:
    """False for VG names that are not concrete countable objects.

    Matches on the **head noun** (the last token), not the whole string, because
    VG's vocabulary is full of modified compounds. Whole-string matching lets
    ``blue sky`` and ``table top`` through while head-noun matching drops them;
    matching *any* token would wrongly drop ``blue jeans``, ``tennis ball`` and
    ``left eye``, whose heads are perfectly good objects.
    """
    tokens = name.replace("-", " ").split()
    if not tokens:
        return False
    return tokens[-1] not in NON_OBJECT_CATEGORIES and name not in NON_OBJECT_CATEGORIES

#: Embedders in the pile. ``patch`` embedders attach ``patch_grid`` and are the
#: only ones that can carry a region-voting arm.
#: Deliberately three, not five. ``siglip`` is the shipped default and
#: ``siglip2_l`` the premium end; the middles (``siglip_l``, ``siglip2``) were
#: dropped because a study learns little from interpolating between them, and
#: the compute is better spent on more runs of the endpoints.
#:
#: The cost of that: ``siglip`` -> ``siglip2_l`` moves generation (1 -> 2) and
#: capacity (base -> SO400M) at the same time, so a difference between them
#: cannot be attributed to either alone. Rebuild a middle column if a result
#: ever needs that split -- ``build_pile.py --embedders siglip2`` restores one.
EMBEDDERS: dict[str, dict] = {
    "siglip": {"patch": False},
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


def require_demo_source(dataset: str) -> None:
    """Fail loudly if a demo dataset's source is not staged in the datadir.

    The demo downloaders treat a *missing* extraction dir as "not downloaded
    yet" and go fetch it. On a datadir that lost its symlink into the shared
    demo cache, that silently substitutes a partial re-download for the real
    dataset: the build still succeeds, but the cell holds a truncated subset
    and disagrees with its sibling cells. Cheaper to block than to detect.
    """
    name = DATASETS.get(dataset, {}).get("source_dir")
    if not name:
        return
    src = DATADIR / name
    if not src.exists():
        raise SystemExit(
            f"{dataset}: demo source {src} is missing, so the loader would re-download it.\n"
            f"  Link the shared cache in first, e.g.\n"
            f"    ln -s {DEMO_CACHE}/{name} {src}"
        )
    if not any(src.iterdir()):
        raise SystemExit(f"{dataset}: demo source {src} is empty (an empty dir reads as 'download complete')")


def setup_env() -> None:
    """Point vtscore + HF at the pile. Call before importing anything vtscore."""
    import sys

    os.environ.setdefault("VTSEARCH_DATA_DIR", str(DATADIR))
    os.environ.setdefault("VTSEARCH_MODELS_DIR", str(MODELS))
    os.environ.setdefault("HF_HOME", str(MODELS))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for var in ("VTSEARCH_DATA_DIR", "VTSEARCH_MODELS_DIR", "HF_HOME"):
        Path(os.environ[var]).mkdir(parents=True, exist_ok=True)

    # Default to the checkout this file lives in, rather than requiring VTS_REPO.
    # Depending on the env var is a live hazard: with it unset, ``import vtscore``
    # falls through to the venv's editable install, which points at the *main*
    # checkout -- 592 commits stale at the time of writing, and missing embedders
    # this pile uses. A build that resolved there would embed against different
    # code with no error. (This is how the shadow-module trap actually bites:
    # `VAR=x cmd1 && cmd2` applies VAR to cmd1 only, so the second command
    # silently ran against the wrong tree.)
    repo = os.environ.get("VTS_REPO") or str(Path(__file__).resolve().parents[3])
    if repo not in sys.path:
        sys.path.insert(0, repo)
    os.environ["VTS_REPO"] = repo  # so calibration's common.py agrees with us
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
