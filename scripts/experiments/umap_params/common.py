"""Shared config, dataset roster, and taxonomy builders for the UMAP sweep.

The sweep runs four embedders — ``clap`` (audio) and ``clip`` / ``siglip`` /
``siglip_l`` (image) — over a roster of demo datasets plus two deep-taxonomy
additions (iNaturalist subset, FSD50K eval). Each dataset carries a class
taxonomy used by ``metric.taxonomy_separability``; this module knows how to
turn a per-item leaf label into the full ``level -> member-masks`` structure the
metric consumes.

Dev-only experiment code. Imports VTSearch only inside functions so the module
is importable for its constants without the heavy stack.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --- paths ------------------------------------------------------------------
DATA_ROOT = Path(os.environ.get("UMAP_EXP_DATA", "/exp/scale26/datasets/external/vtsearch-demos"))
RESULTS_ROOT = Path(os.environ.get("UMAP_EXP_RESULTS", "/exp/scale26/datasets/external/vtsearch-demos/umap-sweep"))
MATRIX_DIR = RESULTS_ROOT / "matrices"  # cached (ids, matrix, labels) npz per (dataset, embedder)
CSV_DIR = RESULTS_ROOT / "rows"  # per-(dataset,embedder) sweep-row CSVs
FIG_DIR = RESULTS_ROOT / "figures"

IMAGE_EMBEDDERS = ["clip", "siglip", "siglip_l"]
AUDIO_EMBEDDERS = ["clap"]

# --- ESC-50: canonical 50-class order → 5 super-categories -------------------
# ESC-50 groups its 50 classes into 5 super-categories of 10, in this canonical
# order (target // 10). The demo pickle keeps only the leaf `category` string,
# so we map back through this well-known ordering.
ESC50_SUPERS = ["Animals", "Natural", "Human", "Interior", "Exterior"]
ESC50_ORDER = [
    # Animals
    "dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects", "sheep", "crow",
    # Natural soundscapes & water
    "rain", "sea_waves", "crackling_fire", "crickets", "chirping_birds",
    "water_drops", "wind", "pouring_water", "toilet_flush", "thunderstorm",
    # Human, non-speech
    "crying_baby", "sneezing", "clapping", "breathing", "coughing",
    "footsteps", "laughing", "brushing_teeth", "snoring", "drinking_sipping",
    # Interior / domestic
    "door_wood_knock", "mouse_click", "keyboard_typing", "door_wood_creaks",
    "can_opening", "washing_machine", "vacuum_cleaner", "clock_alarm",
    "clock_tick", "glass_breaking",
    # Exterior / urban
    "helicopter", "chainsaw", "siren", "car_horn", "engine",
    "train", "church_bells", "airplane", "fireworks", "hand_saw",
]
_ESC50_SUPER_OF = {c: ESC50_SUPERS[i // 10] for i, c in enumerate(ESC50_ORDER)}


@dataclass
class DatasetSpec:
    """One dataset in the roster.

    ``source`` selects how ``prepare_dataset`` gets (ids, matrix, per-item
    labels):
      - ``"pkl"``    reuse an existing embedded pickle verbatim (audio/clap).
      - ``"reembed"`` take file list + labels from a pickle, run image embedders.
      - ``"folder"`` enumerate an image folder (iNaturalist).
      - ``"fsd50k"`` custom FSD50K eval loader.
    ``taxonomy`` names the label→levels builder.
    """

    name: str
    media_type: str  # "audio" | "image"
    source: str
    taxonomy: str
    pkl: str = ""  # basename under data/embeddings for pkl/reembed
    folder: str = ""  # path for folder/fsd50k
    embedders: list[str] = field(default_factory=list)

    def embedder_list(self) -> list[str]:
        return self.embedders or (AUDIO_EMBEDDERS if self.media_type == "audio" else IMAGE_EMBEDDERS)


# The roster. N values are approximate (confirmed at prepare time).
ROSTER: list[DatasetSpec] = [
    # --- audio (clap) ---
    DatasetSpec("esc50_s", "audio", "pkl", "esc50", pkl="esc50_s.pkl"),
    DatasetSpec("esc50_m", "audio", "pkl", "esc50", pkl="esc50_m.pkl"),
    DatasetSpec("esc50_l", "audio", "pkl", "esc50", pkl="esc50_l.pkl"),
    DatasetSpec("gtzan_a", "audio", "pkl", "flat", pkl="gtzan_a.pkl"),
    DatasetSpec("fsd50k_eval", "audio", "fsd50k", "fsd50k", folder=str(DATA_ROOT / "fsd50k")),
    # --- image (clip / siglip / siglip_l) ---
    DatasetSpec("places365_s", "image", "reembed", "places365", pkl="places365_s.pkl"),
    DatasetSpec("places365_m", "image", "folder", "places365", folder=str(DATA_ROOT / "places365")),
    DatasetSpec("places365_l", "image", "folder", "places365", folder=str(DATA_ROOT / "places365")),
    DatasetSpec("caltech256_s", "image", "reembed", "flat", pkl="caltech256_s.pkl"),
    DatasetSpec("caltech256_m", "image", "reembed", "flat", pkl="caltech256_m.pkl"),
    DatasetSpec("inat_val", "image", "folder", "inat", folder=str(DATA_ROOT / "inat2021" / "images" / "val")),
]

ROSTER_BY_NAME = {d.name: d for d in ROSTER}


# --- taxonomy builders ------------------------------------------------------
# Each returns dict[level_name -> list of boolean member-masks (one per node)],
# built from a per-item leaf-label array (and, for iNat/fsd50k, a lineage map).


def _masks_from_labels(labels: np.ndarray, min_members: int = 8) -> list[np.ndarray]:
    """One boolean mask per distinct label with >= min_members items."""
    out = []
    for lab in sorted(set(labels.tolist())):
        m = labels == lab
        if m.sum() >= min_members:
            out.append(m)
    return out


def taxonomy_flat(leaf: np.ndarray, **_) -> dict[str, list[np.ndarray]]:
    """Single-level taxonomy: the leaf class itself (GTZAN, UrbanSound, Caltech)."""
    return {"class": _masks_from_labels(leaf)}


def taxonomy_esc50(leaf: np.ndarray, **_) -> dict[str, list[np.ndarray]]:
    """Two levels: 5 super-categories over 50 leaf classes."""
    supers = np.array([_ESC50_SUPER_OF.get(c, "?") for c in leaf])
    return {
        "supercategory": _masks_from_labels(supers, min_members=8),
        "class": _masks_from_labels(leaf, min_members=6),
    }


def taxonomy_places365(leaf: np.ndarray, **_) -> dict[str, list[np.ndarray]]:
    """Two levels: indoor/outdoor (where the flattened name encodes it) + scene.

    Places365 flattens ``/b/balcony/interior`` → ``balcony_interior``; the
    indoor/outdoor suffix survives on the ~scenes that carried it, so the coarse
    level covers that subset while the scene name is the always-present leaf.
    """
    io = []
    for c in leaf:
        if c.endswith("_indoor") or c.endswith("_interior"):
            io.append("indoor")
        elif c.endswith("_outdoor"):
            io.append("outdoor")
        else:
            io.append("?")  # no suffix → excluded from the coarse level below
    io = np.array(io)
    io_masks = [io == "indoor", io == "outdoor"]
    io_masks = [m for m in io_masks if m.sum() >= 20]
    return {"indoor_outdoor": io_masks, "scene": _masks_from_labels(leaf, min_members=10)}


def taxonomy_inat(leaf: np.ndarray, lineage: dict[str, list[str]] | None = None, **_) -> dict[str, list[np.ndarray]]:
    """Full biological lineage: kingdom → phylum → class → order → family → genus.

    ``lineage`` maps each item's leaf id (species dir name) to its ordered
    ancestor labels. Species (the leaf) is the finest; we score every internal
    level as its own taxonomy tier.
    """
    levels = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
    tax: dict[str, list[np.ndarray]] = {}
    lin = lineage or {}
    per_level = {lv: np.array([lin.get(c, [""] * 7)[i] for c in leaf]) for i, lv in enumerate(levels)}
    for lv in levels:
        masks = _masks_from_labels(per_level[lv], min_members=8)
        if masks:
            tax[lv] = masks
    return tax


def taxonomy_fsd50k(
    leaf: np.ndarray, ml_labels=None, ml_names=None, ml_isroot=None, **_
) -> dict[str, list[np.ndarray]]:
    """Two multi-label tiers: 7 AudioSet roots + the specific FSD50K classes.

    ``ml_labels`` is the dense ``(N, K)`` membership matrix; ``ml_isroot`` marks
    which of the K classes are ontology top-level categories. Each class column
    is a one-vs-rest node (multi-label needs no special handling).
    """
    top, cls = [], []
    for j in range(ml_labels.shape[1]):
        mask = ml_labels[:, j].astype(bool)
        if mask.sum() < 12:
            continue
        (top if ml_isroot[j] else cls).append(mask)
    return {"top": top, "class": cls}


TAXONOMY_BUILDERS = {
    "flat": taxonomy_flat,
    "esc50": taxonomy_esc50,
    "places365": taxonomy_places365,
    "inat": taxonomy_inat,
    "fsd50k": taxonomy_fsd50k,
}


def load_inat_lineage() -> dict[str, list[str]]:
    """species image_dir_name → [kingdom, phylum, class, order, family, genus, species]."""
    sel = json.load(open(DATA_ROOT / "inat2021" / "selection.json"))
    return {
        c["image_dir_name"]: [c["kingdom"], c["phylum"], c["class"], c["order"], c["family"], c["genus"], c["name"]]
        for c in sel
    }


# --- the parameter grid -----------------------------------------------------
N_NEIGHBORS_GRID = [5, 10, 15, 30, 50, 100, 200]
MIN_DIST_GRID = [0.0, 0.05, 0.1, 0.25, 0.5]
COMPACT_GRID = [False, True]  # free axis: both scored from the same fit
SEEDS = [0, 1, 2]
METRIC_K = 20  # kNN scale for separability + guards
MIN_N_FOR_UMAP = 10  # below this fit_projection falls back to PCA; exclude from grid
