#!/usr/bin/env python3
"""Dataset adapter for the small-object-detection sweep.

Turns the staged COCO / LVIS / Visual Genome derived extracts (+ zipped images)
into the per-class structure the metric core needs: positive image ids, a seeded
negative sample, per-image ground-truth boxes for the class (normalized), and a
by-id clean-pixel reader that streams JPEGs straight from the zips (images are
kept zipped on NFS; see each dataset dir's README).

COCO negatives are exhaustively annotated (a non-positive image is a true
negative → clean FPR); LVIS is federated and VG is noisy free-text, so their
"negatives" are only *not-labelled-positive* (FPR is an upper bound). Each
adapter reports ``negatives_exhaustive`` so the orchestrator can flag it per row.

Visual Genome is registered five ways: ``vg`` (the whole annotated corpus) and the
size slices ``vg_s`` / ``vg_m`` / ``vg_l`` / ``vg_a``, which reproduce the GUI demo
datasets ``visual_genome_{s,m,l,a}`` image-for-image so sweep results line up with the
``vtscore.eval --datasets visual_genome_m`` experiments under ``docs/experiments/``.
"""

from __future__ import annotations

import gzip
import io
import json
import re
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image

Box = tuple[float, float, float, float]

# Minimum drawable annotation size in the GUI: a Good-vote box narrower than this
# fraction of the image on EITHER axis is rejected client-side (``MIN_BOX_SIZE`` in
# frontend .../image-viewer/image-viewer.component.ts). We mirror it here so the sweep
# never trains/evaluates on GT boxes a human annotator could not have drawn. ``0.0``
# disables the filter (keep every box).
GUI_MIN_BOX_FRAC = 0.01

EXTERNAL = Path("/exp/scale26/datasets/external")

_VG_BASE = {
    "extract": EXTERNAL / "VisualGenome/derived/objects_flat.jsonl.gz",
    "images": EXTERNAL / "VisualGenome/images",
    "kind": "vg",
    "negatives_exhaustive": False,
}

# Visual Genome size slices, matching the GUI demo datasets ``visual_genome_{s,m,l,a}``
# (vtscore/media/image/_demo_sources.py) so sweep numbers are directly comparable to the
# ``vtscore.eval --datasets visual_genome_m`` experiments in docs/experiments/. VG is
# multi-label, so the demo slices the image list *flat* rather than per-category; see
# ``_demo_universe`` for the exact reproduction of that ordering.
_VG_DEMO_SLICES = {
    "vg_s": (0.0, 1 / 50),
    "vg_m": (1 / 50, 3 / 50),
    "vg_l": (3 / 50, 7 / 50),
    "vg_a": (0.0, None),
}

_CONFIG = {
    "coco": {
        "extract": EXTERNAL / "COCO/derived/objects_flat_val2017.jsonl.gz",
        "images": EXTERNAL / "COCO/images",
        "kind": "coco_lvis",
        "negatives_exhaustive": True,
    },
    "lvis": {
        "extract": EXTERNAL / "LVIS/derived/objects_flat_lvis_v1_val.jsonl.gz",
        "images": EXTERNAL / "LVIS/images",
        "kind": "coco_lvis",
        "negatives_exhaustive": False,
    },
    # Whole corpus: every annotated VG image, no demo vocabulary filter.
    "vg": dict(_VG_BASE),
    **{name: {**_VG_BASE, "demo_slice": frac} for name, frac in _VG_DEMO_SLICES.items()},
}


@lru_cache(maxsize=1)
def _vg_demo_matcher():
    """Return ``names -> bool``: is this VG object one of the demo's 100 categories?

    Reuses the demo loader's own vocabulary and name normalization (case/whitespace
    fold, irregular plurals, naive trailing-``s``) so slice membership is decided by
    exactly the same rule ``_collect_visual_genome_files`` applies.
    """
    from vtscore.media.image._demo_categories import VISUAL_GENOME_CATEGORIES
    from vtscore.media.image._demo_sources import _vg_category_for

    vocab = frozenset(VISUAL_GENOME_CATEGORIES)

    def in_vocab(names: object) -> bool:
        parts = names if isinstance(names, list) else [names]
        return any(p and _vg_category_for(str(p), vocab) is not None for p in parts)

    return in_vocab


def _norm(s: str) -> str:
    """Canonicalize a category so COCO 'stop sign' == LVIS 'stop_sign'."""
    return re.sub(r"[ _]+", " ", str(s).strip().lower())


# ---------------------------------------------------------------------------
# Pixel readers (keep images zipped; read members in-memory)
# ---------------------------------------------------------------------------


class _SplitZipReader:
    """COCO/LVIS: a row's ``file_name`` (``val2017/xxx.jpg``) is the zip member."""

    def __init__(self, images_dir: Path) -> None:
        self._dir = images_dir
        self._handles: dict[str, zipfile.ZipFile] = {}

    def load(self, split: str, file_name: str) -> Image.Image:
        zf = self._handles.get(split)
        if zf is None:
            zf = zipfile.ZipFile(self._dir / f"{split}.zip")
            self._handles[split] = zf
        return Image.open(io.BytesIO(zf.read(file_name))).convert("RGB")

    def close(self) -> None:
        for zf in self._handles.values():
            zf.close()
        self._handles.clear()


class _VgZipReader:
    """VG: image_id -> (zip, member) over images.zip + images2.zip (int stem key)."""

    def __init__(self, images_dir: Path) -> None:
        self._index: dict[int, tuple[Path, str]] = {}
        self._handles: dict[Path, zipfile.ZipFile] = {}
        for name in ("images.zip", "images2.zip"):
            zp = images_dir / name
            if not zp.exists():
                continue
            with zipfile.ZipFile(zp) as zf:
                for member in zf.namelist():
                    if member.endswith(".jpg"):
                        try:
                            self._index[int(Path(member).stem)] = (zp, member)
                        except ValueError:
                            continue

    def has(self, image_id: int) -> bool:
        return image_id in self._index

    def load(self, image_id: int) -> Image.Image:
        zp, member = self._index[image_id]
        zf = self._handles.get(zp)
        if zf is None:
            zf = zipfile.ZipFile(zp)
            self._handles[zp] = zf
        return Image.open(io.BytesIO(zf.read(member))).convert("RGB")

    def close(self) -> None:
        for zf in self._handles.values():
            zf.close()
        self._handles.clear()


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


@dataclass
class ClassSplit:
    """Positive/negative image ids and per-image GT boxes for one class."""

    positive_ids: list[int]
    negative_ids: list[int]
    gt_boxes: dict[int, list[Box]]  # positive id -> normalized boxes for the class
    negatives_exhaustive: bool


class SodDataset:
    """One staged dataset (coco/lvis/vg) with a by-id clean-pixel reader."""

    def __init__(self, name: str) -> None:
        if name not in _CONFIG:
            raise ValueError(f"unknown dataset {name!r}; choices: {sorted(_CONFIG)}")
        self.name = name
        cfg = _CONFIG[name]
        self._extract: Path = cfg["extract"]
        self._kind: str = cfg["kind"]
        self.negatives_exhaustive: bool = cfg["negatives_exhaustive"]
        for p in (self._extract, cfg["images"]):
            if not p.exists():
                raise SystemExit(f"missing {p}; stage the {name.upper()} dataset first.")
        # (start, end) image-list fractions for the VG demo size slices; None = whole corpus.
        self._demo_slice: tuple[float, float | None] | None = cfg.get("demo_slice")
        self._reader: _SplitZipReader | _VgZipReader = (
            _SplitZipReader(cfg["images"]) if self._kind == "coco_lvis" else _VgZipReader(cfg["images"])
        )
        # id -> pixel locator: (split, file_name) for coco/lvis; None for VG (id-keyed).
        self._locator: dict[int, tuple[str, str]] = {}

    def _demo_universe(self, invocab_ids: set[int]) -> set[int]:
        """Reproduce the GUI demo dataset's image set for this size slice.

        ``_collect_visual_genome_files`` keeps images with >=1 in-vocab object, drops
        those whose JPEG is missing, sorts by ``image_id``, and *then* takes the
        fractional slice. The presence check must stay ahead of the slice: 4 in-vocab
        images have no JPEG, and pruning them afterwards would shift every later index.

        Parity is exact for ``vg_s`` and ``vg_m`` and off by one image in 8,386 for
        ``vg_l`` (measured 2026-08-06 against the demo loader over the real corpus).
        The residue is 3 images that ``extract_objects.py`` dropped because their only
        in-vocab object had a degenerate box, which nudges the later slice boundaries.
        """
        assert self._demo_slice is not None
        ids = sorted(i for i in invocab_ids if self._reader.has(i))  # type: ignore[union-attr]
        start_frac, end_frac = self._demo_slice
        n = len(ids)
        start = int(n * start_frac)
        end = int(n * end_frac) if end_frac is not None else n
        return set(ids[start:end])

    def _row_matches(self, row: dict, q_norm: str, q_syn: str) -> bool:
        if self._kind == "coco_lvis":
            return _norm(row.get("name", "")) == q_norm
        # VG: synset-canonical with a case-insensitive name fallback.
        syn = str(row.get("synset", "")).strip().lower()
        if syn:
            return syn.split(".")[0] == q_syn
        return _norm(row.get("name", "")) == q_norm

    def class_split(
        self, category: str, *, neg_multiple: int, seed: int, min_box_frac: float = GUI_MIN_BOX_FRAC
    ) -> ClassSplit:
        """Stream the extract once → positives (+ boxes), and a seeded negative sample.

        ``neg_multiple`` sizes the sampled negative pool as ``neg_multiple × n_positives``
        (capped by the available negatives), so prevalence ≈ ``1/(1+neg_multiple)`` holds
        constant across classes regardless of how many positives a class has.

        ``min_box_frac`` (default: the GUI's :data:`GUI_MIN_BOX_FRAC`) drops any GT box
        below that fraction of the image on **either** axis — the same rule the annotation
        GUI enforces on a drawn box — so the sweep never sees an un-drawable annotation.
        Filtering is per-box: an image stays a positive as long as ≥1 of its class boxes
        survives; only surviving boxes populate ``gt_boxes``. Pass ``0.0`` to keep all boxes.
        """
        import numpy as np

        q_norm = _norm(category)
        q_syn = q_norm.replace(" ", "_")
        pos_boxes: dict[int, list[Box]] = {}
        all_ids: set[int] = set()
        class_ids: set[int] = set()  # every image containing the class, pre size-filter
        locator: dict[int, tuple[str, str]] = {}
        # Demo-slice membership rides along on the single pass we already make.
        invocab_ids: set[int] = set()
        in_demo_vocab = _vg_demo_matcher() if self._demo_slice is not None else None

        with gzip.open(self._extract, "rt", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                iid = int(row["image_id"])
                all_ids.add(iid)
                if in_demo_vocab is not None and iid not in invocab_ids and in_demo_vocab(row.get("names")):
                    invocab_ids.add(iid)
                if self._kind == "coco_lvis":
                    locator[iid] = (str(row["split"]), str(row["file_name"]))
                if self._row_matches(row, q_norm, q_syn):
                    class_ids.add(iid)
                    x0, y0, x1, y1 = float(row["x0"]), float(row["y0"]), float(row["x1"]), float(row["y1"])
                    # Skip boxes a human couldn't draw in the GUI (below the floor on either axis).
                    if (x1 - x0) < min_box_frac or (y1 - y0) < min_box_frac:
                        continue
                    pos_boxes.setdefault(iid, []).append((x0, y0, x1, y1))

        self._locator = locator
        if self._demo_slice is not None:
            universe = self._demo_universe(invocab_ids)
            all_ids &= universe
            class_ids &= universe
            pos_boxes = {i: b for i, b in pos_boxes.items() if i in universe}
        positive_ids = sorted(pos_boxes)
        # Exclude EVERY class-containing image from the negative pool — including images
        # dropped as positives because all their class boxes were sub-floor (they still
        # contain the class, so they'd be false negatives). Such images are ignored entirely.
        cand = [i for i in sorted(all_ids) if i not in class_ids]
        if self._kind == "vg":
            cand = [i for i in cand if self._reader.has(i)]  # type: ignore[union-attr]
            positive_ids = [i for i in positive_ids if self._reader.has(i)]  # type: ignore[union-attr]
        rng = np.random.default_rng(seed)
        neg_count = neg_multiple * len(positive_ids)  # pool = multiple × positives
        take = min(neg_count, len(cand))
        neg_ids = sorted(int(x) for x in rng.choice(cand, size=take, replace=False)) if take else []
        return ClassSplit(
            positive_ids=positive_ids,
            negative_ids=neg_ids,
            gt_boxes={i: pos_boxes[i] for i in positive_ids},
            negatives_exhaustive=self.negatives_exhaustive,
        )

    def load_image(self, image_id: int) -> Image.Image:
        if self._kind == "coco_lvis":
            split, file_name = self._locator[image_id]
            return self._reader.load(split, file_name)  # type: ignore[union-attr]
        return self._reader.load(image_id)  # type: ignore[union-attr]

    def close(self) -> None:
        self._reader.close()

    def __enter__(self) -> SodDataset:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
