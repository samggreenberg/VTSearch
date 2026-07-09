#!/usr/bin/env python3
"""Feature materialization + on-disk cache for the SOD sweep.

Given a staged dataset, a bound :class:`~vtscore.eval.region_sources.RegionSource`,
and a class split, produce the vectors the metric core
(:mod:`vtscore.eval.region_curve`) consumes, caching every image's forward pass to
npz so re-runs and multiple K values reuse embeddings.

Split (seeded, per dataset/class/seed):
* positive images → annotation pool (source of K GT-box exemplars) + test positives
* negative images → training-negative pool (MLP negatives + x-cal) + eval negatives
  (the ~10× pool that measures FPR on the test set)

Two caches (npz):
* ``regions/<dataset>/<embedder>/<proposal_slug>/<id>.npz`` → boxes, vecs, whole_vec
* ``exemplars/<dataset>/<class>/<embedder>/<proposal_slug>/<id>.npz`` → exemplars
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vtscore.eval.region_curve import RegionCurveInputs
from vtscore.eval.region_sources import RegionSource

from datasets import Box, ClassSplit, SodDataset  # sibling module (scripts/sod on sys.path)


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.strip().lower()).strip("-")


# Per-image inference timing (embed + propose), accumulated on cache MISS across
# the run. Keyed by (dataset, embedder, proposal_slug). K-independent, so it's a
# per-config quantity, not a K-curve; the sweep renders it as a bar chart.
_PREP_TIMING: dict[tuple[str, str, str], list[float]] = {}


def _record_prep_time(dataset: str, embedder: str, slug: str, seconds: float) -> None:
    rec = _PREP_TIMING.setdefault((dataset, embedder, slug), [0.0, 0])
    rec[0] += seconds
    rec[1] += 1


def prep_timing_summary() -> list[dict]:
    """Total + mean per-image embed+propose time per config (only cache misses this run).

    ``embedder`` is the registry name and ``slug`` the proposal slug, so callers can
    join this against result rows (which now carry ``reg_name``/``proposal_slug``) to
    combine embedding cost with the MLP ``compute_ms`` into a total-time figure.
    """
    out: list[dict] = []
    for (d, e, s), (tot, cnt) in sorted(_PREP_TIMING.items()):
        if cnt:
            out.append(
                {
                    "dataset": d,
                    "embedder": e,
                    "proposal": s.split("_")[0],
                    "slug": s,
                    "embed_s": round(tot, 3),
                    "mean_ms": round(1000.0 * tot / cnt, 3),
                    "count": cnt,
                }
            )
    return out


class FeatureCache:
    """Per-image region/exemplar vectors, cached to npz under ``cache_dir``."""

    def __init__(self, cache_dir: Path, dataset: str, embedder: str, proposal_slug: str) -> None:
        self._regions_dir = cache_dir / "regions" / dataset / embedder / proposal_slug
        self._exem_root = cache_dir / "exemplars" / dataset
        self._dataset = dataset
        self._embedder = embedder
        self._proposal_slug = proposal_slug
        self._regions_dir.mkdir(parents=True, exist_ok=True)

    def regions(self, source: RegionSource, image_id: int, image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(region_boxes (R,4), region_vecs (R,D), whole_vec (D,)) for an image, cached."""
        path = self._regions_dir / f"{image_id}.npz"
        if path.exists():
            with np.load(path) as z:
                return z["boxes"], z["vecs"], z["whole_vec"]
        t0 = time.perf_counter()
        prep = source.prepare(image)  # embed + propose: the per-image inference cost
        _record_prep_time(self._dataset, self._embedder, self._proposal_slug, time.perf_counter() - t0)
        np.savez_compressed(path, boxes=prep.boxes, vecs=prep.vecs, whole_vec=prep.whole_vec)
        return prep.boxes, prep.vecs, prep.whole_vec

    def exemplars(self, source: RegionSource, class_slug: str, image_id: int, image, gt_boxes) -> np.ndarray:
        """(M,D) GT-box exemplar vectors for a positive image, cached."""
        d = self._exem_root / class_slug / self._embedder / self._proposal_slug
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{image_id}.npz"
        if path.exists():
            with np.load(path) as z:
                return z["exemplars"]
        prep = source.prepare(image, gt_boxes=list(gt_boxes))
        np.savez_compressed(path, exemplars=prep.exemplars)
        return prep.exemplars


def _partition(ids: list[int], test_fraction: float, rng: np.random.Generator) -> tuple[list[int], list[int]]:
    """Shuffle and split ids into (pool, test) by ``test_fraction`` for the test side."""
    order = rng.permutation(ids).tolist()
    n_test = int(round(len(order) * test_fraction))
    return order[n_test:], order[:n_test]  # (pool, test)


@dataclass
class Split:
    """Concrete train/test partition of image ids for one (dataset, class, seed)."""

    train_pos: list[int]  # training positives (source of K GT-box exemplars)
    test_pos: list[int]  # held-out positive test images
    train_neg: list[int]  # training-negative pool (MLP negatives + x-cal)
    test_neg: list[int]  # test negatives (measure FPR on the test set)
    gt_boxes: dict[int, list[Box]]  # positive id -> normalized GT boxes for the class
    negatives_exhaustive: bool


def partition_split(split: ClassSplit, test_fraction: float, seed: int) -> Split:
    """Deterministically partition a class split into train/test pos/neg buckets.

    Depends only on the class split + ``seed`` (the same rng-draw order as
    ``build_curve_inputs`` used), so callers that don't touch embedders (e.g. the
    viz tool) recover the exact same buckets.
    """
    rng = np.random.default_rng(seed)
    train_pos, test_pos = _partition(split.positive_ids, test_fraction, rng)
    train_neg, test_neg = _partition(split.negative_ids, test_fraction, rng)
    return Split(train_pos, test_pos, train_neg, test_neg, split.gt_boxes, split.negatives_exhaustive)


def dump_split(
    path: Path, *, dataset: str, class_name: str, split_seed: int, neg_count: int, test_fraction: float, split: Split
) -> None:
    """Write the split's image ids + counts to JSON for inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "class": class_name,
                "split_seed": split_seed,
                "neg_count": neg_count,
                "test_fraction": test_fraction,
                "negatives_exhaustive": split.negatives_exhaustive,
                "counts": {
                    "train_pos": len(split.train_pos),
                    "test_pos": len(split.test_pos),
                    "train_neg": len(split.train_neg),
                    "test_neg": len(split.test_neg),
                },
                "train_pos": split.train_pos,
                "test_pos": split.test_pos,
                "train_neg": split.train_neg,
                "test_neg": split.test_neg,
            },
            indent=2,
        )
    )


def build_curve_inputs(
    dataset: SodDataset,
    source: RegionSource,
    split: Split,
    cache: FeatureCache,
    *,
    class_name: str,
    meta: dict,
    neg_regions: bool = False,
) -> RegionCurveInputs:
    """Materialize (cached) vectors from a partitioned :class:`Split`.

    ``neg_regions`` selects the MLP's negative training distribution: when False
    (default), negatives are the *whole-image* vector of each training-negative
    image; when True, negatives are that image's *proposed-region* vectors (same
    distribution as the test regions the head is scored on), which better matches
    train to test for crop/HAC proposals. No-op for ``whole`` (its only region is
    the full frame). Reads region vecs from the same cached npz — no re-embedding.
    """
    class_slug = slugify(class_name)

    # Positive exemplar pool: every GT-box exemplar across training-positive images.
    pos_ex_list: list[np.ndarray] = []
    for iid in split.train_pos:
        ex = cache.exemplars(source, class_slug, iid, dataset.load_image(iid), split.gt_boxes[iid])
        if ex.shape[0] > 0:
            pos_ex_list.append(ex)
    pos_exemplars = np.vstack(pos_ex_list) if pos_ex_list else np.zeros((0, source.input_dim), np.float32)

    # Training negatives: whole-image vectors, or proposed-region vectors when
    # --neg-regions (matches the test-region distribution).
    neg_list: list[np.ndarray] = []
    for iid in split.train_neg:
        _boxes, vecs, whole = cache.regions(source, iid, dataset.load_image(iid))
        neg_list.append(vecs if neg_regions else whole[None, :])
    neg_train = np.vstack(neg_list) if neg_list else np.zeros((0, source.input_dim), np.float32)

    # Test set: region matrices + boxes + GT boxes for held-out positives + test negatives.
    test_mats: list[np.ndarray] = []
    test_boxes: list[np.ndarray] = []
    test_gt: list[list] = []
    test_labels: list[int] = []
    for iid in split.test_pos:
        boxes, vecs, _whole = cache.regions(source, iid, dataset.load_image(iid))
        test_mats.append(vecs)
        test_boxes.append(boxes)
        test_gt.append(split.gt_boxes.get(iid, []))
        test_labels.append(1)
    for iid in split.test_neg:
        boxes, vecs, _whole = cache.regions(source, iid, dataset.load_image(iid))
        test_mats.append(vecs)
        test_boxes.append(boxes)
        test_gt.append([])
        test_labels.append(0)

    return RegionCurveInputs(
        pos_exemplars=pos_exemplars,
        neg_train_wholes=neg_train,
        test_region_mats=test_mats,
        test_labels=test_labels,
        input_dim=source.input_dim,
        test_region_boxes=test_boxes,
        test_gt_boxes=test_gt,
        meta={
            **meta,
            "n_train_pos": len(split.train_pos),
            "n_train_neg": len(split.train_neg),
            "n_pos_exemplars": int(pos_exemplars.shape[0]),
        },
    )
