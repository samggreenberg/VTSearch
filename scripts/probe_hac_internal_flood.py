"""Should a Bad vote flood the internal HAC nodes too?

Issue #2731's "Related, smaller" note.  ``bad_negative_vecs`` floods the CLS
node plus the HAC **leaves**; ``_score_all_media`` max-pools **every** region
node, internals included.  So a Bad vote leaves scored rows it never trains
down.  The gap was justified by calling internals redundant pools of leaves
already suppressed.  This probe answers two questions:

1. **Are internals actually dominated by their leaves?**  No.
   ``build_hac_tree`` renormalises each merge (``_l2_normalize(sum_a +
   sum_b)``), so the merged vector is the descendants' convex-hull point
   projected back onto the unit sphere - a ~1.5x gain.  Under a linear head
   that scales the logit by the same factor, and an internal out-scores every
   one of its own leaves on ~5% of node/direction pairs.

2. **Does flooding them anyway help?**  No - it measurably hurts ranking.  Over
   24 synthetic patch detectors the paired AP change is about -0.06 with a 95%
   CI that excludes zero.  Suppressing a rejected image's renormalised mean
   directions also suppresses the geometry a *positive* image's concept-blob
   internal node lives in, and that costs more than the ~2 points of
   negative-internal wins it buys back.

So the flood stays leaves-only, and the real reason is recorded rather than the
refuted redundancy argument.  Pinned by
``tests_lib/detectors/test_hac_internal_flood_gap.py``.

Usage::

    python scripts/probe_hac_internal_flood.py --seeds 24

Synthetic only - no model downloads, no dataset.  Runs in well under a minute.
"""

from __future__ import annotations

import argparse

import numpy as np

from vtscore.media.patch_embed import PatchEmbedOutput, build_region_tree


DIM = 32
GRID = 14
K = 12
EMB = "dinov3_patch"


def _unit(v: np.ndarray, axis: int = -1) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), 1e-12)


def _make_media(cid, rng, concept, backgrounds, positive):
    """One synthetic patch image; positives carry a contiguous concept blob."""
    idx = rng.integers(0, len(backgrounds), size=(GRID, GRID, 2))
    w = rng.random((GRID, GRID, 2)).astype(np.float32)
    grid = (backgrounds[idx] * w[..., None]).sum(axis=2)
    grid += 0.35 * rng.standard_normal((GRID, GRID, DIM)).astype(np.float32)

    box = None
    if positive:
        h = int(rng.integers(3, 5))
        r0 = int(rng.integers(0, GRID - h))
        c0 = int(rng.integers(0, GRID - h))
        grid[r0 : r0 + h, c0 : c0 + h] = concept + 0.30 * rng.standard_normal((h, h, DIM)).astype(np.float32)
        box = (c0 / GRID, r0 / GRID, (c0 + h) / GRID, (r0 + h) / GRID)

    grid = _unit(grid)
    sal = rng.random((GRID, GRID)).astype(np.float32) + 0.2
    if positive:
        sal[r0 : r0 + h, c0 : c0 + h] += 1.0
    sal /= sal.sum()
    # A real CLS token is a separate pooled representation, not the exact
    # saliency-weighted patch mean.  Making them equal would put the HAC root
    # on top of the CLS node, so the flood would cover the root by accident and
    # the gap under test would be understated.
    pooled = (grid * sal[..., None]).reshape(-1, DIM).sum(axis=0)
    cls = _unit(pooled + 0.8 * rng.standard_normal(DIM).astype(np.float32))

    output = PatchEmbedOutput(cls_vec=cls, patch_grid=grid, patch_saliency=sal)
    return {
        "id": cid,
        "md5": f"md5-{cid:05x}",
        "media_type": "image",
        "embedder": EMB,
        "embeddings": {EMB: cls},
        "patch_regions": build_region_tree(output, k=K),
        "_positive": positive,
        "_box": box,
    }


def build_dataset(seed: int, n: int = 240, pos_frac: float = 0.25):
    rng = np.random.default_rng(seed)
    concept = _unit(rng.standard_normal(DIM))
    backgrounds = _unit(rng.standard_normal((6, DIM)))
    clips = {cid: _make_media(cid, rng, concept, backgrounds, rng.random() < pos_frac) for cid in range(n)}
    return clips, rng


def _node_kinds(regions) -> np.ndarray:
    """0 = CLS, 1 = HAC leaf, 2 = HAC internal."""
    return np.asarray([0 if i == 0 else (2 if r.children is not None else 1) for i, r in enumerate(regions)])


def _descendant_leaves(regions, i: int) -> list[int]:
    node = regions[i]
    if node.children is None:
        return [i]
    a, b = node.children
    return _descendant_leaves(regions, a) + _descendant_leaves(regions, b)


# ---------------------------------------------------------------------------
# Q1: is an internal node dominated by its leaves?
# ---------------------------------------------------------------------------


def probe_geometry(n_images: int = 60, n_directions: int = 20) -> None:
    rng = np.random.default_rng(0)
    exceed = total = 0
    gaps: list[float] = []
    hull_norms: list[float] = []
    for seed in range(3):
        clips, _ = build_dataset(seed, n=n_images)
        for media in clips.values():
            regions = media["patch_regions"]
            kinds = _node_kinds(regions)
            vecs = np.stack([np.asarray(r.vec, dtype=np.float32) for r in regions])
            directions = _unit(rng.standard_normal((n_directions, DIM)).astype(np.float32), axis=1)
            proj = vecs @ directions.T
            for i in np.flatnonzero(kinds == 2):
                leaves = _descendant_leaves(regions, int(i))
                leaf_max = proj[leaves].max(axis=0)
                exceed += int((proj[i] > leaf_max + 1e-6).sum())
                total += proj.shape[1]
                gaps.append(float(np.mean(proj[i] - leaf_max)))
                hull_norms.append(float(np.linalg.norm(vecs[leaves].mean(axis=0))))

    print("Q1  internal vs its own descendant leaves, random linear heads")
    print(f"    node/direction pairs                : {total}")
    print(f"    internal EXCEEDS its leaf max       : {exceed / total:.3f}")
    print(f"    mean (internal - leafmax) projection: {np.mean(gaps):+.4f}")
    print(f"    mean ||leaf-mean|| (hull point)     : {np.mean(hull_norms):.4f}")
    print(f"    => renormalisation gain             : {1 / np.mean(hull_norms):.3f}x")


# ---------------------------------------------------------------------------
# Q2: does flooding internals help?
# ---------------------------------------------------------------------------


def _flood_all(media, embedder_name=None):
    regions = media.get("patch_regions")
    if regions:
        return [np.asarray(r.vec, dtype=np.float32) for r in regions]
    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

    return [media_embedding(media, embedder_name)]


def _run_arm(clips, good, bad, boxes, *, flood_internals: bool) -> dict[str, float]:
    import torch  # noqa: PLC0415

    from vtscore.detectors import training as T  # noqa: PLC0415

    original = T.bad_negative_vecs
    if flood_internals:
        T.bad_negative_vecs = _flood_all
    try:
        X, y, groups, score_rows = T._build_vote_xy(clips, good, bad, boxes, EMB)
        _results, threshold, model = T._train_and_score_xy(
            X,
            y,
            clips,
            inclusion_value=50,
            safe_thresholds=False,
            calibrate_count=2,
            calibration_fraction=0.3,
            det_ctx=None,
            groups=groups,
            score_rows=score_rows,
        )
    finally:
        T.bad_negative_vecs = original

    ids, scores, best = T._score_all_media(model, clips, EMB)
    labels = np.asarray([clips[c]["_positive"] for c in ids], dtype=bool)
    s = np.asarray(scores)

    lab = labels[np.argsort(-s)]
    prec = np.cumsum(lab) / np.arange(1, len(lab) + 1)
    ap = float((prec * lab).sum() / max(lab.sum(), 1))

    pred = s >= threshold
    neg_int_win = sum(
        1
        for cid, b, is_pos in zip(ids, best, labels, strict=True)
        if not is_pos and _node_kinds(clips[cid]["patch_regions"])[b] == 2
    )
    del torch
    return {
        "ap": ap,
        "threshold": float(threshold),
        "fpr": float((pred & ~labels).sum() / max((~labels).sum(), 1)),
        "fnr": float((~pred & labels).sum() / max(labels.sum(), 1)),
        "neg_int_win": neg_int_win / max(int((~labels).sum()), 1),
    }


def probe_ab(n_seeds: int) -> None:
    rows_a: list[dict[str, float]] = []
    rows_b: list[dict[str, float]] = []
    for seed in range(n_seeds):
        clips, rng = build_dataset(seed)
        pos_ids = [c for c in clips if clips[c]["_positive"]]
        neg_ids = [c for c in clips if not clips[c]["_positive"]]
        good = {c: None for c in rng.permutation(pos_ids)[:6].tolist()}
        bad = {c: None for c in rng.permutation(neg_ids)[:6].tolist()}
        boxes = {c: clips[c]["_box"] for c in good}
        rows_a.append(_run_arm(clips, good, bad, boxes, flood_internals=False))
        rows_b.append(_run_arm(clips, good, bad, boxes, flood_internals=True))

    print(f"\nQ2  flood A = CLS + leaves (production)  vs  B = every region node   [{n_seeds} seeds]")
    print(f"    {'metric':>12} {'A':>10} {'B':>10}")
    for key in ("ap", "threshold", "fpr", "fnr", "neg_int_win"):
        print(f"    {key:>12} {np.mean([r[key] for r in rows_a]):>10.4f} {np.mean([r[key] for r in rows_b]):>10.4f}")
    print()
    for key in ("ap", "fpr", "fnr"):
        d = np.asarray([b[key] - a[key] for a, b in zip(rows_a, rows_b, strict=True)])
        ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        verdict = "excludes 0" if abs(d.mean()) > ci else "straddles 0"
        print(
            f"    paired B-A {key:>4}: {d.mean():+.4f} +/- {ci:.4f} ({verdict}), B wins {int((d > 0).sum())}/{len(d)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=24, help="paired A/B seeds (default 24)")
    args = parser.parse_args()
    probe_geometry()
    probe_ab(args.seeds)


if __name__ == "__main__":
    main()
