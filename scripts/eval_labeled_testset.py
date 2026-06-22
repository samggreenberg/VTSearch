#!/usr/bin/env python3
"""Evaluate models against a hand-labeled test set (held-out, binary good/bad).

You have a test set with binary ``good``/``bad`` labels and want to measure how
well various "models" rank/classify it.  This script scores the test set
**held-out**: each model is taken as-is (a saved detector is re-derived from its
own fixed labelset; a text query is embedded as-is) and the test labels are used
*only* as ground truth for metrics — never for training or threshold tuning.

A unit of evaluation is a **scorer** that produces one score per test item::

    scorer = (method ∈ {detector, text-query}, embedder, detector-name | query)

For each scorer the script reports ranking metrics (Average Precision, P@k, R@k)
and — for detectors — binary classification metrics (accuracy / precision /
recall / F1) at the model's own cross-calibrated threshold.

Comparing embedders falls out naturally: the test set is re-embedded once per
``--embedders`` value, and a saved detector's labelset is re-embedded into the
same space by :func:`resolve_or_train_detector`.

Usage::

    # One detector, the test set's own embedder
    python scripts/eval_labeled_testset.py \
        --labels my_test.json --dataset test.pkl --detectors stop_signv2

    # Compare two embedders and a text query on an image folder
    python scripts/eval_labeled_testset.py \
        --labels my_test.json --folder /data/test_imgs --media-type image \
        --detectors stop_signv2 --embedders siglip clip \
        --methods detector text-query --query "a stop sign"

    # Write a machine-readable results file
    python scripts/eval_labeled_testset.py \
        --labels my_test.csv --dataset test.pkl --detectors stop_signv2 \
        --output results.json

Notes / caveats:

* Re-embedding under a different embedder needs the raw media on disk (a
  ``--folder`` of files, or a full pickle with a resolvable companion dir).
  Items that cannot be re-embedded are dropped with a warning.
* Held-out integrity: a detector's threshold is cross-calibrated on its *own*
  labelset, independent of the test set, and is applied as-is.  Test items whose
  md5 also appears in the detector's labelset are training labels, not test
  items, so they are dropped from that detector's metrics by default
  (``--keep-overlap`` to retain).
* ``safe_thresholds`` (if enabled in settings) blends in a GMM over the test
  score distribution; that introduces no *label* leakage but makes the threshold
  mildly test-distribution-dependent.  The active value is printed in the header.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Ensure the repo root is importable when run as a bare script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (mirrors eval/runner.py)."""
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / norm)


def _load_labels(path: Path) -> tuple[dict[str, int], dict[str, tuple[float, float, float, float]]]:
    """Return ``(md5 -> 1/0, md5 -> region_box)`` from a .json or .csv label file.

    Reuses the maintained server label importers so the schema matches the
    rest of the app.  Entries whose ``label`` is neither ``good`` nor ``bad``
    are dropped.  ``region_box`` (normalised ``x0,y0,x1,y1``) is collected for
    ``good`` entries that carry one; it round-trips through JSON only (CSV has
    no region column), so ``md5_to_box`` is empty for CSV inputs.
    """
    suffix = path.suffix.lower()
    if suffix == ".json":
        from vtscore.labels.importers.server_json_file import LABEL_IMPORTER
    elif suffix == ".csv":
        from vtscore.labels.importers.server_csv_file import LABEL_IMPORTER
    else:
        raise SystemExit(f"--labels must be a .json or .csv file, got {path.suffix!r}")

    entries = LABEL_IMPORTER.run_cli({"filepath": str(path)})
    md5_to_label: dict[str, int] = {}
    md5_to_box: dict[str, tuple[float, float, float, float]] = {}
    for entry in entries:
        label = entry.get("label", "")
        md5 = entry.get("md5", "")
        if not (md5 and label in ("good", "bad")):
            continue
        md5_to_label[md5] = 1 if label == "good" else 0
        rb = entry.get("region_box")
        if label == "good" and isinstance(rb, (list, tuple)) and len(rb) == 4:
            md5_to_box[md5] = tuple(float(v) for v in rb)  # type: ignore[assignment]
    if not md5_to_label:
        raise SystemExit(f"No usable good/bad labels found in {path}")
    return md5_to_label, md5_to_box


def _load_test_medias(args: argparse.Namespace) -> tuple[dict[int, dict], str]:
    """Load the test set into a medias dict; return (medias, media_type)."""
    from vtscore.datasets.loader import load_dataset_from_folder, load_dataset_from_pickle

    medias: dict[int, dict] = {}
    if args.dataset:
        # Full load so media_path/companion files are available for re-embedding.
        load_dataset_from_pickle(Path(args.dataset), medias, thin=False)
        if not medias:
            raise SystemExit(f"No medias loaded from {args.dataset}")
        media_type = next(iter(medias.values())).get("media_type", "")
    else:
        media_type = args.media_type
        # thin=True keeps media_path for re-embedding without holding bytes.
        load_dataset_from_folder(Path(args.folder), media_type, medias, thin=True)
        if not medias:
            raise SystemExit(f"No {media_type} medias found under {args.folder}")
    return medias, media_type


def _default_embedders(medias: dict[int, dict], media_type: str) -> list[str]:
    """Embedder to use when --embedders is not given.

    Prefer the embedder the dataset was stored with; fall back to the media
    type's default embedder.
    """
    from vtscore.media import embedders_for_type

    stored = next(iter(medias.values())).get("embedder", "") or ""
    if stored:
        return [stored]
    avail = embedders_for_type(media_type)
    if not avail:
        raise SystemExit(f"No embedder registered for media type {media_type!r}")
    return [avail[0].name]


def _embed_test_set(embedder_name: str, medias: dict[int, dict], media_type: str) -> dict[int, dict] | None:
    """Re-embed the test set under *embedder_name*; return a scored snapshot.

    Returns ``None`` (with a printed reason) when the embedder is incompatible
    with the media type.  Items that fail to embed are dropped with a warning.
    """
    from vtscore.media import get_embedder

    try:
        embedder = get_embedder(embedder_name)
    except KeyError:
        print(f"  WARNING: unknown embedder {embedder_name!r}; skipping", file=sys.stderr)
        return None
    if embedder.media_type_id != media_type:
        print(
            f"  WARNING: embedder {embedder_name!r} is for media type "
            f"{embedder.media_type_id!r}, not {media_type!r}; skipping",
            file=sys.stderr,
        )
        return None

    # Reuse stored vectors where the item is already embedded with this
    # embedder; only items embedded with a *different* (or no) embedder need a
    # fresh forward pass.  This avoids touching the raw media files in the
    # common single-embedder case and only re-embeds when comparing embedders.
    snap: dict[int, dict] = {}
    to_embed: dict[int, dict] = {}
    for media_id, media in medias.items():
        if media.get("embedder") == embedder_name and media.get("embedding") is not None:
            item = dict(media)
            item["embedding"] = np.asarray(media["embedding"], dtype=np.float32)
            snap[media_id] = item
        else:
            to_embed[media_id] = media

    failed = 0
    if to_embed:
        embedder.load_models()
        vectors = embedder.embed_medias(to_embed)
        for media_id, vec in vectors.items():
            if vec is None:
                failed += 1
                continue
            item = dict(medias[media_id])
            item["embedding"] = vec
            item["embedder"] = embedder_name
            snap[media_id] = item
    if failed:
        print(
            f"  WARNING: {failed}/{len(medias)} item(s) failed to embed under "
            f"{embedder_name!r} (raw media file unavailable?); dropped from metrics",
            file=sys.stderr,
        )
    return snap


def _embed_with_regions(embedder_name: str, medias: dict[int, dict], media_type: str) -> dict[int, dict] | None:
    """Re-embed the test set under a patch embedder, attaching ``patch_regions``.

    Unlike :func:`_embed_test_set`, this always recomputes from the raw media
    (the HAC region tree is never persisted in pickles) and requires a
    patch-region embedder.  Returns ``None`` (with a printed reason) when the
    embedder is unusable; items that fail to produce regions are dropped.
    """
    from vtscore.datasets.stages.embedding import embed_missing
    from vtscore.media import get_embedder

    try:
        embedder = get_embedder(embedder_name)
    except KeyError:
        print(f"  WARNING: unknown embedder {embedder_name!r}; skipping", file=sys.stderr)
        return None
    if embedder.media_type_id != media_type:
        print(
            f"  WARNING: embedder {embedder_name!r} is for media type "
            f"{embedder.media_type_id!r}, not {media_type!r}; skipping",
            file=sys.stderr,
        )
        return None
    if not getattr(embedder, "supports_patch_regions", False):
        print(
            f"  WARNING: embedder {embedder_name!r} has no patch regions; "
            f"region-detector eval needs a patch embedder (dinov2_patch, "
            f"dinov3_patch, eupe_patch, face); skipping",
            file=sys.stderr,
        )
        return None

    # Fresh copies with embedding cleared so embed_missing recomputes the CLS
    # vector AND attaches patch_regions/patch_grid under this embedder.
    snap: dict[int, dict] = {}
    for media_id, media in medias.items():
        item = dict(media)
        item["embedding"] = None
        item.pop("patch_regions", None)
        item.pop("patch_grid", None)
        snap[media_id] = item

    embed_missing(snap, embedder_name=embedder_name)

    dropped = [mid for mid, m in snap.items() if m.get("patch_regions") is None]
    for mid in dropped:
        del snap[mid]
    if dropped:
        print(
            f"  WARNING: {len(dropped)}/{len(medias)} item(s) produced no patch "
            f"regions under {embedder_name!r} (raw media unavailable?); dropped",
            file=sys.stderr,
        )
    for m in snap.values():
        m["embedder"] = embedder_name
    return snap


def _score_with_mlp(mlp: Any, snap: dict[int, dict], ids: list[int]) -> dict[int, float]:
    """Return ``id -> score`` in [0, 1] (higher = more 'good')."""
    import torch

    from vtscore.utils.scores import sigmoid_to_finite_scores

    X = torch.from_numpy(np.stack([snap[i]["embedding"] for i in ids]).astype(np.float32))
    with torch.no_grad():
        X = X.to(next(mlp.parameters()).device)
        scores = sigmoid_to_finite_scores(mlp(X))
    return dict(zip(ids, scores))


def _ranking_record(
    scorer: str,
    score_map: dict[int, float],
    snap: dict[int, dict],
    md5_to_label: dict[str, int],
    k_values: list[int],
) -> dict[str, Any]:
    """Compute AP / P@k / R@k for a score map against the good/bad labels."""
    from vtscore.eval.metrics import compute_metrics

    # Only items that are both in the snapshot and labeled count as test items.
    test_ids = [i for i in score_map if snap[i]["md5"] in md5_to_label]
    ranked_ids = sorted(test_ids, key=lambda i: score_map[i], reverse=True)
    relevant_ids = {i for i in test_ids if md5_to_label[snap[i]["md5"]] == 1}

    qm = compute_metrics(ranked_ids, relevant_ids, scorer, "good", k_values)
    return {
        "scorer": scorer,
        "AP": qm.average_precision,
        "precision_at_k": qm.precision_at_k,
        "recall_at_k": qm.recall_at_k,
        "num_relevant": qm.num_relevant,
        "num_total": qm.num_total,
        "binary": None,
    }


def _binary_metrics(
    score_map: dict[int, float],
    threshold: float,
    snap: dict[int, dict],
    md5_to_label: dict[str, int],
) -> dict[str, float]:
    """Accuracy/precision/recall/F1 at *threshold* over labeled test items."""
    from vtscore.eval.metrics import compute_binary_classification_metrics

    test_ids = [i for i in score_map if snap[i]["md5"] in md5_to_label]
    preds = [1 if score_map[i] >= threshold else 0 for i in test_ids]
    labels = [md5_to_label[snap[i]["md5"]] for i in test_ids]
    acc, prec, rec, f1 = compute_binary_classification_metrics(preds, labels)
    return {"threshold": threshold, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def _eval_detector(
    det_name: str,
    snap: dict[int, dict],
    media_type: str,
    md5_to_label: dict[str, int],
    k_values: list[int],
    keep_overlap: bool,
    embedder_name: str,
) -> dict[str, Any] | None:
    """Score one saved detector held-out against the test set."""
    from vtscore.detectors.model_loading import resolve_or_train_detector
    from vtscore.detectors.store import _detector_path, _read_detector

    det_data = _read_detector(_detector_path(det_name))
    if det_data is None:
        print(f"  WARNING: detector {det_name!r} not found; skipping", file=sys.stderr)
        return None
    det_media_type = det_data.get("media_type", media_type)
    if det_media_type != media_type:
        print(
            f"  WARNING: detector {det_name!r} is for {det_media_type!r}, not {media_type!r}; skipping",
            file=sys.stderr,
        )
        return None

    # An item that is part of the detector's own labelset is a training label,
    # not a held-out test item; drop it from this detector's metrics by default.
    labelset_md5s = {e.get("md5", "") for e in det_data.get("labelset", {}).get("labels", [])}
    overlap_ids = [i for i, m in snap.items() if m["md5"] in labelset_md5s and m["md5"] in md5_to_label]
    if overlap_ids:
        msg = f"  WARNING: {len(overlap_ids)} test item(s) are also in {det_name!r}'s labelset (train/test overlap)"
        if keep_overlap:
            print(msg + "; KEPT in metrics (--keep-overlap)", file=sys.stderr)
        else:
            print(msg + "; dropped from metrics", file=sys.stderr)

    mlp, threshold, diag = resolve_or_train_detector(det_name, det_data, media_type, snap)
    if mlp is None:
        print(f"  WARNING: could not train detector {det_name!r}: {diag}", file=sys.stderr)
        return None

    score_ids = list(snap.keys())
    if not keep_overlap:
        drop = set(overlap_ids)
        score_ids = [i for i in score_ids if i not in drop]
    score_map = _score_with_mlp(mlp, snap, score_ids)

    scorer = f"detector:{det_name}@{embedder_name}"
    record = _ranking_record(scorer, score_map, snap, md5_to_label, k_values)
    record["binary"] = _binary_metrics(score_map, threshold, snap, md5_to_label)
    record["method"] = "detector"
    record["embedder"] = embedder_name
    record["model"] = det_name
    return record


def _eval_text_query(
    query: str,
    snap: dict[int, dict],
    media_type: str,
    md5_to_label: dict[str, int],
    k_values: list[int],
    embedder_name: str,
) -> dict[str, Any] | None:
    """Rank the test set by cosine similarity to a text query (ranking only)."""
    from vtscore.embedding.helpers import embed_text_query

    tvec = embed_text_query(query, media_type, embedder_name=embedder_name)
    if tvec is None:
        print(
            f"  WARNING: embedder {embedder_name!r} cannot embed text; skipping text-query",
            file=sys.stderr,
        )
        return None

    score_map = {i: _cosine_similarity(m["embedding"], tvec) for i, m in snap.items()}
    scorer = f"text:{query!r}@{embedder_name}"
    record = _ranking_record(scorer, score_map, snap, md5_to_label, k_values)
    record["method"] = "text-query"
    record["embedder"] = embedder_name
    record["model"] = query
    return record


def _eval_region_detector(
    det_name: str,
    snap: dict[int, dict],
    media_type: str,
    md5_to_label: dict[str, int],
    md5_to_box: dict[str, tuple[float, float, float, float]],
    k_values: list[int],
    iou_thresholds: tuple[float, ...],
    keep_overlap: bool,
    embedder_name: str,
) -> dict[str, Any] | None:
    """Score a region detector held-out and report localization (CorLoc/mIoU).

    Uses the region-aware ``labelset_train_and_score`` path (NOT
    ``resolve_or_train_detector``, which drops ``region_box`` on md5 matches).
    *snap* must carry ``patch_regions`` (see :func:`_embed_with_regions`).
    """
    from vtscore.config import CoreConfig
    from vtscore.datasets.labelset import LabelSet
    from vtscore.detectors.labelset_training import labelset_train_and_score
    from vtscore.detectors.store import _detector_path, _read_detector
    from vtscore.eval.metrics import compute_localization_metrics
    from vtscore.state.core import DetectorContext

    det_data = _read_detector(_detector_path(det_name))
    if det_data is None:
        print(f"  WARNING: detector {det_name!r} not found; skipping", file=sys.stderr)
        return None
    det_media_type = det_data.get("media_type", media_type)
    if det_media_type != media_type:
        print(
            f"  WARNING: detector {det_name!r} is for {det_media_type!r}, not {media_type!r}; skipping",
            file=sys.stderr,
        )
        return None

    labelset_md5s = {e.get("md5", "") for e in det_data.get("labelset", {}).get("labels", [])}
    overlap_ids = {i for i, m in snap.items() if m["md5"] in labelset_md5s and m["md5"] in md5_to_label}
    if overlap_ids:
        msg = f"  WARNING: {len(overlap_ids)} test item(s) are also in {det_name!r}'s labelset (train/test overlap)"
        print(
            msg + ("; KEPT in metrics (--keep-overlap)" if keep_overlap else "; dropped from metrics"), file=sys.stderr
        )

    cfg = CoreConfig.from_settings()
    labelset = LabelSet.from_dict(det_data.get("labelset", {}))
    det_ctx = DetectorContext(detector_id=det_name, media_type=media_type, embedder=embedder_name)
    results, threshold, model = labelset_train_and_score(
        det_ctx,
        labelset,
        media_type=media_type,
        clips_dict=snap,
        inclusion_value=cfg.inclusion,
        safe_thresholds=cfg.safe_thresholds,
        calibrate_count=cfg.calibrate_count,
        calibration_fraction=cfg.calibration_fraction,
    )
    if model is None:
        print(f"  WARNING: could not train region detector {det_name!r} from its labelset", file=sys.stderr)
        return None

    score_map = {r["id"]: r["score"] for r in results}
    box_map = {r["id"]: tuple(r["best_region"]) for r in results if r.get("best_region") is not None}
    if not keep_overlap:
        score_map = {i: s for i, s in score_map.items() if i not in overlap_ids}

    scorer = f"region:{det_name}@{embedder_name}"
    record = _ranking_record(scorer, score_map, snap, md5_to_label, k_values)
    record["binary"] = _binary_metrics(score_map, threshold, snap, md5_to_label)

    # Localization: one (predicted best-region box, GT box) per GT-good item
    # that carries a box. None prediction (no region emitted) counts as a miss.
    pairs: list[tuple[tuple[float, float, float, float] | None, tuple[float, float, float, float]]] = []
    for i in score_map:
        md5 = snap[i]["md5"]
        if md5_to_label.get(md5) == 1 and md5 in md5_to_box:
            pairs.append((box_map.get(i), md5_to_box[md5]))
    loc = compute_localization_metrics(pairs, iou_thresholds=iou_thresholds)
    record["localization"] = {
        "mean_iou": loc.mean_iou,
        "corloc": loc.corloc,
        "num_localizable": loc.num_localizable,
    }
    record["method"] = "region-detector"
    record["embedder"] = embedder_name
    record["model"] = det_name
    return record


def _print_table(records: list[dict[str, Any]], k_values: list[int], iou_thresholds: tuple[float, ...]) -> None:
    """Print a fixed-width metrics table to stdout."""

    def fmt(x: float | None) -> str:
        return "  -  " if x is None else f"{x:.3f}"

    k_cols = [f"P@{k}" for k in k_values] + [f"R@{k}" for k in k_values]
    loc_cols = [f"CorLoc@{t}" for t in iou_thresholds] + ["mIoU"]
    header = ["method", "embedder", "model", "AP", *k_cols, "acc", "prec", "rec", "F1", *loc_cols, "n_rel/n_tot"]
    widths = [15, 14, 22, 6] + [6] * len(k_cols) + [6, 6, 6, 6] + [10] * len(iou_thresholds) + [6, 12]

    def row(cells: list[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths))

    print(row(header))
    print(row(["-" * w for w in widths]))
    for r in records:
        b = r.get("binary")
        loc = r.get("localization")
        cells = [
            str(r.get("method", "")),
            str(r.get("embedder", "")),
            str(r.get("model", ""))[:22],
            fmt(r["AP"]),
            *[fmt(r["precision_at_k"].get(k)) for k in k_values],
            *[fmt(r["recall_at_k"].get(k)) for k in k_values],
            fmt(b["accuracy"] if b else None),
            fmt(b["precision"] if b else None),
            fmt(b["recall"] if b else None),
            fmt(b["f1"] if b else None),
            *[fmt(loc["corloc"].get(t) if loc else None) for t in iou_thresholds],
            fmt(loc["mean_iou"] if loc else None),
            f"{r['num_relevant']}/{r['num_total']}",
        ]
        print(row(cells))


def _write_output(
    path: Path, records: list[dict[str, Any]], k_values: list[int], iou_thresholds: tuple[float, ...]
) -> None:
    """Write results to .json or .csv."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    elif suffix == ".csv":
        import csv

        k_cols = [f"P@{k}" for k in k_values] + [f"R@{k}" for k in k_values]
        loc_cols = [f"CorLoc@{t}" for t in iou_thresholds] + ["mean_iou", "num_localizable"]
        fieldnames = [
            "method",
            "embedder",
            "model",
            "AP",
            *k_cols,
            "accuracy",
            "precision",
            "recall",
            "f1",
            *loc_cols,
            "num_relevant",
            "num_total",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                b = r.get("binary") or {}
                loc = r.get("localization") or {}
                rowd = {
                    "method": r.get("method", ""),
                    "embedder": r.get("embedder", ""),
                    "model": r.get("model", ""),
                    "AP": round(r["AP"], 4),
                    **{f"P@{k}": round(r["precision_at_k"].get(k, 0.0), 4) for k in k_values},
                    **{f"R@{k}": round(r["recall_at_k"].get(k, 0.0), 4) for k in k_values},
                    "accuracy": round(b["accuracy"], 4) if b else "",
                    "precision": round(b["precision"], 4) if b else "",
                    "recall": round(b["recall"], 4) if b else "",
                    "f1": round(b["f1"], 4) if b else "",
                    **{f"CorLoc@{t}": (round(loc["corloc"].get(t, 0.0), 4) if loc else "") for t in iou_thresholds},
                    "mean_iou": round(loc["mean_iou"], 4) if loc else "",
                    "num_localizable": loc.get("num_localizable", "") if loc else "",
                    "num_relevant": r["num_relevant"],
                    "num_total": r["num_total"],
                }
                writer.writerow(rowd)
    else:
        raise SystemExit(f"--output must be .json or .csv, got {path.suffix!r}")
    print(f"\nWrote {len(records)} result(s) to {path}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", required=True, help="Label file (.json or .csv) with md5/label good|bad")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", help="Test set as a full .pkl dataset")
    src.add_argument("--folder", help="Test set as a folder of media files (requires --media-type)")
    parser.add_argument("--media-type", help="Media type for --folder (audio|image|text|video|document)")
    parser.add_argument("--detectors", nargs="*", default=[], help="Saved detector name(s) to evaluate")
    parser.add_argument("--embedders", nargs="*", default=[], help="Embedder name(s); default: test set's own")
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["detector"],
        choices=["detector", "text-query", "region-detector"],
        help="Which scoring methods to run (default: detector)",
    )
    parser.add_argument("--query", help="Text query (required if 'text-query' in --methods)")
    parser.add_argument("--k", nargs="*", type=int, default=[5, 10, 20], help="k values for P@k/R@k")
    parser.add_argument(
        "--iou-thresholds",
        nargs="*",
        type=float,
        default=[0.3, 0.5, 0.7],
        help="IoU thresholds for region-detector CorLoc (default: 0.3 0.5 0.7)",
    )
    parser.add_argument("--output", help="Optional results file (.json or .csv)")
    parser.add_argument(
        "--keep-overlap",
        action="store_true",
        help="Keep test items that also appear in a detector's labelset (default: drop them)",
    )
    args = parser.parse_args(argv)

    if args.folder and not args.media_type:
        parser.error("--folder requires --media-type")
    if "text-query" in args.methods and not args.query:
        parser.error("--query is required when 'text-query' is in --methods")
    if ("detector" in args.methods or "region-detector" in args.methods) and not args.detectors:
        parser.error("--detectors is required when 'detector'/'region-detector' is in --methods")
    iou_thresholds = tuple(args.iou_thresholds)

    from vtscore.embedding import initialize_models
    from vtsearch.shim import register_app_config_builder

    initialize_models()  # configure torch + register media types/embedders
    # Back CoreConfig.from_settings() with the app settings layer so the
    # detector store (detectors_dir), thresholds, and inclusion read from
    # data/settings.json exactly as the running app would.
    register_app_config_builder()

    md5_to_label, md5_to_box = _load_labels(Path(args.labels))
    medias, media_type = _load_test_medias(args)
    test_md5s = {m["md5"] for m in medias.values()}
    matched = len(test_md5s & set(md5_to_label))

    embedders = args.embedders or _default_embedders(medias, media_type)

    from vtscore.config import CoreConfig

    safe_thresholds = CoreConfig.from_settings().safe_thresholds

    print(
        f"Test set: {len(medias)} {media_type} item(s); labels matched: {matched}/{len(md5_to_label)}", file=sys.stderr
    )
    print(
        f"Embedders: {', '.join(embedders)} | methods: {', '.join(args.methods)} | safe_thresholds={safe_thresholds}",
        file=sys.stderr,
    )
    if matched == 0:
        raise SystemExit("No labels matched any test-set media by md5; nothing to evaluate")

    whole_image = "detector" in args.methods or "text-query" in args.methods
    records: list[dict[str, Any]] = []
    for embedder_name in embedders:
        if whole_image:
            print(f"\n=== Embedding test set with {embedder_name!r} ===", file=sys.stderr)
            snap = _embed_test_set(embedder_name, medias, media_type)
            if snap:
                if "detector" in args.methods:
                    for det_name in args.detectors:
                        rec = _eval_detector(
                            det_name, snap, media_type, md5_to_label, args.k, args.keep_overlap, embedder_name
                        )
                        if rec:
                            records.append(rec)
                if "text-query" in args.methods:
                    rec = _eval_text_query(args.query, snap, media_type, md5_to_label, args.k, embedder_name)
                    if rec:
                        records.append(rec)
        if "region-detector" in args.methods:
            print(f"\n=== Embedding test set with regions under {embedder_name!r} ===", file=sys.stderr)
            region_snap = _embed_with_regions(embedder_name, medias, media_type)
            if region_snap:
                for det_name in args.detectors:
                    rec = _eval_region_detector(
                        det_name,
                        region_snap,
                        media_type,
                        md5_to_label,
                        md5_to_box,
                        args.k,
                        iou_thresholds,
                        args.keep_overlap,
                        embedder_name,
                    )
                    if rec:
                        records.append(rec)

    if not records:
        raise SystemExit("No scorers produced results (see warnings above)")

    print()
    _print_table(records, args.k, iou_thresholds)
    if args.output:
        _write_output(Path(args.output), records, args.k, iou_thresholds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
