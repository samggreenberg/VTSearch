"""Eval runner - loads datasets and measures sorting quality.

Two evaluation modes:

1. **Text sort**: For each query, embed the text, rank all medias by cosine
   similarity to the query embedding, and measure how well medias of the
   target category float to the top (AP, P@k, R@k).

2. **Learned sort**: For each category, randomly split its medias into
   train/test.  Simulate votes (target = good, rest = bad) on the
   train set, run ``train_and_score``, and measure classification
   quality on the held-out test set.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import numpy as np

from vtscore.embedding.media_vectors import media_embedding
from vtscore.eval.config import EVAL_DATASETS, EvalQuery
from vtscore.eval.labels import evaluable_pool, media_is_positive
from vtscore.eval.metrics import (
    DatasetResult,
    LearnedSortMetrics,
    compute_binary_classification_metrics,
    compute_metrics,
)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


def _loaded_embedder_name(medias: dict[int, dict[str, Any]]) -> str:
    """The embedder the loaded *medias* actually carry vectors for.

    Read back off the medias rather than taken from the ``--embedder`` flag,
    because the flag is empty for a default-embedder run while the medias
    always name the concrete embedder that produced them.
    """
    from vtscore.embedding.media_vectors import media_embedder_names

    first = next(iter(medias.values()), {})
    names = media_embedder_names(first)
    return names[0] if names else ""


def _run_text_sort_query(
    query: EvalQuery,
    medias: dict[int, dict[str, Any]],
    media_type: str,
    enrich: bool = False,
    embedder_name: str = "",
) -> list[dict[str, Any]]:
    """Embed the query text and rank medias by cosine similarity.

    *embedder_name* must be the embedder the medias were embedded with: the
    query has to land in the same vector space as the media vectors it is
    compared against.  Omitting it falls back to the media type's *default*
    embedder, which silently scores a non-default ``--embedder`` run across
    two unrelated spaces and reports near-chance mAP.  The app threads the
    dataset's bound embedder through the same argument
    (``vtsearch/routes/sorting.py``).

    Returns a list of ``{"id": int, "similarity": float}`` sorted descending.
    """
    from vtscore.embedding.helpers import embed_text_query

    text_vec = embed_text_query(query.text, media_type, enrich=enrich, embedder_name=embedder_name)
    if text_vec is None:
        raise RuntimeError(f"Could not embed query {query.text!r} for media type {media_type}")

    results = []
    for media_id, media in medias.items():
        sim = _cosine_similarity(media_embedding(media), text_vec)
        results.append({"id": media_id, "similarity": sim})

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results


def eval_text_sort(
    medias: dict[int, dict[str, Any]],
    queries: list[EvalQuery],
    media_type: str,
    k_values: list[int] | None = None,
    enrich: bool = False,
    start_time: float | None = None,
    embedder_name: str = "",
) -> list:
    """Run text-sort evaluation for a list of queries.

    For each query, computes AP, P@k, and R@k treating medias whose
    ``"category"`` matches ``query.target_category`` as relevant.

    Args:
        medias: Loaded media dict (``{id: media_data}``).
        queries: List of :class:`EvalQuery` to evaluate.
        media_type: The media type string for embedding dispatch.
        k_values: Optional k values for P@k/R@k.
        enrich: If ``True``, use enriched (wrapper-averaged) text embeddings.
        start_time: Monotonic timestamp from the start of the eval run.
            When provided, each result records ``elapsed_seconds``.
        embedder_name: The embedder *medias* were embedded with, so the query
            lands in the same space. Empty means the media type's default.

    Returns:
        List of :class:`~vtscore.eval.metrics.QueryMetrics`.
    """
    from vtscore.eval.metrics import QueryMetrics

    results: list[QueryMetrics] = []
    for query in queries:
        # Excluded media are neither relevant nor irrelevant, so they must leave
        # the ranking too — left in, they would sit in it as false positives.
        pool = evaluable_pool(medias, query.target_category)
        ranked = _run_text_sort_query(query, pool, media_type, enrich=enrich, embedder_name=embedder_name)
        ranked_ids = [r["id"] for r in ranked]
        relevant_ids = {cid for cid, c in pool.items() if media_is_positive(c, query.target_category)}

        qm = compute_metrics(ranked_ids, relevant_ids, query.text, query.target_category, k_values)
        if start_time is not None:
            qm.elapsed_seconds = time.monotonic() - start_time
        results.append(qm)

    return results


def eval_learned_sort(
    medias: dict[int, dict[str, Any]],
    queries: list[EvalQuery],
    train_fraction: float = 0.5,
    seed: int = 42,
    calibrate_count: int = 2,
    calibration_fraction: float = 0.5,
    start_time: float | None = None,
    region_voting: bool = False,
) -> list[LearnedSortMetrics]:
    """Run learned-sort evaluation via simulated voting.

    For each query/category:
      1. Partition medias into target-category (positive) and others (negative).
      2. Randomly split both pools into train and test by ``train_fraction``.
      3. Build ``good_votes`` from train positives, ``bad_votes`` from train
         negatives.
      4. Call ``train_and_score`` on the full media set.
      5. Measure accuracy/precision/recall/F1 on the test set using the
         cross-calibrated threshold.

    Args:
        medias: Loaded media dict.
        queries: List of :class:`EvalQuery` (one per category to test).
        train_fraction: Fraction of medias to use for training (rest for test).
        seed: Random seed for reproducible splits.
        calibrate_count: Number of random Train/Calibrate splits for threshold
            calibration (default 2).
        calibration_fraction: Fraction of labelled data reserved for
            calibration in each split (default 0.5).
        start_time: Monotonic timestamp from the start of the eval run.
            When provided, each result records ``elapsed_seconds``.
        region_voting: When ``True``, each Good vote passes the media's
            ground-truth region box for the target category to
            ``train_and_score`` (the minimal box covering every annotated
            instance), so the positive is region-pooled instead of trained on
            the whole image.  Requires a patch dataset with stored ``regions``
            and ``patch_grid``; on any other dataset the boxes are absent and
            this is a no-op.

    Returns:
        List of :class:`LearnedSortMetrics`, one per query.
    """
    from vtscore.detectors.training import train_and_score
    from vtscore.eval.labels import region_box_for_category

    rng = np.random.RandomState(seed)
    results: list[LearnedSortMetrics] = []

    for query in queries:
        # Split medias into target vs. other, over the scorable pool only.
        pool = evaluable_pool(medias, query.target_category)
        target_ids = [cid for cid, c in pool.items() if media_is_positive(c, query.target_category)]
        other_ids = [cid for cid, c in pool.items() if not media_is_positive(c, query.target_category)]

        if len(target_ids) < 2 or len(other_ids) < 2:
            continue  # not enough data

        # Shuffle and split
        rng.shuffle(target_ids)
        rng.shuffle(other_ids)

        n_target_train = max(1, int(len(target_ids) * train_fraction))
        n_other_train = max(1, int(len(other_ids) * train_fraction))

        train_good = target_ids[:n_target_train]
        test_good = target_ids[n_target_train:]
        train_bad = other_ids[:n_other_train]
        test_bad = other_ids[n_other_train:]

        if not test_good or not test_bad:
            continue  # empty test set

        # Build vote dicts
        good_votes: dict[int, None] = {cid: None for cid in train_good}
        bad_votes: dict[int, None] = {cid: None for cid in train_bad}

        # When region voting, a Good vote carries the ground-truth box for the
        # target category (minimal box over all annotated instances); media
        # without an annotated box are simply omitted and train_and_score falls
        # back to their whole-image vector.
        vote_region_boxes: dict[int, tuple[float, float, float, float]] | None = None
        if region_voting:
            vote_region_boxes = {
                cid: box
                for cid in train_good
                if (box := region_box_for_category(medias[cid], query.target_category)) is not None
            }

        # Run train_and_score
        scored, threshold, _model = train_and_score(
            medias,
            good_votes,
            bad_votes,
            calibrate_count=calibrate_count,
            calibration_fraction=calibration_fraction,
            vote_region_boxes=vote_region_boxes,
        )

        # Evaluate on test set
        score_map = {r["id"]: r["score"] for r in scored}
        test_ids = test_good + test_bad
        predictions = [1 if score_map.get(cid, 0) >= threshold else 0 for cid in test_ids]
        labels = [1] * len(test_good) + [0] * len(test_bad)

        acc, prec, rec, f1 = compute_binary_classification_metrics(predictions, labels)

        elapsed = (time.monotonic() - start_time) if start_time is not None else 0.0
        results.append(
            LearnedSortMetrics(
                accuracy=acc,
                precision=prec,
                recall=rec,
                f1=f1,
                num_train=len(train_good) + len(train_bad),
                num_test=len(test_ids),
                target_category=query.target_category,
                elapsed_seconds=elapsed,
            )
        )

    return results


def run_eval(
    dataset_ids: list[str] | None = None,
    mode: str = "both",
    k_values: list[int] | None = None,
    train_fraction: float = 0.5,
    seed: int = 42,
    enrich: bool = False,
    calibrate_count: int = 2,
    calibration_fraction: float = 0.5,
    embedder_name: str = "",
    region_voting: bool = False,
) -> list[DatasetResult]:
    """Run evaluation on one or more eval datasets.

    This is the main entry point.  It loads the demo dataset (downloading
    and embedding if needed), then runs text-sort and/or learned-sort
    evaluation.

    Args:
        dataset_ids: Which eval datasets to run.  ``None`` means all.
        mode: ``"text"`` for text-sort only, ``"learned"`` for learned-sort
            only, ``"both"`` for both.
        k_values: k values for P@k/R@k.
        train_fraction: Train/test split ratio for learned-sort.
        seed: Random seed.
        enrich: If ``True``, use enriched (wrapper-averaged) text embeddings
            for text-sort evaluation.
        calibrate_count: Number of random Train/Calibrate splits for threshold
            calibration (default 2).
        calibration_fraction: Fraction of labelled data reserved for
            calibration in each split (default 0.5).
        embedder_name: Optional embedder to build each demo dataset with
            (empty = the media type's default).  Pass a patch embedder
            (e.g. ``"dinov3_patch"``) to make ``region_voting`` meaningful -
            only patch embedders produce the ``patch_grid`` region pooling
            needs.
        region_voting: When ``True``, learned-sort Good votes are region-pooled
            from each media's ground-truth box (see :func:`eval_learned_sort`).
            Only affects patch datasets with stored ``regions`` (Visual Genome).

    Returns:
        List of :class:`DatasetResult`, one per evaluated dataset.
    """
    from vtscore.datasets.config import DEMO_DATASETS
    from vtscore.datasets.loader import load_demo_dataset

    if dataset_ids is None:
        dataset_ids = list(EVAL_DATASETS.keys())

    start_time = time.monotonic()
    all_results: list[DatasetResult] = []

    for ds_id in dataset_ids:
        if ds_id not in EVAL_DATASETS:
            print(f"WARNING: unknown eval dataset {ds_id!r}, skipping", file=sys.stderr)
            continue

        eval_cfg = EVAL_DATASETS[ds_id]
        demo_id = eval_cfg["demo_dataset"]

        if demo_id not in DEMO_DATASETS:
            print(f"WARNING: demo dataset {demo_id!r} not found, skipping {ds_id!r}", file=sys.stderr)
            continue

        demo_info = DEMO_DATASETS[demo_id]
        media_type = demo_info.get("media_type", "audio")

        print(f"\n{'=' * 60}")
        print(f"Evaluating: {ds_id}  (media_type={media_type})")
        print(f"{'=' * 60}")

        # Load the demo dataset into a fresh medias dict
        medias: dict[int, dict] = {}
        try:
            load_demo_dataset(demo_id, medias, embedder_name=embedder_name)
        except Exception as e:
            print(f"ERROR loading dataset {demo_id}: {e}", file=sys.stderr)
            continue

        print(f"Loaded {len(medias)} medias across categories: ", end="")
        categories = sorted({c.get("category", "?") for c in medias.values()})
        print(", ".join(categories))

        queries = eval_cfg["queries"]
        ds_result = DatasetResult(dataset_id=ds_id, media_type=media_type)

        # --- Text sort ---
        if mode in ("text", "both"):
            print(f"\n--- Text Sort Evaluation ({len(queries)} queries) ---")
            # Score the query in the *dataset's* space, not the media type's
            # default — see _run_text_sort_query.
            text_results = eval_text_sort(
                medias,
                queries,
                media_type,
                k_values,
                enrich=enrich,
                start_time=start_time,
                embedder_name=_loaded_embedder_name(medias),
            )
            ds_result.text_sort = text_results

            for qm in text_results:
                p5 = qm.precision_at_k.get(5, 0)
                p10 = qm.precision_at_k.get(10, 0)
                print(
                    f"  [{qm.target_category:20s}] AP={qm.average_precision:.3f}  "
                    f"P@5={p5:.2f}  P@10={p10:.2f}  "
                    f"({qm.num_relevant} relevant / {qm.num_total} total)  "
                    f"t={qm.elapsed_seconds:.1f}s"
                )
            print(f"  mAP = {ds_result.mean_average_precision:.4f}")

        # --- Learned sort ---
        if mode in ("learned", "both"):
            print(f"\n--- Learned Sort Evaluation ({len(queries)} categories) ---")
            learned_results = eval_learned_sort(
                medias,
                queries,
                train_fraction,
                seed,
                calibrate_count=calibrate_count,
                calibration_fraction=calibration_fraction,
                start_time=start_time,
                region_voting=region_voting,
            )
            ds_result.learned_sort = learned_results

            for lm in learned_results:
                print(
                    f"  [{lm.target_category:20s}] Acc={lm.accuracy:.3f}  "
                    f"P={lm.precision:.3f}  R={lm.recall:.3f}  F1={lm.f1:.3f}  "
                    f"(train={lm.num_train}, test={lm.num_test})  "
                    f"t={lm.elapsed_seconds:.1f}s"
                )
            print(f"  Mean F1 = {ds_result.mean_learned_f1:.4f}")

        all_results.append(ds_result)

    return all_results


def format_results_json(results: list[DatasetResult]) -> str:
    """Serialise a list of :class:`DatasetResult` to a JSON string."""
    return json.dumps([r.to_dict() for r in results], indent=2)
