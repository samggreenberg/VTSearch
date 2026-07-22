"""Text-sort seed scores for the Autopilot voting simulation.

The Autopilot flow the voting-iterations eval reproduces starts from a *text
sort*: the user types a query, the tool ranks every item by cosine similarity to
that query's embedding, and the first Good votes come from the top of that
ranking (where the positives cluster).  :func:`simulate_voting_iterations`
accepts that ranking as ``seed_scores`` — a ``{media_id: similarity}`` map — but
leaves *building* it to the caller, because the text a user would type is
dataset-specific.

This module builds that map from the evaluation queries already defined per
dataset in :mod:`vtscore.eval.config` (each :class:`~vtscore.eval.config.EvalQuery`
pairs the text a user would type with the ground-truth category it targets).
For each query it embeds the text into the media type's vector space via
:func:`vtscore.embedding.helpers.embed_text_query` and takes the cosine to every
media's embedding.  Embeddings are L2-normalised at ingest, so the cosine is a
dot product over the stacked matrix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import numpy as np

from vtscore.embedding.media_vectors import media_embedding


def _unit(vec: np.ndarray) -> np.ndarray:
    """Return *vec* L2-normalised (a zero vector is returned unchanged)."""
    import numpy as np  # noqa: PLC0415

    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def build_seed_scores(
    dataset_clips: dict[str, dict[int, dict[str, Any]]],
    *,
    media_type: str = "image",
    embedder_name: str = "siglip",
    categories: Optional[dict[str, list[str]]] = None,
) -> dict[str, dict[str, dict[int, float]]]:
    """Build ``{dataset: {category: {media_id: cosine}}}`` text-sort rankings.

    For every dataset in *dataset_clips* that has an entry in
    :data:`vtscore.eval.config.EVAL_DATASETS`, each of its ``EvalQuery`` texts is
    embedded and scored against every media, producing the per-category cosine
    ranking :func:`run_voting_iterations_eval` expects as ``seed_scores``.

    Args:
        dataset_clips: Loaded medias keyed by dataset id.
        media_type: Media type whose embedder space the query is embedded into
            (``"image"`` for the SigLIP image study).
        embedder_name: Registered embedder to embed the query with — must match
            the embedder the medias were embedded with (``"siglip"``).
        categories: Optional ``{dataset: [category, ...]}`` restriction; when a
            dataset is present only those categories' queries are embedded.

    Returns:
        Nested mapping suitable for ``run_voting_iterations_eval(seed_scores=…)``.
        Datasets with no eval-config entry, and queries whose text fails to
        embed, are simply omitted (the autopilot then seeds those from random
        known-good examples instead).
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.eval.config import EVAL_DATASETS  # noqa: PLC0415
    from vtscore.embedding.helpers import embed_text_query  # noqa: PLC0415

    result: dict[str, dict[str, dict[int, float]]] = {}
    for ds_name, clips in dataset_clips.items():
        info = EVAL_DATASETS.get(ds_name)
        if not info or not clips:
            continue
        wanted = set(categories[ds_name]) if categories and ds_name in categories else None

        ids = list(clips.keys())
        matrix = np.stack([_unit(media_embedding(clips[cid])) for cid in ids])  # (N, D), unit rows

        ds_out: dict[str, dict[int, float]] = {}
        for query in info["queries"]:
            cat = query.target_category
            if wanted is not None and cat not in wanted:
                continue
            qvec = embed_text_query(query.text, media_type, embedder_name=embedder_name)
            if qvec is None:
                continue
            cos = matrix @ _unit(np.asarray(qvec, dtype=np.float32))  # (N,)
            # Keep the strongest query per category if several target the same one.
            existing = ds_out.get(cat)
            scores = {ids[k]: float(cos[k]) for k in range(len(ids))}
            if existing is None or max(scores.values()) > max(existing.values()):
                ds_out[cat] = scores
        if ds_out:
            result[ds_name] = ds_out
    return result
