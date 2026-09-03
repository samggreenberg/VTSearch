"""External-query sorts of the active dataset: example media, and label files.

These helpers backed the ``/api/example-sort`` and ``/api/label-file-sort``
route handlers in ``vtsearch/routes/sorting.py`` (and, by cross-blueprint
reach-in, the three server-media example-sort routes in
``vtsearch/routes/media/server.py``).  None of it touches Flask or the
request context: every function takes plain paths, file objects and vectors,
reads the active dataset through :mod:`vtscore.state`, and returns results or
raises :class:`ValueError`.  So it belongs in the library tier, where the CLI
can reach it and tests can exercise it without a Flask client.  Same move,
same reason, as :mod:`vtscore.detectors.learned_sort`.

The routes are now request↔library glue: they materialise the upload into a
temp file, call in here, and translate a :class:`ValueError` into the HTTP
error envelope.

The primitives these compose over live next door:
:func:`vtscore.training.region_similarity.cosine_sort_with_boxes` scores a
media snapshot against a query vector, and
:mod:`vtscore.detectors.training` owns the train→threshold→score pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vtscore.media.embedder import MediaEmbedder


def cosine_sort_active(query_vec, *, role: str = "score", snap=None) -> tuple[list[dict], float]:
    """Sort every media in the active dataset by cosine similarity to *query_vec*.

    Returns ``(results, threshold)`` where *results* is a list of
    ``{"id": …, "similarity": …}`` dicts sorted descending, and
    *threshold* is the GMM-based boundary (rounded to 4 decimals).

    *role* selects which bound embedder the haystack is scored against (the
    v3 routing table, see :meth:`DatasetContext.routed_embedder`): ``"text"``
    for a text query, ``"score"`` (patch-else-text) for an example/cosine
    query.  *query_vec* must have been embedded by that same embedder.

    For datasets embedded with a patch-aware embedder (DINOv2, DINOv3,
    EUPE), each result also carries a ``best_region`` field containing the
    bounding box of the region that scored highest, in normalised
    image coordinates ``[x0, y0, x1, y1]``.  Single-vector embedders
    take a fast vectorised numpy path with no per-result box.

    Both paths live in :mod:`vtscore.training.region_similarity`.

    *snap* lets the caller thread in a medias snapshot it already took, so a
    single handler doesn't copy the full medias dict under ``_state_lock`` more
    than once per request; when ``None`` a fresh snapshot is taken.
    """
    from vtscore.state import snapshot_medias
    from vtscore.state.core import get_active_context
    from vtscore.training.region_similarity import cosine_sort_with_boxes
    from vtscore.training.thresholds import calculate_gmm_threshold

    ctx = get_active_context()
    embedder_name = ctx.routed_embedder(role)
    # Region vectors belong to the patch embedder; the per-region max-pool is
    # valid only when the query was scored against that same embedder.
    region_aware = embedder_name is not None and embedder_name == ctx.patch_embedder

    if snap is None:
        snap = snapshot_medias()
    results, sims_list = cosine_sort_with_boxes(snap, query_vec, embedder_name, region_aware=region_aware)
    threshold = calculate_gmm_threshold(sims_list)
    return results, round(threshold, 4)


def score_embedder_for_active(snap=None) -> tuple[MediaEmbedder | None, str | None]:
    """Return ``(embedder, embedder_name)`` for the active dataset's score embedder.

    The score embedder is the patch slot if bound, else the text slot (the v3
    routing table; see :meth:`DatasetContext.routed_embedder`).  Used to embed
    an example/label query so it shares the space the haystack is scored
    against.  A slot-less single-vector dataset falls back to the embedder
    resolved from the medias themselves and a ``None`` name (the matrix layer
    then reads the primary vector); for single-embedder datasets the two
    coincide.

    *snap* threads in an already-taken medias snapshot to avoid re-copying the
    medias dict under ``_state_lock``; when ``None`` a fresh snapshot is taken.
    """
    from vtscore.media import embedder_for_medias, get_embedder
    from vtscore.state import snapshot_medias
    from vtscore.state.core import get_active_context

    score_name = get_active_context().routed_embedder("score")
    if score_name is not None:
        try:
            return get_embedder(score_name), score_name
        except KeyError:
            pass
    if snap is None:
        snap = snapshot_medias()
    return embedder_for_medias(snap), score_name


def example_sort_from_paths(file_paths: list[Path]) -> tuple[list[dict], float]:
    """Embed one or more media files and sort all loaded medias by similarity.

    Returns ``(results_list, threshold)`` on success, or raises
    :class:`ValueError` when there are no example files, no medias loaded, no
    embedder for the dataset, or a file that the embedder cannot embed.

    Each file is embedded using the score embedder of the currently loaded
    dataset.  A single example sorts by cosine similarity to its vector;
    multiple examples sort against their centroid (the mean of the
    L2-normalised example vectors), so each example contributes equally
    regardless of its embedding norm.
    """
    import numpy as np

    from vtscore.media.embedder import media_from_path
    from vtscore.state import snapshot_medias

    if not file_paths:
        raise ValueError("No example files provided")

    snap = snapshot_medias()
    if not snap:
        raise ValueError("No medias loaded")

    # Embed the examples with the dataset's score embedder so the query shares
    # the space the haystack is scored against.
    emb, _score_name = score_embedder_for_active(snap)
    if emb is None:
        raise ValueError("No embedder available for loaded dataset")

    medias = [media_from_path(p) for p in file_paths]
    embeddings = []
    for path, media in zip(file_paths, medias, strict=True):
        vec = emb.embed_media(media)
        if vec is None:
            raise ValueError(f"Failed to embed media file: {path.name}")
        embeddings.append(np.asarray(vec, dtype=np.float32))

    if len(embeddings) == 1:
        query_vec = embeddings[0]
    else:
        normed = [v / n if (n := float(np.linalg.norm(v))) > 0 else v for v in embeddings]
        query_vec = np.mean(np.stack(normed), axis=0)

    results, threshold = cosine_sort_active(query_vec, snap=snap)

    # Stage-2 structural re-rank (a no-op for non-structural datasets): for a
    # SIFT/VLAD dataset, geometrically verify the VLAD shortlist against the
    # uploaded example's own local features.  The example is the template; any
    # crop was already applied to the file above, so it restricts the template.
    # Geometric verification needs a single template, so the multi-example
    # centroid path skips it and keeps the pure cosine ranking.
    if len(medias) == 1 and getattr(emb, "supports_geometric_verification", False):
        from vtscore.training.structural_similarity import maybe_structural_rerank_example

        example_features = emb.local_features_forward(medias[0])
        results, threshold = maybe_structural_rerank_example(
            results, threshold, snap, example_features, score_key="similarity"
        )

    return results, threshold


def apply_crop_or_keep(temp_path: Path, crop_params: dict | None) -> Path:
    """Apply *crop_params* to *temp_path* in-place when set; otherwise keep file.

    Resolves the target media type from the loaded dataset's first media
    item (the embedder is the same one we're about to use).  Writes the
    cropped bytes back to *temp_path* and returns it.
    """
    if not crop_params:
        return temp_path

    from vtscore.media.cropping import crop_file_bytes
    from vtscore.state import snapshot_medias

    snap = snapshot_medias()
    if not snap:
        return temp_path
    first_media = next(iter(snap.values()))
    media_type = first_media.get("media_type", "")

    cropped = crop_file_bytes(temp_path, media_type, crop_params)
    temp_path.write_bytes(cropped)
    return temp_path


def parse_label_file(fp) -> list[dict]:
    """Read *fp* as a JSON label file and return its ``labels`` list.

    *fp* is any binary file-like object (an upload stream, an open file).
    Raises :class:`ValueError` when the bytes are not valid UTF-8 JSON, or
    when the document carries no non-empty ``labels`` list; the caller maps
    that onto whatever error surface it owns.
    """
    try:
        label_data = json.loads(fp.read().decode("utf-8"))
    except Exception as exc:
        raise ValueError("Invalid label file format") from exc
    if not isinstance(label_data, dict):
        raise ValueError("Invalid label file format")
    labels = label_data.get("labels", [])
    if not labels:
        raise ValueError("No labels found in file")
    return labels


def embed_external_labels(labels: list[dict], emb) -> tuple[list, list[float], int, int]:
    """Embed every well-formed entry in *labels* using *emb*.

    Returns ``(X_list, y_list, loaded_count, skipped_count)``. Entries are
    skipped (not raised on) when the label is malformed, the path is missing
    or escapes the allowed directory, the file doesn't exist, or the
    embedder returns None.
    """
    import vtscore.security.path_validation as _paths
    from vtscore.media.embedder import media_from_path

    X_list: list = []
    y_list: list[float] = []
    loaded = 0
    skipped = 0
    file_base = _paths.get_file_access_base_dir()

    for entry in labels:
        label = entry.get("label")
        if label not in ("good", "bad"):
            skipped += 1
            continue

        raw_path = entry.get("path") or entry.get("file") or entry.get("filename")
        if not raw_path:
            skipped += 1
            continue

        try:
            # Embed the approved path, not the raw one: under confinement the
            # check anchors a relative path at the user's data dir while
            # ``Path(...)`` would anchor it at the process CWD.
            media_path = Path(_paths.confine_server_filepath(str(raw_path), file_base))
        except ValueError:
            skipped += 1
            continue
        if not media_path.exists():
            skipped += 1
            continue

        embedding = emb.embed_media(media_from_path(media_path))
        if embedding is None:
            skipped += 1
            continue

        X_list.append(embedding)
        y_list.append(1.0 if label == "good" else 0.0)
        loaded += 1

    return X_list, y_list, loaded, skipped


def train_and_score_active(
    X_list: list, y_list: list[float], embedder_name: str | None = None
) -> tuple[list[dict[str, Any]], float]:
    """Train an MLP on (X, y), then score every media in the active dataset.

    *embedder_name* is the embedder the external labels in *X_list* were
    embedded with; scoring sources the haystack vectors from the same embedder
    so the trained MLP and the scored vectors share one space.  ``None`` reads
    each media's primary vector.  The same name is handed to
    :func:`vtscore.detectors.training.train_and_threshold` so the safe-threshold
    GMM is fitted on exactly the score distribution returned here - including
    the region max-pool on a patch dataset, where results also gain a
    ``best_region`` box.
    """
    from vtscore.detectors.training import score_media_with_model, train_and_threshold
    from vtscore.state import snapshot_medias

    snap = snapshot_medias()
    model, threshold = train_and_threshold(X_list, y_list, snap=snap, embedder_name=embedder_name)
    return score_media_with_model(model, snap, embedder_name), threshold
