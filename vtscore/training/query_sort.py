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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import numpy as np

    from vtscore.media.embedder import MediaEmbedder
    from vtscore.training.thresholds import TextSortCuts


@dataclass(frozen=True, eq=False)
class SortLine:
    """A text or example sort's display line, kept so a balance change can redraw it (#4760).

    A cosine sort's ranking does not depend on the balance, but its display line
    does: a typed query's is the count line at beta 1 or below (#4603), the
    Goods' centroid's is the count line at every balance (#4732), and a tiled
    structural example's rises at the precision end (#4479).  The sort cache
    keeps this beside the ranking, so ``GET /api/sort/line`` redraws the line at
    the balance set now instead of re-scoring the haystack, the move
    :class:`~vtscore.detectors.centroid_head.CentroidLine` makes for a centroid
    detector.  The acquisition cut does not move with the balance, so it is not
    here.

    *rule* names what draws the line: ``"text"``, a typed query's
    (:func:`~vtscore.training.thresholds.text_sort_cuts`); ``"centroid"``, an
    example's or several examples' centroid's
    (:func:`~vtscore.detectors.centroid_head.centroid_cut`); ``"structural"``, a
    geometrically verified example sort's inlier gate.  *scores* are the cosines
    the first two are drawn on, as float64 in the order the sort drew on them
    (the mixture fit subsamples by position): every rule reads them as float64,
    so :meth:`threshold_at` draws the sort's own number at the sort's own
    balance, bit for bit, while the cache holds 8 bytes a media rather than a
    list's 32.  *tiled* is whether a structural sort's dataset is tiled.
    """

    rule: Literal["text", "centroid", "structural"]
    scores: np.ndarray | None = None
    tiled: bool = False

    def threshold_at(self, beta: float | None) -> float:
        """The display line at *beta*, rounded to 4 decimals as the sort rounds it."""
        if self.rule == "structural":
            from vtscore.training.structural_similarity import _example_line

            return _example_line(self.tiled, beta)
        if self.scores is None:
            raise ValueError(f"a {self.rule!r} sort line needs the scores it was drawn on")
        if self.rule == "text":
            from vtscore.training.thresholds import text_sort_cuts

            return round(text_sort_cuts(self.scores.tolist(), beta=beta).threshold, 4)
        from vtscore.detectors.centroid_head import centroid_cut

        return round(centroid_cut(self.scores, beta=beta), 4)


def _cosine_sort_scored(query_vec, *, role: str, snap) -> tuple[list[dict], list[float]]:
    """Score every media in the active dataset against *query_vec*: ``(results, sims)``.

    The ranking half of :func:`cosine_sort_active` and :func:`text_sort_active`,
    which differ only in the line(s) they draw over *sims*.

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

    ctx = get_active_context()
    embedder_name = ctx.routed_embedder(role)
    # Region vectors belong to the patch embedder; the per-region max-pool is
    # valid only when the query was scored against that same embedder.
    region_aware = embedder_name is not None and embedder_name == ctx.patch_embedder

    if snap is None:
        snap = snapshot_medias()
    return cosine_sort_with_boxes(snap, query_vec, embedder_name, region_aware=region_aware)


def text_sort_active(query_vec, *, snap=None, beta: float | None = None) -> tuple[list[dict], TextSortCuts]:
    """Sort every media in the active dataset by cosine similarity to a typed query's vector.

    Returns ``(results, cuts)``: *results* as :func:`cosine_sort_active` gives
    them, and *cuts* the sort's two lines (issue #4136), each rounded to 4
    decimals - ``threshold``, the display line drawn by
    :func:`~vtscore.training.thresholds.text_sort_threshold`'s rule, and
    ``acq_threshold``, the midpoint Autopilot's opening samples at.  The text
    route sends the pair as the response's ``threshold`` / ``acq_threshold``,
    the same two fields a learned sort carries, so the Hard select reads the
    acquisition cut and everything the user sees reads the display line.

    *query_vec* must have been embedded by the dataset's text embedder
    (``role="text"``).  *snap* as in :func:`cosine_sort_active`.  *beta* is the
    balance the display line is drawn at (#4603); ``None`` reads the active one
    (:func:`vtscore.state.get_beta`), which is the user's setting before any
    detector exists.  *cuts* carries the display line's :class:`SortLine`, to
    redraw it at another balance (#4760).
    """
    from dataclasses import replace

    import numpy as np

    from vtscore.training.thresholds import text_sort_cuts

    if beta is None:
        from vtscore.state import get_beta

        beta = get_beta()
    results, sims_list = _cosine_sort_scored(query_vec, role="text", snap=snap)
    cuts = text_sort_cuts(sims_list, beta=beta)
    return results, replace(
        cuts,
        threshold=round(cuts.threshold, 4),
        acq_threshold=round(cuts.acq_threshold, 4),
        line=SortLine("text", np.asarray(sims_list, dtype=np.float64)),
    )


def cosine_sort_cuts(
    query_vec, *, role: str = "score", snap=None, beta: float | None = None
) -> tuple[list[dict], TextSortCuts]:
    """Sort every media in the active dataset by cosine similarity to *query_vec*, with both its lines.

    Returns ``(results, cuts)`` as :func:`text_sort_active` does: *results* as
    :func:`cosine_sort_active` gives them, and *cuts* the sort's display line
    (``threshold``) and acquisition cut (``acq_threshold``), each rounded to 4
    decimals.  A ``"text"`` query draws them by the typed query's rules (#3826,
    #4136, #4603).  Any other query - an example's vector, several examples'
    centroid - is the Goods' centroid's sort, so its display line is the one
    :func:`~vtscore.detectors.centroid_head.centroid_cut` draws for a centroid
    head on the same corpus (#4732), and its acquisition cut stays the
    two-Gaussian midpoint, where Autopilot's Hard select has always sampled an
    example sort.  *beta* is the balance the display line is drawn at; ``None``
    reads the active one (:func:`vtscore.state.get_beta`).  *cuts* carries the
    display line's :class:`SortLine`, which draws it, to redraw it at another
    balance (#4760).
    """
    if beta is None:
        from vtscore.state import get_beta

        beta = get_beta()
    if role == "text":
        return text_sort_active(query_vec, snap=snap, beta=beta)
    import numpy as np

    from vtscore.detectors.centroid_head import CENTROID_LINE_RULE
    from vtscore.training.thresholds import TextSortCuts, calculate_gmm_threshold

    results, sims_list = _cosine_sort_scored(query_vec, role=role, snap=snap)
    acq = calculate_gmm_threshold(sims_list)
    line = SortLine("centroid", np.asarray(sims_list, dtype=np.float64))
    return results, TextSortCuts(line.threshold_at(beta), round(acq, 4), CENTROID_LINE_RULE, line)


def cosine_sort_active(query_vec, *, role: str = "score", snap=None) -> tuple[list[dict], float]:
    """Sort every media in the active dataset by cosine similarity to *query_vec*.

    Returns ``(results, threshold)`` where *results* is a list of
    ``{"id": …, "similarity": …}`` dicts sorted descending, and
    *threshold* is the sort's display line (rounded to 4 decimals):
    :func:`cosine_sort_cuts`'s ``threshold``.  A caller that also needs the
    acquisition cut calls :func:`cosine_sort_cuts` instead.

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
    results, cuts = cosine_sort_cuts(query_vec, role=role, snap=snap)
    return results, cuts.threshold


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
    """:func:`example_sort_cuts_from_paths` with the display line alone: ``(results_list, threshold)``."""
    results, cuts = example_sort_cuts_from_paths(file_paths)
    return results, cuts.threshold


def example_sort_cuts_from_paths(file_paths: list[Path]) -> tuple[list[dict], TextSortCuts]:
    """Embed one or more media files and sort all loaded medias by similarity.

    Returns ``(results_list, cuts)`` on success - *cuts* the display line and
    the acquisition cut, as :func:`cosine_sort_cuts` draws them - or raises
    :class:`ValueError` when there are no example files, no medias loaded, no
    embedder for the dataset, or a file that the embedder cannot embed.

    Each file is embedded using the score embedder of the currently loaded
    dataset.  A single example sorts by cosine similarity to its vector;
    multiple examples sort against their centroid (the mean of the
    L2-normalised example vectors), so each example contributes equally
    regardless of its embedding norm.  On a structural (SIFT/VLAD) dataset
    that Stage-1 order is then geometrically re-ranked against *every*
    example as a template, max-over-templates, exactly as the voted path
    treats its RegionYes templates.
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

    results, cuts = cosine_sort_cuts(query_vec, snap=snap)

    # Stage-2 structural re-rank (a no-op for non-structural datasets): for a
    # SIFT/VLAD dataset, geometrically verify the VLAD shortlist against the
    # uploaded examples' own local features.  Every example is a template and
    # a candidate scores as the max over templates - the same rule the voted
    # path applies to its RegionYes templates - so several crops of one mark
    # (or several marks) widen what verifies instead of collapsing into the
    # Stage-1 centroid alone, which on a structural dataset ranks at chance.
    # Any crop was already applied to the file above, so it restricts the
    # template.
    if getattr(emb, "supports_geometric_verification", False):
        from vtscore.state import get_beta
        from vtscore.training.structural_similarity import maybe_structural_rerank_example

        example_features = [emb.local_features_forward(m) for m in medias]
        reranked, threshold = maybe_structural_rerank_example(
            results, cuts.threshold, snap, example_features, score_key="similarity", beta=get_beta()
        )
        # The verified ranking draws one line, which the Hard select reads too.
        # A rerank that ran hands back a new list; one that could not (no usable
        # template, no matcher) hands back the cosine ranking and its line, which
        # still redraws as the centroid's.
        from vtscore.training.structural_stage1 import snapshot_has_tiles
        from vtscore.training.thresholds import TextSortCuts

        line = cuts.line if reranked is results else SortLine("structural", tiled=snapshot_has_tiles(snap))
        results = reranked
        cuts = TextSortCuts(threshold, threshold, "structural", line)

    return results, cuts


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
