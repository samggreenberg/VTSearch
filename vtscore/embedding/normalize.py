"""Canonical L2-normalization for embeddings (single ingest chokepoint).

VTSearch is direction-only nearly everywhere: every similarity comparison
treats embeddings as points on the unit sphere.  Rather than re-normalizing
on every comparison (the old per-call cost in
:mod:`vtscore.training.region_similarity`), we make the unit vector the
*stored* form: every embedding written into ``medias[cid]["embedding"]`` and
every text-query vector is L2-normalized exactly once, at ingest.

This module is the **one** place that performs that normalization.  It is
applied at each point a vector enters the system:

* fresh embeds — :meth:`MediaEmbedder.embed_media` /
  :meth:`~MediaEmbedder.embed_media_bulk` (covers every embedder subclass);
* text queries — :meth:`MediaEmbedder.embed_text` (covers every caller,
  funnelled or direct);
* pickle-loaded stored vectors —
  :func:`vtscore.datasets.loader_pickle._build_pickle_full_media` /
  ``_build_pickle_thin_media``;
* re-ingested-from-origin vectors —
  :func:`vtscore.datasets.ingest._build_media_data`.

Because the invariant holds at the store, downstream consumers (the cached
embedding matrix, the diversity tree's k-means, MLP training, region
similarity, and the VTSBrowse UMAP projection) all consume unit vectors
without re-normalizing.  See ``docs/plans/vtsbrowse.md``
§Prerequisite for the full rationale and the behaviour changes this implies
(angular diversity clustering; MLP input scale).

The module imports only :mod:`numpy` so it stays a leaf with no risk of an
import cycle through the embedding package façade.
"""

from __future__ import annotations

import numpy as np


def l2_normalize(vec: object) -> np.ndarray:
    """Return *vec* as a unit-norm ``float32`` array.

    A zero vector (or one whose norm is non-finite) is returned unchanged
    as ``float32`` rather than dividing — that avoids minting ``inf`` /
    ``nan`` rows that would poison every downstream consumer.  The function
    is idempotent: normalizing an already-unit vector returns it unchanged
    up to ``float32`` rounding, so applying it at several chokepoints (or to
    an embedder that already normalizes its output) is harmless.
    """
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0 or not np.isfinite(norm):
        return arr
    return arr / norm
