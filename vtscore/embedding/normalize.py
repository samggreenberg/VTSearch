"""Canonical L2-normalization for embeddings (single ingest chokepoint).

VTSearch is direction-only nearly everywhere: every similarity comparison
treats embeddings as points on the unit sphere.  Rather than re-normalizing
on every comparison (the old per-call cost in
:mod:`vtscore.training.region_similarity`), we make the unit vector the
*stored* form: every embedding written into ``medias[cid]["embeddings"]`` and
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
embedding matrix, the coverage atlas's k-means, head training, region
similarity, and the VTSBrowse UMAP projection) all consume unit vectors
without re-normalizing.  See ``docs/plans/vtsbrowse.md``
§Prerequisite for the full rationale and the behaviour changes this implies
(angular diversity clustering; head input scale).

The module imports only :mod:`numpy` so it stays a leaf with no risk of an
import cycle through the embedding package façade.
"""

from __future__ import annotations

import numpy as np

#: How far from 1 a vector's norm may sit and still count as already unit.  A
#: vector this module normalizes rounds each component to ``float32`` once, so
#: its norm sits within about 6e-8 of 1; this leaves room for vectors other code
#: normalized in ``float32`` arithmetic, and no consumer can tell 1e-6 from 1.
_UNIT_NORM_TOLERANCE = 1e-6


def l2_normalize(vec: object) -> np.ndarray:
    """Return *vec* as a unit-norm ``float32`` array.

    A zero vector (or one whose norm is non-finite) is returned unchanged
    as ``float32`` rather than dividing — that avoids minting ``inf`` /
    ``nan`` rows that would poison every downstream consumer.

    The function is idempotent **bit for bit**: a vector already unit-norm
    (within :data:`_UNIT_NORM_TOLERANCE`) comes back with the very bytes it
    went in with, so applying it at several chokepoints is harmless.  Dividing
    a unit vector by its own norm again moved the last bit of about a third of
    a dataset's vectors (#4892): the pickle loader re-normalizes what ingest
    stored, so a reloaded dataset held other bytes than the process that
    imported it, and the calibration folds' split, seeded from those bytes
    (:func:`vtscore.training.thresholds.conformal._split_dither_rng`), drew
    another split, and with it another labels line.
    """
    arr = np.asarray(vec, dtype=np.float32)
    wide = arr.astype(np.float64)
    norm = float(np.linalg.norm(wide))
    if norm == 0.0 or not np.isfinite(norm):
        return arr
    if abs(norm - 1.0) <= _UNIT_NORM_TOLERANCE:
        return arr.copy()
    return (wide / norm).astype(np.float32)
