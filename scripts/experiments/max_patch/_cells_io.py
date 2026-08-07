"""Loss-free per-(dataset, embedder) media serialization for the Max-Patch run.

The demo *cache* pickle written by ``load_demo_dataset`` only round-trips the
fields declared in each media type's ``pickle_extra_fields`` (for images:
``width``/``height``/``thumbnail_bytes``).  It silently drops exactly the
fields this experiment depends on: the per-patch grid (``patch_grid``), the HAC
patch grid (``patch_grid``), the ground-truth region boxes (``regions``),
and the multi-label ``categories`` list.  Copying that pickle would leave every
arm scoring on the whole-image vector alone — MaxHAC, MaxPatch, and whole_image
would collapse to one curve.

So prepare serializes the *in-memory* medias dict (which carries all of the
above, freshly built by the loader) directly, dropping only the two bulky
raster fields the cell stage never reads (``media_bytes``, ``thumbnail_bytes``);
exemplar crops are already pre-computed in prepare, and cell-time scoring works
purely on vectors.  ``patch_grid`` (an fp16 ndarray) pickles losslessly.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

#: Fields dropped from the cell pickle — large rasters the voting simulation
#: never touches (image decoding happens only at prepare time, for crops).
_DROP_FIELDS = ("media_bytes", "thumbnail_bytes")


def dump_medias(medias: dict[int, dict[str, Any]], path: str | Path) -> int:
    """Pickle *medias* minus the bulky raster fields; return bytes written."""
    thin = {cid: {k: v for k, v in m.items() if k not in _DROP_FIELDS} for cid, m in medias.items()}
    path = Path(path)
    with path.open("wb") as fh:
        pickle.dump(thin, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path.stat().st_size


def load_medias(path: str | Path) -> dict[int, dict[str, Any]]:
    """Load a cell pickle written by :func:`dump_medias`."""
    with Path(path).open("rb") as fh:
        return pickle.load(fh)  # noqa: S301 - our own prepare-written cache, not untrusted input
