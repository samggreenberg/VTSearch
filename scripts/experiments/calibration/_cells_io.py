"""Loss-free per-(dataset, embedder) media serialization for the calibration run.

Identical in spirit to the Max-Patch runner's ``_cells_io``: the demo *cache*
pickle written by ``load_demo_dataset`` only round-trips each media type's
``pickle_extra_fields`` and silently drops the fields this experiment depends on
(``patch_grid``, ``regions``, multi-label ``categories``).  So
prepare serializes the *in-memory* medias dict directly, dropping only the two
bulky raster fields the cell stage never reads.  The resulting pickles are
byte-compatible with the Max-Patch runner's, so its ``visual_genome_m__*.pkl``
can be symlinked in and read here unchanged.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

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
