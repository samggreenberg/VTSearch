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

#: Suffixes ``run_cells.py`` writes *beside* each cell's main frame, one per
#: side frame.  These are separate long-format tables with their own columns, so
#: an analyzer that concatenates them into the main frame gets a ragged
#: DataFrame whose extra rows silently enter every aggregate.
#:
#: Kept in one place because the failure mode is invisible: ``glob("task_*.csv")``
#: matches the side frames too, and each new side frame has historically had to
#: be excluded by hand in ~8 analyzers - which is exactly as reliable as it
#: sounds.  Add the suffix here when you add a frame, not at the call sites.
SIDE_FRAME_SUFFIXES = ("__sweep", "__cutdiag", "__cutincl")


def main_frame_files(cells_dir: str | Path) -> list[Path]:
    """Every cell's **main** metric CSV under *cells_dir*, side frames excluded."""
    return sorted(p for p in Path(cells_dir).glob("task_*.csv") if not any(s in p.name for s in SIDE_FRAME_SUFFIXES))


def side_frame_files(cells_dir: str | Path, suffix: str) -> list[Path]:
    """Every cell's side frame of one kind, e.g. ``suffix="__cutincl"``."""
    return sorted(Path(cells_dir).glob(f"task_*{suffix}.csv"))


def dump_medias(medias: dict[int, dict[str, Any]], path: str | Path) -> int:
    """Pickle *medias* minus the bulky raster fields; return bytes written."""
    thin = {cid: {k: v for k, v in m.items() if k not in _DROP_FIELDS} for cid, m in medias.items()}
    path = Path(path)
    with path.open("wb") as fh:
        pickle.dump(thin, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path.stat().st_size


class _StaleRegionVector:
    """Stand-in for the ``RegionVector`` that #2886 removed from the app tier.

    Pickles cached before #2886 (the Max-Patch ``visual_genome_m__dinov3_patch``
    one every region-voting arm reuses) store a per-image HAC tree under
    ``media["patch_regions"]``, whose nodes were
    ``vtscore.media.patch_embed.RegionVector``.  Production went tree-free, so
    that class no longer exists at that path and a plain ``pickle.load`` dies
    with ``AttributeError`` before returning a single media.

    Nothing under ``vtscore/`` reads ``media["patch_regions"]`` any more - the
    shipped ``max_patch`` style scores straight off ``patch_grid`` - so these
    nodes are inert baggage.  They are revived into this placeholder purely so
    the surrounding dict loads; the experiment never touches them.  The class is
    deliberately *not* aliased to ``vtscore.eval.patch_styles.RegionVector``:
    that one is a different, narrower dataclass (the old one also carried a
    ``cell_mask``), and pretending they are the same type would be a lie that
    some later reader could act on.
    """

    def __setstate__(self, state: Any) -> None:
        self.__dict__.update(state if isinstance(state, dict) else {})


class _StaleClassUnpickler(pickle.Unpickler):
    """Unpickler that tolerates exactly the one class #2886 deleted."""

    def find_class(self, module: str, name: str) -> Any:
        if name == "RegionVector" and module == "vtscore.media.patch_embed":
            return _StaleRegionVector
        return super().find_class(module, name)


def load_medias(path: str | Path) -> dict[int, dict[str, Any]]:
    """Load a cell pickle written by :func:`dump_medias`.

    Tolerates the pre-#2886 ``RegionVector`` nodes in cached pickles; see
    :class:`_StaleRegionVector`.
    """
    with Path(path).open("rb") as fh:
        # S301 - our own prepare-written cache, not untrusted input.
        return _StaleClassUnpickler(fh).load()  # noqa: S301
