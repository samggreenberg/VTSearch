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

import pandas as pd

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
#:
#: ``__picks`` (the #3267 per-click log) was added to ``run_cells.py`` and *not*
#: here, which is the exact miss this constant exists to prevent: it is written
#: unconditionally under ``CALIB_EMIT_PICKS`` (default on), so every analyzer
#: calling :func:`main_frame_files` was concatenating one long-format table per
#: cell into its metric frame.  The pick log shares ``seed``/``dataset``/
#: ``category``/``t`` with the main frame and has no ``cost``, so the extra rows
#: do not raise - they land as NaN in every metric column and change every
#: ``groupby`` denominator silently.
SIDE_FRAME_SUFFIXES = ("__sweep", "__cutdiag", "__cutincl", "__picks", "__fitq")


def main_frame_files(cells_dir: str | Path) -> list[Path]:
    """Every cell's **main** metric CSV under *cells_dir*, side frames excluded."""
    return sorted(p for p in Path(cells_dir).glob("task_*.csv") if not any(s in p.name for s in SIDE_FRAME_SUFFIXES))


def side_frame_files(cells_dir: str | Path, suffix: str) -> list[Path]:
    """Every cell's side frame of one kind, e.g. ``suffix="__cutincl"``."""
    return sorted(Path(cells_dir).glob(f"task_*{suffix}.csv"))


#: The base row's ``pool_variant``.  Whole-image styles emit ``"max"`` here (not
#: blank) - the re-pool variants #2781 added are ``topk``/``pnorm``, and only the
#: raw-patch tree arm emits them.  Filtering on "blank" instead drops *every*
#: row, which is a silent empty analysis, so the accepted set is explicit.
BASE_POOL_VARIANTS = ("", "max")


def _blank(s: pd.Series) -> pd.Series:
    """True where a tag column is empty/NaN - i.e. the arm's own base row."""
    return s.isna() | (s.astype(str).str.strip().isin(("", "nan", "None")))


def _base_rows(df: pd.DataFrame) -> pd.DataFrame:
    """The production rows of one cell: no variant tag, base pooling only.

    Applied per file rather than after the concat; see :func:`load_arm`.
    """
    for col in ("gmm_variant", "schedule"):
        if col in df.columns:
            df = df[_blank(df[col])]
    if "pool_variant" in df.columns:
        pv = df["pool_variant"].fillna("").astype(str).str.strip()
        df = df[pv.isin(BASE_POOL_VARIANTS)]
    return df


def load_arm(arm_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Concatenate one arm's cell CSVs, keeping only its base rows.

    Returns ``(frame, provenance)``; provenance counts the files read, the
    unreadable ones, and the zero-byte ones, because an analysis that silently
    drops cells is how a disk incident becomes a wrong verdict.

    Written for the #2847 spike study and lived in ``analyze_spikes.py`` until
    #3409 moved it here.  It had become the de-facto arm loader for six callers
    across five studies -- including ``curves.py``, which every study's figure
    pair goes through, so the *shared* layer was importing a *study*.  Nothing
    about it was ever spike-specific; ``analyze_spikes`` re-exports the name so
    its own callers are unaffected.
    """
    files = main_frame_files(arm_dir / "cells")
    files = [f for f in files if "__" not in f.name]  # skip __sweep / __cutdiag
    frames, bad, empty, headless, no_base = [], [], [], [], []
    n_rows_all = 0
    for f in files:
        if f.stat().st_size == 0:
            empty.append(f.name)
            continue
        try:
            fr = pd.read_csv(f)
        except Exception:  # noqa: BLE001
            bad.append(f.name)
            continue
        # Filter to base rows PER FILE, not after the concat.  A cell emits one
        # row per (step, gmm_variant, pool_variant) and only ~1 in 34 of them
        # survives this filter, so concatenating first holds 34x the frame that
        # is actually wanted - which is fine at #2847's grid size and is where a
        # long-horizon run with hundreds of cells dies, AFTER the cells have
        # been paid for.  The counts below are accumulated rather than measured
        # off the concat so `n_rows_all` still reports what was read.
        n_rows_all += len(fr)
        # A header-only cell is not a failure: the simulator emits a row only
        # once it has at least one good AND one bad vote, so a rare category
        # whose votes never turned up a positive legitimately writes none.
        # That is the extreme of the positive-starvation regime the #2847 study
        # is about, so it is counted and reported rather than silently dropped -
        # and it differs per arm, which is why paired tests lose those cells.
        #
        # Decided on what the cell WROTE, before the base-row filter: "never
        # found both classes" and "wrote only variant rows" are different facts
        # and only the first is a result.
        if fr.empty:
            headless.append(f.name)
            continue
        fr = _base_rows(fr)
        if fr.empty:
            no_base.append(f.name)
            continue
        frames.append(fr)
    prov = {
        "n_files": len(files),
        "n_read": len(frames),
        "unreadable": bad,
        "zero_byte": empty,
        "no_positive_found": headless,
        #: Wrote rows, none of them base rows.  A tag-column bug, never a
        #: legitimate result, so it is named apart from the cells above.
        "no_base_rows": no_base,
    }
    if no_base:
        raise SystemExit(f"{arm_dir}: base-row filter kept 0 rows in {len(no_base)} cells - check tag columns")
    if not frames:
        return pd.DataFrame(), prov
    df = pd.concat(frames, ignore_index=True)
    prov["n_rows_all"] = int(n_rows_all)
    prov["n_rows"] = int(len(df))
    return df.reset_index(drop=True), prov


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


#: Columns ``run_cells.py`` writes to record **how each cell opened**: the app's
#: two real starts are a text sort and three random known-goods, and since #3276
#: a paired arm's text sort can run in a different embedding space than the arm
#: it is labelled with.
OPENING_COLUMNS = ("seed_mode", "seed_embedder")


def assert_one_opening(frame, where: str = "") -> None:
    """Refuse to pool cells of one environment that did not open the same way.

    ``region_voting_for`` made the *voting mode* a per-cell premise rather than a
    flag, because a run that requests region voting and silently gets whole-image
    training reads as the experiment it is not (#2877).  The **opening** has
    exactly that shape and was the last piece of it left unasserted (#3278): a
    cell records ``seed_mode`` and ``seed_embedder``, and cells of one
    ``(dataset, embedder, category)`` split across a text sort and a known-good
    start are two experiments averaged into one number.

    The way to land there is a **resume**.  Cells are skipped when their CSV
    already exists, so re-running a grid across the #3269 seeding fix -- or after
    an arm was paired -- leaves the old cells in place and writes the new ones
    beside them.  Every column is populated, every count says "N/N cells", and
    the mean is over two openings.

    **Grouped by environment, not by arm**, and the difference is the whole
    argument of the #3278 lesson.  A *category* with no typed query takes the
    known-good start on **every** arm, so it shifts them all together and cancels
    in any contrast between them -- that is a legitimate grid, and a study that
    wants it gone sets ``CALIB_REQUIRE_SEED_QUERY=1``.  An *arm* that opens
    differently from its neighbours does not cancel, and that is what
    ``CALIB_REQUIRE_OPENING`` and preflight check 14 refuse before the array
    rather than after it.  What is left for the analyzer is the case neither can
    see: one environment holding both, which no launcher ever declared and only a
    resume can produce.  Style is not in the key either -- one cell writes every
    style from one opening.

    Cells written before the columns existed carry no value at all; that reads as
    ``unrecorded`` so a mix of old and new fails here rather than averaging
    quietly.  A frame with neither column is entirely pre-#3269 and has nothing
    to compare.

    :raises ValueError: if any ``(dataset, embedder, category)`` holds more than
        one opening.
    """
    if frame is None or len(frame) == 0:
        return
    present = [c for c in OPENING_COLUMNS if c in frame.columns]
    key = ["dataset", "embedder", "category"]
    if not present or not set(key) <= set(frame.columns):
        return
    mixed = []
    for group_key, group in frame.groupby(key, dropna=False):
        openings = {
            tuple(str(row[c]) if row[c] == row[c] and row[c] != "" else "unrecorded" for c in present)
            for _, row in group[present].drop_duplicates().iterrows()
        }
        if len(openings) > 1:
            got = ", ".join(sorted("/".join(o) for o in openings))
            mixed.append(f"{' x '.join(str(k) for k in group_key)}: {got}")
    if mixed:
        raise ValueError(
            (f"{where}: " if where else "")
            + "these environments pooled cells that opened differently -- "
            + "; ".join(mixed[:8])
            + (f" (and {len(mixed) - 8} more)" if len(mixed) > 8 else "")
            + ".  A text sort and a known-good start are different rankings, so their cells are "
            + "not seeds of one experiment; delete the stale cells and re-run them, or analyse "
            + "the two runs separately."
        )
