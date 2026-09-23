"""Loss-free per-(dataset, embedder) media serialization for the calibration run.

Identical in spirit to the Max-Patch runner's ``_cells_io``: the demo *cache*
pickle written by ``load_demo_dataset`` only round-trips each media type's
``pickle_extra_fields`` and silently drops the fields this experiment depends on
(``patch_grid``, ``regions``, multi-label ``categories``).  So
prepare serializes the *in-memory* medias dict directly, dropping only the two
bulky raster fields the cell stage never reads.  The resulting pickles are
byte-compatible with the Max-Patch runner's, so its ``visual_genome_m__*.pkl``
can be symlinked in and read here unchanged.

A cell has two on-disk shapes.  ``dump_medias`` writes the original one -- a
single pickled dict -- and is still the default; ``CellWriter`` writes the
chunked one, for a cell too large to hold in memory whole (#3842).
``load_medias`` and ``iter_medias`` read either, and the chunked shape opens
with a header object so a reader that bypasses them cannot mistake one chunk
for the cell.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from collections.abc import Callable, Iterator
from typing import Any

import pandas as pd

# Re-exported: the discovery half of this module lives in a pandas-free file so
# the csv-and-stdlib figure scripts share the exact same rule.
from _cells_paths import (
    SIDE_FRAME_SUFFIXES,  # noqa: F401 - re-exported; callers import it from here
    main_frame_files,
    side_frame_files,  # noqa: F401 - re-exported; callers import it from here
)

_DROP_FIELDS = ("media_bytes", "thumbnail_bytes")

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


def load_cells(
    cells_dir: str | Path,
    *,
    where: str = "",
    per_file: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Read every main frame under *cells_dir* into one frame plus provenance.

    **The one cell reader.**  Eight analyzers had grown their own, and they
    diverged on exactly the three guards a grid run needs and on nothing else:
    whether a zero-byte cell is skipped, whether an unreadable one is caught,
    and whether a header-only one is counted or silently dropped.  Four of the
    eight had none of the three, so a study that analysed 1295 of 1344 cells
    reported neither number -- which is how a transient ENOSPC on the shared
    volume becomes a wrong verdict rather than a re-run.  This implements the
    union of them; the callers keep their own study-specific post-processing,
    which is the part that legitimately differs.

    The three drops are different facts and are reported apart:

    * **zero-byte** -- the cell died mid-write.  Data loss.
    * **unreadable** -- it parsed as far as a truncated row.  Also data loss,
      counted rather than raised so one bad cell does not lose the run.
    * **header-only** -- the simulator emitted no row because the cell never
      held one Good and one Bad vote at once.  *Not* a failure: it is the
      extreme of the positive-starvation regime several of these studies are
      about, so it is counted and reported rather than assumed away.  It is
      also invisible to a zero-byte check and to ``find -size 0``, which is why
      the count has to be taken here.

    *where*, when given, labels an :func:`assert_one_opening` check on the
    concatenated frame.  *per_file* filters each cell's rows **before** the
    concat -- see :func:`load_arm`, where holding the unfiltered frame is what
    kills a long-horizon run after the cells have been paid for.

    Returns ``(frame, provenance)``.  The frame is empty when nothing loaded;
    the provenance is always complete.
    """
    files = main_frame_files(cells_dir)
    frames: list[pd.DataFrame] = []
    bad: list[tuple[str, str]] = []
    empty: list[str] = []
    headless: list[str] = []
    filtered: list[str] = []
    n_rows_all = 0
    for f in files:
        if f.stat().st_size == 0:
            empty.append(f.name)
            continue
        try:
            # `low_memory=False` because pandas otherwise types each chunk of a
            # column independently and warns on `cut_fallback` /
            # `cut_fallback_kind`, whose values differ between the base rows and
            # the variant rows.  The warning is noise, but it fires once per
            # cell and buries the load line that reports what was dropped.
            fr = pd.read_csv(f, low_memory=False)
        except Exception as exc:  # noqa: BLE001 - a truncated cell is data loss to report, not a crash
            bad.append((f.name, repr(exc)[:80]))
            continue
        n_rows_all += len(fr)
        # Decided on what the cell WROTE, before any filter: "never found both
        # classes" and "wrote only rows the caller filters out" are different
        # facts and only the first is a result.
        if fr.empty:
            headless.append(f.name)
            continue
        if per_file is not None:
            fr = per_file(fr)
            if fr.empty:
                filtered.append(f.name)
                continue
        frames.append(fr)
    prov = {
        "cells_dir": str(cells_dir),
        "n_files": len(files),
        "n_read": len(frames),
        "zero_byte": empty,
        "unreadable": bad,
        "header_only": headless,
        "filtered_out": filtered,
        "n_rows_all": int(n_rows_all),
    }
    if not frames:
        prov["n_rows"] = 0
        return pd.DataFrame(), prov
    df = pd.concat(frames, ignore_index=True).reset_index(drop=True)
    prov["n_rows"] = int(len(df))
    if where:
        assert_one_opening(df, where)
    return df, prov


#: What :func:`describe_load` counts as **data loss**, and every spelling each
#: count goes by.  Two loaders write these dicts and they do not agree on the
#: names: ``load_arm`` renames ``filtered_out`` to ``no_base_rows`` (and
#: ``header_only`` to ``no_positive_found``; see :data:`STARVED_KEYS`) because
#: its filter is its own contract rather than the reader's.  Accepting both
#: spellings here is what keeps the shared sentence shared -- it used to read
#: only the ``load_cells`` names, so any caller handing it ``load_arm``'s
#: provenance reported *zero* of these however many there were (#3808).
LOSS_KEYS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("zero-byte", ("zero_byte",)),
    ("unreadable", ("unreadable",)),
    ("no rows after filter", ("filtered_out", "no_base_rows")),
)

#: The starved count, under both loaders' spellings.  Held apart from
#: :data:`LOSS_KEYS` because it is **not** a loss: the simulator emits no row
#: until one Good and one Bad vote coexist, so a cell that never found a
#: positive legitimately writes none.  That distinction is the whole reason
#: ``load_arm`` renamed the key, and the sentence now says it too.
STARVED_KEYS: tuple[str, ...] = ("header_only", "no_positive_found")

#: Every provenance key the sentence knows how to name.  A loader that invents
#: a key outside this set drops silently out of the coverage line, which is the
#: failure #3808 was; ``tests_lib/meta/test_calibration_cells_io.py`` holds both
#: loaders to it.
DESCRIBED_KEYS: frozenset[str] = frozenset([key for _label, keys in LOSS_KEYS for key in keys] + list(STARVED_KEYS))


def describe_load(prov: dict) -> str:
    """One line naming everything a load dropped, for a study's coverage block.

    Every study prints this differently and several printed nothing; a shared
    sentence is what makes "N of M cells" mean the same thing in two reports.

    Accepts the provenance of **either** loader -- see :data:`LOSS_KEYS`.  The
    starved cells are reported last and worded apart from the losses above
    them, because they are a result of the experiment rather than a hole in it.
    """
    parts = [f"{prov['n_read']}/{prov['n_files']} cells with data", f"{prov.get('n_rows', 0):,} rows"]
    for label, keys in LOSS_KEYS:
        n = sum(len(prov.get(key) or ()) for key in keys)
        if n:
            parts.append(f"{n} {label}")
    starved = sum(len(prov.get(key) or ()) for key in STARVED_KEYS)
    if starved:
        parts.append(f"{starved} starved (header-only; not data loss)")
    return ", ".join(parts)


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

    The reading and the three drop counts are :func:`load_cells`'; what is left
    here is the base-row filter, which is this loader's own contract.  A cell
    emits one row per (step, gmm_variant, pool_variant) and only ~1 in 34 of
    them survives it, so the filter runs per file rather than after the concat:
    concatenating first holds 34x the frame that is wanted, which is fine at
    #2847's grid size and is where a long-horizon run with hundreds of cells
    dies, AFTER the cells have been paid for.
    """
    df, base = load_cells(arm_dir / "cells", per_file=_base_rows)
    prov = {
        "n_files": base["n_files"],
        "n_read": base["n_read"],
        "unreadable": [name for name, _ in base["unreadable"]],
        "zero_byte": base["zero_byte"],
        # A cell that emitted no row at all: the simulator writes one only once
        # it has at least one good AND one bad vote, so a rare category whose
        # votes never turned up a positive legitimately writes none.  That is
        # the extreme of the positive-starvation regime #2847 is about, and it
        # differs per arm, which is why paired tests lose those cells.
        #
        # These two keys are renames of `load_cells`' `header_only` /
        # `filtered_out`.  `describe_load` reads both spellings (see
        # `LOSS_KEYS` / `STARVED_KEYS`); a THIRD spelling would fall out of its
        # sentence silently, so add it there in the same commit.
        "no_positive_found": base["header_only"],
        #: Wrote rows, none of them base rows.  A tag-column bug, never a
        #: legitimate result, so it is named apart from the cells above.
        "no_base_rows": base["filtered_out"],
    }
    if prov["no_base_rows"]:
        raise SystemExit(
            f"{arm_dir}: base-row filter kept 0 rows in {len(prov['no_base_rows'])} cells - check tag columns"
        )
    if df.empty:
        return df, prov
    prov["n_rows_all"] = base["n_rows_all"]
    prov["n_rows"] = base["n_rows"]
    return df, prov


def _thin(medias: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """*medias* minus the bulky raster fields, as the cell stores them."""
    return {cid: {k: v for k, v in m.items() if k not in _DROP_FIELDS} for cid, m in medias.items()}


def dump_medias(medias: dict[int, dict[str, Any]], path: str | Path) -> int:
    """Pickle *medias* minus the bulky raster fields; return bytes written.

    One dict, one ``pickle.dump``, unchanged since the Max-Patch runner wrote
    the first cell -- the format the five scripts that read a cell with a bare
    ``pickle.load`` depend on.  :class:`CellWriter` is the streaming half, for a
    cell too large to hold whole; see its docstring for why it is a separate
    entry point rather than a flag here.
    """
    path = Path(path)
    with path.open("wb") as fh:
        pickle.dump(_thin(medias), fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path.stat().st_size


#: Key of the header object that opens a chunked cell.  Its **value** is the
#: format name, so a later shape can be told from this one rather than guessed
#: at.  A one-shot cell is a dict keyed by integer media id, so the key cannot
#: collide with a real cid.
CHUNKED_MARKER = "__vts_chunked_cell__"
CHUNKED_FORMAT = "chunked-v1"


class CellWriter:
    """Write a cell in chunks, so the whole of it is never in memory at once.

    A cell has always been assembled as one dict and pickled at the end, which
    is fine while the dict fits.  It stops fitting: DocMarks tier ``l`` is
    200,000 pages at ~169 KB of ``local_features`` each (measured, #3842), so
    the ``sift_vlad`` cell alone is ~34 GB *before* the page bytes each media
    carries until :func:`_thin` drops them.  Streaming turns that into one
    chunk at a time.

    **Why a separate entry point rather than a flag on** :func:`dump_medias`.
    Five scripts under ``scripts/experiments/`` read a cell with a bare
    ``pickle.load`` rather than through :func:`load_medias`, and a chunked cell
    handed to one of those would otherwise return its *first chunk* -- a
    perfectly well-formed dict of medias that is silently a fraction of the
    cell.  So the two formats are written by two functions, one-shot stays the
    default and stays byte-identical (the Max-Patch runner's ``_cells_io`` is a
    trimmed copy of this one, and cells are symlinked between the two), and the
    chunked shape opens with :data:`CHUNKED_MARKER`: a bare ``pickle.load`` on
    a streamed cell gets the header, never medias, and fails loudly at the
    first ``.get`` on what it thinks is a media.

    Writes to a ``.part`` file and renames on clean exit, so a run that dies at
    hour 40 leaves no half-cell for ``--verify`` to accept as whole.

        with CellWriter(path) as writer:
            for chunk in chunks:
                writer.write(embed(chunk))
        print(writer.nbytes)
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._tmp = self.path.with_name(self.path.name + ".part")
        self._fh: Any = None
        self._ids: set[int] = set()
        self.n_medias = 0
        self.n_chunks = 0
        self.nbytes = 0

    def __enter__(self) -> "CellWriter":
        self._tmp.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._tmp.open("wb")
        pickle.dump({CHUNKED_MARKER: CHUNKED_FORMAT}, self._fh, protocol=pickle.HIGHEST_PROTOCOL)
        return self

    def write(self, medias: dict[int, dict[str, Any]]) -> int:
        """Append one chunk, thinned as :func:`dump_medias` thins it.

        Media ids must be unique across the whole cell: the reader merges the
        chunks into one dict, so a repeated id would be a silently *shorter*
        cell rather than an error.  That is exactly the mistake a chunked
        writer invites -- indexing each chunk from zero instead of from a
        running offset -- so it is refused here rather than left to a later
        count that looks plausible.
        """
        if self._fh is None:
            raise RuntimeError("CellWriter.write outside the `with` block")
        thin = _thin(medias)
        repeated = self._ids & set(thin)
        if repeated:
            raise ValueError(
                f"{self.path.name}: media id(s) {sorted(repeated)[:5]} already written -- "
                "chunk indices must continue across chunks, not restart at 0"
            )
        self._ids.update(thin)
        pickle.dump(thin, self._fh, protocol=pickle.HIGHEST_PROTOCOL)
        self.n_medias += len(thin)
        self.n_chunks += 1
        return len(thin)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        if exc_type is not None:
            self._tmp.unlink(missing_ok=True)
            return False
        self._tmp.replace(self.path)
        self.nbytes = self.path.stat().st_size
        return False


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


def iter_medias(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(cid, media)`` from a cell without holding the whole of it.

    Reads either shape: a one-shot cell from :func:`dump_medias` (one pickled
    dict) or a chunked one from :class:`CellWriter` (a header then one pickled
    dict per chunk).  Only the chunked shape is actually streamed -- a one-shot
    cell is one pickle and there is no way to read half of it -- so this buys
    nothing on the cells that already fit and everything on the ones that do
    not.

    Use it wherever a pass only *visits* medias (counting, verifying,
    summarising).  :func:`load_medias` is the same iteration collected into the
    dict every caller that indexes or mutates a cell still wants.

    Tolerates the pre-#2886 ``RegionVector`` nodes; see
    :class:`_StaleRegionVector`.
    """
    with Path(path).open("rb") as fh:
        # S301 - our own prepare-written cache, not untrusted input.
        first = _StaleClassUnpickler(fh).load()  # noqa: S301
        if not (isinstance(first, dict) and first.get(CHUNKED_MARKER)):
            yield from first.items()
            return
        while True:
            try:
                chunk = _StaleClassUnpickler(fh).load()  # noqa: S301
            except EOFError:
                return
            yield from chunk.items()


#: How far a stored vector's norm may sit from 1 before the harness renormalises
#: it. Loose on purpose: float32 rounding leaves unit vectors a few 1e-7 off,
#: and re-dividing those would change the bits of every cell in the pile and
#: every published run with them. Only a genuinely raw vector is touched.
UNIT_NORM_TOL = 1e-3


def repair_norms(medias: dict[int, dict[str, Any]], where: str = "") -> int:
    """Renormalise any whole-image vector that is not unit-norm; return how many.

    The app sends every vector through ``l2_normalize`` at ingest, and the
    harness reads cells with a raw ``pickle.load`` that bypassed that chokepoint:
    `coco_val__siglip` shipped norms of 12-19 for seven weeks, and every harness
    run on it trained a detector the app would never have built (#4095, #4099).
    This puts the harness through the same rule. Patch grids are untouched.
    """
    import numpy as np  # noqa: PLC0415

    fixed = 0
    for media in medias.values():
        embs = media.get("embeddings")
        if not isinstance(embs, dict):
            continue
        for name, vec in embs.items():
            arr = np.asarray(vec, dtype=np.float32)
            if arr.ndim != 1:
                continue
            norm = float(np.linalg.norm(arr))
            if norm > 0 and np.isfinite(norm) and abs(norm - 1.0) > UNIT_NORM_TOL:
                embs[name] = arr / norm
                fixed += 1
    if fixed:
        import sys  # noqa: PLC0415

        print(f"_cells_io: renormalised {fixed} non-unit vectors{f' in {where}' if where else ''}", file=sys.stderr)
    return fixed


def load_medias(path: str | Path, repair: bool = False) -> dict[int, dict[str, Any]]:
    """Load a cell pickle written by :func:`dump_medias` or :class:`CellWriter`.

    Returns the cell exactly as stored. ``repair=True`` renormalises non-unit
    vectors (:func:`repair_norms`), which is what a harness that trains and
    scores the APP's detector must ask for; it is not the default because this
    reader is shared with pipelines whose vectors never pass through the app
    (DocMarks' structural descriptors among them), where a norm is data.

    Tolerates the pre-#2886 ``RegionVector`` nodes in cached pickles; see
    :class:`_StaleRegionVector`.
    """
    medias = dict(iter_medias(path))
    if repair:
        repair_norms(medias, where=Path(path).name)
    return medias


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
