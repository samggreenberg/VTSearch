"""Which files in a ``cells/`` directory are a cell's *main* frame.

Deliberately import-free — no pandas, no ``common.setup_env`` — so that the
csv-and-stdlib figure scripts can share the rule with the pandas analyzers.
That was the whole reason the rule kept getting re-typed: half the readers in
this directory avoid pandas on purpose, ``_cells_io`` needs it, and so eleven
scripts each carried their own copy of a one-line filter.  Re-exported from
:mod:`_cells_io`, which is where the pandas-side callers already look.

``run_cells.py`` writes one **main** metric frame per cell, ``task_NNNN.csv``,
and one **side** frame per extra table beside it, ``task_NNNN__<suffix>.csv``.
Side frames are separate long-format tables with their own columns, so an
analyzer that concatenates one into the main frame gets a ragged frame whose
extra rows enter every aggregate.  Nothing raises: a pick row shares
``seed``/``dataset``/``category``/``t`` with the main frame and has no ``cost``,
so the extra rows land as NaN in every metric column and move every ``groupby``
denominator.  The number still looks like a number.
"""

from __future__ import annotations

from pathlib import Path

#: Every side frame ``run_cells.py`` writes, as a registry — **not** as the
#: filter.  :func:`main_frame_files` excludes side frames structurally, on the
#: ``__`` in the stem, because the allowlist shape is one a human has to
#: remember to extend and twice did not: ``__picks`` (#3267) and ``__fitq``
#: (#3329) were both added to ``run_cells.py`` and to no list anywhere, and
#: ``bench_cells.py``'s private three-of-five copy was reading the per-click
#: pick log into four bench analyzers' metric frames as a result (#3407).
#:
#: What the registry is still for is :func:`side_frame_files`, which asks for a
#: frame by name, and the meta-test that holds it to what the runner writes
#: (``tests_lib/meta/test_calibration_cells_io.py``).
SIDE_FRAME_SUFFIXES = ("__sweep", "__cutdiag", "__cutincl", "__picks", "__fitq")


#: A run under ``CALIB_CELLS_GZIP=1`` writes every frame as ``task_NNNN.csv.gz``
#: (#4184: a COCO Better cell is ~3.3 MB as text and ~180 KB gzipped, and that
#: study's 5,040 cells did not fit the shared volume uncompressed).  pandas
#: picks the codec off the suffix, so only the glob has to know.
CELL_SUFFIXES = (".csv", ".csv.gz")


def _cell_files(cells_dir: str | Path, pattern: str) -> list[Path]:
    """*pattern* in either spelling, refusing a cell written in both.

    Both at once means a directory was resumed with the knob flipped, and
    reading both would count that cell twice.
    """
    files = sorted(p for ext in CELL_SUFFIXES for p in Path(cells_dir).glob(pattern + ext))
    names = [p.name.removesuffix(".gz") for p in files]
    twice = sorted({n for n in names if names.count(n) > 1})
    if twice:
        raise ValueError(
            f"{cells_dir}: {len(twice)} cell file(s) exist both plain and gzipped, e.g. {twice[0]} - "
            "delete one spelling before reading"
        )
    return files


def main_frame_files(cells_dir: str | Path) -> list[Path]:
    """Every cell's **main** metric CSV under *cells_dir*, side frames excluded."""
    return [p for p in _cell_files(cells_dir, "task_*") if "__" not in p.name]


def side_frame_files(cells_dir: str | Path, suffix: str) -> list[Path]:
    """Every cell's side frame of one kind, e.g. ``suffix="__cutincl"``."""
    return _cell_files(cells_dir, f"task_*{suffix}")


def pframe_files(cells_dir: str | Path) -> list[Path]:
    """Every cell's #4220 precision-frame archive, ``task_NNNN__pframes.npz``.

    Not a side *frame* - an npz of arrays, written only under
    ``CALIB_PFRAME_STEPS`` - so it is outside the CSV registry, and its
    cell is named by the main frame sharing its stem.
    """
    return sorted(Path(cells_dir).glob("task_*__pframes.npz"))
