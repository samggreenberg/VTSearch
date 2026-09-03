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


def main_frame_files(cells_dir: str | Path) -> list[Path]:
    """Every cell's **main** metric CSV under *cells_dir*, side frames excluded."""
    return sorted(p for p in Path(cells_dir).glob("task_*.csv") if "__" not in p.stem)


def side_frame_files(cells_dir: str | Path, suffix: str) -> list[Path]:
    """Every cell's side frame of one kind, e.g. ``suffix="__cutincl"``."""
    return sorted(Path(cells_dir).glob(f"task_*{suffix}.csv"))
