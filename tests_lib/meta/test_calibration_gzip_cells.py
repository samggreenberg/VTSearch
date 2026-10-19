"""``gzip_cells.py`` converts a finished run's cell frames in place (#5092).

The converter rewrites files that analyzers read by default, on a volume with no
snapshots, so these hold it to the three things that make that safe: a gzipped
run loads exactly as the plain one did; a pass never leaves a cell in both
spellings (``_cells_paths`` refuses those) except for the one frame an
interruption can strand, which the next pass finishes; and the frames it must
not touch (empty failed cells, npz archives, a run still writing) are left as
they were.  Plus the default itself: since #5092 a run gzips unless told not to.
"""

from __future__ import annotations

import gzip
import importlib.util
import os
import sys
import time
from pathlib import Path

import pytest

CALIB = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "calibration"

#: The columns ``_cells_io._base_rows`` filters on, so a fabricated cell survives it.
_ROW = "dataset,embedder,category,t,gmm_variant,schedule,pool_variant\nd,e,c,0.5,,,max\n"


def _load(name: str):
    """Import a calibration module by path, its directory importable only while loading."""
    spec = importlib.util.spec_from_file_location(f"_calib_{name}_5092", CALIB / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # a dataclass resolves its module through sys.modules
    sys.path.insert(0, str(CALIB))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(CALIB))
    return module


@pytest.fixture
def gz():
    return _load("gzip_cells")


def _plain_run(root: Path, *, aged: bool = True) -> Path:
    cells = root / "results" / "cells"
    cells.mkdir(parents=True)
    (cells / "task_0000.csv").write_text(_ROW)
    (cells / "task_0001.csv").write_text(_ROW.replace("0.5", "0.7"))
    (cells / "task_0000__picks.csv").write_text("t,media_id\n1,7\n")
    (cells / "task_0000__testscores.npz").write_bytes(b"npz bytes")
    if aged:
        old = time.time() - 3600
        for p in cells.iterdir():
            os.utime(p, (old, old))
    return cells


def test_a_gzipped_run_loads_exactly_like_the_plain_one(gz, tmp_path: Path) -> None:
    cells_io = _load("_cells_io")
    plain = _plain_run(tmp_path / "plain")
    packed = _plain_run(tmp_path / "packed")
    assert gz.main([str(tmp_path / "packed"), "--apply"]) == 0
    names = sorted(p.name for p in packed.iterdir())
    assert names == ["task_0000.csv.gz", "task_0000__picks.csv.gz", "task_0000__testscores.npz", "task_0001.csv.gz"]
    a, a_prov = cells_io.load_arm(plain.parent)
    b, b_prov = cells_io.load_arm(packed.parent)
    assert a.equals(b)
    assert (a_prov["n_files"], a_prov["n_read"]) == (b_prov["n_files"], b_prov["n_read"]) == (2, 2)
    assert gzip.decompress((packed / "task_0000__picks.csv.gz").read_bytes()) == b"t,media_id\n1,7\n"


def test_a_dry_run_changes_nothing(gz, tmp_path: Path) -> None:
    cells = _plain_run(tmp_path / "run")
    before = sorted((p.name, p.stat().st_mtime) for p in cells.iterdir())
    assert gz.main([str(tmp_path / "run")]) == 0
    assert sorted((p.name, p.stat().st_mtime) for p in cells.iterdir()) == before


def test_an_interrupted_pass_is_finished_by_the_next(gz, tmp_path: Path) -> None:
    """A pass killed between the rename and the unlink leaves one frame in both
    spellings and maybe a ``.part``; the next pass removes both leftovers."""
    cells = _plain_run(tmp_path / "run")
    (cells / "task_0000.csv.gz").write_bytes(gzip.compress((cells / "task_0000.csv").read_bytes()))
    (cells / ".task_0001.csv.gz.part").write_bytes(b"half a gzip")
    assert gz.main([str(cells), "--apply", "--force"]) == 0
    assert not list(cells.glob("*.csv")) and not list(cells.glob(".*.part"))
    assert gzip.decompress((cells / "task_0000.csv.gz").read_bytes()).decode() == _ROW


def test_two_spellings_that_differ_are_a_conflict_and_both_stay(gz, tmp_path: Path) -> None:
    cells = _plain_run(tmp_path / "run")
    (cells / "task_0000.csv.gz").write_bytes(gzip.compress(b"something else\n"))
    assert gz.main([str(cells), "--apply", "--force"]) == 1
    assert (cells / "task_0000.csv").exists() and (cells / "task_0000.csv.gz").exists()


def test_an_empty_failed_cell_stays_plain_and_empty(gz, tmp_path: Path) -> None:
    """Launchers find failed cells with ``find -size 0``; a gzipped empty file is not 0 bytes."""
    cells = _plain_run(tmp_path / "run")
    (cells / "task_0002.csv").write_text("")
    assert gz.main([str(cells), "--apply", "--force"]) == 0
    assert (cells / "task_0002.csv").stat().st_size == 0
    assert not (cells / "task_0002.csv.gz").exists()


def test_a_run_still_writing_is_skipped_without_force(gz, tmp_path: Path) -> None:
    cells = _plain_run(tmp_path / "run", aged=False)
    assert gz.main([str(tmp_path / "run"), "--apply"]) == 1
    assert (cells / "task_0000.csv").exists() and not (cells / "task_0000.csv.gz").exists()


def test_every_cells_directory_under_a_run_is_found(gz, tmp_path: Path) -> None:
    for arm in ("a", "b"):
        (tmp_path / "run" / "results" / arm / "cells").mkdir(parents=True)
    found = gz.cells_dirs(tmp_path / "run")
    assert [p.parent.name for p in found] == ["a", "b"]


def _common(root: Path, target: Path) -> Path:
    """A ``-common`` cells directory: one symlink per frame of *target* (2026-10-06-b*-common)."""
    cells = root / "results" / "cells"
    cells.mkdir(parents=True)
    for f in target.iterdir():
        if f.suffix == ".csv":
            (cells / f.name).symlink_to(f)
    return cells


@pytest.mark.parametrize("order", ["target first", "links first"])
def test_linked_frames_are_re_pointed_not_copied(gz, tmp_path: Path, order: str) -> None:
    """#5092's first pass stopped on a ``-common`` directory whose links its target's conversion had broken."""
    cells_io = _load("_cells_io")
    target = _plain_run(tmp_path / "run")
    common = _common(tmp_path / "run-common", target)
    roots = [tmp_path / "run", tmp_path / "run-common"]
    assert gz.main([str(r) for r in (roots if order == "target first" else roots[::-1])] + ["--apply"]) == 0
    links = sorted(common.iterdir())
    assert [p.name for p in links] == ["task_0000.csv.gz", "task_0000__picks.csv.gz", "task_0001.csv.gz"]
    assert all(p.is_symlink() and p.resolve().parent == target.resolve() for p in links)
    a, _ = cells_io.load_arm(target.parent)
    b, _ = cells_io.load_arm(common.parent)
    assert a.equals(b)


def test_a_link_whose_target_is_still_plain_waits(gz, tmp_path: Path) -> None:
    target = _plain_run(tmp_path / "run")
    common = _common(tmp_path / "run-common", target)
    assert gz.main([str(tmp_path / "run-common"), "--apply"]) == 0
    assert sorted(p.name for p in common.iterdir()) == ["task_0000.csv", "task_0000__picks.csv", "task_0001.csv"]
    assert (target / "task_0000.csv").exists()


def test_open_frame_reads_either_spelling_for_the_csv_module(tmp_path: Path) -> None:
    """The stdlib readers (``csv.DictReader``) cannot take a ``.gz`` frame as text; ``open_frame`` can."""
    import csv

    paths = _load("_cells_paths")
    (tmp_path / "task_0000.csv").write_text(_ROW)
    (tmp_path / "task_0001.csv.gz").write_bytes(gzip.compress(_ROW.encode()))
    rows = []
    for name in ("task_0000.csv", "task_0001.csv.gz"):
        with paths.open_frame(tmp_path / name) as fh:
            rows.append(list(csv.DictReader(fh)))
    assert rows[0] == rows[1] and rows[0][0]["pool_variant"] == "max"


def test_cells_are_gzipped_by_default() -> None:
    """#5092: the State of the App never opted in, and its plain cells filled the volume."""
    src = (CALIB / "experiment_config.py").read_text()
    assert 'CELLS_GZIP = os.environ.get("CALIB_CELLS_GZIP", "1") == "1"' in src
