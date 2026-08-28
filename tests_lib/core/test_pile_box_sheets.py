"""``box_sheets.py`` must read COCO pixels where they are, or say it cannot (#3305).

The sheet resolved COCO images through ``pile_config.COCO_IMAGES`` -- a
directory the staging area has never held; it holds ``val2017.zip``, which is
what ``build_pile._load_coco`` has always read. Nothing failed: the resolver
returned ``None`` for every media and the sheet came out with no thumbnails at
all, which is the one output that looks like an answer and contains none.

So both halves are pinned here: the zip is actually read, and an unresolvable
image source is an error rather than a file. As in
``test_pile_box_scan.py``, the script runs in a subprocess -- ``common.setup_env()``
edits ``os.environ`` and ``sys.meta_path`` at import.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PILE_DIR = _REPO_ROOT / "scripts" / "experiments" / "pile"
_BOX_SHEETS = _PILE_DIR / "box_sheets.py"

#: One positive per synthetic image; four fills a sheet row.
_N_IMAGES = 4


def _jpeg(path: Path, size: tuple[int, int] = (64, 48)) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "#4488cc").save(path, quality=90)


def _names(n: int = _N_IMAGES) -> list[str]:
    return [f"{i:012d}.jpg" for i in range(1, n + 1)]


def _cell(pile: Path, dataset: str, names: list[str], category: str, embedder: str = "siglip") -> None:
    """A cell pickle whose medias are all positives of *category*, with a box."""
    embeddings = pile / "datadir" / "embeddings"
    embeddings.mkdir(parents=True, exist_ok=True)
    medias = {
        i: {
            "id": i,
            "media_type": "image",
            "filename": name,
            "media_string": None,
            "categories": [category],
            "category": category,
            # Normalised, as everything in the pile stores them.
            "regions": [{"box": [0.1, 0.1, 0.1 + 0.02 * i, 0.1 + 0.02 * i], "label": category}],
        }
        for i, name in enumerate(names, start=1)
    }
    with (embeddings / f"{dataset}__{embedder}.pkl").open("wb") as fh:
        pickle.dump(medias, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _zip_images(zip_path: Path, names: list[str], tmp: Path) -> None:
    """A ``val2017.zip`` shaped like COCO's: members under a top-level folder."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    staging = tmp / "_zipsrc"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name in names:
            src = staging / name
            _jpeg(src)
            zf.write(src, f"val2017/{name}")


def _run(pile: Path, coco_root: Path, demo_cache: Path, args: list[str]) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        # `common.setup_env()` defaults these to cluster paths, and neutralises
        # the editable install, so `import vtscore` must be pointed at the tree
        # under test explicitly.
        "VTS_REPO": str(_REPO_ROOT),
        "CALIB_EXP": str(pile / "calib"),
        "CALIB_RESULTS": str(pile / "calib" / "results"),
        "VTS_PILE": str(pile),
        "VTSEARCH_DATA_DIR": str(pile / "datadir"),
        "VTSEARCH_MODELS_DIR": str(pile / "models"),
        "HF_HOME": str(pile / "models"),
        "VTS_COCO_ROOT": str(coco_root),
        "VTS_DEMO_CACHE": str(demo_cache),
    }
    code = f"import sys, box_sheets; sys.exit(box_sheets.main({args!r}))"
    return subprocess.run(  # noqa: S603  # interpreter + test-controlled source
        [sys.executable, "-c", code],
        cwd=str(_PILE_DIR),
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )


@pytest.fixture
def tree(tmp_path: Path):
    """``(pile, coco_root, demo_cache, out)`` -- empty; each test stages what it needs."""
    pytest.importorskip("PIL")
    return tmp_path / "pile", tmp_path / "coco", tmp_path / "demos", tmp_path / "sheet.jpg"


class TestCocoPixelsComeFromTheZip:
    def test_a_coco_sheet_is_drawn(self, tree, tmp_path: Path) -> None:
        """The bug: this used to exit 1 having found no image at all."""
        pile, coco, demos, out = tree
        names = _names()
        _cell(pile, "coco_val", names, "cat")
        _zip_images(coco / "images" / "val2017.zip", names, tmp_path)
        result = _run(pile, coco, demos, ["--dataset", "coco_val", "--category", "cat", "--out", str(out)])
        assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
        assert f"{len(names)} of {len(names)} positives drawable" in result.stdout
        assert out.exists() and out.stat().st_size > 0

    def test_an_extracted_directory_still_works(self, tree, tmp_path: Path) -> None:
        """`COCO_IMAGES` stays a fast path for anyone who unpacks the zip."""
        pile, coco, demos, out = tree
        names = _names()
        _cell(pile, "coco_val", names, "cat")
        for name in names:
            _jpeg(coco / "images" / "val2017" / name)
        result = _run(pile, coco, demos, ["--dataset", "coco_val", "--category", "cat", "--out", str(out)])
        assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
        assert out.exists()

    def test_visual_genome_reads_its_directories(self, tree, tmp_path: Path) -> None:
        """The path that always worked, so the zip fallback cannot cost it."""
        pile, coco, demos, out = tree
        names = _names()
        _cell(pile, "vg_scale", names, "bird@small")
        for name in names:
            _jpeg(demos / "visual_genome" / "VG_100K" / name)
        result = _run(pile, coco, demos, ["--dataset", "vg_scale", "--category", "bird@small", "--out", str(out)])
        assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
        assert out.exists()


class TestAnEmptySheetIsAnError:
    """A sheet with no thumbnails answers the question with nothing."""

    def test_no_image_source_at_all_is_named(self, tree) -> None:
        pile, coco, demos, out = tree
        _cell(pile, "coco_val", _names(), "cat")
        result = _run(pile, coco, demos, ["--dataset", "coco_val", "--category", "cat", "--out", str(out)])
        assert result.returncode != 0
        assert "no image source exists" in result.stderr
        assert "val2017.zip" in result.stderr
        assert not out.exists()

    def test_a_source_that_holds_none_of_the_images_is_an_error(self, tree, tmp_path: Path) -> None:
        """The #3305 shape: the source exists, and resolves nothing in it."""
        pile, coco, demos, out = tree
        _cell(pile, "coco_val", _names(), "cat")
        _zip_images(coco / "images" / "val2017.zip", ["999999999999.jpg"], tmp_path)
        result = _run(pile, coco, demos, ["--dataset", "coco_val", "--category", "cat", "--out", str(out)])
        assert result.returncode != 0
        assert "refusing to write an empty sheet" in result.stderr
        assert not out.exists()

    def test_partial_coverage_warns_but_draws(self, tree, tmp_path: Path) -> None:
        """Some pixels are still worth a sheet -- with the shortfall said out loud."""
        pile, coco, demos, out = tree
        names = _names()
        _cell(pile, "coco_val", names, "cat")
        _zip_images(coco / "images" / "val2017.zip", names[:2], tmp_path)
        result = _run(pile, coco, demos, ["--dataset", "coco_val", "--category", "cat", "--out", str(out)])
        assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
        assert "2 with unreachable pixels" in result.stdout
        assert "WARNING" in result.stdout
        assert out.exists()

    def test_no_positive_is_reported_as_such(self, tree, tmp_path: Path) -> None:
        """Distinct from unreachable pixels: nothing to draw, not nowhere to look."""
        pile, coco, demos, out = tree
        names = _names()
        _cell(pile, "coco_val", names, "cat")
        _zip_images(coco / "images" / "val2017.zip", names, tmp_path)
        result = _run(pile, coco, demos, ["--dataset", "coco_val", "--category", "dog", "--out", str(out)])
        assert result.returncode != 0
        assert "nothing to draw" in result.stderr
        assert not out.exists()


def test_the_zip_path_is_never_spelled_inline() -> None:
    """Pin the *identity*, not the behaviour -- the guard #3299's lesson asked for.

    The bug existed because two places named the COCO pixels independently and
    disagreed. ``pile_config.COCO_VAL_ZIP`` is the one name; a literal here
    would let them drift apart again with nothing to notice.
    """
    source = _BOX_SHEETS.read_text()
    assert "pc.COCO_VAL_ZIP" in source
    # Prose may name the file; a string literal would be a second definition.
    assert '"val2017.zip"' not in source
    assert "'val2017.zip'" not in source
