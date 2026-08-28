"""The ``vg_box_*`` rebuild path must survive both eras of the box-scan file (#3297).

``build_pile.py`` selects a band's categories from ``vg_box_scale.json``. That
file changed shape on 2026-08-17 (``fb4f4ec03`` wrapped the stats dict in a
``{"meta": ..., "categories": ...}`` envelope) while the file on scratch stayed
pre-envelope, so every ``vg_box_*`` rebuild died with ``KeyError: 'categories'``
for eleven days. Nothing caught it, because the *built* cells kept loading fine
and the rebuild path shares no code with the load path.

These tests run the real selector in a subprocess against synthetic scans of
both shapes. The subprocess is not incidental: ``pile_config.setup_env()`` edits
``os.environ`` and ``sys.meta_path`` at import, which is not something to do to
the shared test process.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path

_PILE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "pile"

#: One synthetic category per slot, spread across the band so the stratified
#: selector has a real range to work with rather than a single point.
_N_CATEGORIES = 200


def _stats(n: int = _N_CATEGORIES) -> dict[str, dict]:
    """Categories that all qualify: inside the ``small`` band, well supported."""
    return {
        f"thing{i:03d}": {
            "voted_area": 0.001 + i * 1e-5,
            "n_images": 100 + i,
            "union_inflation": 1.0,
        }
        for i in range(n)
    }


def _run(pile: Path, code: str, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "VTS_PILE": str(pile),
        "VTSEARCH_DATA_DIR": str(pile / "datadir"),
        "VTSEARCH_MODELS_DIR": str(pile / "models"),
        "HF_HOME": str(pile / "models"),
        **(extra_env or {}),
    }
    return subprocess.run(  # noqa: S603  # interpreter + test-controlled source
        [sys.executable, "-c", code],
        cwd=str(_PILE_DIR),
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )


def _select(pile: Path, band: str = "small") -> subprocess.CompletedProcess:
    """Run the real band selector and print what it chose, as JSON."""
    return _run(
        pile,
        f"import json, build_pile; print(json.dumps(build_pile._band_categories({band!r})))",
    )


def _chosen(result: subprocess.CompletedProcess) -> list[str]:
    assert result.returncode == 0, f"selector failed:\n{result.stdout}\n{result.stderr}"
    return json.loads(result.stdout.strip().splitlines()[-1])


def _write_scan(pile: Path, payload: object) -> None:
    pile.mkdir(parents=True, exist_ok=True)
    (pile / "vg_box_scale.json").write_text(json.dumps(payload))


class TestBothScanShapes:
    def test_envelope_scan_selects(self, tmp_path: Path) -> None:
        """The current shape: stats wrapped in an envelope."""
        _write_scan(tmp_path, {"meta": {"n_images_scanned": 108000}, "categories": _stats()})
        assert _chosen(_select(tmp_path))

    def test_pre_envelope_scan_selects(self, tmp_path: Path) -> None:
        """The 2026-08-12 shape: the bare stats dict. This is the #3297 regression."""
        _write_scan(tmp_path, _stats())
        assert _chosen(_select(tmp_path))

    def test_both_shapes_choose_the_same_categories(self, tmp_path: Path) -> None:
        """The point of tolerating the old shape rather than re-scanning.

        Re-running ``scan_vg_boxes.py`` would regenerate the file in the current
        format, but with per-image compact filtering and per-band supply, which
        qualify categories differently -- silently redefining three datasets
        whose numbers are published in #3129 and #3156. Reading the old file
        instead has to select exactly what the envelope would from the same
        statistics, and that is what this pins.
        """
        old, new = tmp_path / "old", tmp_path / "new"
        stats = _stats()
        _write_scan(old, stats)
        _write_scan(new, {"meta": {"n_images_scanned": 108000}, "categories": stats})
        assert _chosen(_select(old)) == _chosen(_select(new))


class TestMalformedScansFailByName:
    """A bad scan should say which era it is from, not raise a bare KeyError."""

    def test_missing_field_is_named(self, tmp_path: Path) -> None:
        stats = _stats()
        for s in stats.values():
            del s["union_inflation"]
        _write_scan(tmp_path, stats)
        result = _select(tmp_path)
        assert result.returncode != 0
        assert "union_inflation" in result.stderr
        assert "KeyError" not in result.stderr

    def test_empty_scan_is_named(self, tmp_path: Path) -> None:
        _write_scan(tmp_path, {})
        result = _select(tmp_path)
        assert result.returncode != 0
        assert "re-run scan_vg_boxes.py" in result.stderr

    def test_a_category_named_categories_is_not_an_envelope(self, tmp_path: Path) -> None:
        """VG's vocabulary is free text, so "categories" is a name it could hold.

        The envelope is detected by shape rather than by the key being present,
        so a bare scan carrying such a category is still read as a bare scan.
        """
        stats = _stats()
        stats["categories"] = {"voted_area": 0.0015, "n_images": 500, "union_inflation": 1.0}
        _write_scan(tmp_path, stats)
        assert "categories" in _chosen(_select(tmp_path))


class TestRebuildableCanary:
    """``--rebuildable`` is what would have caught #3297 the day it landed."""

    def _canary(self, pile: Path) -> subprocess.CompletedProcess:
        return _run(
            pile,
            "import sys, build_pile; sys.exit(build_pile.rebuildable(['vg_box_small']))",
        )

    def test_passes_on_a_readable_scan(self, tmp_path: Path) -> None:
        _write_scan(tmp_path, _stats())
        result = self._canary(tmp_path)
        assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
        assert "categories selected" in result.stdout

    def test_fails_on_the_break_it_exists_to_catch(self, tmp_path: Path) -> None:
        """An envelope-era reader against a pre-envelope file: the #3297 shape."""
        _write_scan(tmp_path, {"meta": {}, "categories": {}})
        result = self._canary(tmp_path)
        assert result.returncode == 1
        assert "REBUILD-BROKEN" in result.stdout

    def test_reports_every_broken_dataset_not_just_the_first(self, tmp_path: Path) -> None:
        """One run should surface the whole list; a canary that stops at the
        first problem costs a round trip per dataset."""
        _write_scan(tmp_path, {"meta": {}, "categories": {}})
        result = _run(
            tmp_path,
            "import sys, build_pile; "
            "sys.exit(build_pile.rebuildable(['vg_box_small', 'vg_box_medium', 'vg_box_large']))",
        )
        assert result.returncode == 1
        assert result.stdout.count("REBUILD-BROKEN") == 3

    def test_unknown_dataset_is_rejected(self, tmp_path: Path) -> None:
        _write_scan(tmp_path, _stats())
        result = _run(tmp_path, "import build_pile; build_pile.rebuildable(['nope'])")
        assert result.returncode != 0
        assert "unknown dataset" in result.stderr


class TestCanaryCatchesASilentDatasetChange:
    """ "Would a rebuild run?" is weaker than "would a rebuild produce *this*?".

    #3297 had two candidate repairs and both made selection run again; only one
    kept picking the categories the published ``vg_box_*`` sets hold. Taking the
    other would have redefined three datasets with the right media count, the
    right vectors and nothing visible to say so. #3299 checked that by hand
    against the live cells; this is the same check, in the canary.
    """

    def _cell(self, pile: Path, categories: list[str], embedder: str = "siglip") -> None:
        """Write a `vg_box_small` cell holding exactly *categories*."""
        embeddings = pile / "datadir" / "embeddings"
        embeddings.mkdir(parents=True, exist_ok=True)
        medias = {i: {"id": i, "categories": [c]} for i, c in enumerate(categories)}
        with (embeddings / f"vg_box_small__{embedder}.pkl").open("wb") as fh:
            pickle.dump(medias, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def _canary(self, pile: Path) -> subprocess.CompletedProcess:
        return _run(pile, "import sys, build_pile; sys.exit(build_pile.rebuildable(['vg_box_small']))")

    def _selected(self, pile: Path) -> list[str]:
        _write_scan(pile, _stats())
        return _chosen(_select(pile))

    def test_agreeing_cell_passes(self, tmp_path: Path) -> None:
        self._cell(tmp_path, self._selected(tmp_path))
        result = self._canary(tmp_path)
        assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

    def test_a_changed_vocabulary_is_reported_as_broken(self, tmp_path: Path) -> None:
        chosen = self._selected(tmp_path)
        self._cell(tmp_path, [*chosen[:-1], "something_else_entirely"])
        result = self._canary(tmp_path)
        assert result.returncode == 1
        assert "REBUILD-BROKEN" in result.stdout
        assert "would NOT reproduce" in result.stdout

    def test_no_built_cell_is_not_a_failure(self, tmp_path: Path) -> None:
        """A purged pile has nothing to reproduce; that is the rebuild case."""
        _write_scan(tmp_path, _stats())
        assert self._canary(tmp_path).returncode == 0


class TestCanaryChecksThePathTheBuildReads:
    """The canary must name the *same* source the builder opens (#3299).

    Its first real run reported ``coco_val`` REBUILD-BROKEN. Nothing was
    broken: ``_load_coco`` reads ``images/val2017.zip``, which was present,
    while the canary checked ``images/val2017`` -- an extracted directory the
    staging area has never held. A canary pointed at a path the build never
    touches raises a false alarm exactly as loudly as a true one, which is the
    fastest way to teach people to ignore it.
    """

    def _coco_canary(self, pile: Path, root: Path) -> subprocess.CompletedProcess:
        return _run(
            pile,
            "import sys, build_pile; sys.exit(build_pile.rebuildable(['coco_val']))",
            extra_env={"VTS_COCO_ROOT": str(root)},
        )

    def _stage(self, root: Path, *, zip_present: bool = True, extracted: bool = False) -> None:
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "derived").mkdir(parents=True, exist_ok=True)
        (root / "derived" / "objects_flat_val2017.jsonl.gz").write_bytes(b"")
        if zip_present:
            (root / "images" / "val2017.zip").write_bytes(b"")
        if extracted:
            (root / "images" / "val2017").mkdir(exist_ok=True)

    def test_the_staged_zip_alone_is_a_rebuildable_source(self, tmp_path: Path) -> None:
        """How the source actually sits on the cluster: a zip, never unpacked."""
        _write_scan(tmp_path, _stats())
        root = tmp_path / "COCO"
        self._stage(root)
        result = self._coco_canary(tmp_path, root)
        assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
        assert "REBUILD-BROKEN" not in result.stdout

    def test_a_missing_zip_is_still_reported(self, tmp_path: Path) -> None:
        """Relaxing the check must not turn it off: an extracted directory is
        not a substitute for the zip the builder opens."""
        _write_scan(tmp_path, _stats())
        root = tmp_path / "COCO"
        self._stage(root, zip_present=False, extracted=True)
        result = self._coco_canary(tmp_path, root)
        assert result.returncode == 1
        assert "REBUILD-BROKEN" in result.stdout
        assert "val2017.zip" in result.stdout

    def test_the_canary_and_the_builder_name_one_path(self, tmp_path: Path) -> None:
        """Pins the repair itself: both sides go through ``pc.COCO_VAL_ZIP``.

        Spelling the zip inline in the builder while a constant named the
        directory is what let the two drift apart in the first place.
        """
        source = (_PILE_DIR / "build_pile.py").read_text()
        loader = source.split("def _load_coco(", 1)[1].split("\ndef ", 1)[0]
        assert "pc.COCO_VAL_ZIP" in loader
        assert 'val2017.zip"' not in loader, "the builder is spelling the zip path inline again"


def test_selector_reads_only_fields_the_pre_envelope_scan_carries() -> None:
    """The tolerance above is only sound while this stays true.

    ``fb4f4ec03`` added ``bands``, ``bands_compact``, ``n_compact`` and
    ``compact_frac`` per category, none of which a 2026-08-12 scan has. Reading
    an old scan is safe precisely because the selector wants none of them. If
    someone teaches the selector a new field, the old shape must stop being
    silently acceptable -- so this pins the fields it may touch, and fails when
    a new one appears rather than when a rebuild does.
    """
    source = (_PILE_DIR / "build_pile.py").read_text()
    body = source.split("def _band_categories(", 1)[1].split("\ndef ", 1)[0]
    allowed = {"voted_area", "n_images", "union_inflation"}
    read = set(re.findall(r'\["(\w+)"\]', body))
    assert read <= allowed, (
        f"_band_categories reads scan fields a pre-envelope file does not carry: "
        f"{sorted(read - allowed)}. Either drop the read or stop accepting the bare shape."
    )
