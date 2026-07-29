"""Tests for the Max-Patch study's scale-stratified category selection.

``scripts/experiments/max_patch/experiment_config.py`` decides which categories
the study runs on, which is to say it decides what the study can conclude.  The
first run selected by *prevalence*, leaving scale coverage to chance: 7 of 12
categories landed below HAC-leaf scale and only 5 above, so the crossover the
study exists to locate rested on those 5 mixed-sign points.  These tests pin the
scale-band selection that replaced it.

The config module is a loose script, not a package member, so it is loaded by
path.  Everything runs on hand-built region dicts - no dataset, no models.
"""

import importlib.util
import math
from pathlib import Path

import pytest

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "max_patch" / "experiment_config.py"


@pytest.fixture(scope="module")
def cfg():
    spec = importlib.util.spec_from_file_location("_maxpatch_experiment_config", _CONFIG_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _square_box(area: float, x0: float = 0.0, y0: float = 0.0) -> list[float]:
    """An axis-aligned square of exactly *area*, anchored at (x0, y0)."""
    side = math.sqrt(area)
    return [x0, y0, x0 + side, y0 + side]


def _medias(spec: dict[str, float], n: int = 25) -> dict[int, dict]:
    """``n`` medias per category, each carrying one box of the given area.

    Single-instance, so voted area == instance area and union inflation is 1.
    """
    medias: dict[int, dict] = {}
    mid = 0
    for cat, area in spec.items():
        for _ in range(n):
            medias[mid] = {"id": mid, "categories": [cat], "regions": [{"box": _square_box(area), "label": cat}]}
            mid += 1
    return medias


def _counts(spec: dict[str, float], n: int = 25) -> dict[str, int]:
    return {cat: n for cat in spec}


class TestScaleBands:
    def test_bands_tile_the_range_without_gaps_or_overlap(self, cfg):
        bands = cfg.SCALE_BANDS
        assert bands[0][1] == 0.0
        assert bands[-1][2] > 1.0, "the top band must contain a whole-image box"
        # Deliberately not strict: pairing each band with its successor is one
        # shorter than the band list.
        for (_, _, hi), (_, lo_next, _) in zip(bands, bands[1:], strict=False):
            assert hi == lo_next, "bands must abut exactly - a gap silently drops categories"

    def test_band_edges_straddle_the_leaf_scale(self, cfg):
        """The hypothesis' crossover is the HAC leaf, so a band edge sits there."""
        edges = {lo for _, lo, _ in cfg.SCALE_BANDS} | {hi for _, _, hi in cfg.SCALE_BANDS}
        assert any(abs(e - cfg.LEAF_AREA) < 1e-9 for e in edges)
        assert any(abs(e - cfg.PATCH_AREA) < 1e-9 for e in edges)

    @pytest.mark.parametrize(
        ("area", "expected"),
        [
            (0.002, "sub_patch"),
            (0.02, "patch_to_leaf"),
            (0.15, "leaf_to_4x"),
            (0.50, "above_4x"),
            (1.00, "above_4x"),
        ],
    )
    def test_band_for_area(self, cfg, area, expected):
        assert cfg.band_for_area(area) == expected


class TestSelectCategoriesByScale:
    def test_each_band_gets_its_own_categories(self, cfg):
        spec = {"tiny": 0.002, "small": 0.02, "big": 0.15, "huge": 0.50}
        selected, report = cfg.select_categories_by_scale(_medias(spec), _counts(spec))

        assert sorted(selected) == ["big", "huge", "small", "tiny"]
        assert report["bands"]["sub_patch"]["selected"] == ["tiny"]
        assert report["bands"]["patch_to_leaf"]["selected"] == ["small"]
        assert report["bands"]["leaf_to_4x"]["selected"] == ["big"]
        assert report["bands"]["above_4x"]["selected"] == ["huge"]

    def test_whole_image_votes_are_dropped_and_reported(self, cfg):
        """A near-frame voted box is an image-level vote wearing a box.

        Keeping it would rebuild the exact confound that made the boxless
        Caltech-101 arm uninformative about large targets.
        """
        spec = {"bed": 0.50, "sky": 0.95}
        selected, report = cfg.select_categories_by_scale(_medias(spec), _counts(spec))

        assert selected == ["bed"]
        dropped = dict(report["dropped_above_max_voted_area"])
        assert "sky" in dropped
        assert dropped["sky"] == pytest.approx(0.95)
        assert "bed" not in dropped

    def test_low_union_inflation_wins_a_contested_band(self, cfg):
        """When a band is oversubscribed, prefer clean single-object votes.

        ``scattered`` and ``single`` have the same voted-box area, so they land
        in the same band; only ``scattered``'s box is a union over instances
        nowhere near each other, which is not a region a user would drag.
        """
        medias: dict[int, dict] = {}
        mid = 0
        for _ in range(25):
            # One 0.15-area object: voted area == instance area.
            medias[mid] = {
                "id": mid,
                "categories": ["single"],
                "regions": [{"box": _square_box(0.15), "label": "single"}],
            }
            mid += 1
            # Two far-apart specks whose union is also ~0.15 area.
            side = math.sqrt(0.15)
            medias[mid] = {
                "id": mid,
                "categories": ["scattered"],
                "regions": [
                    {"box": [0.0, 0.0, 0.02, 0.02], "label": "scattered"},
                    {"box": [side - 0.02, side - 0.02, side, side], "label": "scattered"},
                ],
            }
            mid += 1

        counts = {"single": 25, "scattered": 25}
        _sel, report = cfg.select_categories_by_scale(medias, counts, n_per_band=1)
        band = report["bands"]["leaf_to_4x"]
        assert band["selected"] == ["single"]
        assert band["not_selected"] == ["scattered"]
        assert band["scales"]["single"]["union_inflation"] == pytest.approx(1.0)

    def test_n_per_band_caps_each_band_independently(self, cfg):
        spec = {f"c{i}": 0.02 for i in range(5)}
        spec["lonely"] = 0.15
        selected, report = cfg.select_categories_by_scale(_medias(spec), _counts(spec), n_per_band=2)

        assert len(report["bands"]["patch_to_leaf"]["selected"]) == 2
        assert len(report["bands"]["patch_to_leaf"]["not_selected"]) == 3
        assert report["bands"]["leaf_to_4x"]["selected"] == ["lonely"]
        assert len(selected) == 3

    def test_rare_categories_are_dropped(self, cfg):
        spec = {"common": 0.02, "rare": 0.02}
        counts = {"common": 25, "rare": cfg._MIN_CATEGORY_COUNT - 1}
        selected, _report = cfg.select_categories_by_scale(_medias(spec), counts)
        assert selected == ["common"]

    def test_selection_is_deterministic(self, cfg):
        spec = {f"c{i}": 0.02 for i in range(8)}
        medias, counts = _medias(spec), _counts(spec)
        a, _ = cfg.select_categories_by_scale(medias, counts, n_per_band=3)
        b, _ = cfg.select_categories_by_scale(medias, counts, n_per_band=3)
        assert a == b


class TestSelectCategoriesDispatch:
    def test_boxed_dataset_uses_scale_bands(self, cfg):
        spec = {"tiny": 0.002, "big": 0.15}
        selected, report = cfg.select_categories(_medias(spec), _counts(spec))
        assert report["mode"] == "scale_bands"
        assert sorted(selected) == ["big", "tiny"]

    def test_boxless_dataset_falls_back_to_prevalence(self, cfg):
        """Caltech-101's shape: categories, no regions, so no scale axis."""
        medias = {i: {"id": i, "category": "a" if i < 30 else "b"} for i in range(60)}
        counts = {"a": 30, "b": 30}
        selected, report = cfg.select_categories(medias, counts)
        assert report["mode"] == "prevalence"
        assert "no ground-truth region boxes" in report["reason"]
        assert sorted(selected) == ["a", "b"]
