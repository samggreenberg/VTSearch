"""Unit tests for region-localization metrics (``box_iou`` + CorLoc sweep).

Pure-numpy library-tier tests: no models, no Flask, deterministic.
"""

from __future__ import annotations

import math

from vtscore.eval.metrics import (
    RegionLocalizationMetrics,
    box_iou,
    compute_localization_metrics,
)


class TestBoxIou:
    def test_identical_boxes(self):
        assert box_iou((0.1, 0.1, 0.5, 0.5), (0.1, 0.1, 0.5, 0.5)) == 1.0

    def test_disjoint_boxes(self):
        assert box_iou((0.0, 0.0, 0.2, 0.2), (0.5, 0.5, 0.8, 0.8)) == 0.0

    def test_half_overlap_known_value(self):
        # Two unit-ish boxes overlapping in exactly half their area.
        # a = [0,0,1,1] area 1; b = [0.5,0,1.5,1] area 1; intersection = 0.5x1 = 0.5
        # union = 1 + 1 - 0.5 = 1.5 -> IoU = 0.5 / 1.5 = 1/3
        assert math.isclose(box_iou((0.0, 0.0, 1.0, 1.0), (0.5, 0.0, 1.5, 1.0)), 1.0 / 3.0)

    def test_contained_box(self):
        # b fully inside a: intersection = area_b, union = area_a
        # a area = 1.0, b area = 0.25 -> IoU = 0.25
        assert math.isclose(box_iou((0.0, 0.0, 1.0, 1.0), (0.25, 0.25, 0.75, 0.75)), 0.25)

    def test_degenerate_box_is_zero(self):
        assert box_iou((0.2, 0.2, 0.2, 0.5), (0.2, 0.2, 0.5, 0.5)) == 0.0


class TestComputeLocalizationMetrics:
    def test_empty_pairs_all_zero(self):
        m = compute_localization_metrics([])
        assert isinstance(m, RegionLocalizationMetrics)
        assert m.mean_iou == 0.0
        assert m.num_localizable == 0
        assert m.corloc == {0.3: 0.0, 0.5: 0.0, 0.7: 0.0}

    def test_perfect_localization(self):
        gt = (0.1, 0.1, 0.5, 0.5)
        pairs = [(gt, gt), (gt, gt)]
        m = compute_localization_metrics(pairs)
        assert m.mean_iou == 1.0
        assert m.num_localizable == 2
        assert m.corloc[0.3] == 1.0
        assert m.corloc[0.5] == 1.0
        assert m.corloc[0.7] == 1.0

    def test_none_prediction_counts_as_miss(self):
        gt = (0.0, 0.0, 1.0, 1.0)
        # One perfect, one missing (None) -> IoUs [1.0, 0.0]
        m = compute_localization_metrics([(gt, gt), (None, gt)])
        assert math.isclose(m.mean_iou, 0.5)
        assert m.corloc[0.5] == 0.5  # only the perfect one clears 0.5
        assert m.num_localizable == 2

    def test_threshold_sweep_separates_iou_levels(self):
        gt = (0.0, 0.0, 1.0, 1.0)
        # IoU exactly 1/3 (~0.333): clears 0.3 but not 0.5 or 0.7.
        pred = (0.5, 0.0, 1.5, 1.0)
        m = compute_localization_metrics([(pred, gt)], iou_thresholds=(0.3, 0.5, 0.7))
        assert m.corloc[0.3] == 1.0
        assert m.corloc[0.5] == 0.0
        assert m.corloc[0.7] == 0.0
