"""Integrity tests for the written-down exact demo-dataset counts.

These guard ``vtscore.datasets.demo_counts.DEMO_MEDIA_COUNTS`` — the table that
makes the dataset picker's advertised ``# Media`` figure accurate for sources
with uneven category sizes (see the module docstring).
"""

from __future__ import annotations

from vtscore.datasets.config import DEMO_DATASETS
from vtscore.datasets.demo_counts import DEMO_MEDIA_COUNTS, exact_demo_count


class TestDemoMediaCounts:
    def test_keys_are_real_demo_dataset_ids(self):
        unknown = set(DEMO_MEDIA_COUNTS) - set(DEMO_DATASETS)
        assert not unknown, f"DEMO_MEDIA_COUNTS has ids not in DEMO_DATASETS: {sorted(unknown)}"

    def test_values_are_positive_ints(self):
        for did, count in DEMO_MEDIA_COUNTS.items():
            assert isinstance(count, int), f"{did} count must be int, got {type(count)}"
            assert count > 0, f"{did} count must be positive, got {count}"

    def test_keys_sorted_for_clean_diffs(self):
        keys = list(DEMO_MEDIA_COUNTS)
        assert keys == sorted(keys), "keep DEMO_MEDIA_COUNTS sorted by id"

    def test_exact_demo_count_lookup(self):
        for did, count in DEMO_MEDIA_COUNTS.items():
            assert exact_demo_count(did) == count

    def test_exact_demo_count_unknown_is_none(self):
        assert exact_demo_count("definitely_not_a_dataset") is None

    def test_caltech101_recorded_counts(self):
        # Verified against the loader: S/M confirmed from the cached pkls
        # (412/838), L/A from the same real per-category folder sizes via the
        # loader's exact slice formula (1704/2954). Locks in the audit fix
        # (these were advertised as 300/600/1250/2150 under the old estimate).
        assert DEMO_MEDIA_COUNTS["caltech101_s"] == 412
        assert DEMO_MEDIA_COUNTS["caltech101_m"] == 838
        assert DEMO_MEDIA_COUNTS["caltech101_l"] == 1704
        assert DEMO_MEDIA_COUNTS["caltech101_a"] == 2954

    def test_caltech101_partitions_sum_to_full(self):
        # The S/M/L fractional slices [0,1/7], [1/7,3/7], [3/7,1] partition each
        # category, so their counts must sum to the (A)ll variant.
        s = DEMO_MEDIA_COUNTS["caltech101_s"]
        m = DEMO_MEDIA_COUNTS["caltech101_m"]
        ll = DEMO_MEDIA_COUNTS["caltech101_l"]
        a = DEMO_MEDIA_COUNTS["caltech101_a"]
        assert s + m + ll == a
