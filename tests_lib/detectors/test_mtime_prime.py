"""The rehydrate pre-check's TTL mtime cache is primed by our own writes (#3853).

``ensure_votes_match_active_dataset`` compares the detector context's
``cached_labelset_mtime`` stamp against a TTL-cached stat of the detector
file.  A writer that stamps the context (the per-vote labelset rewrite) and
does not also refresh that cache leaves every request in the next second
comparing the new stamp against the old cached mtime, re-parsing the whole
detector JSON to discover under the lock that nothing changed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vtscore.detectors import dataset_sync


@pytest.fixture(autouse=True)
def _clean_cache():
    dataset_sync.reset_mtime_cache_for_tests()
    yield
    dataset_sync.reset_mtime_cache_for_tests()


def test_primed_value_is_served_without_a_stat(monkeypatch, tmp_path):
    path = tmp_path / "det.json"
    path.write_text("{}")
    dataset_sync.prime_detector_file_mtime(path, 1234.5)

    def _no_stat(_path):
        raise AssertionError("pre-check should not stat a freshly primed file")

    monkeypatch.setattr(dataset_sync, "_detector_file_mtime", _no_stat)
    assert dataset_sync._detector_file_mtime_cached(path) == 1234.5


def test_prime_replaces_a_stale_cached_value(tmp_path):
    path = tmp_path / "det.json"
    path.write_text("{}")
    stale = dataset_sync._detector_file_mtime_cached(path)  # caches the real stat
    dataset_sync.prime_detector_file_mtime(path, stale + 10.0)
    assert dataset_sync._detector_file_mtime_cached(path) == stale + 10.0


def test_prime_is_keyed_by_path_string(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    for p in (a, b):
        p.write_text("{}")
    dataset_sync.prime_detector_file_mtime(a, 1.0)
    dataset_sync.prime_detector_file_mtime(Path(str(b)), 2.0)
    assert dataset_sync._detector_file_mtime_cached(a) == 1.0
    assert dataset_sync._detector_file_mtime_cached(b) == 2.0
