"""Regression (audit follow-up #1): detector registry multi-process safety.

The detector registry mirrors the dataset registry and had the identical
fixed-``.tmp`` + stale-cache clobber bug. Every mutation now routes through
:func:`vtscore.detectors.registry._read_modify_write` (fresh re-read under a
cross-process ``file_lock`` + atomic unique-temp write). See the dataset twin
in ``tests_lib/datasets/test_registry_multiprocess_safety.py``.
"""

from __future__ import annotations

import json
import threading

from vtscore.detectors import registry


def _read_disk_ids() -> set[str]:
    return {e["id"] for e in json.loads(registry.REGISTRY_PATH.read_text(encoding="utf-8"))}


def _commit_sibling_entry(entry_id: str, name: str = "sibling") -> None:
    on_disk = json.loads(registry.REGISTRY_PATH.read_text(encoding="utf-8"))
    on_disk.append({"id": entry_id, "name": name, "media_type": "audio", "created_by": "default"})
    registry.REGISTRY_PATH.write_text(json.dumps(on_disk), encoding="utf-8")


class TestDetectorRegistryMultiprocessSafety:
    def test_register_merges_concurrent_sibling_write(self):
        registry.reset_for_tests()
        a = registry.register_detector(name="A", media_type="audio")
        _commit_sibling_entry("sibling-b")

        c = registry.register_detector(name="C", media_type="audio")

        assert _read_disk_ids() == {a["id"], "sibling-b", c["id"]}

    def test_unregister_preserves_concurrent_sibling_write(self):
        registry.reset_for_tests()
        a = registry.register_detector(name="A", media_type="audio")
        _commit_sibling_entry("sibling-b")

        assert registry.unregister_detector(a["id"]) is True

        assert _read_disk_ids() == {"sibling-b"}

    def test_record_embedder_merges_concurrent_sibling_write(self):
        registry.reset_for_tests()
        a = registry.register_detector(name="A", media_type="audio")
        _commit_sibling_entry("sibling-b")

        registry.record_detector_embedder(a["id"], "laion_clap")

        ids = _read_disk_ids()
        assert ids == {a["id"], "sibling-b"}

    def test_record_embedder_unchanged_skips_write(self):
        # The per-training-cycle fast path must not rewrite the manifest when the
        # embedder is already stamped.
        registry.reset_for_tests()
        a = registry.register_detector(name="A", media_type="audio", embedder="laion_clap")
        before = registry.REGISTRY_PATH.read_text(encoding="utf-8")

        registry.record_detector_embedder(a["id"], "laion_clap")

        assert registry.REGISTRY_PATH.read_text(encoding="utf-8") == before

    def test_temp_file_is_not_a_fixed_name(self):
        registry.reset_for_tests()
        registry.register_detector(name="A", media_type="audio")
        fixed_tmp = registry.REGISTRY_PATH.with_suffix(".tmp")
        assert not fixed_tmp.exists()

    def test_concurrent_registers_all_survive(self):
        registry.reset_for_tests()
        n = 12
        barrier = threading.Barrier(n)

        def worker(i: int) -> None:
            barrier.wait()
            registry.register_detector(name=f"D{i}", media_type="audio")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(_read_disk_ids()) == n
