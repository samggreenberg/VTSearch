"""Regression: detector registry reads must reflect the manifest on disk (#3627).

``vtscore.detectors.registry`` filled its in-memory cache once and thereafter
refreshed it only from *this* process's own mutations, so every read was blind
to a sibling writer — a CLI run against the same data dir (the very case
:func:`_read_modify_write` takes a cross-process lock for), a second server, a
hand-edited manifest.  When a CLI process cleared six finished detectors, the
running app's ``GET /api/detectors/registry`` — the view a person is actually
looking at — kept listing 15 detectors while the manifest and
``GET /api/detectors`` (which reads the detector *files*) both said 9.  Nothing
failed loudly; the only workaround was a restart or an incidental mutation.

The fix ports the dataset registry's ``(mtime_ns, size)`` stamp (#3167): reads
re-parse exactly when the file has actually changed.  These tests write to the
on-disk manifest directly to stand in for the sibling process's commit.
"""

from __future__ import annotations

import json

from vtscore.detectors import registry


def _write_disk(entries: list[dict[str, object]]) -> None:
    registry.REGISTRY_PATH.write_text(json.dumps(entries), encoding="utf-8")


def _read_disk() -> list[dict[str, object]]:
    return json.loads(registry.REGISTRY_PATH.read_text(encoding="utf-8"))


def _commit_sibling_entry(entry_id: str, name: str = "sibling") -> None:
    """Append an entry straight to the on-disk manifest.

    Stands in for a separate process committing a registration while *this*
    process holds a cache filled before that write.
    """
    on_disk = _read_disk()
    on_disk.append({"id": entry_id, "name": name, "media_type": "audio", "created_by": "default", "readers": []})
    _write_disk(on_disk)


class TestDetectorRegistryReadsSeeDiskWrites:
    def test_list_detectors_picks_up_a_sibling_write(self):
        registry.reset_for_tests()
        a = registry.register_detector(name="A", media_type="audio")
        assert {e["id"] for e in registry.list_detectors()} == {a["id"]}

        _commit_sibling_entry("sibling-b")

        assert {e["id"] for e in registry.list_detectors()} == {a["id"], "sibling-b"}, (
            "a detector registered by another process must be visible without a restart"
        )

    def test_get_detector_picks_up_a_sibling_write(self):
        registry.reset_for_tests()
        registry.register_detector(name="A", media_type="audio")
        assert registry.get_detector("sibling-b") is None

        _commit_sibling_entry("sibling-b")

        entry = registry.get_detector("sibling-b")
        assert entry is not None and entry["name"] == "sibling", (
            f"the entry is on disk; the API must not deny it exists, got {entry!r}"
        )

    def test_list_detectors_picks_up_a_sibling_deletion(self):
        """The reported failure: a sibling clears detectors, the dashboard keeps listing them."""
        registry.reset_for_tests()
        a = registry.register_detector(name="A", media_type="audio")
        registry.register_detector(name="B", media_type="audio")
        assert len(registry.list_detectors()) == 2

        _write_disk([e for e in _read_disk() if e["id"] == a["id"]])

        assert {e["id"] for e in registry.list_detectors()} == {a["id"]}, (
            "a detector deleted by another process must disappear without a restart"
        )

    def test_this_process_own_mutation_stays_visible(self):
        """The freshness check must not discard a write this process just made."""
        registry.reset_for_tests()
        a = registry.register_detector(name="A", media_type="audio")
        assert registry.rename_detector(a["id"], "A-renamed") is True

        renamed = registry.get_detector(a["id"])
        assert renamed is not None and renamed["name"] == "A-renamed"
        assert [e["name"] for e in registry.list_detectors()] == ["A-renamed"]

    def test_unchanged_manifest_is_not_re_read(self, monkeypatch):
        """The freshness check is a stat, not a re-parse on every read."""
        registry.reset_for_tests()
        registry.register_detector(name="A", media_type="audio")

        reads = []
        real_load = registry._load
        monkeypatch.setattr(registry, "_load", lambda: (reads.append(1), real_load())[1])

        registry.list_detectors()
        registry.list_detectors()

        assert reads == [], f"an unchanged manifest should not be parsed again, got {len(reads)} reads"
