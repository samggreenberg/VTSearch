"""Regression (audit follow-up #1): dataset registry multi-process safety.

The old persistence layer wrote the manifest through a fixed ``.tmp`` name and
mutated a process-local in-memory cache, then dumped that cache back to disk.
A second process (e.g. a CLI ``--autodetect`` run against the same data dir)
that registered a dataset would be silently erased the next time the server
process wrote its stale cache back.

The fix routes every mutation through :func:`vtscore.datasets.registry._read_modify_write`,
which re-reads the manifest fresh from disk under a cross-process ``file_lock``
before mutating and writing atomically (unique per-writer temp name).  A writer
therefore merges into the current on-disk state instead of clobbering a
sibling's commit.  These tests reproduce the clobber scenario by writing to the
on-disk manifest directly to stand in for the sibling process's commit.
"""

from __future__ import annotations

import json
import threading

from vtscore.datasets import registry


def _read_disk_ids() -> set[str]:
    return {e["id"] for e in json.loads(registry.REGISTRY_PATH.read_text(encoding="utf-8"))}


def _commit_sibling_entry(entry_id: str, name: str = "sibling") -> None:
    """Append an entry straight to the on-disk manifest.

    Stands in for a separate process committing a registration while *this*
    process holds a stale in-memory cache.
    """
    on_disk = json.loads(registry.REGISTRY_PATH.read_text(encoding="utf-8"))
    on_disk.append({"id": entry_id, "name": name, "media_type": "audio", "pkl_path": f"{entry_id}.pkl"})
    registry.REGISTRY_PATH.write_text(json.dumps(on_disk), encoding="utf-8")


class TestRegistryMultiprocessSafety:
    def test_register_merges_concurrent_sibling_write(self):
        registry.reset_for_tests()
        a = registry.register_dataset(name="A", media_type="audio", num_items=1, pkl_path="a.pkl")

        # A sibling process commits its own registration to disk. This process's
        # in-memory cache does not know about it.
        _commit_sibling_entry("sibling-b")

        # Registering here must NOT drop the sibling's entry.
        c = registry.register_dataset(name="C", media_type="audio", num_items=1, pkl_path="c.pkl")

        assert _read_disk_ids() == {a["id"], "sibling-b", c["id"]}

    def test_unregister_preserves_concurrent_sibling_write(self):
        registry.reset_for_tests()
        a = registry.register_dataset(name="A", media_type="audio", num_items=1, pkl_path="a.pkl")
        _commit_sibling_entry("sibling-b")

        assert registry.unregister_dataset(a["id"]) is True

        # Removing A must leave the sibling's B intact, not rewind to the stale
        # cache (which never contained B).
        assert _read_disk_ids() == {"sibling-b"}

    def test_rename_preserves_concurrent_sibling_write(self):
        registry.reset_for_tests()
        a = registry.register_dataset(name="A", media_type="audio", num_items=1, pkl_path="a.pkl")
        _commit_sibling_entry("sibling-b")

        assert registry.rename_dataset(a["id"], "A-renamed") is True

        assert _read_disk_ids() == {a["id"], "sibling-b"}

    def test_temp_file_is_not_a_fixed_name(self):
        # A fixed ``<name>.tmp`` lets two concurrent writers truncate each
        # other's in-flight temp file. The atomic writer now embeds pid + uuid,
        # so no fixed sibling temp file is left behind after a write.
        registry.reset_for_tests()
        registry.register_dataset(name="A", media_type="audio", num_items=1, pkl_path="a.pkl")
        fixed_tmp = registry.REGISTRY_PATH.with_suffix(".tmp")
        assert not fixed_tmp.exists()

    def test_unregister_deletes_sidecar_files_sharing_pkl_stem(self, tmp_path):
        """``unregister_dataset`` must also delete any sidecar file that shares
        the pkl's stem (e.g. a future mmap embedding-matrix cache), not just
        the pkl itself, so such sidecars can't be orphaned on delete/expiry."""
        registry.reset_for_tests()
        pkl_path = tmp_path / "ds_deadbeef.pkl"
        pkl_path.write_bytes(b"pkl-bytes")
        sidecar = tmp_path / "ds_deadbeef.emb.npy"
        sidecar.write_bytes(b"sidecar-bytes")
        cids_sidecar = tmp_path / "ds_deadbeef.cids.npy"
        cids_sidecar.write_bytes(b"cids-bytes")
        unrelated = tmp_path / "ds_other.pkl"
        unrelated.write_bytes(b"unrelated-bytes")

        entry = registry.register_dataset(name="A", media_type="audio", num_items=1, pkl_path=str(pkl_path))

        assert registry.unregister_dataset(entry["id"]) is True

        assert not pkl_path.exists()
        assert not sidecar.exists()
        assert not cids_sidecar.exists()
        assert unrelated.exists(), "unregister must not touch files with a different stem"

    def test_concurrent_registers_all_survive(self):
        # Many threads registering at once must all land on disk; the flock plus
        # fresh re-read serialises the read-modify-write so none clobber another.
        registry.reset_for_tests()
        n = 12
        barrier = threading.Barrier(n)

        def worker(i: int) -> None:
            barrier.wait()
            registry.register_dataset(name=f"D{i}", media_type="audio", num_items=1, pkl_path=f"d{i}.pkl")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(_read_disk_ids()) == n
