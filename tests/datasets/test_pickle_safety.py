"""Tests for the restricted pickle unpickler (RCE prevention).

Verifies that ``RestrictedUnpickler`` and ``safe_pickle_load`` block
arbitrary code execution while still allowing legitimate VTSearch
dataset pickles (plain Python types + numpy arrays).
"""

import io
import pickle

import numpy as np
import pytest

from vtscore.datasets.loader import (
    export_dataset_to_file,
    load_dataset_from_pickle,
    load_dataset_from_pickle_chunked,
    safe_pickle_load,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ArbitraryObj:
    """A plain class with no __reduce__; used to test that arbitrary classes
    are rejected by the restricted unpickler."""

    pass


def _dumps(obj):
    """Pickle an object to bytes."""
    buf = io.BytesIO()
    pickle.dump(obj, buf)
    return buf.getvalue()


def _safe_loads(data: bytes, **kwargs):
    """Deserialise bytes via safe_pickle_load."""
    return safe_pickle_load(io.BytesIO(data), **kwargs)


def _make_malicious_pickle() -> bytes:
    """Build a pickle payload that would execute ``os.system('echo pwned')``
    if loaded with the standard ``pickle.load``.
    """
    # Uses the classic __reduce__ exploit: the unpickler calls
    # os.system("echo pwned") and returns the result.
    import os

    class Exploit:
        def __reduce__(self):
            return (os.system, ("echo pwned",))

    return _dumps(Exploit())


# ---------------------------------------------------------------------------
# RestrictedUnpickler — blocks malicious payloads
# ---------------------------------------------------------------------------


class TestRestrictedUnpicklerBlocks:
    """Verify that dangerous pickle payloads are rejected."""

    def test_blocks_os_system(self):
        """os.system() call via __reduce__ must be rejected."""
        payload = _make_malicious_pickle()
        with pytest.raises(pickle.UnpicklingError, match="Forbidden pickle class"):
            _safe_loads(payload)

    def test_blocks_subprocess(self):
        """subprocess.Popen via __reduce__ must be rejected."""
        import subprocess

        class Exploit:
            def __reduce__(self):
                return (subprocess.Popen, (["echo", "pwned"],))

        payload = _dumps(Exploit())
        with pytest.raises(pickle.UnpicklingError, match="Forbidden pickle class"):
            _safe_loads(payload)

    def test_blocks_eval(self):
        """builtins.eval via __reduce__ must be rejected."""

        class Exploit:
            def __reduce__(self):
                return (eval, ("1+1",))

        payload = _dumps(Exploit())
        with pytest.raises(pickle.UnpicklingError, match="Forbidden pickle class"):
            _safe_loads(payload)

    def test_blocks_exec(self):
        """builtins.exec via __reduce__ must be rejected."""

        class Exploit:
            def __reduce__(self):
                return (exec, ("import os",))

        payload = _dumps(Exploit())
        with pytest.raises(pickle.UnpicklingError, match="Forbidden pickle class"):
            _safe_loads(payload)

    def test_blocks_arbitrary_class(self):
        """An arbitrary user-defined class must be rejected."""
        # Build a pickle stream that references a module-level class by
        # manually constructing the payload (local classes can't be pickled).
        payload = _dumps(_ArbitraryObj())
        with pytest.raises(pickle.UnpicklingError, match="Forbidden pickle class"):
            _safe_loads(payload)

    def test_error_message_contains_module_and_name(self):
        """The error message should identify which class was blocked."""
        payload = _make_malicious_pickle()
        # On Linux, os.system is stored as posix.system in the pickle stream
        with pytest.raises(pickle.UnpicklingError, match=r"system"):
            _safe_loads(payload)


# ---------------------------------------------------------------------------
# RestrictedUnpickler — allows safe types
# ---------------------------------------------------------------------------


class TestRestrictedUnpicklerAllows:
    """Verify that legitimate data types load correctly."""

    def test_plain_dict(self):
        data = {"key": "value", "num": 42, "nested": {"a": [1, 2, 3]}}
        assert _safe_loads(_dumps(data)) == data

    def test_list_and_tuple(self):
        data = [1, "two", (3.0, None, True, False)]
        assert _safe_loads(_dumps(data)) == data

    def test_bytes_and_bytearray(self):
        data = {"raw": b"\x00\x01\x02", "ba": bytearray(b"\x03\x04")}
        result = _safe_loads(_dumps(data))
        assert result["raw"] == b"\x00\x01\x02"
        assert result["ba"] == bytearray(b"\x03\x04")

    def test_numpy_array(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = _safe_loads(_dumps(arr))
        np.testing.assert_array_equal(result, arr)

    def test_numpy_in_dict(self):
        data = {"embedding": np.zeros(10, dtype=np.float32), "id": 1}
        result = _safe_loads(_dumps(data))
        np.testing.assert_array_equal(result["embedding"], data["embedding"])
        assert result["id"] == 1

    def test_ordered_dict(self):
        from collections import OrderedDict

        data = OrderedDict([("a", 1), ("b", 2)])
        result = _safe_loads(_dumps(data))
        assert list(result.keys()) == ["a", "b"]

    def test_set_and_frozenset(self):
        data = {"s": {1, 2, 3}, "fs": frozenset([4, 5])}
        result = _safe_loads(_dumps(data))
        assert result["s"] == {1, 2, 3}
        assert result["fs"] == frozenset([4, 5])

    def test_none_and_booleans(self):
        data = {"none": None, "true": True, "false": False}
        assert _safe_loads(_dumps(data)) == data


# ---------------------------------------------------------------------------
# Integration: export_dataset_to_file → load_dataset_from_pickle round-trip
# ---------------------------------------------------------------------------


class TestSafePickleRoundTrip:
    """Exported datasets must load successfully through the restricted unpickler."""

    def test_export_import_round_trip(self, tmp_path):
        """A dataset exported by VTSearch should load cleanly via safe_pickle_load."""
        medias = {
            1: {
                "id": 1,
                "type": "audio",
                "duration": 1.0,
                "file_size": 1024,
                "md5": "abc123",
                "embedding": np.random.RandomState(42).randn(512).astype(np.float32),
                "filename": "test.wav",
                "category": "test",
                "origin": {"importer": "test", "params": {}},
                "origin_name": "test.wav",
                "media_bytes": b"\x00" * 100,
                "media_string": None,
                "media_path": None,
                "word_count": None,
                "character_count": None,
                "width": None,
                "height": None,
            },
        }
        data_bytes = export_dataset_to_file(medias)
        pkl_path = tmp_path / "test.pkl"
        pkl_path.write_bytes(data_bytes)

        loaded = {}
        load_dataset_from_pickle(pkl_path, loaded)
        assert len(loaded) == 1
        assert loaded[1]["md5"] == "abc123"

    def test_chunked_round_trip(self, tmp_path):
        """Chunked loading should also work through the restricted unpickler."""
        medias = {}
        for i in range(5):
            medias[i + 1] = {
                "id": i + 1,
                "type": "audio",
                "duration": 1.0,
                "file_size": 1024,
                "md5": f"md5_{i}",
                "embedding": np.random.RandomState(42).randn(512).astype(np.float32),
                "filename": f"test_{i}.wav",
                "category": "test",
                "origin": None,
                "origin_name": f"test_{i}.wav",
                "media_bytes": b"\x00" * 100,
                "media_string": None,
                "media_path": None,
                "word_count": None,
                "character_count": None,
                "width": None,
                "height": None,
            }
        data_bytes = export_dataset_to_file(medias)
        pkl_path = tmp_path / "test.pkl"
        pkl_path.write_bytes(data_bytes)

        chunks = list(load_dataset_from_pickle_chunked(pkl_path, chunk_size=2, thin=True))
        total = sum(len(c) for c in chunks)
        assert total == 5


# ---------------------------------------------------------------------------
# Integration: malicious pickle rejected by load_dataset_from_pickle
# ---------------------------------------------------------------------------


class TestMaliciousPickleInLoader:
    """Verify that malicious pickles are rejected when loaded through the
    main dataset loading functions."""

    def test_load_dataset_from_pickle_rejects_rce(self, tmp_path):
        """load_dataset_from_pickle must reject an RCE payload."""
        pkl_path = tmp_path / "evil.pkl"
        pkl_path.write_bytes(_make_malicious_pickle())
        target = {}
        with pytest.raises(pickle.UnpicklingError, match="Forbidden pickle class"):
            load_dataset_from_pickle(pkl_path, target)

    def test_load_dataset_from_pickle_chunked_rejects_rce(self, tmp_path):
        """load_dataset_from_pickle_chunked must reject an RCE payload."""
        pkl_path = tmp_path / "evil.pkl"
        pkl_path.write_bytes(_make_malicious_pickle())
        with pytest.raises(pickle.UnpicklingError, match="Forbidden pickle class"):
            list(load_dataset_from_pickle_chunked(pkl_path, chunk_size=10))

    def test_stage_file_rejects_rce(self, client, tmp_path):
        """The /api/dataset/stage-file endpoint must not execute RCE payloads."""
        payload = _make_malicious_pickle()
        data = {"file": (io.BytesIO(payload), "evil.pkl")}
        resp = client.post(
            "/api/dataset/stage-file",
            data=data,
            content_type="multipart/form-data",
        )
        # The endpoint catches exceptions and returns count=0 for unparseable files
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["count"] == 0
        assert body["media_type"] == "unknown"
