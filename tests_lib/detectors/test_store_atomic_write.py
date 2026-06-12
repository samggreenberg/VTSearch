"""Regression tests for the atomic detector-file writer in ``vtscore.detectors.store``.

A renamed (or otherwise re-saved) detector used to write its JSON through a
*fixed* sibling tmp path ``<name>.json.tmp``.  Two writers racing on the same
target (e.g. a double-fired rename request, or a rename overlapping a label
sync) therefore shared one tmp file: the first ``os.replace`` consumed it and
the second chased a tmp that was already renamed away, surfacing as

    FileNotFoundError: '<name>.json.tmp' -> '<name>.json'

The writer now uses a per-writer unique tmp suffix (PID + UUID), matching
``vtsearch.settings._atomic_write`` and ``vtscore.io.atomic_write_text``.
"""

from __future__ import annotations

import re

import pytest

from vtscore.detectors import store
from vtscore.detectors.store import _read_detector, _write_detector


class TestDetectorAtomicWrite:
    def test_tmp_filename_is_unique_per_writer(self, tmp_path, monkeypatch):
        """The tmp file must NOT be the fixed ``<name>.json.tmp`` (the racy
        name); it carries a PID + hex suffix so concurrent writers can't
        clobber or chase each other's in-flight tmp.
        """
        seen: list[str] = []
        real_open = store.open

        def spy_open(file, *args, **kwargs):
            seen.append(str(file))
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr(store, "open", spy_open, raising=False)

        path = tmp_path / "mammals.json"
        _write_detector(path, {"name": "mammals", "labelset": {"labels": []}})

        assert _read_detector(path) == {"name": "mammals", "labelset": {"labels": []}}
        assert len(seen) == 1
        tmp_name = seen[0].rsplit("/", 1)[-1]
        # Racy fixed name is rejected; unique pattern is required.
        assert tmp_name != "mammals.json.tmp"
        assert re.match(r"^mammals\.json\.\d+\.[0-9a-f]{32}\.tmp$", tmp_name), tmp_name

    def test_no_tmp_left_behind_on_success(self, tmp_path):
        path = tmp_path / "mammals.json"
        _write_detector(path, {"name": "mammals"})
        leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
        assert leftovers == []

    def test_tmp_cleaned_up_on_replace_error(self, tmp_path, monkeypatch):
        """A failed ``os.replace`` must not leak the half-written tmp file."""
        path = tmp_path / "mammals.json"

        def boom(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(store.os, "replace", boom)
        with pytest.raises(OSError, match="disk full"):
            _write_detector(path, {"name": "mammals"})

        leftovers = [p.name for p in tmp_path.iterdir()]
        assert leftovers == []
