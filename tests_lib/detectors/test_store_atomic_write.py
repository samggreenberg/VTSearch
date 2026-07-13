"""Regression tests for the atomic detector-file writer in ``vtscore.detectors.store``.

A renamed (or otherwise re-saved) detector used to write its JSON through a
*fixed* sibling tmp path ``<name>.json.tmp``.  Two writers racing on the same
target (e.g. a double-fired rename request, or a rename overlapping a label
sync) therefore shared one tmp file: the first ``os.replace`` consumed it and
the second chased a tmp that was already renamed away, surfacing as

    FileNotFoundError: '<name>.json.tmp' -> '<name>.json'

The writer now uses a per-writer unique tmp suffix (PID + UUID), matching
``vtsearch.settings_store._atomic_write`` and ``vtscore.io.atomic_write_text``.
"""

from __future__ import annotations

import re

import pytest

from vtscore.detectors import store
from vtscore.detectors.store import _MAX_SLUG_LENGTH, _read_detector, _slug, _write_detector


class TestDetectorAtomicWrite:
    def test_tmp_filename_is_unique_per_writer(self, tmp_path, monkeypatch):
        """The tmp file must NOT be the fixed ``<name>.json.tmp`` (the racy
        name); it carries a PID + hex suffix so concurrent writers can't
        clobber or chase each other's in-flight tmp.
        """
        seen: list[str] = []
        real_replace = store.os.replace

        def spy_replace(src, dst, *args, **kwargs):
            seen.append(str(src))
            return real_replace(src, dst, *args, **kwargs)

        monkeypatch.setattr(store.os, "replace", spy_replace)

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

    def test_slug_truncates_overlong_name(self):
        """A pathological name must slug to a length that keeps ``<slug>.json``

        and its ``.tmp`` sibling under the filesystem ``NAME_MAX`` (255), so a
        write can't raise ``OSError`` (and leak the absolute path).
        """
        slug = _slug("a" * 1000)
        assert len(slug) <= _MAX_SLUG_LENGTH

    def test_slug_overlong_names_do_not_collide(self):
        """Two distinct long names sharing a prefix get distinct slugs (via the

        appended content hash), so truncation can't silently overwrite files.
        """
        a = _slug("z" * 500 + "_alpha")
        b = _slug("z" * 500 + "_beta")
        assert a != b

    def test_slug_short_name_unchanged(self):
        """The truncation path is inert for ordinary names."""
        assert _slug("Mammals of North America") == "mammals_of_north_america"

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
