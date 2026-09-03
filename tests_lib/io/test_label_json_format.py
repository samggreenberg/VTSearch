"""Tests for ``vtscore.labels.json_format``.

The server-JSON label importer and the server-JSON labelset source both read
this shape, and both used to validate it with their own copy of the check.
These tests pin the shared behaviour so the two paths can't drift apart again.
"""

from __future__ import annotations

import pytest

from vtscore.labels.json_format import extract_labels, require_label_object


class TestRequireLabelObject:
    def test_returns_the_object_unchanged(self):
        data = {"labels": [], "detector_meta": {"name": "d"}}
        assert require_label_object(data) is data

    @pytest.mark.parametrize("data", [[], "text", 3, None])
    def test_rejects_non_object_top_level(self, data):
        with pytest.raises(ValueError, match="object at the top level"):
            require_label_object(data)

    def test_rejects_missing_labels_key(self):
        with pytest.raises(ValueError, match="top-level 'labels' list"):
            require_label_object({"data": []})

    def test_rejects_non_list_labels(self):
        with pytest.raises(ValueError, match="top-level 'labels' list"):
            require_label_object({"labels": {"md5": "abc"}})


class TestExtractLabels:
    def test_returns_entries(self):
        entries = [{"md5": "abc", "label": "good"}, {"md5": "def", "label": "bad"}]
        assert extract_labels({"labels": entries}) == entries

    def test_empty_list_is_valid(self):
        assert extract_labels({"labels": []}) == []

    def test_drops_non_dict_entries(self):
        """One stray row shouldn't cost the user the rest of a hand-edited file."""
        data = {"labels": [{"md5": "abc"}, None, "junk", 7, {"md5": "def"}]}
        assert extract_labels(data) == [{"md5": "abc"}, {"md5": "def"}]

    def test_ignores_sibling_keys(self):
        data = {"labels": [{"md5": "abc"}], "detector_meta": {"name": "d"}}
        assert extract_labels(data) == [{"md5": "abc"}]

    @pytest.mark.parametrize("data", [[], {"data": []}, {"labels": "nope"}])
    def test_rejects_wrong_shapes(self, data):
        with pytest.raises(ValueError):
            extract_labels(data)


class TestPluginsShareTheHelper:
    """The importer and the source must agree, entry for entry and error for error."""

    def _importer(self):
        from vtscore.labels.importers import get_label_importer

        return get_label_importer("server_json_file")

    def _source(self):
        from vtscore.labels.sources import get_labelset_source

        return get_labelset_source("server_json_file")

    def test_both_drop_the_same_junk_entries(self, tmp_path):
        import json

        p = tmp_path / "labels.json"
        p.write_text(json.dumps({"labels": [{"md5": "abc"}, None, {"md5": "def"}]}))

        via_importer = self._importer().run({"filepath": str(p)})
        via_source = self._source().load({"filepath": str(p)})
        assert via_importer == via_source == [{"md5": "abc"}, {"md5": "def"}]

    def test_both_reject_a_non_object_top_level(self, tmp_path):
        import json

        p = tmp_path / "labels.json"
        p.write_text(json.dumps([{"md5": "abc"}]))

        with pytest.raises(ValueError, match="object at the top level"):
            self._importer().run({"filepath": str(p)})
        with pytest.raises(ValueError, match="object at the top level"):
            self._source().load({"filepath": str(p)})
