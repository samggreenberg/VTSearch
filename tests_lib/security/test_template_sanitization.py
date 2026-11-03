"""Tests for ``sanitize_template_value`` (path-template field sanitization).

Server-side sync sources and exporters splice user-controlled field values
into admin-defined path templates (``data/labels/{detector_name}.json``).
``sanitize_template_value`` must neutralise any value that could escape the
intended directory when substituted (M32).
"""

from __future__ import annotations

import pytest

from vtscore.security.path_validation import sanitize_template_value


class TestSanitizeTemplateValue:
    @pytest.mark.parametrize("nav", ["", ".", "..", "...", "....", "........"])
    def test_navigation_and_empty_tokens_collapse_to_underscore(self, nav):
        """Empty and any all-dots token (including ``...``) become ``_`` so
        they can't address a parent/current directory (M32)."""
        assert sanitize_template_value(nav) == "_"

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("../etc/passwd", ".._etc_passwd"),
            ("a/b", "a_b"),
            ("a\\b", "a_b"),
            ("dog\0bark", "dog_bark"),
            ("dog_bark", "dog_bark"),
            ("normal-name.json", "normal-name.json"),
            (".hidden", ".hidden"),
            ("v1.2.3", "v1.2.3"),
        ],
    )
    def test_separators_replaced_dotted_names_preserved(self, value, expected):
        assert sanitize_template_value(value) == expected

    def test_idempotent(self):
        """Sanitising an already-sanitised value is a no-op."""
        for v in ["", ".", "..", "...", "../x", "a/b", "dog_bark", ".hidden"]:
            once = sanitize_template_value(v)
            assert sanitize_template_value(once) == once
