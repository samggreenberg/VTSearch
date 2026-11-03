"""Tests for the :class:`vtscore.sync.SyncSource` base contract.

``SyncSource`` is the abstract, library-tier base shared by
``SettingsSource`` and ``LabelsetSource``.  Its public ``load`` / ``save`` /
``peek_version`` methods are framework-owned wrappers that normalize
*field_values* (whitespace-strip, template-resolve, URL/path-validate) on a
*copy* before delegating to the underscored ``_do_*`` template methods.  The
concrete subclasses are covered elsewhere; here we pin the base guarantees
directly with a minimal fake subclass carrying a plain text field.
"""

from __future__ import annotations

import pytest

from vtscore.plugins import PluginField
from vtscore.sync import SyncSource


class _FakeSource(SyncSource):
    """Minimal concrete sync source with a single whitespace-trimmable field."""

    name = "fake_sync"
    display_name = "Fake Sync"
    description = "Test double."
    fields = [PluginField(key="label", label="Label", field_type="text", required=False)]

    def __init__(self) -> None:
        self.loaded_with: dict | None = None
        self.saved_data = None
        self.saved_with: dict | None = None
        self.peeked_with: dict | None = None

    def _do_load(self, field_values):
        self.loaded_with = field_values
        return ["item"]

    def _do_save(self, data, /, field_values):
        self.saved_data = data
        self.saved_with = field_values

    def _do_peek_version(self, field_values):
        self.peeked_with = field_values
        return "v1"


# ---------------------------------------------------------------------------
# normalize-before-delegate
# ---------------------------------------------------------------------------


class TestNormalizeBeforeDelegate:
    def test_load_strips_whitespace_before_delegating(self):
        src = _FakeSource()
        result = src.load({"label": "  padded  "})
        assert result == ["item"]
        assert src.loaded_with == {"label": "padded"}

    def test_save_strips_whitespace_before_delegating(self):
        src = _FakeSource()
        src.save({"a": 1}, field_values={"label": "  x "})
        assert src.saved_data == {"a": 1}
        assert src.saved_with == {"label": "x"}

    def test_peek_version_normalizes_then_delegates(self):
        src = _FakeSource()
        token = src.peek_version({"label": " y "})
        assert token == "v1"
        assert src.peeked_with == {"label": "y"}

    def test_callers_dict_is_not_mutated(self):
        src = _FakeSource()
        original = {"label": "  keep spaces  "}
        src.load(original)
        # The normalize pass works on a copy; the caller's dict is untouched.
        assert original == {"label": "  keep spaces  "}

    def test_missing_optional_field_becomes_empty_string(self):
        src = _FakeSource()
        src.load({})
        assert src.loaded_with == {"label": ""}


# ---------------------------------------------------------------------------
# required-field validation surfaces through the wrapper
# ---------------------------------------------------------------------------


class _RequiredSource(_FakeSource):
    name = "required_sync"
    fields = [PluginField(key="label", label="Label", field_type="text", required=True)]


class TestRequiredValidation:
    def test_empty_required_field_raises_before_delegate(self):
        src = _RequiredSource()
        with pytest.raises(ValueError, match="required"):
            src.load({"label": "   "})
        # _do_load never ran.
        assert src.loaded_with is None


# ---------------------------------------------------------------------------
# peek_version error handling + defaults
# ---------------------------------------------------------------------------


class TestPeekVersion:
    def test_peek_version_swallows_normalize_errors_to_none(self):
        # A required-field violation during normalize must not crash the
        # caller's cheap freshness probe — it degrades to None.
        src = _RequiredSource()
        assert src.peek_version({"label": ""}) is None
        assert src.peeked_with is None

    def test_default_do_peek_version_returns_none(self):
        class _NoPeek(SyncSource):
            name = "no_peek"
            display_name = "No Peek"
            description = ""
            fields = []

            def _do_load(self, field_values):
                return None

            def _do_save(self, data, /, field_values):
                pass

        assert _NoPeek().peek_version({}) is None


# ---------------------------------------------------------------------------
# abstract template methods raise until overridden
# ---------------------------------------------------------------------------


class TestAbstractHooks:
    def test_do_load_not_implemented(self):
        class _Bare(SyncSource):
            name = "bare"
            display_name = "Bare"
            description = ""
            fields = []

        with pytest.raises(NotImplementedError, match="_do_load"):
            _Bare()._do_load({})

    def test_do_save_not_implemented(self):
        class _Bare(SyncSource):
            name = "bare2"
            display_name = "Bare2"
            description = ""
            fields = []

        with pytest.raises(NotImplementedError, match="_do_save"):
            _Bare()._do_save({}, {})

    def test_default_icon_is_sync_arrows(self):
        assert SyncSource.icon == "\U0001f504"
