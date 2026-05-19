"""Tests for the Sync Sources system (SettingsSource + LabelsetSource).

Covers:
- SettingsSource and LabelsetSource base classes
- Auto-discovery registries
- ServerFileSettingsSource load/save round-trip
- ServerFileLabelsetSource load/save round-trip
- Settings sync-on-change: changing a setting triggers source export
- Settings sync-on-load: sync_from_settings_source imports from source
- Circular guard: import doesn't trigger re-export
- Template resolution ({username}, {detector_id}, {detector_name})
- No source configured: no sync behavior (no errors)
- Source file missing on load: graceful fallback
- API routes for settings sources and labelset sources
- DetectorContext.labelset_source field
- Labelset sync_to_labelset_source / sync_from_labelset_source
"""

from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------------------
# SettingsSource base class
# ---------------------------------------------------------------------------


class TestSettingsSourceBase:
    def test_load_raises_not_implemented(self):
        from vtsearch.settings_io.sources.base import SettingsSource

        src = SettingsSource()
        with pytest.raises(NotImplementedError):
            src.load({})

    def test_save_raises_not_implemented(self):
        from vtsearch.settings_io.sources.base import SettingsSource

        src = SettingsSource()
        with pytest.raises(NotImplementedError):
            src.save({}, {})

    def test_to_dict_contains_standard_keys(self):
        from vtsearch.settings_io.sources.base import SettingsSource

        class Minimal(SettingsSource):
            name = "minimal"
            display_name = "Minimal"
            description = "Minimal source."
            fields = []

            def load(self, _fv):
                return {}

            def save(self, _data, _fv):
                pass

        d = Minimal().to_dict()
        assert d["name"] == "minimal"
        assert d["display_name"] == "Minimal"
        assert "icon" in d
        assert "fields" in d

    def test_default_icon_is_sync_arrows(self):
        from vtsearch.settings_io.sources.base import SettingsSource

        assert SettingsSource.icon == "\U0001f504"


# ---------------------------------------------------------------------------
# LabelsetSource base class
# ---------------------------------------------------------------------------


class TestLabelsetSourceBase:
    def test_load_raises_not_implemented(self):
        from vtsearch.labels.sources.base import LabelsetSource

        src = LabelsetSource()
        with pytest.raises(NotImplementedError):
            src.load({})

    def test_save_raises_not_implemented(self):
        from vtsearch.labels.sources.base import LabelsetSource

        src = LabelsetSource()
        with pytest.raises(NotImplementedError):
            src.save(None, {})  # pyright: ignore[reportArgumentType]

    def test_to_dict_contains_standard_keys(self):
        from vtsearch.labels.sources.base import LabelsetSource

        class Minimal(LabelsetSource):
            name = "minimal"
            display_name = "Minimal"
            description = "Minimal source."
            fields = []

            def load(self, _fv):
                return []

            def save(self, _labelset, _fv):
                pass

        d = Minimal().to_dict()
        assert d["name"] == "minimal"
        assert d["display_name"] == "Minimal"
        assert "icon" in d


# ---------------------------------------------------------------------------
# Registry discovery
# ---------------------------------------------------------------------------


class TestSettingsSourceRegistry:
    def test_list_settings_sources(self):
        from vtsearch.settings_io.sources import list_settings_sources

        sources = list_settings_sources()
        names = [s.name for s in sources]
        assert "server_json_file" in names

    def test_get_settings_source(self):
        from vtsearch.settings_io.sources import get_settings_source

        src = get_settings_source("server_json_file")
        assert src is not None
        assert src.name == "server_json_file"

    def test_get_nonexistent_returns_none(self):
        from vtsearch.settings_io.sources import get_settings_source

        assert get_settings_source("nonexistent") is None


class TestLabelsetSourceRegistry:
    def test_list_labelset_sources(self):
        from vtsearch.labels.sources import list_labelset_sources

        sources = list_labelset_sources()
        names = [s.name for s in sources]
        assert "server_json_file" in names

    def test_get_labelset_source(self):
        from vtsearch.labels.sources import get_labelset_source

        src = get_labelset_source("server_json_file")
        assert src is not None
        assert src.name == "server_json_file"

    def test_get_nonexistent_returns_none(self):
        from vtsearch.labels.sources import get_labelset_source

        assert get_labelset_source("nonexistent") is None


# ---------------------------------------------------------------------------
# ServerFileSettingsSource load/save round-trip
# ---------------------------------------------------------------------------


class TestServerFileSettingsSource:
    def test_load_nonexistent_returns_empty(self, tmp_path):
        from vtsearch.settings_io.sources import get_settings_source

        src = get_settings_source("server_json_file")
        result = src.load({"filepath": str(tmp_path / "missing.json")})
        assert result == {}

    def test_load_valid_file(self, tmp_path):
        from vtsearch.settings_io.sources import get_settings_source

        filepath = tmp_path / "settings.json"
        filepath.write_text(json.dumps({"volume": 0.5, "theme": "light"}))

        src = get_settings_source("server_json_file")
        result = src.load({"filepath": str(filepath)})
        assert result["volume"] == 0.5
        assert result["theme"] == "light"

    def test_save_creates_file(self, tmp_path):
        from vtsearch.settings_io.sources import get_settings_source

        filepath = tmp_path / "subdir" / "settings.json"
        src = get_settings_source("server_json_file")
        src.save({"volume": 0.8}, {"filepath": str(filepath)})

        assert filepath.exists()
        data = json.loads(filepath.read_text())
        assert data["volume"] == 0.8

    def test_load_save_round_trip(self, tmp_path):
        from vtsearch.settings_io.sources import get_settings_source

        filepath = tmp_path / "settings.json"
        src = get_settings_source("server_json_file")

        original = {"volume": 0.3, "theme": "highviz", "show_metadata": False}
        src.save(original, {"filepath": str(filepath)})

        loaded = src.load({"filepath": str(filepath)})
        assert loaded == original

    def test_load_invalid_json_raises(self, tmp_path):
        from vtsearch.settings_io.sources import get_settings_source

        filepath = tmp_path / "bad.json"
        filepath.write_text("not valid json {{{")

        src = get_settings_source("server_json_file")
        with pytest.raises(ValueError, match="Invalid JSON"):
            src.load({"filepath": str(filepath)})

    def test_load_non_dict_raises(self, tmp_path):
        from vtsearch.settings_io.sources import get_settings_source

        filepath = tmp_path / "list.json"
        filepath.write_text(json.dumps([1, 2, 3]))

        src = get_settings_source("server_json_file")
        with pytest.raises(ValueError, match="JSON object"):
            src.load({"filepath": str(filepath)})

    def test_empty_filepath_raises(self):
        from vtsearch.settings_io.sources import get_settings_source

        src = get_settings_source("server_json_file")
        with pytest.raises(ValueError, match="file path is required"):
            src.load({"filepath": ""})

    def test_username_template_resolution(self, tmp_path, monkeypatch):
        from vtsearch.settings_io.sources.server_json_file import _resolve_filepath

        monkeypatch.setattr("vtsearch.auth.get_current_user", lambda: "alice")

        result = _resolve_filepath({"filepath": str(tmp_path / "{username}.settings.json")})
        assert "alice.settings.json" in result
        assert "{username}" not in result


# ---------------------------------------------------------------------------
# ServerFileLabelsetSource load/save round-trip
# ---------------------------------------------------------------------------


class TestServerFileLabelsetSource:
    def test_load_nonexistent_returns_empty(self, tmp_path):
        from vtsearch.labels.sources import get_labelset_source

        src = get_labelset_source("server_json_file")
        result = src.load({"filepath": str(tmp_path / "missing.json")})
        assert result == []

    def test_load_valid_file(self, tmp_path):
        from vtsearch.labels.sources import get_labelset_source

        filepath = tmp_path / "labels.json"
        labels = {"labels": [{"md5": "abc123", "label": "good"}, {"md5": "def456", "label": "bad"}]}
        filepath.write_text(json.dumps(labels))

        src = get_labelset_source("server_json_file")
        result = src.load({"filepath": str(filepath)})
        assert len(result) == 2
        assert result[0]["md5"] == "abc123"
        assert result[0]["label"] == "good"

    def test_save_creates_file(self, tmp_path):
        from vtsearch.datasets.labelset import LabelSet, LabeledElement
        from vtsearch.labels.sources import get_labelset_source

        filepath = tmp_path / "labels.json"
        src = get_labelset_source("server_json_file")

        ls = LabelSet(
            [
                LabeledElement(md5="abc", label="good"),
                LabeledElement(md5="def", label="bad"),
            ]
        )
        src.save(ls, {"filepath": str(filepath)})

        assert filepath.exists()
        data = json.loads(filepath.read_text())
        assert "labels" in data
        assert len(data["labels"]) == 2

    def test_load_save_round_trip(self, tmp_path):
        from vtsearch.datasets.labelset import LabelSet, LabeledElement
        from vtsearch.labels.sources import get_labelset_source

        filepath = tmp_path / "labels.json"
        src = get_labelset_source("server_json_file")

        original = LabelSet(
            [
                LabeledElement(md5="abc", label="good"),
                LabeledElement(md5="def", label="bad"),
            ]
        )
        src.save(original, {"filepath": str(filepath)})

        loaded = src.load({"filepath": str(filepath)})
        assert len(loaded) == 2
        assert loaded[0]["md5"] == "abc"

    def test_load_invalid_json_raises(self, tmp_path):
        from vtsearch.labels.sources import get_labelset_source

        filepath = tmp_path / "bad.json"
        filepath.write_text("not valid json")

        src = get_labelset_source("server_json_file")
        with pytest.raises(ValueError, match="Invalid JSON"):
            src.load({"filepath": str(filepath)})

    def test_load_no_labels_key_raises(self, tmp_path):
        from vtsearch.labels.sources import get_labelset_source

        filepath = tmp_path / "no_labels.json"
        filepath.write_text(json.dumps({"data": []}))

        src = get_labelset_source("server_json_file")
        with pytest.raises(ValueError, match="labels"):
            src.load({"filepath": str(filepath)})


# ---------------------------------------------------------------------------
# Settings source config (get/set)
# ---------------------------------------------------------------------------


class TestSettingsSourceConfig:
    def test_get_settings_source_config_default_is_none(self):
        from vtsearch import settings

        assert settings.get_settings_source_config() is None

    def test_set_and_get_settings_source_config(self, isolated_settings):
        from vtsearch import settings

        config = {"source_name": "server_json_file", "field_values": {"filepath": "/tmp/test.json"}}
        settings.set_settings_source_config(config)

        result = settings.get_settings_source_config()
        assert result is not None
        assert result["source_name"] == "server_json_file"
        assert result["field_values"]["filepath"] == "/tmp/test.json"

    def test_clear_settings_source_config(self, isolated_settings):
        from vtsearch import settings

        config = {"source_name": "server_json_file", "field_values": {"filepath": "/tmp/test.json"}}
        settings.set_settings_source_config(config)
        settings.set_settings_source_config(None)

        assert settings.get_settings_source_config() is None

    def test_settings_source_excluded_from_defaults(self):
        from vtsearch.settings import _EXCLUDE_FROM_DEFAULTS

        assert "settings_source" in _EXCLUDE_FROM_DEFAULTS

    def test_settings_source_excluded_from_source_export(self):
        from vtsearch.settings import _EXCLUDE_FROM_SOURCE_EXPORT

        assert "settings_source" in _EXCLUDE_FROM_SOURCE_EXPORT


# ---------------------------------------------------------------------------
# Settings sync-on-change
# ---------------------------------------------------------------------------


class TestSettingsSyncOnChange:
    def test_setting_change_triggers_source_export(self, tmp_path, isolated_settings):
        from vtsearch import settings

        source_file = tmp_path / "synced.json"
        config = {
            "source_name": "server_json_file",
            "field_values": {"filepath": str(source_file)},
        }
        settings.set_settings_source_config(config)

        # Change a setting — should auto-export to source
        settings.set_volume(0.42)

        assert source_file.exists()
        data = json.loads(source_file.read_text())
        assert data["volume"] == 0.42

    def test_source_export_excludes_settings_source_key(self, tmp_path, isolated_settings):
        from vtsearch import settings

        source_file = tmp_path / "synced.json"
        config = {
            "source_name": "server_json_file",
            "field_values": {"filepath": str(source_file)},
        }
        settings.set_settings_source_config(config)
        settings.set_volume(0.5)

        data = json.loads(source_file.read_text())
        assert "settings_source" not in data

    def test_no_source_configured_no_error(self, isolated_settings):
        from vtsearch import settings

        # No source configured — should not raise
        settings.set_volume(0.5)


# ---------------------------------------------------------------------------
# Settings sync-from-source
# ---------------------------------------------------------------------------


class TestSettingsSyncFromSource:
    def test_sync_from_source_imports_settings(self, tmp_path, isolated_settings):
        from vtsearch import settings

        source_file = tmp_path / "remote.json"

        config = {
            "source_name": "server_json_file",
            "field_values": {"filepath": str(source_file)},
        }
        settings.set_settings_source_config(config)

        # Write source file AFTER config is set (since setting config
        # triggers an initial sync-to-source that creates the file).
        source_file.write_text(json.dumps({"volume": 0.77, "theme": "light"}))

        result = settings.sync_from_settings_source()
        assert result is not None
        assert result["volume"] == 0.77

        # Settings should now reflect the imported values
        assert settings.get_volume() == 0.77
        assert settings.get_theme() == "light"

    def test_sync_from_source_no_source_returns_none(self, isolated_settings):
        from vtsearch import settings

        assert settings.sync_from_settings_source() is None

    def test_sync_from_source_missing_file_returns_none(self, tmp_path, isolated_settings):
        from vtsearch import settings

        # Use a path in a subdirectory that doesn't exist yet, so _sync_to_source
        # from set_settings_source_config doesn't auto-create it.
        config = {
            "source_name": "server_json_file",
            "field_values": {"filepath": str(tmp_path / "deep" / "nested" / "nonexistent.json")},
        }
        settings.set_settings_source_config(config)

        # Delete the file that was auto-created by _sync_to_source
        source_path = tmp_path / "deep" / "nested" / "nonexistent.json"
        if source_path.exists():
            source_path.unlink()

        result = settings.sync_from_settings_source()
        assert result is None

    def test_sync_from_source_does_not_reexport(self, tmp_path, isolated_settings):
        """Import from source should NOT trigger re-export (circular guard)."""
        from vtsearch import settings

        source_file = tmp_path / "remote.json"

        config = {
            "source_name": "server_json_file",
            "field_values": {"filepath": str(source_file)},
        }
        settings.set_settings_source_config(config)

        # Overwrite with specific settings AFTER config is set
        source_file.write_text(json.dumps({"volume": 0.33}))

        # Record mtime before sync
        mtime_before = source_file.stat().st_mtime_ns

        settings.sync_from_settings_source()

        # Source file should NOT have been re-written during import
        mtime_after = source_file.stat().st_mtime_ns
        assert mtime_after == mtime_before


# ---------------------------------------------------------------------------
# _apply_settings (module-level, no route dependency)
# ---------------------------------------------------------------------------


class TestApplySettings:
    def test_apply_settings_updates_volume(self, isolated_settings):
        from vtsearch import settings

        settings._apply_settings({"volume": 0.42})
        assert settings.get_volume() == 0.42

    def test_apply_settings_skips_unknown_keys(self, isolated_settings):
        from vtsearch import settings

        # Should not raise — unknown keys are silently ignored
        settings._apply_settings({"nonexistent_key_xyz": "hello"})

    def test_apply_settings_skips_invalid_values(self, isolated_settings):
        from vtsearch import settings

        orig_theme = settings.get_theme()
        # "neon" is not a valid theme — should be silently skipped
        settings._apply_settings({"theme": "neon"})
        assert settings.get_theme() == orig_theme

    def test_apply_settings_available_from_routes(self):
        """The routes module re-exports _apply_settings from settings."""
        from vtsearch.routes.settings.io import _apply_settings
        from vtsearch.settings import _apply_settings as canonical

        assert _apply_settings is canonical


# ---------------------------------------------------------------------------
# Startup auto-import from settings source
# ---------------------------------------------------------------------------


class TestStartupAutoImport:
    """sync_from_settings_source() is called at app startup to import
    settings from the active source.  This simulates that flow."""

    def test_startup_import_applies_source_settings(self, tmp_path, isolated_settings):
        from vtsearch import settings

        source_file = tmp_path / "startup.settings.json"

        config = {
            "source_name": "server_json_file",
            "field_values": {"filepath": str(source_file)},
        }
        settings.set_settings_source_config(config)

        # Overwrite the source file with the settings we want at startup
        source_file.write_text(json.dumps({"volume": 0.33, "theme": "highviz"}))

        # Simulate startup auto-import
        result = settings.sync_from_settings_source()
        assert result is not None
        assert settings.get_volume() == 0.33
        assert settings.get_theme() == "highviz"

    def test_startup_import_noop_when_no_source(self, isolated_settings):
        from vtsearch import settings

        # No source configured — should return None and not error
        assert settings.sync_from_settings_source() is None

    def test_startup_import_noop_when_source_file_missing(self, tmp_path, isolated_settings):
        from vtsearch import settings

        config = {
            "source_name": "server_json_file",
            "field_values": {"filepath": str(tmp_path / "sub" / "missing.json")},
        }
        settings.set_settings_source_config(config)

        # Remove the auto-created file
        auto_created = tmp_path / "sub" / "missing.json"
        if auto_created.exists():
            auto_created.unlink()

        # Should gracefully return None
        assert settings.sync_from_settings_source() is None


# ---------------------------------------------------------------------------
# DetectorContext.labelset_source field
# ---------------------------------------------------------------------------


class TestDetectorContextLabelsetSource:
    def test_default_is_none(self):
        from vtsearch.state.core import DetectorContext

        ctx = DetectorContext("test")
        assert ctx.labelset_source is None

    def test_can_set_and_read(self):
        from vtsearch.state.core import DetectorContext

        ctx = DetectorContext("test")
        ctx.labelset_source = {
            "source_name": "server_json_file",
            "field_values": {"filepath": "/tmp/labels.json"},
        }
        assert ctx.labelset_source["source_name"] == "server_json_file"


# ---------------------------------------------------------------------------
# Labelset sync_to_labelset_source / sync_from_labelset_source
# ---------------------------------------------------------------------------


class TestLabelsetSync:
    def test_sync_to_source_no_source_no_error(self):
        """sync_to_labelset_source should not raise when no source is configured."""
        from vtsearch.labels.sync import sync_to_labelset_source

        sync_to_labelset_source()  # Should silently do nothing

    def test_sync_from_source_no_source_returns_none(self):
        from vtsearch.labels.sync import sync_from_labelset_source

        assert sync_from_labelset_source() is None

    def test_sync_to_source_writes_labels(self, tmp_path):
        from vtsearch.labels.sync import sync_to_labelset_source
        from vtsearch.state.core import (
            DetectorContext,
            register_detector_context,
            set_thread_detector_context,
            unregister_detector_context,
        )
        from vtsearch.state import medias, good_votes

        # Create and register a detector context with a labelset source
        ctx = DetectorContext("test_sync", name="test_sync")
        ctx.labelset_source = {
            "source_name": "server_json_file",
            "field_values": {"filepath": str(tmp_path / "labels.json")},
        }
        register_detector_context(ctx)
        set_thread_detector_context(ctx)

        try:
            # Add some votes
            saved_medias = dict(medias)

            # Add a media item with an md5
            mid = max(medias.keys(), default=0) + 10000
            medias[mid] = {"id": mid, "md5": "test_md5", "type": "audio"}
            good_votes[mid] = None

            sync_to_labelset_source()

            # Check the file was written
            filepath = tmp_path / "labels.json"
            assert filepath.exists()
            data = json.loads(filepath.read_text())
            assert "labels" in data
            assert any(e["md5"] == "test_md5" for e in data["labels"])
        finally:
            # Clean up
            good_votes.pop(mid, None)
            medias.pop(mid, None)
            medias.update(saved_medias)
            set_thread_detector_context(None)
            unregister_detector_context("test_sync")

    def test_sync_from_source_applies_labels(self, tmp_path):
        from vtsearch.labels.sync import sync_from_labelset_source
        from vtsearch.state.core import (
            DetectorContext,
            register_detector_context,
            set_thread_detector_context,
            unregister_detector_context,
        )
        from vtsearch.state import medias, good_votes

        # Create a labels file with an md5 that matches a test media
        # Get an existing media's md5
        if not medias:
            pytest.skip("No test medias available")

        first_id = next(iter(medias))
        first_md5 = medias[first_id].get("md5", "")
        if not first_md5:
            pytest.skip("Test media has no md5")

        filepath = tmp_path / "labels.json"
        filepath.write_text(json.dumps({"labels": [{"md5": first_md5, "label": "good"}]}))

        ctx = DetectorContext("test_import", name="test_import")
        ctx.labelset_source = {
            "source_name": "server_json_file",
            "field_values": {"filepath": str(filepath)},
        }
        register_detector_context(ctx)
        set_thread_detector_context(ctx)

        try:
            result = sync_from_labelset_source()
            assert result is not None
            assert len(result) == 1

            # Check the label was applied
            assert first_id in good_votes
        finally:
            set_thread_detector_context(None)
            unregister_detector_context("test_import")

    def test_sync_to_source_emits_detector_meta(self, tmp_path):
        """sync_to_labelset_source writes the detector's input_spec / threshold."""
        from vtsearch.detectors.store import _detector_path, _write_detector
        from vtsearch.labels.sync import sync_to_labelset_source
        from vtsearch.state.core import (
            DetectorContext,
            register_detector_context,
            set_thread_detector_context,
            unregister_detector_context,
        )
        from vtsearch.state import medias, good_votes

        det_name = "meta_export"
        det_path = _detector_path(det_name)
        _write_detector(
            det_path,
            {
                "name": det_name,
                "media_type": "audio",
                "labelset": {"labels": []},
                "input_spec": {
                    "clipper": "sound_tiling",
                    "clipper_params": {"duration": "2.0"},
                },
            },
        )

        ctx = DetectorContext(det_name, name=det_name, media_type="audio")
        # Stand in for a trained MLP so the threshold flows through.
        ctx.model = object()
        ctx.threshold = 0.31
        ctx.labelset_source = {
            "source_name": "server_json_file",
            "field_values": {"filepath": str(tmp_path / "labels.json")},
        }
        register_detector_context(ctx)
        set_thread_detector_context(ctx)

        saved_medias = dict(medias)
        mid = max(medias.keys(), default=0) + 20000
        try:
            medias[mid] = {"id": mid, "md5": "fake_md5", "type": "audio"}
            good_votes[mid] = None
            sync_to_labelset_source()

            data = json.loads((tmp_path / "labels.json").read_text())
            meta = data.get("detector_meta")
            assert meta is not None
            assert meta["media_type"] == "audio"
            assert meta["input_spec"] == {
                "clipper": "sound_tiling",
                "clipper_params": {"duration": "2.0"},
            }
            assert meta["threshold"] == pytest.approx(0.31)
        finally:
            good_votes.pop(mid, None)
            medias.pop(mid, None)
            medias.update(saved_medias)
            set_thread_detector_context(None)
            unregister_detector_context(det_name)
            if det_path.exists():
                det_path.unlink()

    def test_sync_to_source_skips_threshold_without_model(self, tmp_path):
        """A detector that hasn't been trained yet has no live threshold to emit."""
        from vtsearch.detectors.store import _detector_path, _write_detector
        from vtsearch.labels.sync import sync_to_labelset_source
        from vtsearch.state.core import (
            DetectorContext,
            register_detector_context,
            set_thread_detector_context,
            unregister_detector_context,
        )
        from vtsearch.state import medias, good_votes

        det_name = "meta_no_model"
        det_path = _detector_path(det_name)
        _write_detector(
            det_path,
            {"name": det_name, "media_type": "audio", "labelset": {"labels": []}},
        )

        ctx = DetectorContext(det_name, name=det_name, media_type="audio")
        ctx.model = None
        ctx.threshold = 0.99  # noise — should NOT be emitted without a model
        ctx.labelset_source = {
            "source_name": "server_json_file",
            "field_values": {"filepath": str(tmp_path / "labels.json")},
        }
        register_detector_context(ctx)
        set_thread_detector_context(ctx)

        saved_medias = dict(medias)
        mid = max(medias.keys(), default=0) + 20100
        try:
            medias[mid] = {"id": mid, "md5": "fake_md5_2", "type": "audio"}
            good_votes[mid] = None
            sync_to_labelset_source()

            data = json.loads((tmp_path / "labels.json").read_text())
            meta = data.get("detector_meta")
            assert meta is not None
            assert "threshold" not in meta
            assert meta["media_type"] == "audio"
        finally:
            good_votes.pop(mid, None)
            medias.pop(mid, None)
            medias.update(saved_medias)
            set_thread_detector_context(None)
            unregister_detector_context(det_name)
            if det_path.exists():
                det_path.unlink()

    def test_sync_from_source_writes_input_spec_to_detector(self, tmp_path):
        """An inbound detector_meta updates the receiving detector JSON's input_spec."""
        from vtsearch.detectors.store import _detector_path, _read_detector, _write_detector
        from vtsearch.labels.sync import sync_from_labelset_source
        from vtsearch.state.core import (
            DetectorContext,
            register_detector_context,
            set_thread_detector_context,
            unregister_detector_context,
        )

        det_name = "meta_import"
        det_path = _detector_path(det_name)
        _write_detector(
            det_path,
            {"name": det_name, "media_type": "audio", "labelset": {"labels": []}},
        )

        filepath = tmp_path / "labels.json"
        filepath.write_text(
            json.dumps(
                {
                    "labels": [],
                    "detector_meta": {
                        "media_type": "audio",
                        "input_spec": {
                            "clipper": "sound_tiling",
                            "clipper_params": {"duration": "2.0"},
                        },
                        # Threshold is informational; receiver retrains and
                        # must NOT persist it.
                        "threshold": 0.5,
                    },
                }
            )
        )

        ctx = DetectorContext(det_name, name=det_name, media_type="audio")
        ctx.labelset_source = {
            "source_name": "server_json_file",
            "field_values": {"filepath": str(filepath)},
        }
        register_detector_context(ctx)
        set_thread_detector_context(ctx)

        try:
            sync_from_labelset_source()
            updated = _read_detector(det_path)
            assert updated is not None
            assert updated["input_spec"] == {
                "clipper": "sound_tiling",
                "clipper_params": {"duration": "2.0"},
            }
            assert "threshold" not in updated
        finally:
            set_thread_detector_context(None)
            unregister_detector_context(det_name)
            if det_path.exists():
                det_path.unlink()

    def test_sync_from_source_without_detector_meta_leaves_detector_alone(self, tmp_path):
        """Legacy labelsets (no detector_meta) don't mutate the receiving detector."""
        from vtsearch.detectors.store import _detector_path, _read_detector, _write_detector
        from vtsearch.labels.sync import sync_from_labelset_source
        from vtsearch.state.core import (
            DetectorContext,
            register_detector_context,
            set_thread_detector_context,
            unregister_detector_context,
        )

        det_name = "meta_legacy"
        det_path = _detector_path(det_name)
        _write_detector(
            det_path,
            {
                "name": det_name,
                "media_type": "audio",
                "labelset": {"labels": []},
                "input_spec": {"clipper": "sound_tiling"},
            },
        )

        filepath = tmp_path / "labels.json"
        filepath.write_text(json.dumps({"labels": []}))

        ctx = DetectorContext(det_name, name=det_name, media_type="audio")
        ctx.labelset_source = {
            "source_name": "server_json_file",
            "field_values": {"filepath": str(filepath)},
        }
        register_detector_context(ctx)
        set_thread_detector_context(ctx)

        try:
            sync_from_labelset_source()
            updated = _read_detector(det_path)
            assert updated is not None
            # Receiver's existing input_spec is preserved.
            assert updated["input_spec"] == {"clipper": "sound_tiling"}
        finally:
            set_thread_detector_context(None)
            unregister_detector_context(det_name)
            if det_path.exists():
                det_path.unlink()


# ---------------------------------------------------------------------------
# API routes: settings sources
# ---------------------------------------------------------------------------


class TestSettingsSourcesAPI:
    def test_list_settings_sources(self, client):
        resp = client.get("/api/settings-sources")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, list)
        names = [s["name"] for s in data]
        assert "server_json_file" in names

    def test_get_active_source_default_null(self, client):
        resp = client.get("/api/settings-sources/active")
        assert resp.status_code == 200
        assert resp.get_json() is None

    def test_set_and_get_active_source(self, client, isolated_settings):
        # Set
        resp = client.put(
            "/api/settings-sources/active",
            json={"source_name": "server_json_file", "field_values": {"filepath": "/tmp/test.json"}},
        )
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True

        # Get
        resp = client.get("/api/settings-sources/active")
        assert resp.status_code == 200
        cfg = resp.get_json()
        assert cfg["source_name"] == "server_json_file"

    def test_clear_active_source(self, client, isolated_settings):
        # Set
        client.put(
            "/api/settings-sources/active",
            json={"source_name": "server_json_file", "field_values": {"filepath": "/tmp/test.json"}},
        )
        # Clear
        resp = client.put("/api/settings-sources/active", json={})
        assert resp.status_code == 200

        resp = client.get("/api/settings-sources/active")
        assert resp.get_json() is None

    def test_set_unknown_source_returns_404(self, client):
        resp = client.put(
            "/api/settings-sources/active",
            json={"source_name": "nonexistent", "field_values": {}},
        )
        assert resp.status_code == 404

    def test_sync_no_source_configured(self, client):
        resp = client.post("/api/settings-sources/sync")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is False

    def test_sync_from_source(self, client, tmp_path, isolated_settings):
        source_file = tmp_path / "remote.json"

        # Set active source (this auto-creates the file via sync-to-source)
        client.put(
            "/api/settings-sources/active",
            json={"source_name": "server_json_file", "field_values": {"filepath": str(source_file)}},
        )

        # Overwrite with specific settings AFTER config is set
        source_file.write_text(json.dumps({"volume": 0.99}))

        # Sync
        resp = client.post("/api/settings-sources/sync")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert "volume" in data["keys"]


# ---------------------------------------------------------------------------
# API routes: labelset sources
# ---------------------------------------------------------------------------


class TestLabelsetSourcesAPI:
    def test_list_labelset_sources(self, client):
        resp = client.get("/api/labelset-sources")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, list)
        names = [s["name"] for s in data]
        assert "server_json_file" in names

    def test_get_detector_source_unknown_detector(self, client):
        resp = client.get("/api/detectors/nonexistent/labelset-source")
        # Without a loaded DetectorContext we just report "no source" (null).
        assert resp.status_code == 200
        assert resp.get_json() is None

    def test_set_detector_source_not_loaded(self, client):
        resp = client.put(
            "/api/detectors/nonexistent/labelset-source",
            json={"source_name": "server_json_file", "field_values": {}},
        )
        assert resp.status_code == 404

    def test_set_unknown_labelset_source_returns_404(self, client):
        from vtsearch.state.core import (
            DetectorContext,
            register_detector_context,
            unregister_detector_context,
        )

        ctx = DetectorContext("api_test", name="api_test")
        register_detector_context(ctx)

        try:
            resp = client.put(
                "/api/detectors/api_test/labelset-source",
                json={"source_name": "nonexistent", "field_values": {}},
            )
            assert resp.status_code == 404
        finally:
            unregister_detector_context("api_test")
