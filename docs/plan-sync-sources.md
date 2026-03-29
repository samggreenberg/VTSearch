# Sync Sources: Persistent Bidirectional Sync for Settings and Labelsets

## Problem

Currently, importing settings or labels is a **one-shot** operation. If you import labelset A into detector A and then add labels (making detector A'), the original labelset A is stale. You'd have to manually re-export. Same for settings: import from a file, tweak a setting, and the file is out of date.

## Solution

Introduce **sync sources** — a layer that pairs an importer and exporter behind a single abstraction, providing:

- **Auto-import on load**: When the app starts (or a detector is loaded), pull from the source.
- **Auto-export on change**: When settings change or votes are cast, push back to the source.

Two new plugin families:

- `SettingsSource` — syncs the global settings.
- `LabelsetSource` — syncs a detector's labels.

## Design Principles

1. **Sources are additive.** Existing standalone importers and exporters remain fully functional. A user can still do a one-shot export via `LocalFileSettingsExporter` even if a `ServerFileSettingsSource` is active.

2. **Sources are plugins.** Same `PluginRegistry` discovery pattern used everywhere else. Adding a `DropboxSettingsSource` later is just a new sub-package.

3. **In-memory state is canonical.** The `DetectorContext` vote dicts and `settings.py` module remain the source of truth at runtime. Sources are a sync side-channel, not a replacement.

4. **App-level configuration.** Which source is active is configured outside of the synced data itself (avoiding chicken-and-egg). For settings, an app config or CLI arg specifies the source. For labelsets, the source is attached per-detector.

## Architecture

### SettingsSource

```
vtsearch/settings_io/sources/
  ├── __init__.py                    # PluginRegistry, sentinel SETTINGS_SOURCE
  ├── base.py                        # SettingsSource ABC
  └── server_json_file/
      └── __init__.py                # ServerFileSettingsSource
```

**Base class:**

```python
class SettingsSource(PluginBase):
    """A bidirectional sync target for settings."""

    def load(self, field_values: dict) -> dict[str, Any]:
        """Import settings from the source. Returns settings dict."""
        raise NotImplementedError

    def save(self, settings_data: dict[str, Any], field_values: dict) -> None:
        """Export settings to the source."""
        raise NotImplementedError
```

**Concrete: `ServerFileSettingsSource`**

- `load()`: reads JSON from a server filepath (reuses `ServerFileSettingsImporter` logic).
- `save()`: writes JSON to the same filepath (reuses `ServerFileSettingsExporter` logic).
- Fields: `filepath` (server_path) — supports `{username}` template resolved at runtime.

**Sync hooks:**

- **On load**: At app startup, if a settings source is configured, call `source.load()` and apply via setters. This happens after loading local `settings.json` (so we know the source config) but overlays remote values.
- **On change**: In `settings._save()`, after the local atomic write, call `source.save(get_all(), field_values)`. A `_syncing` guard flag prevents re-export during an import pass.

**Configuration:**

The active source is configured via `settings.json` itself, under a key excluded from sync:

```json
{
  "settings_source": {
    "source_name": "server_json_file",
    "field_values": {"filepath": "data/{username}.settings.json"}
  }
}
```

This key is in `_EXCLUDE_FROM_DEFAULTS` and excluded from `get_all()` export, so it doesn't get synced circularly. The `{username}` template is resolved at runtime via `get_current_user()`.

### LabelsetSource

```
vtsearch/labels/sources/
  ├── __init__.py                    # PluginRegistry, sentinel LABELSET_SOURCE
  ├── base.py                       # LabelsetSource ABC
  └── server_json_file/
      └── __init__.py                # ServerFileLabelsetSource
```

**Base class:**

```python
class LabelsetSource(PluginBase):
    """A bidirectional sync target for detector labels."""

    def load(self, field_values: dict) -> list[dict]:
        """Import labels from the source. Returns label dicts."""
        raise NotImplementedError

    def save(self, labelset: LabelSet, field_values: dict) -> None:
        """Export labels to the source."""
        raise NotImplementedError
```

**Concrete: `ServerFileLabelsetSource`**

- `load()`: reads JSON label file from server filesystem.
- `save()`: writes `LabelSet.to_dict()` to the same filepath.
- Fields: `filepath` (server_path) — supports `{detector_id}` template.

**Sync hooks:**

- **On load**: When a detector is loaded and has a linked source, call `source.load()` and apply labels via `apply_label()`.
- **On change**: After `toggle_vote()` and `apply_label()` in `state_votes.py`, if the active detector has a linked source, call `source.save()`. A `_syncing` guard prevents re-export during import.

**Per-detector attachment:**

`DetectorContext` gets a new field:

```python
labelset_source: dict | None = None
# Example: {"source_name": "server_json_file", "field_values": {"filepath": "..."}}
```

When a label import creates a detector and the import came from a source-capable importer, the source is automatically attached. Sources can also be attached/detached via API.

## API Endpoints

### Settings Sources

```
GET  /api/settings-sources              → list available source plugins
GET  /api/settings-sources/active       → get the active source config (or null)
PUT  /api/settings-sources/active       → set or clear the active source
POST /api/settings-sources/sync         → force a manual sync (import from source)
```

### Labelset Sources

```
GET  /api/labelset-sources                          → list available source plugins
GET  /api/detectors/<name>/labelset-source           → get this detector's source (or null)
PUT  /api/detectors/<name>/labelset-source           → set or clear the source
POST /api/detectors/<name>/labelset-source/sync      → force manual sync
```

## Circular Trigger Prevention

When a source imports settings/labels, each applied value triggers `_save()` / vote mutation, which would normally trigger an export back to the source. To prevent this:

```python
_syncing = threading.local()

def _is_syncing() -> bool:
    return getattr(_syncing, "active", False)

def _sync_guard():
    """Context manager that suppresses source export during import."""
    _syncing.active = True
    try:
        yield
    finally:
        _syncing.active = False
```

The export hook checks `_is_syncing()` and skips if true.

## Template Variables

Source field values support template strings resolved at runtime:

| Variable | Resolves to | Available in |
|---|---|---|
| `{username}` | `get_current_user()` | SettingsSource |
| `{detector_id}` | Active detector's ID | LabelsetSource |
| `{detector_name}` | Active detector's name | LabelsetSource |

## Future Extensibility

New source types are just new sub-packages:

- `DropboxSettingsSource` — sync via Dropbox API
- `S3LabelsetSource` — sync to S3 bucket
- `DatabaseSettingsSource` — sync to a shared DB

For network-based sources, the `save()` method should be debounced (batch writes on a short timer) rather than called per-mutation. The base class could provide an optional `debounce_ms` attribute that the sync hook respects.

## Implementation Order

1. **SettingsSource base + server_json_file** — base class, plugin registry, concrete implementation
2. **Settings sync hooks** — hook `_save()`, add `_syncing` guard, startup auto-import
3. **Settings source API routes** — list, get/set active, manual sync
4. **LabelsetSource base + server_json_file** — same pattern
5. **Labelset sync hooks** — hook `toggle_vote()`/`apply_label()`, DetectorContext field
6. **Labelset source API routes** — list, get/set per-detector, manual sync
7. **Tests** — for both source types, sync behavior, circular prevention, template resolution

## Test Plan

- Source plugin discovery and registry
- `load()` / `save()` round-trip for server_json_file sources
- Sync-on-change: changing a setting triggers source export
- Sync-on-load: app startup imports from source
- Circular guard: import doesn't trigger re-export
- Template resolution (`{username}`, `{detector_id}`)
- No source configured: no sync behavior (no errors)
- Source file missing on load: graceful fallback
- Standalone importers/exporters still work independently
