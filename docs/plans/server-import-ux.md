# Server / Services dataset-import UX

**Status: Phase 1 shipped.** Came out of a hands-on audit of the three
real-world import paths (Services extension plugin, server folder, server file
with pre-computed vectors), driving the live UI. Open follow-ups first; shipped
detail below.

**Reversal note:** single-user path confinement (the `SERVER_ROOTS` work in the
shipped list) was later **reversed** — in single-user / no-auth mode server-path
import, export, and browse are now unrestricted (the lone trusted user may reach
any server-readable path), and `SERVER_ROOTS` / `VTSEARCH_SERVER_ROOTS` were
removed. Multi-user per-user confinement is unchanged. The "Browse-root vs.
import-root seam" follow-up below is therefore moot for single-user mode (browse
is now rooted at `/`, matching the unrestricted import validation).

## Open follow-ups

- **Empty Services tab.** The base `DatasetImporter.category` default is
  `"services"`, and every services-category importer ships
  `hidden_from_picker=True` (recaller, http_archive, pickle, combine_datasets),
  so a stock install shows an empty Services tab ("No importers in this
  category."). `visibleImporterTabs` renders all declared tabs unconditionally,
  contradicting `tabs.py`'s documented "only show tabs with ≥1 visible
  importer." Follow-up: hide tabs with zero visible importers and/or give the
  Services tab an empty-state that explains it's the extension surface. (Audited
  but intentionally not changed in Phase 1.)
- **Relative-entry resolution in server_files manifests.** Relative paths inside
  a `.txt`/`.npz` resolve against the **manifest file's own directory**, not
  CWD — easy to trip. Consider documenting in the field hint or resolving
  against a more predictable base.
- **Browse-root vs. import-root seam (server_folder).** Moot in single-user mode
  after the confinement reversal (both browse and import now rooted at `/`).
  Retained only as a note for multi-user: the `server_folder` browser +
  media-type detection were rooted at `/` while import validation confined to
  `SERVER_ROOTS`, so a user could navigate to and select a folder outside the
  root and only get rejected at Import.

## What shipped

- **Unified Server-tab path policy** — `server_folder`'s `path` validated through
  `validate_server_filepath` like `server_files`; closed a multi-user confinement
  hole; fixed the misleading `get_file_access_base_dir` docstring. *(Later
  reversed — see status note; `SERVER_ROOTS` removed.)*
- **Client-side required-field gating** in the generic `form` picker
  (`formCanSubmit` disables Import until required fields are filled).
- **Server path browser** for both Server importers — `<vt-file-browser>` on
  `server_files`' `server_path`, a "Browse…" `<vt-folder-browser>` toggle on
  `server_folder` (backed by `/api/browse-media-files?source=server_fs`).
