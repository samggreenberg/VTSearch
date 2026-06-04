# Server / Services dataset-import UX

Status: **Phase 1 shipped.** Came out of a hands-on audit of the three real-world
import paths (Services extension plugin, server folder, server file with
pre-computed vectors), driving the live UI. The Demo importer is well-trodden;
these three are what real deployments use and were comparatively rough.

## What shipped

1. **Unified Server-tab path policy.** `server_folder`'s `path` field
   (`field_type="folder"`) is now validated through
   `validate_server_filepath` exactly like `server_files`' `server_path`
   field, instead of bypassing validation entirely. Both Server importers
   now confine to `SERVER_ROOTS` (single-user) / the per-user data dir
   (multi-user). This also closed a multi-user confinement hole where
   `server_folder` accepted any absolute path.
   - `validate_server_filepath` now checks the resolved path against **every**
     entry of `SERVER_ROOTS` (not just `[0]`), so `VTSEARCH_SERVER_ROOTS=/a:/b`
     actually permits both roots. The error message names the allowed roots and
     points at `VTSEARCH_SERVER_ROOTS`.
   - **BREAKING:** importing a folder outside the install dir now requires
     setting `VTSEARCH_SERVER_ROOTS` to include it (previously folder imports
     accepted any server-readable path). `server_files` already behaved this way.
   - Fixed the misleading `get_file_access_base_dir` docstring/comment that
     called single-user mode "unrestricted" — it confines to `SERVER_ROOTS`.

2. **Client-side required-field gating** in the generic `form` picker view.
   Import is disabled until every required field has a value
   (`formCanSubmit`), instead of letting an empty required field submit and
   surface a raw 422 toast. Mirrors what the `server_folder` view already did
   via `sfAbsolutePath`.

3. **Server path browser** for both Server importers, delivering on the
   "Browse the server's filesystem" copy that previously fronted a plain
   text box:
   - `server_files` (paths file): the `server_path` field now renders
     `<vt-file-browser>` (text input + Browse button + inline `/api/browse`
     folder browser, filtered by the field's `accept` extensions). Browse is
     rooted at `SERVER_ROOTS[0]`, matching `server_files` validation.
   - `server_folder` (path): a "Browse…" toggle reveals `<vt-folder-browser>`
     backed by `/api/browse-media-files?source=server_fs`, which returns
     `root_path` so the picked folder resolves to a true absolute path fed
     into the existing typed-path + detection flow. Typing/pasting a path
     still works.

## Open follow-ups

- **Browse-root vs. import-root seam (server_folder).** The `server_folder`
  browser + media-type detection are rooted at `/` (the `server_fs` source in
  `_resolve_browse_root`), but import validation now confines to `SERVER_ROOTS`.
  So you can navigate to and select a folder *outside* `SERVER_ROOTS` and only
  get rejected at Import (with a clear message). The `server_files` side has no
  such seam (its `/api/browse` root and validation root are both
  `SERVER_ROOTS[0]`). Closing this for `server_folder` means re-rooting
  `server_fs` browse/detect at `SERVER_ROOTS[0]` **and** reworking the `sf-*`
  path flow to pass root-relative paths (today `sfApplyPathInput` anchors at
  `/` and detection resolves against `/`). Deferred because that path-model
  rework is broad and the current behavior is correct, just not maximally tidy.

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
