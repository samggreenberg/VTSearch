# Server / Services dataset-import UX

Status: **Phase 1 shipped.** Single-user path confinement (item 1 below) was
later **reversed**: in single-user / no-auth mode server-path import, export,
and browse are now unrestricted (the lone trusted user may reach any
server-readable path), and `SERVER_ROOTS` / `VTSEARCH_SERVER_ROOTS` were
removed. Multi-user per-user confinement is unchanged. The "Browse-root vs.
import-root seam" follow-up below is therefore moot for single-user mode
(browse is now rooted at `/`, matching the unrestricted import validation).

Came out of a hands-on audit of the three real-world
import paths (Services extension plugin, server folder, server file with
pre-computed vectors), driving the live UI. The Demo importer is well-trodden;
these three are what real deployments use and were comparatively rough.

## ~~What shipped~~

All struck through (note: the single-user `SERVER_ROOTS` confinement below was
later reversed — see the status header):

- ~~**Unified Server-tab path policy.**~~ `server_folder`'s `path` validated
  through `validate_server_filepath` like `server_files`; checks against every
  `SERVER_ROOTS` entry; closed a multi-user confinement hole; fixed the
  misleading `get_file_access_base_dir` docstring.
- ~~**Client-side required-field gating**~~ in the generic `form` picker
  (`formCanSubmit` disables Import until required fields are filled).
- ~~**Server path browser**~~ for both Server importers — `<vt-file-browser>` on
  `server_files`' `server_path`, a "Browse…" `<vt-folder-browser>` toggle on
  `server_folder` (backed by `/api/browse-media-files?source=server_fs`).

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
