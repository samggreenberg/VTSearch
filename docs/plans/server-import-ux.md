# Server / Services dataset-import UX

**Status:** Phase 1 shipped; the remaining work is the open follow-ups below
(empty Services tab, relative-entry resolution in manifests, and the multi-user
browse-root vs. import-root seam).

**Background:** Single-user path confinement (`SERVER_ROOTS` /
`VTSEARCH_SERVER_ROOTS`) was removed — in single-user / no-auth mode server-path
import, export, and browse are now unrestricted (the lone trusted user may reach
any server-readable path); multi-user per-user confinement is unchanged. This
makes the "Browse-root vs. import-root seam" follow-up below moot for single-user
mode (browse is now rooted at `/`, matching the unrestricted import validation).

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
