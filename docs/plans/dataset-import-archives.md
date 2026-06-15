# Dataset import from local archives

**Status: shipped.** Folder and URL importers can now read media out of a
local zip/tar/rar archive. Open follow-ups are listed at the bottom.

## Motivation

Before this change there was no way to import a local archive of media:

- `server_folder` scanned a directory by media-extension and silently skipped
  any `.zip` / `.tar` inside it.
- `http_archive` could extract archives but only from an `http(s)://` URL (it
  streamed via `download_file_with_progress`); a local path in its
  "Path or URL" field was rejected.

## What shipped

- **`vtscore/datasets/archive.py`** — central home for archive handling:
  - `extract_archive` (moved here from the `http_archive` importer, with the
    `_reject_traversal` guard) — extracts `.zip` / `.tar(.gz|.bz2|.xz)` /
    `.rar`, validating every member against path traversal first.
  - `extract_archive_cached` — extracts into a directory under `DATA_DIR`
    keyed by the archive's path + size + mtime, so re-imports and later
    resolves reuse one extraction and a replaced archive busts the cache.
  - `is_archive_path`, `find_archives`, `append_medias`.
  - `iter_archive_chunks` / `load_archive_into` — load an archive's contents
    through the same direct/PDF/converter pipeline as the folder loader.
- **`local_archive` origin** — media loaded from an archive carry
  `{"importer": "local_archive", "params": {"path": <abs archive>, "media_type": ...}}`.
  Only the archive path is persisted; the system re-derives
  `origin → archive → extracted file → embedding` on demand (no-persist rule).
  - Resolved by **`vtscore/datasets/sources/local_archive.py`**
    (`LocalArchiveSource`, auto-discovered as `name="local_archive"`), which
    extracts (cached) and delegates to `LocalFolderSource`.
  - Converter-derived media re-derive via the converter origin's
    `parent_importer` / `parent_path`; PDF pages via the resolver's generic
    `params.path` fallback against the cached extraction dir.
- **`server_folder`** — the folder field accepts a single archive file; a new
  opt-in **`dig_archives`** checkbox extracts archives found inside the scanned
  folder. The CLI `is_dir` guard was relaxed to allow archive paths.
- **`http_archive`** — accepts a local server path (not just a URL); local
  paths route through `load_archive_into` and emit `local_archive` origins.
- **Frontend** — "Dig into archives" checkbox added to the server-folder view;
  the existing typed folder-path input already accepts an archive path.
- The PDF page-expansion helper moved to `vtscore/datasets/pdf.py`
  (`load_pdf_images_into`), shared by the folder importer and archive loader.

## Open follow-ups

- **Media-type detection on archive paths.** The server-folder picker's
  auto-detect (`detectMediaType('server_fs', path, recursive)`) runs against a
  directory; when the path is an archive it returns nothing and the user picks
  the media type manually. Could peek inside the archive to suggest a type.
- **PDF-in-archive provenance.** PDF *pages* extracted from a PDF inside an
  archive carry a `pdf` origin whose `params.path` points at the cached
  extraction directory (resolved via the generic path fallback). This works as
  long as the cache survives; it is not re-derivable through the archive the
  way direct/converter media are. Niche (a zip of PDFs imported as *image*
  pages — importing PDFs as the `document` type goes through the direct path
  and is fully re-derivable).
- **Slow CLI subprocess tests** (`-m slow`, ~290s) were not run for this
  change; the fast suite plus new `run_cli` unit tests cover the paths.
