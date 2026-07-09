# Dataset import from local archives

**Status: shipped.** Folder and URL importers can now read media out of a
local zip/tar/rar archive. Open follow-ups first; shipped detail below.

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

## What shipped

- **`vtscore/datasets/archive.py`** — central archive handling: `extract_archive`
  (`.zip`/`.tar(.gz|.bz2|.xz)`/`.rar`, validates every member against traversal via
  `_reject_traversal`); `extract_archive_cached` (extracts into a `DATA_DIR` dir
  keyed by path+size+mtime, so re-imports reuse one extraction and a replaced
  archive busts the cache); `is_archive_path`, `find_archives`, `append_medias`,
  `iter_archive_chunks`/`load_archive_into`.
- **`local_archive` origin** — media carry `{"importer": "local_archive", "params":
  {"path": <abs archive>, "media_type": ...}}`; only the archive path is persisted,
  resolved by `vtscore/datasets/sources/local_archive.py` (`LocalArchiveSource`,
  extracts cached then delegates to `LocalFolderSource`). Converter media re-derive
  via the converter origin's `parent_importer`/`parent_path`; PDF pages via the
  resolver's generic `params.path` fallback.
- **`server_folder`** — folder field accepts a single archive file; opt-in
  `dig_archives` checkbox extracts archives inside the scanned folder; CLI `is_dir`
  guard relaxed for archive paths.
- **`http_archive`** — accepts a local server path (not just a URL); routes through
  `load_archive_into` and emits `local_archive` origins.
- **Frontend** — "Dig into archives" checkbox on the server-folder view; the typed
  folder-path input already accepts an archive path.
- **`vtscore/datasets/pdf.py`** — PDF page-expansion helper (`load_pdf_images_into`)
  shared by the folder importer and archive loader.
