# Dataset import from local archives

**Status:** The open follow-ups below remain; core archive import (folder and
URL importers reading media out of a local zip/tar/rar archive) has shipped.

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
