# #5339: Add Dataset's media-type choice leaves out the types the server hides

The harness supplies no face models, so the server hides Face, but the Folder
importer's single-dataset media-type choice still offered a bare `face` entry.
It now offers only the types `/api/media-types` lists.

- `step-import-train` — the Folder importer's media-type choice no longer lists `face`
- `step-import-test` — the same importer, the same choice
- `importer-form` — the Folder importer with its file browser open
