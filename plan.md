# Move demo dataset definitions from MediaType into the demo importer

## Problem

Every `MediaType` subclass carries hundreds of lines of demo dataset constants
(category lists, folder paths, `DemoDataset` objects) in its `demo_datasets`
property. These are **only** consumed by the demo importer pipeline
(`DemoDatasetImporter` → `load_demo_dataset()` → `load_demo_source()`). If the
demo importer is removed, all this code stays behind as dead weight in each
media type.

## Plan

### 1. Create `vtsearch/datasets/importers/demo/datasets.py`

Move **all** demo dataset definitions here — one function per media type that
returns a `list[DemoDataset]`:

- `_audio_demo_datasets() -> list[DemoDataset]` — ESC-50, GTZAN, Speech Commands, UrbanSound8K category lists + DemoDataset objects
- `_image_demo_datasets() -> list[DemoDataset]` — Caltech-101/256, Oxford Flowers, Food-101, EuroSAT, Stanford Dogs, UCSF Documents category lists + DemoDataset objects
- `_text_demo_datasets() -> list[DemoDataset]` — 20 Newsgroups, AG News, BBC News, IMDB category lists + DemoDataset objects
- `_video_demo_datasets() -> list[DemoDataset]` — UCF-101 category lists + DemoDataset objects

Plus a public `all_demo_datasets() -> dict` that assembles the flat
`{dataset_id: info_dict}` mapping (currently in `vtsearch/media/__init__.py`).

### 2. Update `vtsearch/datasets/config.py`

Change the import from `vtsearch.media.all_demo_datasets` to
`vtsearch.datasets.importers.demo.datasets.all_demo_datasets`.

### 3. Remove from each `MediaType`

- Delete the `demo_datasets` property and all `_DEMO_CATEGORIES_*` class
  constants from `AudioMediaType`, `ImageMediaType`, `TextMediaType`,
  `VideoMediaType`, and `DocumentMediaType`.
- Remove the `@abstractmethod demo_datasets` from `MediaType` base class.
- Remove `DemoDataset` import from media type modules that no longer need it.

### 4. Remove `all_demo_datasets()` from `vtsearch/media/__init__.py`

This function is no longer needed since the demo importer owns the definitions.

### 5. Keep `load_demo_source()` on `MediaType`

`load_demo_source()` contains the actual media-type-specific download +
embedding logic. It stays on `MediaType` since it needs access to
media-type-specific internals (embedder loading, PIL processing, audio
waveform handling, etc.). The demo importer calls it via `load_demo_dataset()`
passing the params it owns.

### 6. Update `DemoDataset` dataclass

Move `DemoDataset` from `vtsearch/media/base.py` to
`vtsearch/datasets/importers/demo/datasets.py` since it's only used by the
demo system. Update imports accordingly.

### 7. Update tests

- Tests that import `DemoDataset` from `vtsearch.media.base` need updated
  imports.
- Tests that check `mt.demo_datasets` need to instead check the new
  `all_demo_datasets()` or the per-type functions.
- The `required_folder` comment fix in `test_datasets.py` stays.

### 8. Run tests

Run `./run-tests.sh` to verify nothing breaks.

## Files changed

- **New**: `vtsearch/datasets/importers/demo/datasets.py`
- **Modified**: `vtsearch/datasets/config.py` (import change)
- **Modified**: `vtsearch/media/base.py` (remove `DemoDataset` class, remove `demo_datasets` abstract property)
- **Modified**: `vtsearch/media/__init__.py` (remove `all_demo_datasets()`)
- **Modified**: `vtsearch/media/audio/media_type.py` (remove demo constants + property)
- **Modified**: `vtsearch/media/image/media_type.py` (remove demo constants + property)
- **Modified**: `vtsearch/media/text/media_type.py` (remove demo constants + property)
- **Modified**: `vtsearch/media/video/media_type.py` (remove demo constants + property)
- **Modified**: `vtsearch/media/document/media_type.py` (remove empty demo_datasets property)
- **Modified**: Various test files (import path updates)
