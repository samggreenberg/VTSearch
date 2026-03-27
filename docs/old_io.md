# Retired IO Modules — Reference Implementations

These importers and exporters were removed from the UI but are preserved here
as reference implementations for developers building new IO plugins.  They are
**not loaded, tested, or supported** — treat them as code samples.

> **Warning:** These code samples may be outdated. In particular, the current
> codebase uses the media type ID `"text"` (not `"paragraph"`). Several code
> examples below still use `"paragraph"` — substitute `"text"` if reusing
> any of this code. Review and test thoroughly.

See [EXTENDING.md](EXTENDING.md) for the current plugin interfaces and the
S3 importer skeleton.

---

## Exporters

### Local JSON File Exporter

Returns auto-detect results as JSON for browser download (GUI) or writes to a
local file path (CLI).  No extra dependencies.

```python
# Was: vtsearch/exporters/local_json_file/__init__.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vtsearch.exporters.base import ExporterField, LabelsetExporter


class LocalJsonLabelsetExporter(LabelsetExporter):
    name = "local_json_file"
    display_name = "Local JSON File"
    description = "Download the results as a JSON file to your local machine."
    icon = "\U0001f4be"  # floppy disk
    fields = [
        ExporterField(
            key="filepath",
            label="File Path",
            field_type="text",
            description=(
                "Filename for the downloaded JSON file (used by the CLI; "
                "the browser uses its own save-as dialog)."
            ),
            placeholder="autodetect_results.json",
            default="autodetect_results.json",
            required=False,
        ),
    ]

    def export(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        total_hits = sum(r.get("total_hits", 0) for r in results.get("results", {}).values())
        return {
            "message": (
                f"Prepared {total_hits} hit(s) across "
                f"{results.get('detectors_run', 0)} detector(s) for download."
            ),
            "download_content": json.dumps(results, indent=2),
            "download_filename": field_values.get("filepath", "").strip() or "autodetect_results.json",
            "download_content_type": "application/json",
        }

    def export_cli(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        filepath_str = (field_values.get("filepath") or "").strip() or "autodetect_results.json"
        filepath = Path(filepath_str)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(json.dumps(results, indent=2), encoding="utf-8")
        total_hits = sum(r.get("total_hits", 0) for r in results.get("results", {}).values())
        return {
            "message": (
                f"Saved {total_hits} hit(s) across "
                f"{results.get('detectors_run', 0)} detector(s) "
                f"to {filepath.resolve()}."
            ),
            "filepath": str(filepath.resolve()),
        }


EXPORTER = LocalJsonLabelsetExporter()
```

### Local CSV File Exporter

Returns auto-detect results as CSV for browser download.  No extra dependencies.

```python
# Was: vtsearch/exporters/local_csv_file/__init__.py

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from vtsearch.exporters.base import ExporterField, LabelsetExporter


class LocalCsvLabelsetExporter(LabelsetExporter):
    name = "local_csv_file"
    display_name = "Local CSV File"
    description = "Download the results as a CSV file to your local machine."
    icon = "\U0001f4ca"  # bar chart
    fields = [
        ExporterField(
            key="filepath",
            label="File Path",
            field_type="text",
            description=(
                "Filename for the downloaded CSV file (used by the CLI; "
                "the browser uses its own save-as dialog)."
            ),
            placeholder="autodetect_results.csv",
            default="autodetect_results.csv",
            required=False,
        ),
    ]

    def export(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        csv_content, total_hits = _build_csv_string(results)
        return {
            "message": (
                f"Prepared {total_hits} hit(s) across "
                f"{results.get('detectors_run', 0)} detector(s) for download."
            ),
            "download_content": csv_content,
            "download_filename": field_values.get("filepath", "").strip() or "autodetect_results.csv",
            "download_content_type": "text/csv",
        }

    def export_cli(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        filepath_str = (field_values.get("filepath") or "").strip() or "autodetect_results.csv"
        filepath = Path(filepath_str)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        csv_content, total_hits = _build_csv_string(results)
        filepath.write_text(csv_content, encoding="utf-8")
        return {
            "message": (
                f"Saved {total_hits} hit(s) across "
                f"{results.get('detectors_run', 0)} detector(s) "
                f"to {filepath.resolve()}."
            ),
            "filepath": str(filepath.resolve()),
        }


def _build_csv_string(results: dict[str, Any]) -> tuple[str, int]:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["detector", "threshold", "filename", "category", "score", "origin", "origin_name"])
    total_hits = 0
    for det_result in results.get("results", {}).values():
        detector_name = det_result.get("detector_name", "unknown")
        threshold = det_result.get("threshold", "")
        for hit in det_result.get("hits", []):
            origin = hit.get("origin")
            origin_str = ""
            if origin:
                from vtsearch.datasets.origin import Origin
                origin_str = Origin.from_dict(origin).display()
            writer.writerow([
                detector_name, threshold,
                hit.get("filename", ""), hit.get("category", ""),
                hit.get("score", ""), origin_str, hit.get("origin_name", ""),
            ])
            total_hits += 1
    return buf.getvalue(), total_hits


EXPORTER = LocalCsvLabelsetExporter()
```

---

## Label Importers

### Local JSON Label Importer

Loads labels from a `.json` file uploaded via the browser.

```python
# Was: vtsearch/labels/importers/local_json_file/__init__.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vtsearch.labels.importers.base import LabelImporter, LabelImporterField


class LocalJsonLabelImporter(LabelImporter):
    name = "local_json_file"
    display_name = "Local JSON File"
    description = "Import labels from a VTSearch-format JSON file uploaded from your local machine."
    icon = "\U0001f4c4"  # page facing up
    fields = [
        LabelImporterField(
            key="file",
            label="Labels JSON File",
            field_type="file",
            accept=".json",
            description="A JSON file with a 'labels' list of {md5, label} objects.",
        ),
    ]

    def run(self, field_values: dict[str, Any]) -> list[dict[str, str]]:
        file_storage = field_values.get("file")
        if file_storage is None:
            raise ValueError("No file provided.")
        try:
            raw = file_storage.read()
        except AttributeError:
            raise ValueError("Expected a file upload, not a string. Use run_cli for CLI usage.")
        return _parse_json_bytes(raw)

    def run_cli(self, field_values: dict[str, Any]) -> list[dict[str, str]]:
        filepath = field_values.get("file", "").strip()
        if not filepath:
            raise ValueError("--file is required.")
        raw = Path(filepath).read_bytes()
        return _parse_json_bytes(raw)


def _parse_json_bytes(raw: bytes) -> list[dict[str, str]]:
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc
    labels = data.get("labels")
    if not isinstance(labels, list):
        raise ValueError("JSON must contain a top-level 'labels' list.")
    return [entry for entry in labels if isinstance(entry, dict)]


LABEL_IMPORTER = LocalJsonLabelImporter()
```

### Local CSV Label Importer

Loads labels from a `.csv` file uploaded via the browser.

```python
# Was: vtsearch/labels/importers/local_csv_file/__init__.py

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from vtsearch.labels.importers.base import LabelImporter, LabelImporterField


class LocalCsvLabelImporter(LabelImporter):
    name = "local_csv_file"
    display_name = "Local CSV File"
    description = "Import labels from a CSV file uploaded from your local machine."
    icon = "\U0001f4ca"  # bar chart
    fields = [
        LabelImporterField(
            key="file",
            label="Labels CSV File",
            field_type="file",
            accept=".csv",
            description="A CSV file with header row containing 'md5' and 'label' columns.",
        ),
    ]

    def run(self, field_values: dict[str, Any]) -> list[dict[str, str]]:
        file_storage = field_values.get("file")
        if file_storage is None:
            raise ValueError("No file provided.")
        try:
            raw = file_storage.read()
        except AttributeError:
            raise ValueError("Expected a file upload, not a string. Use run_cli for CLI usage.")
        return _parse_csv_bytes(raw)

    def run_cli(self, field_values: dict[str, Any]) -> list[dict[str, str]]:
        filepath = field_values.get("file", "").strip()
        if not filepath:
            raise ValueError("--file is required.")
        raw = Path(filepath).read_bytes()
        return _parse_csv_bytes(raw)


def _parse_csv_bytes(raw: bytes) -> list[dict[str, str]]:
    try:
        text = raw.decode("utf-8-sig")  # strip BOM if present
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ValueError("CSV file appears to be empty.")
    normalised = {k.strip().lower(): k for k in reader.fieldnames if k}
    if "md5" not in normalised or "label" not in normalised:
        raise ValueError("CSV must have 'md5' and 'label' column headers.")
    results = []
    for row in reader:
        md5 = row.get(normalised["md5"], "").strip()
        label = row.get(normalised["label"], "").strip().lower()
        if md5 and label:
            results.append({"md5": md5, "label": label})
    return results


LABEL_IMPORTER = LocalCsvLabelImporter()
```

---

## Processor Importers

### Local Detector File Importer

Loads a detector from a `.json` file uploaded via the browser.

```python
# Was: vtsearch/processors/importers/detector_file/__init__.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vtsearch.processors.importers.base import ProcessorImporter, ProcessorImporterField


class LocalFileProcessorImporter(ProcessorImporter):
    name = "detector_file"
    display_name = "Local Detector File (.json)"
    description = "Import a pre-trained detector from a JSON file uploaded from your local machine."
    icon = "\U0001f4c4"  # page facing up
    fields = [
        ProcessorImporterField(
            key="file",
            label="Detector JSON File",
            field_type="file",
            accept=".json",
            description="A VTSearch detector JSON file with weights and threshold.",
        ),
    ]

    def run(self, field_values: dict[str, Any]) -> dict[str, Any]:
        file_storage = field_values.get("file")
        if file_storage is None:
            raise ValueError("No file provided.")
        try:
            raw = file_storage.read()
        except AttributeError:
            raise ValueError("Expected a file upload, not a string. Use run_cli for CLI usage.")
        return _parse_detector_json(raw)

    def run_cli(self, field_values: dict[str, Any]) -> dict[str, Any]:
        filepath = field_values.get("file", "").strip()
        if not filepath:
            raise ValueError("--file is required.")
        raw = Path(filepath).read_bytes()
        return _parse_detector_json(raw)


def _parse_detector_json(raw: bytes) -> dict[str, Any]:
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc
    weights = data.get("weights")
    if not weights:
        raise ValueError("Detector file missing 'weights' field.")
    threshold = data.get("threshold", 0.5)
    media_type = data.get("media_type", "audio")
    suggested_name = data.get("name", "")
    result: dict[str, Any] = {
        "media_type": media_type,
        "weights": weights,
        "threshold": threshold,
    }
    if suggested_name:
        result["name"] = suggested_name
    return result


PROCESSOR_IMPORTER = LocalFileProcessorImporter()
```

### Label File Processor Importer

Trains a detector from a JSON file of labelled media file paths.  Requires
embedding models to be available.

```python
# Was: vtsearch/processors/importers/label_file/__init__.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vtsearch.processors.importers.base import ProcessorImporter, ProcessorImporterField


class LabelFileProcessorImporter(ProcessorImporter):
    name = "label_file"
    display_name = "Label File (.json)"
    description = "Train a new detector from a JSON file listing labelled media paths."
    icon = "\U0001f3f7\ufe0f"  # label
    fields = [
        ProcessorImporterField(
            key="file", label="Labels JSON File", field_type="file",
            accept=".json",
            description="A JSON file with a 'labels' list of {path, label} objects.",
        ),
        ProcessorImporterField(
            key="media_type", label="Media Type", field_type="select",
            options=["", "audio", "image", "video", "paragraph"],
            default="", required=False,
            description="Override auto-detected media type (leave blank to auto-detect).",
        ),
    ]

    def run(self, field_values: dict[str, Any]) -> dict[str, Any]:
        file_storage = field_values.get("file")
        if file_storage is None:
            raise ValueError("No file provided.")
        try:
            raw = file_storage.read()
        except AttributeError:
            raise ValueError("Expected a file upload, not a string.")
        media_type_hint = (field_values.get("media_type") or "").strip()
        return _train_from_labels(raw, media_type_hint)

    def run_cli(self, field_values: dict[str, Any]) -> dict[str, Any]:
        filepath = field_values.get("file", "").strip()
        if not filepath:
            raise ValueError("--file is required.")
        raw = Path(filepath).read_bytes()
        media_type_hint = (field_values.get("media_type") or "").strip()
        return _train_from_labels(raw, media_type_hint)


# Extension-to-media-type lookup
_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".webm", ".mkv"}
_TEXT_EXTS = {".txt", ".md"}


def _media_type_for_path(p: Path) -> str | None:
    ext = p.suffix.lower()
    if ext in _AUDIO_EXTS: return "audio"
    if ext in _IMAGE_EXTS: return "image"
    if ext in _VIDEO_EXTS: return "video"
    if ext in _TEXT_EXTS:  return "paragraph"
    return None


def _train_from_labels(raw: bytes, media_type_hint: str) -> dict[str, Any]:
    """Parse label JSON, embed referenced files, and train an MLP detector."""
    import numpy as np
    import torch
    from vtsearch.models import calculate_cross_calibration_threshold, train_model
    from vtsearch.utils import get_inclusion

    data = json.loads(raw.decode("utf-8"))
    labels = data.get("labels", [])
    if not labels:
        raise ValueError("No labels found in file.")

    X_list, y_list = [], []
    loaded_count = skipped_count = 0
    detected_media_type = media_type_hint or None

    for entry in labels:
        label = entry.get("label")
        if label not in ("good", "bad"):
            skipped_count += 1; continue
        file_path_str = entry.get("path") or entry.get("file") or entry.get("filename")
        if not file_path_str:
            skipped_count += 1; continue
        file_path = Path(file_path_str)
        if not file_path.exists():
            skipped_count += 1; continue
        mt = media_type_hint or _media_type_for_path(file_path)
        if mt is None:
            skipped_count += 1; continue
        if detected_media_type is None:
            detected_media_type = mt
        elif detected_media_type != mt:
            skipped_count += 1; continue
        # embedding = _embed(mt, file_path)  # requires model loading
        # X_list.append(embedding); y_list.append(1.0 if label == "good" else 0.0)
        loaded_count += 1

    # ... train model, return weights + threshold ...


PROCESSOR_IMPORTER = LabelFileProcessorImporter()
```

### CSV Label File Processor Importer

Same as Label File but reads a CSV instead of JSON.  Depends on the
Label File importer's `_media_type_for_path` and `_embed` helpers.

```python
# Was: vtsearch/processors/importers/csv_label_file/__init__.py

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from vtsearch.processors.importers.base import ProcessorImporter, ProcessorImporterField


class CsvLabelFileProcessorImporter(ProcessorImporter):
    name = "csv_label_file"
    display_name = "Label File (.csv)"
    description = "Train a new detector from a CSV file listing labelled media paths."
    icon = "\U0001f4ca"  # bar chart
    fields = [
        ProcessorImporterField(
            key="file", label="Labels CSV File", field_type="file",
            accept=".csv",
            description="A CSV file with 'path' and 'label' columns.",
        ),
        ProcessorImporterField(
            key="media_type", label="Media Type", field_type="select",
            options=["", "audio", "image", "video", "paragraph"],
            default="", required=False,
            description="Override auto-detected media type (leave blank to auto-detect).",
        ),
    ]

    def run(self, field_values: dict[str, Any]) -> dict[str, Any]:
        file_storage = field_values.get("file")
        if file_storage is None:
            raise ValueError("No file provided.")
        try:
            raw = file_storage.read()
        except AttributeError:
            raise ValueError("Expected a file upload, not a string.")
        media_type_hint = (field_values.get("media_type") or "").strip()
        return _train_from_csv_labels(raw, media_type_hint)


def _parse_csv_bytes(raw: bytes) -> list[dict[str, str]]:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ValueError("CSV file appears to be empty.")
    normalised = {k.strip().lower(): k for k in reader.fieldnames if k}
    path_key = None
    for candidate in ("path", "file", "filename"):
        if candidate in normalised:
            path_key = normalised[candidate]; break
    if path_key is None:
        raise ValueError("CSV must have a 'path' (or 'file' or 'filename') column header.")
    if "label" not in normalised:
        raise ValueError("CSV must have a 'label' column header.")
    label_key = normalised["label"]
    results = []
    for row in reader:
        path_val = row.get(path_key, "").strip()
        label_val = row.get(label_key, "").strip().lower()
        if path_val and label_val:
            results.append({"path": path_val, "label": label_val})
    return results


def _train_from_csv_labels(raw: bytes, media_type_hint: str) -> dict[str, Any]:
    # Uses same training logic as label_file importer
    entries = _parse_csv_bytes(raw)
    if not entries:
        raise ValueError("No labels found in CSV file.")
    # ... embed files, train model, return weights + threshold ...


PROCESSOR_IMPORTER = CsvLabelFileProcessorImporter()
```

---

## Dataset Importers

### RSS / Podcast Feed Importer

Downloads audio enclosures from an RSS feed.  **Requires:** `feedparser`.

```python
# Was: vtsearch/datasets/importers/rss_feed/__init__.py

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Iterator

import requests

from vtsearch.config import DATA_DIR
from vtsearch.datasets.importers.base import DatasetImporter, ImporterField


class RSSDatasetImporter(DatasetImporter):
    name = "rss_feed"
    display_name = "Generate from RSS Podcast Feed"
    description = "Import media files from an RSS or podcast feed URL."
    icon = "\U0001f3b5"
    fields = [
        ImporterField(key="url", label="Feed URL", field_type="url",
                      description="URL of an RSS or Atom feed with media enclosures."),
        ImporterField(key="media_type", label="Media Type", field_type="select",
                      options=all_folder_names(), default="sounds",  # from vtsearch.media
                      description="Type of media to extract from the feed."),
        ImporterField(key="max_episodes", label="Max Episodes", field_type="text",
                      default="50", required=False,
                      description="Maximum number of episodes to download (0 = all)."),
    ]

    def run(self, field_values: dict[str, Any], medias: dict, thin: bool = False) -> None:
        import feedparser
        from vtsearch.datasets.loader import load_dataset_from_folder
        from vtsearch.media import get_by_folder_name

        url = field_values["url"]
        media_type = field_values.get("media_type", "sounds")
        max_episodes = int(field_values.get("max_episodes", "0") or "0")

        feed = feedparser.parse(url)
        if feed.bozo and not feed.entries:
            raise ValueError(f"Failed to parse feed: {feed.bozo_exception}")

        mt = get_by_folder_name(media_type)
        valid_exts = {ext.lstrip("*.").lower() for ext in mt.file_extensions}

        enclosures = []
        for entry in feed.entries:
            for link in getattr(entry, "enclosures", []):
                href = link.get("href", "")
                if href:
                    enclosures.append((entry.get("title", ""), href))
            if max_episodes and len(enclosures) >= max_episodes:
                enclosures = enclosures[:max_episodes]; break

        if not enclosures:
            raise ValueError("No media enclosures found in the feed.")

        download_dir = DATA_DIR / "rss_feed_download"
        download_dir.mkdir(parents=True, exist_ok=True)

        for i, (title, href) in enumerate(enclosures, 1):
            url_filename = href.split("?")[0].rstrip("/").split("/")[-1] or f"episode_{i}"
            short_hash = hashlib.md5(href.encode()).hexdigest()[:8]
            dest = download_dir / f"{short_hash}_{url_filename}"
            resp = requests.get(href, stream=True, timeout=120)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

        load_dataset_from_folder(download_dir, media_type, medias, thin=thin)


IMPORTER = RSSDatasetImporter()
```

### YouTube Playlist Importer

Downloads videos via yt-dlp and embeds them.  **Requires:** `yt-dlp`.

```python
# Was: vtsearch/datasets/importers/youtube_playlist/__init__.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterator

from vtsearch.config import DATA_DIR
from vtsearch.datasets.importers.base import DatasetImporter, ImporterField


class YouTubePlaylistDatasetImporter(DatasetImporter):
    name = "youtube_playlist"
    display_name = "Generate from YouTube Playlist"
    description = "Download videos from a YouTube playlist or channel URL via yt-dlp."
    icon = "\U0001f3ac"
    fields = [
        ImporterField(key="url", label="Playlist / Channel URL", field_type="url",
                      description="URL of a YouTube playlist, channel, or single video."),
        ImporterField(key="media_type", label="Media Type", field_type="select",
                      options=["videos", "sounds"], default="videos",
                      description="How to treat the downloaded files."),
        ImporterField(key="max_videos", label="Max Videos", field_type="text",
                      default="20", required=False,
                      description="Maximum number of videos to download (0 = all)."),
    ]

    def run(self, field_values: dict[str, Any], medias: dict, thin: bool = False) -> None:
        import yt_dlp
        from vtsearch.datasets.loader import load_dataset_from_folder

        url = field_values["url"]
        media_type = field_values.get("media_type", "videos")
        max_videos = int(field_values.get("max_videos", "0") or "0")

        download_dir = DATA_DIR / "youtube_playlist_download"
        download_dir.mkdir(parents=True, exist_ok=True)

        ydl_opts = {
            "format": "mp4[height<=720]/best[height<=720]/mp4/best",
            "outtmpl": str(download_dir / "%(id)s.%(ext)s"),
            "quiet": True, "no_warnings": True, "ignoreerrors": True,
        }
        if max_videos:
            ydl_opts["playlistend"] = max_videos
        if media_type == "sounds":
            ydl_opts["format"] = "bestaudio/best"
            ydl_opts["postprocessors"] = [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }]

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        if not list(download_dir.iterdir()):
            raise ValueError("No videos were downloaded.")

        load_dataset_from_folder(download_dir, media_type, medias, thin=thin)


IMPORTER = YouTubePlaylistDatasetImporter()
```
