"""Dataset loading and management utilities.

All public functions that perform I/O accept an optional ``on_progress``
callback with the signature
``(status: str, message: str, current: int, total: int) -> None``.
When omitted the functions fall back to the application-wide
:func:`~vtsearch.utils.update_progress` reporter; pass an explicit callback
to use these functions outside the Flask app.
"""

import csv
import hashlib
import io
import pickle
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import numpy as np
from PIL import Image

from vtsearch.config import EMBEDDINGS_DIR
from vtsearch.datasets.config import DEMO_DATASETS

ProgressCallback = Callable[[str, str, int, int], None]


def _default_progress() -> ProgressCallback:
    """Lazily resolve the application-wide progress callback."""
    from vtsearch.utils import update_progress

    return update_progress


def load_esc50_metadata(esc50_dir: Path) -> dict[str, dict[str, Any]]:
    """Load clip metadata from the ESC-50 ``esc50.csv`` metadata file.

    Reads ``<esc50_dir>/meta/esc50.csv`` and builds a mapping from audio
    filename to its associated metadata fields.

    Args:
        esc50_dir: Path to the root ESC-50 dataset directory (the directory that
            contains the ``meta/`` and ``audio/`` subdirectories).

    Returns:
        A dict mapping audio filename (e.g. ``"1-100032-A-0.wav"``) to a dict
        with the keys:

        - ``"category"`` (``str``): Human-readable sound category label.
        - ``"esc10"`` (``bool``): Whether the clip belongs to the ESC-10 subset.
        - ``"target"`` (``int``): Integer class index.
        - ``"fold"`` (``int``): Cross-validation fold number (1–5).
    """
    meta_file = esc50_dir / "meta" / "esc50.csv"

    metadata = {}
    with open(meta_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            metadata[filename] = {
                "category": row["category"],
                "esc10": row["esc10"] == "True",
                "target": int(row["target"]),
                "fold": int(row["fold"]),
            }
    return metadata


def load_cifar10_batch(file_path: Path) -> tuple[np.ndarray, list[int], list[str]]:
    """Load a CIFAR-10 pickle batch file and return images, labels, and label names.

    Args:
        file_path: Path to a CIFAR-10 batch file (e.g. ``data_batch_1``) in the
            unpickled binary format used by the original dataset.

    Returns:
        A 3-tuple ``(images, labels, label_names)`` where:

        - ``images`` is a ``numpy.ndarray`` of shape ``(N, 32, 32, 3)`` with
          ``uint8`` pixel values in RGB order.
        - ``labels`` is a list of integer class indices (one per image), each in
          the range ``[0, 9]``.
        - ``label_names`` is a fixed list of 10 human-readable class name strings
          (e.g. ``"airplane"``, ``"automobile"``, …, ``"truck"``), ordered so that
          ``label_names[i]`` corresponds to label value ``i``.
    """
    with open(file_path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")

    # CIFAR-10 label names
    label_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    images = batch[b"data"]
    labels = batch[b"labels"]

    # Reshape images from (10000, 3072) to (10000, 32, 32, 3)
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    return images, labels, label_names


def load_video_metadata_from_folders(video_dir: Path, categories: list[str]) -> dict[str, dict[str, Any]]:
    """Scan category subdirectories and collect video file metadata.

    Iterates over immediate subdirectories of ``video_dir``, keeping only those
    whose name appears in ``categories``, and collects paths for all video files
    with common extensions (``mp4``, ``avi``, ``mov``, ``webm``, ``mkv``).

    Args:
        video_dir: Root directory whose immediate subdirectories represent
            category folders.
        categories: List of category folder names to include. Subdirectories
            not in this list are skipped.

    Returns:
        A dict mapping video filename (basename only) to a dict with the keys:

        - ``"category"`` (``str``): Name of the category folder.
        - ``"path"`` (``Path``): Full path to the video file.
    """
    metadata = {}

    for category_folder in video_dir.iterdir():
        if not category_folder.is_dir():
            continue

        category_name = category_folder.name
        if category_name not in categories:
            continue

        # Find all video files in this category
        # Use category/filename as key to avoid collisions across categories.
        for ext in ["*.mp4", "*.avi", "*.mov", "*.webm", "*.mkv"]:
            for video_path in category_folder.glob(ext):
                metadata[f"{category_name}/{video_path.name}"] = {
                    "category": category_name,
                    "path": video_path,
                }

    return metadata


def load_image_metadata_from_folders(image_dir: Path, categories: list[str]) -> dict[str, dict[str, Any]]:
    """Scan category subdirectories and collect image file metadata.

    Iterates over immediate subdirectories of ``image_dir``, keeping only those
    whose name appears in ``categories``, and collects paths for all image files
    with common extensions (``png``, ``jpg``, ``jpeg``, ``gif``, ``bmp``, ``webp``).

    Args:
        image_dir: Root directory whose immediate subdirectories represent
            category folders.
        categories: List of category folder names to include. Subdirectories
            not in this list are skipped.

    Returns:
        A dict mapping image filename (basename only) to a dict with the keys:

        - ``"category"`` (``str``): Name of the category folder.
        - ``"path"`` (``Path``): Full path to the image file.
    """
    metadata = {}

    for category_folder in image_dir.iterdir():
        if not category_folder.is_dir():
            continue

        category_name = category_folder.name
        if category_name not in categories:
            continue

        # Find all image files in this category
        # Use category/filename as key to avoid collisions across categories
        # (e.g. Caltech-101 uses image_XXXX.jpg in every category folder).
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.webp"]:
            for image_path in category_folder.glob(ext):
                metadata[f"{category_name}/{image_path.name}"] = {
                    "category": category_name,
                    "path": image_path,
                }

    return metadata


def load_paragraph_metadata_from_folders(text_dir: Path, categories: list[str]) -> dict[str, dict[str, Any]]:
    """Scan category subdirectories and collect text file metadata.

    Iterates over immediate subdirectories of ``text_dir``, keeping only those
    whose name appears in ``categories``, and collects paths for all plain-text
    files with extensions ``txt`` or ``md``.

    Args:
        text_dir: Root directory whose immediate subdirectories represent
            category folders.
        categories: List of category folder names to include. Subdirectories
            not in this list are skipped.

    Returns:
        A dict mapping text filename (basename only) to a dict with the keys:

        - ``"category"`` (``str``): Name of the category folder.
        - ``"path"`` (``Path``): Full path to the text file.
    """
    metadata = {}

    for category_folder in text_dir.iterdir():
        if not category_folder.is_dir():
            continue

        category_name = category_folder.name
        if category_name not in categories:
            continue

        # Find all text files in this category
        # Use category/filename as key to avoid collisions across categories.
        for ext in ["*.txt", "*.md"]:
            for text_path in category_folder.glob(ext):
                metadata[f"{category_name}/{text_path.name}"] = {
                    "category": category_name,
                    "path": text_path,
                }

    return metadata


def _streaming_md5(file_path: Path) -> str:
    """Compute MD5 hash of a file using constant memory."""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dataset_from_folder(
    folder_path: Path,
    media_type: str,
    clips: dict[int, dict[str, Any]],
    content_vectors: dict[str, Any] | None = None,
    content_md5s: dict[str, str] | None = None,
    on_progress: Optional[ProgressCallback] = None,
    origin: dict[str, Any] | None = None,
    thin: bool = False,
) -> None:
    """Generate a dataset in-place from a flat folder of media files.

    Scans ``folder_path`` for all files matching the extensions for ``media_type``,
    embeds each file using the appropriate model, and populates ``clips`` with
    the resulting clip dicts. Progress is reported via :func:`update_progress`.

    Files whose basename appears in ``content_vectors`` will use the supplied
    embedding instead of running the embedding model.  This allows importers
    that already provide content vectors to avoid redundant computation.

    Similarly, files whose basename appears in ``content_md5s`` will use the
    supplied hash instead of computing it from the file contents.

    The ``clips`` dict is cleared before loading begins.

    ``media_type`` is looked up in the media type registry by
    :attr:`~vtsearch.media.base.MediaType.folder_import_name` (e.g.
    ``"sounds"``, ``"videos"``, ``"images"``, ``"paragraphs"``).  Adding a
    new media type to the registry automatically makes it available here
    without any changes to this function.

    Args:
        folder_path: Path to a flat directory containing media files.
        media_type: Folder-import alias for the media type (e.g. ``"sounds"``).
        clips: Dict to populate in-place. Existing entries are removed before
            loading. Keys are sequential integer clip IDs starting from 1.
        content_vectors: Optional mapping of filename (basename) to a
            pre-computed embedding ``numpy.ndarray``.  When a file's name is
            found in this dict the supplied vector is used directly and the
            embedding model is not invoked for that file.
        content_md5s: Optional mapping of filename (basename) to a
            pre-computed MD5 hex digest string.  When a file's name is found
            in this dict the supplied hash is used directly and no MD5
            calculation is performed for that file.
        origin: Optional serialised
            :class:`~vtsearch.datasets.origin.Origin` dict to attach to each
            clip (as ``clip["origin"]``).  When ``None`` no origin is set
            and the caller is expected to set it afterwards.
        thin: When ``True``, store a ``media_path`` reference to the file on
            disk instead of reading all bytes into ``clip_bytes``.  This saves
            memory for CLI workflows that only need embeddings for scoring.
            MD5 is still computed via streaming (constant memory).

    Raises:
        ValueError: If ``media_type`` is not recognised, or if no matching
            files are found in ``folder_path``.
    """
    from vtsearch.media import get_by_folder_name

    if on_progress is None:
        on_progress = _default_progress()

    on_progress("loading", "Scanning media files...", 0, 0)

    try:
        mt = get_by_folder_name(media_type)
    except KeyError:
        raise ValueError(f"Invalid media type: {media_type}")

    # Eagerly load models before starting the embedding timer so that
    # download / weight-loading time does not pollute the progress bar.
    if getattr(mt, "_model", None) is None:
        mt.load_models()

    # Find all files of the specified media type
    media_files = []
    for ext in mt.file_extensions:
        media_files.extend(folder_path.glob(ext))

    if not media_files:
        raise ValueError(f"No {media_type} files found in folder")

    clips.clear()
    clip_id = 1
    total_files = len(media_files)

    for i, file_path in enumerate(media_files):
        on_progress(
            "embedding",
            f"Embedding {media_type} {file_path.name}...",
            i + 1,
            total_files,
        )

        if content_vectors and file_path.name in content_vectors:
            embedding = content_vectors[file_path.name]
        else:
            embedding = mt.embed_media(file_path)
            if embedding is None:
                continue

        if thin:
            # Thin mode: store file path reference, skip loading bytes.
            # Use stat for file_size and streaming hash for MD5.
            if content_md5s and file_path.name in content_md5s:
                md5 = content_md5s[file_path.name]
            else:
                md5 = _streaming_md5(file_path)
            clip_data: dict[str, Any] = {
                "id": clip_id,
                "type": mt.type_id,
                "file_size": file_path.stat().st_size,
                "md5": md5,
                "embedding": embedding,
                "filename": file_path.name,
                "category": "custom",
                "origin": origin,
                "origin_name": file_path.name,
                "clip_bytes": None,
                "clip_string": None,
                "media_path": str(file_path.resolve()),
                "duration": 0,
            }
        else:
            with open(file_path, "rb") as f:
                file_bytes = f.read()

            if content_md5s and file_path.name in content_md5s:
                md5 = content_md5s[file_path.name]
            else:
                md5 = hashlib.md5(file_bytes).hexdigest()

            # Build the base clip dict
            clip_data = {
                "id": clip_id,
                "type": mt.type_id,
                "file_size": len(file_bytes),
                "md5": md5,
                "embedding": embedding,
                "filename": file_path.name,
                "category": "custom",
                "origin": origin,
                "origin_name": file_path.name,
                # Null-out optional media fields so clips from different types
                # stored in the same dict have consistent keys.
                "clip_bytes": None,
                "clip_string": None,
                "media_path": str(file_path.resolve()),
                "duration": 0,
            }

            # Merge in media-specific fields from the media type
            clip_data.update(mt.load_clip_data(file_path))

        clips[clip_id] = clip_data
        clip_id += 1

    on_progress("idle", f"Loaded {len(clips)} {media_type} clips from folder")


def load_dataset_from_folder_chunked(
    folder_path: Path,
    media_type: str,
    chunk_size: int,
    content_vectors: dict[str, Any] | None = None,
    content_md5s: dict[str, str] | None = None,
    on_progress: Optional[ProgressCallback] = None,
    origin: dict[str, Any] | None = None,
    thin: bool = False,
) -> Iterator[dict[int, dict[str, Any]]]:
    """Yield chunks of clips from a flat folder of media files.

    Works identically to :func:`load_dataset_from_folder` but yields the
    clips in groups of at most *chunk_size*.  Each yielded dict is a
    self-contained clips dict with IDs starting at 1.  After the caller
    has processed a chunk, the dict can be discarded to free memory.

    Args:
        folder_path: Path to a flat directory containing media files.
        media_type: Folder-import alias (e.g. ``"sounds"``).
        chunk_size: Maximum number of clips per chunk.
        content_vectors: Optional pre-computed embeddings keyed by filename.
        content_md5s: Optional pre-computed MD5s keyed by filename.
        origin: Optional origin dict to attach to each clip.
        thin: When ``True``, store ``media_path`` instead of ``clip_bytes``.

    Yields:
        A dict mapping int clip IDs (starting at 1) to clip data dicts.
        Each yielded dict contains at most *chunk_size* clips.

    Raises:
        ValueError: If ``media_type`` is not recognised, or if no matching
            files are found in ``folder_path``.
    """
    from vtsearch.media import get_by_folder_name

    if on_progress is None:
        on_progress = _default_progress()

    on_progress("loading", "Scanning media files...", 0, 0)

    try:
        mt = get_by_folder_name(media_type)
    except KeyError:
        raise ValueError(f"Invalid media type: {media_type}")

    # Eagerly load models before starting the embedding timer so that
    # download / weight-loading time does not pollute the progress bar.
    if getattr(mt, "_model", None) is None:
        mt.load_models()

    # Find all files of the specified media type
    media_files: list[Path] = []
    for ext in mt.file_extensions:
        media_files.extend(folder_path.glob(ext))

    if not media_files:
        raise ValueError(f"No {media_type} files found in folder")

    total_files = len(media_files)

    # Process in groups of chunk_size
    for start in range(0, total_files, chunk_size):
        batch = media_files[start : start + chunk_size]
        chunk_clips: dict[int, dict[str, Any]] = {}
        clip_id = 1

        for i, file_path in enumerate(batch):
            global_idx = start + i
            on_progress(
                "embedding",
                f"Embedding {media_type} {file_path.name} (chunk {start // chunk_size + 1})...",
                global_idx + 1,
                total_files,
            )

            if content_vectors and file_path.name in content_vectors:
                embedding = content_vectors[file_path.name]
            else:
                embedding = mt.embed_media(file_path)
                if embedding is None:
                    continue

            if thin:
                if content_md5s and file_path.name in content_md5s:
                    md5 = content_md5s[file_path.name]
                else:
                    md5 = _streaming_md5(file_path)
                clip_data: dict[str, Any] = {
                    "id": clip_id,
                    "type": mt.type_id,
                    "file_size": file_path.stat().st_size,
                    "md5": md5,
                    "embedding": embedding,
                    "filename": file_path.name,
                    "category": "custom",
                    "origin": origin,
                    "origin_name": file_path.name,
                    "clip_bytes": None,
                    "clip_string": None,
                    "media_path": str(file_path.resolve()),
                    "duration": 0,
                }
            else:
                with open(file_path, "rb") as f:
                    file_bytes = f.read()

                if content_md5s and file_path.name in content_md5s:
                    md5 = content_md5s[file_path.name]
                else:
                    md5 = hashlib.md5(file_bytes).hexdigest()

                clip_data = {
                    "id": clip_id,
                    "type": mt.type_id,
                    "file_size": len(file_bytes),
                    "md5": md5,
                    "embedding": embedding,
                    "filename": file_path.name,
                    "category": "custom",
                    "origin": origin,
                    "origin_name": file_path.name,
                    "clip_bytes": None,
                    "clip_string": None,
                    "media_path": str(file_path.resolve()),
                    "duration": 0,
                }
                clip_data.update(mt.load_clip_data(file_path))

            chunk_clips[clip_id] = clip_data
            clip_id += 1

        if chunk_clips:
            yield chunk_clips

    on_progress("idle", f"Finished chunked loading of {total_files} {media_type} files")


def load_dataset_from_pickle(
    file_path: Path,
    clips: dict[int, dict[str, Any]],
    thin: bool = False,
) -> dict[str, Any] | None:
    """Load a dataset from a pickle file into the clips dict in-place.

    Supports two pickle formats:

    - **New format**: A dict with a ``"clips"`` key mapping to clip data dicts.
      May also include ``"audio_dir"``, ``"video_dir"``, ``"image_dir"``, or
      ``"text_dir"`` keys pointing to directories containing the raw media files
      when the bytes are not stored inline.
    - **Old format**: A plain dict mapping clip ID to clip data dict (no wrapping
      ``"clips"`` key).

    If media bytes are not stored inline in the pickle, the function attempts to
    load them from the companion directory entry in the pickle. Clips for which
    no media bytes can be resolved are silently skipped (a warning is printed to
    stdout after loading).

    The ``clips`` dict is cleared before loading begins.

    Args:
        file_path: Path to a ``.pkl`` file previously created by
            :func:`export_dataset_to_file` or :func:`load_demo_dataset`.
        clips: Dict to populate in-place. Existing entries are removed before
            loading. Keys are clip IDs (int); values are clip data dicts.
        thin: When ``True``, skip loading media bytes into memory.  Inline
            bytes from the pickle are discarded and external-dir files are
            referenced by ``media_path`` instead of read.  Useful for CLI
            workflows that only need embeddings for scoring.

    Returns:
        ``None``.  (Formerly returned ``creation_info``; that field has been
        removed.)
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    clips.clear()

    # Handle both old format (just clips dict) and new format (with metadata)
    if isinstance(data, dict) and "clips" in data:
        clips_data = data["clips"]
        # Old pickles may contain creation_info; extract a fallback origin
        # for clips that predate per-element origin tracking.
        creation_info = data.get("creation_info")
    else:
        clips_data = data
        creation_info = None

    fallback_origin = None
    if creation_info:
        fallback_origin = {
            "importer": creation_info.get("importer", "unknown"),
            "params": creation_info.get("field_values", {}),
        }

    # Build the dir_key mapping dynamically from the media type registry.
    # Also build the legacy-bytes-key mapping for backward compat with old pickles.
    from vtsearch.media import all_types

    _dir_keys: dict[str, str] = {}
    _legacy_bytes: dict[str, list[str]] = {}
    for mt in all_types():
        _dir_keys[mt.type_id] = mt.dir_key
        _legacy_bytes[mt.type_id] = mt.legacy_bytes_keys

    # Convert to the app's clip format
    missing_media = 0
    for clip_id, clip_info in clips_data.items():
        # Determine media type
        media_type = clip_info.get("type", "audio")

        if thin:
            # ── Thin mode: skip bytes, store media_path if available ──
            media_path: str | None = clip_info.get("media_path")

            # Try to resolve a media_path from the external directory
            if not media_path:
                dir_key = _dir_keys.get(media_type)
                if dir_key and dir_key in data and "filename" in clip_info:
                    candidate = Path(data[dir_key]) / clip_info["filename"]
                    if candidate.exists():
                        media_path = str(candidate.resolve())

            # We still need the embedding to be useful
            if "embedding" not in clip_info:
                missing_media += 1
                continue

            fname = clip_info.get("filename", f"clip_{clip_id}.{media_type}")
            clip_data: dict[str, Any] = {
                "id": clip_id,
                "type": media_type,
                "duration": clip_info.get("duration", 0),
                "file_size": clip_info.get("file_size", 0),
                "md5": clip_info.get("md5", ""),
                "embedding": np.array(clip_info["embedding"]),
                "clip_bytes": None,
                "clip_string": None,
                "media_path": media_path,
                "filename": fname,
                "category": clip_info.get("category", "unknown"),
                "origin": clip_info.get("origin", fallback_origin),
                "origin_name": clip_info.get("origin_name", fname),
            }
            # Preserve any extra metadata fields from the pickle
            for extra_key in ("width", "height", "word_count", "character_count"):
                if extra_key in clip_info:
                    clip_data[extra_key] = clip_info[extra_key]

            clips[clip_id] = clip_data
            continue

        # ── Full mode (original behaviour) ──
        # Load the actual media content.
        # Support both new key names (clip_bytes/clip_string) and legacy
        # per-media-type key names for backward compatibility with old pickles.
        clip_bytes = None
        clip_string = None
        media_bytes = None
        media_path = None

        # Try clip_bytes first (binary media), then legacy keys
        clip_bytes = clip_info.get("clip_bytes")
        if clip_bytes is None:
            for legacy_key in _legacy_bytes.get(media_type, []):
                clip_bytes = clip_info.get(legacy_key)
                if clip_bytes is not None:
                    break

        # Try clip_string (text media), then legacy keys
        clip_string = clip_info.get("clip_string")
        if clip_string is None:
            for legacy_key in _legacy_bytes.get(media_type, []):
                val = clip_info.get(legacy_key)
                if isinstance(val, str):
                    clip_string = val
                    break

        if clip_bytes is not None:
            media_bytes = clip_bytes
        elif clip_string is not None:
            media_bytes = clip_string.encode("utf-8")
        else:
            # Try loading from the external directory
            dir_key = _dir_keys.get(media_type)
            if dir_key and dir_key in data and "filename" in clip_info:
                ext_path = Path(data[dir_key]) / clip_info["filename"]
                if ext_path.exists():
                    if clip_string is None and ext_path.suffix in (".txt", ".md"):
                        with open(ext_path, "r", encoding="utf-8") as f:
                            clip_string = f.read()
                            media_bytes = clip_string.encode("utf-8")
                    else:
                        with open(ext_path, "rb") as f:
                            clip_bytes = f.read()
                            media_bytes = clip_bytes
                    media_path = str(ext_path.resolve())
                else:
                    missing_media += 1

        if media_bytes:
            fname = clip_info.get("filename", f"clip_{clip_id}.{media_type}")
            clip_data = {
                "id": clip_id,
                "type": media_type,
                "duration": clip_info.get("duration", 0),
                "file_size": clip_info.get("file_size", len(media_bytes)),
                "md5": clip_info.get("md5") or hashlib.md5(media_bytes).hexdigest(),
                "embedding": np.array(clip_info["embedding"]),
                "clip_bytes": clip_bytes,
                "clip_string": clip_string,
                "media_path": media_path or clip_info.get("media_path"),
                "filename": fname,
                "category": clip_info.get("category", "unknown"),
                "origin": clip_info.get("origin", fallback_origin),
                "origin_name": clip_info.get("origin_name", fname),
            }
            # Preserve any extra metadata fields from the pickle
            for extra_key in ("width", "height", "word_count", "character_count"):
                if extra_key in clip_info:
                    clip_data[extra_key] = clip_info[extra_key]

            clips[clip_id] = clip_data

    if missing_media > 0:
        print(f"WARNING: {missing_media} media files missing from {file_path}", flush=True)

    return None


def load_dataset_from_pickle_chunked(
    file_path: Path,
    chunk_size: int,
    thin: bool = False,
) -> Iterator[dict[int, dict[str, Any]]]:
    """Yield chunks of clips from a pickle dataset file.

    Works identically to :func:`load_dataset_from_pickle` but yields the
    clips in groups of at most *chunk_size*.  Each yielded dict is a
    self-contained clips dict with IDs starting at 1.

    The entire pickle is deserialized once (unavoidable for ``.pkl``
    format), but media bytes are dropped or skipped per-chunk so that
    only one chunk's worth of clip data is alive at a time.

    Args:
        file_path: Path to a ``.pkl`` dataset file.
        chunk_size: Maximum number of clips per yielded chunk.
        thin: When ``True``, skip loading media bytes into memory.

    Yields:
        A dict mapping int clip IDs (starting at 1) to clip data dicts.
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "clips" in data:
        clips_data = data["clips"]
        creation_info = data.get("creation_info")
    else:
        clips_data = data
        creation_info = None

    fallback_origin = None
    if creation_info:
        fallback_origin = {
            "importer": creation_info.get("importer", "unknown"),
            "params": creation_info.get("field_values", {}),
        }

    _dir_keys = {
        "audio": "audio_dir",
        "video": "video_dir",
        "image": "image_dir",
        "paragraph": "text_dir",
    }

    all_clip_ids = sorted(clips_data.keys())

    for start in range(0, len(all_clip_ids), chunk_size):
        batch_ids = all_clip_ids[start : start + chunk_size]
        chunk_clips: dict[int, dict[str, Any]] = {}
        new_id = 1

        for clip_id in batch_ids:
            clip_info = clips_data[clip_id]
            media_type = clip_info.get("type", "audio")

            if thin:
                media_path: str | None = clip_info.get("media_path")
                if not media_path:
                    dir_key = _dir_keys.get(media_type)
                    if dir_key and dir_key in data and "filename" in clip_info:
                        candidate = Path(data[dir_key]) / clip_info["filename"]
                        if candidate.exists():
                            media_path = str(candidate.resolve())

                if "embedding" not in clip_info:
                    continue

                fname = clip_info.get("filename", f"clip_{clip_id}.{media_type}")
                clip_data: dict[str, Any] = {
                    "id": new_id,
                    "type": media_type,
                    "duration": clip_info.get("duration", 0),
                    "file_size": clip_info.get("file_size", 0),
                    "md5": clip_info.get("md5", ""),
                    "embedding": np.array(clip_info["embedding"]),
                    "clip_bytes": None,
                    "clip_string": None,
                    "media_path": media_path,
                    "filename": fname,
                    "category": clip_info.get("category", "unknown"),
                    "origin": clip_info.get("origin", fallback_origin),
                    "origin_name": clip_info.get("origin_name", fname),
                }
                if media_type == "image":
                    clip_data["width"] = clip_info.get("width")
                    clip_data["height"] = clip_info.get("height")
                elif media_type == "paragraph":
                    clip_data["word_count"] = clip_info.get("word_count")
                    clip_data["character_count"] = clip_info.get("character_count")

                chunk_clips[new_id] = clip_data
                new_id += 1
                continue

            # Full mode — same logic as load_dataset_from_pickle
            clip_bytes = None
            clip_string = None
            media_bytes = None
            media_path = None

            if media_type == "audio":
                clip_bytes = clip_info.get("clip_bytes") or clip_info.get("wav_bytes")
                if clip_bytes is not None:
                    media_bytes = clip_bytes
                elif "filename" in clip_info and "audio_dir" in data:
                    audio_path = Path(data["audio_dir"]) / clip_info["filename"]
                    if audio_path.exists():
                        with open(audio_path, "rb") as f:
                            clip_bytes = f.read()
                            media_bytes = clip_bytes
                        media_path = str(audio_path.resolve())

            elif media_type == "video":
                clip_bytes = clip_info.get("clip_bytes") or clip_info.get("video_bytes")
                if clip_bytes is not None:
                    media_bytes = clip_bytes
                elif "filename" in clip_info and "video_dir" in data:
                    video_path = Path(data["video_dir"]) / clip_info["filename"]
                    if video_path.exists():
                        with open(video_path, "rb") as f:
                            clip_bytes = f.read()
                            media_bytes = clip_bytes
                        media_path = str(video_path.resolve())

            elif media_type == "image":
                clip_bytes = clip_info.get("clip_bytes") or clip_info.get("image_bytes")
                if clip_bytes is not None:
                    media_bytes = clip_bytes
                elif "filename" in clip_info and "image_dir" in data:
                    image_path = Path(data["image_dir"]) / clip_info["filename"]
                    if image_path.exists():
                        with open(image_path, "rb") as f:
                            clip_bytes = f.read()
                            media_bytes = clip_bytes
                        media_path = str(image_path.resolve())

            elif media_type == "paragraph":
                clip_string = clip_info.get("clip_string") or clip_info.get("text_content")
                if clip_string is not None:
                    media_bytes = clip_string.encode("utf-8")
                elif "filename" in clip_info and "text_dir" in data:
                    text_path = Path(data["text_dir"]) / clip_info["filename"]
                    if text_path.exists():
                        with open(text_path, "r", encoding="utf-8") as f:
                            clip_string = f.read()
                            media_bytes = clip_string.encode("utf-8")
                        media_path = str(text_path.resolve())

            if media_bytes:
                fname = clip_info.get("filename", f"clip_{clip_id}.{media_type}")
                clip_data = {
                    "id": new_id,
                    "type": media_type,
                    "duration": clip_info.get("duration", 0),
                    "file_size": clip_info.get("file_size", len(media_bytes)),
                    "md5": clip_info.get("md5") or hashlib.md5(media_bytes).hexdigest(),
                    "embedding": np.array(clip_info["embedding"]),
                    "clip_bytes": clip_bytes,
                    "clip_string": clip_string,
                    "media_path": media_path or clip_info.get("media_path"),
                    "filename": fname,
                    "category": clip_info.get("category", "unknown"),
                    "origin": clip_info.get("origin", fallback_origin),
                    "origin_name": clip_info.get("origin_name", fname),
                }
                if media_type == "image":
                    clip_data["width"] = clip_info.get("width")
                    clip_data["height"] = clip_info.get("height")
                elif media_type == "paragraph":
                    clip_data["word_count"] = clip_info.get("word_count")
                    clip_data["character_count"] = clip_info.get("character_count")

                chunk_clips[new_id] = clip_data
                new_id += 1

        if chunk_clips:
            yield chunk_clips


def embed_image_file_from_pil(image: Image.Image) -> Optional[np.ndarray]:
    """Generate a CLIP embedding vector for a PIL Image object.

    A convenience wrapper for cases where the image is already in memory
    (e.g. reconstructed from a NumPy array during CIFAR-10 loading).

    Delegates to :meth:`~vtsearch.media.image.media_type.ImageMediaType.embed_pil_image`.

    Args:
        image: A PIL Image in any mode.

    Returns:
        A 1-D ``numpy.ndarray`` of shape ``(embedding_dim,)``, or ``None`` if
        the CLIP model is not loaded or an exception occurs.
    """
    from vtsearch.media import get as media_get

    return media_get("image").embed_pil_image(image)


def load_demo_dataset(
    dataset_name: str,
    clips: dict[int, dict[str, Any]],
    e5_model: Any = None,
    on_progress: Optional[ProgressCallback] = None,
) -> None:
    """Load a named demo dataset into the clips dict, downloading and embedding as needed.

    Checks for a cached ``.pkl`` file in ``EMBEDDINGS_DIR``; if found, loads
    from that file. If the cache is missing or the media bytes it references can
    no longer be found on disk, the raw data is re-downloaded and re-embedded.

    Each media type implements its own
    :meth:`~vtsearch.media.base.MediaType.load_demo_source` method that
    handles downloading, embedding, and populating clips for its demo sources.
    This function simply orchestrates pickle caching around that delegation.

    Progress throughout the operation is reported via :func:`update_progress`.

    Args:
        dataset_name: Key into ``DEMO_DATASETS`` identifying which demo dataset
            to load.  Raises ``ValueError`` if the key is not found.
        clips: Dict to populate in-place. Existing entries are removed before
            loading. Keys are integer clip IDs; values are clip data dicts.
        e5_model: Deprecated — kept for backward compatibility but no longer
            used.  The text embedding model is obtained from the media type
            registry.

    Raises:
        ValueError: If ``dataset_name`` is not in ``DEMO_DATASETS``, or if the
            media type does not support the requested demo source.
    """
    if on_progress is None:
        on_progress = _default_progress()

    if dataset_name not in DEMO_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset_info = DEMO_DATASETS[dataset_name]
    media_type_id = dataset_info.get("media_type", "audio")

    # Check if already embedded
    pkl_file = EMBEDDINGS_DIR / f"{dataset_name}.pkl"
    if pkl_file.exists():
        on_progress("loading", f"Loading {dataset_name} dataset...", 0, 0)
        load_dataset_from_pickle(pkl_file, clips)

        # Check if any clips were actually loaded
        if len(clips) == 0:
            # Pickle file exists but media files are missing, delete and re-embed
            on_progress("loading", f"Media files missing, re-embedding {dataset_name}...", 0, 0)
            pkl_file.unlink()
        else:
            on_progress("idle", f"Loaded {dataset_name} dataset")
            return

    # Delegate to the media type's load_demo_source() method
    from vtsearch.media import get as media_get

    mt = media_get(media_type_id)

    source = dataset_info.get("source", "")
    categories = dataset_info["categories"]
    slice_start = dataset_info.get("slice_start", 0)
    slice_end = dataset_info.get("slice_end")

    clips.clear()
    external_dir = mt.load_demo_source(
        source=source,
        categories=categories,
        slice_start=slice_start,
        slice_end=slice_end,
        clips=clips,
        on_progress=on_progress,
    )

    # Stamp the demo origin on all clips
    demo_origin: dict[str, Any] = {"importer": "demo", "params": {"name": dataset_name}}
    for clip in clips.values():
        clip["origin"] = demo_origin

    # Build the pickle cache payload
    # For types with external media dirs (audio, video), exclude clip_bytes
    # from the pickle and store the dir path so reloading can find the files.
    if external_dir is not None:
        pkl_data: dict[str, Any] = {
            "name": dataset_name,
            "clips": {
                cid: {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in clip.items() if k != "clip_bytes"}
                for cid, clip in clips.items()
            },
            mt.dir_key: external_dir,
        }
    else:
        pkl_data = {
            "name": dataset_name,
            "clips": {
                cid: {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in clip.items()}
                for cid, clip in clips.items()
            },
        }

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(pkl_file, "wb") as f:
        pickle.dump(pkl_data, f)

    on_progress("idle", f"Loaded {dataset_name} dataset")


def export_dataset_to_file(
    clips: dict[int, dict[str, Any]],
) -> bytes:
    """Serialise the current clip dataset to a pickle-formatted byte string.

    Converts the in-memory ``clips`` dict to a portable format (converting any
    ``numpy.ndarray`` embeddings to plain Python lists) and returns it as bytes
    suitable for writing to a ``.pkl`` file or sending as an HTTP response.

    The resulting bytes can be reloaded with :func:`load_dataset_from_pickle`.

    Args:
        clips: Mapping of clip ID to clip data dict.

    Returns:
        Raw bytes of the pickled dataset dict.
    """
    data: dict[str, Any] = {
        "clips": {
            cid: {
                "id": clip["id"],
                "type": clip.get("type", "audio"),
                "duration": clip["duration"],
                "file_size": clip["file_size"],
                "md5": clip["md5"],
                "embedding": clip["embedding"].tolist()
                if isinstance(clip["embedding"], np.ndarray)
                else clip["embedding"],
                "filename": clip.get("filename", f"clip_{cid}.wav"),
                "category": clip.get("category", "unknown"),
                "origin": clip.get("origin"),
                "origin_name": clip.get("origin_name", clip.get("filename", "")),
                "clip_bytes": clip.get("clip_bytes"),
                "clip_string": clip.get("clip_string"),
                "media_path": clip.get("media_path"),
                "word_count": clip.get("word_count"),
                "character_count": clip.get("character_count"),
                "width": clip.get("width"),
                "height": clip.get("height"),
            }
            for cid, clip in clips.items()
        }
    }

    buf = io.BytesIO()
    pickle.dump(data, buf)
    buf.seek(0)
    return buf.getvalue()
