"""Dataset loading and management utilities.

All public functions that perform I/O accept an optional ``on_progress``
callback with the signature
``(status: str, message: str, current: int, total: int) -> None``.
When omitted the functions fall back to the application-wide
:func:`~vtsearch.utils.update_progress` reporter; pass an explicit callback
to use these functions outside the Flask app.
"""

from __future__ import annotations

import csv
import gc
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


# ---------------------------------------------------------------------------
# Restricted unpickler – prevents arbitrary code execution from .pkl files
# ---------------------------------------------------------------------------

# Allowlist of (module, name) pairs that may be instantiated during unpickling.
# VTSearch pickles only contain plain Python containers and numpy arrays.
_PICKLE_SAFE_CLASSES: set[tuple[str, str]] = {
    # builtins (needed for dict/list/set/bytes subclasses & booleans)
    ("builtins", "set"),
    ("builtins", "frozenset"),
    ("builtins", "bytes"),
    ("builtins", "bytearray"),
    ("builtins", "complex"),
    # collections
    ("collections", "OrderedDict"),
    # numpy – reconstruction helpers used by numpy's __reduce__
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy", "_core.multiarray._reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy.core.numeric", "_frombuffer"),
    ("numpy._core.numeric", "_frombuffer"),
}


class RestrictedUnpickler(pickle.Unpickler):
    """An ``Unpickler`` that refuses to instantiate classes outside an allowlist.

    Plain Python primitives (int, float, str, None, bool, dict, list, tuple)
    are handled by the pickle protocol directly and never trigger
    ``find_class``.  This restricts only explicit class/callable references
    in the pickle stream.
    """

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in _PICKLE_SAFE_CLASSES:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Forbidden pickle class: {module}.{name}. Only plain Python types and numpy arrays are allowed."
        )


def safe_pickle_load(f: io.BufferedIOBase, **kwargs: Any) -> Any:
    """Deserialise a pickle stream using the restricted unpickler.

    Drop-in replacement for ``pickle.load(f)`` that blocks arbitrary code
    execution.  Any extra keyword arguments (e.g. ``encoding``) are
    forwarded to the underlying ``Unpickler``.
    """
    return RestrictedUnpickler(f, **kwargs).load()


def _default_progress() -> ProgressCallback:
    """Lazily resolve the application-wide progress callback."""
    from vtsearch.utils import update_progress

    return update_progress


def load_esc50_metadata(esc50_dir: Path) -> dict[str, dict[str, Any]]:
    """Load media metadata from the ESC-50 ``esc50.csv`` metadata file.

    Reads ``<esc50_dir>/meta/esc50.csv`` and builds a mapping from audio
    filename to its associated metadata fields.

    Args:
        esc50_dir: Path to the root ESC-50 dataset directory (the directory that
            contains the ``meta/`` and ``audio/`` subdirectories).

    Returns:
        A dict mapping audio filename (e.g. ``"1-100032-A-0.wav"``) to a dict
        with the keys:

        - ``"category"`` (``str``): Human-readable sound category label.
        - ``"esc10"`` (``bool``): Whether the media belongs to the ESC-10 subset.
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


def load_audio_metadata_from_folders(audio_dir: Path, categories: list[str]) -> dict[str, dict[str, Any]]:
    """Scan category subdirectories and collect audio file metadata.

    Iterates over immediate subdirectories of ``audio_dir``, keeping only those
    whose name appears in ``categories``, and collects paths for all audio files
    with common extensions (``wav``, ``mp3``, ``flac``, ``ogg``, ``m4a``, ``au``).

    Args:
        audio_dir: Root directory whose immediate subdirectories represent
            category folders.
        categories: List of category folder names to include. Subdirectories
            not in this list are skipped.

    Returns:
        A dict mapping ``category/filename`` to a dict with the keys:

        - ``"category"`` (``str``): Name of the category folder.
        - ``"path"`` (``Path``): Full path to the audio file.
    """
    metadata: dict[str, dict[str, Any]] = {}

    for category_folder in audio_dir.iterdir():
        if not category_folder.is_dir():
            continue

        category_name = category_folder.name
        if category_name not in categories:
            continue

        for ext in ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a", "*.au"]:
            for audio_path in category_folder.glob(ext):
                metadata[f"{category_name}/{audio_path.name}"] = {
                    "category": category_name,
                    "path": audio_path,
                }

    return metadata


def load_urbansound8k_metadata(us8k_dir: Path) -> dict[str, dict[str, Any]]:
    """Load metadata from the UrbanSound8K CSV file.

    Reads ``<us8k_dir>/metadata/UrbanSound8K.csv`` and builds a mapping from
    audio filename to its associated metadata fields.

    Args:
        us8k_dir: Path to the root UrbanSound8K directory (the directory that
            contains the ``metadata/`` and ``audio/`` subdirectories).

    Returns:
        A dict mapping audio filename (e.g. ``"100032-3-0-0.wav"``) to a dict
        with the keys:

        - ``"category"`` (``str``): Human-readable class label.
        - ``"fold"`` (``int``): Fold number (1–10).
        - ``"class_id"`` (``int``): Integer class index.
        - ``"path"`` (``Path``): Full path to the audio file.
    """
    meta_file = us8k_dir / "metadata" / "UrbanSound8K.csv"
    metadata: dict[str, dict[str, Any]] = {}

    with open(meta_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["slice_file_name"]
            fold = int(row["fold"])
            audio_path = us8k_dir / "audio" / f"fold{fold}" / filename
            metadata[filename] = {
                "category": row["class"],
                "fold": fold,
                "class_id": int(row["classID"]),
                "path": audio_path,
            }

    return metadata


def load_oxford_flowers_metadata(flowers_dir: Path, categories: list[str]) -> dict[str, dict[str, Any]]:
    """Load metadata for the Oxford Flowers 102 dataset.

    Reads the ``imagelabels.mat`` file and maps numeric labels (1–102) to
    category names using the provided ``categories`` list.  Images are
    stored in a flat ``jpg/`` directory with filenames ``image_NNNNN.jpg``.

    Args:
        flowers_dir: Path to the root Oxford Flowers directory (contains
            ``jpg/`` and ``imagelabels.mat``).
        categories: Ordered list of 102 flower species names (index 0
            corresponds to label 1 in the MAT file).

    Returns:
        A dict mapping image filename to a dict with the keys:

        - ``"category"`` (``str``): Flower species name from *categories*.
        - ``"path"`` (``Path``): Full path to the image file.
    """
    import scipy.io  # noqa: PLC0415

    mat_path = flowers_dir / "imagelabels.mat"
    mat = scipy.io.loadmat(str(mat_path))
    labels = mat["labels"][0]  # 1-indexed array of length 8189

    jpg_dir = flowers_dir / "jpg"
    metadata: dict[str, dict[str, Any]] = {}

    for i, label in enumerate(labels):
        # Labels are 1-indexed; categories list is 0-indexed.
        cat_idx = int(label) - 1
        if cat_idx < 0 or cat_idx >= len(categories):
            continue
        cat_name = categories[cat_idx]
        # Oxford Flowers images are named image_00001.jpg .. image_08189.jpg
        fname = f"image_{i + 1:05d}.jpg"
        img_path = jpg_dir / fname
        metadata[fname] = {
            "category": cat_name,
            "path": img_path,
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
        batch = safe_pickle_load(f, encoding="bytes")

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
        A dict mapping ``category/filename`` to a dict with the keys:

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
    medias: dict[int, dict[str, Any]],
    content_vectors: dict[str, Any] | None = None,
    content_md5s: dict[str, str] | None = None,
    on_progress: Optional[ProgressCallback] = None,
    origin: dict[str, Any] | None = None,
    thin: bool = False,
    embedder_name: str = "",
    custom_metadata_map: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Generate a dataset in-place from a flat folder of media files.

    Scans ``folder_path`` for all files matching the extensions for ``media_type``,
    embeds each file using the appropriate model, and populates ``medias`` with
    the resulting media dicts. Progress is reported via :func:`update_progress`.

    Files whose basename appears in ``content_vectors`` will use the supplied
    embedding instead of running the embedding model.  This allows importers
    that already provide content vectors to avoid redundant computation.

    Similarly, files whose basename appears in ``content_md5s`` will use the
    supplied hash instead of computing it from the file contents.

    The ``medias`` dict is cleared before loading begins.

    ``media_type`` is looked up in the media type registry by
    :attr:`~vtsearch.media.base.MediaType.folder_import_name`.  Adding a
    new media type to the registry automatically makes it available here
    without any changes to this function.

    Args:
        folder_path: Path to the root directory containing media files.
            Subdirectories are scanned recursively.
        media_type: Folder-import alias for the media type (e.g. ``"sounds"``).
        medias: Dict to populate in-place. Existing entries are removed before
            loading. Keys are sequential integer media IDs starting from 1.
        content_vectors: Optional mapping of filename to a pre-computed
            embedding ``numpy.ndarray``.  Keys may be relative paths
            (``"subdir/file.wav"``) or basenames (``"file.wav"``); relative
            paths are checked first for an exact match, then basenames as a
            fallback.
        content_md5s: Optional mapping of filename to a pre-computed MD5 hex
            digest string.  Keys follow the same lookup logic as
            ``content_vectors`` (relative path first, then basename).
        origin: Optional serialised
            :class:`~vtsearch.datasets.origin.Origin` dict to attach to each
            media (as ``media["origin"]``).  When ``None`` no origin is set
            and the caller is expected to set it afterwards.
        thin: When ``True``, store a ``media_path`` reference to the file on
            disk instead of reading all bytes into ``media_bytes``.  This saves
            memory for CLI workflows that only need embeddings for scoring.
            MD5 is still computed via streaming (constant memory).
        embedder_name: Optional name of a registered embedder to use.
            When empty, the first registered embedder for the media type
            is used.
        custom_metadata_map: Optional mapping of filename to a metadata dict.
            Keys follow the same lookup logic as ``content_vectors`` (relative
            path first, then basename).  When a metadata dict contains a
            non-empty ``"md5"`` key, that value is used as the media's MD5
            instead of computing it from the file contents.  The metadata dict
            is also attached to the media as ``custom_metadata``.

    Raises:
        ValueError: If ``media_type`` is not recognised, or if no matching
            files are found in ``folder_path``.
    """
    from vtsearch.media import embedders_for_type, get_by_folder_name, get_embedder

    if on_progress is None:
        on_progress = _default_progress()

    on_progress("loading", "Scanning media files...", 0, 0)

    try:
        mt = get_by_folder_name(media_type)
    except KeyError:
        raise ValueError(f"Invalid media type: {media_type}")

    # Resolve the embedder
    emb = None
    if embedder_name:
        try:
            emb = get_embedder(embedder_name)
        except KeyError:
            raise ValueError(f"Unknown embedder: {embedder_name}")
    else:
        avail = embedders_for_type(mt.type_id)
        if avail:
            emb = avail[0]

    # Eagerly load models before starting the embedding timer so that
    # download / weight-loading time does not pollute the progress bar.
    if emb is not None and getattr(emb, "_model", None) is None:
        from vtsearch.media.base import intercept_tqdm_progress

        on_progress("loading", "Loading embedding model…", 0, 0)
        with intercept_tqdm_progress(on_progress):
            emb.load_models()

    # Find all files of the specified media type (recursive so that
    # subdirectory structures are preserved).
    media_files = []
    for ext in mt.file_extensions:
        media_files.extend(folder_path.rglob(ext))

    if not media_files:
        raise ValueError(f"No {media_type} files found in folder")

    medias.clear()
    media_id = 1
    total_files = len(media_files)

    try:
        for i, file_path in enumerate(media_files):
            # Preserve relative path from the import root so that files in
            # different subdirectories with the same basename stay distinct.
            rel_path = file_path.relative_to(folder_path).as_posix()

            on_progress(
                "embedding",
                f"Embedding {media_type} {rel_path}...",
                i + 1,
                total_files,
            )

            # Look up pre-computed vectors by relative path first, then
            # fall back to basename for backward compatibility.
            if content_vectors and rel_path in content_vectors:
                embedding = content_vectors[rel_path]
            elif content_vectors and file_path.name in content_vectors:
                embedding = content_vectors[file_path.name]
            else:
                if emb is None:
                    continue
                embedding = emb.embed_media(file_path)
                if embedding is None:
                    continue

            embedder_id = emb.name if emb else ""

            # Look up per-file custom metadata (relative path first, then
            # basename fallback — same lookup order as content_vectors).
            file_cm: dict[str, Any] | None = None
            if custom_metadata_map:
                if rel_path in custom_metadata_map:
                    file_cm = custom_metadata_map[rel_path]
                elif file_path.name in custom_metadata_map:
                    file_cm = custom_metadata_map[file_path.name]

            # Resolve MD5: custom_metadata > content_md5s > computed
            cm_md5 = (file_cm.get("md5") or "") if file_cm else ""

            if thin:
                # Thin mode: store file path reference, skip loading bytes.
                # Use stat for file_size and streaming hash for MD5.
                if cm_md5:
                    md5 = cm_md5
                elif content_md5s and rel_path in content_md5s:
                    md5 = content_md5s[rel_path]
                elif content_md5s and file_path.name in content_md5s:
                    md5 = content_md5s[file_path.name]
                else:
                    md5 = _streaming_md5(file_path)
                media_data: dict[str, Any] = {
                    "id": media_id,
                    "type": mt.type_id,
                    "embedder": embedder_id,
                    "file_size": file_path.stat().st_size,
                    "md5": md5,
                    "embedding": embedding,
                    "filename": rel_path,
                    "category": "custom",
                    "origin": origin,
                    "origin_name": rel_path,
                    "media_bytes": None,
                    "media_string": None,
                    "media_path": str(file_path.resolve()),
                    "duration": 0,
                }
            else:
                with open(file_path, "rb") as f:
                    file_bytes = f.read()

                if cm_md5:
                    md5 = cm_md5
                elif content_md5s and rel_path in content_md5s:
                    md5 = content_md5s[rel_path]
                elif content_md5s and file_path.name in content_md5s:
                    md5 = content_md5s[file_path.name]
                else:
                    md5 = hashlib.md5(file_bytes).hexdigest()

                # Build the base media dict
                media_data = {
                    "id": media_id,
                    "type": mt.type_id,
                    "embedder": embedder_id,
                    "file_size": len(file_bytes),
                    "md5": md5,
                    "embedding": embedding,
                    "filename": rel_path,
                    "category": "custom",
                    "origin": origin,
                    "origin_name": rel_path,
                    # Null-out optional media fields so medias from different types
                    # stored in the same dict have consistent keys.
                    "media_bytes": None,
                    "media_string": None,
                    "media_path": str(file_path.resolve()),
                    "duration": 0,
                }

                # Merge in media-specific fields from the media type
                media_data.update(mt.load_media_data(file_path))

            if file_cm:
                media_data["custom_metadata"] = file_cm

            medias[media_id] = media_data
            media_id += 1
    except MemoryError:
        medias.clear()
        gc.collect()
        raise MemoryError(
            f"Out of memory after loading {media_id - 1} of {total_files} files. "
            "Try a smaller dataset or free up system RAM."
        )

    on_progress("idle", f"Loaded {len(medias)} {media_type} medias from folder")


def apply_custom_metadata_md5(media_dict: dict[int, dict[str, Any]]) -> int:
    """Use MD5 hashes from custom_metadata when available.

    If a media item's ``custom_metadata`` contains a non-empty ``"md5"`` key,
    that value is used as the item's ``"md5"`` instead of whatever was
    calculated during loading.  This lets importers supply authoritative hashes
    from their data source without recalculation.

    Args:
        media_dict: The mutable medias dict.  Modified in place.

    Returns:
        The number of media items whose MD5 was replaced.
    """
    count = 0
    for media in media_dict.values():
        cm = media.get("custom_metadata")
        if not cm:
            continue
        cm_md5 = cm.get("md5")
        if cm_md5:
            media["md5"] = cm_md5
            count += 1
    return count


def load_dataset_from_folder_chunked(
    folder_path: Path,
    media_type: str,
    chunk_size: int,
    content_vectors: dict[str, Any] | None = None,
    content_md5s: dict[str, str] | None = None,
    on_progress: Optional[ProgressCallback] = None,
    origin: dict[str, Any] | None = None,
    thin: bool = False,
    embedder_name: str = "",
    custom_metadata_map: dict[str, dict[str, Any]] | None = None,
) -> Iterator[dict[int, dict[str, Any]]]:
    """Yield chunks of medias from a folder of media files.

    Works identically to :func:`load_dataset_from_folder` but yields the
    medias in groups of at most *chunk_size*.  Each yielded dict is a
    self-contained medias dict with IDs starting at 1.  After the caller
    has processed a chunk, the dict can be discarded to free memory.

    Args:
        folder_path: Path to the root directory containing media files.
            Subdirectories are scanned recursively.
        media_type: Folder-import alias (e.g. ``"sounds"``).
        chunk_size: Maximum number of medias per chunk.
        content_vectors: Optional pre-computed embeddings keyed by filename
            (relative path or basename; relative path is tried first).
        content_md5s: Optional pre-computed MD5s keyed by filename (same
            lookup logic as *content_vectors*).
        origin: Optional origin dict to attach to each media.
        thin: When ``True``, store ``media_path`` instead of ``media_bytes``.
        embedder_name: Optional name of a registered embedder to use.
            When empty, the first registered embedder for the media type
            is used.
        custom_metadata_map: Optional mapping of filename to a metadata dict.
            Same semantics as in :func:`load_dataset_from_folder`.

    Yields:
        A dict mapping int media IDs (starting at 1) to media data dicts.
        Each yielded dict contains at most *chunk_size* medias.

    Raises:
        ValueError: If ``media_type`` is not recognised, or if no matching
            files are found in ``folder_path``.
    """
    from vtsearch.media import embedders_for_type, get_by_folder_name, get_embedder

    if on_progress is None:
        on_progress = _default_progress()

    on_progress("loading", "Scanning media files...", 0, 0)

    try:
        mt = get_by_folder_name(media_type)
    except KeyError:
        raise ValueError(f"Invalid media type: {media_type}")

    # Resolve the embedder
    emb = None
    if embedder_name:
        try:
            emb = get_embedder(embedder_name)
        except KeyError:
            raise ValueError(f"Unknown embedder: {embedder_name}")
    else:
        avail = embedders_for_type(mt.type_id)
        if avail:
            emb = avail[0]

    # Eagerly load models before starting the embedding timer so that
    # download / weight-loading time does not pollute the progress bar.
    if emb is not None and getattr(emb, "_model", None) is None:
        from vtsearch.media.base import intercept_tqdm_progress

        on_progress("loading", "Loading embedding model…", 0, 0)
        with intercept_tqdm_progress(on_progress):
            emb.load_models()

    # Find all files of the specified media type (recursive so that
    # subdirectory structures are preserved).
    media_files: list[Path] = []
    for ext in mt.file_extensions:
        media_files.extend(folder_path.rglob(ext))

    if not media_files:
        raise ValueError(f"No {media_type} files found in folder")

    total_files = len(media_files)
    embedder_id = emb.name if emb else ""

    # Process in groups of chunk_size
    for start in range(0, total_files, chunk_size):
        batch = media_files[start : start + chunk_size]
        chunk_medias: dict[int, dict[str, Any]] = {}
        media_id = 1

        for i, file_path in enumerate(batch):
            global_idx = start + i
            rel_path = file_path.relative_to(folder_path).as_posix()

            on_progress(
                "embedding",
                f"Embedding {media_type} {rel_path} (chunk {start // chunk_size + 1})...",
                global_idx + 1,
                total_files,
            )

            if content_vectors and rel_path in content_vectors:
                embedding = content_vectors[rel_path]
            elif content_vectors and file_path.name in content_vectors:
                embedding = content_vectors[file_path.name]
            else:
                if emb is None:
                    continue
                embedding = emb.embed_media(file_path)
                if embedding is None:
                    continue

            # Look up per-file custom metadata (same logic as non-chunked).
            file_cm: dict[str, Any] | None = None
            if custom_metadata_map:
                if rel_path in custom_metadata_map:
                    file_cm = custom_metadata_map[rel_path]
                elif file_path.name in custom_metadata_map:
                    file_cm = custom_metadata_map[file_path.name]

            cm_md5 = (file_cm.get("md5") or "") if file_cm else ""

            if thin:
                if cm_md5:
                    md5 = cm_md5
                elif content_md5s and rel_path in content_md5s:
                    md5 = content_md5s[rel_path]
                elif content_md5s and file_path.name in content_md5s:
                    md5 = content_md5s[file_path.name]
                else:
                    md5 = _streaming_md5(file_path)
                media_data: dict[str, Any] = {
                    "id": media_id,
                    "type": mt.type_id,
                    "embedder": embedder_id,
                    "file_size": file_path.stat().st_size,
                    "md5": md5,
                    "embedding": embedding,
                    "filename": rel_path,
                    "category": "custom",
                    "origin": origin,
                    "origin_name": rel_path,
                    "media_bytes": None,
                    "media_string": None,
                    "media_path": str(file_path.resolve()),
                    "duration": 0,
                }
            else:
                with open(file_path, "rb") as f:
                    file_bytes = f.read()

                if cm_md5:
                    md5 = cm_md5
                elif content_md5s and rel_path in content_md5s:
                    md5 = content_md5s[rel_path]
                elif content_md5s and file_path.name in content_md5s:
                    md5 = content_md5s[file_path.name]
                else:
                    md5 = hashlib.md5(file_bytes).hexdigest()

                media_data = {
                    "id": media_id,
                    "type": mt.type_id,
                    "embedder": embedder_id,
                    "file_size": len(file_bytes),
                    "md5": md5,
                    "embedding": embedding,
                    "filename": rel_path,
                    "category": "custom",
                    "origin": origin,
                    "origin_name": rel_path,
                    "media_bytes": None,
                    "media_string": None,
                    "media_path": str(file_path.resolve()),
                    "duration": 0,
                }
                media_data.update(mt.load_media_data(file_path))

            if file_cm:
                media_data["custom_metadata"] = file_cm

            chunk_medias[media_id] = media_data
            media_id += 1

        if chunk_medias:
            yield chunk_medias

    on_progress("idle", f"Finished chunked loading of {total_files} {media_type} files")


def load_dataset_from_pickle(
    file_path: Path,
    medias: dict[int, dict[str, Any]],
    thin: bool = False,
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any] | None:
    """Load a dataset from a pickle file into the medias dict in-place.

    Supports two pickle formats:

    - **New format**: A dict with a ``"medias"`` key mapping to media data dicts.
      May also include ``"audio_dir"``, ``"video_dir"``, ``"image_dir"``, or
      ``"text_dir"`` keys pointing to directories containing the raw media files
      when the bytes are not stored inline.
    - **Old format**: A plain dict mapping media ID to media data dict (no wrapping
      ``"medias"`` key).

    If media bytes are not stored inline in the pickle, the function attempts to
    load them from the companion directory entry in the pickle. Clips for which
    no media bytes can be resolved are silently skipped (a warning is printed to
    stdout after loading).

    The ``medias`` dict is cleared before loading begins.

    Args:
        file_path: Path to a ``.pkl`` file previously created by
            :func:`export_dataset_to_file` or :func:`load_demo_dataset`.
        medias: Dict to populate in-place. Existing entries are removed before
            loading. Keys are media IDs (int); values are media data dicts.
        thin: When ``True``, skip loading media bytes into memory.  Inline
            bytes from the pickle are discarded and external-dir files are
            referenced by ``media_path`` instead of read.  Useful for CLI
            workflows that only need embeddings for scoring.

    Returns:
        ``None``.  (Formerly returned ``creation_info``; that field has been
        removed.)
    """
    if on_progress is not None:
        on_progress("loading", f"Reading {file_path.name}…", 0, 0)

    try:
        with open(file_path, "rb") as f:
            data = safe_pickle_load(f)
    except MemoryError:
        gc.collect()
        raise MemoryError(
            f"Out of memory while reading {file_path.name}. The pickle file is too large for available RAM."
        )

    medias.clear()

    # Handle both old format (just medias dict) and new format (with metadata).
    # Also support legacy "clips" key from pickles saved before the rename.
    if isinstance(data, dict) and ("medias" in data or "clips" in data):
        medias_data = data["medias"] if "medias" in data else data["clips"]
        # Old pickles may contain creation_info; extract a fallback origin
        # for medias that predate per-element origin tracking.
        creation_info = data.get("creation_info")
    else:
        medias_data = data
        creation_info = None

    fallback_origin = None
    if creation_info:
        fallback_origin = {
            "importer": creation_info.get("importer", "unknown"),
            "params": creation_info.get("field_values", {}),
        }

    # Build lookup tables dynamically from the media type registry.
    from vtsearch.media import all_types

    _dir_keys: dict[str, str] = {}
    _legacy_bytes: dict[str, list[str]] = {}
    _extra_fields: dict[str, list[str]] = {}
    for mt in all_types():
        _dir_keys[mt.type_id] = mt.dir_key
        _legacy_bytes[mt.type_id] = mt.legacy_bytes_keys
        _extra_fields[mt.type_id] = mt.pickle_extra_fields

    # Convert to the app's media format
    missing_media = 0
    loaded_count = 0
    total_count = len(medias_data)
    # Report progress every ~2% or at least every 50 items
    _progress_interval = max(1, min(50, total_count // 50)) if total_count > 0 else 1
    if on_progress is not None:
        on_progress("loading", f"Processing 0 of {total_count} items…", 0, total_count)
    try:
        for media_id, media_info in medias_data.items():
            # Determine media type
            media_type = media_info.get("type", "audio")

            if thin:
                # ── Thin mode: skip bytes, store media_path if available ──
                media_path: str | None = media_info.get("media_path")

                # Try to resolve a media_path from the external directory
                if not media_path:
                    dir_key = _dir_keys.get(media_type)
                    if dir_key and dir_key in data and "filename" in media_info:
                        candidate = Path(data[dir_key]) / media_info["filename"]
                        if candidate.exists():
                            media_path = str(candidate.resolve())

                # We still need the embedding to be useful
                if "embedding" not in media_info:
                    missing_media += 1
                    continue

                fname = media_info.get("filename", f"media_{media_id}.{media_type}")
                media_data: dict[str, Any] = {
                    "id": media_id,
                    "type": media_type,
                    "embedder": media_info.get("embedder", ""),
                    "duration": media_info.get("duration", 0),
                    "file_size": media_info.get("file_size", 0),
                    "md5": media_info.get("md5", ""),
                    "embedding": np.array(media_info["embedding"]),
                    "media_bytes": None,
                    "media_string": None,
                    "media_path": media_path,
                    "filename": fname,
                    "category": media_info.get("category", "unknown"),
                    "origin": media_info.get("origin", fallback_origin),
                    "origin_name": media_info.get("origin_name", fname),
                }
                for field in _extra_fields.get(media_type, []):
                    media_data[field] = media_info.get(field)
                cm = media_info.get("custom_metadata")
                if cm:
                    media_data["custom_metadata"] = cm

                medias[media_id] = media_data
                loaded_count += 1
                if on_progress is not None and loaded_count % _progress_interval == 0:
                    on_progress(
                        "loading", f"Processing {loaded_count} of {total_count} items…", loaded_count, total_count
                    )
                continue

            # ── Full mode (original behaviour) ──
            # Load the actual media content.
            # Support both new key names (media_bytes/media_string) and legacy
            # key names (clip_bytes/clip_string, wav_bytes/video_bytes/image_bytes/
            # text_content) for backward compatibility with old pickles.
            media_bytes = None
            media_string = None
            media_path = None

            # Try media_bytes first (binary media), then legacy keys via registry
            bytes_val = media_info.get("media_bytes") or media_info.get("clip_bytes")
            if bytes_val is None:
                for legacy_key in _legacy_bytes.get(media_type, []):
                    val = media_info.get(legacy_key)
                    if isinstance(val, bytes):
                        bytes_val = val
                        break

            # Try media_string (text media), then legacy keys
            string_val = media_info.get("media_string") or media_info.get("clip_string")
            if string_val is None:
                for legacy_key in _legacy_bytes.get(media_type, []):
                    val = media_info.get(legacy_key)
                    if isinstance(val, str):
                        string_val = val
                        break

            if bytes_val is not None:
                media_bytes = bytes_val
            elif string_val is not None:
                media_string = string_val
                media_bytes = string_val.encode("utf-8")
            else:
                # Try loading from the external directory via registry dir_key
                dir_key = _dir_keys.get(media_type)
                if dir_key and dir_key in data and "filename" in media_info:
                    ext_path = Path(data[dir_key]) / media_info["filename"]
                    if ext_path.exists():
                        if media_string is None and ext_path.suffix in (".txt", ".md"):
                            with open(ext_path, "r", encoding="utf-8") as f:
                                media_string = f.read()
                                media_bytes = media_string.encode("utf-8")
                        else:
                            with open(ext_path, "rb") as f:
                                media_bytes = f.read()
                        media_path = str(ext_path.resolve())
                    else:
                        missing_media += 1

            if media_bytes is not None:
                fname = media_info.get("filename", f"media_{media_id}.{media_type}")
                media_data = {
                    "id": media_id,
                    "type": media_type,
                    "embedder": media_info.get("embedder", ""),
                    "duration": media_info.get("duration", 0),
                    "file_size": media_info.get("file_size", len(media_bytes)),
                    "md5": media_info.get("md5") or hashlib.md5(media_bytes).hexdigest(),
                    "embedding": np.array(media_info["embedding"]),
                    "media_bytes": media_bytes,
                    "media_string": media_string,
                    "media_path": media_path or media_info.get("media_path"),
                    "filename": fname,
                    "category": media_info.get("category", "unknown"),
                    "origin": media_info.get("origin", fallback_origin),
                    "origin_name": media_info.get("origin_name", fname),
                }
                for field in _extra_fields.get(media_type, []):
                    media_data[field] = media_info.get(field)
                cm = media_info.get("custom_metadata")
                if cm:
                    media_data["custom_metadata"] = cm

                medias[media_id] = media_data
                loaded_count += 1
                if on_progress is not None and loaded_count % _progress_interval == 0:
                    on_progress(
                        "loading", f"Processing {loaded_count} of {total_count} items…", loaded_count, total_count
                    )
    except MemoryError:
        medias.clear()
        del data
        gc.collect()
        raise MemoryError(
            f"Out of memory after loading {loaded_count} of {total_count} medias from "
            f"{file_path.name}. Try a smaller dataset or free up system RAM."
        )

    # Release the raw pickle data now that medias are built
    del data  # noqa: F821 — ruff cannot see past `del data` in the except branch (which always re-raises)
    gc.collect()

    if missing_media > 0:
        print(f"WARNING: {missing_media} media files missing from {file_path}", flush=True)

    return None


def load_dataset_from_pickle_chunked(
    file_path: Path,
    chunk_size: int,
    thin: bool = False,
) -> Iterator[dict[int, dict[str, Any]]]:
    """Yield chunks of medias from a pickle dataset file.

    Works identically to :func:`load_dataset_from_pickle` but yields the
    medias in groups of at most *chunk_size*.  Each yielded dict is a
    self-contained medias dict with IDs starting at 1.

    The entire pickle is deserialized once (unavoidable for ``.pkl``
    format), but media bytes are dropped or skipped per-chunk so that
    only one chunk's worth of media data is alive at a time.

    Args:
        file_path: Path to a ``.pkl`` dataset file.
        chunk_size: Maximum number of medias per yielded chunk.
        thin: When ``True``, skip loading media bytes into memory.

    Yields:
        A dict mapping int media IDs (starting at 1) to media data dicts.
    """
    with open(file_path, "rb") as f:
        data = safe_pickle_load(f)

    if isinstance(data, dict) and ("medias" in data or "clips" in data):
        medias_data = data["medias"] if "medias" in data else data["clips"]
        creation_info = data.get("creation_info")
    else:
        medias_data = data
        creation_info = None

    fallback_origin = None
    if creation_info:
        fallback_origin = {
            "importer": creation_info.get("importer", "unknown"),
            "params": creation_info.get("field_values", {}),
        }

    # Build lookup tables dynamically from the media type registry,
    # matching the approach used by load_dataset_from_pickle.
    from vtsearch.media import all_types

    _dir_keys: dict[str, str] = {}
    _legacy_bytes: dict[str, list[str]] = {}
    _extra_fields: dict[str, list[str]] = {}
    for mt in all_types():
        _dir_keys[mt.type_id] = mt.dir_key
        _legacy_bytes[mt.type_id] = mt.legacy_bytes_keys
        _extra_fields[mt.type_id] = mt.pickle_extra_fields

    all_media_ids = sorted(medias_data.keys())

    for start in range(0, len(all_media_ids), chunk_size):
        batch_ids = all_media_ids[start : start + chunk_size]
        chunk_medias: dict[int, dict[str, Any]] = {}
        new_id = 1

        for media_id in batch_ids:
            media_info = medias_data[media_id]
            media_type = media_info.get("type", "audio")

            if thin:
                media_path: str | None = media_info.get("media_path")
                if not media_path:
                    dir_key = _dir_keys.get(media_type)
                    if dir_key and dir_key in data and "filename" in media_info:
                        candidate = Path(data[dir_key]) / media_info["filename"]
                        if candidate.exists():
                            media_path = str(candidate.resolve())

                if "embedding" not in media_info:
                    continue

                fname = media_info.get("filename", f"media_{media_id}.{media_type}")
                media_data: dict[str, Any] = {
                    "id": new_id,
                    "type": media_type,
                    "duration": media_info.get("duration", 0),
                    "file_size": media_info.get("file_size", 0),
                    "md5": media_info.get("md5", ""),
                    "embedding": np.array(media_info["embedding"]),
                    "media_bytes": None,
                    "media_string": None,
                    "media_path": media_path,
                    "filename": fname,
                    "category": media_info.get("category", "unknown"),
                    "origin": media_info.get("origin", fallback_origin),
                    "origin_name": media_info.get("origin_name", fname),
                }
                for field in _extra_fields.get(media_type, []):
                    media_data[field] = media_info.get(field)
                cm = media_info.get("custom_metadata")
                if cm:
                    media_data["custom_metadata"] = cm

                chunk_medias[new_id] = media_data
                new_id += 1
                continue

            # Full mode — same logic as load_dataset_from_pickle
            media_bytes = None
            media_string = None
            media_path = None

            # Try media_bytes first (binary media), then legacy keys via registry
            bytes_val = media_info.get("media_bytes") or media_info.get("clip_bytes")
            if bytes_val is None:
                for legacy_key in _legacy_bytes.get(media_type, []):
                    val = media_info.get(legacy_key)
                    if isinstance(val, bytes):
                        bytes_val = val
                        break

            # Try media_string (text media), then legacy keys
            string_val = media_info.get("media_string") or media_info.get("clip_string")
            if string_val is None:
                for legacy_key in _legacy_bytes.get(media_type, []):
                    val = media_info.get(legacy_key)
                    if isinstance(val, str):
                        string_val = val
                        break

            if bytes_val is not None:
                media_bytes = bytes_val
            elif string_val is not None:
                media_string = string_val
                media_bytes = string_val.encode("utf-8")
            else:
                # Try loading from the external directory via registry dir_key
                dir_key = _dir_keys.get(media_type)
                if dir_key and dir_key in data and "filename" in media_info:
                    ext_path = Path(data[dir_key]) / media_info["filename"]
                    if ext_path.exists():
                        if media_string is None and ext_path.suffix in (".txt", ".md"):
                            with open(ext_path, "r", encoding="utf-8") as f:
                                media_string = f.read()
                                media_bytes = media_string.encode("utf-8")
                        else:
                            with open(ext_path, "rb") as f:
                                media_bytes = f.read()
                        media_path = str(ext_path.resolve())

            if media_bytes is not None:
                fname = media_info.get("filename", f"media_{media_id}.{media_type}")
                media_data = {
                    "id": new_id,
                    "type": media_type,
                    "duration": media_info.get("duration", 0),
                    "file_size": media_info.get("file_size", len(media_bytes)),
                    "md5": media_info.get("md5") or hashlib.md5(media_bytes).hexdigest(),
                    "embedding": np.array(media_info["embedding"]),
                    "media_bytes": media_bytes,
                    "media_string": media_string,
                    "media_path": media_path or media_info.get("media_path"),
                    "filename": fname,
                    "category": media_info.get("category", "unknown"),
                    "origin": media_info.get("origin", fallback_origin),
                    "origin_name": media_info.get("origin_name", fname),
                }
                for field in _extra_fields.get(media_type, []):
                    media_data[field] = media_info.get(field)
                cm = media_info.get("custom_metadata")
                if cm:
                    media_data["custom_metadata"] = cm

                chunk_medias[new_id] = media_data
                new_id += 1

        if chunk_medias:
            yield chunk_medias


def embed_image_file_from_pil(image: Image.Image, embedder_name: str = "") -> Optional[np.ndarray]:
    """Generate a CLIP embedding vector for a PIL Image object.

    A convenience wrapper for cases where the image is already in memory
    (e.g. reconstructed from a NumPy array during CIFAR-10 loading).

    Delegates to the image embedder's ``embed_pil_image`` method.

    Args:
        image: A PIL Image in any mode.
        embedder_name: Optional name of a registered embedder.  When empty,
            the first image embedder is used.

    Returns:
        A 1-D ``numpy.ndarray`` of shape ``(embedding_dim,)``, or ``None`` if
        no image embedder is available or an exception occurs.
    """
    from vtsearch.media import embedders_for_type, get_embedder

    if embedder_name:
        emb = get_embedder(embedder_name)
    else:
        avail = embedders_for_type("image")
        if not avail:
            return None
        emb = avail[0]
    return emb.embed_pil_image(image)


def _write_embedder_sidecar(pkl_path: Path, embedder_name: str) -> None:
    """Write a small ``<name>.embedder`` file next to *pkl_path*.

    The file stores the embedder name used to produce the pickle so that
    ``demo_dataset_list`` can cheaply check whether the cached embeddings
    match the user's current embedder selection without loading the full pkl.
    """
    sidecar = pkl_path.with_suffix(".embedder")
    sidecar.write_text(embedder_name, encoding="utf-8")


def _write_clipper_sidecar(pkl_path: Path, clipper_name: str) -> None:
    """Write a small ``<name>.clipper`` file next to *pkl_path*."""
    sidecar = pkl_path.with_suffix(".clipper")
    sidecar.write_text(clipper_name, encoding="utf-8")


def read_pkl_clipper(pkl_path: Path) -> str | None:
    """Return the clipper name stored for *pkl_path*, or ``None`` if unknown.

    Checks the lightweight ``.clipper`` sidecar.  Returns ``None`` if the
    sidecar does not exist (legacy pickles created before clipper tracking).
    """
    sidecar = pkl_path.with_suffix(".clipper")
    if sidecar.exists():
        return sidecar.read_text(encoding="utf-8").strip()
    return None


def read_pkl_embedder(pkl_path: Path) -> str | None:
    """Return the embedder name stored for *pkl_path*, or ``None`` if unknown.

    Checks the lightweight ``.embedder`` sidecar first.  Falls back to loading
    the pickle and inspecting the first media entry's ``"embedder"`` field,
    then writes the sidecar for future fast lookups.
    """
    sidecar = pkl_path.with_suffix(".embedder")
    if sidecar.exists():
        return sidecar.read_text(encoding="utf-8").strip()

    # Fallback: peek into the pkl itself.
    if not pkl_path.exists():
        return None
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)  # noqa: S301
        medias_dict = data.get("medias", {}) if isinstance(data, dict) else {}
        for media in medias_dict.values():
            name = media.get("embedder", "")
            if name:
                # Cache it for next time.
                _write_embedder_sidecar(pkl_path, name)
                return name
    except Exception:
        pass
    return None


def _stamp_demo_origin(
    medias: dict[int, dict[str, Any]],
    dataset_name: str,
    converter_name: str = "",
) -> None:
    """Stamp the demo origin on all medias (fresh dict per media).

    Ensures every media has ``origin = {"importer": "demo", "params": {"name": ...}}``.
    Called both for freshly-embedded medias and for pickle-cached loads so
    that old pickles (created before origin stamping was added) get corrected.
    """
    demo_origin_params: dict[str, str] = {"name": dataset_name}
    if converter_name:
        demo_origin_params["converter"] = converter_name
    for media in medias.values():
        media["origin"] = {"importer": "demo", "params": dict(demo_origin_params)}


def load_demo_dataset(
    dataset_name: str,
    medias: dict[int, dict[str, Any]],
    e5_model: Any = None,
    on_progress: Optional[ProgressCallback] = None,
    embedder_name: str = "",
    converter_name: str = "",
    clipper_name: str = "",
) -> None:
    """Load a named demo dataset into the medias dict, downloading and embedding as needed.

    Checks for a cached ``.pkl`` file in ``EMBEDDINGS_DIR``; if found, loads
    from that file. If the cache is missing or the media bytes it references can
    no longer be found on disk, the raw data is re-downloaded and re-embedded.

    Each media type implements its own
    :meth:`~vtsearch.media.base.MediaType.load_demo_source` method that
    handles downloading, embedding, and populating clips for its demo sources.
    This function simply orchestrates pickle caching around that delegation.

    When *converter_name* is given (e.g. ``"video2image"``), the demo data is
    loaded using its original media type, then each media is converted via the
    named converter.  The resulting dataset contains the *target* type and
    is cached under a separate pickle key.

    Progress throughout the operation is reported via :func:`update_progress`.

    Args:
        dataset_name: Key into ``DEMO_DATASETS`` identifying which demo dataset
            to load.  Raises ``ValueError`` if the key is not found.
        medias: Dict to populate in-place. Existing entries are removed before
            loading. Keys are integer media IDs; values are media data dicts.
        e5_model: Deprecated — kept for backward compatibility but no longer
            used.  The text embedding model is obtained from the media type
            registry.
        embedder_name: Optional name of a registered embedder to use.
            When empty, the first registered embedder for the media type
            is used.
        converter_name: Optional name of a converter (e.g. ``"video2image"``).
            When given, the demo is loaded in its native type and then
            converted.
        clipper_name: Optional name of a registered clipper.  Recorded in
            a ``.clipper`` sidecar next to the pickle for status tracking.

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

    # When a converter is specified, use a separate pickle cache key.
    cache_key = f"{dataset_name}__{converter_name}" if converter_name else dataset_name

    # Check if already embedded
    pkl_file = EMBEDDINGS_DIR / f"{cache_key}.pkl"
    if pkl_file.exists():
        on_progress("loading", f"Loading {dataset_name} dataset...", 0, 0)
        load_dataset_from_pickle(pkl_file, medias)

        # Check if any medias were actually loaded
        if len(medias) == 0:
            # Pickle file exists but media files are missing, delete and re-embed
            on_progress("loading", f"Media files missing, re-embedding {dataset_name}...", 0, 0)
            pkl_file.unlink()
            pkl_file.with_suffix(".embedder").unlink(missing_ok=True)
            pkl_file.with_suffix(".clipper").unlink(missing_ok=True)
        else:
            # Stamp demo origin on cached medias so that cross-dataset
            # resolution always has the dataset name in the origin params.
            # Old pickles (created before origin stamping) may have empty
            # params — this ensures they are corrected on load.
            _stamp_demo_origin(medias, dataset_name, converter_name)
            on_progress("idle", f"Loaded {dataset_name} dataset")
            return

    # Resolve the embedder
    from vtsearch.media import embedders_for_type, get as media_get, get_embedder

    embedder = None
    if embedder_name:
        try:
            embedder = get_embedder(embedder_name)
        except KeyError:
            raise ValueError(f"Unknown embedder: {embedder_name}")
    else:
        avail = embedders_for_type(media_type_id)
        if avail:
            embedder = avail[0]

    mt = media_get(media_type_id)

    source = dataset_info.get("source", "")
    categories = dataset_info["categories"]
    slice_start = dataset_info.get("slice_start", 0)
    slice_end = dataset_info.get("slice_end")

    medias.clear()
    external_dir = mt.load_demo_source(
        source=source,
        categories=categories,
        slice_start=slice_start,
        slice_end=slice_end,
        clips=medias,
        on_progress=on_progress,
        embedder=embedder,
    )

    # Stamp the demo origin on all medias
    _stamp_demo_origin(medias, dataset_name, converter_name)

    # --- Apply converter if requested ---
    if converter_name:
        _apply_converter_to_demo(
            converter_name=converter_name,
            dataset_name=dataset_name,
            medias=medias,
            embedder_name=embedder_name,
            on_progress=on_progress,
        )

    # Build the pickle cache payload
    # For types with external media dirs (audio, video), exclude media_bytes
    # from the pickle and store the dir path so reloading can find the files.
    # When a converter was applied, external_dir is no longer relevant (the
    # converted medias carry their own bytes/strings).
    if external_dir is not None and not converter_name:
        pkl_data: dict[str, Any] = {
            "name": dataset_name,
            "medias": {
                cid: {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in media.items() if k != "media_bytes"}
                for cid, media in medias.items()
            },
            mt.dir_key: external_dir,
        }
    else:
        pkl_data = {
            "name": dataset_name,
            "medias": {
                cid: {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in media.items()}
                for cid, media in medias.items()
            },
        }

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(pkl_file, "wb") as f:
        pickle.dump(pkl_data, f)

    # Write a lightweight sidecar that records which embedder produced this pkl.
    resolved_name = getattr(embedder, "name", "") if embedder is not None else ""
    _write_embedder_sidecar(pkl_file, resolved_name)

    # Write a clipper sidecar so the demo list can check readiness.
    _write_clipper_sidecar(pkl_file, clipper_name)

    on_progress("idle", f"Loaded {dataset_name} dataset")


def _apply_converter_to_demo(
    converter_name: str,
    dataset_name: str,
    medias: dict[int, dict[str, Any]],
    embedder_name: str = "",
    on_progress: Optional[ProgressCallback] = None,
) -> None:
    """Convert all medias in-place using the named converter.

    After conversion, *medias* contains the converted outputs (target type)
    instead of the original source-type medias.  Each converted media's
    origin records the demo dataset and the converter used.
    """
    from vtsearch.converters import get_converter
    from vtsearch.converters.runner import _compute_md5, _embed_converted_output
    from vtsearch.media import embedders_for_type, get_embedder

    converter = get_converter(converter_name)
    if converter is None:
        raise ValueError(f"Unknown converter: {converter_name}")

    if on_progress is None:
        on_progress = _default_progress()

    # Resolve embedder for the *target* type.
    target_emb = None
    if embedder_name:
        try:
            target_emb = get_embedder(embedder_name)
        except KeyError:
            pass
    if target_emb is None:
        avail = embedders_for_type(converter.target_type)
        if avail:
            target_emb = avail[0]
    if target_emb is not None and getattr(target_emb, "_model", None) is None:
        target_emb.load_models()

    # Convert each original media and collect the outputs.
    source_items = list(medias.items())
    converted: dict[int, dict[str, Any]] = {}
    new_id = 1

    on_progress(
        "converting",
        f"Converting {len(source_items)} items via {converter.display_name or converter.name}...",
        0,
        len(source_items),
    )

    for idx, (_, src) in enumerate(source_items):
        on_progress(
            "converting",
            f"Converting {src.get('filename', '')} via {converter.display_name or converter.name}...",
            idx + 1,
            len(source_items),
        )

        try:
            outputs = converter.convert(src)
        except Exception:
            continue
        if not outputs:
            continue

        origin = {
            "importer": "converter",
            "params": {
                "converter": converter_name,
                "source_file": src.get("filename", ""),
                "parent_importer": "demo",
                "parent_demo": dataset_name,
            },
        }

        for output in outputs:
            output_filename = output.get("filename", f"converted_{new_id}")
            source_name = src.get("filename", str(src.get("id", "")))
            origin_name = f"{source_name}\u2192{output_filename}"

            embedding = _embed_converted_output(target_emb, output)
            if embedding is None:
                continue

            md5 = _compute_md5(output)

            media_data: dict[str, Any] = {
                "id": new_id,
                "type": converter.target_type,
                "embedder": target_emb.name if target_emb else "",
                "file_size": len(output.get("media_bytes", b"") or output.get("media_string", "").encode()),
                "md5": md5,
                "embedding": embedding,
                "filename": origin_name,
                "category": src.get("category", "custom"),
                "origin": origin,
                "origin_name": origin_name,
                "media_bytes": None,
                "media_string": None,
                "media_path": src.get("media_path", ""),
                "duration": output.get("duration", 0),
            }

            if "media_bytes" in output:
                media_data["media_bytes"] = output["media_bytes"]
            if "media_string" in output:
                media_data["media_string"] = output["media_string"]
            if "width" in output:
                media_data["width"] = output["width"]
            if "height" in output:
                media_data["height"] = output["height"]
            if "word_count" in output:
                media_data["word_count"] = output["word_count"]
            if "character_count" in output:
                media_data["character_count"] = output["character_count"]

            converted[new_id] = media_data
            new_id += 1

    medias.clear()
    medias.update(converted)


def export_dataset_to_file(
    medias: dict[int, dict[str, Any]],
) -> bytes:
    """Serialise the current media dataset to a pickle-formatted byte string.

    Converts the in-memory ``medias`` dict to a portable format (converting any
    ``numpy.ndarray`` embeddings to plain Python lists) and returns it as bytes
    suitable for writing to a ``.pkl`` file or sending as an HTTP response.

    The resulting bytes can be reloaded with :func:`load_dataset_from_pickle`.

    Args:
        medias: Mapping of media ID to media data dict.

    Returns:
        Raw bytes of the pickled dataset dict.
    """
    data: dict[str, Any] = {
        "medias": {
            cid: {
                "id": media["id"],
                "type": media.get("type", "audio"),
                "duration": media["duration"],
                "file_size": media["file_size"],
                "md5": media["md5"],
                "embedder": media.get("embedder", ""),
                "embedding": media["embedding"].tolist()
                if isinstance(media["embedding"], np.ndarray)
                else media["embedding"],
                "filename": media.get("filename", f"media_{cid}.wav"),
                "category": media.get("category", "unknown"),
                "origin": media.get("origin"),
                "origin_name": media.get("origin_name", media.get("filename", "")),
                "media_bytes": media.get("media_bytes"),
                "media_string": media.get("media_string"),
                "media_path": media.get("media_path"),
                "word_count": media.get("word_count"),
                "character_count": media.get("character_count"),
                "width": media.get("width"),
                "height": media.get("height"),
                "custom_metadata": media.get("custom_metadata"),
            }
            for cid, media in medias.items()
        }
    }

    buf = io.BytesIO()
    pickle.dump(data, buf)
    buf.seek(0)
    return buf.getvalue()
