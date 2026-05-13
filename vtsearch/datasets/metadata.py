"""Dataset metadata extraction functions.

Functions for loading media metadata from CSV files, MAT files, CIFAR-10
pickle batches, and folder-based category structures.  Each function returns
a dict mapping a key (typically filename or ``category/filename``) to a
metadata dict.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from vtsearch.datasets.pickle_security import safe_pickle_load


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


def load_places365_metadata(places_dir: Path, categories: list[str]) -> dict[str, dict[str, Any]]:
    """Load metadata for the Places365 validation set.

    Reads the ``places365_val.txt`` label file and maps each image filename
    to a scene category name via the integer index into ``categories``.
    Images live in a flat ``val_256/`` directory with names like
    ``Places365_val_00000001.jpg``.

    Args:
        places_dir: Path to the root Places365 directory (contains
            ``val_256/`` and ``places365_val.txt``).
        categories: Ordered list of 365 scene category names (index 0
            corresponds to category index 0 in the label file).

    Returns:
        A dict mapping image filename to a dict with the keys:

        - ``"category"`` (``str``): Scene category name from *categories*.
        - ``"path"`` (``Path``): Full path to the image file.
    """
    labels_path = places_dir / "places365_val.txt"
    images_dir = places_dir / "val_256"
    metadata: dict[str, dict[str, Any]] = {}

    with open(labels_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fname, _, idx_str = line.rpartition(" ")
            if not fname:
                continue
            try:
                idx = int(idx_str)
            except ValueError:
                continue
            if idx < 0 or idx >= len(categories):
                continue
            metadata[fname] = {
                "category": categories[idx],
                "path": images_dir / fname,
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
