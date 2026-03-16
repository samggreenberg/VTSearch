"""Image media type — JPEG/PNG/GIF/BMP/WEBP files."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from vtsearch.media.base import (
    MediaResponse,
    MediaType,
    ProgressCallback,
    _noop_progress,
    intercept_tqdm_progress,
)


_IMAGE_MIME_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


class ImageMediaType(MediaType):
    """Handles image medias — file import, HTTP serving, and demo datasets.

    Embedding is handled by :class:`~vtsearch.media.image.embedder.ImageClipEmbedder`.
    """

    def __init__(self) -> None:
        self._on_progress: ProgressCallback = _noop_progress

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def type_id(self) -> str:
        return "image"

    @property
    def name(self) -> str:
        return "Image"

    @property
    def icon(self) -> str:
        return "🖼️"

    # ------------------------------------------------------------------
    # File import
    # ------------------------------------------------------------------

    @property
    def file_extensions(self) -> list:
        return ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.webp"]

    @property
    def folder_import_name(self) -> str:
        return "images"

    @property
    def tab_title(self) -> str:
        return "Images"

    @property
    def dir_key(self) -> str:
        return "image_dir"

    @property
    def legacy_bytes_keys(self) -> list[str]:
        return ["image_bytes"]

    @property
    def pickle_extra_fields(self) -> list[str]:
        return ["width", "height"]

    # ------------------------------------------------------------------
    # Display metadata
    # ------------------------------------------------------------------

    def display_metadata(self, media: dict) -> dict:
        result: dict = {}
        cat = media.get("category")
        if cat and cat not in ("unknown", "custom"):
            result["Category"] = cat
        w, h = media.get("width"), media.get("height")
        if w and h:
            result["Dimensions"] = f"{w}\u00d7{h}"
        fs = media.get("file_size")
        if fs:
            result["File Size"] = fs
        return result

    # ------------------------------------------------------------------
    # Viewer
    # ------------------------------------------------------------------

    @property
    def loops(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Demo dataset loading
    # ------------------------------------------------------------------

    def load_demo_source(self, source, categories, slice_start, slice_end, clips, on_progress=None, embedder=None):
        import hashlib  # noqa: PLC0415
        import io as _io  # noqa: PLC0415

        from PIL import Image  # noqa: PLC0415

        if on_progress is None:
            from vtsearch.utils import update_progress

            on_progress = update_progress

        if embedder is None:
            from vtsearch.media import embedders_for_type

            avail = embedders_for_type(self.type_id)
            if not avail:
                raise ValueError(f"No embedders registered for media type {self.type_id!r}")
            embedder = avail[0]

        from vtsearch.datasets.loader import load_image_metadata_from_folders  # noqa: PLC0415

        demo_origin: dict = {"importer": "demo", "params": {}}

        def _embed_file_images(selected):
            """Embed a list of (img_path, category) tuples."""
            if getattr(embedder, "_model", None) is None:
                on_progress("loading", "Loading image embedding model…", 0, 0)
                with intercept_tqdm_progress(on_progress):
                    embedder.load_models()

            clip_id = 1
            total = len(selected)
            on_progress("embedding", f"Starting embedding for {total} images...", 0, total)

            for i, (img_path, category) in enumerate(selected):
                on_progress("embedding", f"Embedding {category}/{img_path.name} ({i + 1}/{total})", i + 1, total)
                embedding = embedder.embed_media(img_path)
                if embedding is None:
                    continue
                with open(img_path, "rb") as f:
                    image_bytes = f.read()
                try:
                    img = Image.open(img_path)
                    width, height = img.width, img.height
                except Exception:
                    width, height = None, None
                clips[clip_id] = {
                    "id": clip_id,
                    "type": self.type_id,
                    "embedder": embedder.name,
                    "duration": 0,
                    "file_size": len(image_bytes),
                    "md5": hashlib.md5(image_bytes).hexdigest(),
                    "embedding": embedding,
                    "media_bytes": image_bytes,
                    "media_string": None,
                    "filename": f"{category}/{img_path.name}",
                    "category": category,
                    "width": width,
                    "height": height,
                    "origin": demo_origin,
                    "origin_name": f"{category}/{img_path.name}",
                }
                clip_id += 1

        if source in ("caltech101", "caltech256"):
            if source == "caltech101":
                from vtsearch.datasets.downloader import download_caltech101  # noqa: PLC0415

                img_dir = download_caltech101(on_progress=on_progress)
            else:
                from vtsearch.datasets.downloader import download_caltech256  # noqa: PLC0415

                img_dir = download_caltech256(on_progress=on_progress)

            metadata = load_image_metadata_from_folders(img_dir, categories)
            by_cat: dict[str, list[tuple[Path, str]]] = {}
            for fname, meta in sorted(metadata.items()):
                cat = meta["category"]
                by_cat.setdefault(cat, []).append((meta["path"], cat))

            selected: list[tuple[Path, str]] = []
            for cat in categories:
                selected.extend(by_cat.get(cat, [])[slice_start:slice_end])

            _embed_file_images(selected)
            return None

        elif source == "oxford_flowers_102":
            from vtsearch.datasets.downloader import download_oxford_flowers  # noqa: PLC0415
            from vtsearch.datasets.loader import load_oxford_flowers_metadata  # noqa: PLC0415

            flowers_dir = download_oxford_flowers(on_progress=on_progress)
            from vtsearch.datasets.importers.demo.datasets import OXFORD_FLOWERS_CATEGORIES  # noqa: PLC0415

            metadata = load_oxford_flowers_metadata(flowers_dir, OXFORD_FLOWERS_CATEGORIES)

            by_cat: dict[str, list[tuple[Path, str]]] = {}
            for _fname, meta in sorted(metadata.items()):
                cat = meta["category"]
                if cat in categories:
                    by_cat.setdefault(cat, []).append((meta["path"], cat))

            selected: list[tuple[Path, str]] = []
            for cat in categories:
                selected.extend(by_cat.get(cat, [])[slice_start:slice_end])

            _embed_file_images(selected)
            return None

        elif source in ("food101", "eurosat"):
            if source == "food101":
                from vtsearch.datasets.downloader import download_food101  # noqa: PLC0415

                img_dir = download_food101(on_progress=on_progress)
            else:
                from vtsearch.datasets.downloader import download_eurosat  # noqa: PLC0415

                img_dir = download_eurosat(on_progress=on_progress)

            metadata = load_image_metadata_from_folders(img_dir, categories)
            by_cat = {}
            for _fname, meta in sorted(metadata.items()):
                cat = meta["category"]
                by_cat.setdefault(cat, []).append((meta["path"], cat))

            selected = []
            for cat in categories:
                selected.extend(by_cat.get(cat, [])[slice_start:slice_end])

            _embed_file_images(selected)
            return None

        elif source == "stanford_dogs":
            from vtsearch.datasets.downloader import download_stanford_dogs  # noqa: PLC0415

            images_dir = download_stanford_dogs(on_progress=on_progress)

            breed_to_folder: dict[str, Path] = {}
            if images_dir.exists():
                for folder in images_dir.iterdir():
                    if folder.is_dir() and "-" in folder.name:
                        breed_name = folder.name.split("-", 1)[1]
                        breed_to_folder[breed_name] = folder

            by_cat = {}
            for cat in categories:
                folder = breed_to_folder.get(cat)
                if folder is None:
                    continue
                for ext in ["*.jpg", "*.jpeg", "*.png"]:
                    for img_path in sorted(folder.glob(ext)):
                        by_cat.setdefault(cat, []).append((img_path, cat))

            selected = []
            for cat in categories:
                selected.extend(by_cat.get(cat, [])[slice_start:slice_end])

            _embed_file_images(selected)
            return None

        elif source == "ucsf_documents":
            from vtsearch.datasets.downloader import download_ucsf_documents  # noqa: PLC0415
            from vtsearch.datasets.pdf import render_pdf_pages  # noqa: PLC0415

            docs_dir = download_ucsf_documents(categories, on_progress=on_progress)

            by_cat_pages: dict[str, list[tuple[str, "Image.Image"]]] = {}
            for cat in categories:
                cat_dir = docs_dir / cat
                if not cat_dir.is_dir():
                    continue
                for pdf_path in sorted(cat_dir.glob("*.pdf")):
                    try:
                        pages = render_pdf_pages(pdf_path, dpi=150)
                        if pages:
                            by_cat_pages.setdefault(cat, []).append(pages[0])
                    except Exception:
                        continue

            selected_pages: list[tuple[str, "Image.Image", str]] = []
            for cat in categories:
                for page_name, pil_image in by_cat_pages.get(cat, [])[slice_start:slice_end]:
                    selected_pages.append((page_name, pil_image, cat))

            if getattr(embedder, "_model", None) is None:
                on_progress("loading", "Loading image embedding model…", 0, 0)
                with intercept_tqdm_progress(on_progress):
                    embedder.load_models()

            clip_id = 1
            total = len(selected_pages)
            on_progress("embedding", f"Starting embedding for {total} document pages...", 0, total)

            for i, (page_name, pil_image, category) in enumerate(selected_pages):
                on_progress("embedding", f"Embedding {page_name} ({i + 1}/{total})", i + 1, total)
                embedding = embedder.embed_pil_image(pil_image)
                if embedding is None:
                    continue
                img_buffer = _io.BytesIO()
                pil_image.save(img_buffer, format="PNG")
                image_bytes = img_buffer.getvalue()
                rel_name = f"{category}/{page_name}"
                clips[clip_id] = {
                    "id": clip_id,
                    "type": self.type_id,
                    "embedder": embedder.name,
                    "duration": 0,
                    "file_size": len(image_bytes),
                    "md5": hashlib.md5(image_bytes).hexdigest(),
                    "embedding": embedding,
                    "media_bytes": image_bytes,
                    "media_string": None,
                    "filename": f"{rel_name}.png",
                    "category": category,
                    "width": pil_image.width,
                    "height": pil_image.height,
                    "origin": demo_origin,
                    "origin_name": rel_name,
                }
                clip_id += 1
            return None

        elif source == "cifar10_sample" or not source:
            from vtsearch.datasets.downloader import download_cifar10  # noqa: PLC0415
            from vtsearch.datasets.loader import load_cifar10_batch  # noqa: PLC0415

            cifar_dir = download_cifar10(on_progress=on_progress)
            batch_file = cifar_dir / "data_batch_1"
            images, labels, label_names = load_cifar10_batch(batch_file)
            category_indices = {label_names[i]: i for i in range(len(label_names))}

            selected_images = []
            selected_labels = []
            for cat in categories:
                if cat in category_indices:
                    cat_idx = category_indices[cat]
                    cat_mask = [i for i, lbl in enumerate(labels) if lbl == cat_idx]
                    for idx in cat_mask[slice_start:(slice_end or len(cat_mask))]:
                        selected_images.append(images[idx])
                        selected_labels.append(cat)

            if getattr(embedder, "_model", None) is None:
                on_progress("loading", "Loading image embedding model…", 0, 0)
                embedder.load_models()

            clip_id = 1
            total = len(selected_images)
            on_progress("embedding", f"Starting embedding for {total} images...", 0, total)

            for i, (image_array, category) in enumerate(zip(selected_images, selected_labels)):
                on_progress("embedding", f"Embedding {category}: image {i + 1}/{total}", i + 1, total)
                img = Image.fromarray(image_array.astype("uint8"), "RGB")
                img_buffer = _io.BytesIO()
                img.save(img_buffer, format="PNG")
                image_bytes = img_buffer.getvalue()
                embedding = embedder.embed_pil_image(img)
                if embedding is None:
                    continue
                fname = f"{category}/{category}_{clip_id}.png"
                clips[clip_id] = {
                    "id": clip_id,
                    "type": self.type_id,
                    "embedder": embedder.name,
                    "duration": 0,
                    "file_size": len(image_bytes),
                    "md5": hashlib.md5(image_bytes).hexdigest(),
                    "embedding": embedding,
                    "media_bytes": image_bytes,
                    "media_string": None,
                    "filename": fname,
                    "category": category,
                    "width": img.width,
                    "height": img.height,
                    "origin": demo_origin,
                    "origin_name": fname,
                }
                clip_id += 1
            return None

        else:
            raise ValueError(f"Unsupported image source: {source!r}")

    # ------------------------------------------------------------------
    # Clip data
    # ------------------------------------------------------------------

    def load_media_data(self, file_path: Path) -> dict:
        from PIL import Image  # noqa: PLC0415

        with open(file_path, "rb") as f:
            media_bytes = f.read()
        try:
            img = Image.open(file_path)
            width, height = img.width, img.height
        except Exception:
            width, height = None, None
        return {
            "media_bytes": media_bytes,
            "duration": 0,
            "width": width,
            "height": height,
        }

    # ------------------------------------------------------------------
    # HTTP serving
    # ------------------------------------------------------------------

    def media_response(self, media: dict) -> MediaResponse:
        filename = media.get("filename", "")
        ext = Path(filename).suffix.lower() if filename else ".jpg"
        mimetype = _IMAGE_MIME_TYPES.get(ext, "image/jpeg")
        data = self._resolve_media_bytes(media)
        if data is None:
            return MediaResponse(data=b"", mimetype=mimetype, download_name=f"media_{media['id']}{ext}")
        return MediaResponse(
            data=data,
            mimetype=mimetype,
            download_name=f"media_{media['id']}{ext}",
        )
