"""Synthetic dataset importer — generates fake media offline.

Useful for trying out VTSearch without an internet connection. Pick a
``media_type`` and a ``size`` and the importer renders that many synthetic
files (smiley faces / shapes for images; tones / chords / drums / rain /
wind / bird chirps for audio; bouncing balls / walking smileys / rotating
shapes / marquees for video). Files are cached to ``data/synthetic/...``
so subsequent reloads skip regeneration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vtsearch.config import DATA_DIR
from vtsearch.datasets.importers.base import DatasetImporter, ImporterField
from vtsearch.datasets.loader import load_dataset_from_folder

_SUPPORTED_MEDIA_TYPES = ["image", "audio", "video"]
_DEFAULT_SEED = 42


def _cache_dir(media_type: str, size: int) -> Path:
    return DATA_DIR / "synthetic" / f"{media_type}_{size}"


class SyntheticDatasetImporter(DatasetImporter):
    """Generate a fake dataset of images, audio, or video — no network needed.

    The user only picks a media type and a size; everything else is
    deterministic (fixed seed) so repeated loads of the same parameters
    reuse cached files.
    """

    name = "synthetic"
    display_name = "Synthetic Media"
    description = "Generate fake media offline — useful for demos and field testing without internet"
    # 🏭 — frontend renders this as a line-drawing factory icon (see
    # frontend/src/app/components/icon/icon.component.ts).
    icon = "\U0001f3ed"
    category = "demo"

    fields = [
        ImporterField(
            key="media_type",
            label="Media Type",
            field_type="select",
            description="What kind of media to generate.",
            options=_SUPPORTED_MEDIA_TYPES,
            default="image",
        ),
        ImporterField(
            key="size",
            label="Size",
            field_type="text",
            description="How many medias to generate (e.g. 100, 1000, 10000).",
            default="100",
            placeholder="100",
        ),
    ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_size(field_values: dict[str, Any]) -> int:
        raw = field_values.get("size", "")
        if isinstance(raw, int):
            n = raw
        else:
            try:
                n = int(str(raw).strip())
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid size: {raw!r}") from exc
        if n <= 0:
            raise ValueError(f"Size must be positive, got {n}")
        return n

    @staticmethod
    def _media_type(field_values: dict[str, Any]) -> str:
        mt = (field_values.get("media_type") or "").strip()
        if mt not in _SUPPORTED_MEDIA_TYPES:
            raise ValueError(f"Unsupported media_type {mt!r} (expected one of {_SUPPORTED_MEDIA_TYPES})")
        return mt

    def _generate(self, media_type: str, size: int) -> Path:
        """Render files into the cache dir, return the dir path."""
        from vtsearch.concurrency.progress import get_thread_progress  # noqa: PLC0415

        on_progress = get_thread_progress()
        out_dir = _cache_dir(media_type, size)
        if media_type == "image":
            from vtsearch.utils.synthetic import generate_image_dataset  # noqa: PLC0415

            generate_image_dataset(out_dir, size, seed=_DEFAULT_SEED, on_progress=on_progress)
        elif media_type == "audio":
            from vtsearch.utils.synthetic import generate_audio_dataset  # noqa: PLC0415

            generate_audio_dataset(out_dir, size, seed=_DEFAULT_SEED, on_progress=on_progress)
        elif media_type == "video":
            from vtsearch.utils.synthetic import generate_video_dataset  # noqa: PLC0415

            generate_video_dataset(out_dir, size, seed=_DEFAULT_SEED, on_progress=on_progress)
        else:
            raise ValueError(f"Unsupported media_type {media_type!r}")
        return out_dir

    # ------------------------------------------------------------------
    # Importer interface
    # ------------------------------------------------------------------

    def run(self, field_values: dict[str, Any], medias: dict, thin: bool = False) -> None:
        media_type = self._media_type(field_values)
        size = self._parse_size(field_values)
        out_dir = self._generate(media_type, size)
        origin = self.build_origin({"media_type": media_type, "size": str(size)})
        load_dataset_from_folder(
            out_dir,
            media_type,
            medias,
            origin=origin,
            thin=thin,
        )

    def run_cli(self, field_values: dict[str, Any], medias: dict, thin: bool = False) -> None:
        self.run(field_values, medias, thin=thin)

    def default_display_name(self, field_values: dict[str, Any]) -> str:
        try:
            mt = self._media_type(field_values)
            n = self._parse_size(field_values)
        except ValueError:
            return self.display_name
        return f"Synthetic {mt} ({n})"

    # ------------------------------------------------------------------
    # Origin handling
    # ------------------------------------------------------------------

    def origin_display(self, origin: dict[str, Any]) -> str:
        params = origin.get("params", {})
        return f"synthetic:{params.get('media_type', '')}_{params.get('size', '')}"

    def can_reload_from_origin(self, origin: dict[str, Any]) -> bool:
        params = origin.get("params", {})
        return params.get("media_type", "") in _SUPPORTED_MEDIA_TYPES and bool(params.get("size", ""))

    def reload_from_origin(self, origin: dict[str, Any]) -> dict[str, Any] | None:
        params = origin.get("params", {})
        mt = params.get("media_type", "")
        size = params.get("size", "")
        if mt not in _SUPPORTED_MEDIA_TYPES or not size:
            return None
        return {"media_type": mt, "size": size}

    def resolve_file(
        self,
        origin: dict[str, Any],
        origin_name: str = "",
        filename: str = "",
    ) -> Path | None:
        params = origin.get("params", {})
        mt = params.get("media_type", "")
        size_str = params.get("size", "")
        if mt not in _SUPPORTED_MEDIA_TYPES or not size_str:
            return None
        try:
            size = int(size_str)
        except (TypeError, ValueError):
            return None
        root = _cache_dir(mt, size)
        for name in (origin_name, filename):
            if not name:
                continue
            candidate = root / name
            if candidate.is_file():
                return candidate
            basename = Path(name).name
            candidate = root / basename
            if candidate.is_file():
                return candidate
        return None


IMPORTER = SyntheticDatasetImporter()
