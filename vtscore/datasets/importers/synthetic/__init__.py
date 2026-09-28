"""Synthetic dataset importer: generates fake media offline.

Useful for trying out VTSearch without an internet connection. Pick a
``media_type`` and a ``size`` and the importer renders that many synthetic
files (cartoon smiley faces, shapes and scenes for images; tones / chords /
drums / rain / wind / bird chirps for audio; bouncing balls / walking smileys
/ rotating shapes / marquees for video). A ``seed`` picks which set of that
mix: the same seed always makes the same files, and two seeds make two sets
that share nothing - a pile to train a detector on and a second one it has
never seen. Files are cached to ``data/synthetic/...`` so subsequent reloads
skip regeneration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vtscore.config import DATA_DIR
from vtscore.datasets.importers.base import ImporterBase, PluginField
from vtscore.datasets.loader import load_dataset_from_folder

_SUPPORTED_MEDIA_TYPES = ["image", "audio", "video"]
_DEFAULT_SEED = 1

#: How many times a generator's drawing has changed since its caches were
#: first written. A generator keeps any file already on disk under its naming
#: scheme, so this is part of the cache folder's name: bump it with any change
#: to what a generator draws, or its old files are served as the new ones.
_GENERATOR_VERSIONS = {"image": 2}


def _cache_dir(media_type: str, size: int, seed: int) -> Path:
    """The folder the files for ``(media_type, size, seed)`` are cached in."""
    version = _GENERATOR_VERSIONS.get(media_type)
    suffix = f"_v{version}" if version else ""
    return DATA_DIR / "synthetic" / f"{media_type}_{size}_seed{seed}{suffix}"


class SyntheticDatasetImporter(ImporterBase):
    """Generate a fake dataset of images, audio, or video (no network needed).

    The user picks a media type, a size and (optionally) a seed; everything
    else is deterministic, so repeated loads of the same parameters reuse
    cached files.
    """

    name = "synthetic"
    display_name = "Synthetic Media"
    description = "Generate fake media offline (useful for demos and field testing without internet)"
    # 🏭 (frontend renders this as a line-drawing factory icon; see
    # frontend/src/app/components/icon/icon.component.ts).
    icon = "\U0001f3ed"
    category = "demo"

    fields = [
        PluginField(
            key="media_type",
            label="Dataset media type",
            field_type="select",
            description="What kind of media to generate.",
            options=_SUPPORTED_MEDIA_TYPES,
            default="image",
            required=False,
        ),
        PluginField(
            key="size",
            label="Size",
            field_type="number",
            description="How many medias to generate (e.g. 100, 1000, 10000).",
            default="100",
            placeholder="100",
            min="1",
            step="1",
            required=False,
        ),
        PluginField(
            key="seed",
            label="Seed",
            field_type="number",
            description=(
                "Which set to generate. The same seed always makes the same media; "
                "a different seed makes a different set of the same mix."
            ),
            default=str(_DEFAULT_SEED),
            placeholder=str(_DEFAULT_SEED),
            min="0",
            step="1",
            required=False,
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
    def _parse_seed(field_values: dict[str, Any]) -> int:
        raw = field_values.get("seed", "")
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            return _DEFAULT_SEED
        try:
            seed = raw if isinstance(raw, int) else int(str(raw).strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid seed: {raw!r}") from exc
        if seed < 0:
            raise ValueError(f"Seed must not be negative, got {seed}")
        return seed

    @staticmethod
    def _media_type(field_values: dict[str, Any]) -> str:
        mt = (field_values.get("media_type") or "").strip()
        if mt not in _SUPPORTED_MEDIA_TYPES:
            raise ValueError(f"Unsupported media_type {mt!r} (expected one of {_SUPPORTED_MEDIA_TYPES})")
        return mt

    def _generate(self, media_type: str, size: int, seed: int) -> Path:
        """Render files into the cache dir, return the dir path."""
        from vtscore.concurrency.progress import get_thread_progress  # noqa: PLC0415

        on_progress = get_thread_progress()
        out_dir = _cache_dir(media_type, size, seed)
        if media_type == "image":
            from vtscore.utils.synthetic import generate_image_dataset  # noqa: PLC0415

            generate_image_dataset(out_dir, size, seed=seed, on_progress=on_progress)
        elif media_type == "audio":
            from vtscore.utils.synthetic import generate_audio_dataset  # noqa: PLC0415

            generate_audio_dataset(out_dir, size, seed=seed, on_progress=on_progress)
        elif media_type == "video":
            from vtscore.utils.synthetic import generate_video_dataset  # noqa: PLC0415

            generate_video_dataset(out_dir, size, seed=seed, on_progress=on_progress)
        else:
            raise ValueError(f"Unsupported media_type {media_type!r}")
        return out_dir

    # ------------------------------------------------------------------
    # Importer interface
    # ------------------------------------------------------------------

    def run(self, field_values: dict[str, Any], medias: dict, thin: bool = False) -> None:
        media_type = self._media_type(field_values)
        size = self._parse_size(field_values)
        seed = self._parse_seed(field_values)
        out_dir = self._generate(media_type, size, seed)
        origin = self.build_origin({"media_type": media_type, "size": str(size), "seed": str(seed)})
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
            seed = self._parse_seed(field_values)
        except ValueError:
            return self.display_name
        if seed != _DEFAULT_SEED:
            return f"Synthetic {mt} ({n}, seed {seed})"
        return f"Synthetic {mt} ({n})"

    # ------------------------------------------------------------------
    # Origin handling
    # ------------------------------------------------------------------

    def origin_display(self, origin: dict[str, Any]) -> str:
        params = origin.get("params", {})
        seed = params.get("seed") or _DEFAULT_SEED
        return f"synthetic:{params.get('media_type', '')}_{params.get('size', '')}_seed{seed}"

    def can_reload_from_origin(self, origin: dict[str, Any]) -> bool:
        params = origin.get("params", {})
        return params.get("media_type", "") in _SUPPORTED_MEDIA_TYPES and bool(params.get("size", ""))

    def reload_from_origin(self, origin: dict[str, Any]) -> dict[str, Any] | None:
        params = origin.get("params", {})
        mt = params.get("media_type", "")
        size = params.get("size", "")
        if mt not in _SUPPORTED_MEDIA_TYPES or not size:
            return None
        return {"media_type": mt, "size": size, "seed": params.get("seed") or str(_DEFAULT_SEED)}

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
            seed = self._parse_seed(params)
        except (TypeError, ValueError):
            return None
        root = _cache_dir(mt, size, seed)
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
