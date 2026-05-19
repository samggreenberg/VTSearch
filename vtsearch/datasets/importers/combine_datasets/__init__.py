"""Combine-datasets importer -- merge multiple pickle datasets into one.

All source datasets must share the same media type.  Duplicate entries
(identified by MD5 hash) are kept only once.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Callable, Iterator

from vtsearch.datasets.importers.base import DatasetImporter, ImporterField


def _get_progress():
    from vtsearch.concurrency.progress import update_progress

    return update_progress


def _load_clips_from_pickle(file_path: Path, thin: bool = False) -> dict[int, dict[str, Any]]:
    """Load medias from a pickle file without clearing a target dict.

    Returns a fresh dict mapping media-id to media-data.  This is a
    simplified version of :func:`~vtsearch.datasets.loader.load_dataset_from_pickle`
    that avoids side-effects on any global state.
    """
    from vtsearch.datasets.loader import load_dataset_from_pickle

    temp_medias: dict[int, dict[str, Any]] = {}
    load_dataset_from_pickle(file_path, temp_medias, thin=thin)
    return temp_medias


def _parse_dataset_paths(raw: Any) -> list[Path]:
    """Parse the ``datasets`` field into a validated list of Paths.

    Accepts either a comma-separated string or a list of path strings.
    Raises ``ValueError`` if fewer than two paths are supplied, and
    ``FileNotFoundError`` if any path does not exist on disk.
    """
    if isinstance(raw, list):
        paths = [Path(p) for p in raw if p]
    else:
        paths = [Path(p.strip()) for p in raw.split(",") if p.strip()]

    if len(paths) < 2:
        raise ValueError("At least two datasets are required to combine.")

    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Dataset file not found: {p}")

    return paths


def _iter_unique_source_clips(
    paths: list[Path],
    thin: bool,
    seen_md5s: set[str],
    mtype_state: list[str | None],
    progress: Callable[..., Any],
) -> Iterator[tuple[Path, list[dict[str, Any]], int]]:
    """Yield ``(pkl_path, deduped_medias, dupe_count)`` for each source pickle.

    The generator walks ``paths`` in order, emits a "Loading dataset
    i/N" progress update for each, and loads + dedups each pickle in
    turn. For empty pickles it yields ``(pkl_path, [], 0)`` so the
    caller can decide whether to emit a "skipping empty" notice.

    Mutates the shared ``seen_md5s`` set (appending newly-seen MD5s)
    and ``mtype_state[0]`` (latching the first observed media type).
    Raises ``ValueError`` on a media-type mismatch.
    """
    for i, pkl_path in enumerate(paths):
        progress(
            "loading",
            f"Loading dataset {i + 1}/{len(paths)}: {pkl_path.name}...",
            i + 1,
            len(paths),
        )
        source_clips = _load_clips_from_pickle(pkl_path, thin=thin)

        if not source_clips:
            yield pkl_path, [], 0
            continue

        first_clip = next(iter(source_clips.values()))
        source_media_type = first_clip.get("type", "audio")
        if mtype_state[0] is None:
            mtype_state[0] = source_media_type
        elif source_media_type != mtype_state[0]:
            raise ValueError(
                f"Media type mismatch: expected '{mtype_state[0]}' but "
                f"'{pkl_path.name}' contains '{source_media_type}' medias."
            )

        deduped: list[dict[str, Any]] = []
        dupes = 0
        for media in source_clips.values():
            md5 = media.get("md5", "")
            if md5 and md5 in seen_md5s:
                dupes += 1
                continue
            if md5:
                seen_md5s.add(md5)
            deduped.append(media)

        del source_clips
        gc.collect()
        yield pkl_path, deduped, dupes


class CombineDatasetsImporter(DatasetImporter):
    """Merge two or more existing ``.pkl`` datasets into a single dataset.

    All datasets must be of the same media type.  Entries with duplicate
    MD5 hashes are included only once (the first occurrence wins).
    """

    name = "combine_datasets"
    display_name = "Combined Datasets"
    description = "Merge multiple saved datasets into one, automatically removing duplicates"
    icon = "\U0001f500"  # twisted rightwards arrows
    ui_mode = "custom"
    hidden_from_picker = True
    # The combined output type is determined by the source pickles, not
    # by a user-chosen ``media_type``.  Flag set to keep the in-tree
    # importer set uniformly off the legacy shim.
    multi_media = True
    fields = [
        ImporterField(
            key="datasets",
            label="Dataset Files",
            field_type="text",
            description="Comma-separated paths to .pkl dataset files.",
        ),
        ImporterField(
            key="name",
            label="Name",
            field_type="text",
            description="Display name for the new combined dataset.",
        ),
    ]

    def default_display_name(self, field_values: dict[str, Any]) -> str:
        name = (field_values.get("name") or "").strip()
        return name or self.display_name

    def run(self, field_values: dict[str, Any], medias: dict, thin: bool = False) -> None:
        """Combine datasets specified by *field_values['datasets']*.

        ``field_values["datasets"]`` may be either:
        - a comma-separated string of file paths, or
        - a Python list of path strings (when called from the API route).
        """
        paths = _parse_dataset_paths(field_values.get("datasets", ""))
        progress = _get_progress()

        all_clips: list[dict[str, Any]] = []
        seen_md5s: set[str] = set()
        mtype_state: list[str | None] = [None]
        total_dupes = 0

        try:
            for i, (pkl_path, deduped, dupes) in enumerate(
                _iter_unique_source_clips(paths, thin, seen_md5s, mtype_state, progress)
            ):
                if not deduped:
                    progress("loading", f"Skipping empty dataset: {pkl_path.name}", i + 1, len(paths))
                    continue
                all_clips.extend(deduped)
                total_dupes += dupes
        except MemoryError:
            all_clips.clear()
            medias.clear()
            gc.collect()
            raise MemoryError(
                "Out of memory while combining datasets. "
                "Try combining fewer or smaller datasets, or free up system RAM."
            )

        if not all_clips:
            raise ValueError("No medias found in any of the selected datasets.")

        # Assign fresh sequential IDs and populate the target medias dict
        medias.clear()
        for new_id, media in enumerate(all_clips, start=1):
            media["id"] = new_id
            medias[new_id] = media

        msg = f"Combined {len(medias)} medias from {len(paths)} datasets"
        if total_dupes:
            msg += f" ({total_dupes} duplicate(s) skipped)"
        progress("idle", msg)

    def run_cli(self, field_values: dict[str, Any], medias: dict, thin: bool = False) -> None:
        """CLI entry point -- *datasets* is a comma-separated path string."""
        self.run(field_values, medias, thin=thin)

    @property
    def supports_chunked(self) -> bool:
        return True

    def run_chunked(
        self,
        field_values: dict[str, Any],
        chunk_size: int,
        thin: bool = False,
    ) -> Iterator[dict[int, dict[str, Any]]]:
        """Yield one chunk per source pickle, deduplicating across chunks.

        Each source pickle is loaded as its own chunk (with fresh IDs
        starting at 1).  Cross-source deduplication by MD5 is maintained
        across yields via a running ``seen_md5s`` set.
        """
        paths = _parse_dataset_paths(field_values.get("datasets", ""))
        progress = _get_progress()
        seen_md5s: set[str] = set()
        mtype_state: list[str | None] = [None]

        for _pkl_path, deduped, _dupes in _iter_unique_source_clips(
            paths, thin, seen_md5s, mtype_state, progress
        ):
            if not deduped:
                continue
            chunk_medias: dict[int, dict[str, Any]] = {}
            for new_id, media in enumerate(deduped, start=1):
                media["id"] = new_id
                chunk_medias[new_id] = media
            yield chunk_medias

    def run_chunked_cli(
        self,
        field_values: dict[str, Any],
        chunk_size: int,
        thin: bool = False,
    ) -> Iterator[dict[int, dict[str, Any]]]:
        yield from self.run_chunked(field_values, chunk_size, thin=thin)

    def build_origin(self, field_values: dict[str, Any]) -> dict[str, Any]:
        """Build an origin dict listing the source dataset paths."""
        raw = field_values.get("datasets", "")
        if isinstance(raw, list):
            datasets_str = ",".join(raw)
        else:
            datasets_str = raw
        return {"importer": self.name, "params": {"datasets": datasets_str}}


IMPORTER = CombineDatasetsImporter()
