"""Import a filtered subset of media straight out of tar/zip shards.

This importer is the no-extraction path for WebDataset-style corpora: tens of
thousands of audio/video chunks live inside a handful of multi-GB ``shard_*.tar``
files, and a sidecar manifest already pairs a chosen subset of those chunks with
**pre-computed** embedding vectors.  Rather than extract a second on-disk copy
of the shards (multivent-raw's ``videos/`` alone is 4.1 TB), each imported media
records only ``{archive path, member name}`` and re-derives its bytes on demand
through :mod:`vtscore.datasets.archive_stream` -- so a media plays back by
streaming a single tar member with HTTP Range, never persisting it.

Input is a single ``.npz`` manifest holding, per row, an archive path, a member
name within that archive, and a pre-computed vector (see
:func:`~vtscore.datasets.importers._npz_vectors.read_npz_archive_member_rows`).
Because the embeddings are supplied, the framework embed stage is skipped: the
import needs no GPU and reads no member bytes (it only walks each referenced
shard's tar headers to confirm the member exists and record its size).

Each row becomes one whole-member media.  Sub-file *windowing* (multiple 10 s
clip items per member, carrying ``clip_start`` / ``clip_end``) is the planned
Phase B extension -- see ``docs/plans/webdataset-tar-import.md``.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from vtscore.datasets.archive_stream import (
    ArchiveMemberError,
    LOCAL_ARCHIVE_MEMBER_IMPORTER,
    member_size,
)
from vtscore.datasets.importers._npz_vectors import read_npz_archive_member_rows, read_npz_embedder_name
from vtscore.datasets.importers.base import ImporterBase, PluginField
from vtscore.embedding.media_vectors import init_embeddings

logger = logging.getLogger(__name__)


def _synthetic_md5(archive: str, member: str) -> str:
    """Return a stable content-id for a member without reading its bytes.

    The real point of this importer is to avoid touching member *data* (the
    corpora are far too large to hash in full), so the dedup/label key is
    derived from the globally-unique ``archive::member`` pair instead of the
    member's content.  This keeps in-dataset dedup, voting, and label
    persistence working; the one thing it gives up is *content*-based label
    transfer across unrelated datasets, which these precomputed-embedding
    corpora don't use.
    """
    return hashlib.md5(f"{archive}::{member}".encode()).hexdigest()


class LocalArchiveMemberImporter(ImporterBase):
    """Import a manifest-selected subset of archive members with no extraction.

    The user supplies a ``.npz`` manifest pairing tar/zip members with
    pre-computed embeddings; each row imports as one media whose bytes stream
    from the shard on demand.  Nothing is written to disk and no member data is
    read at import time.
    """

    name = LOCAL_ARCHIVE_MEMBER_IMPORTER
    display_name = "Archive members (no extract)"
    description = (
        "Import a manifest-selected subset of media stored inside tar/zip shards, "
        "with pre-computed embeddings, without extracting the archives. Bytes stream "
        "a single member on demand -- built for WebDataset-style corpora too large to copy."
    )
    icon = "\U0001f4e6"  # 📦
    picker_view = "form"
    category = "server"

    fields = [
        PluginField(
            key="media_type",
            label="Dataset MediaType",
            field_type="select",
            description="Type of media the referenced members hold (e.g. video or audio).",
            default="video",
            required=False,
        ),
        PluginField(
            key="manifest",
            label="Manifest (.npz)",
            field_type="server_path",
            description="A .npz manifest pairing archive members with pre-computed embedding vectors.",
            hint=(
                "Expected arrays:\n"
                " • vectors / embeddings - (N, D) float embeddings, one per row.\n"
                " • members - (N,) member names within their archive.\n"
                " • archives - (N,) archive paths (or a single value for one shard);\n"
                "   relative paths resolve against the manifest's directory.\n"
                " • filenames (optional) - (N,) display names.\n"
                " • embedder_name (optional) - the embedder that produced the vectors.\n"
                "Members are streamed on demand; the archives are never extracted."
            ),
            accept=".npz",
        ),
    ]

    def __init__(self) -> None:
        super().__init__()
        from vtscore.media import all_folder_names  # noqa: PLC0415

        for f in self.fields:
            if f.key == "media_type":
                f.options = all_folder_names()
                break

    def run(self, field_values: dict[str, Any], medias: dict, thin: bool = False) -> None:
        from vtscore.media import get_by_folder_name  # noqa: PLC0415

        manifest = Path(field_values["manifest"])
        if not manifest.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest}")
        output_type = get_by_folder_name(field_values.get("media_type", "video")).type_id

        rows = read_npz_archive_member_rows(manifest)
        embedder_name = read_npz_embedder_name(manifest)

        next_id = max(medias.keys(), default=0) + 1
        skipped = 0
        for row in rows:
            archive = row["archive"]
            member = row["member"]
            try:
                size = member_size(archive, member)
            except (ArchiveMemberError, OSError) as exc:
                # The manifest references a member we can't locate in its shard
                # (missing archive, renamed member). Skip it rather than import
                # a media whose bytes will never resolve.
                logger.warning("local_archive_member: skipping %s::%s (%s)", archive, member, exc)
                skipped += 1
                continue

            origin = {
                "importer": self.name,
                "params": {
                    "archive_path": archive,
                    "member": member,
                    "media_type": output_type,
                    "manifest": str(manifest.resolve()),
                    "embedder_name": embedder_name,
                },
            }
            medias[next_id] = {
                "id": next_id,
                "media_type": output_type,
                "embedder": embedder_name,
                "file_size": size,
                "md5": _synthetic_md5(archive, member),
                "embeddings": init_embeddings(embedder_name, row["vector"]),
                "filename": row["filename"],
                "category": "custom",
                "origin": origin,
                "origin_name": f"{archive}::{member}",
                "media_bytes": None,
                "media_string": None,
                "duration": 0,
                "archive_member": {"path": archive, "member": member},
            }
            next_id += 1

        if not medias:
            raise ValueError(
                f"No importable members in {manifest}"
                + (f" ({skipped} referenced member(s) could not be located)" if skipped else "")
            )

    def run_cli(self, field_values: dict[str, Any], medias: dict, thin: bool = False) -> None:
        manifest = Path(field_values["manifest"])
        if not manifest.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest}")
        if not manifest.is_file():
            raise IsADirectoryError(f"Manifest must be a file: {manifest}")
        self.run(field_values, medias, thin=thin)

    def default_display_name(self, field_values: dict[str, Any]) -> str:
        manifest = (field_values.get("manifest") or "").strip()
        if manifest:
            stem = Path(manifest).stem
            if stem:
                return stem
        return self.display_name

    def origin_display(self, origin: dict[str, Any]) -> str:
        params = origin.get("params", {})
        member = params.get("member", "")
        archive = params.get("archive_path", "")
        if archive and member:
            return f"{self.name}:{Path(archive).name}::{member}"
        return self.name

    def can_reload_from_origin(self, origin: dict[str, Any]) -> bool:
        manifest = origin.get("params", {}).get("manifest", "")
        return bool(manifest) and Path(manifest).is_file()

    def reload_from_origin(self, origin: dict[str, Any]) -> dict[str, Any] | None:
        params = origin.get("params", {})
        manifest = params.get("manifest", "")
        if not manifest or not Path(manifest).is_file():
            return None
        return {"manifest": manifest, "media_type": params.get("media_type", "video")}


IMPORTER = LocalArchiveMemberImporter()
