"""Multivent/microvent WebDataset importer for VTSearch.

Creates one audio :class:`~vtscore.media.audio.media_type.AudioMediaType` per
10-second CLAP window from a microvent or multivent-raw corpus directory.
Pre-computed CLAP embeddings are injected directly — the CLAP model is not
re-run, so no GPU is needed.

Directory layout expected (same for microvent and multivent-raw)::

    <data_dir>/
        audio/
            catalog.csv          (chunk_id, shard_index, has_audio, ...)
            shard_000000.tar     (<chunk_id>.m4a members)
            ...
        embeddings/
            audemb_largerclapgeneral/
                shard_000000.tar (<chunk_id>.audemb_largerclapgeneral.npz members)
                ...

The importer extracts 10-second audio clips from the audio tar shards into
a persistent cache directory and builds a ``content_vectors`` mapping so
that :func:`~vtscore.datasets.loader_folder.load_dataset_from_folder` skips
re-embedding.  On subsequent imports the cached clips are reused and only
the embedding NPZs are re-read.

Text queries in VTSearch use the ``clap_general`` encoder
(``laion/larger_clap_general``), which shares the 512-dimensional embedding
space of the pre-computed ``audemb_largerclapgeneral`` vectors.
"""

from __future__ import annotations

import io
import logging
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vtscore.datasets.importers.base import DatasetImporter, ImporterField
from vtscore.datasets.loader_folder import load_dataset_from_folder

_log = logging.getLogger(__name__)

_WINDOW_SEC = 10.0
_EMBEDDER_NAME = "clap_general"
_EMB_TAG = "audemb_largerclapgeneral"


class MultiventDatasetImporter(DatasetImporter):
    """Import microvent or multivent-raw as 10-second audio windows.

    Each Media corresponds to one ``(chunk_id, t_offset)`` CLAP window.
    Pre-computed ``laion/larger_clap_general`` embeddings are injected
    directly so VTSearch text queries work without re-running the model.

    First import extracts audio clips via ffmpeg (a few minutes for
    microvent; longer for multivent-raw).  Subsequent imports reuse the
    cached clips.
    """

    name = "multivent"
    display_name = "Multivent / microvent"
    description = (
        "Import microvent or multivent-raw WebDataset as 10-second audio "
        "windows with pre-computed CLAP embeddings (no GPU required)"
    )
    icon = "\U0001f3a7"  # 🎧
    category = "server"
    extra_origin_keys = ("data_dir", "cache_dir")

    fields = [
        ImporterField(
            key="data_dir",
            label="Dataset directory",
            field_type="folder",
            description=(
                "Absolute path to the microvent or multivent-raw dataset root "
                "(the directory that contains audio/ and embeddings/)."
            ),
        ),
        ImporterField(
            key="cache_dir",
            label="Clip cache directory",
            field_type="text",
            description=(
                "Directory where extracted 10-second audio clips are stored. "
                "Created automatically on first import and reused on "
                "subsequent loads. Leave blank to use "
                "<data_dir>/../vtsearch_clips/<dataset_name>."
            ),
            required=False,
        ),
        ImporterField(
            key="max_shards",
            label="Max shards (for testing)",
            field_type="text",
            description=(
                "Limit the number of embedding shards to load.  Useful for "
                "a quick smoke-test.  Leave blank to import all shards."
            ),
            required=False,
        ),
    ]

    # ------------------------------------------------------------------ #
    # main entry point
    # ------------------------------------------------------------------ #

    def run(self, field_values: dict[str, Any], medias: dict, thin: bool = False) -> None:
        data_dir = Path(field_values["data_dir"])

        raw_cache = (field_values.get("cache_dir") or "").strip()
        cache_dir = (
            Path(raw_cache)
            if raw_cache
            else data_dir.parent / "vtsearch_clips" / data_dir.name
        )
        cache_dir.mkdir(parents=True, exist_ok=True)

        max_shards_raw = (field_values.get("max_shards") or "").strip()
        max_shards = int(max_shards_raw) if max_shards_raw.isdigit() else None

        audio_shard_dir = data_dir / "audio"
        emb_shard_dir = data_dir / "embeddings" / _EMB_TAG

        if not audio_shard_dir.is_dir():
            raise FileNotFoundError(f"Audio directory not found: {audio_shard_dir}")
        if not emb_shard_dir.is_dir():
            raise FileNotFoundError(
                f"Embedding directory not found: {emb_shard_dir}. "
                "Make sure the microvent-features embeddings are present "
                f"(expected: {_EMB_TAG})."
            )

        # audio catalog: chunk_id → (shard_index, has_audio)
        audio_cat = pd.read_csv(audio_shard_dir / "catalog.csv")
        chunk_to_audio_shard: dict[str, int] = dict(
            zip(audio_cat["chunk_id"], audio_cat["shard_index"])
        )
        chunks_with_audio: set[str] = set(
            audio_cat.loc[audio_cat["has_audio"].astype(bool), "chunk_id"]
        )

        content_vectors: dict[str, np.ndarray] = {}
        custom_metadata: dict[str, dict[str, Any]] = {}

        emb_shards = sorted(emb_shard_dir.glob("shard_*.tar"))
        if max_shards is not None:
            emb_shards = emb_shards[:max_shards]

        for emb_shard_path in emb_shards:
            self._process_embedding_shard(
                emb_shard_path,
                audio_shard_dir,
                chunk_to_audio_shard,
                chunks_with_audio,
                cache_dir,
                content_vectors,
                custom_metadata,
            )

        _log.info("Multivent importer: %d audio windows ready in %s", len(content_vectors), cache_dir)

        if not content_vectors:
            return

        origin = self.build_origin(field_values)
        load_dataset_from_folder(
            cache_dir,
            "audio",
            medias,
            origin=origin,
            thin=thin,
            content_vectors=content_vectors if content_vectors else None,
            content_embedder_name=_EMBEDDER_NAME,
            custom_metadata_map=custom_metadata if custom_metadata else None,
        )

    # ------------------------------------------------------------------ #
    # per-shard processing
    # ------------------------------------------------------------------ #

    def _process_embedding_shard(
        self,
        emb_shard_path: Path,
        audio_shard_dir: Path,
        chunk_to_audio_shard: dict[str, int],
        chunks_with_audio: set[str],
        cache_dir: Path,
        content_vectors: dict[str, np.ndarray],
        custom_metadata: dict[str, dict[str, Any]],
    ) -> None:
        # Load all embeddings from this shard
        shard_emb: dict[str, tuple[list[str], np.ndarray]] = {}
        with tarfile.open(emb_shard_path) as t:
            for member in t.getmembers():
                chunk_id = member.name.split(".")[0]
                if chunk_id not in chunks_with_audio:
                    continue
                try:
                    raw = t.extractfile(member)
                    if raw is None:
                        continue
                    data = np.load(io.BytesIO(raw.read()))
                    shard_emb[chunk_id] = (list(data["keyframe_ids"]), data["embeddings"])
                except Exception:
                    _log.warning("Failed to load NPZ %s from %s", member.name, emb_shard_path, exc_info=True)

        # Group chunks by audio shard to minimise tar re-opens
        by_audio_shard: dict[int, list[str]] = {}
        for chunk_id in shard_emb:
            a_shard = chunk_to_audio_shard.get(chunk_id)
            if a_shard is None:
                _log.debug("chunk %s not in audio catalog; skipping", chunk_id)
                continue
            by_audio_shard.setdefault(a_shard, []).append(chunk_id)

        for a_shard_idx, chunk_ids in by_audio_shard.items():
            audio_shard_path = audio_shard_dir / f"shard_{a_shard_idx:06d}.tar"
            if not audio_shard_path.is_file():
                _log.warning("Missing audio shard: %s", audio_shard_path)
                continue
            with tarfile.open(audio_shard_path) as audio_tar:
                for chunk_id in chunk_ids:
                    self._process_chunk(
                        chunk_id, audio_tar, audio_shard_path,
                        shard_emb[chunk_id], cache_dir,
                        content_vectors, custom_metadata,
                    )

    def _process_chunk(
        self,
        chunk_id: str,
        audio_tar: tarfile.TarFile,
        audio_shard_path: Path,
        emb_data: tuple[list[str], np.ndarray],
        cache_dir: Path,
        content_vectors: dict[str, np.ndarray],
        custom_metadata: dict[str, dict[str, Any]],
    ) -> None:
        m4a_member = f"{chunk_id}.m4a"
        try:
            fobj = audio_tar.extractfile(m4a_member)
            if fobj is None:
                return
            m4a_bytes = fobj.read()
        except KeyError:
            _log.warning("Missing audio member %s in %s", m4a_member, audio_shard_path)
            return

        window_ids, embeddings = emb_data
        for t_label, emb_vec in zip(window_ids, embeddings):
            t_sec = int(t_label[1:])  # 't000030' → 30
            clip_name = f"{chunk_id}_{t_label}.m4a"
            clip_path = cache_dir / clip_name

            if not clip_path.exists():
                ok = _ffmpeg_extract_clip(m4a_bytes, t_sec, _WINDOW_SEC, clip_path)
                if not ok:
                    _log.warning("ffmpeg failed: %s @ %s", chunk_id, t_label)
                    continue

            content_vectors[clip_name] = emb_vec.astype(np.float32)
            custom_metadata[clip_name] = {
                "chunk_id": chunk_id,
                "t_offset": t_label,
                "t_start_sec": t_sec,
            }


# ------------------------------------------------------------------ #
# module-level helpers
# ------------------------------------------------------------------ #


def _ffmpeg_extract_clip(
    m4a_bytes: bytes, t_start: float, duration: float, out_path: Path
) -> bool:
    """Cut a time-bounded segment from *m4a_bytes* and write to *out_path*.

    Uses fast keyframe-aligned seek (``-ss`` before ``-i``) for speed.
    The output is re-muxed without re-encoding (``-c:a copy``).
    Returns True on success.
    """
    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp:
        tmp.write(m4a_bytes)
        tmp_path = tmp.name
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(int(t_start)),
                "-i", tmp_path,
                "-t", str(int(duration)),
                "-c:a", "copy",
                str(out_path),
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            _log.debug("ffmpeg stderr: %s", result.stderr.decode(errors="replace"))
        return result.returncode == 0
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


IMPORTER = MultiventDatasetImporter()
