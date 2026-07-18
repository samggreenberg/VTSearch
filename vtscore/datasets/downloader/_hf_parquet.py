"""Shared helpers for HuggingFace parquet-hosted image demo datasets.

Some demo datasets (RICO-Screen2Words, RVL-CDIP) are distributed on the Hub as
parquet shards whose images live in an ``Image``-feature column (a struct of
``{bytes, path}``) rather than as loose files.  The rest of the image demo
pipeline (``_collect_simple_folder_files`` → ``_embed_file_images``) and the
on-demand file re-resolution in the demo importer both expect *loose files on
disk grouped into ``<category>/`` folders*.  So the download step for these
datasets pulls the parquet shards, decodes each row's image bytes to a
``<out_dir>/<category>/<id>.<ext>`` file, and then deletes the parquet to
reclaim disk — after which the dataset behaves exactly like the folder-per-class
demos (Caltech, Food-101, EuroSAT).

``pyarrow`` is a declared dependency but only these two demos need it, so it is
imported lazily and a missing install surfaces as a short, actionable
``RuntimeError`` rather than an ``ImportError`` (mirroring the ``py7zr`` handling
for the Clotho ``.7z`` archive).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator, Optional

from vtscore.datasets.downloader.core import ProgressCallback


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover - exercised only without pyarrow installed
        raise RuntimeError(
            "Reading this demo dataset needs the 'pyarrow' package to decode its "
            "parquet shards. Install it with 'pip install pyarrow' and try again."
        ) from exc
    return pq


def list_parquet_shards(repo_id: str, path_prefix: str) -> list[str]:
    """Return the dataset repo's parquet files whose path starts with *path_prefix*.

    Used when shard filenames carry a volatile content hash (e.g.
    ``data/train-00000-of-00001-<hash>.parquet``) that would rot if hardcoded.
    """
    from huggingface_hub import HfApi  # noqa: PLC0415

    from vtscore.security.hf_auth import get_token  # noqa: PLC0415

    files = HfApi().list_repo_files(repo_id, repo_type="dataset", token=get_token())
    return sorted(f for f in files if f.startswith(path_prefix) and f.endswith(".parquet"))


def download_parquet_shards(
    repo_id: str,
    filenames: list[str],
    dest_dir: Path,
    dataset_name: str,
    on_progress: ProgressCallback,
) -> list[Path]:
    """Download the named parquet *filenames* from a HuggingFace dataset repo.

    Returns the local paths in the same order.  Reports coarse per-shard
    progress (byte-level progress within a single ``hf_hub_download`` is not
    surfaced by the Hub client, so we tick once per completed shard).
    """
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    from vtscore.security.hf_auth import get_token  # noqa: PLC0415

    dest_dir.mkdir(parents=True, exist_ok=True)
    token = get_token()
    total = len(filenames)
    paths: list[Path] = []
    for i, name in enumerate(filenames):
        on_progress("downloading", f"Downloading {dataset_name} shard {i + 1}/{total}...", i, total)
        local = hf_hub_download(
            repo_id=repo_id,
            filename=name,
            repo_type="dataset",
            local_dir=str(dest_dir),
            token=token,
        )
        paths.append(Path(local))
    on_progress("downloading", f"Downloading {dataset_name} shards...", total, total)
    return paths


def iter_parquet_rows(shard_path: Path, columns: list[str], batch_size: int = 256) -> Iterator[dict]:
    """Yield each row of *shard_path* as a dict over *columns*.

    Streamed in record batches so a multi-hundred-MB shard is never fully
    materialised at once.  An ``Image``-feature column decodes to a dict
    ``{"bytes": <png/jpg bytes>, "path": ...}``.
    """
    pq = _require_pyarrow()
    pf = pq.ParquetFile(str(shard_path))
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        cols = {c: batch.column(c).to_pylist() for c in columns}
        for i in range(batch.num_rows):
            yield {c: cols[c][i] for c in columns}


def extract_images_to_folders(
    shard_paths: list[Path],
    *,
    image_col: str,
    out_dir: Path,
    category_of: Callable[[dict], Optional[str]],
    id_of: Callable[[dict, int], str],
    ext: str,
    dataset_name: str,
    on_progress: ProgressCallback,
    columns: Optional[list[str]] = None,
) -> None:
    """Decode image bytes from parquet *shard_paths* into ``out_dir/<category>/``.

    ``category_of(row)`` returns the display category folder name for a row, or
    ``None`` to skip it.  ``id_of(row, running_index)`` returns the stem for the
    written file.  Rows whose image bytes are missing are skipped.  The scan is
    idempotent at the call level: callers gate on ``out_dir`` already being
    populated before invoking this.
    """
    read_cols = columns or [image_col]
    written = 0
    global_idx = 0  # monotonic across shards so id_of(row, idx) never collides
    for si, shard in enumerate(shard_paths):
        on_progress(
            "extracting",
            f"Extracting {dataset_name} images (shard {si + 1}/{len(shard_paths)})...",
            si,
            len(shard_paths),
        )
        for row in iter_parquet_rows(shard, read_cols):
            idx = global_idx
            global_idx += 1
            category = category_of(row)
            if category is None:
                continue
            img = row.get(image_col)
            data = img.get("bytes") if isinstance(img, dict) else None
            if not data:
                continue
            cat_dir = out_dir / category
            cat_dir.mkdir(parents=True, exist_ok=True)
            (cat_dir / f"{id_of(row, idx)}.{ext}").write_bytes(data)
            written += 1
            if written % 500 == 0:
                on_progress(
                    "extracting", f"Extracting {dataset_name} images ({written} written)...", si, len(shard_paths)
                )
    on_progress("extracting", f"Extracting {dataset_name} images...", len(shard_paths), len(shard_paths))
