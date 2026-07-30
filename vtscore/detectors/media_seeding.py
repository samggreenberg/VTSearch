"""Seed good votes from a model's media examples.

Provides :func:`seed_good_votes_from_examples` which reads example media
files, matches them to loaded dataset medias by MD5, and votes them good.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any

from vtscore.utils.hashing import content_md5


def _ensure_embedder(embedder, dataset_embedder_name: str, dataset_media_type: str):
    """Resolve an embedder for example media, loading it lazily.

    Returns the already-loaded *embedder* if non-``None``; otherwise tries
    the dataset's named embedder, then falls back to the first embedder
    available for *dataset_media_type*.  Returns ``None`` when no embedder
    is available.
    """
    if embedder is not None:
        return embedder

    from vtscore.media import embedders_for_type, get_embedder

    if dataset_embedder_name:
        try:
            embedder = get_embedder(dataset_embedder_name)
        except KeyError:
            pass
    if embedder is None:
        avail = embedders_for_type(dataset_media_type)
        embedder = avail[0] if avail else None
    return embedder


def _real_example_origin(ex: dict[str, Any]) -> dict[str, Any] | None:
    """Return the example's validated real origin dict, or ``None``.

    An example saved by a datasource importer carries the item's durable
    origin (``{"importer": "url_download", "params": {"url": ...}}``,
    ``{"importer": "server_file", "params": {"path": ...}}``, ...) so the
    seeded media points back at its source instead of at the dead-end
    ``example_media`` sentinel.  Examples saved before origins existed (and
    genuinely un-re-derivable ones: uploads, add-to-pile snapshots) have no
    ``origin`` key and fall back to the sentinel.

    Security: the origin comes from a detector JSON / request body, so its
    path-like params must pass the same per-user confinement the ingress
    applies; an origin that fails the check is discarded (sentinel
    fallback).  URL params are re-validated by the url_download source at
    fetch time (``validate_url`` plus per-redirect-hop checks in the
    downloader).
    """
    origin = ex.get("origin")
    if not isinstance(origin, dict):
        return None
    importer = origin.get("importer")
    if not isinstance(importer, str) or not importer or importer == "example_media":
        return None
    if not isinstance(origin.get("params"), dict):
        return None

    from vtscore.security.origin_validation import check_origin_param_confinement

    try:
        check_origin_param_confinement(origin)
    except ValueError:
        return None
    return origin


def _example_origin_name(origin: dict[str, Any] | None, filename: str) -> str:
    """Human-meaningful origin name: the URL / path for a real origin, else the cache filename."""
    if origin is not None:
        params = origin.get("params", {})
        name = params.get("url") or params.get("path") or ""
        if isinstance(name, str) and name:
            return name
    return filename


def _example_file_path(server_media_dir: Path, filename: str) -> Path | None:
    """The example's path inside ``example_media/``, or ``None`` on a traversal attempt."""
    file_path = server_media_dir / filename
    try:
        file_path.resolve().relative_to(server_media_dir.resolve())
    except ValueError:
        return None
    return file_path


def _insert_example_media(
    origin: dict[str, Any] | None,
    origin_name: str,
    filename: str,
    file_bytes: bytes,
    file_md5: str,
    embedding,
    dataset_media_type: str,
    dataset_embedder_name: str,
) -> None:
    """Insert an embedded example into the active context's medias and vote it good."""
    from vtscore.state import _state_lock, apply_label, get_active_context, next_media_id

    with _state_lock:
        active_medias = get_active_context().medias
        new_id = next_media_id(active_medias)
        active_medias[new_id] = {
            "id": new_id,
            "media_type": dataset_media_type,
            "embedder": dataset_embedder_name,
            "md5": file_md5,
            "embeddings": {dataset_embedder_name: embedding},
            "media_bytes": file_bytes,
            "filename": filename,
            "file_size": len(file_bytes),
            "category": "",
            "origin": origin
            if origin is not None
            else {
                "importer": "example_media",
                "params": {"filename": filename},
            },
            "origin_name": origin_name,
        }

    apply_label(new_id, "good", record_achievement=False)


def _seed_one_example(
    ex: dict[str, Any],
    server_media_dir: Path,
    md5_lookup: dict,
    dataset_media_type: str,
    dataset_embedder_name: str,
    embedder,
) -> tuple[int, Any]:
    """Seed a single media example.

    Returns ``(1 or 0, embedder)`` - whether the example was seeded, plus
    the (possibly lazily-loaded) embedder for the caller to reuse.
    """
    from vtscore.detectors.resolver import resolve_file_context
    from vtscore.state import apply_label

    filename = ex["value"].strip()
    file_path = _example_file_path(server_media_dir, filename)
    if file_path is None:
        return 0, embedder

    origin = _real_example_origin(ex)
    origin_name = _example_origin_name(origin, filename)

    # Hold any resolver-backed source alive (its temp file must survive
    # until the example is read and embedded below).
    with ExitStack() as stack:
        if file_path.is_file():
            embed_path: Path | None = file_path
        elif origin is not None:
            # The example_media/ cache file is gone; the origin is the
            # canonical form, so re-derive the bytes from it.
            embed_path = stack.enter_context(resolve_file_context(origin, origin_name, filename))
        else:
            embed_path = None
        if embed_path is None or not embed_path.is_file():
            return 0, embedder

        file_bytes = embed_path.read_bytes()
        file_md5 = content_md5(file_bytes)
        cids = md5_lookup.get(file_md5, [])

        if cids:
            # Example matches existing dataset media - just vote good.
            # System-driven seeding from a detector's saved examples, not a
            # user vote action, so don't credit achievement counters.
            for cid in cids:
                apply_label(cid, "good", record_achievement=False)
            return 1, embedder

        # Example is NOT in the dataset - embed and insert as new media.
        embedder = _ensure_embedder(embedder, dataset_embedder_name, dataset_media_type)
        if embedder is None:
            # No embedder available; the caller's remaining examples will
            # skip here too.
            return 0, embedder

        from vtscore.media.embedder import media_from_path  # noqa: PLC0415

        embedding = embedder.embed_media(media_from_path(embed_path))
        if embedding is None:
            return 0, embedder

        _insert_example_media(
            origin,
            origin_name,
            filename,
            file_bytes,
            file_md5,
            embedding,
            dataset_media_type,
            dataset_embedder_name,
        )
        return 1, embedder


def seed_good_votes_from_examples(examples: list[dict]) -> int:
    """Seed good votes from a model's media examples.

    For each ``type: "media"`` example, reads the file from
    ``data/example_media/`` (re-fetching it from the example's ``origin``
    when the cached file is gone) and adds it to ``good_votes``:

    * **Match by MD5** - if a loaded media has the same content hash,
      that media is voted good (keeping its original dataset origin).
    * **No match** - the example file is embedded using the dataset's
      embedder, inserted into the ``medias`` dict as a new item, and voted
      good.  This makes it available for training (its embedding is in the
      medias snapshot) and for label export (LabelSet picks it up from
      medias + votes).  The inserted media carries the example's real
      origin when it has one (so the resulting labels resolve against any
      dataset), falling back to the ``example_media`` sentinel origin for
      examples with no re-derivable source.

    Returns the number of example entries successfully seeded.
    """

    from vtscore.config import DATA_DIR
    from vtscore.state import cached_md5_lookup, snapshot_medias

    media_examples = [
        ex for ex in examples if isinstance(ex, dict) and ex.get("type") == "media" and ex.get("value", "").strip()
    ]
    if not media_examples:
        return 0

    snap = snapshot_medias()
    if not snap:
        return 0

    md5_lookup = cached_md5_lookup()
    server_media_dir = DATA_DIR / "example_media"

    # Determine the embedder and media type from the loaded dataset so we
    # can embed example files that aren't already in the dataset.
    first_media = next(iter(snap.values()))
    dataset_media_type = first_media.get("media_type", "audio")
    dataset_embedder_name = first_media.get("embedder", "")
    embedder = None  # lazily loaded only when needed

    seeded = 0
    for ex in media_examples:
        delta, embedder = _seed_one_example(
            ex,
            server_media_dir,
            md5_lookup,
            dataset_media_type,
            dataset_embedder_name,
            embedder,
        )
        seeded += delta

    return seeded
