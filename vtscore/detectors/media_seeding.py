"""Turn a detector's media examples into labels and votes.

Two entry points, both keyed off the same example → origin identity:

* :func:`labeled_elements_from_examples` converts the media examples the
  user supplied at create time into ``good`` :class:`LabeledElement`s, so
  the detector's labelset carries them from the moment it exists - no
  dataset, embedder, or vote required.  This is what makes an exemplar a
  *label* rather than a transient hint.
* :func:`seed_good_votes_from_examples` reads the example media files,
  matches them to loaded dataset medias by MD5, and votes them good, so
  the same exemplars are usable for training against whatever dataset is
  active.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vtscore.utils.hashing import content_md5

if TYPE_CHECKING:
    from vtscore.datasets.labelset import LabeledElement, LabelSet


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
    params must pass the same per-user confinement the ingress applies; an
    origin that fails the check is discarded (sentinel fallback).  The
    *confined* copy is returned so the seeded media resolves the path the
    check approved rather than one anchored at the process CWD.  URL params
    are re-validated by the url_download source at fetch time
    (``validate_url`` plus per-redirect-hop checks in the downloader).
    """
    origin = ex.get("origin")
    if not isinstance(origin, dict):
        return None
    importer = origin.get("importer")
    if not isinstance(importer, str) or not importer or importer == "example_media":
        return None
    if not isinstance(origin.get("params"), dict):
        return None

    from vtscore.security.origin_validation import confine_origin_params

    try:
        return confine_origin_params(origin)
    except ValueError:
        return None


def _sentinel_origin(filename: str) -> dict[str, Any]:
    """The dead-end ``example_media`` origin for an example with no real source.

    Uploads and add-to-pile snapshots have no re-derivable origin: the bytes
    only exist in the ``example_media/`` cache.  The sentinel still gives the
    element a stable identity key (see
    :func:`~vtscore.datasets.labelset.element_key`), so the label dedupes and
    survives merges the same way a real origin does.
    """
    return {"importer": "example_media", "params": {"filename": filename}}


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
            "origin": origin if origin is not None else _sentinel_origin(filename),
            "origin_name": origin_name,
        }

    apply_label(new_id, "good", record_achievement=False)


def labeled_elements_from_examples(examples: list[dict]) -> list["LabeledElement"]:
    """Build one ``good`` :class:`LabeledElement` per ``type: "media"`` example.

    A user who supplies three Good exemplars instead of a Good text
    description has cast three good votes, so those exemplars belong in the
    detector's labelset - not merely in its ``examples`` list, which nothing
    downstream of training reads.  This runs at create time, before any
    dataset is involved: the element's identity is its **origin**, which is
    dataset-agnostic by construction, so an ``https://`` exemplar is kept
    verbatim even when the dataset it will be used against is entirely local
    files (issue #3045).

    Each element carries:

    * the example's validated real origin (``url_download``, ``server_file``,
      ...) when it has one, else the ``example_media`` sentinel for genuinely
      un-re-derivable exemplars (uploads, add-to-pile snapshots);
    * the MD5 of the ``example_media/`` cache file when it is still there, so
      the label can also match dataset media by content hash.  A missing
      cache file leaves ``md5`` empty and costs nothing: origin is the
      preferred identity key either way.

    Text examples are skipped - they are queries, not labeled media.  The
    origin / origin_name derivation is shared with
    :func:`seed_good_votes_from_examples`, so the label a create emits and
    the media a later seed inserts collapse onto one identity key instead of
    double-counting.
    """
    from vtscore.datasets.labelset import LabeledElement
    from vtscore.security.path_validation import example_media_dir
    from vtscore.utils.hashing import file_md5

    server_media_dir = example_media_dir()
    elements: list[LabeledElement] = []
    for ex in examples:
        if not isinstance(ex, dict) or ex.get("type") != "media":
            continue
        filename = (ex.get("value") or "").strip()
        if not filename:
            continue

        origin = _real_example_origin(ex)
        origin_name = _example_origin_name(origin, filename)

        md5 = ""
        file_path = _example_file_path(server_media_dir, filename)
        if file_path is not None and file_path.is_file():
            try:
                md5 = file_md5(file_path)
            except OSError:
                md5 = ""

        elements.append(
            LabeledElement(
                md5=md5,
                label="good",
                origin=origin if origin is not None else _sentinel_origin(filename),
                origin_name=origin_name,
                filename=filename,
            )
        )
    return elements


def merge_examples_into_labelset(existing: "LabelSet", examples: list[dict]) -> "LabelSet":
    """Return *existing* plus a good label for each media example not already in it.

    Purely additive: an exemplar whose identity key is already present keeps
    whatever label it has (the user may have since voted it Bad, and a
    re-supplied example must not silently flip that back), and no existing
    element is dropped.
    """
    from vtscore.datasets.labelset import LabelSet, element_key

    seen = {element_key(el) for el in existing.elements}
    added = [el for el in labeled_elements_from_examples(examples) if element_key(el) not in seen]
    if not added:
        return existing
    return LabelSet(list(existing.elements) + added, detector_meta=existing.detector_meta)


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

    For each ``type: "media"`` example, reads the file from the current
    user's ``example_media/`` cache (see
    :func:`~vtscore.security.path_validation.example_media_dir`, and
    re-fetching it from the example's ``origin`` when the cached file is
    gone) and adds it to ``good_votes``:

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

    from vtscore.security.path_validation import example_media_dir
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
    server_media_dir = example_media_dir()

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
