"""Seed good votes from a model's media examples.

Provides :func:`seed_good_votes_from_examples` which reads example media
files, matches them to loaded dataset medias by MD5, and votes them good.
"""

from __future__ import annotations


def seed_good_votes_from_examples(examples: list[dict]) -> int:  # noqa: C901
    """Seed good votes from a model's media examples.

    For each ``type: "media"`` example, reads the file from
    ``data/example_media/`` and adds it to ``good_votes``:

    * **Match by MD5** — if a loaded media has the same content hash,
      that media is voted good (keeping its original dataset origin).
    * **No match** — the example file is embedded using the dataset's
      embedder, inserted into the ``medias`` dict as a new item with
      an ``example_media`` origin, and voted good.  This makes it
      available for training (its embedding is in the medias snapshot)
      and for label export (LabelSet picks it up from medias + votes).

    Returns the number of example entries successfully seeded.
    """
    import hashlib

    from vtscore.config import DATA_DIR
    from vtsearch.state import (
        _state_lock,
        apply_label,
        build_media_lookup,
        medias,
        next_media_id,
        snapshot_medias,
    )

    media_examples = [
        ex for ex in examples if isinstance(ex, dict) and ex.get("type") == "media" and ex.get("value", "").strip()
    ]
    if not media_examples:
        return 0

    snap = snapshot_medias()
    if not snap:
        return 0

    _, md5_lookup, _ = build_media_lookup(snap)
    server_media_dir = DATA_DIR / "example_media"

    # Determine the embedder and media type from the loaded dataset so we
    # can embed example files that aren't already in the dataset.
    first_media = next(iter(snap.values()))
    dataset_media_type = first_media.get("type", "audio")
    dataset_embedder_name = first_media.get("embedder", "")
    embedder = None  # lazily loaded only when needed

    seeded = 0
    for ex in media_examples:
        filename = ex["value"].strip()
        file_path = server_media_dir / filename
        # Prevent directory traversal
        try:
            file_path.resolve().relative_to(server_media_dir.resolve())
        except ValueError:
            continue
        if not file_path.is_file():
            continue

        file_bytes = file_path.read_bytes()
        file_md5 = hashlib.md5(file_bytes).hexdigest()
        cids = md5_lookup.get(file_md5, [])

        if cids:
            # Example matches existing dataset media — just vote good.
            for cid in cids:
                apply_label(cid, "good")
            seeded += 1
        else:
            # Example is NOT in the dataset — embed and insert as new media.
            if embedder is None:
                from vtscore.media import embedders_for_type, get_embedder

                if dataset_embedder_name:
                    try:
                        embedder = get_embedder(dataset_embedder_name)
                    except KeyError:
                        pass
                if embedder is None:
                    avail = embedders_for_type(dataset_media_type)
                    embedder = avail[0] if avail else None
                if embedder is None:
                    # No embedder available; skip remaining examples.
                    continue

            from vtscore.media.embedder import media_from_path  # noqa: PLC0415

            embedding = embedder.embed_media(media_from_path(file_path))
            if embedding is None:
                continue

            with _state_lock:
                new_id = next_media_id(medias)
                medias[new_id] = {
                    "id": new_id,
                    "type": dataset_media_type,
                    "embedder": dataset_embedder_name,
                    "md5": file_md5,
                    "embedding": embedding,
                    "media_bytes": file_bytes,
                    "filename": filename,
                    "file_size": len(file_bytes),
                    "category": "",
                    "origin": {
                        "importer": "example_media",
                        "params": {"filename": filename},
                    },
                    "origin_name": filename,
                }

            apply_label(new_id, "good")
            seeded += 1

    return seeded
