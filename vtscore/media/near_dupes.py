"""Near-duplicate detection and collapsing (images + text).

Exact-duplicate collapsing (identical MD5) lives in
:mod:`vtscore.state.media_lookup`.  This module adds *near*-duplicate
collapsing: items that are the **same content** under a different encoding,
resize, recompression, or trivial edit.  That is distinct from *semantic*
similarity (what the CLAP / SigLIP / X-CLIP / E5 embeddings measure): a
perceptual hash answers "these *are* the same thing", an embedding answers
"these *mean* the same thing".

Signals (both reduce to a 64-bit hash + Hamming distance, so grouping is
uniform across modalities):

* **Images** - classic **pHash** (DCT perceptual hash).  Implemented with
  PIL (already a dependency) plus a manual numpy DCT-II, so no
  ``imagehash`` / ``scipy`` dependency is added.  Computed from each image's
  stored ``thumbnail_bytes``.
* **Text** - **SimHash** over word k-shingles, hashed with ``blake2b``
  (deterministic, unlike Python's salted ``hash()``).  Computed from
  ``media_string``.

"pHashes are close" is not transitive (A≈B, B≈C, but A≉C).  We pick a *tight*
per-pair Hamming threshold so any positive is high-confidence (missed links
are false negatives, which we tolerate), then take **connected components**
over the closeness graph as if it were an equivalence relation.

Collapsing reuses the exact-duplicate ``dupe_set`` origin structure so that
label export (which already expands ``dupe_set`` representatives into one
element per member) fans a near-dup group out to its full membership for free.

Converter outputs follow their sources rather than being hashed themselves:
:func:`follow_source_merges` collapses the faces of a multi-dataset import's
face dataset exactly where its image dataset merged the photos they were cut
from, so the two datasets agree on which photos are one.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Callable
from typing import Any

import numpy as np

from vtscore.media.provenance import SOURCE_BOX_FIELD

# Per-modality Hamming thresholds (out of 64 bits).  Deliberately tight: a
# positive should be a near-certain near-duplicate, so missed links fail
# *negative*.  Not user-exposed - there is no "equalness points" slider.
_THRESHOLDS: dict[str, int] = {"image": 4, "text": 4}

# --- Image pHash ---------------------------------------------------------
_DCT_N = 32  # downsample square edge before the DCT
_LOW_FREQ = 8  # top-left low-frequency block kept from the DCT (-> 64 bits)


def _dct_basis(n: int) -> np.ndarray:
    """Unnormalised DCT-II basis matrix.

    The overall positive scale factor is irrelevant: the hash compares each
    coefficient against the *median* of the kept block, which is scale
    invariant, so this reproduces classic ``imagehash`` pHash bits without
    needing scipy's exact normalisation.
    """
    k = np.arange(n).reshape(-1, 1)
    x = np.arange(n).reshape(1, -1)
    return np.cos(np.pi * (2 * x + 1) * k / (2 * n))


_DCT_BASIS = _dct_basis(_DCT_N)


def _pack_bits(flat: np.ndarray) -> int:
    """Pack a 1-D boolean array of length 64 (most-significant first) into an int.

    ``np.packbits`` packs MSB-first into bytes, so a big-endian
    ``int.from_bytes`` of the 8 packed bytes reproduces the old
    bit-by-bit shift loop exactly — but as one vectorised call instead of a
    64-iteration Python loop per hash.
    """
    return int.from_bytes(np.packbits(flat).tobytes(), "big")


def phash_image(thumbnail_bytes: bytes | None) -> int | None:
    """Return a 64-bit DCT perceptual hash for *thumbnail_bytes*.

    Returns ``None`` when the bytes are missing or cannot be decoded (e.g.
    SVG / corrupt thumbnails); such media are simply left ungrouped.
    """
    if not thumbnail_bytes:
        return None
    try:
        from PIL import Image  # noqa: PLC0415

        with Image.open(io.BytesIO(thumbnail_bytes)) as im:
            gray = im.convert("L").resize((_DCT_N, _DCT_N), Image.Resampling.LANCZOS)
            arr = np.asarray(gray, dtype=np.float64)
    except Exception:
        return None
    dct = _DCT_BASIS @ arr @ _DCT_BASIS.T
    block = dct[:_LOW_FREQ, :_LOW_FREQ]
    bits = block > np.median(block)
    return _pack_bits(bits.flatten())


# --- Text SimHash --------------------------------------------------------
_SHINGLE_K = 4  # word-shingle length


def simhash_text(text: str | None) -> int | None:
    """Return a 64-bit SimHash over word k-shingles of *text*.

    Falls back to single words for texts shorter than the shingle length.
    Returns ``None`` for empty/whitespace-only text.
    """
    if not text:
        return None
    words = text.lower().split()
    if not words:
        return None
    if len(words) >= _SHINGLE_K:
        shingles = {" ".join(words[i : i + _SHINGLE_K]) for i in range(len(words) - _SHINGLE_K + 1)}
    else:
        shingles = set(words)
    if not shingles:
        return None
    # Hash every shingle to an 8-byte blake2b digest, then reduce the whole
    # batch with numpy instead of two nested Python loops (shingles x 64 bits).
    # ``np.unpackbits`` on the big-endian digest bytes yields, per shingle, the
    # hash bits in most-significant-first order (bit 63 .. bit 0); mapping
    # {0,1} -> {-1,+1}, summing across shingles, and thresholding >0 reproduces
    # the classic SimHash bit vote bit-for-bit, and ``np.packbits`` re-packs it
    # MSB-first into the same 64-bit integer the shift loop produced.
    digests = b"".join(hashlib.blake2b(sh.encode("utf-8"), digest_size=8).digest() for sh in shingles)
    bits = np.unpackbits(np.frombuffer(digests, dtype=np.uint8).reshape(len(shingles), 8), axis=1)
    votes = (bits.astype(np.int32) * 2 - 1).sum(axis=0)  # order: bit 63 .. bit 0
    return _pack_bits(votes > 0)


# --- Grouping ------------------------------------------------------------
def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class _DSU:
    """Disjoint-set / union-find over an arbitrary set of hashable items."""

    def __init__(self, items: list[int]) -> None:
        self.parent: dict[int, int] = {x: x for x in items}

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _band_offsets(bands: int, total_bits: int = 64) -> list[tuple[int, int]]:
    """Split *total_bits* into *bands* near-equal ``(offset, width)`` segments."""
    base, rem = divmod(total_bits, bands)
    offsets: list[tuple[int, int]] = []
    off = 0
    for i in range(bands):
        width = base + (1 if i < rem else 0)
        offsets.append((off, width))
        off += width
    return offsets


def _union_exact(dsu: _DSU, hashes: dict[int, int]) -> list[tuple[int, int]]:
    """Union ids that share an identical hash; return one ``(id, hash)`` per
    distinct hash value.  Folding exact duplicates up front keeps a giant
    bucket of identical hashes (e.g. all-black thumbnails) from blowing up the
    pairwise band check below.
    """
    by_exact: dict[int, list[int]] = {}
    for cid, h in hashes.items():
        by_exact.setdefault(h, []).append(cid)
    distinct: list[tuple[int, int]] = []
    for h, grp in by_exact.items():
        for c in grp[1:]:
            dsu.union(grp[0], c)
        distinct.append((grp[0], h))
    return distinct


def _union_within_bands(dsu: _DSU, distinct: list[tuple[int, int]], threshold: int) -> None:
    """Union ``(id, hash)`` pairs within Hamming *threshold* via banding.

    Pigeonhole: two hashes within Hamming distance ``threshold`` must share at
    least one of ``threshold + 1`` disjoint bands exactly, so we only verify
    pairs that collide within a band rather than all O(n^2) pairs.
    """
    for off, width in _band_offsets(threshold + 1):
        mask = (1 << width) - 1
        buckets: dict[int, list[tuple[int, int]]] = {}
        for cid, h in distinct:
            buckets.setdefault((h >> off) & mask, []).append((cid, h))
        for bucket in buckets.values():
            for i in range(len(bucket)):
                ci, hi = bucket[i]
                for cj, hj in bucket[i + 1 :]:
                    if _hamming(hi, hj) <= threshold:
                        dsu.union(ci, cj)


def _connected_components(hashes: dict[int, int], threshold: int) -> list[list[int]]:
    """Group ids by Hamming-distance closeness of their 64-bit *hashes*."""
    dsu = _DSU(list(hashes.keys()))
    _union_within_bands(dsu, _union_exact(dsu, hashes), threshold)
    comps: dict[int, list[int]] = {}
    for cid in hashes:
        comps.setdefault(dsu.find(cid), []).append(cid)
    return [sorted(v) for v in comps.values()]


def _hash_for(media: dict[str, Any], media_type: str) -> int | None:
    if media_type == "image":
        return phash_image(media.get("thumbnail_bytes"))
    if media_type == "text":
        return simhash_text(media.get("media_string") or "")
    return None


def _members_of(media: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the dupe-set member dicts a media contributes to a merged group.

    A media that is already an exact ``dupe_set`` representative contributes
    its existing members (flattened, never nested); any other media
    contributes a single fresh member carrying its own ``md5`` so export can
    re-derive each near-dup member from its real content hash.
    """
    origin = media.get("origin")
    if isinstance(origin, dict) and origin.get("importer") == "dupe_set":
        return [dict(m) for m in origin.get("members", [])]
    return [
        {
            "md5": media.get("md5", ""),
            "origin": media.get("origin"),
            "origin_name": media.get("origin_name", ""),
            "filename": media.get("filename", ""),
            "category": media.get("category", ""),
        }
    ]


def _choose_representative(media_dict: dict[int, dict[str, Any]], comp: list[int]) -> int:
    """Pick the displayed representative: file_size desc, centroid asc, id asc.

    ``file_size`` is a universal fidelity proxy (works for every media type,
    unlike pixel resolution); the embedding-centroid distance breaks ties
    toward the most "typical" member; the media id is a deterministic final
    tie-break (required by the test-suite flakiness rules).
    """
    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

    embs: dict[int, np.ndarray] = {}
    for cid in comp:
        e = media_embedding(media_dict[cid])
        if e is not None:
            embs[cid] = np.asarray(e, dtype=np.float32)
    centroid = np.mean(np.stack(list(embs.values())), axis=0) if embs else None

    def sort_key(cid: int) -> tuple[int, float, int]:
        size = media_dict[cid].get("file_size") or 0
        if centroid is not None and cid in embs:
            dist = float(np.linalg.norm(embs[cid] - centroid))
        else:
            dist = float("inf")
        return (-size, dist, cid)

    return min(comp, key=sort_key)


# Report hashing progress every this many items (a whole-percent-ish stride so
# the callback overhead stays negligible on large datasets).
_HASH_PROGRESS_STRIDE = 64

# Only spin up a decode thread pool past this many images: pool setup costs more
# than it saves for the handful-of-items case (and the test suite's tiny groups).
_THREAD_MIN_IMAGES = 128


def _hash_media_items(
    media_dict: dict[int, dict[str, Any]],
    cids: list[int],
    media_type: str,
    on_tick: Callable[[], None],
) -> dict[int, int]:
    """Compute the 64-bit hash for every id in *cids*, calling *on_tick* per item.

    Image pHash is dominated by PIL thumbnail *decode* + resize, which releases
    the GIL, so past :data:`_THREAD_MIN_IMAGES` it runs on a small thread pool;
    text SimHash is numpy-vectorised and cheap enough to stay inline.  The hash
    of each item is independent, so results are order-independent and the
    grouping stays deterministic regardless of completion order.
    """
    hashes: dict[int, int] = {}
    if media_type == "image" and len(cids) >= _THREAD_MIN_IMAGES:
        import os  # noqa: PLC0415
        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

        workers = min(8, (os.cpu_count() or 2))

        def _one(cid: int) -> tuple[int, int | None]:
            return cid, phash_image(media_dict[cid].get("thumbnail_bytes"))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for cid, h in pool.map(_one, cids):
                if h is not None:
                    hashes[cid] = h
                on_tick()
    else:
        for cid in cids:
            h = _hash_for(media_dict[cid], media_type)
            if h is not None:
                hashes[cid] = h
            on_tick()
    return hashes


def _find_near_dup_groups(
    media_dict: dict[int, dict[str, Any]],
    on_progress: Callable[[int, int], None] | None = None,
) -> list[list[int]]:
    """Return the near-dup groups (size >= 2) across supported media types.

    *on_progress* (if given) is driven across the **hashing** phase — the
    dominant cost — so the bar advances while every image/text is fingerprinted,
    rather than sitting at its floor until the cheap collapse phase begins.
    """
    by_type: dict[str, list[int]] = {}
    for cid, m in media_dict.items():
        mt = m.get("media_type")
        if mt in _THRESHOLDS:
            by_type.setdefault(mt, []).append(cid)

    total = sum(len(cids) for cids in by_type.values())
    done = 0

    def tick() -> None:
        nonlocal done
        done += 1
        if on_progress and done % _HASH_PROGRESS_STRIDE == 0:
            on_progress(done, total)

    if on_progress:
        on_progress(0, total)

    groups: list[list[int]] = []
    for mt, cids in by_type.items():
        hashes = _hash_media_items(media_dict, cids, mt, tick)
        groups.extend(comp for comp in _connected_components(hashes, _THRESHOLDS[mt]) if len(comp) >= 2)
    return groups


def _fold_group(media_dict: dict[int, dict[str, Any]], rep_id: int, others: list[int]) -> None:
    """Fold *others* into *rep_id* as one ``dupe_set`` representative, and remove them."""
    members: list[dict[str, Any]] = []
    for cid in [rep_id, *others]:
        members.extend(_members_of(media_dict[cid]))
    rep = media_dict[rep_id]
    first_name = rep.get("origin_name", rep.get("filename", ""))
    rep["origin"] = {"importer": "dupe_set", "params": {"name": first_name}, "members": members}
    rep["origin_name"] = first_name
    for cid in others:
        del media_dict[cid]


def _collapse_group(media_dict: dict[int, dict[str, Any]], comp: list[int]) -> None:
    """Collapse one near-dup group into a ``dupe_set`` representative."""
    rep_id = _choose_representative(media_dict, comp)
    _fold_group(media_dict, rep_id, [c for c in comp if c != rep_id])


def collapse_near_duplicates(
    media_dict: dict[int, dict[str, Any]],
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """Collapse near-duplicate medias (images + text) into representatives.

    Runs *after* :func:`vtscore.state.media_lookup.collapse_duplicates`, so
    some group members may already be exact ``dupe_set`` representatives;
    their members are merged into the near-dup group rather than nested.  For
    each near-dup group the chosen representative (see
    :func:`_choose_representative`) keeps a ``dupe_set`` origin listing every
    member's provenance, and the rest are removed from *media_dict*.

    Only ``image`` and ``text`` media are considered; other types are left
    untouched.  Returns the number of near-dup groups collapsed.

    *on_progress* is driven across the dominant **hashing** phase inside
    :func:`_find_near_dup_groups`; the subsequent collapse is cheap dict surgery,
    so the callback only fires once more (at completion) after it finishes.
    """
    groups = _find_near_dup_groups(media_dict, on_progress=on_progress)
    for comp in groups:
        _collapse_group(media_dict, comp)
    if on_progress:
        on_progress(1, 1)
    return len(groups)


# --- Following a source dataset's merges ---------------------------------
#: How much two converter outputs' ``source_box`` es must overlap (IoU) to be
#: the same thing cut from two copies of one source: a face sits at nearly the
#: same place in a photo and in its resized copy, and two faces side by side
#: overlap far less than this.
_SOURCE_BOX_IOU = 0.5


def _box_iou(a: Any, b: Any) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _same_output_score(a: dict[str, Any], b: dict[str, Any], ref_a: dict[str, Any], ref_b: dict[str, Any]) -> float:
    """How surely *a* and *b* are one output of two copies of a source; ``0.0`` when they are not.

    Outputs that record where in the source they sat (a face's ``source_box``)
    match by overlap; outputs that record no place match by their position in
    the converter's output list (the one caption of each photo).
    """
    if ref_a.get("converter") != ref_b.get("converter"):
        return 0.0
    box_a, box_b = a.get(SOURCE_BOX_FIELD), b.get(SOURCE_BOX_FIELD)
    if box_a is not None and box_b is not None:
        iou = _box_iou(box_a, box_b)
        return iou if iou >= _SOURCE_BOX_IOU else 0.0
    if box_a is None and box_b is None and ref_a.get("converter_out_index") == ref_b.get("converter_out_index"):
        return 1.0
    return 0.0


def follow_source_merges(
    media_dict: dict[int, dict[str, Any]],
    source_medias: dict[int, dict[str, Any]],
) -> int:
    """Collapse converter outputs whose sources were merged in *source_medias*.

    The datasets of one multi-dataset import are built from one scan, so a
    face dataset holds the faces cut from every photo the image dataset beside
    it holds.  When that image dataset merged two copies of a photo
    (:func:`collapse_near_duplicates`, or the exact-duplicate pass), the faces
    must merge with them: otherwise each copy leaves its own crop of the same
    face behind.

    Each output of *media_dict* is resolved to the item of *source_medias* it
    was cut from (:class:`~vtscore.datasets.source_item.SourceResolver`, which
    finds a merged copy through its representative's ``dupe_set`` members).
    Under one source item, outputs cut from different copies collapse when
    they are the same output of each (:func:`_same_output_score`), never two
    of one copy.  The output cut from the copy the source dataset kept leads
    its group, so the kept face is the one from the photo still shown; the
    rest are folded into its ``dupe_set`` origin, so exporting it exports every
    copy, as for the photo.  An output with no counterpart (a face found in one
    copy only) stays.  Returns the number of groups collapsed.
    """
    from vtscore.datasets.source_item import SourceResolver, converter_refs  # noqa: PLC0415

    resolver = SourceResolver(source_medias)
    by_source: dict[int, list[int]] = {}
    for cid, media in media_dict.items():
        source_id = resolver.resolve(media)
        if source_id is not None:
            by_source.setdefault(source_id, []).append(cid)

    collapsed = 0
    for source_id, cids in by_source.items():
        refs = {cid: converter_refs(media_dict[cid])[0] for cid in cids}
        kept_file = source_medias[source_id].get("origin_name") or ""
        for group in _copy_groups(media_dict, cids, refs, kept_file):
            if len(group) > 1:
                _fold_group(media_dict, group[0], group[1:])
                collapsed += 1
    return collapsed


def _copy_groups(
    media_dict: dict[int, dict[str, Any]],
    cids: list[int],
    refs: dict[int, dict[str, Any]],
    kept_file: str,
) -> list[list[int]]:
    """Group the outputs *cids* of one source item, each group one output of every copy.

    *refs* holds each output's converter params; *kept_file* names the copy the
    source dataset kept.  Its outputs come first, so each leads its group; the
    rest join the group they best match, largest first.
    """
    files = {cid: refs[cid].get("source_file") or "" for cid in cids}
    if len(set(files.values())) < 2:
        return []  # every output came from one copy: nothing was merged here
    order = sorted(cids, key=lambda c: (files[c] != kept_file, -(media_dict[c].get("file_size") or 0), c))
    groups: list[list[int]] = []
    for cid in order:
        best: list[int] | None = None
        best_score = 0.0
        for group in groups:
            if any(files[m] == files[cid] for m in group):
                continue  # never two outputs of one copy
            score = _same_output_score(media_dict[cid], media_dict[group[0]], refs[cid], refs[group[0]])
            if score > best_score:
                best, best_score = group, score
        if best is None:
            groups.append([cid])
        else:
            best.append(cid)
    return groups
