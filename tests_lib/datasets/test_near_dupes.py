"""Tests for near-duplicate detection and collapsing (images + text).

Covers the pure library-tier logic in :mod:`vtscore.media.near_dupes`:
pHash / SimHash, connected-components grouping over a tight Hamming
threshold, representative selection (file_size -> centroid -> id), the
``dupe_set`` collapse (including merging pre-existing exact dupe_sets), that
label export fans a near-dup representative out to its full membership
with each member's own md5 (the bug the feature also fixes), and that
converter outputs (faces) follow the merges of the photos they were cut from.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from vtscore.datasets.labelset import LabelSet
from vtscore.state import (
    collapse_duplicates,
    collapse_near_duplicates,
    phash_image,
    simhash_text,
)
from vtscore.datasets.source_item import SourceResolver
from vtscore.media.near_dupes import _hamming, follow_source_merges

# A realistic document-length paragraph.  Near-dup text detection (tight
# SimHash threshold) is meant to catch reformatted/re-encoded copies of a
# document, not paraphrases: whitespace/punctuation reflows leave the hash
# essentially unchanged, while genuine rewrites move well past the threshold.
_DOC = (
    "Machine learning models require careful evaluation across diverse datasets "
    "to ensure that reported accuracy reflects genuine generalization rather than "
    "overfitting to a particular benchmark distribution that may not represent "
    "the real world conditions encountered after deployment in production systems"
)
# Same document, re-cased and reflowed with different whitespace/newlines (a
# near-dup that exact-MD5 dedup would miss).  SimHash lowercases and splits on
# whitespace, so an all-tokens-preserved reflow hashes identically.
_DOC_REFLOWED = "\n    ".join(_DOC.upper().split())


# --- helpers -------------------------------------------------------------
def _ph(thumbnail_bytes: bytes) -> int:
    h = phash_image(thumbnail_bytes)
    assert h is not None
    return h


def _sh(text: str) -> int:
    h = simhash_text(text)
    assert h is not None
    return h


def _img_bytes(arr: np.ndarray) -> bytes:
    """Encode a uint8 HxW(xC) array as PNG bytes."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _photo_arr(seed: int, size: int = 128) -> np.ndarray:
    """A structured, photo-like RGB array (low-frequency noise upsampled).

    Smooth gradients are a pathological pHash case (their low-freq DCT
    coefficients all cluster near the median, so trivial perturbations flip
    many bits).  Upsampled blocky noise has the broad coefficient spread real
    photos do, so its pHash is stable under recompression/resize.
    """
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (8, 8), dtype=np.uint8)
    gray = np.asarray(Image.fromarray(small).resize((size, size), Image.Resampling.BICUBIC))
    return np.stack([gray] * 3, axis=-1)


def _photo(seed: int) -> bytes:
    """PNG bytes of a structured photo-like image."""
    return _img_bytes(_photo_arr(seed))


def _recompress(arr: np.ndarray, quality: int = 70) -> bytes:
    """Re-encode an array as JPEG (a lossy near-duplicate of the original)."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _make_image_media(cid, thumb, *, md5=None, file_size=100, dim=4):
    rng = np.random.default_rng(cid)
    return {
        "id": cid,
        "media_type": "image",
        "duration": 0,
        "file_size": file_size,
        "md5": md5 or f"md5_{cid}",
        "embeddings": {"siglip": rng.standard_normal(dim).astype(np.float32)},
        "embedder": "siglip",
        "thumbnail_bytes": thumb,
        "filename": f"img_{cid}.png",
        "category": "cat",
        "origin": {"importer": "server_folder", "params": {"path": f"/data/{cid}"}},
        "origin_name": f"img_{cid}.png",
    }


def _make_text_media(cid, text, *, md5=None, file_size=100):
    return {
        "id": cid,
        "media_type": "text",
        "duration": 0,
        "file_size": file_size,
        "md5": md5 or f"md5_{cid}",
        "embeddings": {"e5": np.zeros(4, dtype=np.float32)},
        "embedder": "e5",
        "media_string": text,
        "filename": f"text_{cid}.txt",
        "category": "cat",
        "origin": {"importer": "server_folder", "params": {"path": f"/data/{cid}"}},
        "origin_name": f"text_{cid}.txt",
    }


# --- pHash ---------------------------------------------------------------
class TestPhash:
    def test_deterministic_and_64_bit(self):
        b = _photo(1)
        assert phash_image(b) == phash_image(b)
        assert 0 <= _ph(b) < (1 << 64)

    def test_none_for_missing_or_undecodable(self):
        assert phash_image(None) is None
        assert phash_image(b"") is None
        assert phash_image(b"not an image") is None

    def test_recompressed_image_is_near(self):
        """A re-encoded (JPEG) copy stays within the tight image threshold."""
        arr = _photo_arr(2)
        assert _hamming(_ph(_img_bytes(arr)), _ph(_recompress(arr, 70))) <= 4

    def test_different_images_are_far(self):
        assert _hamming(_ph(_photo(3)), _ph(_photo(4))) > 4


# --- SimHash -------------------------------------------------------------
class TestSimhash:
    def test_deterministic(self):
        t = "the quick brown fox jumps over the lazy dog"
        assert simhash_text(t) == simhash_text(t)

    def test_none_for_empty(self):
        assert simhash_text(None) is None
        assert simhash_text("   ") is None

    def test_reflowed_document_is_close(self):
        """A whitespace/punctuation-reflowed copy stays within the threshold."""
        assert _hamming(_sh(_DOC), _sh(_DOC_REFLOWED)) <= 3

    def test_unrelated_text_is_far(self):
        other = (
            "An entirely different essay about coastal fermentation traditions and "
            "the culinary history of preserved seafood across the northern islands"
        )
        assert _hamming(_sh(_DOC), _sh(other)) > 3


# --- grouping / collapse -------------------------------------------------
class TestCollapseNearDuplicates:
    def test_groups_near_dup_images(self):
        arr = _photo_arr(10)
        # A JPEG recompression of the same photo is a near-duplicate; a
        # different photo (seed 11) is not.
        media = {
            1: _make_image_media(1, _img_bytes(arr)),
            2: _make_image_media(2, _recompress(arr, 75)),
            3: _make_image_media(3, _photo(11)),
        }
        count = collapse_near_duplicates(media)
        assert count == 1
        # 1 & 2 merged; 3 stands alone.
        assert len(media) == 2
        assert 3 in media

    def test_representative_is_largest_file_size(self):
        thumb = _photo(12)
        media = {
            1: _make_image_media(1, thumb, file_size=100),
            2: _make_image_media(2, thumb, file_size=500),  # largest -> representative
            3: _make_image_media(3, thumb, file_size=200),
        }
        collapse_near_duplicates(media)
        assert set(media.keys()) == {2}
        rep = media[2]
        assert rep["origin"]["importer"] == "dupe_set"
        members = rep["origin"]["members"]
        assert len(members) == 3
        # Representative listed first, and every member carries its own md5.
        assert members[0]["md5"] == "md5_2"
        assert {m["md5"] for m in members} == {"md5_1", "md5_2", "md5_3"}

    def test_non_supported_types_untouched(self):
        media = {
            1: {"id": 1, "media_type": "audio", "md5": "x", "file_size": 1, "media_bytes": b"\x00"},
            2: {"id": 2, "media_type": "audio", "md5": "y", "file_size": 1, "media_bytes": b"\x00"},
        }
        assert collapse_near_duplicates(media) == 0
        assert len(media) == 2

    def test_groups_near_dup_text(self):
        media = {
            1: _make_text_media(1, _DOC),
            2: _make_text_media(2, _DOC_REFLOWED),  # reflowed near-dup of 1
            3: _make_text_media(3, "an entirely different document discussing macroeconomic monetary policy at length"),
        }
        count = collapse_near_duplicates(media)
        assert count == 1
        assert 3 in media
        assert len(media) == 2

    def test_merges_existing_exact_dupe_set(self):
        """Near-dup over a group where one member is already an exact dupe_set.

        The exact dupe_set's members are flattened into the near-dup group's
        member list (never nested), and each carries its own md5.
        """
        thumb = _photo(20)
        media = {
            1: _make_image_media(1, thumb, md5="A", file_size=100),
            2: _make_image_media(2, thumb, md5="A", file_size=100),  # exact dup of 1
            3: _make_image_media(3, thumb, md5="B", file_size=500),  # near-dup, bigger
        }
        # Exact dedup first (collapses 1 & 2 into rep 1).
        collapse_duplicates(media)
        assert set(media.keys()) == {1, 3}
        # Near-dup merge: 3 (largest) becomes representative of {1(=exact set), 3}.
        collapse_near_duplicates(media)
        assert set(media.keys()) == {3}
        members = media[3]["origin"]["members"]
        # 3 members total: the two original exact dups (md5 A x2) + media 3 (md5 B).
        assert len(members) == 3
        md5s = sorted(m["md5"] for m in members)
        assert md5s == ["A", "A", "B"]
        # No nested dupe_set origins leaked into the member list.
        assert all(
            not (isinstance(m.get("origin"), dict) and m["origin"].get("importer") == "dupe_set") for m in members
        )


class TestNearDupeProgress:
    def test_progress_reported_during_hashing(self):
        """Progress advances across the (dominant) hashing phase, not just the
        cheap collapse — the fix for the "dead air" progress bar.

        Uses > _THREAD_MIN_IMAGES distinct photos so both the decode thread pool
        and the mid-hash progress stride (every 64 items) are exercised.
        """
        media = {cid: _make_image_media(cid, _photo(cid)) for cid in range(1, 201)}
        calls: list[tuple[int, int]] = []
        collapse_near_duplicates(media, on_progress=lambda cur, tot: calls.append((cur, tot)))

        # At least one tick fired *during* hashing (current > 0 against the
        # per-item total of 200), proving the bar moves while fingerprinting
        # rather than sitting at its floor until collapse.
        mid_hash = [(c, t) for c, t in calls if t == 200 and c > 0]
        assert mid_hash, f"expected mid-hash progress ticks, got {calls}"
        assert max(c for c, _ in mid_hash) >= 128  # advanced well into the set
        # current never exceeds total within the hashing phase.
        assert all(c <= t for c, t in calls)

    def test_threaded_and_inline_hashing_agree(self, monkeypatch):
        """The decode thread pool yields the exact same grouping as the inline
        path — each item's hash is independent of completion order.
        """
        import copy  # noqa: PLC0415

        from vtscore.media import near_dupes  # noqa: PLC0415

        arr = _photo_arr(500)
        base = {
            1: _make_image_media(1, _img_bytes(arr), file_size=500),  # PNG
            2: _make_image_media(2, _recompress(arr, 75), file_size=100),  # near-dup JPEG
        }
        for cid in range(3, 60):
            base[cid] = _make_image_media(cid, _photo(cid))

        inline = copy.deepcopy(base)
        monkeypatch.setattr(near_dupes, "_THREAD_MIN_IMAGES", 10**9)  # force the inline path
        n_inline = collapse_near_duplicates(inline)

        threaded = copy.deepcopy(base)
        monkeypatch.setattr(near_dupes, "_THREAD_MIN_IMAGES", 1)  # force the thread pool
        n_threaded = collapse_near_duplicates(threaded)

        assert n_inline == n_threaded
        assert set(inline.keys()) == set(threaded.keys())
        assert 2 not in inline  # the JPEG near-dup collapsed into the larger PNG


# --- export expansion ----------------------------------------------------
class TestNearDupeExportExpansion:
    def test_export_expands_with_per_member_md5(self):
        thumb = _photo(30)
        media = {
            1: _make_image_media(1, thumb, md5="AAA", file_size=500),
            2: _make_image_media(2, thumb, md5="BBB", file_size=100),
        }
        collapse_near_duplicates(media)
        (rep_id,) = media.keys()
        ls = LabelSet.from_clips_and_votes(media, {rep_id: None}, {})
        # One element per member, each with its OWN md5 (not the rep's md5
        # for every entry - the bug this feature fixes).
        assert len(ls.elements) == 2
        assert {e.md5 for e in ls.elements} == {"AAA", "BBB"}
        assert all(e.label == "good" for e in ls.elements)


# --- following a source dataset's merges ----------------------------------
_LEFT = [0.10, 0.20, 0.30, 0.50]
_LEFT_IN_COPY = [0.11, 0.21, 0.31, 0.51]  # the same face, a resize's jitter away
_RIGHT = [0.60, 0.20, 0.80, 0.50]


def _face(cid, photo, box, *, file_size=50, out_index=0, converter="image2face", media_type="face"):
    """A converter output cut from *photo*, as the converter runner emits it."""
    media = {
        "id": cid,
        "media_type": media_type,
        "file_size": file_size,
        "md5": f"face_md5_{cid}",
        "filename": f"{photo}\u2192face_{out_index}.jpg",
        "category": "cat",
        "origin": {
            "importer": "converter",
            "params": {
                "converter": converter,
                "source_file": photo,
                "source_path": f"/data/{photo}",
                "converter_out_index": str(out_index),
            },
        },
        "origin_name": f"{photo}\u2192face_{out_index}.jpg",
        "media_path": f"/data/{photo}",
    }
    if box is not None:
        media["source_box"] = box
    return media


def _merged_photos():
    """Three photos, the second a smaller recompressed copy of the first, near-dup collapsed."""
    arr = _photo_arr(30)
    photos = {
        1: _make_image_media(1, _img_bytes(arr), file_size=500),
        2: _make_image_media(2, _recompress(arr, 75), file_size=200),
        3: _make_image_media(3, _photo(31)),
    }
    collapse_near_duplicates(photos)
    assert set(photos) == {1, 3}, "the fixture needs photo 2 merged into photo 1"
    return photos


def _member_md5s(media):
    return [m["md5"] for m in media["origin"]["members"]]


class TestFollowSourceMerges:
    def test_the_face_of_a_merged_copy_folds_into_the_kept_photos_face(self):
        photos = _merged_photos()
        faces = {
            11: _face(11, "img_1.png", _LEFT),
            12: _face(12, "img_2.png", _LEFT_IN_COPY, file_size=900),
            13: _face(13, "img_3.png", _LEFT),
        }

        assert follow_source_merges(faces, photos) == 1

        # The face from the photo the image dataset kept stays, larger crop or not.
        assert set(faces) == {11, 13}
        assert faces[11]["origin"]["importer"] == "dupe_set"
        assert _member_md5s(faces[11]) == ["face_md5_11", "face_md5_12"]
        assert faces[11]["origin"]["members"][0]["origin"]["params"]["source_file"] == "img_1.png"
        # ...and still finds the photo it was cut from.
        assert SourceResolver(photos).resolve(faces[11]) == 1

    def test_each_face_folds_into_its_counterpart(self):
        photos = _merged_photos()
        faces = {
            11: _face(11, "img_1.png", _LEFT, out_index=0),
            12: _face(12, "img_1.png", _RIGHT, out_index=1),
            21: _face(21, "img_2.png", [0.61, 0.21, 0.81, 0.51], out_index=0),
            22: _face(22, "img_2.png", _LEFT_IN_COPY, out_index=1),
        }

        assert follow_source_merges(faces, photos) == 2

        assert set(faces) == {11, 12}
        assert _member_md5s(faces[11]) == ["face_md5_11", "face_md5_22"], "matched by place, not by index"
        assert _member_md5s(faces[12]) == ["face_md5_12", "face_md5_21"]

    def test_two_faces_of_one_photo_never_fold_together(self):
        photos = _merged_photos()
        faces = {
            11: _face(11, "img_1.png", _LEFT, out_index=0),
            12: _face(12, "img_1.png", _LEFT_IN_COPY, out_index=1),  # overlaps 11 as much as a copy would
            21: _face(21, "img_2.png", _LEFT),
        }

        assert follow_source_merges(faces, photos) == 1

        assert set(faces) == {11, 12}
        assert _member_md5s(faces[11]) == ["face_md5_11", "face_md5_21"], "the best overlap wins"
        assert faces[12]["origin"]["importer"] == "converter"

    def test_a_face_found_in_one_copy_only_stays(self):
        photos = _merged_photos()
        faces = {
            11: _face(11, "img_1.png", _LEFT),
            21: _face(21, "img_2.png", _LEFT_IN_COPY, out_index=0),
            22: _face(22, "img_2.png", _RIGHT, out_index=1),
        }

        assert follow_source_merges(faces, photos) == 1

        assert set(faces) == {11, 22}

    def test_nothing_merged_leaves_the_faces_alone(self):
        photos = {1: _make_image_media(1, _photo(40)), 2: _make_image_media(2, _photo(41))}
        collapse_near_duplicates(photos)
        assert set(photos) == {1, 2}
        faces = {11: _face(11, "img_1.png", _LEFT), 21: _face(21, "img_2.png", _LEFT)}
        before = {cid: dict(m) for cid, m in faces.items()}

        assert follow_source_merges(faces, photos) == 0
        assert faces == before

    def test_follows_an_exact_duplicate_merge_too(self):
        """Byte-identical photos collapse on every import; faces that differ by a byte still follow."""
        thumb = _photo(42)
        photos = {
            1: _make_image_media(1, thumb, md5="same"),
            2: _make_image_media(2, thumb, md5="same"),
        }
        collapse_duplicates(photos)
        assert set(photos) == {1}
        faces = {11: _face(11, "img_1.png", _LEFT), 21: _face(21, "img_2.png", _LEFT)}

        assert follow_source_merges(faces, photos) == 1
        assert set(faces) == {11}

    def test_outputs_without_a_place_match_by_position(self):
        """A converter that records no ``source_box`` (one caption per photo) matches by output index."""
        photos = _merged_photos()
        captions = {
            11: _face(11, "img_1.png", None, converter="image2text", media_type="text"),
            21: _face(21, "img_2.png", None, converter="image2text", media_type="text"),
        }

        assert follow_source_merges(captions, photos) == 1
        assert set(captions) == {11}
