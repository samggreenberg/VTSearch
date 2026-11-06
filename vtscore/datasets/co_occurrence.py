"""Which faces share a photo, and which people appear together (#5587).

The co-occurrence half of Co-face Browse (``docs/plans/co-face-browse.md``).
A face crop records the photo it was cut from on itself: its converter origin
names the source file (:func:`~vtscore.datasets.source_item.converter_refs`).
So *which faces share a photo* is answerable from the Face dataset alone, with
no Image sibling loaded; the sibling is needed only to *show* the photo.

:class:`PhotoIndex` is that answer, built once from a dataset's medias:

* **A photo key per converter ref**: ``source_path`` when the ref records one,
  else ``source_file``.  ``source_path`` is the resolved file, so two photos
  that share a scan-relative name in different folders stay two photos.
* **Keys a face carries are unioned.**  A ``dupe_set`` representative (two
  byte-identical photos make byte-identical faces, which duplicate collapse
  folds into one) carries a ref per member, so it joins the copies into one
  photo, and every other face of either copy follows (a union-find over the
  keys).  The Image sibling collapsed the same two photos into one item, so
  one photo here is one item there.
* **A photo id** is an ``int``, numbered ``0, 1, …`` in the order of each
  photo's lowest face id.  It is valid for the index that issued it and means
  nothing across a rebuild.

**The photo is the one** :class:`~vtscore.datasets.source_item.SourceResolver`
**finds.**  The resolver joins a ref's ``source_file`` to the sibling's
``origin_name``, settles a name several photos share by ``source_path``, and
reaches a photo that collapse folded under another name through the
representative's member names.  Every face of one photo here therefore
resolves, against the import's own Image sibling, to one sibling item, so a
shared photo later opens as exactly one photo.  That rests on the converter
being deterministic: byte-identical faces come only from byte-identical
photos, which the sibling's own duplicate collapse merged too.

The co-occurrence questions take people as **plain face-id lists**, so this
module knows nothing of how faces were merged into people: :meth:`~PhotoIndex.photos_of`
(the photos a person is in), :meth:`~PhotoIndex.appears_with` (who shares a
photo with a person, and how often), :meth:`~PhotoIndex.shared_photos` (the
photos two people share).

Pure apart from its input and Flask-free.  Build it from a snapshot of the
medias and cache it on ``media_revision`` beside the people partition; it is
ids only and is never persisted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Iterable, Mapping, Sequence

from vtscore.datasets.source_item import converter_refs

__all__ = [
    "CoAppearance",
    "PhotoIndex",
    "SharedPhoto",
    "faces_by_photo",
    "photo_keys",
]

#: A photo key: ``("path", source_path)`` or ``("file", source_file)``.  The
#: kind is part of the key so a scan-relative name can never pass for a path.
PhotoKey = tuple[str, str]


def photo_keys(media: dict[str, Any]) -> list[PhotoKey]:
    """The photo keys *media* carries, one per converter ref, without repeats.

    ``source_path`` when the ref records one, else ``source_file``; a ref with
    neither contributes nothing.  ``[]`` for a media that is no converter
    output, which then has no photo.
    """
    keys: list[PhotoKey] = []
    for params in converter_refs(media):
        path = params.get("source_path") or ""
        if path:
            keys.append(("path", str(path)))
            continue
        name = params.get("source_file") or ""
        if name:
            keys.append(("file", str(name)))
    return list(dict.fromkeys(keys))


class _UnionFind:
    """Disjoint sets over hashable items, with path compression."""

    def __init__(self) -> None:
        self._parent: dict[Hashable, Hashable] = {}

    def find(self, item: Hashable) -> Hashable:
        parent = self._parent
        root = item
        while parent.setdefault(root, root) != root:
            root = parent[root]
        while parent[item] != root:
            parent[item], item = root, parent[item]
        return root

    def union(self, a: Hashable, b: Hashable) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self._parent[root_b] = root_a


@dataclass(frozen=True)
class CoAppearance:
    """Another group that shares photos with a person: its index in the groups, and the photos."""

    group: int
    #: The photo ids it shares with the person, ascending.
    photos: tuple[int, ...]

    @property
    def shared(self) -> int:
        """How many photos it shares with the person."""
        return len(self.photos)


@dataclass(frozen=True)
class SharedPhoto:
    """A photo two groups share, with every face cut from it (not only theirs), ascending."""

    photo: int
    faces: tuple[int, ...]


class PhotoIndex:
    """Faces keyed by the photo they were cut from; build once, ask many.

    *medias* is a dataset's medias dict (a snapshot is fine).  A face with no
    photo key (no converter ref) is in no photo, and the questions below
    ignore it, as they ignore an id that is not in *medias* at all.
    """

    def __init__(self, medias: Mapping[int, dict[str, Any]]) -> None:
        sets = _UnionFind()
        keyed: list[tuple[int, PhotoKey]] = []
        for media_id in sorted(medias):
            keys = photo_keys(medias[media_id])
            if not keys:
                continue
            for key in keys[1:]:
                sets.union(keys[0], key)
            keyed.append((media_id, keys[0]))

        # Number photos in order of their lowest face id (``keyed`` is
        # ascending by id), so the same medias always give the same ids.
        photo_of_root: dict[Hashable, int] = {}
        faces: list[list[int]] = []
        photo_of: dict[int, int] = {}
        for media_id, key in keyed:
            root = sets.find(key)
            photo = photo_of_root.get(root)
            if photo is None:
                photo = photo_of_root[root] = len(faces)
                faces.append([])
            faces[photo].append(media_id)
            photo_of[media_id] = photo
        self._photo_of = photo_of
        self._faces = [tuple(ids) for ids in faces]

    def __len__(self) -> int:
        """How many photos the faces were cut from."""
        return len(self._faces)

    @property
    def faces_by_photo(self) -> dict[int, tuple[int, ...]]:
        """Photo id to the ids of the faces cut from it, ascending; a fresh dict."""
        return dict(enumerate(self._faces))

    def photo_of(self, face_id: int) -> int | None:
        """The photo *face_id* was cut from, or ``None`` when it has none."""
        return self._photo_of.get(face_id)

    def faces_in(self, photo: int) -> tuple[int, ...]:
        """Every face cut from *photo*, ascending.  Raises ``IndexError`` for an unknown photo."""
        if not 0 <= photo < len(self._faces):
            raise IndexError(f"no photo {photo} in this index")
        return self._faces[photo]

    def photos_of(self, face_ids: Iterable[int]) -> list[int]:
        """The photos a person (*face_ids*) is in, ascending, each once."""
        return sorted(self._photo_set(face_ids))

    def appears_with(self, groups: Sequence[Sequence[int]], person: int) -> list[CoAppearance]:
        """The other groups that share a photo with ``groups[person]``.

        *groups* are people as face-id lists (the partition's order);
        *person* is an index into them.  Each other group that has a face in
        one of the person's photos is listed once, with the photos it shares;
        most shared first, ties in group order.  A group sharing nothing is
        left out, and so is the person itself.  Raises ``IndexError`` when
        *person* is out of range.
        """
        if not 0 <= person < len(groups):
            raise IndexError(f"no group {person} among {len(groups)}")
        own = self._photo_set(groups[person])
        if not own:
            return []
        found = []
        for group, face_ids in enumerate(groups):
            if group == person:
                continue
            shared = own & self._photo_set(face_ids)
            if shared:
                found.append(CoAppearance(group, tuple(sorted(shared))))
        found.sort(key=lambda co: (-co.shared, co.group))
        return found

    def shared_photos(self, group_a: Iterable[int], group_b: Iterable[int]) -> list[SharedPhoto]:
        """The photos both groups have a face in, ascending, each with every face cut from it."""
        shared = self._photo_set(group_a) & self._photo_set(group_b)
        return [SharedPhoto(photo, self._faces[photo]) for photo in sorted(shared)]

    def _photo_set(self, face_ids: Iterable[int]) -> set[int]:
        photo_of = self._photo_of
        return {photo_of[face_id] for face_id in face_ids if face_id in photo_of}


def faces_by_photo(medias: Mapping[int, dict[str, Any]]) -> dict[int, tuple[int, ...]]:
    """Photo id to the ids of the faces cut from it; see :class:`PhotoIndex`.

    One-shot: build a :class:`PhotoIndex` to ask more than this.  A face with
    no converter ref has no photo and is left out.
    """
    return PhotoIndex(medias).faces_by_photo
