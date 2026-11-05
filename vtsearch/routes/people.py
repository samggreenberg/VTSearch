"""The ``people`` blueprint: a Face dataset's people, and who appears with whom (#5589).

The app-tier slice of Co-face Browse (``docs/plans/co-face-browse.md``).  It
serves the two ids-only structures the library derives from a Face dataset to
the People view:

* the **people partition** (:mod:`vtscore.datasets.people`, #5586): the
  dataset's face crops merged into people, fit at build or on the first ask;
* the **photo index** (:mod:`vtscore.datasets.co_occurrence`, #5587): which
  faces were cut from one photo, read off each face's own converter refs, so
  it needs no Image sibling loaded.  It is built per request from the faces,
  as cheaply as listing them.

Routes, all on the requesting Face dataset (``X-Dataset-Id``):

* ``GET /api/people``: the partition's status and its people.  It answers
  ``building`` while a fit runs, and the client polls it, as
  ``POST /api/medias/similarity-order`` is polled.
* ``POST /api/people/build``: the same answer, fitting again with ``force``
  (Rebuild people).
* ``GET /api/people/<id>``: one person's faces, photos and who appears with
  them.
* ``GET /api/people/<id>/with/<other>``: the photos two people share, each
  with every face in it and, when it resolves, the photo as an item of the
  Image sibling.  Only this route needs the sibling.

**A person's id is a rank and holds only within one partition.**  A fit can
reorder people, and removing faces can drop one.  Every answer carries the
partition's identity (:func:`partition_identity`), and the per-person routes
take it back as ``?partition=``.  A client holding an id from another
partition gets a 409 ``people_changed``, which it cures by listing the people
again, never an answer about someone else.  The identity is a digest of the
partition as served, so a forced fit that finds the same people keeps it and
the client's ids stay good.

The per-person routes read the stored partition and never start a fit.  While
a forced fit runs they keep answering from the partition it will replace,
under that partition's identity; once it lands, an id read under the old one
is a 409 if the people moved.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, NoReturn

from flask_smorest import Blueprint, abort

from vtscore.datasets import source_item
from vtscore.datasets.co_occurrence import PhotoIndex
from vtscore.datasets.people import NothingToPartition, people_for, request_people, wants_people
from vtscore.media.provenance import SOURCE_BOX_FIELD
from vtsearch.routes._context import MISSING_DATASET_HEADER, require_dataset_header
from vtsearch.routes._source import abort_source_not_loaded, source_locator
from vtsearch.schemas.people import (
    PeopleBuildRequestSchema,
    PeopleListSchema,
    PersonQuerySchema,
    PersonSchema,
    SharedPhotosSchema,
)
from vtsearch.state import get_active_context, snapshot_medias

if TYPE_CHECKING:
    from vtscore.datasets.people import PeoplePartition

people_bp = Blueprint(
    "people",
    __name__,
    description="A Face dataset's people, and the people who appear with each in the same photos.",
)

_NO_PEOPLE = (
    "The dataset has no people to find (``error_code: no_people``): it is not a Face dataset, "
    "or its faces have no vectors."
)
_PERSON_CONFLICT = (
    "The dataset has no people to find (``error_code: no_people``); its people are not found yet "
    "(``people_not_ready``: ask ``GET /api/people``); or the people were found again since ``partition`` "
    "was read (``people_changed``: list them again)."
)
_UNKNOWN_PERSON = "No person has this id in the partition."


def partition_identity(partition: PeoplePartition) -> str:
    """A short digest of *partition* as served: its people, unplaced faces and stamps.

    Two partitions with one identity give every person the same id, so an id
    read under one is good under the other.  A fit that finds the same people
    keeps the identity; one that moves anyone, or a removed face, changes it.
    """
    body = json.dumps(partition.to_payload(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def _abort_conflict(message: str, error_code: str) -> NoReturn:
    """A 409 the client branches on by ``error_code``: ``no_people``, ``people_not_ready``, ``people_changed``."""
    abort(409, message=message, error_code=error_code)
    raise RuntimeError("unreachable (abort() raises)")  # for type narrowing


def _people_answer(*, force: bool) -> dict[str, Any]:
    """What ``GET /api/people`` and ``POST /api/people/build`` answer, from :func:`request_people`."""
    try:
        answer = request_people(get_active_context(), force=force)
    except NothingToPartition as exc:
        _abort_conflict(str(exc), "no_people")
    if answer["status"] != "ready":
        return answer
    snap = snapshot_medias()
    partition = answer["people"].restricted_to(snap.keys())
    index = PhotoIndex(snap)
    return {
        "status": "ready",
        "partition": partition_identity(partition),
        "people": [
            {
                "id": rank,
                "face_count": len(faces),
                "photo_count": len(index.photos_of(faces)),
                "representative": faces[0],
            }
            for rank, faces in enumerate(partition.people, start=1)
        ],
        "unplaced": len(partition.unplaced),
        "unplaced_ids": list(partition.unplaced),
        "cutoff": partition.cutoff,
        "compat": partition.compat,
        "method": partition.method,
        "linkage": partition.linkage,
    }


def _served_partition(asked: str) -> tuple[PeoplePartition, dict[int, dict[str, Any]]]:
    """The stored partition over the live faces, and those faces, if it is the one *asked* names.

    Never starts a fit: one not stored yet is a 409 ``people_not_ready``, and
    one whose identity is not *asked* a 409 ``people_changed``.
    """
    ctx = get_active_context()
    if not wants_people(ctx):
        _abort_conflict("People are found in Face datasets only.", "no_people")
    stored = people_for(ctx, fit=False)
    if stored is None:
        _abort_conflict("This dataset's people are not found yet; ask GET /api/people.", "people_not_ready")
    snap = snapshot_medias()
    partition = stored.restricted_to(snap.keys())
    if partition_identity(partition) != asked:
        _abort_conflict("The people were found again since this partition was read; list them again.", "people_changed")
    return partition, snap


def _faces_of(partition: PeoplePartition, person_id: int) -> tuple[int, ...]:
    """Person *person_id*'s faces (representative first); 404 for an id the partition has no person at."""
    if not 1 <= person_id <= len(partition.people):
        abort(404, message=_UNKNOWN_PERSON)
    return partition.people[person_id - 1]


def _box(media: dict[str, Any]) -> list[float] | None:
    box = media.get(SOURCE_BOX_FIELD)
    return [float(v) for v in box] if box is not None else None


def _source_of(
    locator: source_item.SourceLocator, faces: tuple[int, ...], snap: dict[int, dict[str, Any]]
) -> tuple[str | None, int | None]:
    """The sibling item a photo (its *faces*) resolves to, or ``(None, None)``; 409 for an unloaded sibling.

    Every face of one photo resolves to the same item
    (:mod:`vtscore.datasets.co_occurrence`), so the first that resolves names it.
    """
    for face_id in faces:
        found = locator.locate(snap[face_id])
        if found.status == source_item.SIBLING_NOT_LOADED:
            abort_source_not_loaded(found.dataset_id)
        if found.status == source_item.RESOLVED:
            return found.dataset_id, found.media_id
    return None, None


@people_bp.route("/api/people", methods=["GET"])
@people_bp.response(200, PeopleListSchema)
@people_bp.alt_response(409, description=_NO_PEOPLE)
@people_bp.alt_response(400, description=MISSING_DATASET_HEADER)
@require_dataset_header
def list_people():
    """List the dataset's people: each one's rank, face and photo counts, and representative (#5589).

    A Face dataset's people are its face crops merged at the face ball's cut
    (:mod:`vtscore.datasets.people`), fit when the dataset was built, or on the
    first ask for one built before that.  That ask answers ``building`` with
    the fit's progress, and the client asks again until it is ``ready``.  A
    face space the ball has no cut for answers ``not_calibrated`` and is never
    fit.  ``partition`` names this partition; pass it back to name a person.
    """
    return _people_answer(force=False)


@people_bp.route("/api/people/build", methods=["POST"])
@people_bp.arguments(PeopleBuildRequestSchema, required=False)
@people_bp.response(200, PeopleListSchema)
@people_bp.alt_response(409, description=_NO_PEOPLE)
@people_bp.alt_response(400, description=MISSING_DATASET_HEADER)
@require_dataset_header
def build_people(body: dict | None = None):
    """Find the dataset's people, again with ``force`` (Rebuild people).

    Answers as ``GET /api/people`` does.  Without ``force`` it is that ask; with
    it, a stored partition is fit again and the answer is ``building`` (the
    running fit's, when one already is).  Poll ``GET /api/people``.
    """
    return _people_answer(force=bool((body or {}).get("force")))


@people_bp.route("/api/people/<int:person_id>", methods=["GET"])
@people_bp.arguments(PersonQuerySchema, location="query")
@people_bp.response(200, PersonSchema)
@people_bp.alt_response(404, description=_UNKNOWN_PERSON)
@people_bp.alt_response(409, description=_PERSON_CONFLICT)
@people_bp.alt_response(400, description=MISSING_DATASET_HEADER)
@require_dataset_header
def get_person(query: dict, person_id: int):
    """One person: their faces, the photos they are in, and the people who appear with them.

    ``appears_with`` lists every other person with a face in one of this
    person's photos, with how many photos the two share, most first.  Needs no
    Image sibling: which faces share a photo is read off the faces.
    """
    partition, snap = _served_partition(query["partition"])
    faces = _faces_of(partition, person_id)
    index = PhotoIndex(snap)
    own: dict[int, list[int]] = {}
    for face_id in faces:
        photo = index.photo_of(face_id)
        if photo is not None:
            own.setdefault(photo, []).append(face_id)
    return {
        "partition": query["partition"],
        "id": person_id,
        "representative": faces[0],
        "face_ids": list(faces),
        "photos": [{"photo": photo, "face_ids": own[photo]} for photo in sorted(own)],
        "appears_with": [
            {"id": co.group + 1, "shared": co.shared, "representative": partition.people[co.group][0]}
            for co in index.appears_with(partition.people, person_id - 1)
        ],
    }


@people_bp.route("/api/people/<int:person_id>/with/<int:other_id>", methods=["GET"])
@people_bp.arguments(PersonQuerySchema, location="query")
@people_bp.response(200, SharedPhotosSchema)
@people_bp.alt_response(404, description=_UNKNOWN_PERSON)
@people_bp.alt_response(
    409,
    description=(
        f"{_PERSON_CONFLICT} Or the photos' Image dataset is registered but not loaded "
        "(``error_code: source_not_loaded``); ``dataset_id`` names it for ``POST /api/datasets/registry/<id>/load``."
    ),
)
@people_bp.alt_response(400, description=MISSING_DATASET_HEADER)
@require_dataset_header
def shared_photos(query: dict, person_id: int, other_id: int):
    """The photos two people share, each with every face cut from it.

    Each face carries its box in the photo and its person (null when it is
    unplaced), so a photo can be drawn with everyone in it outlined.  The photo
    itself is named as an item of the Image dataset the same import made
    (``source_dataset_id`` / ``source_media_id``), found as
    ``GET /api/medias/<id>/source`` finds a face's.  That dataset must be
    loaded; when it is not, the 409 names it.  One the user cannot read leaves
    both null.
    """
    partition, snap = _served_partition(query["partition"])
    faces_a = _faces_of(partition, person_id)
    faces_b = _faces_of(partition, other_id)
    index = PhotoIndex(snap)
    person_of = partition.person_of()
    locator = source_locator()
    photos = []
    for shared in index.shared_photos(faces_a, faces_b):
        dataset_id, media_id = _source_of(locator, shared.faces, snap)
        photos.append(
            {
                "photo": shared.photo,
                "source_dataset_id": dataset_id,
                "source_media_id": media_id,
                "faces": [
                    {
                        "media_id": face_id,
                        "box": _box(snap[face_id]),
                        "person": person_of[face_id] + 1 if face_id in person_of else None,
                    }
                    for face_id in shared.faces
                ],
            }
        )
    return {"partition": query["partition"], "id": person_id, "other": other_id, "photos": photos}
