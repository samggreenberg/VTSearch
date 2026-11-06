"""Schemas for the People routes: a Face dataset's people, and who appears with whom (#5589).

``vtsearch/routes/people.py``:

* ``GET  /api/people`` -> :class:`PeopleListSchema`
* ``POST /api/people/build`` -> :class:`PeopleBuildRequestSchema` -> :class:`PeopleListSchema`
* ``GET  /api/people/<id>`` -> :class:`PersonQuerySchema` -> :class:`PersonSchema`
* ``GET  /api/people/<id>/with/<other>`` -> :class:`PersonQuerySchema` -> :class:`SharedPhotosSchema`

A person's ``id`` is its rank in one partition, ``1`` for the person with the
most faces.  It means nothing under another partition, so every answer carries
the partition's identity (``partition``), and the two per-person routes take it
back as a query parameter.
"""

from __future__ import annotations

from marshmallow import Schema, fields, validate

_PARTITION = (
    "The identity of the partition the people were read from.  A person's ``id`` holds only under it: "
    "pass it back as ``?partition=`` to name a person."
)


class PersonSummarySchema(Schema):
    """One row of ``GET /api/people``: a person, without their faces."""

    id = fields.Integer(
        required=True,
        metadata={"description": "The person's rank in this partition: 1 for the person with the most faces."},
    )
    face_count = fields.Integer(required=True, metadata={"description": "How many faces the person has."})
    photo_count = fields.Integer(
        required=True,
        metadata={"description": "How many photos the person's faces were cut from."},
    )
    representative = fields.Integer(
        required=True,
        metadata={"description": "The face that stands for the person: their face with the highest Face Quality."},
    )


class PeopleListSchema(Schema):
    """Response for ``GET /api/people`` and ``POST /api/people/build``: the people, or why there are none yet."""

    status = fields.String(
        required=True,
        validate=validate.OneOf(["ready", "building", "error", "not_calibrated"]),
        metadata={
            "description": (
                "``ready``: the people are listed. ``building``: they are being found; ask ``GET /api/people`` "
                "again. ``error``: that fit failed (said once; the next ask starts over). ``not_calibrated``: "
                "the dataset's face embedder has no cut to find people at, and ``message`` says so."
            )
        },
    )
    partition = fields.String(load_default=None, metadata={"description": f"When ``ready``: {_PARTITION}"})
    people = fields.List(
        fields.Nested(PersonSummarySchema),
        load_default=None,
        metadata={"description": "When ``ready``: every person, the most faces first."},
    )
    unplaced = fields.Integer(
        load_default=None,
        metadata={"description": "When ``ready``: how many faces are in no person (hard views that matched no one)."},
    )
    unplaced_ids = fields.List(
        fields.Integer(),
        load_default=None,
        metadata={"description": "When ``ready``: the faces in no person, ascending."},
    )
    cutoff = fields.Float(
        load_default=None,
        metadata={
            "description": (
                "When ``ready``: the cosine a face cleared to its person's centroid (the face ball's beta 1 floor)."
            )
        },
    )
    compat = fields.String(
        load_default=None,
        metadata={
            "description": (
                "The face space (``Face Compat``) the people were cut in, when ``ready``; the space with no cut, "
                "when ``not_calibrated`` (null when the faces carry no single one)."
            )
        },
    )
    method = fields.String(load_default=None, metadata={"description": "When ``ready``: how the people were fit."})
    linkage = fields.String(
        load_default=None,
        metadata={"description": "When ``ready``: the clustering linkage the fit used."},
    )
    # Fit progress (when status == "building"), as POST /api/medias/similarity-order reports it.
    # ``message`` is the uncalibrated wording when status == "not_calibrated".
    job_id = fields.String(load_default=None)
    current = fields.Integer(load_default=None)
    total = fields.Integer(load_default=None)
    message = fields.String(load_default=None)
    step = fields.Integer(load_default=None)
    total_steps = fields.Integer(load_default=None)
    overall = fields.Float(load_default=None)
    overall_step_end = fields.Float(load_default=None)
    eta_seconds = fields.Float(load_default=None)
    error = fields.String(load_default=None)


class PeopleBuildRequestSchema(Schema):
    """Body for ``POST /api/people/build``; every field is optional."""

    force = fields.Boolean(
        load_default=False,
        metadata={"description": "Fit the people again even when a partition is stored (Rebuild people)."},
    )


class PersonQuerySchema(Schema):
    """Query for the per-person routes: the partition the person's id was read under."""

    partition = fields.String(required=True, metadata={"description": _PARTITION})

    class Meta:
        # Request-context params (dataset_id) smuggled in as query args, as on /api/sort/page.
        unknown = "exclude"


class PersonPhotoSchema(Schema):
    """A photo a person is in, in ``GET /api/people/<id>``."""

    photo = fields.Integer(
        required=True,
        metadata={
            "description": (
                "The photo's id: a number for the photo while the dataset's faces are unchanged, "
                "the same one the shared-photos route gives it."
            )
        },
    )
    face_ids = fields.List(
        fields.Integer(),
        required=True,
        metadata={"description": "The person's faces cut from the photo, in the person's order (usually one)."},
    )


class CoPersonSchema(Schema):
    """Another person who shares a photo with a person, in ``GET /api/people/<id>``."""

    id = fields.Integer(required=True, metadata={"description": "The other person's rank in this partition."})
    shared = fields.Integer(required=True, metadata={"description": "How many photos the two share."})
    representative = fields.Integer(required=True, metadata={"description": "The other person's representative."})


class PersonSchema(Schema):
    """Response for ``GET /api/people/<id>``: a person's faces, photos, and who appears with them."""

    partition = fields.String(required=True, metadata={"description": _PARTITION})
    id = fields.Integer(required=True)
    representative = fields.Integer(required=True, metadata={"description": "The person's representative face."})
    face_ids = fields.List(
        fields.Integer(),
        required=True,
        metadata={"description": "The person's faces: the representative first, then from the closest to them out."},
    )
    photos = fields.List(
        fields.Nested(PersonPhotoSchema),
        required=True,
        metadata={"description": "The photos the person is in, by photo id."},
    )
    appears_with = fields.List(
        fields.Nested(CoPersonSchema),
        required=True,
        metadata={"description": "The people who share a photo with this person: most shared first, ties by rank."},
    )


class SharedPhotoFaceSchema(Schema):
    """One face cut from a shared photo, in ``GET /api/people/<id>/with/<other>``."""

    media_id = fields.Integer(required=True, metadata={"description": "The face's media id."})
    box = fields.List(
        fields.Float(),
        required=True,
        allow_none=True,
        metadata={
            "description": (
                "Where the face sits in its photo, normalised ``[x0, y0, x1, y1]`` "
                "(its ``source_box``), or null when its converter records none."
            )
        },
    )
    person = fields.Integer(
        required=True,
        allow_none=True,
        metadata={"description": "The face's person (their rank in this partition), or null when it is unplaced."},
    )


class SharedPhotoSchema(Schema):
    """A photo two people share, with every face cut from it."""

    photo = fields.Integer(
        required=True, metadata={"description": "The photo's id, as ``GET /api/people/<id>`` gives it."}
    )
    source_dataset_id = fields.String(
        required=True,
        allow_none=True,
        metadata={
            "description": (
                "The Image dataset holding the photo, when it could be named; null when no dataset the user "
                "can read holds it."
            )
        },
    )
    source_media_id = fields.Integer(
        required=True,
        allow_none=True,
        metadata={"description": "The photo's media id in ``source_dataset_id``, or null with it."},
    )
    faces = fields.List(
        fields.Nested(SharedPhotoFaceSchema),
        required=True,
        metadata={"description": "Every face cut from the photo, not only the two people's, ascending."},
    )


class SharedPhotosSchema(Schema):
    """Response for ``GET /api/people/<id>/with/<other>``: the photos two people share."""

    partition = fields.String(required=True, metadata={"description": _PARTITION})
    id = fields.Integer(required=True)
    other = fields.Integer(required=True)
    photos = fields.List(
        fields.Nested(SharedPhotoSchema),
        required=True,
        metadata={"description": "The photos both people have a face in, by photo id."},
    )
