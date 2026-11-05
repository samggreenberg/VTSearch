# People

[← Back to API index](../API.md)

> Every People route is scoped to the requesting **Face** dataset through the
> [`X-Dataset-Id` context header](../API.md#context-headers-x-dataset-id--x-detector-id),
> and answers 400 without it.
>
> This page covers purpose and non-obvious semantics. Exact request and
> response fields are in the [OpenAPI spec](../API.md#machine-readable-schema)
> (`frontend/openapi.json`).

---

A Face dataset's **people** are its face crops merged into discrete persons,
and two people **appear with** each other when their faces were cut from the
same photo. The routes serve two structures the library derives from the
dataset's own vectors and origins, both ids only:

- **The partition** (`vtscore.datasets.people`): the faces merged at the face
  ball's cut, so a face belongs to a person when it passes the test Find
  applies. It is fit when the dataset is built, kept in its container, and fit
  on the first ask for a dataset built before that.
- **The photo index** (`vtscore.datasets.co_occurrence`): which faces were cut
  from one photo, read off each face's own source refs. It needs no Image
  dataset loaded, so only the [shared-photos](#the-photos-two-people-share)
  route, which names the photo itself, needs one.

Both are described in
[`vtscore/docs/packages/datasets.md`](../../vtscore/docs/packages/datasets.md#people).

## Person ids and the partition

A person's `id` is their **rank** in one partition: `1` for the person with
the most faces (ties to the lowest face id). A fit can reorder people and a
removed face can drop one, so an id names someone only under the partition it
was read from. Every answer carries that partition's identity as
`partition`, and the per-person routes take it back as `?partition=`:

- **The identity is a digest of the partition as served**: its people, its
  unplaced faces and the cut. A forced fit that finds the same people keeps
  it, so ids read before the fit stay good.
- **An id read under another partition is a 409 `people_changed`**, never an
  answer about someone else. List the people again and pick the person anew.
- **The per-person routes never start a fit.** They read the stored
  partition, and answer 409 `people_not_ready` when there is none yet. While
  a forced fit runs they keep answering from the partition it will replace,
  under that partition's identity.

A **photo id** (`photo`) is a number for one photo, valid while the dataset's
faces are unchanged. The routes give it out but never take it.

## List the people

```
GET /api/people
X-Dataset-Id: <face dataset id>
```

→ `{ "status": "ready", "partition", "people": [{id, face_count, photo_count, representative}], "unplaced", "unplaced_ids", "cutoff", "compat", "method", "linkage" }`

`people` lists every person, the most faces first. `representative` is the
face that stands for them, their face with the highest Face Quality, for a
thumbnail. `unplaced` counts the faces in no person (hard views that matched
no one's centroid), and `unplaced_ids` lists them. `cutoff` is the cosine a
face cleared to its person's centroid, in the face space `compat`.

The other answers:

- `{ "status": "building", "job_id", ... }` while the people are being found,
  with the progress fields of
  [`POST /api/medias/similarity-order`](medias.md#similarity-order). Ask
  again until it is `ready`.
- `{ "status": "error", "error" }` once, for a fit that failed. The next ask
  starts over.
- `{ "status": "not_calibrated", "compat", "message" }` when the dataset's
  face embedder has no cut to find people at (the face ball is calibrated for
  one face space). No people are fit, and `message` is the wording the face
  ball's own warning uses.

A dataset that is not a Face dataset, or whose faces have no vectors, is a
409 `no_people`.

## Rebuild the people

```
POST /api/people/build
X-Dataset-Id: <face dataset id>
Content-Type: application/json

{ "force": true }
```

Answers as [`GET /api/people`](#list-the-people) does. Without `force` (or
without a body) it is that ask. With `force`, a stored partition is fit again
and the answer is `building`, or the running fit's progress when one already
is; poll `GET /api/people`.

## One person

```
GET /api/people/<id>?partition=<partition>
X-Dataset-Id: <face dataset id>
```

→ `{ "partition", "id", "representative", "face_ids", "photos": [{photo, face_ids}], "appears_with": [{id, shared, representative}] }`

`face_ids` are the person's faces, the representative first, then from the
closest to the person's centroid out. `photos` are the photos the person is
in, each with the person's faces cut from it (usually one). `appears_with`
lists every other person with a face in one of those photos, with how many
photos the two share; most shared first, ties by rank. Needs no Image dataset.

404 for an id with no person in the partition; 409 `no_people`,
`people_not_ready` or `people_changed` as above.

## The photos two people share

```
GET /api/people/<id>/with/<other>?partition=<partition>
X-Dataset-Id: <face dataset id>
```

→ `{ "partition", "id", "other", "photos": [{photo, source_dataset_id, source_media_id, faces: [{media_id, box, person}]}] }`

Every photo both people have a face in, with **every** face cut from it, not
only theirs, so the photo can be drawn with everyone outlined. Each face
carries its `box` in the photo (normalised `[x0, y0, x1, y1]`, its
`source_box`) and its `person` (rank), or `null` for an unplaced face.

The photo is named as an item of the Image dataset the same import produced
(`source_dataset_id` / `source_media_id`), found as
[`GET /api/medias/<id>/source`](medias.md#resolve-a-derived-items-source)
finds a face's photo:

- **The Image dataset must be loaded.** When it is registered but not, the
  answer is the shared 409 `source_not_loaded`, with `dataset_id` naming it
  for [`POST /api/datasets/registry/<id>/load`](datasets.md#load-registered-dataset). Load it and ask
  again.
- **One the user cannot read** leaves `source_dataset_id` and
  `source_media_id` `null`; the faces are still listed.

404 for an id with no person; 409 `no_people`, `people_not_ready` or
`people_changed` as above.
