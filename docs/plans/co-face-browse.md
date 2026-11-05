# Design: Co-face Browse — the people in a Face dataset, and who shares a photo

A view on a Face dataset that shows its **people** (face crops merged into
discrete person designators) and, for each person, **who they appear with**:
the other people whose faces were cut from the same photos of the Image
dataset the import made beside the faces. Nothing here trains, votes or
scores; it is a browser over two derived structures, both ids-only, both
rebuilt from the dataset's own vectors and origins.

## Background

### What was asked

The owner's request (#5582): add a view to Face datasets that shows the
people in the dataset — merged face instances into discrete people — and
which people share a photo in the original image dataset, the way a face crop
can already show the photo it came from. The questions were where it would
live, what has to be written, and what frontend work it needs.

### What the repo already has

Every input the view needs exists; the work is two derived structures and a
surface over them.

- **An identity space.** A face crop carries a 512-d vector in the face
  provider's space (`vtscore/media/face/embedder_imagekit.py`; buffalo_s, tag
  `oss-arcface-mbf:1` in `custom_metadata["Face Compat"]`), L2-normalised at
  ingest so cosine is a dot product, and cached as the `(N, 512)` matrix
  `get_embedding_matrix` serves (`vtscore/embedding/matrix.py`). Every face
  also carries a `Face Quality`, and `HARD_VIEW_QUALITY`
  (`vtscore/media/face/imagekit_bridge.py`) names the cut under which a face
  is a *hard view* that a search started from it tends to miss (#5372).
- **The photo join.** A face's `origin.params` records the photo it was cut
  from (`source_file`, `source_path`) and `source_box` says where in it
  (`vtscore/media/provenance.py`). `vtscore/datasets/source_item.py` resolves
  that to the photo as an item of the Image sibling (`SourceResolver`,
  `SourceLocator`, with `converter_refs` for a `dupe_set` representative), and
  `vtscore/datasets/source_groups.py` groups a ranking by photo for Collapse
  to photos (#4751). The join key lives on the face itself, so *which faces
  share a photo* is answerable from the Face dataset alone; the sibling is
  needed only to show the photo.
- **The surfaces that show a face's photo.** Show in photo (#4750,
  `vt-source-photo-modal`, one outlined box over `vt-image-viewer`), Collapse
  to photos (#4751, `vt-photo-grid`, every ranked face outlined on its photo),
  and `SiblingLoadService` with the shared 409 `source_not_loaded` path for an
  Image sibling that is registered but not loaded
  (`vtsearch/routes/_source.py`).
- **The pattern for an ids-only derived cache.** The 1-D similarity order
  (`vtscore/projection/similarity_order.py`, #5464, #5465) is exactly the
  shape a people partition needs: a pure function of the pickle's medias,
  stored in the container (`similarity_order.json`, `vtscore/datasets/container.py`),
  cached on `DatasetContext` in its own reset group, checked against the live
  id set on every read, fit at build for Face datasets by a load stage
  (`vtscore/datasets/stages/projection.py`) and on first ask otherwise, with a
  `request_*` call that answers "ready", "building" or "error" for a client
  that polls. It is the sanctioned carve-out from the no-persisted-vectors
  rule: ids and a method, never a vector.
- **A measured notion of "the same person".** The face ball
  (`vtscore/detectors/centroid_head.py`, #5474, #5478) is the app's shipped
  test for whether a face is one of a person's: cosine to the Goods' centroid
  at or above a floor per balance (`FACE_BALL_FLOORS`, 0.4425 at β 1, moved by
  the Goods' spread). On FHIBE's 562 people, 86% are perfectly separable by
  cosine to their centroid, the one floor lands inside 80% of the gaps between
  a person and everyone else, and 14% have no such gap
  ([2026-10-30-face-balls-5474](../experiments/2026-10-30-face-balls-5474/REPORT.md)).
  The floors are the embedder's, not the idea's: they fail by 0.65 on FaceNet
  ([2026-11-02-face-ball-embedders-5486](../experiments/2026-11-02-face-ball-embedders-5486/REPORT.md)),
  and #5497 is moving them into a per-compat file with a fallback for an
  uncalibrated tag.
- **Ground truth to measure a partition against.** FHIBE (on the GRID) has
  562 identities with 7–10 one-person photos each; the Celebrities in Context
  demo (`celebrity_together_*`) names one person per photo for 60 people, in
  scenes that hold other people too; VGGFace2 frames one labelled person per
  photo ([demos](../demos.md)).
- **A cut Browse item this revives.** `vtsbrowse.md` cut *Sibling
  highlighting* (highlight the bins holding faces from the same source file)
  for want of "source-file group ids in the tile payload or a server-side
  lookup endpoint". The photo index below is that plumbing.

## The shape

### Two derived structures, library tier, ids only

**The people partition** shipped in #5586 as `vtscore.datasets.people`: the
fit, the container entry, the context cache and `request_people` are
documented in [`vtscore/docs/packages/datasets.md`](../../vtscore/docs/packages/datasets.md#people).

**The photo index.** Faces keyed by the photo they were cut from, read off
each face's own converter refs (`source_path`, else `source_file`; a
`dupe_set` representative contributes every member's), with keys unioned
where one face carries two (two copies of one photo that the sibling merged).
It needs no sibling loaded. From it and the partition: for each person, the
photos they are in, and for each pair of people, the photos they share. This
is cheap enough to build on demand from the partition and cache beside it;
it need not be persisted.

The photo index shipped in #5587 as `vtscore.datasets.co_occurrence`, beside
`people.py`, `source_item.py` and `source_groups.py`, which own the
face-to-photo join
([`datasets.md`](../../vtscore/docs/packages/datasets.md#on-disk-dataset-registry)).

### Merging faces into people

The fit shipped in #5586
([`datasets.md`](../../vtscore/docs/packages/datasets.md#people)): a
neighbour graph, centroid linkage on the unit sphere and hard views attached
last, all at the face ball's beta 1 floor, so a face belongs to a person when
it passes the test Find applies. Two rules from its design still bind the
work left:

**Prefer a split to a merge.** A person shown as two is visible and
recoverable; two people shown as one hides someone. Where the calibration
offers a choice, take the higher cut.

**An uncalibrated compat tag** (an `ext` provider's space, #5497) has no
floor to cut at, so it gets no partition: the view says People is not
calibrated for this face embedder, with the same wording as the ball's
warning. A fixed fallback floor with a warning was the alternative; the owner
chose the message (2026-11-05), so nothing plausible-but-wrong is shown.

### Where the view lives

A **new dataset-only view**, `/people/:datasetId`, beside Browse in the route
table, entered from the dataset card's menu (a **People** item, offered on a
Face dataset only) and from Browse's toolbar on a Face dataset. Not a layer on
the Browse canvas, and not a modal (the owner's call, 2026-11-05):

- The Browse map already lays faces out in identity space, so a person is
  already a blob there; what the map cannot show is a relation between blobs.
  Co-occurrence is a graph over people, and a list-and-detail layout reads it;
  a canvas would need its own rendering path in a component that is already
  the largest in the app.
- A modal is too small for a browse: the user moves from person to person to
  photo and back, and wants the ranking-sized panel the Train view gives a
  list.

The view is three panes, in the Browse view's layout idiom:

- **People list** (left, virtual-scrolled, as `vt-browse-selection-panel`
  scrolls): one row per person, the representative's thumbnail, `N faces ·
  M photos`; sorted by faces, with a filter for a minimum count so the long
  tail of one-face people folds away. An **Unplaced** row at the end holds
  the faces in no person.
- **Person** (centre): the person's faces as a thumbnail grid,
  `vt-browse-bin-member-grid` reused as the bin details panel reuses it, the
  representative first; clicking a face shows it large with its metadata,
  as the bin details pane does.
- **Appears with** (right): the people who share a photo with this person,
  each with the shared count and their representative, most shared first;
  and under the selected co-person, the shared photos. A photo renders as
  `vt-photo-grid` renders a collapsed row — the photo with every face in it
  outlined — with the outlines coloured per person (this person, the
  co-person, anyone else), and opens in `vt-source-photo-modal`, extended
  from one outline to several.

An unloaded Image sibling is handled as Collapse to photos handles it: the
photo strip offers to load it through `SiblingLoadService`, and the people
list and faces grid need no sibling at all. Hover previews, `OnPush`,
signals and `rxResource` follow `docs/FRONTEND.md`'s rules; the view gets a
component-provided state service as Browse does.

**Rebuild people** sits in the toolbar as **Rebuild Map** does in Browse,
for a partition fit before a cut change. There is no strictness slider in
the first version: the cut is the app's own, and a slider would invite a
partition that disagrees with Find.

### What the view does not do

- No votes, no detectors, no labels: a person is presentation over faces,
  as a collapsed photo is. Every existing count stays a count of faces.
- No names. People are numbered by rank (`Person 1`), and the number is not
  stable across a rebuild. Naming a person, persisted by the representative's
  origin and re-attached after a rebuild by member overlap, is a follow-up.
- No merge/split corrections by hand. Also a follow-up; the data model
  (ids per person) can carry an override list without changing shape.
- Nothing persisted but ids. The centroids the fit uses are in-memory; the
  container entry carries ids, method, cutoff and compat tag.

## Settled when the slices were filed

The open questions were taken as proposed (2026-11-05): centroid linkage
ships and #5591 measures it against average linkage, with the
split-over-merge rule as the tiebreak; the entry points are the dataset
card's menu and Browse's toolbar; the view is *People* and the relation
*Appears with*. A *Show person* action from Train's face list is not filed,
and is worth revisiting once the view ships.

## Open work

<!-- item-sep -->

- [ ] #5586 — People partition: merge a Face dataset's crops into people at the face ball's floor

<!-- item-sep -->

- [ ] #5587 — Photo index: which faces share a photo, and which people appear together

<!-- item-sep -->

- [ ] #5589 — People routes: list, person, shared photos, build

<!-- item-sep -->

- [ ] #5588 — Image viewer and Show in photo: outline several faces, coloured by person

<!-- item-sep -->

- [ ] #5590 — People view: a Face dataset's people and who appears with whom

<!-- item-sep -->

- [ ] #5591 — Calibrate the People cut against FHIBE and the VGGFace2 demo (experiment)

<!-- item-sep -->

- **Sibling highlighting on the map, revived** — with the photo index
  served, Browse can highlight the bins holding the other faces of the
  hovered face's photo. `vtsbrowse.md` cut this item for want of exactly
  that plumbing, so reviving it is the owner's call, and it is not filed.
  After #5589 if taken up. *Recommended Claude model: Sonnet.*

<!-- item-sep -->

### Dependencies

- #5586 and #5587 are independent in code; #5589 needs both.
- #5588 needs nothing and can run beside the library slices.
- #5590 needs #5589 and #5588.
- #5591 needs #5586 for the fit it measures, and can run beside the
  frontend slices; its outcome is a constant or a linkage choice, so it
  does not block the view.
