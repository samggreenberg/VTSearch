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

**The people partition.** A list of people, each a list of face ids, plus the
faces placed in no person. Every face is in at most one person. Built from the
cached embedding matrix and persisted in the container as a new entry beside
`similarity_order.json` (ids, the method, the cutoff, the compat tag it was
cut in), cached on the context in a reset group of its own, and swept with
the container. A partition that still covers every current face serves after
items were removed; one that misses a face is rebuilt.

**The photo index.** Faces keyed by the photo they were cut from, read off
each face's own converter refs (`source_path`, else `source_file`; a
`dupe_set` representative contributes every member's), with keys unioned
where one face carries two (two copies of one photo that the sibling merged).
It needs no sibling loaded. From it and the partition: for each person, the
photos they are in, and for each pair of people, the photos they share. This
is cheap enough to build on demand from the partition and cache beside it;
it need not be persisted.

Both live in `vtscore/datasets/`, beside `source_item.py` and
`source_groups.py`, which already own the face-to-photo join: a new
`people.py` (the partition: fit, persistence, context cache, the `request_*`
contract) and a new `co_occurrence.py` (the photo index and the pairwise
counts). Flask-free, taking an explicit context, tested under
`tests_lib/datasets/`.

### Merging faces into people

The partition should agree with Find: a face belongs to a person when it
passes the test the face ball already applies, cosine to the person's centroid
at or above the β 1 floor. So the cut is the ball's floor for the dataset's
compat tag, not a new knob.

The fit, in three steps, all on the cached matrix:

1. **A neighbour graph.** Each face's top-k cosine neighbours (k ≈ 20, by
   chunked dot products over the matrix), keeping edges at or above the
   floor. Connected components of that graph are candidate people. This
   bounds the work to O(N·k) and never forms an N×N distance matrix; a Face
   dataset is a few thousand crops, but the matrix would be the one thing
   that scales badly.
2. **A split within each component.** Chaining (A≈B, B≈C, A≉C) is the
   failure mode of a threshold graph, and two people who share a lookalike
   would land in one component. Each component is re-cut with agglomerative
   clustering at the same floor (scipy, already a dependency; a component is
   small, so its distance matrix is not), centroid or average linkage,
   which of the two the calibration below settles. A face stays in the
   cluster whose centroid it clears the floor to.
3. **Hard views last.** A face under `HARD_VIEW_QUALITY` is poor evidence of
   identity and is left out of steps 1 and 2. It is then attached to the
   person whose centroid it clears the floor to, if any, else left unplaced.
   This mirrors the trained head's down-weighting of hard-view Goods (#5413).

Then: people are ordered by face count, descending; each person's
representative is its face with the highest Face Quality (ties to the one
closest to the centroid); a person with one face is a person, listed after
the rest. The fit is deterministic (no UMAP, no seed), so a rebuild on the same
faces gives the same people.

**Prefer a split to a merge.** A person shown as two is visible and
recoverable; two people shown as one hides someone. Where the calibration
offers a choice, take the higher cut.

**An uncalibrated compat tag** (an `ext` provider's space, #5497) has no
floor to cut at. Proposed: no partition, and the view says People is not
calibrated for this face embedder, with the same wording as the ball's
warning. A fixed fallback floor with a warning is the alternative; the owner's
call.

### Where the view lives

A **new dataset-only view**, `/people/:datasetId`, beside Browse in the route
table, entered from the dataset card's menu (a **People** item, offered on a
Face dataset only) and from Browse's toolbar on a Face dataset. Not a layer on
the Browse canvas, and not a modal:

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

## Open questions for the owner

- **Linkage.** Centroid linkage is the ball's own test applied cluster to
  cluster; average linkage is the standard for face clustering. The
  calibration slice measures both; pick by the measurement, with the
  split-over-merge rule above as the tiebreak.
- **Uncalibrated embedders.** No partition and a message, or a fallback
  floor with a warning (see above).
- **Entry points.** The dataset card's menu and Browse's toolbar are
  proposed; the Train view's face list could offer *Show person* on a face
  (the view opened on that person), which is cheap once the view exists.
- **Naming the view.** *People* for the view and *Appears with* for the
  relation; the issue's *Co-face Browse* names the feature.

## What has to be written

By tier, with the slice each belongs to in parentheses.

**Library (`vtscore/`)**

- `vtscore/datasets/`: `people.py` (partition dataclass, the fit, persistence
  through two new container functions beside `append_similarity_order` /
  `read_similarity_order`, the context cache and `request_people`) and
  `co_occurrence.py` (photo index, per-person photos, pairwise shared
  photos). (People partition; Photo index)
- `vtscore/state/dataset_context.py`: two slots (`_people`, `_people_job_id`)
  and a `people` reset group, as `order` is. (People partition)
- `vtscore/datasets/stages/projection.py`: a people stage for Face datasets
  beside the similarity-order stage, best-effort by the same contract.
  (People partition)
- `vtscore/datasets/registry.py`: nothing; the entry rides inside the
  container, so `unregister_dataset` already sweeps it. The same goes for
  the sidecar sweep.
- `vtscore/docs/packages/datasets.md`: the two modules. (People partition)

**App (`vtsearch/`)**

- A `people` blueprint (new module beside `vtsearch/routes/sorting.py`):
  `GET /api/people` (status and the list: id, face count, photo count,
  representative; `building` with a job id while the fit runs, as
  `/api/medias/similarity-order` answers), `GET /api/people/<id>` (the faces,
  the photos, the co-people with counts), `GET /api/people/<id>/with/<other>`
  (the shared photos, each with its faces, boxes and person ids, and the
  sibling item when it resolves) and `POST /api/people/build` with `force`.
  An unloaded sibling is the shared 409. (Routes)
- Schemas in a new `vtsearch/schemas/` module, the OpenAPI snapshot
  regenerated, and a `docs/api/` page indexed from `docs/API.md`. (Routes)

**Frontend (`frontend/src/app/`)**

- Route `people/:datasetId` with the dataset-only guard
  (`guards/browse-context.guard.ts` generalised, or a sibling of it), a
  `PeoplePrepService` that loads the dataset and waits for the partition
  before navigating, as `BrowsePrepService` does for the map. (People view)
- `components/people-view/`: the view, the people list, the person pane and
  the appears-with pane; a `PeopleApiService` wrapper over the generated
  client. (People view)
- `vt-source-photo-modal` and `vt-image-viewer`: an `outlineBoxes` input
  (several boxes, each with a colour class) beside `outlineBox`. (Multi-outline
  photo)
- The dataset card menu (`components/dashboard/card-context-menu-items.ts`)
  and the Browse toolbar: the People entry on a Face dataset. (People view)

**Docs and pictures**

- `docs/ARCHITECTURE.md` (a paragraph beside Collapse to photos),
  `docs/FRONTEND.md` (route table and a feature-area entry), the user guide's
  Faces section, `CHANGELOG.md`. (Docs)
- New screenshots are captured in the session that adds them
  (`scripts/screenshots/refresh.sh <id>`); nothing existing moves, so the
  reshoot queue is untouched. (Docs)

**Tests**

- `tests_lib/datasets/`: the fit on synthetic clusters (two people, a
  chained lookalike, a hard view, a singleton), persistence round trip, the
  live-id check, the stage; the photo index on faces with shared keys and a
  `dupe_set` representative. (People partition; Photo index)
- `tests/`: the routes, the 409, the build/poll contract. (Routes)
- Vitest: the view's zoneless specs, the multi-outline viewer. (People view;
  Multi-outline photo)

## Open work

Each item is a slice that can be filed as an issue when the owner settles
the questions above; the plan then keeps a one-line pointer in its place.

<!-- item-sep -->

- **People partition** — `people.py`, the container entry, the context
  slots and reset group, the load stage, `request_people`, library tests.
  Ships with centroid linkage at the ball's β 1 floor; the calibration may
  move it. *Recommended Claude model: Opus* (cache plumbing that mirrors the
  similarity order exactly, and a clustering whose wrong answer looks
  plausible).

<!-- item-sep -->

- **Photo index and co-occurrence** — `co_occurrence.py`: the key union,
  per-person photos, pairwise shared photos, library tests. Independent of
  the partition in code (it takes any grouping of face ids). *Recommended
  Claude model: Sonnet.*

<!-- item-sep -->

- **People routes** — the blueprint, schemas, OpenAPI snapshot, API page,
  route tests. After the two library slices. *Recommended Claude model:
  Sonnet.*

<!-- item-sep -->

- **Multi-outline photo** — `outlineBoxes` on the image viewer and the source
  photo modal, with per-person colours, and its Vitest specs. Independent of
  the backend; the Collapse to photos grid may adopt it too. *Recommended
  Claude model: Sonnet.*

<!-- item-sep -->

- **People view** — the route, guard and prep service, the three panes, the
  dataset-card and Browse entries, zoneless specs, the user-guide section
  with its new screenshots. After the routes and the multi-outline photo.
  *Recommended Claude model: Opus* (a new view's reactivity and the shared
  sibling-load path).

<!-- item-sep -->

- **Calibrate the cut** — on FHIBE's 562 identities (GRID) and the
  Celebrities in Context demo (laptop): pairwise precision, recall and
  B-cubed F of the partition against the ground truth across floors around
  the ball's, for centroid and average linkage, with and without the
  hard-view step; and the share of people split or merged at the shipped
  cut. Writes a `REPORT.md` under `docs/experiments/`; moves the shipped
  linkage or floor only if it pays. An experiment: it cannot close without
  the run. *Recommended Claude model: Sonnet* for the harness and analysis.

<!-- item-sep -->

- **Sibling highlighting on the map, revived** — with the photo index
  served, Browse can highlight the bins holding the other faces of the
  hovered face's photo, the item `vtsbrowse.md` cut. Optional, after the
  routes. *Recommended Claude model: Sonnet.*

<!-- item-sep -->

### Dependencies

- *People partition* and *Photo index* are independent in code; *People
  routes* needs both.
- *Multi-outline photo* needs nothing and can run beside the library slices.
- *People view* needs *People routes* and *Multi-outline photo*.
- *Calibrate the cut* needs *People partition* for the fit it measures, and
  can run beside the frontend slices; its outcome is a constant or a linkage
  choice, so it does not block the view.
