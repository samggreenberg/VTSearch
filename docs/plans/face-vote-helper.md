# Design: Face vote helper — comparing two faces the way examiners do

Voting on a Face dataset asks the user one question per crop: *is this the
person I am looking for?* The app shows one face and two buttons. This plan
puts a **reference face** beside the candidate, sizes and levels the two so
they can be compared part by part, and adds a short **checklist** of what to
compare, in the order the research says is most useful. Nothing here changes
training, scoring or what a vote means.

## Background

### What was asked

The owner's request (#5639): find outside references for how face comparison
is done by hand, then design a UI that helps with it. The ideas floated were a
reference face kept on the left, aligning the two, and a checklist of the
suggested checks.

### What the outside references say

Two bodies of work, both free to read.

**The practitioner standards** (US: FISWG and the NIST-run OSAC Facial
Identification Subcommittee; Europe: ENFSI):

- [OSAC 2022-S-0007, *Standard Guide for Facial Comparison: Overview and
  Methodology Guidelines*](https://www.nist.gov/document/osac-2022-s-0007-standards-guide-facial-comparison-overview-osac-proposed-registry)
  names three methods and rules on each.
  - **Morphological analysis** is the one to use: compare the face part by
    part (nose, ear, mouth), then the detail of each part (nostrils, ear
    lobes, helix), plus "scars, marks, and tattoos". It "should be the
    primary approach" at every level of rigour.
  - **Superimposition** (overlaying two aligned images) "should be used only
    to aid" and "shall never be used as a stand-alone approach". It applies
    only when both images are "taken from the same viewpoint", and only with
    tools "which preserve shapes". The European manual below is stricter.
  - **Photo-anthropometry** (measuring distances and angles between
    landmarks) "shall not be used". It "does not produce consistent results
    and may be misleading".
  - Just looking at the whole face (*holistic comparison*) "is not recognized
    as a method".
  - Classification schemes ("round face, Roman nose") are "not best
    practice", because observers disagree on them.
  - The process is ACE-V: **A**nalysis (is the image fit to compare at all?),
    **C**omparison, **E**valuation, **V**erification (a second examiner).
- [FISWG, *Facial Image Comparison Feature List for Morphological Analysis*
  v2.0](https://fiswg.org/FISWG_Morph_Analysis_Feature_List_v2.0_20180911.pdf)
  (sold by ASTM as E3149; the FISWG copy is free) is the list of parts the
  method walks. It is a vocabulary, not a procedure: it "does not define the
  comparison process itself".
  - 19 components: skin, face/head outline, face/head composition, hair,
    forehead, eyebrows, eyes, cheeks, nose, ears, mouth, chin/jawline, neck,
    facial hair, facial lines, scars, facial marks, alterations, other.
  - Each splits twice more. The ear table alone has 19 rows, 15 of them named parts (helix,
    antihelix, tragus, concha, lobe "attached or detached", ...); the mouth
    table starts with the philtrum.
  - The order is "from the top of the face to the bottom, not in order of
    importance or priority".
  - Where it says "distance" it means relative size judged by eye, and it
    repeats that photo-anthropometry should "not be used at all".
- [OSAC 2020-S-0002, *Physical Stability of Facial Features of
  Adults*](https://www.nist.gov/document/osac-2020-s-0002-physicalstabilityoffacialfeaturesofadultsosacproposedstandard)
  rates each part High / Medium / Low for how much it changes with
  expression, time, weight, health and deliberate alteration. Its ratings are
  "primarily a consensus opinion of practitioners". The ones that matter here:

  | Part | Rating | Note |
  |---|---|---|
  | Ears | High on almost every column | "The ear is a very stable feature" |
  | Moles, skin tags | Stable short and long term | Acne and blemishes are transient |
  | Scars | Medium | Unstable for the first year |
  | Nose | Stable short term | Lengthens over the long term |
  | Hair, hairline, baldness | Low over time and under alteration | High only against expression |
  | Facial hair, skin appearance | Low under alteration | Makeup, tanning, grooming |

- [ENFSI-BPM-DI-01, *Best Practice Manual for Facial Image
  Comparison*](https://www.mvr.bg/upload/19089/enfsi-bpm-di-01.pdf) (2018)
  is the European counterpart. It adopts the FISWG list as its checklist and
  ACE-V as its workflow, and differs from the OSAC guide in four ways that
  matter here.
  - **It rejects overlays outright.** Superimposition, "proportional
    alignment" and feature classification are all "not recommend[ed]". It
    cites two studies in which chimeric, blinked and wiped overlays "reduced
    matching accuracy compared to full facial images" and biased viewers
    toward calling a match.
  - **It allows resizing and rotating** an image to aid observation, with the
    aspect ratio kept.
  - **It guards the order of viewing.** The questioned image is assessed
    before the reference is studied, so that a sharp reference does not lead
    the viewer to see detail the questioned image cannot resolve.
  - **It wants an inconclusive answer** on any scale of conclusions, and recommends
    that "statements of absolute certainty are not used".

  It also names the useful features: "the ears, nose, and scars and
  blemishes were most diagnostic", and feature-by-feature comparison improved
  accuracy "when the facial images are a true match".

**The psychology** (mostly the UNSW Sydney group):

- [White et al. 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4571699) and
  [Phillips et al. 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6019409):
  trained examiners beat untrained people, most clearly when given time.
- [Towler, White & Kemp 2017](https://pubmed.ncbi.nlm.nih.gov/28045276/):
  having novices rate the similarity of each feature before deciding improved
  their accuracy. For examiners, **ears and facial marks** were the features
  whose similarity best predicted the right answer; **face shape and mouth**
  were the worst.
- [Towler et al. 2021](https://gwern.net/doc/psychology/vision/2021-towler.pdf):
  a 6-minute course telling novices to rely on ears and facial marks raised
  their accuracy by 6%, about half of the examiners' lead. The same course
  pointed at face shape and mouth did nothing. Both courses told trainees not
  to look at the face as a whole or fix on the eyes-and-mouth triangle. The
  ranking the course taught, most useful first: marks, ears, eyes, face
  shape, mouth. Novices left to themselves ranked eyes and face shape first.
- Two cautions from the same literature. The gain was strongest when the
  diagnostic features were clearly visible. And about 70% of trainees said
  the course helped *whichever* course they took, so users will like a helper
  whether or not it works.
- The evidence is not unanimous:
  [Megreya 2018](https://peerj.com/articles/4437) found a feature-by-feature
  strategy had no effect on matching.

**On turning the face upside down.** The issue recalls this as part of the
method. No standard read here recommends it. In the research it is a *probe*:
[White et al. 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4571699) showed
inverted pairs to find out how examiners work, and found them less hurt by
inversion than novices, which is taken as a sign that they compare parts
rather than whole faces. Inversion did not make anyone more accurate. The
viewer already turns an image a quarter turn with `[` and `]`, so the option
exists; this plan adds nothing for it.

### What this means for a design

- **A second face on screen is the core of it.** Every method above is a
  comparison of two images. Today the user compares the candidate against
  their memory of the target.
- **A checklist, short and in diagnostic order.** Marks, ears and nose first,
  face shape and mouth last, with hair, facial hair and skin tone named as
  things *not* to weigh. It is a prompt, not a form: the vote is still one
  key. The FISWG list supplies the wording; its 19 components are far too
  many for a vote.
- **Side by side, never overlaid.** The two standards disagree on overlays
  and the stricter one cites measured harm, so this plan has no overlay,
  blink or wipe. Matching the two faces' size and position, with the aspect
  ratio kept, is within what both allow.
- **No measurements.** No distances, ratios or angles shown as numbers. That
  is the method the standard forbids.
- **Say when a pair is hard to compare.** ACE-V starts with fitness. The app
  already knows a hard view (`HARD_VIEW_QUALITY`, #5372), so the helper says
  so. It cannot offer an inconclusive answer: a vote here is good or bad, and
  there is no way to move on without one. The helper tells the user the pair
  is hard and leaves the vote as it is.
- **The reference is a known quantity here.** ENFSI's order-of-viewing rule
  protects a blurry questioned image from a sharp reference. The helper
  cannot hide the reference, but it can prefer a reference no sharper than
  it needs to be, and the fitness notice covers the rest.

### What the repo already has

- **The voting view.** `vt-center-panel` hosts every media type; a face has
  no viewer of its own and renders in `vt-image-viewer`, with zoom and rotate
  (`frontend/src/app/components/center-panel/`). Vote keys are in
  `keyboard.service.ts`.
- **A reusable viewer.** `vt-image-viewer` takes a `media`, a `datasetId` (so
  it can show an item of another dataset) and box inputs.
  `modals/source-photo-modal/` already hosts a second one. There is no
  side-by-side component.
- **The photo a crop came from.** `GET /api/medias/<id>/source` returns the
  parent photo's item and the crop's `source_box`, when the sibling Image
  dataset is loaded (`vtscore/datasets/source_item.py`).
- **The compare layout.** On a Face dataset with a detector active, the
  center panel is a two-column grid: `vt-face-reference` on the left, the
  candidate on the right, each a `vt-image-viewer` with its own toolbar
  (`components/center-panel/face-reference/`). The reference is a set the
  user steps through with **Next** or `R`: the Good pile once there is one,
  for each candidate the faces turned its way first and by `Face Quality`
  among those (#5670), the detector's example pictures before
  (`face-reference.service.ts`). One can be pinned for the page's lifetime.
  The Compare toggle is the per-user `face_compare` setting.
- **Five landmarks on every crop.** `image2face` gets two eyes, a nose tip
  and two mouth corners per face (`imagekit/face/records.py`), uses them to
  level the crop (`align="roll"`, #4777), and keeps them on the item as
  `face_landmarks`, fractions of the crop as stored, served on
  `POST /api/medias/batch` (#5644). The converter computes no pose; the
  compare layout reads a rough turn of the head from the five points
  (`landmarkTurn`, `face-checklist/face-regions.ts`).
- **Crop padding of 0.25** of the face box on every side by default. Ears and
  hairline are often at or past the crop edge.
- **A wider view.** The compare layout's **Wider** toggle shows both faces
  from their parent photos, framed on twice the crop's box (`WIDER_SCALE`,
  `source-photo.service.ts`), through `vt-image-viewer`'s `fitBox`. The photo
  is shown as taken: the converter does not keep the roll angle it levelled
  the crop by, so the photo cannot be turned to match.
- **The two faces registered, zoom linked.** When both crops have landmarks,
  the reference is resized and shifted so the eye midpoints coincide and the
  eye-to-mouth distances match, and the two panes zoom and pan together
  through a `ViewLink` the centre panel hands both viewers
  (`components/center-panel/image-viewer/face-link.ts`). The candidate's
  viewer leads; a view is a point of *face space* (eye midpoint at the origin,
  one eye-to-mouth distance per unit) at the pane's centre plus a scale, so
  zooming both panes to a region is one `FaceView` handed to the lead.
  Rotation stays per pane. Not under **Wider**, where the photos are not
  levelled.
- **The checklist.** Down the right of the compare layout,
  `vt-face-checklist` (`components/center-panel/face-checklist/`) lists five
  rows in diagnostic order, worded from the FISWG list: marks, ears, nose,
  eyes and brows, then mouth, chin and face outline. Under them is the line
  on what not to weigh. Nothing is ticked off or recorded. While the panes
  are linked, a row is a button that zooms both to its part, a rectangle of
  face space placed from the landmarks (`face-checklist/face-regions.ts`); a
  row whose part is off either crop is greyed. Its fitness notice
  says when the candidate or the reference is a hard view, from each face's
  `face_hard_view` on `POST /api/medias/batch`, or for an example picture
  `GET /api/server-media-files/<f>/face-hard-view`.

## Design

### Registering under Wider

Under **Wider** the panes show the unlevelled photos, so the landmarks, which
are measured on the levelled crop, do not land on them, and the two panes
zoom on their own. Registering there needs the crop's roll angle, which the
converter does not keep yet.

## Open decisions

These are the owner's; the recommendation is first in each.

- **Does the checklist record anything?** Recommended: no. Recording
  per-feature answers would give data on which checks users rely on, at the
  cost of a slower vote.
- **Does voting need an "I can't tell" answer?** Recommended: not in this
  work. The ENFSI manual wants an inconclusive level on any scale of
  conclusions, and the helper makes hard pairs more visible, which is where
  users would reach for one. But votes are good or bad today. A graded vote
  (say -2 to 2, with 0 for "can't tell") makes the label a number, so the
  detector becomes a regression and needs a new kind of detector, not a
  change to this UI. That is its own design.
- **Re-import, or re-detect on demand, for landmarks on existing datasets?**
  Recommended: re-import. Re-detection on demand needs the parent photo and
  the face provider loaded at vote time.

## Open work

Each item is an issue; the issue holds its scope. An arrow means "must land
before"; the dotted one is a soft dependency, for the checklist rows' click to
zoom only.

```text
#5642 compare layout ──> #5643 checklist strip ┄┄┄┄┄┄┐
        │                                             ┆
        ├──────────────> #5645 wider view             ▼
        │                                     #5647 size and position,
        └───────────────────────────────────> linked zoom
                                                      ▲
#5644 keep landmarks ─────────────────────────────────┘
```

#5642 and #5644 can start together. Nothing else is blocked on #5645.

<!-- item-sep -->

- [ ] #5642 — Compare layout with a reference face beside the candidate (Opus)

<!-- item-sep -->

- [ ] #5643 — Checklist strip of what to compare, in diagnostic order (Sonnet)

<!-- item-sep -->

- [ ] #5644 — image2face: keep each face's five landmarks on its item (Sonnet)

<!-- item-sep -->

- [ ] #5647 — Match the two faces' size and position, with linked zoom (Opus)

<!-- item-sep -->

- [ ] #5645 — Wider view of both faces from their parent photos (Sonnet)

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->
