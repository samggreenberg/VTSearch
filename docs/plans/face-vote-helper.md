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
    tools "which preserve shapes".
  - **Photo-anthropometry** (measuring distances and angles between
    landmarks) "shall not be used". It "does not produce consistent results
    and may be misleading".
  - Just looking at the whole face (*holistic comparison*) "is not recognized
    as a method".
  - Classification schemes ("round face, Roman nose") are "not best
    practice", because observers disagree on them.
  - The process is ACE-V: **A**nalysis (is the image fit to compare at all?),
    **C**omparison, **E**valuation, **V**erification (a second examiner).
- [ASTM E3149 / the FISWG feature list](https://store.astm.org/e3149-26.html)
  is the list of parts the method walks: 19 components, top of the face to
  bottom, each split into finer detail. The ASTM text is paid; this plan
  takes the component names from the free stability standard below, which
  follows the same list.
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
  Comparison*](https://www.mvr.bg/upload/19089/enfsi-bpm-di-01.pdf) is the
  European counterpart. It was not read for this plan; it is listed for
  whoever takes up the checklist item.

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
- **A checklist, short and in diagnostic order.** Marks and ears first, face
  shape and mouth last, with hair, facial hair and skin tone named as things
  *not* to weigh. It is a prompt, not a form: the vote is still one key.
- **Alignment as an aid, with shape-preserving transforms only.** Scale, turn
  and shift; never warp. An overlay is offered only when the two faces are
  near the same viewpoint.
- **No measurements.** No distances, ratios or angles shown as numbers. That
  is the method the standard forbids.
- **Say when a pair cannot be compared.** ACE-V starts with fitness. The app
  already knows a hard view (`HARD_VIEW_QUALITY`, #5372); the helper says so
  and reminds the user that moving on without voting is a valid answer.

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
- **Candidates for a reference.** A detector's media examples
  (`PUT /api/detectors/<name>/examples`) and its Good pile (`GET /api/votes`,
  `vote-state.service.ts`). There is no "reference face" concept.
- **Five landmarks, used once and dropped.** `image2face` gets two eyes, a
  nose tip and two mouth corners per face
  (`imagekit/face/records.py`), uses them to level the crop (`align="roll"`,
  #4777), and does not keep them. No pose is computed.
- **Crop padding of 0.25** of the face box on every side by default. Ears and
  hairline are often at or past the crop edge.

## Design

### The compare layout

On a Face dataset with an active detector, the center panel splits in two:
the **reference** on the left, the **candidate** on the right, each a
`vt-image-viewer`. The candidate keeps everything it has today. The layout is
a toggle in the panel's toolbar, on by default for faces, and its state is a
per-user setting.

Zoom and pan are **linked**: zooming to the candidate's left ear zooms the
reference to the same place. This is what makes part-by-part comparison
quick, and it depends on the two faces being registered (below).

### Choosing the reference

A person is not one photo, so the reference is a small set the user cycles
through, not a single image.

- Before any vote: the detector's media examples.
- After votes: the Good pile, best `Face Quality` first, hard views last.
- The user can **pin** one. A pinned reference stays until unpinned.
- A key steps to the next reference without leaving the candidate.

The pin is detector state that names a media item. Whether it is saved with
the detector or kept for the session is an open decision (below).

### Registering the two faces

Both crops are already levelled. What is missing is scale and position: one
face may fill its crop and the other sit small in a corner. With each face's
five landmarks, the frontend scales and shifts the reference so the two eye
midpoints coincide and the eye-to-mouth distances match. That is a similarity
transform, the shape-preserving kind the standard allows.

This needs the landmarks kept. `image2face` would store them on the item as
five points in crop fractions, beside `source_box`. They are geometry, not an
embedding, so the no-persisted-vectors rule does not apply. Existing Face
datasets have none and would show the unregistered layout until re-imported.

### Overlay

A hold-to-blink key swaps the reference into the candidate's pane, registered.
It is offered only when both faces look frontal. With no pose estimate, the
test is a proxy from the landmarks: how far the nose tip sits from the
midline between the eyes. This is this plan's own inference, not a method
from the references, and it needs checking against real crops before it
gates anything.

### Wider view

Ears and hairline are the parts most likely to be cut off. A **Wider** toggle
re-renders each side from its parent photo with a larger box around the same
face. It needs the sibling Image dataset loaded and is hidden otherwise.

### The checklist

A narrow strip beside the pair, in this order:

1. **Marks** — moles, scars, freckles. Same place on both?
2. **Ears** — lobe attached or free, rim shape, how far they stand out.
3. **Eyes and brows** — lid folds, brow shape (not brow grooming).
4. **Nose** — tip, nostrils, bridge.
5. **Mouth, chin, face outline** — weakest evidence; expression moves them.

Under it, one line: *don't weigh hair, beard, makeup or skin tone.*

Clicking an item zooms both panes to that region, located from the
landmarks. Items are not ticked off and nothing is recorded: the research
supports directing attention, and a form would slow every vote. A row is
greyed when its region falls outside either crop.

### Fitness notice

When the candidate or the reference is a hard view, the strip says so in one
line, and that moving on without voting is fine.

## Open decisions

These are the owner's; the recommendation is first in each.

- **Is the pinned reference saved with the detector?** Recommended: session
  only at first. Saving it adds a field to detector JSON for a convenience.
- **Does the checklist record anything?** Recommended: no. Recording
  per-feature answers would give data on which checks users rely on, at the
  cost of a slower vote.
- **Re-import, or re-detect on demand, for landmarks on existing datasets?**
  Recommended: re-import. Re-detection on demand needs the parent photo and
  the face provider loaded at vote time.

## Open work

None of these is filed as an issue yet.

<!-- item-sep -->

- **Compare layout** — the split panel, the reference set drawn from
  examples and the Good pile, pin and cycle, and the per-user toggle. Usable
  on its own, unregistered. *Recommended Claude model: Opus, for the
  center-panel reactivity.*

<!-- item-sep -->

- **Checklist strip** — the five rows, the do-not-weigh line and the fitness
  notice, as static guidance. Read the ENFSI manual first and reconcile the
  wording with it. After *Compare layout*. *Recommended Claude model: Sonnet.*

<!-- item-sep -->

- **Keep the landmarks** — `image2face` stores the five points in crop
  fractions; the media API serves them. *Recommended Claude model: Sonnet.*

<!-- item-sep -->

- **Registration and linked zoom** — scale and shift the reference from the
  landmarks; link zoom and pan; checklist rows zoom to their region. After
  *Keep the landmarks*. *Recommended Claude model: Opus.*

<!-- item-sep -->

- **Wider view** — both sides re-rendered from their parent photos with a
  larger box. After *Compare layout*. *Recommended Claude model: Sonnet.*

<!-- item-sep -->

- **Overlay** — hold-to-blink, gated on the frontal proxy. The proxy has to
  be checked on real crops first. After *Registration and linked zoom*.
  *Recommended Claude model: Sonnet.*

<!-- item-sep -->

- **Does it help?** — the research warns that users rate a helper well
  regardless. A measurement would compare vote accuracy with and without the
  helper on FHIBE, where identity is known. It needs people voting, so it is
  a study to design, not a harness run.

<!-- item-sep -->
