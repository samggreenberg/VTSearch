# A hard starting face: the Face Quality cut — 2026-10-28 (#5372)

**A one-photo face search started from a face with Face Quality under 14 tends to miss the person. The app
now warns when every face picked to start a detector is under that cut. The warning is advice and never
blocks Create.** The cut comes from buffalo_s (`oss-arcface-mbf:1`, the GRID default). It holds at both crop
sizes, and it never fires on a four-photo start.

| one starting photo, β 1 | 1024 px | 640 px |
|---|---:|---:|
| sessions that fail at every β | 6 | 6 |
| AUC: low start quality ranks the failures first | 0.998 | 0.999 |
| sessions with a start under 14 | 7 (1.2%) | 7 (1.2%) |
| … of which fail at every β | 5 | 6 |
| F at 50 votes, start under 14 / the rest | 0.16 / 0.93 | 0.04 / 0.91 |
| pile faces under 14 | 2.1% | 1.9% |

Face Quality is the L2 norm of the raw ArcFace embedding (`embedding_quality` in `imagekit/face/records.py`).
It measures how readable the face is (pose, cover, blur, size), not who it is: `w600k_mbf` is plain ArcFace,
not MagFace. The pile's median is 23.

## Where the cut came from

#5348 found it on one set: one photo at 1024 px, β 1, 562 sessions. There, the six failures (#4777's bar:
final AP below 0.1 while the person's full-label ceiling reaches 0.5) all started from that person's
lowest-quality photo. Six is few, so this reads the cut on every other set of #5325's shipped arm
(`2026-10-27-5325z55-ik`, the sessions the app runs since #5369). No new sessions were run.

- **640 px.** These are the same 562 people, but the face is cut and embedded again from a smaller photo,
  and the failures change: 4 of 6 are the same people, and 2 are new (`16c82ff4`, `548d8580`). All six start
  under 14 (10.7–14.0). The cut of 13 read at 1024 px would catch only 3 of them, so the cut is 14, not 13.
- **Every β.** A failure fails at every β, so the AUC (0.998 / 0.999) and the flags are the same at β ¼, 1
  and 4.
- **Four photos.** No four-photo session fails (559 at 1024 px, 558 at 640 px), even where the worst of the
  four is under 12 (11 sessions). The best of the four is never under 17. So one clear face is enough, and
  the warning keys on **the best of the starting faces**: it shows only when every face picked is under 14.

| cut (one photo, β 1) | 13 | 14 | 16 |
|---|---:|---:|---:|
| flagged, 1024 / 640 | 5 / 3 | 7 / 7 | 16 / 17 |
| failures caught, of 6 / 6 | 5 / 3 | 5 / 6 | 6 / 6 |
| F at 50 votes of the flagged, 1024 / 640 | 0.00 / 0.00 | 0.16 / 0.04 | 0.48 / 0.50 |

At 14, 11 of the 14 flagged sessions fail. The other three end at F 0.29, 0.86 and 0.29 at 50 votes, so
the warning is right about those too. At 16 it would flag 3% of sessions and catch every failure, but half
of what it flags would be fine. The one failure 14 misses is `b2e04b02` at 1024 px (15.75), a cap and bowed
head; at 640 px its face scores under 14 and is caught.

One more pattern, for the record only: in four-photo sessions at β 4, a start set whose worst face is under
14 reaches F 0.80 at 50 votes, against 0.93 for the rest (1024 px; 0.83 / 0.92 at 640 px). At β 1 the gap is
gone (0.94 / 0.93). These people may simply be harder to find, so this is not a cost of the hard face, and
the warning does not key on it. Telling the two apart is #5376.

The six 1024 px failures were checked in the full-size photos in #5348. Each shows the right person in a hard
view: a full profile, two lying down, a bottle over the face, a cap and bowed head, a head turned away. The
labels are right. The two lying-down faces are rotations, and re-detecting them upright is #5373.

## What the app does

`vtscore.media.face.imagekit_bridge.HARD_VIEW_QUALITY` holds the cut per embedding space:
`{"oss-arcface-mbf:1": 14.0}`. A norm means nothing across models, so a face in any other space (buffalo_l/m's
`oss-arcface-r50:1`, an `ext` provider) gets no verdict and no warning.

When New Detector saves a starting face, the response carries `face_hard_view`:

- **Pick from Dataset… and Use as detector seed** read the quality the converter stamped on the face. If the
  face is cropped first, or carries no stamp, the provider reads the saved picture.
- **A face picture dropped on the box** is read by the provider, as the face embedder would read it (its
  first detection).

The Face tab's example stack then shows, under the faces: "This photo is a hard view of the face (turned,
covered or small). Searching from it may miss this person; a clearer photo will help." Adding one clear face
clears the warning, as the four-photo sessions say it should. A face with no verdict counts as clear.

## Files and reproduce

- `cuts.csv`: per K, size, β, start (`only`, or the `min` / `max` of four) and cut: sessions flagged, failures
  among them, F at 50 votes flagged and the rest, and the share of the pile's faces under the cut.
- `auc.csv`: per K, size and β, the failures and the AUC of a low start quality.
- `fails.csv`: the sessions that fail at every β, with their start quality.

All three come from `scripts/experiments/fhibe/face_quality_5372.py`, run on a compute node with the app venv:

```
python scripts/experiments/fhibe/face_quality_5372.py --out docs/experiments/2026-10-28-face-quality-5372
```
