# Face Value on the face path as it ships: buffalo_m sessions, paired with buffalo_l — 2026-10-27 (#5336)

**The face review re-run on buffalo_m, the pack the owner chose in #5330, gives `hold-the-line`'s Face
Value slide the face path the app runs: SCRFD `det_2.5g` + ArcFace `w600k_r50`.** From one starting photo
at 1024 px, the session reaches F 0.97 / 0.91 / 0.93 by 50 votes at β ¼ / 1 / 4, and the end-of-run check
leaves it there. Paired person by person against #5321's buffalo_l sessions, buffalo_m costs about 0.01 of
the session mean at every preset. That is the size #5330's static read predicted (−0.009 AP).

| one starting photo, 1024 px | β ¼ | β 1 | β 4 |
|---|---:|---:|---:|
| F at the start photo (the example sort) | 0.86 | 0.64 | 0.76 |
| F at 50 votes | 0.97 | 0.91 | 0.93 |
| F after the check | 0.97 | 0.91 | 0.93 |
| precision / recall at 50 votes | 0.98 / 0.95 | 0.87 / 0.97 | 0.56 / 0.98 |
| faces returned at 50 votes, mean | 3.7 | 4.6 | 7.7 |

## Against buffalo_l

Both sets are 562 people (561 from four photos, #5328), 150 votes, one seed, the same identities and
starting photos. buffalo_l is #5321's `ik` arm, run at `e7bc6fc1b`; buffalo_m ran at `76274d50b`. The two
commits differ in the embedder's name (#5327) and the pack, so the pairing is across a merge: read the
deltas as the pack's cost plus anything else #5327 moved.

| | K | β ¼ | β 1 | β 4 |
|---|---:|---:|---:|---:|
| session mean F, votes 1–150, buffalo_m − buffalo_l | 1 | −0.0057 ± 0.0037 | −0.010 ± 0.0053 | −0.0082 ± 0.0035 |
| | 4 | −0.0010 ± 0.0017 | −0.0093 ± 0.0046 | −0.0076 ± 0.0023 |
| F after the check, change | 1 | −0.0057 ± 0.0039 | −0.0063 ± 0.0056 | −0.0047 ± 0.0036 |
| | 4 | +0.0003 ± 0.0017 | −0.0086 ± 0.0050 | −0.0038 ± 0.0024 |
| final AP, change | 1 | −0.0090 ± 0.0036 | −0.0089 ± 0.0036 | −0.0090 ± 0.0036 |
| | 4 | −0.0041 ± 0.0017 | −0.0041 ± 0.0017 | −0.0041 ± 0.0017 |
| failed sessions, buffalo_l → buffalo_m | 1 | 0 → 3 | 0 → 3 | 0 → 3 |
| | 4 | 0 → 0 | 0 → 0 | 0 → 0 |

- **Over half the one-photo AP cost is three people.** The same three people fail at every β from one
  starting photo: final AP about 1.0 on buffalo_l and 0.001 on buffalo_m. They contribute 3/562 ≈ 0.0053
  of the −0.009. A failed session is final AP below 0.1 while the person's full-label ceiling reaches 0.5
  (#4777's bar). From four photos none of them fails, which fits #5330's finding: buffalo_m's loss sits in
  the few photos its landmarks misplace, and one bad starting photo sinks a session that four photos
  carry.
- **The rest is small and even.** Win and loss shares are close (β 1, one photo: 43% of people gain, 48%
  lose), and faces returned at 50 votes barely move (4.5 → 4.6 at β 1).

## Files and reproduce

`curves.csv` (the objective at every vote, per K and β; `control` = buffalo_l, `arm` = buffalo_m) and
`ab.csv` (the session sets' aggregates) are `scripts/experiments/fhibe/weak_check_4778.py`'s output at
1024 px. The per-run table names FHIBE subject ids and stays on the GRID with the run dirs.

```bash
SOTA_FACE_ARM=ik SOTA_FACE_DATE=2026-10-27 SOTA_FACE_TAG=5336 sota_face.sh dirs   # grids 2026-10-27-prep-ik-k{1,4}
SOTA_FACE_ARM=ik SOTA_FACE_DATE=2026-10-27 SOTA_FACE_TAG=5336 sota_face.sh tasks  # minus #5328's 12 unopenable K=4 tasks
CALIB_PARTITION=cpu sota_face.sh pack 4; sota_face.sh pack 2                      # one SOTA_FACE_CLAIMS_DIR
sota_face.sh baseline; sota_face_analyze.sh <runs>/2026-10-27-5336-ik-analysis 1  # and 4
python weak_check_4778.py --runs <runs> --faces fhibe_faces_ik --size 1024 --out <dir> \
    --control 2026-10-26-5321-ik::<runs>/2026-10-26-5321-ik-analysis \
    --arm 2026-10-27-5336-ik::<runs>/2026-10-27-5336-ik-analysis
```

#5321's click-0 baseline files its rows under the embedder's old name, `imagekit_face`. The pairing reads
`face`, so it ran on a private copy of that baseline with the column renamed. Without the rename the
control's example sort reads as missing.

Face Value is drawn by `slides/figs/src/make-sota-figs.py` from `curves.csv` and `ab.csv`. The 640 px and
four-photo sets ran too (`<runs>/2026-10-27-5336-ik-*`), and #5325 can pair against them as its unrefit
control on buffalo_m.
