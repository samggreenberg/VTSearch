# Face Value on buffalo_s, the GRID default face pack — 2026-10-28 (#5348)

**`hold-the-line`'s Face Value slide now charts the face path the app runs: imagekit on InsightFace
buffalo_s (SCRFD `det_500m` + ArcFace `w600k_mbf`, the GRID default since #5341), with the centroid's
face-only line at z 5.5 (#5325).** From one starting photo at 1024 px, the session opens at F 0.89 / 0.85 /
0.84 on that photo's own sort, at β ¼ / 1 / 4. By 50 votes it reaches 0.96 / 0.92 / 0.92, and the
end-of-run check leaves it there.

| one starting photo, 1024 px | β ¼ | β 1 | β 4 |
|---|---:|---:|---:|
| F at the start photo (the example sort) | 0.89 | 0.85 | 0.84 |
| F at 50 votes | 0.96 | 0.92 | 0.92 |
| F after the check | 0.96 | 0.92 | 0.92 |
| precision / recall at 50 votes | 0.97 / 0.87 | 0.93 / 0.93 | 0.63 / 0.96 |
| faces returned at 50 votes, mean | 3.4 | 3.9 | 6.6 |
| session mean F, votes 1–150 | 0.96 | 0.91 | 0.92 |

There are 562 people, one seed each, 150 votes. The person has 3.8 withheld photos on average. Six
one-photo sessions fail at every β: final AP is below 0.1 while the person's full-label ceiling reaches 0.5
(#4777's bar). In those six, the starting photo's vector misses the person. They stay in every mean above.
From four photos (559 people, `K = 4` rows) no session fails.

## The sessions

These are #5325's arm, `2026-10-27-5325z55-ik`, run at `5e50355ac` with
`CALIB_CENTROID_LINE=count_gap_floor_skip25_b1_z5.5` on the `fhibe_faces_ik_1024` cells as rebuilt on
buffalo_s (#5341). #5369 shipped that rule as the face default. From a frozen tree with no override set, 18
cells matched the arm in every value. So these are the sessions the app runs now, and no new run was
needed: #5325 covers one photo at 1024 px for β ¼, 1 and 4.

The slide charts buffalo_s alone. The files here carry no control arm.

## Files and reproduce

- `curves.csv`: the objective at every vote, per K and β. These are the `K, beta, vote, arm` columns of
  `2026-10-27-face-lines-5325/curves.csv`, with `arm` renamed to `fbeta`.
- `ab.csv`: the set's aggregates per K and β. These are that report's `ab.csv` `arm_*` columns with the
  prefix dropped, and the vote windows (`1_25` …) renamed `mean_1_25` …

Both come from `scripts/experiments/fhibe/weak_check_4778.py` at 1024 px (#5325's reproduce block).
Face Value is drawn by `slides/figs/src/make-sota-figs.py` from these two files.
