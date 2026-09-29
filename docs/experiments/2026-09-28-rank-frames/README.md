# Rank frames: where the positives sit, for precision-floor studies off the GRID

Compact exports of the #4220 and #4222 precision frames. A precision-floor rule
is judged by one thing: of the top *k* items of a corpus, how many are
positive. Answering that needs only **the ranks of the positives** in each
corpus's ranking, a few dozen integers per frame, instead of the ~0.6 MB
precision frame each row was cut from. So a study of cut rules that read the
top of the ranking (random verification, audit sampling, #4224 follow-ups) can
run anywhere, from these files, with numpy and pandas.

Built by `scripts/experiments/calibration/export_rank_frames.py` from:

| file | pool prevalence | source run | frames |
|---|---|---|---|
| `rank_frames_0p44pct.csv.gz` | 0.44% (COCO Better's default) | #4220, today's app (r7) | 2,662 |
| `rank_frames_0p1pct.csv.gz` | 0.1% (positives thinned) | #4222, today's app (G = 3) | 2,328 |
| `rank_frames_5pct.csv.gz` | 5% (haystack thinned) | #4220, today's app; the corpus thinned to 5% as its "same" scenario did | 2,871 |

These prevalences are scenarios, not claims about users' collections. All three
use COCO Better, SigLIP, binary voting, 144 class@band cells × 5 seeds, with
checkpoints at votes 25, 50, 100 and 150. Starved cells (no detector) are left
out.

## Columns

- `world`, `category`, `band`, `seed`, `t`: the frame. `t` is the number of
  votes cast so far.
- `n_corpus`, `n_pos`: the **corpus** is the cell's held-out test half, the
  images the detector is run over. It is never voted on, so it plays "media
  nobody voted on".
- `pos_ranks`: space-separated 0-based ranks of the corpus's positives when the
  corpus is sorted by the session's final model score, highest first (ties
  broken stably). The precision of the top *k* is
  `count(pos_ranks < k) / k`, and its recall is `count(pos_ranks < k) / n_pos`.
- `n_votes`, `n_vote_pos`: votes cast so far, and how many were Good.
- `n_cal_pos`: positives among the calibration folds' held-out votes. The
  shipped estimator's gate is 10.
- For X in 25/50/75%, the #4220 precision-floor estimator
  (`vtscore.training.thresholds.precision_floor_cut`, library defaults):
  - `shipped_status_xNN`, `shipped_k_xNN`: as the library ships it, with the
    reference pool including the voted items. `status` is `promised`,
    `unreachable` or `insufficient_evidence`; `k` is the number of corpus items
    returned (0 when nothing is promised).
  - `consistent_status_xNN`, `consistent_k_xNN`: the same call with the voted
    items removed from the reference pool, as the fold haystacks already do.
- **Best effort** (#4267: what a floor returns if it never refuses), per X:
  - `fallback_k`: the app's own line, the shipped cut at its Inclusion. It is
    what a floor that promises nothing falls back to today (#4247). One column,
    independent of X.
  - `best_{med,lb}_{shipped,consistent}_k_xNN`: the same estimator's curve
    (`precision_lower_bound_curve`), read **without the gate**. `med` reads
    it at the median of the bootstrap refits, a point estimate; `lb` reads it
    at the shipped 10th percentile. `k` is the largest top k the curve puts
    at >= X. Where the curve never gets there, it is the k with the highest
    estimate. Where no curve can be formed (no calibration positive), it is
    `fallback_k`.
  - `best_*_how_xNN`: which of those three it was: `reached`, `short` or
    `none`.

## Sanity numbers (X = 50%)

Use these to check your reading of the files. A promise is **broken** when the
precision of the returned top *k* is below X.

| file | frames past the gate | shipped: promised / broken | consistent: promised / broken |
|---|---|---|---|
| 0.44% | 126 | 116 / 6.0% | 125 / 88% |
| 5% | 1,227 | 605 / 1.8% | 1,152 / 87% |
| 0.1% | 0 | 0 / – | 0 / – |

Best effort, scored on **every** frame by mean shortfall, max(0, X - precision
of the returned top k), where returning nothing counts as X (0.6% of frames at
0.44%, 3.2% at 0.1%):

| file | today: promise, else `fallback_k` | `best_med_shipped` | `best_lb_consistent` | read the top 32 |
|---|---|---|---|---|
| 0.44% | 0.46 | 0.31 | 0.30 | 0.066 |
| 5% | 0.17 | 0.069 | 0.15 | 0.004 |
| 0.1% | 0.50 | 0.43 | 0.41 | 0.15 |

"Read the top 32" returns the longest prefix of the top 32 at >= X, else the
prefix with the highest precision.

The shipped estimator's apparent safety is an artifact: its reference pool
includes the voted items, scored by a model trained on them, which pushes
unseen positives down the percentile scale. With a consistent pool, the same
estimator breaks most of its promises. See #4221's comment of 2026-09-28.
