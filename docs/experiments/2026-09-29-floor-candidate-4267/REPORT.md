# How big should the spot check's candidate be at a low floor?

**At 10%, start at the top 128 and at 25%, the top 64; audit 5 picks a round,
and halve to 32 on a failed round.** The owner's #4267 ruling earns the precision
floor's promise with one spot check of the top 32 (#4257's `a:top32`). That
candidate is fixed, so every floor returns the same 32 items when promised, and a
lower floor only makes the promise easier to earn. With a 10% preset added "for
users willing to dig through results" (owner, 2026-09-29), the owner ruled that
**the candidate grows as the floor falls**, and picked this schedule from the grid
below. At COCO Better's default pool (0.44%) and X = 10% it spends **12 votes**
(90th percentile 15). It promises in **59%** of sessions (the fixed top 32: 52%),
returns **68 items** on average when it does (32), and recovers **0.63** of the
oracle's recall (0.43). It hands over **17** matches per session that the user
never saw (11) and breaks **0.07%** of its promises. It is the first rule in this
series to beat reading the top 32 yourself on recall at the default pool:
**+0.058 ± 0.006**, for 12 votes against 32. At 25% it spends 9 votes and
recovers 0.55 (0.44). Above 50% nothing changes.

**Then the promise stopped being make-or-break** (owner, 2026-09-29): "Do your
best. How close did we get?" The line always keeps the set the check ended on, and
the control shows a **likely range** for how much of it is right. The range's
tails are at the level the check's rounds are tested at, so a check confirms X
exactly when the range's lower end clears X. That range contains the true precision
in **99%** of sessions at every floor and richer pool. At 10% and 0.44%, a session
now returns **53 items** on average, **40%** right, and meets the floor in **84%**
of sessions. When the check falls short, the top 32 it ends on is 18% right on
average, and the range it shows is about 2–55%. See
[Do your best](#do-your-best-how-close-did-we-get) below.

Part of #4224; follows #4257 and the #4267 ruling. Data: the #4224 rank frames
([`2026-09-28-rank-frames`](../2026-09-28-rank-frames/README.md)). Analysis:
`scripts/experiments/calibration/analyze_floor_candidate_4267.py`, which reuses
`analyze_random_verification.py`'s audit simulation unchanged (planted-answer
test: `selftest_analyze_floor_candidate_4267.py`). Nothing here needed the GRID.

## What was run

- **Frames.** The #4257 frames: 7,861 frames of today's app on COCO Better
  (SigLIP, binary voting), 144 class@band cells × 5 seeds, at t = 25–150 votes, in
  three pool scenarios (0.44%, 5%, 0.1%). Audits are exact, simulated as
  hypergeometric draws from each frame's positive ranks: 20 draws a frame, seed
  4267.
- **The bound.** As in #4257: promise a set if the one-sided Clopper–Pearson lower
  bound on its audited hit rate is ≥ X at level α = 5%. A rule with R rounds runs
  each at α / R.
- **Rules** (`summary.csv`, `part = grid`, at X = 10%, 25% and 50%):
  - `a:topK`, one round of m picks from the top K, for K = 32, 64, 128;
  - `b:topK>32`, m fresh picks a round from the top K, halving on failure down to
    32, for K = 64 to 512.
- **The schedule** (`part = schedule`, at all five presets):
  - The candidate starts at K(X) = 32 · 2^max(0, ⌊log₂(0.5 / X)⌋): the top 128 at
    10%, the top 64 at 25%, and the top 32 at 50% and above.
  - It halves on a failed round down to 32, so it runs R = log₂(K / 32) + 1
    rounds.
  - Each round audits m(X) = max(5, ⌈ln(α / R) / ln X⌉) fresh picks. That is the
    fewest that can reach X at level α / R when all are right. At R = 1 it is the
    ruled m(X): 5 up to 54.9%, 11 at 75% and 29 at 90%. At 10% and 25% it is 5.
- **Reference:** reading the top 32 yourself (`read:32`). You vote on all 32 and
  get the longest prefix at ≥ X. It is never wrong and costs 32 votes.
- **Metrics:** those of #4257's `summary.csv`, plus `returned`, the mean size of a
  promised set.

## The schedule, at every preset

X, α = 5%, pooled over t. "Δ recall vs reading" is paired on the same frames
(cluster SE in brackets).

| pool | X | candidate | picks a round | votes, mean (p90) | promised | broken, of promises | recall ÷ oracle | unseen matches | returned | oracle's cut (median) | Δ recall vs reading the top 32 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.44% | 10% | 128 → 32 | 5 | 12 (15) | 59% | 0.07% | 0.63 | 17 | 68 | 370 | +0.058 (0.006) |
| 0.44% | 25% | 64 → 32 | 5 | 9.1 (10) | 43% | 0.14% | 0.55 | 12 | 45 | 112 | −0.045 (0.006) |
| 0.44% | 50% | 32 | 5 | 5 | 28% | 0.29% | 0.38 | 6.9 | 32 | 38 | −0.15 (0.006) |
| 0.44% | 75% | 32 | 11 | 11 | 20% | 0.26% | 0.35 | 4.0 | 32 | 14 | −0.16 (0.007) |
| 0.44% | 90% | 32 | 29 | 29 | 12% | 0% | 0.26 | 0.37 | 32 | 6 | −0.17 (0.008) |
| 5% | 10% | 128 → 32 | 5 | 10 (15) | 90% | 0% | 0.73 | 31 | 74 | 490 | +0.20 (0.005) |
| 5% | 25% | 64 → 32 | 5 | 8.1 (10) | 71% | 0.03% | 0.61 | 23 | 49 | 184 | +0.029 (0.007) |
| 5% | 50% | 32 | 5 | 5 | 50% | 0.16% | 0.42 | 13 | 32 | 86 | −0.18 (0.005) |
| 0.1% | 10% | 128 → 32 | 5 | 15 (15) | 10% | 1.6% | 0.17 | 0.67 | 38 | 60 | −0.36 (0.011) |
| 0.1% | 25% | 64 → 32 | 5 | 10 (10) | 0.79% | 9.5% | 0.017 | 0.056 | 36 | 16 | −0.41 (0.015) |
| 0.1% | 50% | 32 | 5 | 5 | 0.06% | 85% | 0.001 | 0.004 | 32 | 6 | −0.36 (0.015) |

- **At low floors the honest set is large, and the fixed top 32 leaves most of it
  behind.** The oracle's cut is 370 items at 10% and 112 at 25% (0.44%), against
  38 at 50%. Growing the candidate turns that into returned items: 68 at 10%,
  where the fixed top 32 returns 32.
- **Each extra round costs 5 votes, and pays mostly at 10%.** At 10% the schedule
  beats reading the top 32 on recall at both richer pools. At 25% it trails
  reading at 0.44% (−0.045), but costs 9 votes against 32, and it hands over 12
  matches the user never saw where reading hands over none.
- **At 0.1% the rare promises stay mostly luck.** At 25% the schedule promises in
  0.79% of sessions, and 9.5% of those break. At 50%, 85% of its 26 promises break
  (#4267's caveat). Per session, the breaks stay far under α: 0.17%, 0.08% and
  0.05% of all frames. As at 50%, the no-cause failure wording covers this.

## Do your best: how close did we get?

After #4267 the owner changed what the check is for: not a make-or-break promise,
but "do your best, and say how close we got" (2026-09-29). So:

- **The line always keeps the set the check ended on.** That is the promised set
  after a passed check, and the top 32 after a short one. Before any check, and in
  headless runs, it is the schedule's starting candidate, unchecked. Nothing falls
  back to today's cut any more.
- **The control shows a likely range** for how much of that set is right. It is a
  Clopper–Pearson interval from the labels inside the set, with each tail at α / R,
  the level each round is tested at (`range_tail`). It is exact for a census.
  Because the lower end is the very bound the check tested, **a check confirms X
  iff the range's lower end is at least X.** The analyzer asserts this on every
  frame and draw.

`best_attempt.csv`, pooled over t (20 draws a frame):

| pool | X | votes | check confirms X | returned | right, mean | set meets X | recall ÷ oracle | range contains the truth | range width | after a short check: right, range |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.44% | 10% | 12 | 59% | 53 | 40% | 84% | 0.70 | 99.3% | 0.63 | 18%, 2–55% |
| 0.44% | 25% | 9.1 | 43% | 38 | 49% | 71% | 0.75 | 99.4% | 0.58 | 27%, 6–64% |
| 0.44% | 50% | 5 | 28% | 32 | 53% | 53% | 0.79 | 98.8% | 0.57 | 38%, 11–73% |
| 0.44% | 75% | 11 | 20% | 32 | 53% | 37% | 0.96 | 99.2% | 0.39 | 42%, 22–65% |
| 0.44% | 90% | 29 | 12% | 32 | 53% | 26% | 1.14 | 100% | 0.23 | 46%, 34–59% |
| 5% | 10% | 10 | 90% | 70 | 56% | 99.7% | 0.76 | 99.3% | 0.70 | 37%, 4–67% |
| 5% | 25% | 8.1 | 71% | 44 | 70% | 98% | 0.71 | 99.5% | 0.61 | 49%, 12–79% |
| 5% | 50% | 5 | 50% | 32 | 78% | 85% | 0.68 | 99.0% | 0.57 | 60%, 19–87% |
| 0.1% | 10% | 15 | 10% | 33 | 16% | 57% | 0.90 | 99.5% | 0.53 | 14%, 2–52% |
| 0.1% | 50% | 5 | 0.1% | 32 | 16% | 0.3% | 1.28 | 99.2% | 0.57 | 16%, 3–60% |

- **The range is honest, but wide.** Five picks can't pin a set's precision down.
  At 50% the range is 0.57 wide on average, and 0.23 wide at 90%, where the check
  reads 29 of 32. That width *is* the answer to "how close did we get?" after 5
  votes.
- **Recall now counts every session.** A short check still returns its top 32, so
  recall ÷ oracle rises from 0.63 to 0.70 at 10% (0.44%). Above 1 at 75% and 90%,
  it means the returned 32 hold more positives than the largest set that really is
  that pure, bought with lower precision. The range says so.
- **Before a check, the starting set is often already there.** With no labels at
  all (headless, or before the user checks), the top 128 at 10% is 22% right on
  average and meets 10% in 76% of sessions at 0.44% (99.5% at 5%). At 50% the top
  32 is 53% right and meets it in 53%. An unchecked set carries no range; the
  control says it is unchecked.

**Why each tail is at α / R, not 5%.** The same labels decide the check and draw
the range. A check that passes in its first round did so partly by luck, so its
labels run high. With a plain 90% range (each tail 5%), at 10% and 0.44% a check
confirmed in its first round showed a range above the truth **11%** of the time.
With each tail at α / 3 it is **4.0%**, and the range is 0.10 wider on average.
Checks that end in rounds 2 and 3 cover the truth 99.5% and 99.8% of the time.
At one round the two are the same range. At 0.1% the rare early passes (0.2% of
sessions) are the check's α failures. No range drawn from their labels can
contain the truth.

| X = 10% | check ended in | share of sessions | range contains the truth | range above the truth |
|---|---|---|---|---|
| 0.44% | round 1 (confirmed at 128) | 12% | 96.0% | 4.0% |
| 0.44% | round 2 (confirmed at 64) | 30% | 99.5% | 0.5% |
| 0.44% | round 3 (32, confirmed or short) | 58% | 99.8% | 0.1% |
| 5% | round 1 | 22% | 97.6% | 2.4% |

## How the schedule was chosen

X = 10% and 25% at 0.44%, α = 5% (`part = grid`). The chosen schedule is in bold.

| X | rule | picks a round | votes, mean (p90) | promised | recall ÷ oracle | unseen | returned | Δ recall vs reading |
|---|---|---|---|---|---|---|---|---|
| 10% | `a:top32` (as first ruled) | 5 | 5 | 52% | 0.43 | 11 | 32 | −0.067 |
| 10% | `a:top64` | 5 | 5 | 36% | 0.44 | 13 | 64 | −0.064 |
| 10% | `a:top64` | 10 | 10 | 50% | 0.58 | 15 | 64 | +0.030 |
| 10% | `b:top64>32` | 5 | 8.2 (10) | 57% | 0.58 | 16 | 52 | +0.030 |
| 10% | **`b:top128>32`** | **5** | **12 (15)** | **59%** | **0.63** | **17** | **67** | **+0.058** |
| 10% | `b:top256>32` | 5 | 17 (20) | 58% | 0.63 | 17 | 75 | +0.060 |
| 10% | `b:top512>32` | 5 | 22 (25) | 58% | 0.64 | 17 | 78 | +0.065 |
| 10% | `b:top128>32` | 10 | 22 (30) | 66% | 0.70 | 18 | 82 | +0.11 |
| 10% | `b:top128>32` | 20 | 39 (59) | 83% | 0.77 | 17 | 84 | +0.15 |
| 25% | `a:top32` (as first ruled) | 5 | 5 | 41% | 0.44 | 9.6 | 32 | −0.10 |
| 25% | **`b:top64>32`** | **5** | **9.1 (10)** | **43%** | **0.55** | **12** | **46** | **−0.045** |
| 25% | `b:top128>32` | 5 | 14 (15) | 44% | 0.58 | 12 | 50 | −0.033 |
| 25% | `b:top64>32` | 10 | 17 (20) | 52% | 0.68 | 13 | 51 | +0.019 |
| 25% | `b:top128>32` | 20 | 49 (59) | 70% | 0.81 | 9.7 | 56 | +0.090 |

- **A single bigger round fails at 5 picks.** At a low floor, a top 64 or 128 is
  far from all positive. The bound needs 3 hits of 5 at 10% and 4 of 5 at 25%,
  and five picks from a big candidate rarely hold them. So `a:top64` at m = 5
  promises less than the fixed top 32.
- **Starting past the knee buys nothing.** Beyond the top 128 at 10% and the top
  64 at 25%, a larger start keeps recall (0.63–0.64 and 0.55–0.58) and adds
  5 votes a round.
- **More picks a round buy recall, at a steep price in votes.** At 10%, 10 picks a
  round reach 0.70 of the oracle for 22 votes, and 20 picks reach 0.77 for 39. The
  owner kept 5 picks a round, the budget #4267 ruled for 50%.

## By object size (0.44%)

| X | band | reachable | promised | recall ÷ oracle | unseen matches | returned | votes | oracle's cut (median) |
|---|---|---|---|---|---|---|---|---|
| 10% | large | 99% | 84% | 0.73 | 27 | 71 | 11 | 460 |
| 10% | medium | 93% | 53% | 0.52 | 13 | 62 | 13 | 310 |
| 10% | small | 81% | 32% | 0.55 | 9.5 | 69 | 13 | 110 |
| 25% | large | 98% | 70% | 0.64 | 20 | 46 | 8.5 | 172 |
| 25% | medium | 88% | 31% | 0.43 | 7.8 | 44 | 9.4 | 84 |
| 25% | small | 67% | 23% | 0.50 | 6.4 | 46 | 9.5 | 16 |

Large objects gain most, because their honest sets are largest. By vote
checkpoint nothing moves: at 10%, recall ÷ oracle is 0.62–0.64 at every t from
25 to 150 (`summary.csv`, `slice = t=…`).

## What this changes for #4272 and #4273

- **The line never falls back.** It keeps the set the check ended on, or the
  unchecked starting candidate. The control shows the likely range, and the check
  confirms X iff the range's lower end clears X.
- **The candidate and the pick count both depend on X,** by the two formulas
  above. The presets resolve to (K, rounds, picks a round) = (128, 3, 5) at 10%,
  (64, 2, 5) at 25%, and (32, 1, m(X)) at 50%, 75% and 90%. The API's other
  floors resolve by the same formulas.
- **A check can take several rounds.** Each round's picks are fresh, uniform
  draws from the current candidate. Labels already seen inside the halved
  candidate are kept, which keeps every round's sample uniform (#4257). The step
  (#4273) has to present a second and a third round of 5 picks.
- **A promise's size varies: 128, 64 or 32 at 10%.** The control's count has to
  come from the check's result, not from the preset.

## Caveats

- **Open loop.** As in #4257, audit votes are not fed back. In the app they train
  the model, which would move the ranking between rounds. #4272 owns that
  lifecycle question.
- **The bound treats sampling without replacement as a binomial draw.** That is
  conservative for a small candidate.
- **One dataset, one embedder, and pools that are scenarios.**

## Files

| file | what |
|---|---|
| `summary.csv` | every rule × m × X × pool, per slice (all, by t, by band): #4257's metrics, plus `returned` and the oracle's median cut. `part` is `schedule` (the chosen rule and reading the top 32, at all five presets) or `grid` (the candidates, at 10–50%) |
| `best_attempt.csv` | the schedule under "do your best", per pool × X × slice (all, by t, by band, and by the round a check ended in): whether the check confirmed X, the returned set's size and true precision, whether it meets X, recall ÷ oracle, how often the likely range contains the truth (overall, confirmed, short; `se_coverage` is clustered by cell), its width, and the unchecked starting candidate's precision |
| `provenance.json` | input hashes, α, draws, seed, the schedule, the grid, and the range's definition |

Rebuild: `python scripts/experiments/calibration/analyze_floor_candidate_4267.py`
(about 30 seconds on one core).
