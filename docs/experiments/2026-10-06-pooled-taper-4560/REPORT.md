# Why does Lean the Threshold's farthest-shallow preset read low, and what fixes it? (#4560)

**The top bands above the line were never audited, and the prior that stood
in for them was the wrong one.** At beta 4 a line has 6–9 bands above it; the
first pass audits them from the line upward, and the 20-pick budget (#4558)
ran out before the top bands had a round (2.4–3.6 picks on average, against
the 5 of a round). An unaudited band reads at the pooled rate, which the
bands near the line set, about 45% right, while the top bands are about 78%
right. So a preset whose count falls in them (a beta-4 session's beta-1/4
option, about a quarter of the line) read low. **Shipped: the budget waits
until every band above the line has had a round, and each band pools only
with its neighbours.** The farthest-shallow preset's F-beta range now holds
the truth in **0.87–0.90** of sessions (was 0.61–0.80). The line's own
precision range holds 0.925–0.99 (was 0.936–0.99). A beta-4 Test costs
**64–67 picks** (was 56–60); beta 1/4 and 1 cost 1–3 more.

Part of #4520; follows #4539, #4540, #4558. Data: #4523's 3,669 saved photo
sessions replayed at the shipped 0.20 / 20 / 40, four Test seeds each, for
every variant below (`variants.csv`; `summary_shipped.csv` is the shipped
variant's full summary). Harness: `price_pooled_taper_4560.py`, which sets
the estimator's study seams (`pooled_weight`, `POOL_RADIUS`,
`FIRST_PASS_BEFORE_BUDGET` in `vtscore/training/thresholds/line_test.py`)
and runs the #4523 replay.

## What was tried

At beta 4, the farthest-shallow preset's F-beta coverage, the lowest
line-precision coverage over every world and beta, and the mean picks:

| variant | preset 0.1% | 0.44% | 5% | line precision held, min | picks |
|---|---:|---:|---:|---:|---:|
| today (one pool, flat weight) | 0.80 | 0.73 | 0.61 | 0.94 | 57 |
| weaker pull toward the pool (linear / half / weak taper) | 0.85–0.86 | 0.82–0.85 | 0.75–0.80 | 0.91–0.94 | 57 |
| budget waits for the first pass | 0.88 | 0.85 | 0.80 | **0.87** | 61 |
| neighbour pool (radius 1) | 0.82 | 0.79 | 0.77 | 0.93 | 57 |
| **neighbour pool (radius 1) + first pass** | **0.89** | **0.90** | **0.87** | **0.925** | **61** |
| neighbour pool (radius 2) + first pass | 0.88 | 0.90 | 0.85 | 0.91 | 61 |

- **Tapering the pool's weight** (the issue's first idea) barely moved the
  bias (−0.165 → −0.156 at 0.44%): an unaudited band has no picks, so its
  estimate is the prior's mean whatever the prior's weight. Coverage rose
  only because the ranges widened, and the weakest pool broke the line's
  range (0.91).
- **Waiting for the first pass** fixed the presets but broke the line's own
  range on sparse beta-4 lines (0.87). Once the rich top bands are audited,
  one pool over every band above the line lifts the large, nearly empty
  bands near the line with them: every session that lost coverage had its
  range above the truth, on lines 1–8% right.
- **Pooling with neighbours only** stops that: rich bands borrow from rich
  bands and sparse ones from sparse ones. Alone it changes little, because
  the top bands are still unaudited; with the first pass it clears the
  preset target in every world and keeps the line's range within one
  standard error of the 0.93 bar (0.925 at 0.44% / beta 4, SE ≈ 0.008).
  Radius 2 mixes too far again (0.91).

## What it costs

Mean picks to Done, today → shipped:

| withheld half | beta 1/4 | beta 1 | beta 4 |
|---|---:|---:|---:|
| 0.1% | 52 → 55 | 55 → 60 | 56 → 64 |
| 0.44% | 56 → 58 | 59 → 62 | 59 → 67 |
| 5% | 57 → 57 | 60 → 61 | 60 → 65 |

The 90th percentile at beta 4 rises from 60 to 70–80. That gives back about
half of what the 20-pick budget saved at beta 4 (#4558), and nothing at beta
1/4. The owner chose it on these numbers (2026-10-06): a beta-4 user pays
5–8 picks for a *Lean* option that is right 87–90% of the time instead of
61–80%.

## The other presets

Every preset's coverage is in `variants.csv`. Beyond the farthest-shallow
one, the shipped variant moves the beta-4 session's beta-1 preset from
0.80–0.85 to 0.92–0.95, and leaves the deeper presets of beta-1/4 sessions
where they were (0.81–0.96). The line's F-beta range holds 0.88–0.97.

## Decisions

- Radius 1 and the first-pass rule ship as module defaults
  (`POOL_RADIUS = 1`, `FIRST_PASS_BEFORE_BUDGET = True`). They are module
  constants, not `LineBudgets` fields, because they are estimator design,
  not something a session varies; `pooled_weight` stays flat, since the
  tapers did not earn a place.
- The 0.925 cell is reported as under the bar by less than its standard
  error, not rounded up.
