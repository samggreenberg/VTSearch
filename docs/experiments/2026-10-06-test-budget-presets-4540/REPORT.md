# How many picks should a Test spend above the line, and does Lean the Threshold show the right ranges? (#4540)

**Twenty picks above the line, not forty: the same coverage or better, for up
to 13 fewer picks.** With the pooled prior (#4539) and the deeper walk below
the line (#4542) in place, a Test at 0.20 / 40 / 40 spends 54–73 picks on
photos (90th percentile 65–80). Capping the phase above the line at 20 and
keeping 40 below cuts that to 52–60 (90th percentile 60), and the precision
range holds the truth in **0.94–0.99** of sessions in every world and beta
(was 0.92–0.99). The saving is largest at beta 4, where lines are biggest:
71 → 59 picks at COCO Better's default pool, with precision coverage
0.92 → 0.94 and recall 0.85 → 0.91. The width target barely matters: 0.30
instead of 0.20 saves about one more pick. **The default is now 0.20 / 20
above the line, 40 below.**

**Lean the Threshold's own-balance range is right; its alternatives are not.**
The shipped Test view shows no ship / lean / retrain reading. *Lean the
Threshold* lists the precision, recall and F-beta range each balance preset
would ship, re-estimated from the picks, and the user chooses. The preset at
the session's own balance is the line itself and holds 0.94–0.99. The other
presets hold **0.45–0.89**, and they miss one way:

- a **shallower** preset (fewer items kept) under-reads its precision by up
  to 0.19 and its F-beta by up to 0.17, so Lean makes keeping fewer look
  worse than it is;
- a **deeper** preset over-reads its recall by up to 0.14.

That is the under-read gain #4523's verdict study saw, located: the rule it
priced was the harness's proposal, but the ranges it read are the ones the
app shows. #4540 stays open for the estimator fix.

Part of #4520; follows #4523, #4539, #4542. Data: #4523's 3,669 saved photo
sessions replayed under dev at 8bb7cfde9 + this branch (the arm now records the
presets), widths 0.20 / 0.30 × budgets 40 / 40, 30 / 40, 20 / 40, four Test
seeds each. Documents have no class model, so no preset line to re-estimate.

## The split budget

At width 0.20 (`summary.csv`, `budget` = above the line, `budget_below` =
below):

| withheld half | beta | picks 40/40 | **20/40** | p90 40/40 | p90 20/40 | precision held 40/40 | 20/40 | recall held 40/40 | 20/40 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1% | 1/4 | 54 | **52** | 75 | 60 | 0.99 | 0.99 | 0.81 | 0.81 |
| 0.1% | 1 | 59 | **55** | 80 | 60 | 0.99 | 0.99 | 0.82 | 0.82 |
| 0.1% | 4 | 64 | **56** | 80 | 60 | 0.96 | 0.98 | 0.83 | 0.83 |
| 0.44% | 1/4 | 58 | **56** | 65 | 60 | 0.97 | 0.98 | 0.81 | 0.81 |
| 0.44% | 1 | 65 | **59** | 80 | 60 | 0.95 | 0.96 | 0.82 | 0.83 |
| 0.44% | 4 | 71 | **59** | 80 | 60 | 0.92 | 0.94 | 0.85 | 0.91 |
| 5% | 1/4 | 58 | **57** | 65 | 60 | 0.99 | 0.99 | 0.96 | 0.96 |
| 5% | 1 | 67 | **60** | 80 | 60 | 0.95 | 0.95 | 0.97 | 0.97 |
| 5% | 4 | 73 | **60** | 80 | 60 | 0.94 | 0.94 | 0.98 | 0.98 |
| documents | 1/4 – 4 | 23–26 | **19–21** | 41–45 | 25–28 | 0.94–1.00 | 0.94–1.00 | 0.65–0.67 | 0.65–0.67 |

Why fewer picks above the line costs nothing: once every band above the line
has had a round (the #4539 rule), the pooled range is already near its width
target, and the extra rounds the old budget allowed went to the bands where
the F-beta range shrank most, which at beta 4 are the deep, nearly empty ones
whose share the pool already pins. The 30 / 40 point lands between the two.

## What Lean the Threshold shows

At 20 / 40, the share of sessions whose range for each preset held the truth,
and the mean point − truth (`presets.csv`):

| withheld half | session beta | preset | kept vs line | precision held | precision bias | recall held | recall bias | F-beta held |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.44% | 1 | **1 (own)** | 1.0 | **0.96** | | **0.83** | | **0.90** |
| 0.44% | 1 | 1/4 | 0.58 | 0.79 | −0.050 | 0.85 | +0.009 | 0.84 |
| 0.44% | 1 | 4 | 2.1 | 0.82 | −0.006 | 0.83 | +0.135 | 0.88 |
| 0.44% | 4 | 1/4 | 0.29 | 0.61 | −0.187 | 0.80 | −0.057 | 0.58 |
| 0.44% | 4 | 1 | 0.52 | 0.76 | −0.076 | 0.86 | −0.046 | 0.75 |
| 0.1% | 4 | 1/4 | 0.24 | 0.65 | −0.138 | 0.70 | −0.143 | 0.63 |
| 5% | 4 | 1/4 | 0.35 | 0.56 | −0.187 | 0.59 | −0.101 | 0.45 |
| 5% | 1/4 | 4 | 2.6 | 0.89 | −0.010 | 0.90 | +0.016 | 0.89 |

The further the preset is from the line, the worse its range, and the
direction follows the side:

- **Shallower** presets fall inside the bands above the line. `estimate_at`
  splits a band's matches in proportion to how much of the band the count
  takes, which assumes the band is uniform. The top of a band is richer than
  its bottom, and the pooled prior also pulls the rich top bands toward the
  line's average, so the shallow preset's precision is under-read.
- **Deeper** presets add bands below the line, whose counts lean on the
  model's tail, which the walk corrects only as deep as it reaches.

**The fix to try (#4540).** The class model's posteriors rank the items
within a band. Splitting a band's matches by posterior mass instead of by
size would move the shallow presets' estimate up where the model says the
top is richer, at no cost in picks. A pooled weight that falls with distance
from the top is a second lever. Either is priced by the same replay.

## Decisions made without the owner

- The default goes to 20 / 40 from this replay alone: every cell keeps or
  improves coverage, and none needed the owner's trade-off.
- Width stays at 0.20; 0.30 saves about one pick and gives back up to two
  points of coverage at beta 4.
- The harness's ship / lean / retrain rule (`read_verdict`) stays as a
  measurement, not a recommendation: the app does not implement one, and
  the presets' coverage is the number that matters for what it does show.
