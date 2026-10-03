# One correct Good can wreck a document session (#4170): measured, not fixed

**Question.** #4170 found that a Good whose box is mostly printed text makes
a bad template: its glyphs match every typed page. How often does that cost a
real session? Does either cheap candidate fix it? The candidates were keeping
the query crop, and the descriptor stop-list (#4432).

**Answer.** It is rare but severe, and neither candidate removes it. The text
mask, the third candidate, is postponed: the app has no text detector, and the
`text_components` the issue cites are SPODS ground truth.

## How often

On the shipped path (FullMarks v5.0 tier `m`, 36 classes, 2 replicates, the
second with the click and test halves swapped), **5 of 1,328 Good clicks cost
≥ 0.2 test AP. All five are in three classes, and four of them are click 1:**

| class | replicate | click | test AP | back within 0.05 at | AP at 50 clicks |
|---|---|---|---|---|---|
| `staver/stamp_stampds-00213_1` | 1 | 3 | 0.80 → 0.24 | click 13 | 0.72 |
| `tobacco800/logo_asg54f00_1` | 1 | 1 | 0.67 → 0.36 | never | 0.40 |
| `tobacco800/logo_asg54f00_1` | 2 | 1 | 0.31 → 0.05 | click 38 | 0.50 |
| `ucsf/logo_p_lorillard_crest` | 1 | 1 | 0.63 → 0.12 | click 2 | 0.79 |
| `ucsf/logo_p_lorillard_crest` | 2 | 1 | 0.67 → 0.00 | click 41 | 0.81 |

## The review's opening is the app's

The review opens with an example sort on the query crop. From the first Good
on, it builds templates and Stage-1 queries from the Good boxes alone, so the
crop is gone. That is also what the app does.

A detector created from an example seeds the crop as a Good vote
(`seed_good_votes_from_examples`). But the inserted media item carries no
`local_features`, which only the dataset embedding stage attaches. So
`build_templates` and `vote_queries` skip it.

## Candidate: keep the crop all session

The new `sota_documents.py --seed-crop` keeps the crop as an unboxed Good
vote. It was measured with two Stage-1 queries for the crop. Pooled over 2
replicates against the shipped opening:

| test AP, pooled difference [95%] | click 0 | 10 | 25 | 50 |
|---|---|---|---|---|
| crop queried by its **tiles** | −0.28 [−0.37, −0.20] | −0.08 [−0.15, −0.03] | −0.08 [−0.15, −0.03] | −0.09 [−0.16, −0.03] |
| crop queried **whole** (one VLAD) | −0.031 [−0.057, −0.011] | +0.020 [−0.007, +0.061] | +0.020 [−0.004, +0.060] | +0.008 [−0.003, +0.023] |

**Querying by tiles fails, because of scale.** Tiles are fixed fractions of
the image, so a crop's 42–259 tiles are fragments of the mark. They look like
glyphs and match anywhere.

**The whole-crop query ranks as well as the shipped opening from click 10 on,
but the harmful clicks stay.** 3 of the 5 sessions above still have one
(Staver replicate 1, lorillard in both). One new one appears:
`spods/logo_00023_0`, replicate 2, 0.82 → 0.42 at click 1. That fails the
rule set on #4170 in advance. A good crop template does not stop a bad box
template from taking over the max.

## Candidate: the descriptor stop-list

Measured in #4432
([report](../2026-10-02-stoplist-cheap-4432/REPORT.md)). It lifts the
Tobacco800 and Staver classes in both replicates. But it makes lorillard's
replicate-2 session unrecoverable, and pooled AP misses its bar.

## Side finding: GPU Stage 1 fell back to the CPU on 32 GB cards

The first whole-crop run crawled on a V100: ~800 s per class against ~60 s
on an L40S. `_gpu_page_scores` needs 3× the 8.5 GiB tile matrix free, and
memory PyTorch's allocator still held did not count. So from the second
matrix on, it silently scored on the CPU. That was fixed in its own PR, which
empties the cache first and logs a fallback. The deployed app runs on a V100.
The whole-crop replicate 1 ran its first 14 classes before the fix, so its
retrain times are not comparable. Its AP and F1 are unaffected.

## Reproduce

```bash
source scripts/experiments/pile/pile_env.sh
for flags in "--seed-crop tiles" "--seed-crop tiles --swap-halves" \
             "--seed-crop whole" "--seed-crop whole --swap-halves"; do
  python scripts/experiments/fullmarks/sota_documents.py --tier m --max-v 50 $flags \
      --matrix /expscratch/$USER/fullmarks/votes-4162/matrix-m \
      --feature-cache /expscratch/$USER/fullmarks/features --out <dir>/<arm>
done
```

The shipped-opening replicates are #4440's validation run and #4432's
`swap-off` arm. Run directory: `/expscratch/sgreenberg/seedcrop-4170/`.
`measurements/` holds all four candidate arms' `steps.csv`.
