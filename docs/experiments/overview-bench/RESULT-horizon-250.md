# Result — 250 votes instead of 150

Scores [`PREREG-horizon-250.md`](PREREG-horizon-250.md). The question was whether a
user who keeps clicking past 150 votes gets a better detector. **Yes — modestly,
reliably, and not for the reason the question assumes.**

## What the extra 100 votes bought

Paired per run, votes 101–150 against 201–250 (equal windows on the same cell, so
between-category variance cancels). Full tables in `ANALYSIS_HORIZON_*.txt`.

| | `visual_genome_m` + `coco_val` (128 runs) | box bands (140 runs, partial grid) |
|---|---|---|
| cost | **−0.011 ± 0.005** | **−0.023 ± 0.006** |
| fnr | **−0.025 ± 0.005** | **−0.100 ± 0.017** |
| fpr | **+0.014 ± 0.007** | **+0.078 ± 0.017** |
| AP | −0.002 ± 0.002 (not resolvable) | +0.012 ± 0.003 |
| positives held | +1.8 ± 0.3 | +3.1 ± 0.4 |

**The gain is a threshold effect, not a learning effect.** The extra clicks find
1.8–3.1 more positives; ranking barely moves (AP −0.002 on the whole-image
haystacks, +0.012 on the bands) while the operating point shifts hard toward
inclusion — recall improves two to four times as much as precision degrades, and
cost nets out slightly better. What 100 more clicks buy is a cut placed on a less
thin positive set, not a detector that ranks better.

That has a cheaper implication than "ask users to click more": the same gain
should be available from a cut rule that accounts for how few positives it is
calibrating on. Asking for 100 extra clicks to move a threshold is an expensive
way to buy what a better estimator could give for free — and it is the same
conclusion the [main report](REPORT.md) reaches from the regret decomposition,
arriving from the opposite direction.

## Runs that were not starved, only under-clicked

Five runs the 150-vote grid reports as *starved* — never a single positive — find
their first positive after vote 150:

| run | first positive |
|---|---|
| `coco_val` / `sports ball` / seed 1 | vote 159 |
| `vg_box_small` / `tip` / seed 2 (`siglip`) | vote 175 |
| `vg_box_small` / `tip` / seed 2 (`siglip2_l`) | vote 177 |
| `visual_genome_m` / `ball` / seed 1 | vote 224 |
| `coco_val` / `refrigerator` / seed 0 (no boxes) | vote 232 |

So the report's "14 of 504 runs never found a positive (2.8 %)" is a statement
about the **horizon**, not about the method: a third of those runs were still
going to work. What does *not* improve is the runs that are stuck rather than
starved — cells whose deep-regime cost stays above 0.9 number **23 before and 23
after** on the box bands. More clicks help runs that already have traction.

## Scoring the pre-registration

| expectation | outcome |
|---|---|
| 1. Little on the box bands (under ~0.02 cost) | **Wrong.** −0.023 ± 0.006, resolvable, and the largest effect measured. |
| 2. Something on `visual_genome_m` / `coco_val` | **Right**, −0.011 ± 0.005, though smaller than the bands'. |
| 3. Most of the gain on the slow starters | **Half wrong.** Five starved runs did start, but the stuck-run count did not move at all (23 → 23); the gain is spread across runs that were already working. |

The single sizing cell quoted in the pre-registration (`vg_box_small` / `glasses`,
which gained nothing) was not representative — it was one cell chosen for being
cheap to time, and a 140-cell paired comparison says otherwise. Sizing cells size;
they do not measure.

## What this does *not* license

- **The Visual Genome half is provisional.** #3156 is correcting VG's annotations
  — the `bus` sheet in the main report shows frame-filling buses annotated
  `door, tire, window` — and `fb4f4ec03` replaces the box-area banding, whose
  bands currently carry disjoint vocabularies. Every VG number here should be
  re-measured against the corrected data. The COCO half is unaffected: its
  annotation is exhaustive over its classes.
- **The box-band grid is partial.** It was cancelled mid-flight for the reason
  above, after completing `vg_box_small` (30 / 30 / 22 cells by embedder) and the
  two whole-image arms of `vg_box_medium`; `vg_box_large` has no coverage. The
  numbers above are what those 140 runs say, not a full band comparison.
- **The main report is unchanged for now.** Updating it means re-running on
  corrected annotations, which is #3156's work.

## Provenance and a numerical wrinkle

The horizon run reuses each source study's `prepare_info.json` and crops, so it is
the same cells carried further. The overlap check confirms this exactly: 26,538
shared steps on the whole-image grid, 19,930 on the bands and 6,249 without
boxes, **all bit-identical** — except one cell.

`caltech101_m` / `siglip` / `airplanes` / seed 1 diverges from vote 37 onward:
`AP`, `AUROC` and `n_good` are identical, so the votes and the model are
identical, but the **threshold** differs by up to 0.026 and cost through it by
0.020. The category is saturated — positives near 1.0, negatives near 0 — which
leaves the mixture fit that places the cut ill-conditioned, so a float-level
difference between machines flips it. Filed as #3166; it is why
`caltech101_m` is excluded from the whole-image numbers above, and one more
reason to retire that haystack from this sweep.

Grids: `/expscratch/sgreenberg/bench-h250/{wave1,vgbox,binary}/results`, arrays
`507700` (cancelled at 143 / 270), `507713` (189 / 189), `507722` (45 / 45), zero
task failures. Reproduce with `analyze_horizon.py <short_results> <long_results>`.
