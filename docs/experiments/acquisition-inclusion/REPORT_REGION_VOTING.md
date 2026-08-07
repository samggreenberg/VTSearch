# The acquisition cut under region voting — the generalisation check

**Issue #2877 · follow-up to #2876 (the COCO result) and #2878 (which shipped `-3`) ·
base dev `c83b0e08` · branch `claude/acq-incl-vg-2877` · GRID worktree
`/exp/sgreenberg/projects/vts-acq-vg` · experiment `/exp/sgreenberg/acq-vg` ·
SLURM 476552 / 476554 / 476587 / 476589 / 476591 / 476593 / 476595 ·
3864/3864 cells, 0 failures, 86 min, zero GPU**

## BLUF

**The mechanism generalises. The operating point does not.**

On `visual_genome_m × siglip` region voting the lever pulls *harder* than it did
on COCO — the sampling position moves +0.121 in pool percentile against COCO's
+0.058 — and it still buys real positives: **6 → 12 per 100 votes** at the
shipped `k = -3`. But the payoff inverts. On COCO the extra labels *lowered*
cost (−0.011, p=8e-5); here cost degrades roughly monotonically in `|k|`, and
**the shipped −3 fails the pre-registered ship rule**: its 95% CI on the mean
final-cost delta is **[+0.003, +0.022]** against a tolerance of +0.01.

**`k = -1` is the only arm that passes.** It is not an interior optimum in the
sense COCO had one — the lowest-cost arm here is `prod` itself. It is a
*constrained* optimum: the most aggressive offset whose cost penalty is small
enough to certify.

**Recommendation: gate the offset by voting mode** (`-3` binary, `-1` region)
rather than leave `-3` global. That is a production change and belongs in its
own PR, exactly as #2876 (the study) and #2878 (the ship) were split.

## The mechanism is confirmed, not assumed

The same two things that had to hold on COCO hold here, which is what makes the
negative result readable rather than a shrug.

**The lever moved, further than it did on COCO:**

| arm | median `acq_pool_percentile` | shift vs control | steps where acq ≠ reporting |
|---|---:|---:|---:|
| `acq_m4` (k=−4) | 0.9088 | +0.1242 | 96% |
| `acq_m3` (k=−3) | 0.9059 | +0.1213 | 96% |
| `acq_m2` (k=−2) | 0.8904 | +0.1058 | 96% |
| `acq_m1` (k=−1) | 0.8550 | +0.0704 | 96% |
| `prod` (k=0) | 0.7846 | — | 0% |
| `acq_p2` (k=+2) | 0.4315 | **−0.3531** | 97% |
| `rank_pin` (0.959) | 0.9633 | +0.1787 | 100% |

**The falsification arm falsified, on every endpoint.** `acq_p2` (k=+2) moved
the cut down the ranking and made everything worse: positives 6 → **4**
(p<1e-5), final cost +0.033 (p<1e-5), oracle cost +0.016 (p<1e-5), AP −0.018
(p<1e-5), and it is the one arm whose deep-spike incidence rises significantly
(23.7% → 28.5%, p=0.010). Without it none of the rest would be interpretable.

## Result

| arm | positives @100 | positives @50 | final cost | final AP | oracle cost | deep spikes |
|---|---:|---:|---:|---:|---:|---:|
| `acq_m4` (k=−4) | **12** | 7 | 0.455 | 0.370 | 0.414 | 26.7% |
| `acq_m3` (k=−3, shipped) | **12** | 6 | 0.452 | **0.371** | 0.410 | 26.9% |
| `acq_m2` (k=−2) | 10 | 5 | 0.434 | 0.362 | 0.407 | 25.0% |
| **`acq_m1` (k=−1)** | 7 | 4 | 0.437 | 0.350 | 0.401 | 23.5% |
| `prod` (k=0) | 6 | 4 | **0.426** | 0.349 | **0.395** | 23.7% |
| `acq_p2` (k=+2) | 4 | 3 | 0.464 | 0.313 | 0.410 | 28.5% |
| `rank_pin` (0.959) | 11 | 6 | 0.505 | 0.362 | 0.449 | 27.0% |

Paired at the `(category, seed)` cell against `prod`, 540 cells:

| arm | positives Δ | final cost Δ | 95% CI on mean cost Δ | p (cost) | AP Δ | p (AP) |
|---|---:|---:|---:|---:|---:|---:|
| `acq_m4` | **+5** | +0.000 | [−0.0003, +0.0183] | 0.32 | +0.012 | <1e-5 |
| `acq_m3` | **+4** | +0.000 | **[+0.0033, +0.0215]** | 0.25 | +0.012 | <1e-5 |
| `acq_m2` | +3 | +0.000 | [−0.0070, **+0.0109**] | 0.48 | +0.012 | <1e-5 |
| **`acq_m1`** | +1 | −0.002 | **[−0.0095, +0.0075]** | 0.23 | +0.004 | 1e-4 |
| `rank_pin` | +5 | +0.015 | [+0.0196, +0.0401] | <1e-5 | +0.007 | <1e-5 |
| `acq_p2` | **−1** | +0.033 | [+0.0281, +0.0458] | <1e-5 | −0.018 | <1e-5 |

### Ship rule (pre-registered)

Adopt iff positives rise (p<0.05) **and** the 95% upper bound on the final-cost
delta is below +0.01 **and** deep-spike incidence does not rise **and** the lever
moved. Only **`acq_m1`** passes. `acq_m3` and `acq_m4` fail on cost; `rank_pin`
fails on cost outright.

**`acq_m2` fails by 0.0009** — its CI upper bound is +0.0109 against a +0.01
tolerance. That is inside the width of the bootstrap's own resampling noise, and
it should be read as "on the boundary", not as a decision. It matters because
k=−2 is where the *ranking* benefit saturates (below).

## Why it fails — and it is not the threshold estimator

Decomposing final cost into `oracle_cost` (how good the learned ranking is) and
`regret = cost − oracle_cost` (how well the cut is placed on it) separates two
very different failures. Paired against `prod`:

| arm | Δ oracle cost | Δ regret | p (regret) |
|---|---:|---:|---:|
| `acq_m1` | +0.000 | −0.000 | 0.23 |
| `acq_m2` | −0.002 | +0.000 | 0.58 |
| `acq_m3` | +0.000 | +0.000 | 0.59 |
| `acq_m4` | +0.001 | −0.000 | 0.28 |
| `acq_p2` | +0.016 | **+0.007** | **8e-10** |

**Regret is flat in every negative-`k` arm.** The cut estimator is doing its job
exactly as well no matter where acquisition samples; only the falsifier degrades
it. So the cost penalty is not a calibration failure — it is that **the learned
ranking itself gets slightly worse** (oracle cost drifts up from 0.395 to 0.410
at k=−3).

And yet **average precision rises** (0.349 → 0.371, median Δ +0.012, p<1e-5).
Those two facts are not in conflict, and together they are the finding:

> Aggressive acquisition **sharpens the top of the ranking and blurs the rest.**
> AP is dominated by the head of the list, so it improves. `oracle_cost` is
> `min_θ (FPR+FNR)`, a statement about *global* separability, so it degrades.
> Reported cost is a global cut, so it follows `oracle_cost`, not AP.

COCO did not show this because COCO was starved hard enough that any positive
helped everywhere: there, oracle cost *fell* 0.113 → 0.101 (p=1e-5) and AP rose
0.696 → 0.817. Region voting starts from a much less starved place — `prod`
already finds 6 positives per 100 votes here against COCO's 4, on a pool where a
media's score is a max over ~24 region nodes — so the marginal positive is worth
less, and the sampling bias that buys it costs more.

**The ranking benefit saturates before the cost penalty does.** Median AP Δ is
+0.004 at k=−1, then +0.012 at k=−2 and unchanged at −3 and −4. Oracle cost only
begins drifting at −3. Everything k=−3 and −4 buy over k=−2 is extra *positives*
with no further ranking gain and a rising bill.

## What did transfer, unchanged

**The adaptive ramp.** The signature finding of #2876 reproduces exactly:

| arm | acq percentile, t≤20 | t 21–60 | t 61+ | std |
|---|---:|---:|---:|---:|
| `prod` | 0.6471 | 0.7745 | 0.7912 | 0.0671 |
| `acq_m3` | 0.7320 | 0.8882 | 0.9122 | 0.0897 |
| `rank_pin` | 0.9602 | 0.9622 | 0.9659 | **0.0024** |

An inclusion cut starts conservative and climbs as the fit sharpens; a pinned
quantile is constant by construction. **`rank_pin` is now worse in two
environments and for two different reasons** — on COCO it stalled (6 positives
against 18); here it finds positives (11) but pays for them, regressing cost
(+0.015 median, p<1e-5), oracle cost (+0.008, p<1e-5) *and* posting the second
worst deep-spike rate. #2876's "do not ship the pinned form" is reinforced.

## Guardrails and caveats

- **The deep-spike thresholds do not transfer, and I have not pretended they
  do.** `SPIKE_DEEP_COST=0.25` / `DEEP_EXCESS=0.20` were calibrated to COCO,
  where cost sits near 0.137; VG region-voting cost sits near 0.43, so the base
  incidence is **23.7% here against COCO's 5.4%**. The *paired* McNemar contrast
  is still meaningful and is what the ship rule reads; the absolute rate is not
  comparable across the two reports. `acq_m3` is nominally +3.2 pp (p=0.093) —
  not significant, but pointing the same way as the falsifier's +4.8 pp
  (p=0.010).
- **The paired test is biased *against* the offset here, and it still lost.**
  12 of 552 cells never found a positive in 100 votes and drop out. They are the
  *same 12 cells in every arm*, so no arm is advantaged — but any cell where
  `prod` never gets off the ground and a treatment arm does would also be
  dropped, and those are where the treatment should look best.
- **Marginal and paired readings agree here**, unlike COCO. Every arm's marginal
  median cost and its paired delta point the same way, so the reading caveat
  that picked COCO's winner does not arise.
- **Cost deltas are tail effects, not shifts.** For k=−2/−3/−4 the *median*
  paired cost delta is exactly 0.000: most cells are unchanged. The mean penalty
  at k=−3 (+0.012) comes from an asymmetric tail — 25.6% of cells get more than
  0.05 worse against 20.9% at k=−1. What the offset risks on region voting is
  variance, not a uniform tax.

## A sizing note worth keeping

This run is 24 seeds, not #2876's 8. The 8-seed pilot (kept at
`results_8seed/`) reproduced every qualitative finding above but put a 95% CI of
**[−0.014, +0.019]** on the k=−3 cost delta — a null too wide to certify, and
one that could not separate "the offset is free on region voting" from "it costs
something", which are **opposite shipping decisions**.

The lesson is the one the issue already flagged and I initially got wrong: *size
for the cost contrast, not the positives one, and re-derive the size per
environment*. The positives endpoint was over-powered at 8 seeds in both
environments. Cost was not, because VG region-voting costs are ~3× COCO's in
absolute terms and correspondingly noisier: the observed paired SD of 0.111
implies **n≈473** for a ±0.010 half-width, against the 180 that 8 seeds
delivered.

## Recommendation

1. **Do not leave `-3` global.** It fails its own pre-registered ship rule in the
   second environment, and the failure has a mechanism rather than being noise.
2. **Gate the offset by voting mode** — `-3` binary, `-1` region. The codebase
   already has this exact shape in `production_schedule_for(region_voting=...)`
   (`vtscore/training/blend_schedules.py`, from #2849), so this is one
   conditional against an existing precedent and not a new user-facing knob. It
   is a production change and should be its own PR.
3. **If a single global value is wanted instead, it is `-1`**, not `-3`: it is
   the only value that passes the rule in both environments (it passed on COCO
   too, buying +2 positives at −0.007 cost). It leaves most of COCO's win on the
   table, which is the price of one number.
4. **Consider `-2` for region voting** if the owner reads a +0.0009 tolerance
   breach as noise — it captures the *entire* ranking benefit (AP Δ +0.012,
   identical to −3 and −4) for roughly half the cost exposure. I have not
   recommended it as the default because it does not pass as written, and the
   rule should not be relaxed after seeing the number.
5. **Do not widen this grid.** The answer is not "somewhere past −4"; cost
   degrades monotonically in that direction and AP has already saturated.

## Reproducing

```bash
cd /exp/$USER/projects/vts-acq-vg/scripts/experiments/calibration
python selftest_analyze_acq.py               # planted-answer test; run first
bash launch_acq_incl_vg.sh                   # 7 arms x 552 cells, ~86 min, no GPU
python analyze_acq.py                        # once all seven drain
CALIB_N_SEEDS=8 RESULTS_ROOT=$CALIB_EXP/results_8seed bash launch_acq_incl_vg.sh   # the pilot
```

Raw analyzer output for this run is in
[`GENERATED_TABLES_REGION_VOTING.md`](GENERATED_TABLES_REGION_VOTING.md).

## Figures

![frontier](figures/vg_fig1_frontier.png)

![lever verification](figures/vg_fig2_lever_verification.png)

![guardrails](figures/vg_fig3_guardrails.png)
