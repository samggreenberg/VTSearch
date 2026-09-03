# #3319 frontier — `siglip+dinov3_patch x max_patch` — 960 trajectories, 5 arms

Arms present: prod, acq_m3, acq_m4, acq_m5, acq_p2
Cost-regression tolerance: ±0.01

## Did the lever move?

| arm | k | median `acq_pool_percentile` | shift vs prod | cells |
|---|---:|---:|---:|---:|
| `prod` | 0 | 0.8487 | +0.0000 | 192 |
| `acq_m3` | -3 | 0.9398 | +0.0911 | 192 |
| `acq_m4` | -4 | 0.9519 | +0.1032 | 192 |
| `acq_m5` | -5 | 0.9600 | +0.1113 | 192 |
| `acq_p2` | 2 | 0.7049 | -0.1439 | 192 |

## H2 prerequisite — are the half steps distinct operating points?

An arm whose per-cell `acq_pool_percentile` matches a neighbour's is a
duplicate produced by the quantile snap, not a finer grid point. Refused
above 10% of cells, per the plan.

| arm | neighbour | cells compared | identical | verdict |
|---|---|---:|---:|---|
| `acq_p2` | `prod` | 192 | 0.0% | distinct |
| `prod` | `acq_m3` | 192 | 0.0% | distinct |
| `acq_m3` | `acq_m4` | 192 | 0.0% | distinct |
| `acq_m4` | `acq_m5` | 192 | 0.5% | distinct |

## The frontier — paired against `prod` (k=0)

| arm | k | Δ final cost [95% CI] | Δ positives@100 | Δ AP | Δ oracle cost | pairs |
|---|---:|---|---:|---:|---:|---:|
| `acq_p2` | 2 | +0.0368 [+0.0283, +0.0457] | -4.9 | -0.035 | +0.0345 | 192 |
| `acq_m3` | -3 | -0.0020 [-0.0094, +0.0053] | +25.7 | +0.026 | -0.0064 | 192 |
| `acq_m4` | -4 | +0.0011 [-0.0071, +0.0094] | +35.1 | +0.024 | +0.0024 | 192 |
| `acq_m5` | -5 | +0.0044 [-0.0033, +0.0120] | +42.4 | +0.024 | +0.0059 | 192 |

**Falsification arm `acq_p2`**: Δ positives -4.9 — behaves (positives fall, as required)

## H1 — does the frontier turn?

Minimum paired cost delta is at **`acq_m3` (k=-3)**, -0.0020 [-0.0094, +0.0053].

Is any arm deeper than the minimum **resolvably** worse than it?

| deeper arm | k | Δ cost vs the minimum [95% CI] | resolvably worse? |
|---|---:|---|---|
| `acq_m4` | -4 | +0.0031 [-0.0025, +0.0091] | no |
| `acq_m5` | -5 | +0.0064 [+0.0007, +0.0123] | no |

No arm deeper than the minimum is resolvably worse than it: the frontier is **flat past the optimum**, not turning. H1 falsified in its strong form — the knob has a plateau, not a peak, and the practical reading is that anything past the plateau's near edge buys positives for free.

## H4 — the posterior-flip landmark

At prevalence π = 7.1% the prior odds are 0.0764, so a
selector's picks become more likely Good than Bad only at
**k\* = −log₂((1−π)/π) = -3.71**.

Measured minimum: k=-3. Landmark: -3.71. **H4 NOT supported** (pre-registered window [−4.0, −3.5]).

## H2 — is the optimum resolvable at finer than one step?

Each half step paired **arm-to-arm** against both its integer neighbours on
the same cells. A half step 'resolves' only if it beats both by more than
the ±0.01 tolerance.

| half step | vs | Δ final cost [95% CI] | beats it? |
|---|---|---|---|

**H2 FALSIFIED** — no half step beats both of its integer neighbours; the knob's usable resolution is one bit and the integer grid was right.

## The ship comparison — every arm against the incumbent `acq_m3` (k=-3)

| arm | k | Δ final cost [95% CI] | Δ positives | Δ AP | deep spikes | passes ship rule |
|---|---:|---|---:|---:|---|---|
| `acq_m4` | -4 | +0.0031 [-0.0025, +0.0091] | +9.4 | -0.002 | 1.6% → 1.6% | YES |
| `acq_m5` | -5 | +0.0064 [+0.0007, +0.0123] | +16.7 | -0.002 | 1.6% → 1.0% | no |
