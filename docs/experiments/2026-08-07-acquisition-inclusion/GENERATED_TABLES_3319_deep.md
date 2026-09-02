# #3319 frontier — `siglip x whole_image` — 768 trajectories, 4 arms

Arms present: prod, acq_m1, acq_m3, acq_m4
Cost-regression tolerance: ±0.01

## Did the lever move?

| arm | k | median `acq_pool_percentile` | shift vs prod | cells |
|---|---:|---:|---:|---:|
| `prod` | 0 | 0.7501 | +0.0000 | 192 |
| `acq_m1` | -1 | 0.8326 | +0.0825 | 192 |
| `acq_m3` | -3 | 0.9434 | +0.1933 | 192 |
| `acq_m4` | -4 | 0.9713 | +0.2213 | 192 |

## H2 prerequisite — are the half steps distinct operating points?

An arm whose per-cell `acq_pool_percentile` matches a neighbour's is a
duplicate produced by the quantile snap, not a finer grid point. Refused
above 10% of cells, per the plan.

| arm | neighbour | cells compared | identical | verdict |
|---|---|---:|---:|---|
| `prod` | `acq_m1` | 192 | 0.0% | distinct |
| `acq_m1` | `acq_m3` | 192 | 0.0% | distinct |
| `acq_m3` | `acq_m4` | 192 | 0.5% | distinct |

## The frontier — paired against `prod` (k=0)

| arm | k | Δ final cost [95% CI] | Δ positives@100 | Δ AP | Δ oracle cost | pairs |
|---|---:|---|---:|---:|---:|---:|
| `acq_m1` | -1 | -0.0163 [-0.0218, -0.0108] | +16.8 | +0.063 | -0.0210 | 192 |
| `acq_m3` | -3 | -0.0329 [-0.0387, -0.0269] | +90.1 | +0.123 | -0.0313 | 192 |
| `acq_m4` | -4 | -0.0324 [-0.0387, -0.0262] | +99.9 | +0.128 | -0.0324 | 192 |

**Falsification arm missing — verdict withheld.**

## H1 — does the frontier turn?

Minimum paired cost delta is at **`acq_m3` (k=-3)**, -0.0329 [-0.0387, -0.0269].

Is any arm deeper than the minimum **resolvably** worse than it?

| deeper arm | k | Δ cost vs the minimum [95% CI] | resolvably worse? |
|---|---:|---|---|
| `acq_m4` | -4 | +0.0004 [-0.0029, +0.0037] | no |

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
| `acq_m1` | -1 | +0.0166 [+0.0113, +0.0219] | -73.3 | -0.060 | 5.7% → 1.0% | no |
| `acq_m4` | -4 | +0.0004 [-0.0029, +0.0037] | +9.7 | +0.004 | 5.7% → 2.1% | YES |
