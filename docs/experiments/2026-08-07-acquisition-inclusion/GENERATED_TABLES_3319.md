# #3319 frontier — 2304 trajectories, 12 arms

Arms present: prod, acq_m1, acq_m2, acq_m2h, acq_m3, acq_m3h, acq_m4, acq_m4h, acq_m5, acq_m6, acq_m8, acq_p2
Cost-regression tolerance: ±0.01

## Did the lever move?

| arm | k | median `acq_pool_percentile` | shift vs prod | cells |
|---|---:|---:|---:|---:|
| `prod` | 0 | 0.7252 | +0.0000 | 192 |
| `acq_m1` | -1 | 0.8074 | +0.0822 | 192 |
| `acq_m2` | -2 | 0.8596 | +0.1344 | 192 |
| `acq_m2h` | -2.5 | 0.8753 | +0.1501 | 192 |
| `acq_m3` | -3 | 0.8990 | +0.1738 | 192 |
| `acq_m3h` | -3.5 | 0.9144 | +0.1892 | 192 |
| `acq_m4` | -4 | 0.9298 | +0.2046 | 192 |
| `acq_m4h` | -4.5 | 0.9415 | +0.2164 | 192 |
| `acq_m5` | -5 | 0.9480 | +0.2228 | 192 |
| `acq_m6` | -6 | 0.9601 | +0.2349 | 192 |
| `acq_m8` | -8 | 0.9733 | +0.2482 | 192 |
| `acq_p2` | 2 | 0.4009 | -0.3243 | 192 |

## H2 prerequisite — are the half steps distinct operating points?

An arm whose per-cell `acq_pool_percentile` matches a neighbour's is a
duplicate produced by the quantile snap, not a finer grid point. Refused
above 10% of cells, per the plan.

| arm | neighbour | cells compared | identical | verdict |
|---|---|---:|---:|---|
| `acq_p2` | `prod` | 192 | 0.0% | distinct |
| `prod` | `acq_m1` | 192 | 0.0% | distinct |
| `acq_m1` | `acq_m2` | 192 | 0.0% | distinct |
| `acq_m2` | `acq_m2h` | 192 | 0.0% | distinct |
| `acq_m2h` | `acq_m3` | 192 | 0.0% | distinct |
| `acq_m3` | `acq_m3h` | 192 | 0.0% | distinct |
| `acq_m3h` | `acq_m4` | 192 | 0.0% | distinct |
| `acq_m4` | `acq_m4h` | 192 | 0.0% | distinct |
| `acq_m4h` | `acq_m5` | 192 | 0.0% | distinct |
| `acq_m5` | `acq_m6` | 192 | 0.0% | distinct |
| `acq_m6` | `acq_m8` | 192 | 0.0% | distinct |

## The frontier — paired against `prod` (k=0)

| arm | k | Δ final cost [95% CI] | Δ positives@100 | Δ AP | Δ oracle cost | pairs |
|---|---:|---|---:|---:|---:|---:|
| `acq_p2` | 2 | +0.0630 [+0.0533, +0.0731] | -2.9 | -0.047 | +0.0458 | 192 |
| `acq_m1` | -1 | -0.0113 [-0.0190, -0.0039] | +3.6 | +0.027 | -0.0138 | 192 |
| `acq_m2` | -2 | -0.0301 [-0.0381, -0.0224] | +10.1 | +0.058 | -0.0266 | 192 |
| `acq_m2h` | -2.5 | -0.0338 [-0.0412, -0.0268] | +13.0 | +0.073 | -0.0301 | 192 |
| `acq_m3` | -3 | -0.0313 [-0.0405, -0.0222] | +17.7 | +0.083 | -0.0281 | 192 |
| `acq_m3h` | -3.5 | -0.0337 [-0.0427, -0.0247] | +22.8 | +0.088 | -0.0273 | 192 |
| `acq_m4` | -4 | -0.0328 [-0.0418, -0.0239] | +27.7 | +0.102 | -0.0257 | 192 |
| `acq_m4h` | -4.5 | -0.0329 [-0.0423, -0.0236] | +32.4 | +0.103 | -0.0249 | 192 |
| `acq_m5` | -5 | -0.0298 [-0.0391, -0.0205] | +36.6 | +0.108 | -0.0236 | 192 |
| `acq_m6` | -6 | -0.0229 [-0.0328, -0.0136] | +44.6 | +0.113 | -0.0154 | 192 |
| `acq_m8` | -8 | -0.0101 [-0.0209, +0.0004] | +52.1 | +0.111 | -0.0071 | 192 |

**Falsification arm `acq_p2`**: Δ positives -2.9 — behaves (positives fall, as required)

## H1 — does the frontier turn?

Minimum paired cost delta is at **`acq_m2h` (k=-2.5)**, -0.0338 [-0.0412, -0.0268].

Is any arm deeper than the minimum **resolvably** worse than it?

| deeper arm | k | Δ cost vs the minimum [95% CI] | resolvably worse? |
|---|---:|---|---|
| `acq_m3` | -3 | +0.0025 [-0.0049, +0.0099] | no |
| `acq_m3h` | -3.5 | +0.0001 [-0.0075, +0.0079] | no |
| `acq_m4` | -4 | +0.0010 [-0.0060, +0.0079] | no |
| `acq_m4h` | -4.5 | +0.0009 [-0.0060, +0.0082] | no |
| `acq_m5` | -5 | +0.0040 [-0.0031, +0.0111] | no |
| `acq_m6` | -6 | +0.0109 [+0.0032, +0.0186] | no |
| `acq_m8` | -8 | +0.0237 [+0.0152, +0.0324] | YES |

Resolvably worse past the minimum: `acq_m8`. **The frontier turns — H1 supported.**

## H4 — the posterior-flip landmark

At prevalence π = 7.1% the prior odds are 0.0764, so a
selector's picks become more likely Good than Bad only at
**k\* = −log₂((1−π)/π) = -3.71**.

Measured minimum: k=-2.5. Landmark: -3.71. **H4 NOT supported** (pre-registered window [−4.0, −3.5]).

## H2 — is the optimum resolvable at finer than one step?

Each half step paired **arm-to-arm** against both its integer neighbours on
the same cells. A half step 'resolves' only if it beats both by more than
the ±0.01 tolerance.

| half step | vs | Δ final cost [95% CI] | beats it? |
|---|---|---|---|
| `acq_m2h` (k=-2.5) | `acq_m2` (k=-2) | -0.0037 [-0.0105, +0.0027] | no |
| `acq_m2h` (k=-2.5) | `acq_m3` (k=-3) | -0.0025 [-0.0099, +0.0049] | no |
| `acq_m3h` (k=-3.5) | `acq_m3` (k=-3) | -0.0024 [-0.0092, +0.0047] | no |
| `acq_m3h` (k=-3.5) | `acq_m4` (k=-4) | -0.0009 [-0.0075, +0.0062] | no |
| `acq_m4h` (k=-4.5) | `acq_m4` (k=-4) | -0.0001 [-0.0064, +0.0066] | no |
| `acq_m4h` (k=-4.5) | `acq_m5` (k=-5) | -0.0031 [-0.0095, +0.0031] | no |

**H2 FALSIFIED** — no half step beats both of its integer neighbours; the knob's usable resolution is one bit and the integer grid was right.

## The ship comparison — every arm against the incumbent `acq_m3` (k=-3)

| arm | k | Δ final cost [95% CI] | Δ positives | Δ AP | deep spikes | passes ship rule |
|---|---:|---|---:|---:|---|---|
| `acq_m1` | -1 | +0.0201 [+0.0110, +0.0291] | -14.1 | -0.056 | 0.0% → 0.0% | no |
| `acq_m2` | -2 | +0.0013 [-0.0059, +0.0084] | -7.6 | -0.025 | 0.0% → 0.5% | no |
| `acq_m2h` | -2.5 | -0.0025 [-0.0099, +0.0049] | -4.7 | -0.011 | 0.0% → 0.5% | no |
| `acq_m3h` | -3.5 | -0.0024 [-0.0092, +0.0047] | +5.1 | +0.005 | 0.0% → 0.0% | YES |
| `acq_m4` | -4 | -0.0015 [-0.0081, +0.0050] | +9.9 | +0.019 | 0.0% → 0.5% | YES |
| `acq_m4h` | -4.5 | -0.0016 [-0.0087, +0.0053] | +14.7 | +0.020 | 0.0% → 0.0% | YES |
| `acq_m5` | -5 | +0.0015 [-0.0058, +0.0088] | +18.9 | +0.025 | 0.0% → 0.0% | YES |
| `acq_m6` | -6 | +0.0085 [+0.0008, +0.0158] | +26.8 | +0.030 | 0.0% → 1.0% | no |
| `acq_m8` | -8 | +0.0212 [+0.0124, +0.0301] | +34.4 | +0.028 | 0.0% → 2.6% | no |
