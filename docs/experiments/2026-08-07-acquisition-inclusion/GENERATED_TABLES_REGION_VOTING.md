# Acquisition/reporting threshold decoupling — does it buy back the positives?

> # ⚠️ SEEDING CAVEAT — these runs did not start the way the app does
>
> **Recorded 2026-08-26 (#3156).** Autopilot seeds its first three Good votes from
> a **text sort**: the user types a query and votes down that ranking. Until
> PR #3269 this harness instead ranked every item by cosine to a **crop of one
> boxed positive** — a ranking no user ever produces — and passed it as
> `seed_scores`, the argument that `al_strategies`, `EVAL.md` and
> `voting_iterations` all describe as "similarity to the **typed query**".
>
> **What to distrust here:** anything that depends on *how a run starts* —
> positive starvation, stuck or never-got-going runs, `n_good`, and
> early-trajectory cost. Measured on one cell after the fix, text seeding put the
> first positive at **rank 1** with five in the top 20, while the exemplar that
> crop-seeding made look like the dataset's hardest positive ranked **4006 of
> 7749** for its own class.
>
> **What still holds:** within-study contrasts where every arm seeded identically,
> which is most of what these reports conclude — the seeding is a shared baseline
> shift, not an arm-dependent one.
>
> See [the harness seeded from a crop](../../../scripts/experiments/lessons/2026-08-26-the-harness-seeded-from-a-crop.md).


Design: `docs/ML.md` (threshold calibration). Reporting is cut at inclusion 0 in **every** arm; only the selector's cut moves.


## Lever verification — where each arm actually sampled

| arm | median `acq_pool_percentile` | shift vs control | steps where acq ≠ reporting |
|---|---:|---:|---:|
| `acq_m4` — k=-4 | 0.9995 | +0.0127 | 97% |
| `acq_m3` — k=-3 | 0.9990 | +0.0122 | 97% |
| `acq_m2` — k=-2 | 0.9975 | +0.0107 | 97% |
| `acq_m1` — k=-1 | 0.9940 | +0.0071 | 97% |
| `prod` — prod (k=0, shipped) | 0.9869 | +0.0000 | 0% |
| `acq_p2` — k=+2 (falsifier) | 0.9410 | -0.0458 | 97% |
| `rank_pin` — rank-pinned 0.959 | 0.9995 | +0.0126 | 100% |

## Per-arm

| arm | trajectories | positives @100 | positives @50 | final cost | mean warm cost | final AP | oracle cost | genuine blips |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `acq_m4` — k=-4 | 541 | 21.0 | 12.0 | 0.364 | 0.392 | 0.443 | 0.321 | 8.7% |
| `acq_m3` — k=-3 | 541 | 20.0 | 11.0 | 0.361 | 0.390 | 0.443 | 0.314 | 8.9% |
| `acq_m2` — k=-2 | 541 | 19.0 | 10.0 | 0.365 | 0.391 | 0.444 | 0.312 | 8.3% |
| `acq_m1` — k=-1 | 541 | 17.0 | 9.0 | 0.353 | 0.387 | 0.449 | 0.315 | 8.5% |
| `prod` — prod (k=0, shipped) | 541 | 14.0 | 7.0 | 0.362 | 0.387 | 0.447 | 0.310 | 7.9% |
| `acq_p2` — k=+2 (falsifier) | 541 | 7.0 | 4.0 | 0.361 | 0.391 | 0.442 | 0.326 | 9.1% |
| `rank_pin` — rank-pinned 0.959 | 541 | 18.0 | 10.0 | 0.363 | 0.399 | 0.440 | 0.314 | 8.9% |

## Paired against `prod` — cells are `(category, seed)`, never steps

| arm | metric | n | control | arm | median Δ | 95% CI on mean Δ | p |
|---|---|---:|---:|---:|---:|---:|---:|
| `acq_m4` | positives_100 | 541 | 14.000 | 21.000 | +3.000 | [+4.8705, +6.1036] | 0.00000 |
| `acq_m4` | final_cost | 541 | 0.362 | 0.364 | +0.000 | [-0.0028, +0.0107] | 0.46079 |
| `acq_m4` | mean_cost_warm | 541 | 0.387 | 0.392 | +0.000 | [-0.0030, +0.0066] | 0.86228 |
| `acq_m4` | final_oracle_cost | 541 | 0.310 | 0.321 | +0.001 | [+0.0015, +0.0095] | 0.00011 |
| `acq_m4` | final_ap | 541 | 0.447 | 0.443 | +0.000 | [-0.0051, +0.0041] | 0.17338 |
| `acq_m3` | positives_100 | 541 | 14.000 | 20.000 | +3.000 | [+4.3401, +5.5195] | 0.00000 |
| `acq_m3` | final_cost | 541 | 0.362 | 0.361 | +0.000 | [-0.0034, +0.0095] | 0.79581 |
| `acq_m3` | mean_cost_warm | 541 | 0.387 | 0.390 | +0.000 | [-0.0030, +0.0062] | 0.61868 |
| `acq_m3` | final_oracle_cost | 541 | 0.310 | 0.314 | +0.000 | [+0.0002, +0.0078] | 0.00410 |
| `acq_m3` | final_ap | 541 | 0.447 | 0.443 | +0.000 | [-0.0036, +0.0043] | 0.62359 |
| `acq_m2` | positives_100 | 541 | 14.000 | 19.000 | +2.000 | [+3.5194, +4.6063] | 0.00000 |
| `acq_m2` | final_cost | 541 | 0.362 | 0.365 | +0.000 | [-0.0028, +0.0079] | 0.49461 |
| `acq_m2` | mean_cost_warm | 541 | 0.387 | 0.391 | +0.000 | [-0.0045, +0.0042] | 0.43564 |
| `acq_m2` | final_oracle_cost | 541 | 0.310 | 0.312 | +0.001 | [-0.0008, +0.0061] | 0.00505 |
| `acq_m2` | final_ap | 541 | 0.447 | 0.444 | +0.000 | [-0.0034, +0.0041] | 0.34313 |
| `acq_m1` | positives_100 | 541 | 14.000 | 17.000 | +1.000 | [+1.7856, +2.7246] | 0.00000 |
| `acq_m1` | final_cost | 541 | 0.362 | 0.353 | +0.000 | [-0.0085, +0.0035] | 0.19428 |
| `acq_m1` | mean_cost_warm | 541 | 0.387 | 0.387 | +0.000 | [-0.0059, +0.0035] | 0.48402 |
| `acq_m1` | final_oracle_cost | 541 | 0.310 | 0.315 | +0.000 | [-0.0054, +0.0023] | 0.72188 |
| `acq_m1` | final_ap | 541 | 0.447 | 0.449 | +0.000 | [-0.0019, +0.0056] | 0.42394 |
| `acq_p2` | positives_100 | 541 | 14.000 | 7.000 | -4.000 | [-5.1941, -4.2661] | 0.00000 |
| `acq_p2` | final_cost | 541 | 0.362 | 0.361 | +0.001 | [-0.0063, +0.0110] | 0.11398 |
| `acq_p2` | mean_cost_warm | 541 | 0.387 | 0.391 | +0.001 | [-0.0035, +0.0099] | 0.01322 |
| `acq_p2` | final_oracle_cost | 541 | 0.310 | 0.326 | +0.000 | [-0.0010, +0.0101] | 0.19941 |
| `acq_p2` | final_ap | 541 | 0.447 | 0.442 | -0.007 | [-0.0205, -0.0099] | 0.00000 |
| `rank_pin` | positives_100 | 541 | 14.000 | 18.000 | +2.000 | [+2.5009, +3.4104] | 0.00000 |
| `rank_pin` | final_cost | 541 | 0.362 | 0.363 | +0.000 | [-0.0064, +0.0056] | 0.74971 |
| `rank_pin` | mean_cost_warm | 541 | 0.387 | 0.399 | +0.000 | [-0.0041, +0.0056] | 0.56950 |
| `rank_pin` | final_oracle_cost | 541 | 0.310 | 0.314 | +0.000 | [-0.0009, +0.0064] | 0.04202 |
| `rank_pin` | final_ap | 541 | 0.447 | 0.440 | +0.000 | [-0.0037, +0.0041] | 0.22630 |

## Ship rule (pre-registered)

Adopt iff positives rise (p<0.05) **and** the 95% upper bound on the final-cost delta is below +0.01 **and** deep-spike incidence does not rise **and** the lever actually moved.

| arm | positives rose | cost did not regress | spikes did not rise | lever moved | **ADOPT** |
|---|:--:|:--:|:--:|:--:|:--:|
| `acq_m4` | yes | no | yes | yes | **no** |
| `acq_m3` | yes | yes | yes | yes | **yes** |
| `acq_m2` | yes | yes | yes | yes | **yes** |
| `acq_m1` | yes | yes | yes | yes | **yes** |
| `rank_pin` | yes | yes | yes | yes | **yes** |

**Arms passing every criterion:** acq_m3, acq_m2, acq_m1, rank_pin


## Data read

| arm | cell files | trajectories | never found a positive | unreadable | zero-byte |
|---|---:|---:|---:|---:|---:|
| `acq_m4` | 552 | 541 | 11 | 0 | 0 |
| `acq_m3` | 552 | 541 | 11 | 0 | 0 |
| `acq_m2` | 552 | 541 | 11 | 0 | 0 |
| `acq_m1` | 552 | 541 | 11 | 0 | 0 |
| `prod` | 552 | 541 | 11 | 0 | 0 |
| `acq_p2` | 552 | 541 | 11 | 0 | 0 |
| `rank_pin` | 552 | 541 | 11 | 0 | 0 |

## Figures

![fig1_frontier.png](figures/fig1_frontier.png)

![fig2_lever_verification.png](figures/fig2_lever_verification.png)

![fig3_guardrails.png](figures/fig3_guardrails.png)
