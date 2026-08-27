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
| `acq_m4` — k=-4 | 0.9518 | +0.0676 | 98% |
| `acq_m3` — k=-3 | 0.9426 | +0.0583 | 98% |
| `acq_m2` — k=-2 | 0.9357 | +0.0515 | 98% |
| `acq_m1` — k=-1 | 0.9154 | +0.0311 | 98% |
| `prod` — prod (k=0, shipped) | 0.8842 | +0.0000 | 0% |
| `acq_p2` — k=+2 (falsifier) | 0.7364 | -0.1478 | 99% |
| `rank_pin` — rank-pinned 0.959 | 0.9600 | +0.0758 | 100% |

## Per-arm

| arm | trajectories | positives @100 | positives @50 | final cost | mean warm cost | final AP | oracle cost | genuine blips |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `acq_m4` — k=-4 | 147 | 19.0 | 6.0 | 0.134 | 0.169 | 0.802 | 0.101 | 6.1% |
| `acq_m3` — k=-3 | 147 | 18.0 | 5.0 | 0.129 | 0.169 | 0.817 | 0.101 | 5.4% |
| `acq_m2` — k=-2 | 147 | 11.0 | 4.0 | 0.144 | 0.183 | 0.808 | 0.115 | 5.4% |
| `acq_m1` — k=-1 | 147 | 6.0 | 4.0 | 0.138 | 0.181 | 0.772 | 0.113 | 3.4% |
| `prod` — prod (k=0, shipped) | 147 | 4.0 | 3.0 | 0.137 | 0.184 | 0.696 | 0.113 | 5.4% |
| `acq_p2` — k=+2 (falsifier) | 147 | 3.0 | 3.0 | 0.178 | 0.236 | 0.635 | 0.154 | 5.4% |
| `rank_pin` — rank-pinned 0.959 | 147 | 6.0 | 5.0 | 0.142 | 0.175 | 0.815 | 0.110 | 7.5% |

## Paired against `prod` — cells are `(category, seed)`, never steps

| arm | metric | n | control | arm | median Δ | 95% CI on mean Δ | p |
|---|---|---:|---:|---:|---:|---:|---:|
| `acq_m4` | positives_100 | 147 | 4.000 | 19.000 | +11.000 | [+16.1633, +21.8571] | 0.00000 |
| `acq_m4` | final_cost | 147 | 0.137 | 0.134 | -0.009 | [-0.0236, -0.0067] | 0.00073 |
| `acq_m4` | mean_cost_warm | 147 | 0.184 | 0.169 | -0.009 | [-0.0211, -0.0058] | 0.00011 |
| `acq_m4` | final_oracle_cost | 147 | 0.113 | 0.101 | -0.009 | [-0.0199, -0.0059] | 0.00004 |
| `acq_m3` | positives_100 | 147 | 4.000 | 18.000 | +10.000 | [+12.7075, +17.5510] | 0.00000 |
| `acq_m3` | final_cost | 147 | 0.137 | 0.129 | -0.011 | [-0.0254, -0.0045] | 0.00008 |
| `acq_m3` | mean_cost_warm | 147 | 0.184 | 0.169 | -0.010 | [-0.0231, -0.0065] | 0.00008 |
| `acq_m3` | final_oracle_cost | 147 | 0.113 | 0.101 | -0.011 | [-0.0210, -0.0041] | 0.00001 |
| `acq_m2` | positives_100 | 147 | 4.000 | 11.000 | +4.000 | [+7.9388, +11.6190] | 0.00000 |
| `acq_m2` | final_cost | 147 | 0.137 | 0.144 | -0.009 | [-0.0243, -0.0094] | 0.00003 |
| `acq_m2` | mean_cost_warm | 147 | 0.184 | 0.183 | -0.009 | [-0.0224, -0.0080] | 0.00007 |
| `acq_m2` | final_oracle_cost | 147 | 0.113 | 0.115 | -0.006 | [-0.0210, -0.0075] | 0.00007 |
| `acq_m1` | positives_100 | 147 | 4.000 | 6.000 | +2.000 | [+2.6667, +4.6327] | 0.00000 |
| `acq_m1` | final_cost | 147 | 0.137 | 0.138 | -0.007 | [-0.0191, -0.0033] | 0.00960 |
| `acq_m1` | mean_cost_warm | 147 | 0.184 | 0.181 | -0.005 | [-0.0187, -0.0048] | 0.00084 |
| `acq_m1` | final_oracle_cost | 147 | 0.113 | 0.113 | -0.004 | [-0.0157, -0.0025] | 0.00635 |
| `acq_p2` | positives_100 | 147 | 4.000 | 3.000 | -1.000 | [-1.5850, -1.0748] | 0.00000 |
| `acq_p2` | final_cost | 147 | 0.137 | 0.178 | +0.021 | [+0.0198, +0.0398] | 0.00000 |
| `acq_p2` | mean_cost_warm | 147 | 0.184 | 0.236 | +0.021 | [+0.0155, +0.0323] | 0.00000 |
| `acq_p2` | final_oracle_cost | 147 | 0.113 | 0.154 | +0.013 | [+0.0153, +0.0332] | 0.00000 |
| `rank_pin` | positives_100 | 147 | 4.000 | 6.000 | +2.000 | [+4.1903, +6.9660] | 0.00000 |
| `rank_pin` | final_cost | 147 | 0.137 | 0.142 | -0.011 | [-0.0217, -0.0042] | 0.00276 |
| `rank_pin` | mean_cost_warm | 147 | 0.184 | 0.175 | -0.006 | [-0.0199, -0.0037] | 0.00187 |
| `rank_pin` | final_oracle_cost | 147 | 0.113 | 0.110 | -0.009 | [-0.0194, -0.0047] | 0.00082 |

## Ship rule (pre-registered)

Adopt iff positives rise (p<0.05) **and** the 95% upper bound on the final-cost delta is below +0.01 **and** deep-spike incidence does not rise **and** the lever actually moved.

| arm | positives rose | cost did not regress | spikes did not rise | lever moved | **ADOPT** |
|---|:--:|:--:|:--:|:--:|:--:|
| `acq_m4` | yes | yes | yes | yes | **yes** |
| `acq_m3` | yes | yes | yes | yes | **yes** |
| `acq_m2` | yes | yes | yes | yes | **yes** |
| `acq_m1` | yes | yes | yes | yes | **yes** |
| `rank_pin` | yes | yes | yes | yes | **yes** |

**Arms passing every criterion:** acq_m4, acq_m3, acq_m2, acq_m1, rank_pin


## Data read

| arm | cell files | trajectories | never found a positive | unreadable | zero-byte |
|---|---:|---:|---:|---:|---:|
| `acq_m4` | 152 | 147 | 5 | 0 | 0 |
| `acq_m3` | 152 | 147 | 5 | 0 | 0 |
| `acq_m2` | 152 | 147 | 5 | 0 | 0 |
| `acq_m1` | 152 | 147 | 5 | 0 | 0 |
| `prod` | 152 | 147 | 5 | 0 | 0 |
| `acq_p2` | 152 | 147 | 5 | 0 | 0 |
| `rank_pin` | 152 | 147 | 5 | 0 | 0 |

## Figures

![fig1_frontier.png](figures/fig1_frontier.png)

![fig2_lever_verification.png](figures/fig2_lever_verification.png)

![fig3_guardrails.png](figures/fig3_guardrails.png)
