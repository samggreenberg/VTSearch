# Did the voted-media exclusion buy anything? (#3312)

**Status:** pre-registered before the run. The decision rules below are fixed at
submission time; the report records the verdict they produce, including "the
change was right and bought nothing measurable", which is the outcome this study
most expects at production scale.

## The question

PR #3311 (issue #3308) stopped the calibrated threshold from grounding its
quantiles in distributions that include the very votes its models were trained
on. In the slides' vocabulary, M₀(D₋₁) became M₀(D₋₁\D₀): every haystack the
fold-anchored estimator fits or realizes on now covers one identical population,
the unlabeled remainder.

The reasoning is sound and nobody disputes it. What is missing is a number.
Everything justifying the change — and, more pointedly, the
`EXCLUSION_MIN_REMAINDER = 60` floor it ships with — is synthetic:

| Source | What it showed | What it never saw |
|---|---|---|
| A 64-dim Gaussian simulation with acquisition-shaped voting | −0.02 cost at 24% of the corpus voted, −0.003 at 5%, nothing at ≤1% | a real embedding, a real ranker, a real Autopilot |
| One 60-media eval environment (16-dim Gaussians) | unconditional exclusion is catastrophic once the remainder drains (+0.18 cost, FNR 0.7–0.9) — this is what forced the floor | anything above a 60-item haystack |

That is thin evidence for a constant now sitting on the shipped threshold path
of every detector anyone trains. The two sources also disagree about where the
interesting region is, and the shipped floor was placed at the boundary between
them rather than at a measured optimum.

## What is being measured

The cost a VTSearch user ends up paying, on average, under each exclusion
policy — not a mechanism and not an artifact. Everything except the floor is
held at the app's own behaviour: the fused threshold path, the production
linear-SVM head, `calibrate_count=2`, the per-space calibration fraction, the
app's per-mode blend schedule, and the text-sort opening a user gets by typing a
query.

**The live hypothesis is that the change buys nothing measurable at production
scale.** The effect is bounded above by the votes' share of the (≤50k-sampled)
haystack: on `vg_scale_any` at `sim_fraction=0.5` a 150-click horizon votes 7%
of a ~2100-media haystack, where the synthetic curve puts the benefit at
~−0.003. This study is built so that "smaller than we can resolve, and smaller
than we would act on" is a reportable finding rather than a failure.

## Design

**The arm axis is one number.** `exclusion_min_remainder` is the smallest
unlabeled remainder at which the exclusion still fires, so the arms are ordered
and need no sentinel:

| Arm | Floor | What it is |
|---|---|---|
| `off` | `inf` | the exclusion never fires — the pre-#3308 baseline |
| `always` | `0` | unconditional, no floor — what #3311 first implemented |
| `app` | resolved | **the incumbent**: pins nothing, resolves through the app's own `resolve_exclusion_floor` |
| `f250` | `250` | a more conservative floor |

`app` deliberately pins nothing. Pinning `60` would freeze the arm against a
constant that can move underneath the study, and the incumbent has to be
*whatever a live detector does* or the contrast is against a detector nobody
runs.

**Full runs, not paired re-cuts.** The floor sets the threshold, the threshold
sets the acquisition cut, and the cut sets which item Autopilot's Hard pick
samples next — so two arms have collected different votes by their second
trained step. This is the same reason `calibrate_count` (#2897) and
`calibration_fraction` (#3287) each needed a live A/B after their cheap screens:
a knob upstream of acquisition cannot be screened inside one trajectory. The
arms are therefore **not** paired on votes; they are paired on
`(dataset, category, seed, geometry)`, which is what the analyzer bootstraps
over.

**Two stages, because the two questions live in different regimes.**

| | Stage A | Stage B |
|---|---|---|
| `sim_fraction` | 0.5 (~2100-media haystack) | 0.10 (~420-media haystack) |
| horizon | 150 clicks | 380 clicks |
| votes' share of the haystack | 0.2% → 7% | 1% → 90% |
| remainder range | ~2100 → ~1950 | ~419 → ~40 |
| floor binds? | **never** | `f250` at ~170 votes, `app` at ~360, `always` never |
| arms | `off`, `app` | `off`, `always`, `app`, `f250` |
| question | is the exclusion worth anything where users actually work? | is 60 the right floor? |

Stage A carries only two arms **because the floor is inert there by
construction**: with the remainder never below ~1950, `always`, `app` and `f250`
are the same estimator, and spending cells on all four would buy three copies of
one arm.

**`sim_fraction` is the instrument.** It sets the size of the haystack the
threshold's population estimate is fitted on, and therefore the
votes-to-haystack ratio — which is precisely the axis the mechanism runs on. It
was a hard-coded `0.5` before this study.

**Environment.** One dataset, `vg_scale_any` (#3156 + #3115): 12 hand-checked
classes, 300 positives each against one shared 3900-image negative pool, so the
evaluable pool is 4200 at 7.1% prevalence, identical in every cell. That
uniformity is the instrument here as it was in #3287: the swept axis is a *share
of the haystack*, so a grid whose cells' haystacks differed 60-fold in size
would confound the axis with itself. Both voting modes are present via
`siglip` (binary) and the `siglip+dinov3_patch` pair (region), so a per-mode
answer rests on the mode and not on the representation. 4 seeds.

## Reading the axis

Results are banded on the **votes' share of the haystack**, never pooled and
never on raw vote count. The mechanism's size is bounded by that share, so a
pooled average across the horizon is exactly the number that hides it — and raw
vote count would put stage A's 150 votes and stage B's 150 votes in the same
bucket while they consume 7% and 36% of their haystacks.

## Pre-registered decision rules

Cost at inclusion 0 is the metric; `regret_honest` is reported beside it.
`HARM_TOLERANCE = 0.01` is the margin PR #2891 pre-registered for this family of
decisions, and `NULL_BOUND` is set equal to it: an arm that cannot move cost by
as much as we would tolerate losing is not a difference anyone should act on,
whichever way it points.

- **Ship-keeping.** `app` stays shipped unless another arm beats it by more than
  `HARM_TOLERANCE` with the CI excluding 0, in **both** stages. An arm that wins
  one stage and loses the other does not ship: the floor exists precisely to
  serve both regimes.
- **Stage A's null is a result.** If `off` vs `app` is not resolvable, the
  report quotes the bound and says the change is a rigor improvement that costs
  nothing. It must not be written up as a win.
- **Resolved-but-negligible is its own finding**, distinct from both. A
  difference that clears 2 SE while its whole interval sits inside
  ±`NULL_BOUND` means the effect is real *and* not worth a decision. Reporting
  it as "no effect" would discard evidence that the shipped arm does measurable
  work; reporting it as a win would invent a reason to act.
- **Harm.** If `off` beats `app` by more than `HARM_TOLERANCE` with the CI
  excluding 0 at *either* stage, #3308 is a regression: say so, and revert or
  re-scope the exclusion.
- **The floor's own justification.** If `always` is never worse than `app`
  anywhere in stage B, the floor is unjustified complexity and should be
  deleted — the synthetic evidence that motivated it did not survive contact
  with a real environment. If `f250` beats `app`, the floor is in the right
  place but at the wrong value.
- **Pointwise, not just pooled.** No arm ships if it is worse than the incumbent
  by more than `HARM_TOLERANCE` in **any** band. An arm can win overall while
  being worse everywhere a short session actually lives, and a short session is
  most sessions.

## Validity checks the analyzer runs before any verdict

These are not decoration: each one is a way the study could be silently wrong.

- **The trap check.** Two arms whose floors agree above some remainder *are the
  same estimator* above it, so on every step where the remainder clears both
  floors their thresholds must be **identical**, not similar. `always` vs `app`
  must match on every stage-B step with remainder ≥ 60; `f250` vs `app` above
  250. Anything below 1.0 means an arm ran under the wrong environment — the
  "cluster ran the previous commit" failure — and no number in the report can be
  trusted.
- **The floor regime.** Where each arm's exclusion was actually live,
  reconstructed from `n_remainder` alone (exactly the count
  `apply_vote_exclusion` compares against the floor). An arm that never
  excludes, or always does, cannot be a contrast *about the floor*. This checks
  the harness rather than merely describing it.
- **Arm-vs-directory.** Every cell's `exclusion_arm` column is required to agree
  with the directory it was read out of; a mismatch aborts the analysis rather
  than being averaged in.
- **Completeness.** Unreadable, zero-byte and starved cells are counted and
  reported. A cell that never found both classes is dropped from the paired
  difference and counted, because computing an arm's mean over the cells that
  happened to work flatters exactly the arm that starved.
- **One opening.** `assert_one_opening` refuses to pool cells that started
  differently (#3278).

## Deliverables

- `REPORT.md` with the mandatory quality-over-clicks pair (averages and per-seed
  individuals, anchored at click 0 on each cell's own zero-click text sort) and
  `viewer.html`, per the `grid-experiments` skill. Panels are stage × geometry
  and are never averaged together: a stage *is* a haystack size, which is this
  study's axis.
- Whatever the verdict, fold it back into `EXCLUSION_MIN_REMAINDER`'s docstring,
  which currently cites synthetic numbers and should cite this run instead.

## Running it

```bash
bash launch_exclusion_3308.sh prepare          # stage 0, ONCE, shared by every arm
bash launch_exclusion_3308.sh baseline         # the click-0 text-sort anchor
bash launch_exclusion_3308.sh size A 0         # time ONE cell per stage, per geometry
bash launch_exclusion_3308.sh size B 12
bash launch_exclusion_3308.sh arms             # both stages, then one cross-arm analyze
```

Size before submitting. Stage B is **not** stage A scaled: its haystack is 5×
smaller but its horizon is 2.5× longer and it carries far more votes per step,
so its per-cell cost has to be measured rather than divided.
