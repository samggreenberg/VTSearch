# #2877 on the pile — the acquisition cut's generalisation check, in both voting modes

**Pre-registered 2026-08-29, before any arm cell existed.** Written first on
purpose: every decision rule below is one this study is committed to, and the
point of writing them down now is that the run cannot be read into agreeing with
whichever answer arrives.

Issue #2877 · related #2876, #2878, #2905, #2909, #2910 · branch
`run/acq-incl-pile-2877` · base dev `53dd14cb4` · worktree
`/exp/sgreenberg/projects/vts-acq-2877` · study
`/expscratch/sgreenberg/acq-2877`.

## What is being measured

`ACQUISITION_INCLUSION_OFFSET` is the gap between two jobs one number does. The
*reported* threshold is the line the user sees and the line `cost = FPR + FNR`
is scored at. The *acquisition* cut is what Autopilot's Hard pick consumes — and
it does not use it as a decision boundary at all, but as a **rank position**:
rank every item descending, find the first position at or below the threshold,
take the nearest unlabeled item in rank space. The offset moves only the second.

`k` is that offset. A **negative** `k` prices false alarms higher, raises the
cut, moves it *up* the ranking, and returns *more* positives — the opposite of
the direction the cost weights suggest, which is why the falsification arm is
load-bearing.

## Why the check is still owed

Three environments have measured this constant and none of them is the check
#2877 asked for.

| environment | voting mode | verdict on `-3` | status |
|---|---|---|---|
| `coco_val × siglip2` (#2876) | binary | ships — positives 4 → 18, cost −0.011 | stands |
| `visual_genome_m × siglip` (#2877) | **binary**, not region | **fails** — cost CI [+0.003, +0.022] vs +0.01 | stands, but is not the environment it claimed |
| `visual_genome_m × dinov3_patch` (#2905) | region | passed | **void** (PR #3119) |

#2877 pre-registered its arm as region voting and justified the whole check on
region voting's scoring geometry — *"a media's score is a max over ~24
region-node scores, so the Bad mode is an extreme-value statistic."* That is
true of region voting and false of the arm it ran: `visual_genome_m__siglip`
carries no `patch_grid`, so `region_voting=True` fell back to whole-image
training, whole-image scoring, and the **binary** blend schedule. It is a second
*binary* environment, and one that rejects `-3`.

#2905 ran the real region arm and lost it. Dev `b7d528d8` (#2943), two days
later, fixed `_score_pool` scoring the acquisition pool by whole-image vectors
while the threshold was cut on region max-pooled scores. Autopilot's Hard pick
compares `ranking[cid] <= threshold` absolutely, so the two must share a space;
they did not, and the aggressive arms sat pinned above the entire pool on 39% of
`k=−3` steps — clamped precisely where the decision was, while the falsifier
moved *away* from the ceiling and was spared.

So the shipped **−1** (#2909) is a compromise resting on binary evidence alone,
and `docs/ML.md` says so in as many words: *"The region-voting check is
therefore still outstanding."* This is that check.

## The environment, and why it is not `visual_genome_m`

**`vg_scale_any` × {`siglip`, `siglip+dinov3_patch`}**, from the shared
pre-embedded pile — no GPU, no re-embed.

`vg_scale_any` (#3156 + #3115) is 12 hand-checked classes at 300 positives each
against **one shared 3900-image negative pool**, labelled from COCO's exhaustive
annotation of the half of Visual Genome that COCO sourced, and repaired by a
human review pass. Cells are *designated*: prevalence is **7.1% in every one of
them, by construction**.

That uniformity is the instrument, not a convenience:

* **#2910 measured this offset's benefit as a decreasing function of positive
  supply** — AP response slope −0.0207 on log category prevalence, sharply
  positive in starved cells and negative in rich ones. On `visual_genome_m` the
  selected categories run **25 to 1645 positives**, a 60-fold spread. A per-arm
  mean there is an average over the very axis the effect runs on, and the axis
  is not reported.
* `visual_genome_m`'s thin categories produce cells with **no trainable step at
  all** — the simulator writes a header and nothing else, which is non-empty,
  parses cleanly, and counts as present in every "N/N cells" tally (#3115
  launched 208 such cells and its first two completions were header-only).

Holding supply flat does not make the supply question go away; it makes an arm
difference attributable to the arm. The supply question is #2910's, and this run
deliberately does not confound itself with it.

### Two voting modes, one embedder between them

| arm | style | mode |
|---|---|---|
| `siglip` | `whole_image` | binary |
| `siglip+dinov3_patch` | `whole_image` | binary |
| `siglip+dinov3_patch` | `max_patch` | **region** |

The pair runs **both styles inside one task, off one loaded pickle** — one
sim/test split, one startup exemplar, one trajectory-seeding query. So the
region-vs-binary contrast is paired cell-for-cell and differs *only* in the
scoring geometry.

This matters because the obvious grid cannot say that. #3115 reported a per-mode
headline off a grid whose binary cells were all SigLIP and whose region cells
were all DINOv3: the sign flip it measured is real, its attribution to the
voting mode is not, because the embedder moved with it. Preflight check 13b now
refuses that shape, and this grid satisfies it by construction.

The `siglip × whole_image` half is kept because it is the arm directly
comparable to #2876 and #2877, both of which were single-vector whole-image
environments.

**The region arm is the PAIR, not bare `dinov3_patch`.** DINOv3 has no text
tower, so on its own every cell falls back to three random known-goods while
every other arm opens on a typed query (#3269/#3278). `k` is an offset applied
to a rank position in a ranking *the opening creates*, so a seeding contrast
hidden inside the mode contrast would not be a detail. SigLIP ranks the query;
DINOv3 does every piece of learning. `CALIB_REQUIRE_OPENING=text` asserts it per
cell, and `--require-region-voting` asserts the patch geometry is actually
present before anything is submitted (7747/7747 medias carry a `patch_grid`).

## Arms

#2876/#2877/#2905's verbatim, so all four environments stay comparable.

| arm | `CALIB_ACQ_INCLUSION_OFFSET` | role |
|---|---|---|
| `prod` | `0` | control — the pre-#2876 coupled behaviour, one threshold doing both jobs |
| `acq_m1` | `-1` | **the shipped default** (#2909) |
| `acq_m2` | `-2` | where #2877 found the ranking benefit saturates |
| `acq_m3` | `-3` | what #2876 shipped and #2877 rejected |
| `acq_m4` | `-4` | the far end |
| `acq_p2` | `+2` | **falsification arm — must make positives worse** |
| `rank_pin` | `0` + `CALIB_ACQ_RANK_PERCENTILE=0.959` | the pinned-quantile parameterisation |

Every arm names its offset explicitly, `acq_m1` included. An unset offset
resolves to the shipped constant, so leaving one unset would silently duplicate
another arm the day that constant moves — which is the pre-#2878 failure in
reverse.

`acq_p2` is kept. It is the only thing separating "the lever works" from "any
perturbation of the sampling position changes the numbers", and it has behaved
in all three prior environments.

Everything not on the arm axis runs at production: the head is the linear SVM
(#3198, unpinned), `calibration_fraction` is the per-space default (#3290,
unpinned), the voted-media exclusion floor is the app's own (#3308, unpinned),
and the blend schedule is whatever `production_schedule_for` picks for the
cell's voting mode (#2849, unpinned) — so region cells blend under `slow_cap50`
and binary ones under `cap50`. Pinning the schedule would measure the offset
under a schedule no user of that mode runs, and "the schedule is already
mode-gated while the offset is not" is half of what this question is about.

## Endpoints and the ship rule (pre-registered, unchanged from #2876)

* **Decision endpoint:** `final_cost` (FPR + FNR at inclusion 0) at t=100,
  paired at the `(category, seed, style)` cell. Reporting is cut at inclusion 0
  in *every* arm, so cost is comparable across arms.
* **Mechanism endpoints:** positives found by t=100 and by t=50, and
  `final_ap`. AP is what separates "the extra labels taught the model something"
  from "they were redundant", and it is what made #2877's mechanism legible.
* **Guardrails:** deep-spike incidence, worst-step warm regret, oracle cost.

**Adopt an arm iff** positives rise (p < 0.05) **and** the 95% upper bound on
the mean final-cost delta is below **+0.010** **and** deep-spike incidence does
not rise **and** the lever actually moved.

### The verdict is per voting mode

The ship rule is evaluated **separately in each mode**, and the pooled table is
printed as descriptive only. The whole reason this question survived three runs
is that the answer moved between environments; a mean over a grid spanning two
of them is precisely the number that would hide it. `analyze_acq.py` enforces
this, and `selftest_analyze_acq.py` plants a grid where the two modes disagree
by an amount that pooling brings back inside the tolerance.

### One guardrail does not transfer, and is read as a contrast

`SPIKE_DEEP_COST = 0.25` is an **absolute** cost, calibrated to COCO's ~0.137
scale. Costs here are 2–3× that, so the base deep-spike rate will be high in
both modes. This is #2877's own recorded lesson and the thresholds are
deliberately **not** re-tuned: moving them would make the guardrail
incomparable to the three prior environments. The ship rule reads the *paired
McNemar contrast*, which stays valid at any base rate. The base rates are
reported and are not comparable across environments.

## Sizing

**Declared at 24 seeds, run as a prefix.** `CALIB_CELL_ORDER=seed` makes the
array seed-major (index = seed × 12 + category), so every seed block is a
complete design, a truncated run loses *seeds* rather than *categories*, and a
top-up is an array-range extension with every cell already on disk still
counting.

#2877's one unrecoverable mistake was inheriting #2876's seed count with its arm
table: the decision-endpoint CI came back **[−0.014, +0.019]**, which is not a
null — it is an interval containing both "ship it" and "revert it", reported as
though it were one. The sizing input (the paired SD on `final_cost` in *this*
environment) cannot be known before cells exist, so the fix is not a better
guess but a number the run reports: `analyze_acq.py` prints the realized paired
SD per mode and the `n` a ±0.010 half-width needs.

**Wave 1 is 16 seeds** = 192 pairs per arm per mode, which is #2905's derived
`n ≈ 193` at its SD of 0.0709. If the realized SD is larger, the top-up is
`arms 16-23`.

## Cost, measured on this grid

| half | cell | elapsed | peak RSS | `--mem` | `%N` per arm |
|---|---|---|---|---|---|
| `reg` | `siglip+dinov3_patch`, whole_image + max_patch | 16m05s / 16m42s | 5.0 GB | 8G | 12 |
| `bin` | `siglip`, whole_image | 3m21s | 0.9 GB | 3G | 4 |

Measured by `launch_acq_2877.sh size` (SLURM 590192/590193/590194), not
inherited: GRID-PLAYBOOK's table would have said 16G for the region cell, and
the difference is eight slots of concurrency per arm. The throttles are then set
so the two halves finish together and the study stays inside `cpu_limit`
(cpu=240 at 2 charged per task ⇒ 120 concurrent; mem=1100000M).

The halves are separate arrays because a region cell cannot share a memory
request with a whole-image one and memory is the binding per-user quota. They
are two index spaces off **one** prepare, so a task index means one cell.

## Outcomes and what each means

* **`-3` (or nearby) ships in both modes.** The disagreement is by
  *environment*, not by mode, and no gate reaches it. `-1` stays the global
  value; the constant's real successor is #2910's supply-dependent offset.
* **The modes want different values, and the DiD says so.** Then the gate is
  supported — `ACQUISITION_INCLUSION_OFFSET_BY_MODE`, mirroring
  `PRODUCTION_SCHEDULE_BY_MODE`, resolved where `_blend_schedule_for_snap`
  resolves the schedule and keyed on the *scoring geometry*, not on how the user
  voted. (#2909 records why the obvious implementation is wrong:
  `ctx.embedder_type` is the detector's locked type, which is `""` for a legacy
  detector on a patch dataset.)
* **Neither `-1` nor `-3` clears the rule in region voting.** The strongest
  result available: the constant is environment-sensitive in a fourth
  environment too, the gate is a distraction, and #2910 becomes the only live
  proposal.
* **The lever does not diverge from `prod` at all here.** That is also an
  answer, and it is the one #2905's re-run note said to look for first: an
  acquisition cut pinned at the top of the pool is *inert*, and an inert arm
  looks exactly like a lever that does nothing. Probed before launch (below),
  not diagnosed afterwards.

In every case the shipped `-1` should be revisited, because it currently rests
on one voting mode.

## What this run does NOT answer

* **The supply question (#2910).** Prevalence is flat here by construction, so
  this grid cannot place a supply-dependent offset. It also cannot be
  contaminated by supply, which is the trade.
* **Whether the offset transfers to `visual_genome_m`'s free-text vocabulary.**
  `vg_scale_any` is COCO-labelled by design; that is what makes its negatives
  trustworthy (VG's own recall over these classes is 0.76, and 1.4% of the
  images it calls negative hold the object).
* **Anything about the opening.** Every cell opens on a typed query. Numbers
  from #2876/#2877 predate #3269 and seeded from a *crop*, a ranking no user
  produces; cross-run comparisons of anything that depends on how a run starts —
  positive starvation, `n_good`, early-trajectory cost — are qualified by that,
  and `_cells_io.assert_one_opening` refuses to pool the two.
