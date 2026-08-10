# Should VTSearch force safe thresholds on? (issue #2799)

_An Autopilot simulation study on the HLTCOE Grid, run on a fresh `dev` worktree
(`claude/safe-thresholds-2799`, dev @ 0bf4a815). Every number below is generated
by `scripts/experiments/calibration/analyze_ab.py` (the ON/OFF A/B) and
`analyze_safe.py` (the within-run GMM variants) from the per-cell CSVs; the prose
is written on top of those numbers._

## BLUF

**Yes — turn `safe_thresholds` on for everyone.** On the production region-vote
arm (`dinov3_patch` × `max_patch`) with the **production linear head**, at the
steps a user actually sees a detector, over the 6–20-vote window where the blend
has authority (226 paired cells):

| | safe OFF | safe ON | Δ | cells improved | p |
|---|---|---|---|---|---|
| inclusion-weighted cost | 0.5464 | **0.4722** | **−0.0742** | 81 % | 5.6e−23 |
| FNR (missed needles) | 0.4163 | **0.3456** | **−0.0708** | 69 % | 2.9e−13 |
| FPR | 0.1301 | **0.1267** | −0.0034 | — | 8.0e−03 |
| regret vs the oracle cut | 0.1896 | **0.1195** | **−0.0700** | 76 % | 4.7e−21 |

Three things make this a clean recommendation rather than a metric trade:

1. **It is not bought with permissiveness.** The recurring failure mode in the
   #2790 work was a "fix" that games the FPR/FNR-weighted cost by cutting lower
   and wrecking precision. Here FNR drops **and** FPR does not rise on the
   production arm — the cut lands *closer to that model's own oracle* (regret
   −0.070), which is what "better calibrated" is supposed to mean.
2. **It keeps paying after its authority ends.** Past 20 votes the blend is pure
   cross-calibration, yet safe-ON is still ahead (cost −0.0197, FNR −0.0312,
   p = 0.005). That gain can only travel through **selection feedback**: the
   threshold drives Autopilot's Hard pick, so a better-placed cut surfaces better
   items to vote on, and the whole trajectory improves. A within-step
   counterfactual cannot see this at all — it is the reason this study is an A/B.
3. **Ranking is untouched.** Average precision moves +0.001 (n.s.). The blend
   changes *where the line is drawn* and *what gets labelled*, not how well the
   model orders the pool.

The control arm (`siglip` × `whole_image`, single-vector, no region pooling)
agrees in direction and adds the cold-start argument: safe-ON **eliminates
degenerate "admit nothing" thresholds** (0.0078 → 0.0000, p = 0.012) — the
`too_few_default` cold-start failure #2788 flagged. Its FPR rises slightly
(+0.0107) where the max_patch arm's does not.

**Scope note.** This measures VG region voting on the linear head, 30 votes deep.
It does not re-measure the MLP (out of scope by request), audio/video media, or
the ≫30-vote regime.

## The question, and why the cached #2781 cells could not answer it

`safe_thresholds` blends the conformal threshold with a GMM threshold fitted on
the current score distribution (`calculate_safe_threshold`): **full** authority
below 6 votes, majority authority to ~13. It ships **off**
(`vtsearch/settings_models.py`, a per-user checkbox in the settings modal), and
the #2781 calibration study ran every arm with it off, so it had never been
measured under region voting.

Two things forced a fresh two-arm run rather than a re-read of cached data:

1. **The head.** #2781's cached `safe_thresholds False` cells ran the harness's
   historical auto-sized **MLP**. Production has trained a **linear (logistic)**
   head since #2790/#2809. Pairing new safe-ON cells against those would confound
   the blend with the head swap, so both arms were re-run at `head=linear` — new
   in this branch, threading `LINEAR_HEAD` into the final fit *and* the
   calibration folds, the same single-width discipline production uses in
   `_train_and_score_xy`.
2. **The threshold is not output-only.** Autopilot's Hard phase picks the
   unlabeled item nearest the decision threshold
   (`al_strategies._hard_pick_by_index`), so the two settings label *different
   items*. Steps therefore stop being comparable across arms, and the paired unit
   has to be the **(arm, category, seed) cell**.

## Design

| | |
|---|---|
| Dataset | `visual_genome_m`, region voting (ground-truth boxes) |
| Primary arm | `dinov3_patch` × `max_patch` — the production region-vote strategy |
| Control arm | `siglip` × `whole_image` — single-vector, isolates the blend from region max-pooling |
| Head | `linear` (production, #2790/#2809) on both arms |
| Steps / seeds | 30 voting steps · 12 seeds |
| Categories | 24 per embedder, scale-band selection shared with the Max-Patch/#2781 studies |
| Cells | 552 per arm × 2 arms = 1104, all COMPLETED; 511 non-empty per arm |
| Fixed | inclusion 0, `sim_fraction` 0.5, `calibrate_count` 2, `calibration_fraction` 0.5, `autopilot_fidelity=True` |

Deviations from the pre-registered plan, all decided before the run:
**12 seeds instead of 8** (the A/B needs cell-level power on two runs, and a
linear-head cell costs 188 s on CPU); **the production head instead of the plan's
"MLP trainer"** line, which predates #2809; **the inclusion sweep reduced to one
k** (that is #2781's question, and `/exp` is a shared 50 GB volume).

Empty cells (rare small-object categories whose split leaves the test set
single-class) are **identical in both arms** — 36 of 552 in each, same indices —
so the pairing is unbiased by construction.

### Which windows a user can actually see

Vote count `n = n_good + n_bad`, the unit the blend's own ramp uses: **2–5**
(pure GMM), **6–20** (ramp), **21+** (pure cross-cal). Cross-cutting that, the
analyzer splits by **scope**: `app_visible` keeps only steps where the app would
have a trained detector on screen, `all_steps` keeps every trainable step.

**The first app-visible step in this run is at 7 votes.** Below the Good/Bad
quorum the app sorts by text/example cosine, so the entire 2–5 "pure GMM" window
— where the blend has *total* authority — contains **zero** steps any user
experiences. The blend's user-facing life is the ramp, exactly where its
authority is being handed over to cross-calibration. (Numerically, the sub-6
window is also where safe-ON wins biggest — cost −0.068, p = 6e−19 at
`all_steps` scope — but that is a harness-only reading, and the report does not
lean on it.)

## Results

Full tables: `agg/ab_window_by_arm.csv` (A/B, both scopes) and
`agg/ab_paired_cells.csv`; figures `figures/ab_{cost,fnr,degenerate}_vs_votes.png`.

**Production arm, app-visible steps** (Δ = ON − OFF, negative = safe ON better):

| window | metric | OFF | ON | Δ | p |
|---|---|---|---|---|---|
| ramp 6–20 (n=226) | cost | 0.5464 | 0.4722 | −0.0742 | 5.6e−23 |
| | FNR | 0.4163 | 0.3456 | −0.0708 | 2.9e−13 |
| | FPR | 0.1301 | 0.1267 | −0.0034 | 8.0e−03 |
| | regret | 0.1896 | 0.1195 | −0.0700 | 4.7e−21 |
| post-ramp 21+ (n=243) | cost | 0.5093 | 0.4896 | −0.0197 | 5.2e−03 |
| | FNR | 0.3995 | 0.3684 | −0.0312 | 1.8e−04 |
| | FPR | 0.1097 | 0.1212 | +0.0115 | 3.9e−03 |
| all steps (n=243) | cost | 0.5329 | 0.4855 | −0.0473 | 3.5e−15 |

**Control arm, app-visible steps:** ramp cost 0.6110 → 0.5646 (−0.0464,
p = 3.6e−13), FNR 0.3907 → 0.3337 (−0.0571, p = 5.0e−10), FPR +0.0107
(p = 6.4e−04), degenerate rate 0.0078 → 0.0000 (p = 0.012). Post-ramp: no
difference (cost −0.0041, p = 0.88) — on the single-vector arm the selection
feedback does not carry, so the durable post-ramp gain is specific to the region
arm.

**A note on the app-visible filter.** Because the arms label different items, the
`app_trained` filter is applied inside each arm's own trajectory, so the two
compared step sets are not literally the same steps — which is the point: this is
a comparison of *policies*, and when the detector appears is itself part of what
the policy does. The unfiltered `all_steps` scope, which applies no such filter,
agrees in sign and size on every headline number (ramp Δcost −0.0754 vs −0.0742,
ΔFNR −0.0747 vs −0.0708), so nothing here rests on the filter.

### Harness fidelity

The safe-ON run also emits, per step, one row per GMM variant re-cutting the
*same* model. The `pooled_cross` variant is production's own rule, and it
reproduces the run's actual blended threshold **exactly across 13,652 steps**
(`max_abs_diff = 0.0`) — so the counterfactual machinery below is measuring
production, not an approximation of it.

## Secondary: the pre-registered variant questions

Within-run contrasts on the production arm, ramp window, n = 267 cells (each
variant re-cuts one model on identical votes, so these are pure threshold-rule
effects, free of selection feedback):

| contrast | Δcost | ΔFPR | ΔFNR | p | reading |
|---|---|---|---|---|---|
| blend − `xcal_only` | **−0.0638** | −0.0044 | −0.0595 | 1.0e−40 | the blend beats the raw conformal cut on both error types |
| pooled fit − image fit (#2797) | **−0.0792** | −0.2075 | +0.1283 | 9.5e−36 | the geometry fix was worth a lot: the image-level fit cut far too low (FPR 0.367 vs 0.157) |
| crossing − midpoint (#2798/#2801) | **+0.0036** | −0.0114 | +0.0150 | 1.9e−08 | the shipped crossing cut is *slightly worse on cost* — see below |
| logit − sigmoid space | +0.0006 | −0.0010 | +0.0017 | 9.5e−04 | no gain; **close the logit idea** |

Two of these settle open questions:

* **The logit-space fit (#2798's unshipped idea) is dead.** It is
  indistinguishable from the sigmoid fit and, to the extent it moves at all,
  moves the wrong way. Recorded, closed, no code change.
* **The equal-density crossing cut (#2801) fails its own pre-registered keep
  rule.** The rule was "keep it unless `pooled_cross` is significantly worse than
  `pooled_mid` on cost"; it is (+0.0036, p = 1.9e−08, and +0.0059 in the sub-6
  window). The effect is second-order — 5 % of the ON-vs-OFF effect — and it is a
  pure FPR/FNR exchange: crossing buys −0.011 FPR for +0.015 FNR. For a
  needle-finding tool the missed positive is the worse error, which points the
  same way as the cost rule. **Recommendation: revert to the midpoint cut**
  (`_weighted_gaussian_crossing` returning `None` already falls through to
  `fit.midpoint()`), as a separate change from the default flip, so the two
  effects stay attributable. This does not weaken the headline: every ON-vs-OFF
  number above was measured with the crossing cut *in place*, i.e. against
  today's production rule.

## Decision

1. **Ship `safe_thresholds = True` as the default** (`vtsearch/settings_models.py`).
   Note the flip only reaches users who never touched the toggle — anyone with a
   persisted `false` keeps it. If the intent is genuinely "on for all users",
   that needs either a migration of persisted `false` values or removing the
   setting; that is a product call, not something this data decides.
2. **Revert the equal-density crossing cut to the midpoint** (#2798/#2801), as a
   separate PR with its own before/after.
3. **Close the logit-space idea** with the number above.
4. `docs/plans/inclusion-calibration-bias.md`'s "Cold-start calibration" item
   should cite this study as evidence the blend **earns its keep** rather than
   being part of the cold-start problem (it removes the degenerate cuts, and it
   beats the raw conformal cut at every vote count where it has authority).

## Reproducing

```bash
# Grid worktree with this branch at /exp/$USER/projects/vts-safe2799
cd /exp/$USER/projects/vts-safe2799/scripts/experiments/calibration
bash launch_safe_ab.sh     # prepare -> ON + OFF arrays -> analyze_safe + analyze_ab
```

Cells run on the **cpu** partition (a linear-head cell is 188 s), so the array is
not capped by the 4-GPU QOS. `VTS_REPO` must point at the worktree:
`common.setup_env()` puts it at the front of `sys.path`, so without it the tasks
import `vtscore` from whichever worktree the default names — a silent
wrong-code-under-test trap.
