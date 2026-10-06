# A document line that follows the balance (#4458)

**Question.** The photo path's line follows the user's balance, F-beta's beta (#4413). The
document line ignored it: it is the Bad ceiling (#4367), or H1's gated 8-inlier line before
the first Bad (#4440), whatever beta is. Can a per-beta rule do better?

**Answer, after two rounds: at the recall end, yes. Elsewhere, no.**
- **Round 2** used the presets 1/4, 1 and 4 (#4472) and class-split cross-validation.
  - **Beta 4 passed**, and both folds chose the same rule.
  - **What ships at beta >= 2** (owner's ruling): the beta-1 set, plus every verified page with
    at least max(10, ⌈0.25 × the Goods' median leave-one-out inliers⌉) inliers, whatever the
    Bad ceiling and the geometry cuts say. That is a superset of beta 1's set, so the slider stays
    monotone.
  - **Through the app**, against the shipped line (2 replicates, tier `m`):
    - returned-set F4 over clicks 0–25: **+0.054 [+0.023, +0.091]**;
    - ranking AP: +0.007 [+0.003, +0.012];
    - pages returned: median 16 → 26.
  - **Beta 1/4 and 1 keep the shipped line.**
- **Round 1** used the presets 0.5, 1 and 2, with floors fitted on tier `s`. No rule passed on
  tier `m`.

## Round 2: presets 1/4, 1, 4 (2026-10-03)

**The shipped line's share of the best cut** at the new presets (tier `m`, 2 replicates; this
describes the shipped line, with nothing fitted):

| click | 1/4 | 1 | 4 |
|---|---:|---:|---:|
| 0 | 0.56 | 0.66 | 0.89 |
| 10 | 0.76 | 0.80 | 0.86 |
| 25 | 0.88 | 0.90 | 0.89 |

**Test design** (owner's choice, pre-registered on #4458). FullMarks tiers are nested, so every
positive is in every tier, and tier `m`'s frames were scored in round 1. So the test is
**class-split cross-validation** on tier `m`'s frames:
- the 36 classes split in two by the parity of the first byte of sha256(class id);
- each beta's rule is chosen on one fold and scored on the other;
- every class is scored with a rule chosen without it.

**The family:** 6,048 rules per beta (`balance_rules_cv.py`). A page is accepted if it is
verified with inliers >= T, where T = max(`t`, the adaptive floor `q` × the Goods' median
leave-one-out inliers, the Bad ceiling + `d` + 1):
- `t` from 4 to 32;
- `q` off, 0.25, 0.5 or 0.75;
- `d` none, or −4 to +8;
- the geometry cuts on or off in each of three states;
- optionally, a tail of unverified pages above a percentile of the Goods' Stage-1 scores.

A fast scorer matched round 1's on every rule both families share (32,349 comparisons,
difference 0).

| beta | verdict | the folds' choices | held-out vs shipped, clicks 0–25 |
|---|---|---|---|
| **4** | **passes** | both: `t=10`, `q=0.25`, no ceiling, no geometry, no tail | +0.060 [+0.020, +0.104]; worst click lower bound −0.011 |
| 1/4 | fails (per-click) | `t=20, q=0.5, no ceiling` / `t=16, q=0.25, d=+4` | +0.068 [+0.027, +0.112], but clicks 3, 15, 25 and 50 have lower bounds of −0.023 to −0.086 |
| 1 | fails | `t=10, q=0.25, d=+4` / `t=16, q=0.25, d=−2` | +0.029 [−0.001, +0.060]; click 1: −0.067 |

The tier-`s` check (each rule refit on all classes) gave beta 4 +0.073 [+0.031, +0.116]. Full
tables: `measurements/round2/cv.md`.

**The monotone union (post hoc, owner's ruling).** The beta-4 rule alone returns **fewer**
pages than the beta-1 line in 234 of 645 frames (36%), and in 64 of 72 at click 0, where its
floor of 10 is stricter than 8. The union with the beta-1 set fixes that by construction. On the
same folds it scores better than the rule alone:
- tier `m`: +0.066 [+0.030, +0.109], no click worse, and click 0 unchanged;
- tier `s`: +0.064 [+0.026, +0.107].

Its gain is the Bad ceiling no longer holding pages back. The pages the rule dropped below
beta 1's line were worth keeping.

**End to end through the app** (`sota_documents.py --beta 4`, 2 replicates, against the
shipped app's runs on the same halves):

| clicks | returned-set F4, beta 4 minus shipped [95%] | AP | pages returned, median |
|---|---|---|---|
| 0–25 | **+0.054 [+0.023, +0.091]** | +0.007 [+0.003, +0.012] | 16 → 26 |
| 0 | +0.000 (the same line) | 0 | 31 → 31 |
| 1 | +0.081 [+0.023, +0.147] | +0.020 | 15 → 30 |
| 10 | +0.065 [+0.027, +0.109] | +0.010 | 20 → 28 |
| 25 | +0.038 [+0.006, +0.077] | +0.002 | 14 → 26 |
| 50 | +0.045 [+0.008, +0.088] | +0.001 | 14 → 26 |

- **The ranking changes a little.** Before the first Bad, a loose fit that clears the floor is no
  longer demoted, so a few pages reorder and a few sessions diverge.
- **The frames predicted the app** in 303 of 322 steps. The rest is those diverged sessions.
  The mean F4 difference is −0.0007.
- **Retrain p90 is 2.2 s** on a V100.

**Not done:** a stricter rule for precision-leaning users. Beta 1/4's rules gain on average but
dip at single clicks, and the two folds disagree.

---

## Round 1: presets 0.5, 1, 2 (2026-10-03)

**Answer: not with the rule family tried.** The rules fitted on tier `s` did
not pass the pre-registered bar on tier `m` at any beta, so every beta keeps
the shipped line. The one gain that held is the opening: at click 0, a higher
inlier floor lifts beta 0.5 by +0.18 [+0.04, +0.31]. But it was not tested on
its own, and tier `m` has now been seen.

![Share by click](figures/share_by_click.png)

### Method

**Frames instead of closed loops.** The structural line does not change which
page the user clicks next, so the sessions are the same at every beta; only
the returned set differs. So `sota_documents.py --frames` saves per-page
state along the shipped sessions: each page's verified inliers and best-fit
geometry, and the votes' fits. Every candidate rule is then scored exactly,
offline. Click 0 gets its own frame (new here): the example sort's shortlist
and each shortlisted page's fit to the crop.

- **Tier `s`:** 33 classes, clicks 0, 1, 2, 3, 5, 10, 15, 25; used for the
  choice.
- **Tier `m`:** 36 classes × 2 replicates (the second with the halves
  swapped), the same clicks plus 50; used for the bar.

**Check:** the shipped rule on the frames reproduces the harness's
returned-set F1 in 259 of 263 tier-`s` frames and 619 of 645 tier-`m` frames.
Of the 30 that differ, 27 are pages at exactly the Bad ceiling + 1 inliers.
The app dropped those through a rounding bug (#4464, since fixed), so the
frames score the rule as intended.

**The rule family** (pre-registered on #4458, with one correction before any
results), per beta:
- `t`, a minimum inlier count in every state: 8, 10, 12, 16, 20, 24 or 32;
- the geometry cuts (#4434) at click 0, on or off;
- the geometry cuts with votes but no Bad yet, on or off;
- the ceiling offset `d` after the first Bad (accept inliers > ceiling + `d`):
  none, −2, 0, +2, +4 or +8;
- the geometry cuts after the first Bad, on or off.

The shipped line is `t=8`, click 0 off, pre-Bad on, `d=0`, post-Bad off.
The plan held beta 1 at the shipped line. The tier-`s` search found the
8-inlier floor too loose at every beta, which would have left the slider
non-monotone, so **the owner amended the plan to fit beta 1 too**. That was
before any tier-`m` frame was scored.

**Score:** the returned set's F-beta as a share of the best F-beta any cut of
the app's order reaches, on the test half.

**Bar (tier `m`, paired by class):** the mean share over clicks 0–25 above
shipped, with the 95% lower bound > 0, and no click (0–50) with a lower bound
below −0.02.

### Tier `s`: the choice

| beta | chosen | tier `s` share, clicks 0–25 | shipped |
|---|---|---:|---:|
| 0.5 | `t=16`, geometry also after the first Bad | 0.843 | 0.753 |
| 1 | `t=16` | 0.840 | 0.780 |
| 2 | `t=12` | 0.851 | 0.820 |

All three keep the shipped switches otherwise. The 8-inlier floor lets in too
many hard negatives early on, even for a recall-leaning user. (Full ranking:
`measurements/search.md`.)

### Tier `m`: the bar

| beta | mean share, clicks 0–25: shipped → chosen | difference [95%] | worst click (lower bound) | verdict |
|---|---|---|---|---|
| 0.5 | 0.776 → 0.819 | +0.043 [−0.009, +0.091] | click 25: −0.072 | fails |
| 1 | 0.800 → 0.833 | +0.033 [−0.000, +0.068] | click 1: −0.039 | fails |
| 2 | 0.836 → 0.843 | +0.008 [−0.012, +0.027] | click 1: −0.039 | fails |

For beta 1, the lower bound is −0.0001 to −0.0005 across bootstrap seeds, and
click 1 fails the per-click limit outright. Per-click tables are in
`measurements/score-tier-m.md`.

- **The opening gain held, at least for beta 0.5.** Click 0's line is the
  example sort's plain 8-inlier gate, and it over-returns.
  - beta 0.5: 0.59 → 0.77 (+0.18 [+0.04, +0.31]);
  - beta 1: +0.13 [−0.02, +0.25];
  - beta 2: +0.07 [−0.00, +0.14].
- **After click 0 the gain is small and mixed.** A fixed floor hurts classes
  whose true matches have few inliers:
  - beta 1: `ucsf/logo_bw_oval_emblem` −0.19, `ucsf/logo_bat_leaf` −0.15 and
    `tobacco800/logo_aeq93a00_1` −0.11, all faint or small marks;
  - beta 0.5 also hurts the two harmful-click classes, `asg54f00` (−0.42) and
    `stampds-00213` (−0.40).

### Next, on fresh data (round 1's view)

Tier `m` has been seen, click 0 included, so a follow-up needs data it has not
scored. That could be FullMarks tier `l`, or a third split of tier `m` that is
disjoint from both replicates' test halves. Candidates:

1. **A floor at click 0 only:** a beta-dependent minimum for the example
   sort's line, the one place the gain held.
2. **A per-detector floor:** relative to the Goods' own leave-one-out inliers,
   so a faint mark's line sits lower than a crisp logo's.

## Reproduce

```bash
source scripts/experiments/pile/pile_env.sh
cd scripts/experiments/fullmarks
python sota_documents.py --tier s --max-v 25 --frames 0,1,2,3,5,10,15,25 \
    --matrix /expscratch/$USER/fullmarks/votes-4162/matrix-s \
    --feature-cache /expscratch/$USER/fullmarks/features --out <dir>/s
python sota_documents.py --tier m --max-v 50 [--swap-halves] --frames 0,1,2,3,5,10,15,25,50 \
    --matrix /expscratch/$USER/fullmarks/votes-4162/matrix-m \
    --feature-cache /expscratch/$USER/fullmarks/features --out <dir>/m-rep1  # and m-rep2
python balance_rules.py search --frames <dir>/s/frames --cuts measurements/cuts.json --out <dir>/search-s
python balance_rules.py score --frames <dir>/m-rep1/frames --frames <dir>/m-rep2/frames \
    --cuts measurements/cuts.json --chosen <dir>/search-s/chosen.json --out <dir>/score-m
# round 2
python balance_rules_cv.py --frames <dir>/m-rep1/frames --frames <dir>/m-rep2/frames \
    --cuts measurements/cuts.json --check-frames <dir>/s/frames --out <dir>/cv-r2
python sota_documents.py --tier m --max-v 50 --beta 4 [--swap-halves] \
    --matrix /expscratch/$USER/fullmarks/votes-4162/matrix-m \
    --feature-cache /expscratch/$USER/fullmarks/features --out <dir>/beta4-rep1  # and rep2
```

The run directory is `/expscratch/sgreenberg/balance-4458/`, which holds the
frames (94 MB, not in the repo). `measurements/` holds the tier-`s` search,
the chosen rules, the tier-`m` per-click scores, and #4434's geometry cuts.
