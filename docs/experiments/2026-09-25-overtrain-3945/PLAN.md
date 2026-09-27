# #3945: how many votes does it take to over-train the head? (pre-registered plan)

Written 2026-09-25, before any Stage R row beyond a one-cell pilot
(`visual_genome_m × siglip2_l × bag`, used only to size the job) was read.

## What is already known (not re-measured)

- **The shuffled-label control is done** (#3197, `A_control.csv`): every head
  separates coin-flip labels perfectly on its own votes and scores chance
  (AUROC 0.498 ± 0.002) held out. Memorisation of the votes is total and
  invisible from the votes; it only shows held out.
- **In-loop C at 150 clicks** (#3197 Stage B): C = 0.1 ranks better than the
  shipped C = 1 (oracle cost −0.005 ± 0.002, area under 150 clicks), but the
  shipped cut gives it back (regret +0.008). C = 10 is worse on both. Re-tuning
  the cut per C is #4115, **not** this study.
- **Late-session spikes are positive exhaustion, not capacity** (#3547).

What is *not* known is the **shape in the vote count**: at what session depth
C = 1 starts losing to a smaller C, whether that loss grows with depth (the
over-training signature), whether a C that falls with the vote count beats any
fixed C, and whether anything the app can see without ground truth tracks it.

## Design

**Stage R: replay (CPU, off-policy).** The Autopilot vote sets that #3197's
shipped `svm` arm collected (96 environments × 5 seeds = 480 sessions, 150
clicks) are cut at t ∈ {5, 10, 15, 20, 30, 40, 60, 80, 100, 120, 150}. At each
cut the shipped head is refitted at C ∈ {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1,
3, 10, 30} and scored on the harness's held-out half (oracle cost, AP, AUROC),
on its own votes (train AUROC), and on two **vote-only gauges**: the shipped
cross-calibration splits (2 × 30% held out) and 5-fold CV. Script:
`scripts/experiments/overtrain_3945/c_path.py`.

**Stage D: deep sessions (in loop, CPU).** #3197's harness, unchanged, at
**400 clicks**, arms `svm` (shipped) and `svmc01` (C = 0.1), emitting picks.
Stage R is then re-run on the `svm` arm's 400-click vote sets (t up to 400).
The first 150 clicks of each 400-click session must reproduce #3197's
150-click session exactly (same code path, `max_steps` is only a loop bound,
#3547). A mismatch invalidates the pairing and is reported, not patched.

## Primary quantities (defined now)

All are paired within (environment, category, seed); SE clustered on
(environment, category); **resolvable** means |mean| ≥ 2 SE.

1. **Over-training penalty at t**: oracle cost(C = 1) − oracle cost(C\*),
   where C\* is the single best *fixed* C chosen on the other seeds
   (cross-fitted, seeds {0,1,2} ↔ {3,4}). The issue's "how many votes" is the
   smallest t at which the penalty is resolvable, and whether it grows with t.
2. **Best-C curve**: the cross-fitted C\*(t) per t. Over-training in the
   textbook sense predicts C\*(t) falling, or at least a penalty growing, in t.
3. **Schedule arms**, each fitted on seeds {0,1,2}, scored on {3,4} and
   vice versa: fixed C\*; C = c₀·(n/n₀)^(−α) for α ∈ {0.5, 1} (n = votes);
   the same with n = Good votes. Read against fixed C\*, not against C = 1.
4. **Detection**: (a) rank correlation, across t within a session, between
   each gauge at C = 1 and held-out AUROC at C = 1; (b) the held-out oracle
   cost of the C each gauge picks, minus that of C\*(t).
5. **Along-session degradation** (the literal "more votes made it worse"):
   the fraction of sessions whose held-out oracle cost at C = 1 at the last
   cut is worse than its own minimum over earlier cuts by more than 0.02.

## Decision rules (fixed before the full data)

- **"C = 1 over-trains"** requires quantity 1 resolvable and > 0.005 at two
  or more adjacent cuts, and growing (the t = 150 value above the t = 20 value
  by 2 SE). Otherwise the answer is "not at depths up to 150/400".
- **A vote-count schedule is worth an in-loop test** only if it beats fixed C\*
  by ≥ 0.005 oracle cost (resolvable) averaged over cuts. Otherwise a fixed C
  suffices and the question folds into #4115.
- **A gauge "detects over-training"** only if (a) its median within-session
  rank correlation is > 0.5 **and** (b) picking C by it loses < 0.005 against
  C\*(t). A gauge that points the wrong way is reported as such.
- **Nothing ships from this study.** Oracle cost is ranking only; the cut is
  #4115's. Any C or schedule recommendation goes to #4115 as an arm.

## Known limits, stated now

- Replaying the `svm` arm's votes at another C is off-policy: the votes were
  chosen by the C = 1 head. Stage D's `svmc01` arm is the on-policy check for
  one C.
- Whole-image voting and two SigLIP embedders only, as in #3197. Caltech is
  saturated (AUROC ≈ 1) and contributes zeros.
