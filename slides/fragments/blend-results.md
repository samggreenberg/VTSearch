### Iteration 3 — measured

## The single biggest win in the line

- A/B on the production arm: cost **−0.074**, regret −0.070 (p ≈ 1e−23)
- FNR −0.071 with FPR flat — not bought with permissiveness
- Keeps paying *after* handoff: a better cut surfaces better items to vote on

<!-- The A/B (docs/experiments/safe-thresholds/REPORT.md, #2799) is the
     cleanest result in the deck; spend time on why it is clean, not just how
     big. Three reasons. First, it is not bought with permissiveness: the
     recurring failure mode in earlier threshold work was a "fix" that games
     the weighted cost by cutting lower and wrecking precision — here FNR
     drops 0.071 while FPR is flat, meaning the cut lands closer to the
     model's own oracle, which is what "better calibrated" is supposed to
     mean. Second, ranking is untouched (average precision moves +0.001,
     n.s.): the blend changes where the line is drawn, not how the pool is
     ordered.

     Third — the subtle one, worth slowing down for — it keeps paying after
     its authority ends. Past 20 votes the blend IS pure cross-calibration,
     yet the ON arm stays ahead. The only path for that gain is selection
     feedback: the threshold drives Autopilot's Hard pick, so a better cut
     surfaces better items to vote on, and the whole trajectory improves. A
     within-step counterfactual cannot see this at all, which is exactly why
     the study was run as an A/B over whole trajectories. Turned on for
     everyone; it also eliminated the cold-start "admit nothing" cuts on the
     control arm. -->
