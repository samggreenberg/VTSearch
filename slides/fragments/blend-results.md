### Iteration 3 — Measured

## The Single Biggest Win in the Line

- A/B on the production arm: cost **−0.074**, regret **−0.070** (p ≈ 1e−23)
- FNR −0.071 with FPR flat — not bought with permissiveness
- Keeps paying *after* handoff: a better cut surfaces better items to vote on

<!-- This is the cleanest result in the deck; spend time on why it is clean,
     not just on how big it is. Three reasons.

     First, it is not bought with permissiveness. The recurring failure mode
     in threshold work is a "fix" that games the weighted cost by cutting
     lower and wrecking precision. Here the false-negative rate drops by 0.071
     while the false-positive rate does not move, which means the cut landed
     closer to the model's own best line rather than sliding down it. That is
     what "better calibrated" is supposed to mean, and it is worth showing the
     room that the two rates were checked separately.

     Second, the ranking is untouched — average precision moves by 0.001, not
     significant. The blend changes where the line is drawn, not how the pool
     is ordered. Same detector, better decision.

     Third, and this is the subtle one, so slow down: it keeps paying after
     its authority ends. Past twenty votes the blend IS pure
     cross-calibration — the mixture's weight has gone to zero — and yet the
     arm with the blend switched on stays ahead for the rest of the run. The
     only path for that gain is the second job from the loop slide: the
     threshold decides which items get shown, so a better cut early surfaces
     better items to vote on, which improves every retrain after it. A
     within-step comparison cannot see this at all, which is exactly why the
     study was run as an A/B over whole trajectories rather than as a cheaper
     per-step counterfactual. Turned on for everyone; it also eliminated the
     cold-start "admit nothing" cuts on the control arm. -->
