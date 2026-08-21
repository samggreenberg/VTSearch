![bg right:56% fit](figs/calib-blend-schedule.png)

### Iteration 3½ — measured again

## Never hand over completely

- Nine schedules swept: the 20-vote handoff was too fast
- Winners cap x-cal at **half weight, forever**: region −0.058, binary −0.019
- 300 clicks ≈ **13 positives** — x-cal converges in positives, not clicks

<!-- Iteration 3½ went back and swept the three baked-in choices
     (docs/experiments/mixin-schedule/REPORT.md, #2841): nine schedules, and
     re-run at ten times the original horizon to make sure the answer was a
     finding and not an artefact. The surprise is the headline: every single
     schedule that fully hands over to the learned cut gives its advantage
     back at the moment it does, monotone in release point. The winners never
     hand over — they cap cross-calibration at half weight, forever.

     Give the intuition before the numbers: this is NOT because the GMM is
     better in the limit — it is inconsistent and cannot be. It is a horizon
     effect: 300 clicks buys a median of about 13 positives, and the learned
     cut converges in positives, not clicks, so "the limit" where x-cal wins
     outright is simply never reached inside a real session. The first-pass
     answer (a plain slower ramp) was wrong for exactly this reason: it
     reached pure x-cal at 40 labels and decayed to nothing past it.

     What shipped: slow_cap50 for region voting (−0.058), cap50 for binary
     (−0.019) — the two voting modes genuinely want different curves, a mode
     split that returns in iteration 4's caveats. -->
