### Iteration 4 — Measured

## The Production Threshold Today

- Deep regime: **−0.044** paired regret against pure cross-calibration
- Beats the shipped blend on region voting: **−0.026 to −0.032**
- Best single global setting in **6 of 6** environments (κ = 0.3)

<!-- Backup for iteration 4. Two runs stand behind it: run A on the deep regime,
     then run B sweeping the anchor mass two decades wider across six
     environments — three datasets, four embedding models, both voting modes, 568
     cells in all. -->

<!-- This is what ships today, unconditionally: the previous blend's off-switch
     was deleted, and the blend survives only as the fallback for the case where
     folds cannot be formed at all. -->

<!-- Give the caveats yourself, before anyone asks, because they are on record in
     the report: on binary voting the fused fit is a dead heat with the capped
     blend, and one dataset shows a small loss. The win does not transfer
     everywhere, and saying so is part of why the result is credible. -->

<!-- If someone asks where κ = 0.3 comes from: a first sweep over 1 to 100 picked
     the smallest value it measured, and a wider one found the real optimum an
     order of magnitude below that, winning in all six environments. It is a
     measured setting, not a tuned one — and the fact that κ* keeps falling as
     votes accumulate is why the roadmap wants a schedule rather than a
     constant. -->
