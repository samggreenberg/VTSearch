### Iteration 4 — Measured

## The Production Threshold Today

- Deep regime: **−0.044** paired regret against pure cross-calibration
- Beats the shipped blend on region voting: **−0.026 to −0.032**
- Best single global setting in **6 of 6** environments (κ = 0.3)

<!-- Two runs stand behind this slide: run A on the deep regime, then run B
     sweeping the anchor mass two decades wider across six environments —
     three datasets, four embedding models, both voting modes, 568 cells in
     all.

     The headline: the fused fit cuts paired regret against pure
     cross-calibration by 0.044 in the deep regime, beats the shipped blend on
     region voting by 0.026 to 0.032, and one value of the anchor mass is the
     best global setting in all six environments. This is what ships today,
     unconditionally — the previous blend's off-switch was deleted, and the
     blend survives only as the fallback for the case where folds cannot be
     formed at all.

     Then give the caveats yourself, before anyone asks, because they are on
     record in the report: on binary voting the fused fit is a dead heat with
     the capped blend, and one dataset shows a small loss. The win does not
     transfer everywhere, and saying so is part of why the result is credible.

     The next slide is an honest-lessons slide about how this one nearly
     shipped wrong. -->
