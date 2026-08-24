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

<!-- One process note worth volunteering: run A swept κ over 1, 3, 10, 30, 100 and
     κ = 1 won — the smallest value on the grid, which should have been read as
     "the optimum is off the edge". Run B, two decades wider, found it interior at
     κ = 0.3 and winning in all six environments, about a day after the narrower
     recommendation had merged. A winner on the edge of its own grid is a prompt
     to extend the grid. -->
