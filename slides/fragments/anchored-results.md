### Iteration 4 — measured

## The production threshold today

- Deep regime: **−0.044** paired regret vs pure cross-calibration
- Beats the shipped blend on region voting: −0.026 to −0.032
- Best single global setting in **6 of 6** environments (κ = 0.3)

<!-- Two runs stand behind this slide
     (docs/experiments/population-anchored-calibration/REPORT.md, #2852 and
     #2864): run A on the deep regime, then run B sweeping the anchor mass two
     decades wider across six environments — three datasets, four embedders,
     both voting modes, 568 cells in all. Headline: the fused fit cuts paired
     regret against pure cross-calibration by 0.044 in the deep regime, beats
     the shipped blend on region voting by 0.026 to 0.032, and κ = 0.3 is the
     best single global setting in all six environments. This is what ships
     today, unconditionally — #2863 deleted the safe-thresholds off-switch,
     and the blend survives only as the no-folds fallback.

     Then give the caveats yourself, before anyone asks, because they are on
     record in the report: on binary voting fusion is a dead heat with the
     cap50 blend, and caltech101 shows a small loss. The win does not transfer
     everywhere, and saying so is part of why the result is credible. The next
     slide is an honest-lessons slide about how this one nearly shipped
     wrong. -->
