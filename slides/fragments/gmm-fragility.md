### Iteration 2 — measured

## Nearly oracle-close early — and the least robust choice

- Closest of any schedule to the oracle cut (gap 0.023 vs production 0.036)
- But it never learns: pure GMM hits **+0.24** cost when false positives cost 4×
- Use it early. Not alone.

<!-- docs/experiments/mixin-schedule/REPORT.md (#2841). The GMM is an
     inconsistent estimator — its high component is "confidently scored", not
     "true match" (fitted w_hi 0.35 vs true prevalence 0.09, #2836). This slide
     motivates the blend. -->
