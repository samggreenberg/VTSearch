### Iteration 2 — measured

## Nearly oracle-close early — and the least robust choice

- Closest of any schedule to the oracle cut (gap 0.023 vs production 0.036)
- But it never learns: pure GMM hits **+0.24** cost when false positives cost 4×
- Use it early. Not alone.

<!-- The measurement (docs/experiments/mixin-schedule/REPORT.md, #2841) is
     more generous to the GMM than expected, and it is worth being fair about
     that before condemning it: of every schedule measured on region voting,
     pure GMM sits closest to the oracle cut — mean gap 0.023 against
     production's 0.036 — and it was among the best at the shipped operating
     point. "Just use the GMM all the time" was not an obviously bad idea.

     Then the two structural failures. It is an inconsistent estimator: its
     high component means "confidently scored", not "true match" — the fitted
     high-component weight was 0.35 against a true prevalence of 0.09 (#2836)
     — and no number of votes ever corrects it, because it never reads one.
     And that bias is asymmetric in exactly the wrong way: reweight the cost
     so false positives matter four times more — which is just an
     Inclusion-averse user, not a hypothetical — and pure GMM blows up by
     +0.24 cost. So the verdict that motivates the next slide: use it early,
     when the labels have nothing, but never alone. -->
