### Epilogue — the cut-rule line

## Could a smarter rule beat the midpoint?

- Prior-free crossing: derived, shipped, **−0.0044** — FPR *and* FNR both fall
- Gumbel/EVT family: a max over regions is an extreme-value statistic — fit one
- Pre-registered, swept twice, repaired once…

<!-- Change of pace for the epilogue: the iterations so far changed what data
     the threshold sees; this line asked whether, holding the mixture fit
     fixed, a smarter *cut rule* than the naive midpoint could win. It opened
     with a genuine success (docs/experiments/gmm-cut/REPORT.md, #2836): the
     prior-free crossing — the Bayes crossing with the mixture weights
     correctly divided out, weights the midpoint was silently smuggling in —
     shipped at −0.0044 cost with FPR and FNR both falling, capturing about
     60% of the headroom that a label-reading oracle showed was available. A
     small win, but a clean one, and it proved the axis had something on it.

     Then the ambitious idea: on region voting a media score is a max over
     region scores, and the max of many draws is an extreme-value statistic —
     so the high tail should be Gumbel-shaped, and fitting the tail family the
     data actually implies should beat any Gaussian rule. Stress that this
     premise is principled and testable, and that the sweep was pre-registered
     before any result came back — because the next slide is what the
     measurement said, and pre-registration is what makes that answer mean
     something. Leave the ellipsis hanging as the transition. -->
