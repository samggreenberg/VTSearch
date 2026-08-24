### Epilogue — The Cut-Rule Line

## Could a Smarter Rule Beat the Midpoint?

- Prior-free crossing: derived, shipped, **−0.0044** — FPR *and* FNR both fell
- A max over regions is an **extreme-value** statistic — so fit that tail instead
- Pre-registered, swept twice, repaired once…

<!-- Change of pace for the epilogue. Every iteration so far changed what data
     the threshold gets to see. This line asked a different question: holding
     the mixture fit fixed, could a smarter *cut rule* than the naive midpoint
     win?

     It opened with a genuine success. The midpoint between two component
     means is the right cut only when the two components are equally likely;
     doing the Bayes-optimal crossing properly, with the fitted mixture
     weights divided back out rather than silently smuggled in, is a
     three-line change. It shipped at −0.0044 in cost with both error rates
     falling, and it captured about sixty percent of the headroom that a
     label-reading oracle said was available on this axis. Small, but clean,
     and it proved the axis had something on it.

     Then the ambitious idea, and this is the one to set up carefully. Under
     region voting an item's score is the maximum over its regions — that was
     the second slide of the talk. The maximum of many draws is not
     Gaussian-shaped; it is an extreme-value statistic, and the classical
     answer for its tail is the Gumbel family. So fitting the tail family the
     data actually implies ought to beat any rule that assumes a Gaussian.

     Stress two things: the premise is principled and testable, and the sweep
     was pre-registered before any result came back. The next slide is what
     the measurement said, and pre-registration is what makes that answer mean
     something. Leave the ellipsis hanging. -->
