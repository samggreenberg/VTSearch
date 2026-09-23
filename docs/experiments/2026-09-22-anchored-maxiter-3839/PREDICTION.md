# What the `cap2000` trajectory A/B is expected to show (issue #3839)

Written after the gate and before the A/B grid returned, so the direction below
was not read off the result.

- The gate moves the admitted set on **9.6%** of 2,258 fold cases; where it
  moves, `cap2000` admits **fewer** medias in 184 cases and more in 33 (median
  signed move −0.60% of the haystack). A capped fit is stopped mid-migration,
  with the minority component not yet out in the high mode, so its midpoint sits
  low and the cut over-admits - the direction #3825 measured for 1e-3.
- **Expected:** ΔFPR < 0 and ΔFNR ≥ 0, i.e. the arm raises the cut where it acts.
  No prediction on the sign of Δcost, which is those two netted under equal
  inclusion-0 weights.
- **Expected σ of the per-cell Δcost below the ~0.04 of a whole-population arm.**
  The arm is inert on any trajectory in which no fold reaches 200 iterations
  (seeds 0-8 on `caltech101_m` should pair to Δ = 0 exactly), so the cells that
  diverge are fewer and diverge later - #3840's "σ is set by when trajectories
  part".
- **Ship bar (#3825's):** the gate says the change is toward the converged fit on
  every capped case; the A/B must not show a cost regression resolvable at 2 SE.
