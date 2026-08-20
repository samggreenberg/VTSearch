![bg right:56% fit](figs/calib-error-decomposition.png)

### Epilogue — why it closed

## The anchored fit ate the cut axis

- Tail-α at its own argmin: **+0.0069**, worse at p = 0.001 — line closed
- The label-reading oracle of **every** cut rule: −0.0055 vs production, n.s.
- What remains is **sim → test transfer** — no cut rule can touch it

<!-- REMEASURE-2846.md + REPORT-2881.md. The strongest evidence iteration 4
     was the right investment: once anchoring shipped, even a rule that reads
     the true labels can't significantly beat production. Pre-registration is
     what lets a negative close a line for good. -->
