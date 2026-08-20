![bg right:56% fit](figs/regret-decomposition.png)

### Finding 3

## The error is calibration, not the cut rule

- Regret decomposes almost entirely into **calibration shift**
- `rule_inefficiency` is **negative on every arm** — the rules are already fine
- So: **acquisition is the frontier**, not thresholding

<!-- Independently reproduces the #2836 result. Worth saying out loud that this
     is the second time this fell out of a different experiment. -->
