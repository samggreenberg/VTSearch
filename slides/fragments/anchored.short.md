![bg right:56% fit](figs/calib-anchored-em.png)

### Iteration 4 — production today

## Partially-labeled GMM

- Votes clamped one-hot **inside** the fit — no late averaging
- **−0.044** regret vs pure x-cal; best global setting in 6/6 environments

<!-- Brief-deck cut of anchored-gmm + anchored-results. The idea in one line:
     stop averaging two finished answers and fuse the evidence instead — one
     semi-supervised EM fit where the unlabeled haystack contributes soft
     responsibilities and each vote is clamped one-hot to its component at a
     small mass κ. The labels' job is to identify the components, not to
     estimate them; the haystack carries the shape, the votes pin down which
     mound is "Good", and the blend's hand-tuned ramp becomes a derived
     quantity that grows with the vote count on its own.

     The measurement, from 568 Grid cells across six environments: regret down
     0.044 against pure cross-calibration in the deep regime, and κ = 0.3 the
     best single global setting in all six. This is the production threshold
     today, unconditionally. If time allows, add the two honest caveats from
     the report: binary voting is a dead heat with the old blend, and the
     winning κ came from run B after run A had already shipped a value from
     the edge of its grid — the fuller decks make a whole slide of that
     lesson. -->
