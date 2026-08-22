![bg right:56% fit](figs/calib-anchored-em.png)

### Iteration 4 — the idea

## Stop averaging cuts. Fuse the evidence.

- One fit reads the corpus *and* the votes — each vote pinned at mass **κ**
- The labels' job: **identify** the components, not estimate them
- The hand-tuned ramp becomes **derived**: γ = κ*n* / (κ*n* + *N*)

<!-- The blend averages two finished answers. Iteration 4 fuses the evidence
     before answering, which is the difference between asking two people and
     getting one opinion out of everything they both saw.

     Mechanically it is the same two-component mixture as iteration 2, fitted
     the same way, with one addition: the unlabeled corpus contributes softly,
     as before, while each voted item is pinned to the component its vote
     names — a Good vote belongs to the high component with certainty — and
     carries a weight κ instead of 1. One fit sees both data sources; one cut
     falls out of it. The figure shows what that buys: the unanchored fit,
     dashed, puts its cut inside the negative bulk, because with no labels
     nothing tells it which mound is which. A handful of pinned votes moves
     the line to where it belongs.

     The conceptual line to deliver, and it is the one worth remembering: the
     labels' job is to identify the components, not to estimate them. The
     corpus has all the shape information — fifty thousand scores describe two
     mounds perfectly well. What it cannot know is which mound is "Good". Five
     votes settle that. This is why the estimator does not starve the way
     iteration 1 does: identification is a much cheaper question than
     estimation.

     And the previous iteration's hand-tuned ramp is now derived rather than
     designed. The labels' effective share of the fit is their total mass over
     the total mass — κ times the vote count, over that plus the corpus size —
     which grows with the vote count on its own, with nobody choosing
     endpoints. At the shipped setting it is around two and a half percent.

     One refinement to mention if asked: in production the fit is done per
     calibration fold on held-out anchors, and cuts transfer as quantile
     ranks, so no raw score ever crosses between model scales. -->
