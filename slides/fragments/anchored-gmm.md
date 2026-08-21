![bg right:56% fit](figs/calib-anchored-em.png)

### Iteration 4 — the idea

## Stop averaging cuts. Fuse the evidence.

- Semi-supervised EM: each vote clamped one-hot at mass **κ**
- One fit sees haystack *and* labels; one cut falls out
- The blend's hand-tuned ramp becomes **derived**: γ = κn / (κn + N)

<!-- The blend averages two finished answers; iteration 4 fuses the evidence
     before answering. Mechanically it is semi-supervised EM on the same
     two-component mixture: the unlabeled haystack contributes soft
     responsibilities as usual, while each voted item is clamped one-hot to
     its component — a Good vote belongs to the high component with certainty
     — and carries weight κ instead of 1. One fit sees both data sources; one
     cut falls out (fit_anchored_score_gmm, #2852).

     The conceptual line to deliver: the labels' job is to *identify* the
     components, not to estimate them. The haystack has all the shape
     information; what it cannot know is which mound is "Good". A handful of
     clamped votes pins that down. And the blend's hand-tuned ramp is now
     derived instead of designed: the labels' effective share of the fit is
     γ = κn/(κn + N), which grows automatically with the vote count n — at
     κ = 0.3, γ is around 2.5%.

     One refinement to mention (#2853): production fits per calibration fold
     on held-out anchors and transfers cuts as quantile ranks, so no raw score
     ever crosses model scales. Figure: real vtscore fits on synthetic
     scores. -->
