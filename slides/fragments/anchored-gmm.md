![bg right:54% fit](figs/calib-anchored-em.png)

### Iteration 4 — the idea

## Stop averaging cuts. Fuse the evidence.

- Semi-supervised EM: each vote clamped one-hot at mass **κ**
- One fit sees haystack *and* labels; one cut falls out
- The blend's hand-tuned ramp becomes **derived**: γ = κn / (κn + N)

<!-- fit_anchored_score_gmm (#2852), then fold-anchored (#2853): fit per
     calibration fold on held-out anchors, transfer cuts as quantile ranks so
     no raw score crosses model scales. "The labels' job is to identify the
     components, not to estimate them" — at κ=0.3, γ ≈ 2.5%. Figure: real
     vtscore fits on synthetic scores. -->
