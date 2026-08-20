![bg right:56% fit](figs/calib-gmm-cut.png)

### Iteration 2 — the idea

## A cut with no labels at all

- Fit a 2-component mixture to the **whole haystack's scores**
- Cut at the <span class="cut">midpoint</span> between the <span class="neg">low</span> and <span class="pos">high</span> modes
- 50 000 scores instead of tens — nothing to starve

<!-- Flip the data source entirely: instead of the tens of labeled votes, use
     the tens of thousands of *unlabeled* scores the detector already assigns
     to the whole haystack. The score histogram is bimodal — a big mound of
     clear rejects, a smaller mound of high scorers — so fit a two-component
     Gaussian mixture (fit_score_gmm) and cut at the midpoint between the two
     component means (calculate_gmm_threshold). The figure is a real vtscore
     fit on synthetic scores, and the colours match: rust for the low
     component, green for the high, blue for the cut.

     Two asides worth making. The midpoint looks naive and survived two
     attempts to replace it with something smarter — the epilogue will show
     #2836 proving it IS the rate-optimal cut under equal variances. And note
     what this estimator costs: nothing here ever looks at a vote, which is
     both its superpower and, next slide, its ceiling. -->
