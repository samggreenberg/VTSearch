![bg right:56% fit](figs/calib-gmm-cut.png)

### Iteration 2 — the idea

## A cut with no labels at all

- Fit a 2-component mixture to the **whole haystack's scores**
- Cut at the <span class="cut">midpoint</span> between the <span class="neg">low</span> and <span class="pos">high</span> modes
- 50 000 scores instead of tens — nothing to starve

<!-- fit_score_gmm / calculate_gmm_threshold. The figure is the real vtscore
     fit on synthetic scores. The midpoint survived two attempts to replace it
     with a crossing rule — #2836 later proved it IS the rate-optimal cut under
     equal variances. -->
