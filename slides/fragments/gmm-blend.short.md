![bg right:56% fit](figs/calib-blend-schedule.png)

### Iterations 2–3

## A label-free GMM cut, blended in early

- Mixture fit on the unlabeled haystack — nothing to starve
- Averaged against x-cal: cost **−0.074** in the A/B
- Refined: never hand over completely

<!-- Brief-deck cut of gmm-idea + blend + blend-schedule — three moves on one
     slide, so keep each to a sentence or two. Move one: flip the data source.
     Fit a two-component mixture to the 50 000 unlabeled scores of the whole
     haystack and cut between the modes — nothing to starve, but inconsistent:
     its high component means "confidently scored", not "true match", and no
     vote ever corrects it. Move two: average the rival thresholds with a
     weight that ramps along the vote count, GMM early, labels later. That
     A/B was the single biggest win in the line — cost down 0.074, and FNR
     fell with FPR flat, so it was not bought with permissiveness.

     Move three is the refinement in the figure: sweeping nine handoff
     schedules showed every one that fully hands over to the learned cut gives
     the win back when it does — 300 clicks is only ~13 positives, and the
     learned cut converges in positives, not clicks. So the shipped schedules
     cap the handoff at half weight, forever. -->
