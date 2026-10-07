<!-- _class: full -->

![bg fit](figs/test-draws.png)

## Luck of<br>the Draw

<!-- build: figs/test-draws.build1.png -->

<!-- build: figs/test-draws.build2.png -->

<!-- How twelve band posteriors become one range each for Right, Found and
     F₁. Precision and recall are ratios of sums across bands, and no formula
     gives a ratio's range cleanly, so the app simulates: Monte Carlo. -->

<!-- **a** — One draw takes every band's count of matches at once: its picks,
     plus its unseen photos at a share drawn from its posterior. The three
     bands above the line get a column each and the nine below are summed.
     The deepest band, which the walk never reached, adds the model's own
     count, 3.0, to every draw. 4,000 draws at a fixed seed, so the same
     labels always give the same ranges. -->

<!-- **b** — Each draw is a whole possible pile, so it has a Right, a Found
     and an F₁ of its own. Draw 1 keeps 17 matches and leaves 12 below: Right
     is 17 of 32, 53%; Found is 17 of 29, 59%; F₁ 0.56. -->

<!-- **c** — All 4,000, and each range is its column's central 95%: Right
     44–66%, Found 26–70%, F₁ 0.35–0.67. The truth planted here is 56%, 49%
     and 0.52, inside all three. Every number comes from the same draws, so
     the three can never contradict each other, and Lean the Threshold reads
     those draws again at other lines. -->
