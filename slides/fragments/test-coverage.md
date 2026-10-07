<!-- _class: full -->

![bg fit](figs/test-coverage.png)

## Grading<br>the Grader

<div class="asof">Measured 2026-10-06</div>

<!-- build: figs/test-coverage.build1.png -->

<!-- build: figs/test-coverage.build2.png -->

<!-- Does any of it hold? #4523 replayed the unseen halves of 3,669 saved
     COCO Better sessions, four Tests each, every pick answered from the truth,
     in three worlds: matches thinned to 0.1%, the default 0.44%, and 5%. This
     is the estimator the app ships (#4560): 14,676 Tests. -->

<!-- **a** — Right: the share of Tests whose precision range held the truth.
     92.5–99%, against the 95% it claims. The lowest is β 4 at 0.44%, where
     lines are biggest and sparsest, short by less than its standard error. -->

<!-- **b** — Found: 80–87% in the two sparse worlds, 95–98% at 5%. Where
     matches are rare the deep tail rests on the model, and the range is
     short of its claim. That is why the pane leads Found with words, and
     even the words match the truth's only 31–59% of the time. -->

<!-- **c** — The cost: 55–67 picks to Done on average, nine Tests in ten
     within 60–80. And where matches are rarest, 18–28% of sessions had
     nothing to test: the line kept fewer than five photos. -->
