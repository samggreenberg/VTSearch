<!-- _class: full -->

![bg fit](figs/test-coverage.png)

## Grading<br>the Grader

<div class="asof">Measured 2026-10-06</div>

<!-- build: figs/test-coverage.build1.png -->

<!-- build: figs/test-coverage.build2.png -->

<!-- **a** — So does any of it hold? We took the COCO Better sessions from the
     last section and ran almost 15,000 Tests on photos they never trained on,
     with the real answers filling in every pick. We thinned the matches three
     ways: 0.1%, the usual 0.44%, and 5%. Right held the truth 92.5 to 99% of
     the time, against the 95% it claims. The worst is β 4 at 0.44%, a little
     short. -->

<!-- **b** — Found is weaker. Where matches are rare it holds 80 to 87% of the
     time, short of what it claims, because the deep bands lean on the
     detector's count. At 5% it's 95 to 98%. That's why the pane leads Found
     with words, not a number, and even the words only land on the right one 31
     to 59% of the time. -->

<!-- **c** — And the cost: 55 to 67 picks to get to Done on average, and nine
     Tests in ten finish in 60 to 80. Where matches are rarest, 18 to 28% of
     sessions had nothing to test at all: the line kept fewer than five photos. -->
