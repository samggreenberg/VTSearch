<!-- _class: full -->

![bg fit](figs/calib-floor-ask.png)

## Say It<br>Out Loud

<!-- build: figs/calib-floor-ask.build1.png -->

<!-- build: figs/calib-floor-ask.build2.png -->

<!-- build: figs/calib-floor-ask.build3.png -->

<!-- **a** — The problem, first. Under the line, ranked best on the right, is
     what comes back: the **kept** set, the 32 items nobody has voted on above
     the <span class="cut">line</span>. 32 is wherever What to Expect's peak
     fell on this corpus; no cap set it. The question: how many of those 32
     are right? That is the kept set's precision, and the user should be told
     it with an honest range. -->

<!-- **b** — Where the app tells them: the Threshold control, Beta Max's three
     radios, has one line of words under it. Before anyone checks, all it can
     honestly say is how many are kept, and that nobody has checked them. -->

<!-- **c** — The obvious source of an answer: the votes the session already
     holds, at the scores they landed on. A cluster at the top, where
     Autopilot asked for likely matches, and one round the line, where it
     asked the hard ones. The model chose every one of them. -->

<!-- **d** — That is the trouble: a sample the model chose is biased toward
     where the model looked. An estimator that promises a precision from
     those votes alone breaks **83%** of its 50% promises (#4256). So the
     answer has to come from picks the model did not choose, drawn uniformly
     from the kept set, and counted. The next slide turns a count into a
     range. -->
