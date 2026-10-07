<!-- _class: full -->

![bg fit](figs/test-question.png)

## Final Exam

<!-- build: figs/test-question.build1.png -->

<!-- build: figs/test-question.build2.png -->

<!-- build: figs/test-question.build3.png -->

<!-- The Test button. On the Dashboard, tick a dataset and a trained
     detector and click Test instead of Train: the detector scores every
     photo, the Threshold draws its line on this pile, and the view opens on
     the Test autopilot. These slides are what that autopilot measures, and
     how. Every figure runs the shipped code on a planted pile, so the truth
     is known and the numbers are the ones the app would show. -->

<!-- **a** — A pile the detector has never seen: 2,048 photos, ranked, best on
     the right. The line at the middle radio keeps the top 32. Not to scale:
     the 32 are one and a half percent of the pile. -->

<!-- **b** — The first half of the question: **Right**. Of the 32 the line
     would ship, how many are matches? That is the precision AutoRun would
     deliver, unchecked, on every pile like this one. -->

<!-- **c** — The second half: **Found**. Of all the matches in the pile, how
     many did the line keep? That needs the ones below the line counted too,
     and they are few, spread over 2,016 photos. It is the hard half, and
     most of what follows is about it. -->

<!-- **d** — Two rules make the numbers mean anything. Every pick is drawn at
     random, never chosen by the model: an estimator fed the model's own
     votes broke 83% of its promises (#4256). And a test vote never trains
     the detector, or the test would be marking its own homework. The ranking
     is frozen while the test runs, and Add Corrections is the failed-the-test
     exit: use it, and the result is out of date. -->
