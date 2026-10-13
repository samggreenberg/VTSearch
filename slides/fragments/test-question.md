<!-- _class: full -->

![bg fit](figs/test-question.png)

## Final Exam

<!-- build: figs/test-question.build1.png -->

<!-- build: figs/test-question.build2.png -->

<!-- build: figs/test-question.build3.png -->

<!-- **a** — So we've trained a detector. Is it any good? That's the Test
     button: pick a dataset it's never seen, and hit Test instead of Train.
     Here it scored 2,048 new photos, and the line kept the top 32. Not to
     scale: 32 is about one and a half percent of the pile. This pile's made
     up, with the answers planted, so we can check the test itself. -->

<!-- **b** — The test asks two things. First, Right: of the 32 it kept, how
     many are matches? That's what AutoFind would be sending you, with nobody
     checking. -->

<!-- **c** — Second, Found: of all the matches in the pile, how many did the
     line keep? Now we have to count the ones it left behind too, and they're a
     few needles spread over 2,016 photos. That's the hard half, and it's most
     of what's coming. -->

<!-- **d** — And two rules. Every photo we get asked about is picked at random,
     five at a time, never by the detector, because like we saw earlier, the
     votes it picks tell you where it looked, not how right it is. And a test
     vote never trains the detector, or it'd be grading its own homework. -->
