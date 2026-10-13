<!-- _class: full -->

![bg fit](figs/calib-floor-ask.png)

## Say It<br>Out Loud

<!-- build: figs/calib-floor-ask.build1.png -->

<!-- build: figs/calib-floor-ask.build2.png -->

<!-- build: figs/calib-floor-ask.build3.png -->

<!-- **a** — New question. The line keeps 32 photos nobody has voted on. Nobody
     picked 32; it's just where the peak fell on this corpus. But how many of
     those 32 are actually right? That's the precision of what we're handing
     back, and the user deserves an honest answer, with a range on it. -->

<!-- **b** — Here's where we'd tell them: the one line of text under the
     Threshold control. Before anyone checks, all it can honestly say is "Top
     32 kept, unchecked." -->

<!-- **c** — The obvious place to look is the votes we already have. Here they
     are at their scores: a cluster at the top, where Autopilot went looking
     for likely matches, and a bunch around the line, where it asked the hard
     ones. The model chose every one of them. -->

<!-- **d** — And that's the trouble. Those votes tell you where the model
     looked, not how right the kept set is. When we tried estimating precision
     from them, it broke its promises far more often than it said it would. So
     instead we pick from the kept set at random, and count. The next slide
     turns that count into a range. -->
