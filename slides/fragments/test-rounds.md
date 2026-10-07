<!-- _class: full -->

![bg fit](figs/test-rounds.png)

## Pencils<br>Down

<!-- build: figs/test-rounds.build1.png -->

<!-- build: figs/test-rounds.build2.png -->

<!-- build: figs/test-rounds.build3.png -->

<!-- build: figs/test-rounds.build4.png -->

<!-- Where each round of five goes, and when the test stops. Both are pure
     functions of the picks, re-derived on every poll, so the app's view and
     the eval harness cannot disagree about either. -->

<!-- **a** — Both sides of the line in bands: 8, 8, 16 above; 8, 8, 16 and on
     to 992 below. The Threshold is frozen from here to Done, because moving
     the line would move the bands under the picks. -->

<!-- **b** — Check the matches. The first pass gives every band above the line
     one round, from the band holding the line, where the detector is least
     sure and the band is biggest, upward: 2, 4 and 4 of 5. -->

<!-- **c** — Round 4, the allocation rule: which band's next round would
     narrow the F₁ range most, averaged over what that round could find?
     0.078 for the 16, against 0.041 and 0.019 for the eights. It is the
     greedy face of Neyman allocation: picks go where a band is big and
     unsure. 1 of 5, and that is 20 picks, the phase's budget. -->

<!-- **d** — Check the misses. The band just under the line, then one band
     deeper a round, eight rounds to the 40-pick budget. With a class model
     the walk goes to its budget whatever it finds, since its range only holds
     once the model's count has been corrected deep down. -->

<!-- **e** — Done, at 60 picks. The precision range could also have stopped
     the matches early, at 0.20 wide; it is 0.22 here. The budgets are what
     #4540 priced. The band of 992 was never reached: the model's count
     stands in for it. In the app each phase has a light that climbs red,
     yellow, green as it nears its end. -->
