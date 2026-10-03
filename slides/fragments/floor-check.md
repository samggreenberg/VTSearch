<!-- _class: full -->

![bg fit](figs/calib-floor-check.png)

## Spot Check

<!-- build: figs/calib-floor-check.build1.png -->

<!-- build: figs/calib-floor-check.build2.png -->

<!-- build: figs/calib-floor-check.build3.png -->

<!-- build: figs/calib-floor-check.build4.png -->

<!-- **a** — Same corpus, same candidate, same <span class="cut">line</span>,
     same sentence. -->

<!-- **b** — The check, as the floor shipped it: five items drawn uniformly
     at random from the candidate — from anywhere in it, not its top — and the user votes on
     them. Uniform is the whole trick: a pick the model did not choose is a
     sample the model cannot bias. The app shows them as a check, never as the
     ranking. -->

<!-- **c** — Four of the five came back right. They are votes like any other:
     they go into the labelled set and train the model. -->

<!-- **d** — What five votes can honestly say: a Clopper–Pearson range for how
     much of the candidate is right, each tail at 5%, so its lower end is
     exactly the bound the check is tested at. Four of five reads 34 to 99.
     The floor was 50 and the lower end is 34, so the check falls **short** —
     and it says so naming no cause, because a sparse corpus and a weak model
     fail it alike. Five of five would read 55 to 100: confirmed. -->

<!-- **e** — The floor's three states: *unchecked* before a check,
     *confirmed* when the range cleared the floor, *short* when it did not —
     and the line never fell back. The fallback it replaced returned about
     2,300 items at 2% right. -->

<!-- What ships now differs (#4388, #4413), and the gauge on this figure is
     the floor's. The picks are still uniform, but walked in bands — 8, 8 and
     16 under the 32, five picks each, each band's range at α over the bands
     — going deeper while the estimated F-beta rises and keeping the band
     where it peaks. Two states, unchecked and checked, and it reports
     precision and recall in words. Headless runs export the unchecked
     set. -->
