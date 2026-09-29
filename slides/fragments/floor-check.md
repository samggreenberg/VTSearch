<!-- _class: full -->

![bg fit](figs/calib-floor-check.png)

## Spot Check

<!-- build: figs/calib-floor-check.build1.png -->

<!-- build: figs/calib-floor-check.build2.png -->

<!-- build: figs/calib-floor-check.build3.png -->

<!-- build: figs/calib-floor-check.build4.png -->

<!-- **a** — Same corpus, same candidate, same <span class="cut">line</span>,
     same sentence. -->

<!-- **b** — The check: five items drawn uniformly at random from the
     candidate — from anywhere in it, not its top — and the user votes on
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

<!-- **e** — Three states, and the line never falls back. *Unchecked* before a
     check; *confirmed* when the range clears the floor; *short* when it does
     not, keeping the 32 it checked and the range that says how close. The
     fallback this replaces returned about 2,300 items at 2% right. Lower
     floors grow the candidate — 64 at 25%, 128 at 10% — and a short round
     halves it and draws five fresh; higher floors cost more picks, 11 at 75%
     and 29 at 90%. -->

<!-- Measured on the rank frames at 0.44% prevalence, floor 50%: five votes,
     confirmed in 28% of sessions, the 32 returned 53% right on average, the
     range holds the truth 99% of the time and is 0.57 wide — that width *is*
     the honest answer. At 10%: twelve votes, 53 items back, 40% right, on the
     floor in 84% of sessions. Nobody votes in a headless run — autorun, the
     CLI — so those export the unchecked candidate. -->
