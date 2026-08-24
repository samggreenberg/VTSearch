<!-- _class: full -->

![bg fit](figs/calib-cost-knob.png)

## Pick Your Poison

<!-- build: figs/calib-cost-knob.build1.png -->

<!-- build: figs/calib-cost-knob.build2.png -->

<!-- build: figs/calib-cost-knob.build3.png -->

<!-- build: figs/calib-cost-knob.build4.png -->

<!-- Change gear here. Everything so far asked one question — where does the
     line go? The room always has a second one: what if I wanted more false
     positives, or fewer? This slide is what that question *means*, and the
     next four are the machinery answering it. -->

<!-- **a** — A ranking, and a realistic one: mostly right, wrong in the middle.
     Nothing here says where to cut. -->

<!-- **b** — Weigh the two mistakes and you get a rule. At equal prices the
     best cut is here, and it is not obviously the cut anyone else would
     want. -->

<!-- **c** — Price a false alarm four times a miss — a user who cannot afford
     noise — and the whole curve tilts. The cheapest cut moves *up* the
     ranking: fewer items come back, and the ones that do are surer. -->

<!-- **d** — Price a miss four times a false alarm and it moves the other way.
     Three defensible answers on one ranking. The data has not changed; the
     person at the keyboard has. -->

<!-- **e** — So we exposed it. Inclusion, minus ten to plus ten: each step up
     doubles the price of a miss, each step down doubles the price of a false
     alarm. One definition, shared by every rule in this section, so a measured
     arm and the shipped path cannot disagree about what a setting costs.
     The rest of the section is what happened when we tried to make the slider
     mean it. -->
