<!-- _class: full -->

![bg fit](figs/calib-fbeta.png)

## Beta Max

<!-- build: figs/calib-fbeta.build1.png -->

<!-- build: figs/calib-fbeta.build2.png -->

<!-- build: figs/calib-fbeta.build3.png -->

<!-- build: figs/calib-fbeta.build4.png -->

<!-- **a** — The same ranking, its cut taken off: F-ing Metrics scored one
     cut, and the question now is which cut. -->

<!-- **b** — The app keeps its line where F-beta peaks, so score every cut:
     one row per β, the peak dotted up to the cut it picks. At β = 1 a miss
     and a false alarm weigh the same, and the peak is the middle cut, Include
     five: one false alarm against one miss, F 0.80. -->

<!-- **c** — β = 1/4: a miss now weighs a sixteenth of a false alarm, and the
     peak moves to the tightest cut, Include two (0.92). Fewer items come back,
     and the ones that do are surer. -->

<!-- **d** — β = 4: a miss weighs sixteen false alarms, and it moves the other
     way, to Include eight (0.97). Three betas, three answers, each the only
     peak on its own row. The data has not changed; the person at the keyboard
     has. Worth a beat: at Include five all three rows read 0.80. Where
     precision equals recall every F-beta agrees, so β only matters away from
     that point. -->

<!-- **e** — So the app asks. The Threshold control: a False Positives …
     False Negatives track and three radios, with no word or number on any of
     them. Left is β = 4: it returns the most and lets more wrong ones in.
     Right is β = 1/4: only the surest come back, and more are missed. The
     middle is 1, and it is every detector's default (#4413, #4448). It sits
     under the rows because β is all it sets: the trade, not a place on the
     ranking. -->

<!-- The catch, and the next slide: the formula counts the Goods returned
     and the possible Goods, and off the votes the app knows neither. It has
     to *expect* them. -->
