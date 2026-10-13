<!-- _class: full -->

![bg fit](figs/autopilot-smart.png)

## Diminishing<br>Returns

<!-- build: figs/autopilot-smart.build1.png -->

<!-- **a** — Smart is the first light, and it asks: is the detector still
     getting better? Each point is one retrain's detector, scored against all
     the votes we have today, as false alarms plus misses. Fit a line through
     the last ten. Early on it's falling 7% a retrain, so it's yellow: keep
     going. -->

<!-- **b** — Later the same fit is flat, 0.2% a retrain, well inside the 1.5%
     bar, so it's green. A drop also has to be bigger than the points' own
     scatter to count; otherwise a hard category bounces around and the light
     flickers. And it stays red until we have at least five Goods and five
     Bads. -->
