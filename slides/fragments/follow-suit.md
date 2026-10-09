<!-- _class: full -->

![bg fit](figs/follow-suit.png)

## Follow<br>Suit

<div class="asof">Measured 2026-10-08</div>

<!-- build: figs/follow-suit.build1.png -->

<!-- What the work since the first rule bought, scored the way each user
     asked to be scored: one row per preset, each its own F-beta. Grey is
     cross-calibration, the first rule, which reads only the labels and draws
     one line whatever the user wants. Blue is today's app at that preset. On
     all of COCO Better, 144 cells, five seeds, the user's pool thinned to 1%
     positive: each line is the mean of 720 runs. Every line starts at the
     typed query, scored at the line the app draws for it at that preset:
     0.47 at β ¼, 0.37 at β 1, 0.47 at β 4. It holds there through
     Autopilot's opening, until a detector shows at vote 23. The first
     detector then ranks worse than the typed query, so every line dips
     before it climbs. -->

<!-- **a** — Cross-calibration alone, in every row: the same sessions, scored
     at each row's β. Its line runs long and leans to recall: a median of 73
     images back at precision 0.45 and recall 0.70, and 30% of sessions get
     more than 200. -->

<!-- **b** — Today's app at each preset. At β ¼ it is back above the typed
     query by vote 34 and ends at 0.61 against 0.45: +0.17, better in three
     sessions in four. Cross-calibration at β ¼ never clears the typed query
     for good and ends below it (0.45 against 0.47).
     It returns a median of 29 at precision 0.68. At β 1 it is +0.03. At β 4 the
     two lines coincide: cross-calibration's long line was already what a
     recall-minded user wants. So the work since the first rule bought a line
     that follows the user. If asked about the rules in between: the mixture
     cuts all over-return at this prevalence, and mean F1 at β 1 hides the gain.
     The report is docs/experiments/2026-10-05-ladder-fbeta-4519. -->
