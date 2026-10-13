<!-- _class: full -->

![bg fit](figs/calib-inside-job.png)

## Inside<br>Job

<!-- build: figs/calib-inside-job.build1.png -->

<!-- build: figs/calib-inside-job.build2.png -->

<!-- **a** — Here's that rule written out. The fit only reads the votes the
     folds hold out, so the rule depends only on those: Goods up, Bads across,
     per fold. The table says how each color sorts and where it cuts. No Goods,
     white: the typed query. One or two Goods per fold, orange: the trained
     head sorts, but it's too thin to fit, so we count what stands out. -->

<!-- **b** — Three Goods per fold, six between the two, and we're in purple:
     same head, now cut at the F-beta peak. It's one flat edge; the Bads don't
     move it. Which side of it your labels land on is the next slide. -->

<!-- **c** — One more branch, in dark purple. The fit normally assumes one bell
     curve for the non-matches, and if their real scores have a long upper
     tail, it reads that tail as matches and cuts too deep. So with fifty Bads
     per fold that look like a random sample, it uses the Bads' own shape
     instead. Sessions never get here, since Autopilot asks about the top; a
     big imported set, fully labeled, can. -->
