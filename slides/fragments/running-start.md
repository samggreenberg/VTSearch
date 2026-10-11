<!-- _class: full -->

![bg fit](figs/running-start.png)

## Running<br>Start

<div class="asof">Measured 2026-10-11</div>

<!-- build: figs/running-start.build1.png -->

<!-- What Head Start bought. Every person and every COCO session was run
     twice, once with the plain SVM and once with the SVM started from the
     Goods' average; everything else the same. Each line is what a Find
     returns at every vote, F-beta at β 1: 562 FHIBE people on face crops at
     1024 px, from one and from four starting photos, and COCO Better's Binary
     opening, 720 runs. -->

<!-- **a** — The plain SVM. On faces Find dips the moment the first detector
     is trained (vote 7 from one photo, 8 from four), below the Goods' average
     it replaced. On COCO there is no dip to fix. -->

<!-- **b** — The head start. On faces the dip is gone, and it stays ahead of
     the plain SVM for the first thirty-odd votes: over votes 1–50 Find's
     F-beta is up 0.058 from one photo and 0.055 from four, and 0.027 and
     0.024 over the whole session. At every preset, both sizes and both
     starts, it is up 0.037 to 0.073 over votes 1–50, and 68–84% of people do
     better. On COCO it runs on top of the plain SVM, a hair ahead: +0.004
     over votes 1–50, +0.002 over the session. The report is
     docs/experiments/2026-10-11-face-handover-4776. -->
