<!-- _class: full -->

![bg fit](figs/sota-documents.png)

## Looks Good<br>on Paper

<div class="asof">Measured 2026-10-03 and 10-04</div>

<!-- build: figs/sota-documents.build1.png -->

<!-- The photo results' two panels, for documents. FullMarks v5.0, about
     50,000 pages a class, all 36 marks, 50 clicks from the query crop, two
     replicates. Each radio is its own run of the app's path: the review of
     2026-10-03 at β 1 (#4457), and the runs that shipped each end. Scored on
     the half of the pages the user never clicked. -->

<!-- **a** — Left: by ten clicks every radio is past 0.75 at its own β, and
     little moves after 25; β 1 ends at F1 0.87. The ranking under the line:
     AP 0.79 from the crop alone, 0.92 at 25 clicks, 0.94 at 50. -->

<!-- **b** — Right: each radio's set as precision against recall, followed
     through the session; each circle holds its click count, 0, 10 and 50.
     The middle and recall end leave one circle, the crop alone; the
     precision end's rule trims even that. Each climbs most by 10 and then drifts. At 50: 13 pages at
     precision 0.98 and recall 0.79; 14 at 0.93 and 0.87; 25 at 0.72 and
     0.95. The same trade as on photos, higher up. The β 1/4 run is in sample,
     its rule refit on these classes; its cross-validated gain is on the last
     slide. -->

<!-- Weak spots: 35 to 42 positives beyond the 2,000-page shortlist, mostly in
     two classes; the Bad ceiling's recall cost in one Tobacco800 class; and
     the large harms all come from correct Good votes (#4170). A Good click
     cost 5.5 s on the review's V100; #4469 has since brought it to about
     1.5 s. -->
