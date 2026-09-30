<!-- _class: full -->

![bg fit](figs/logo-votes.webp)

## Show, Don't<br>Tell

<!-- build: figs/logo-votes.build1.webp -->

<!-- build: figs/logo-votes.build2.webp -->

<!-- Last, what the votes do. The Coke slide said the edge of a concept
     belongs to whoever asked. This is that, for marks. -->

<!-- **a** — FullMarks' crest class holds two drawings. One is the globe
     crest the query was cut from; the other is the *PM* monogram crest on
     Philip Morris U.S.A. letters. The owner ruled them one mark. SIFT cannot
     know that, because they are different ink. The query gets 6 agreeing
     keypoints on this U.S.A. letter: under the gate of 8, a miss. -->

<!-- **b** — A user votes Good on another U.S.A. letter and draws a box round
     its crest. The keypoints inside the box become a second template. -->

<!-- **c** — Every page is now checked against both templates and scores its
     best, 32 here. One vote taught it the other drawing, and the other
     U.S.A. letters come with it. -->

<!-- On FullMarks (v5.0, 5,000 pages, the same 10 votes for every rule),
     scoring by the best template ranks the pages still unlabelled at AP
     0.87, against 0.74 for the query alone. The app builds these templates,
     but it checks only the page vector's top 50, and from three votes it
     re-scores them with a small learned classifier. Once that classifier has
     seen a Bad, it does worse than the plain count (#4169). The app's path
     scores 0.17 in the same test. Both are open. -->
