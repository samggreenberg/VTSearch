<!-- _class: full -->

![bg fit](figs/logo-stages.webp)

## Short List,<br>Long Look

<!-- build: figs/logo-stages.build1.webp -->

<!-- build: figs/logo-stages.build2.webp -->

<!-- Checking is not free. Matching plus RANSAC costs about a tenth of a
     CPU-second a page. At 50,000 pages that is 100 seconds a query on 40
     cores, and a user voting in a loop needs an answer every click. -->

<!-- **a** — So the pile is not checked. Each page gets one fixed-length
     vector instead, the same kind of object SigLIP makes, so everything the
     app does with vectors (the SVM, the atlas, sorting) works on it
     unchanged. The vector is VLAD. A fixed vocabulary of 64 "visual words",
     typical keypoints found once by clustering a million of them, files each
     keypoint under its nearest word. The vector records, word by word, how
     the page's keypoints differ from it: 64 × 128 = 8,192 numbers. -->

<!-- **b** — Stage 1 compares the query's vector with every page's, all in
     one matrix multiply, and keeps the top few. (On documents it now
     compares tiles instead of whole pages — next slide.) -->

<!-- **c** — Stage 2 checks only those, the slow way, and ranks them by how
     many agree. These four are the real top of Tobacco800's 1,290 pages for
     this crop, after the query's own page. The app keeps the top 2,000 on
     documents (1,000 without a GPU), and 50 on photos. A page
     Stage 1 does not shortlist is never checked, so Stage 1 is the
     ceiling. -->
