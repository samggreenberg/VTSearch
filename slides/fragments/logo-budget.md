<!-- _class: full -->

![bg fit](figs/logo-budget.webp)

## Text Eats<br>the Budget

<!-- First thing that went wrong on documents, and the biggest single fix. SIFT
     doesn't keep every keypoint it finds, just the strongest ones, and the
     default budget of 1,024 was set for photos. A scanned page is mostly type,
     every letter has corners, and the type wins. On this letter the crest gets
     7 keypoints, and 4 agree. That's under 8, so it's a miss, on a page that
     obviously has the crest. -->

<!-- Raise the budget to 8,192 and the crest gets 460, with 101 agreeing. Same
     matcher, same page. On the benchmark coming up, that takes AP, a score for
     the whole ranking where 1 means every right page comes first, from 0.16 to
     0.88. SigLIP gets 0.12 on the same pages. -->
