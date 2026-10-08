<!-- _class: full -->

![bg fit](figs/logo-budget.webp)

## Text Eats<br>the Budget

<!-- The first thing that went wrong on documents, and the biggest single
     fix. -->

<!-- SIFT does not keep every keypoint it finds; it keeps the strongest few.
     The default, 1,024, was set for photographs. A scanned page is mostly
     type, every letter has corners, and the type wins. On this letter the
     typed text and the scanner noise along the bottom edge take most of the
     1,024. The crest gets 7, and 4 of them agree with the query. That is
     under the gate, so this letter is a miss, though it plainly carries the
     crest. -->

<!-- At 8,192 the crest gets 460, and 101 agree. Nothing else changed: same
     matcher, same page. -->

<!-- On FullMarks, the benchmark after these slides (v3.1, the 5,000-page
     tier, every page checked), this is AP
     0.16 at 1,024 against 0.88 at 8,192. SigLIP scores 0.12 on the same
     pages. It ships as its own embedder, *SIFT/VLAD (document scans)*, so a
     dataset records that it chose it. -->

<!-- The price is storage: about 0.8 MB of keypoints a page, or 167 GB for
     FullMarks' 200,000-page tier. The budget is also tied to the resolution
     SIFT runs at. At 8,192, two megapixels is the best, and changing one
     without the other gives most of the gain back. -->
