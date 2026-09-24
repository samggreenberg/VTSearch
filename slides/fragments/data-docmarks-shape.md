<!-- _class: full -->

![bg fit](figs/dataset-docmarks-shape.png)

## What's in<br>the Pile

<!-- Two charts, and each answers a question that decides whether a number
     from this corpus means anything. -->

<!-- The top one is the whole design in one picture. It is a log scale — the
     UCSF archive is seventy times the size of the other three sources
     together. Every page of those three is in the smallest tier, so "how
     does this method degrade as the haystack grows" is a question about
     *real unrelated scans*, and not about harder look-alikes: those are all
     in the small tier already. -->

<!-- UCSF is grey because it is the haystack, not because it is empty: four
     of its letterhead marks are on the roster, with 335 copies between them.
     Those four are the one place a bigger tier adds positives as well as
     distractors, so their numbers are compared within a tier, never across. -->

<!-- The bottom one is why a per-class number needs its **n** printed beside
     it. Copies per mark run from 5 to 399. One miss moves recall by a fifth
     at one end of that roster and a four-hundredth at the other, and a
     league table across classes would read that as difficulty. -->

<!-- For scale, on the headline pool at the 50K tier: SIFT checking every
     page scores AP 0.87, SigLIP alone 0.087, and a control that ranks by
     source and ignores the mark 0.026. The gap between the first two is what
     the benchmark exists to close. -->

<!-- The version number is on the slide on purpose. v5.0 is a major version —
     the roster, not just the labels, moved — and a number from one major
     version is re-run, never compared, against another. -->

<!-- The one bias worth stating out loud: the labels were *completed* by one
     matcher. A copy of a mark too faint or too small for SIFT to find is the
     copy most likely still sitting in the negatives — so a SIFT-family method
     is flattered on exactly the pages that are still unlabelled. Say that
     beside any SIFT-versus-other result. -->
