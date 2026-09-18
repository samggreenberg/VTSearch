<!-- _class: full -->

![bg fit](figs/dataset-docmarks-build.png)

## Needles, Then<br>Haystack

<!-- build: figs/dataset-docmarks-build.build1.png -->

<!-- build: figs/dataset-docmarks-build.build2.png -->

<!-- build: figs/dataset-docmarks-build.build3.png -->

<!-- build: figs/dataset-docmarks-build.build4.png -->

<!-- A different task from everything else in the talk. Not "find me the
     books" — *find me this exact stamp*, from one crop of it, in a pile of
     scanned pages. Same shape: a concept the user can point at and not write
     down, and a haystack too big to read. -->

<!-- **a** — Start where the marks already have outlines. Three sources ship
     them: SPODS, pseudo-official documents made with logos, stamps and
     signatures on them; Tobacco800, binarised scans from 1980s and 90s tobacco
     litigation with the logos boxed; and StaVer, German scanned invoices
     carrying real rubber stamps. Under three thousand pages between them. -->

<!-- **b** — None of the three tells you which marks are *the same* mark.
     A box is a box. So perceptual hashing over every boxed mark proposes the
     groups — and that is all it does, propose. -->

<!-- **c** — Then a person settles it, and this is where most of the work went.
     Four questions, asked separately: is this group one mark? is every member
     really that mark? are these two classes different? is it a mark at all, or
     a plain shape? All 276 pairs of the roster were ruled on, and the rulings
     survive a rebuild — it is the *rulings* that are the dataset, not the
     clustering. -->

<!-- **d** — Then bury them. Nearly two hundred thousand real scanned industry
     pages go in as distractors, in three nested tiers. Every page carrying a
     mark is in the smallest tier, so 5K to 200K adds only distractors and the
     hard same-source near-misses stay constant. -->

<!-- **e** — Last, fix what counts as wrong. A mark is scored against its
     own source's other pages — all of which have been checked — and two
     sources cut from the same archive never score each other. Without that
     last rule a correct find on a Tobacco800 logo, found on an unlabelled
     UCSF page from the same archive, is counted as a false positive. -->

<!-- One thing this slide does not show, and should: a photograph of an actual
     mark. The corpus lives on the cluster, so a container cannot open it, and
     the panel that used to be here was built against an old 41-class version.
     Reshooting it is booked. -->
