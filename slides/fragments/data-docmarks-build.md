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

<!-- **a** — Start where the marks already have outlines. SPODS: made-up
     official documents with logos and stamps. Tobacco800: binarised 1980s–90s
     litigation scans, logos boxed. StaVer: German invoices with real rubber
     stamps. Plus four UCSF letterhead marks, each box checked by hand. -->

<!-- **b** — None of the sources tells you which marks are *the same* mark.
     A box is a box. Perceptual hashing proposes the groups, and a person
     settles them with separate questions: is this group one mark, is every
     member really it, are these two classes different? -->

<!-- **c** — Then look for the copies nobody boxed. SIFT compares every
     query with every page of the small tier, and the owner confirms or
     rejects each candidate. A rejection is kept too, as a permanent hard
     negative no rebuild can lose. -->

<!-- **d** — Then bury them. Nearly two hundred thousand real scanned industry
     pages, in three nested tiers. Every page from the three outlined sources
     is in the smallest tier, so growing the haystack adds unrelated real
     scans — and, for the four UCSF marks alone, more copies, which is why
     their numbers are read per tier. -->

<!-- **e** — Last, fix what counts as wrong. A mark is scored against its own
     source's other pages, every one of which has been checked. Two sources
     cut from the same archive never score each other, or a correct find on a
     Tobacco800 logo, on an unboxed UCSF page from that archive, would be
     marked a false positive. Every number sits beside a mark-blind control
     that ranks by source alone. -->

<!-- One thing this slide does not show, and should: a photograph of an actual
     mark. The corpus lives on the cluster, so drawing one is GRID work. -->
