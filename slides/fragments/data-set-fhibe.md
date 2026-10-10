<!-- _class: full -->
<!-- frames: equal -->
<!-- bookmark: Data, Set: FHIBE -->

![bg fit](figs/data-set-fhibe-zoom.webp)

## Data, Set

<!-- build: figs/data-set-fhibe-grid.webp -->

<!-- **FHIBE**: Sony AI's Fair Human-Centric Image Benchmark (Xiang et al.,
     *Nature*, 2025). 10,901 photos of 2,056 people who consented and were
     paid, from 81 countries: one or two people doing something somewhere
     real, every consenting face boxed, with keypoints, a mask and a
     camera-distance label beside it. A registered download, never passed on:
     the pile reads the owner's copy. Ten of its photos are on these two
     pages, inside the twenty its terms allow a publication and the talks
     about it. -->

<!-- **a** — The grid. Six one-person photos, the blue box the annotated face.
     The caption is the face's size band under COCO Better's rule, so Small
     means what it meant there: under 1/196 of the frame. These frames are
     12–32 MP phone photos, so a small face is still about 170 px across, and
     finding it is only hard once the photo is stored small. -->

<!-- **b** — The zoom, grown out of its corner cell, and down the side the
     same person's other photos: a query's positives. Six photos per person
     at the median, so about five positives each, and everyone else is a
     negative. The 623 two-person photos are left out, as Sony recommends: a
     person can appear in them under a second id. Two things are scored, and
     separately: was the face found, and was the person matched. -->

<!-- Why this and not the face set we had: VGGFace2 is celebrities, whom SigLIP
     knows by name, in face-centred crops, scraped without consent. FHIBE is
     none of those. And what it is not for: finding out who anyone is, or
     anything about them. Matching one photo of a person to another, inside
     the set, is the use it was built for. -->
