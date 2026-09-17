<!-- _class: full -->

![bg fit](figs/dataset-card-coco-val.webp)

## COCO val2017

<!-- Twenty times fewer classes than Visual Genome, and the property that
     makes it the reference the others are corrected against: it is
     **exhaustive**. If COCO touches an image, it annotates every one of its
     eighty classes on that image. So a missing box is evidence of absence,
     which is exactly what VG's silence is not. -->

<!-- The boxes on these twelve frames are real, and drawn at the same scale
     the slide shows them. The top-left one is a lounge with ten classes in
     it; the densest frame in val2017 carries fourteen classes over
     twenty-four boxes. The skier and the stop sign at the bottom are the
     other end of the range — sparse frames are in there too. -->

<!-- Five thousand images, and 48 of them hold nothing from the eighty
     classes. That is not a defect and not a mystery: those 48 carry no label
     to score, which is the whole reason the pile's `coco_val` cell is 4,952
     medias rather than a round five thousand. -->

<!-- Two jobs in this work, and they are different. It is a **dataset** —
     one of the two region-voting cells in the pile, because it has boxes.
     And it is an **instrument**: half of Visual Genome is COCO images too,
     so on that half we can replace VG's labels with COCO's and know what is
     really in the picture. That is the next-but-one slide. -->

<!-- It is not a gold standard, and the review found that out the hard way:
     two prohibition circles and a school-crossing paddle labelled
     `stop sign`, a box on a hedge labelled `umbrella`. Four adjudicated
     COCO errors across twelve classes. Better than anything else available,
     and still made by people. -->
