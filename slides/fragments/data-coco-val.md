<!-- _class: full -->

![bg fit](figs/dataset-card-coco-val.webp)

## COCO val2017

<!-- Twenty times fewer classes than Visual Genome, and the property that
     makes it the one every size result here is built on: it is
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

<!-- Two jobs in this work, and they are different. val2017 is a
     **dataset** — a region-voting cell in the pile, because it has boxes.
     And COCO 2017 as a whole, train and val, is the **source** the size
     benchmark is cut from, for the reason on the next slide. -->

<!-- It is not a gold standard, and the review found that out the hard way:
     two prohibition circles and a school-crossing paddle labelled
     `stop sign`, a box on a hedge labelled `umbrella`. Four adjudicated
     COCO errors across twelve classes. Better than anything else available,
     and still made by people. -->
