<!-- _class: full -->

![bg fit](figs/dataset-vg-scale-build.png)

## Where Buses<br>Come From

<!-- build: figs/dataset-vg-scale-build.build1.png -->

<!-- build: figs/dataset-vg-scale-build.build2.png -->

<!-- build: figs/dataset-vg-scale-build.build3.png -->

<!-- build: figs/dataset-vg-scale-build.build4.png -->

<!-- The shortcutted version. The real order was repair, audit, repair again,
     because each measurement changed what the last one meant — but the steps
     are real and every number on the slide is the dataset's own. -->

<!-- **a** — Start from Visual Genome, for the reason two slides ago: one
     photograph, a dozen named boxes. Free text, one name per object, nothing
     exhaustive. -->

<!-- **b** — Band by how much of the frame the object takes. The edges are
     not a choice — `small` is below **one patch** of the model's own grid,
     `medium` tops out at the smallest region it can pool. That is what makes
     a small-versus-large result a statement about the method rather than
     about a threshold we picked. Size means the *union* box over the class's
     instances -- which is what the harness's simulated Good vote drags. A real
     vote carries one box a person drew, so the union is our modelling choice,
     not a fact about the app. -->

<!-- **c** — Repair. About 51K of these images are COCO images too,
     and there COCO's exhaustive boxes replace VG's outright. The rest went
     in front of a person, in VTSearch, in three passes: the negatives it
     ranked highest, a uniform random stratum, and every positive re-issued
     with its box drawn. If somebody asks why not all of it: reviewing the
     whole of VG is the thing we were trying not to have to do. -->

<!-- **d** — Audit the vocabulary. This is the step nobody plans for. A
     bicycle annotated `bike` was never a `bicycle` positive — and on the
     non-COCO half, where VG's silence is all the evidence there is, it
     became a `bicycle` **negative**. 182 spellings now fold into their
     class; 254 more are withheld from it, because a name can be evidence
     that the class *might* be there without being evidence that it is.
     Numbers if asked: 182 spellings fold, 254 are withheld, and `bike`
     carries 638 of COCO's 3,683 `bicycle` boxes. -->

<!-- **e** — Draw the negatives where absence is provable. Every negative now
     comes from the COCO-scored half, so "this image holds no bus" is a fact
     rather than an inference from VG saying nothing — which we measured
     wrong about **1.4%** of the time. -->

<!-- The honest caveat, if it comes up: reviewing a sub-patch object is at
     the limit of what a person can do. With the box drawn, a human and a
     vision-language model each confirm only about **two thirds** of
     small-band positives. Those are recorded as unconfirmed and the label
     stands — read any small-band number beside that fact. -->
