<!-- _class: full -->

![bg fit](figs/dataset-coco-quarry-build.png)

## Where Buses<br>Come From

<!-- build: figs/dataset-coco-quarry-build.build1.png -->

<!-- build: figs/dataset-coco-quarry-build.build2.png -->

<!-- build: figs/dataset-coco-quarry-build.build3.png -->

<!-- build: figs/dataset-coco-quarry-build.build4.png -->

<!-- `coco_quarry`: a size benchmark cut out of COCO. One question — does
     the size of the thing change how hard it is to find? — and five
     decisions that make the answer mean something. Every number on the
     slide is read out of the config the pile builds against. -->

<!-- **a** — All of COCO 2017, train and val: 123,287 photographs. The
     property everything rests on is the one from the Venn: COCO answers for
     all eighty of its classes on every image, so an unboxed class is absent
     rather than unmentioned. Crowd regions are dropped — a crowd box is a
     region, not an object anybody would drag. -->

<!-- **b** — Merge where COCO's own line does not hold *at the box*. Region
     voting drags a box, so the test is per box: match COCO against LVIS, which
     re-annotated the same pictures, and ask how often an object of one type
     carries the minority label. Car/truck 10.7%, the bags 10.6%, vase/potted
     plant 6.1% — so those become one class each, as does cup/wine glass.
     Skis/snowboard is 3.4% and stays two. -->

<!-- **c** — Band by the **largest** instance, measured against the model's
     own geometry: `small` is below one patch of the grid, `medium` tops out at
     the smallest region it can pool. So small-versus-large is a statement
     about the method, not about a threshold we picked. The largest instance is
     also the box the simulated user drags — one box, the most obvious object —
     and each positive carries exactly that one region. -->

<!-- **d** — COCO sometimes draws one box round a bunch of bananas or a shelf
     of books, and then the band is the size of the pile. LVIS boxes one object
     at a time, so a fruit or book box is kept only if LVIS finds one thing
     inside it; the rule was set on 176 of the owner's votes. An image that
     fails is simply not a positive — it is never scored as a negative for its
     own class. -->

<!-- **e** — A class is in if it clears 100 positives in all three bands, and
     nothing else is asked. Not scatter, not purity: both track how hard a
     class is to detect, and choosing on either would make the benchmark
     easier and every result optimistic. 49 classes pass. Three cells do not
     exist — small bananas, apples and oranges — because no honest supply
     reaches 100. -->
