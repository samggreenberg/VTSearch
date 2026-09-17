<!-- _class: full -->

![bg fit](figs/dataset-card-caltech101.webp)

## Caltech-101

<!-- The control. One object, centred, filling the frame, on a background
     that is usually nothing — and 101 categories that are treated as
     disjoint, so a picture is a leopard or it is a laptop and never both. -->

<!-- Why keep something this easy in the pile: because it is the set where
     region voting has **nothing to point at**. If the object is the whole
     frame, dragging a box around it selects the frame, and a patch-pooling
     detector and a whole-image one are being handed the same thing. Any
     calibration result that holds on Visual Genome *and* here is not a
     result about clutter. -->

<!-- It carries no boxes at all — whole-image labels only — which is the
     other half of the same property, and why it is never a region-voting
     cell. -->

<!-- Two things not on the slide. There is a 102nd directory,
     `BACKGROUND_Google`, which is a pile of assorted web images the original
     paper shipped as a negative class; it is not one of the 101 and the
     count here leaves it out. And two of the categories are `Faces` and
     `Faces_easy` — photographs of identifiable people who sat for a vision
     paper in 2003. They are ordinary members of the dataset and they are
     deliberately not on this wall. -->

<!-- 838 of them are in the pile's `caltech101_m` cell. The `_m` is a
     *dataset size* tier, not a box size — worth saying because the next
     slide's `_small`/`_medium`/`_large` mean something completely
     different. -->
