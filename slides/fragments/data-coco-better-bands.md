<!-- _class: full -->

![bg fit](figs/dataset-coco-better-bands.png)

## Same Words,<br>Three Sizes

<!-- The one-sentence version of why this dataset exists: the same class
     names appear in all three bands, so the only thing that changes between
     `bus@small` and `bus@large` is how big the bus is. Class identity and box
     size are never tangled together. -->

<!-- The boxes are drawn to scale inside the frame, which is the only way to
     say what `small` means without asking the room to take it on trust. A
     sub-patch object is under 1/196 of the picture. That is the blue speck. -->

<!-- Every name is a COCO class or a union of them: Car is COCO's car and
     truck, Cup its cup and wine glass, Vase its vase and potted plant, Bag its
     three bags. Capitalised on purpose — each is *a* definition of a cup, this
     dataset's, not *the* definition. Every class has an awkward edge
     somewhere; nobody writes "bird, alive or dead, not cooked". -->

<!-- Why 49 of COCO's 80: the rest cannot fill a band. A class that owns its
     scene gets photographed filling the frame — there are three small-band
     giraffes in the whole of COCO — so it has no small band to be in. Small
     fruit is the other gap: those three cells are not built at all. -->
