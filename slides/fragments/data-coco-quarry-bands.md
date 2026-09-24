<!-- _class: full -->

![bg fit](figs/dataset-coco-quarry-bands.png)

## Same Words,<br>Three Sizes

<!-- The one-sentence version of why this dataset exists: the same class
     names appear in all three bands, so the only thing that changes between
     `bus@small` and `bus@large` is how big the bus is. Class identity and box
     size are never tangled together. -->

<!-- The boxes are drawn to scale inside the frame, which is the only way to
     say what `small` means without asking the room to take it on trust. A
     sub-patch object is under 1/196 of the picture. That is the blue speck. -->

<!-- Every name is a COCO class or a union of them. The four long ones are
     the merges from the slide before — `enclosed road vehicle` is car and
     truck, `single serving drinking vessel` is cup and wine glass — named for
     what COCO's annotators actually put in them. -->

<!-- Why 49 of COCO's 80: the rest cannot fill a band. A class that owns its
     scene gets photographed filling the frame — there are three small-band
     giraffes in the whole of COCO — so it has no small band to be in. Small
     fruit is the other gap: those three cells are not built at all. -->
