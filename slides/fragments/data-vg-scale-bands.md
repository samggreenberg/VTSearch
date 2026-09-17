<!-- _class: full -->

![bg fit](figs/dataset-vg-scale-bands.png)

## Same Words, Three Sizes

<!-- The one-sentence version of why this dataset exists: the same
     twenty-five class names appear in all three bands, so the only thing
     that changes between `bus@small` and `bus@large` is how big the bus is.
     The slide before this is the version that could not do that. -->

<!-- The boxes are drawn to scale inside the frame, which is the only way to
     say what `small` means without asking the room to take it on trust. A
     sub-patch object is under 1/196 of the picture. That is the blue speck. -->

<!-- Where the class list came from: every one is also a COCO class, which is
     what made the repair pass affordable at all. It was twelve until #3588
     and is twenty-five now — the thirteen added were chosen for their
     *scenes*, deliberately putting `truck` and `car` beside `bus` and
     `fork` and `spoon` beside `knife`, because the original twelve sampled
     context by accident. -->

<!-- The easy end could not be widened, and the reason is a nice one: a class
     that owns its scene gets photographed filling the frame, so it has no
     small band to be in. -->
