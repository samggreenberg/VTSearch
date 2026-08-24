<!-- _class: full -->

## Region of Interest

![bg fit](figs/ui-region-voting.webp)

<!-- Backup. Region voting: instead of "is this item a match?", the user draws a
     box — *which part of it*. A whole-item Good on a mostly-irrelevant image is
     a mostly-wrong training signal; a box says where the evidence actually
     is. -->

<!-- The consequence worth flagging, because it is what the Gumbel slide is built
     on: under region voting an item's score is the **maximum** over its regions,
     not a single number from a single vector. That changes the shape of the
     score distribution the whole talk is about, which is why nearly every result
     is reported separately for the two modes. -->
