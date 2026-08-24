<!-- _class: full -->

## Region of Interest

![bg fit](figs/ui-region-voting.webp)

<!-- Show, don't explain. Point at the box, say "that is the whole
     interaction", move on — but make the distinction stick, because the
     results later in the deck split along it.

     Binary voting asks "is this item a match?" and takes one answer for the
     whole thing. Region voting asks "which part of it?" and takes a box. The
     second is worth having because a whole-item Good on a mostly-irrelevant
     image is a mostly-wrong training signal; a box says where the evidence
     actually is.

     The consequence to flag now, since it comes back three times: under
     region voting an item's score is the *maximum* over its regions, not a
     single number from a single vector. That changes the shape of the score
     distribution the whole talk is about, which is why nearly every result
     later is reported separately for the two modes — and why one late idea in
     the deck is built specifically on the fact that a maximum is a different
     kind of statistic. -->
