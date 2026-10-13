<!-- _class: full -->

![bg fit](figs/logo-stages.webp)

## Short List,<br>Long Look

<!-- build: figs/logo-stages.build1.webp -->

<!-- build: figs/logo-stages.build2.webp -->

<!-- **a** — But checking is slow: about a tenth of a second of computer time
     per page. At fifty thousand pages, that's way too slow for someone who
     wants an answer after every vote. So we don't check the whole pile.
     Instead, every page gets boiled down to one vector, the same kind of thing
     SigLIP makes, so the rest of the app works on it unchanged. -->

<!-- **b** — Stage 1 compares the crop's vector with every page's, all at once,
     which is fast, and keeps a shortlist off the top. On documents it actually
     compares pieces of pages; more on that in a second. -->

<!-- **c** — Stage 2 does the slow check from before, only on the shortlist,
     and ranks those pages by how many matches agree. These four are the real
     top for this crop, out of about thirteen hundred letters. The app
     shortlists two thousand pages, and a page that misses the shortlist never
     gets checked, so Stage 1 is the ceiling. -->
