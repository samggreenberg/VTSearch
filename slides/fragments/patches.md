<!-- _class: full -->

![bg fit](figs/calib-patches.png)

## DINO Might

<!-- build: figs/calib-patches.build1.png -->

<!-- build: figs/calib-patches.build2.png -->

<!-- What sits under a box. Region voting runs on a different model from the
     rest of the talk, and the next slide takes a maximum over what it makes. -->

<!-- **a** — One photograph: Extreme Measures' doll with a book under her
     arm. DINOv3, a vision model from Meta, sends it into content space: the
     same cube as Embed-time Stories, 768 dimensions with three drawn. SigLIP
     sends a whole photo to one point in there. -->

<!-- **b** — DINOv3 cuts it up first: a 224-pixel square becomes a 14 by 14
     grid of 16-pixel patches, 196 of them. -->

<!-- **c** — And sends each patch to its own point, described in context.
     SigLIP says what a photo is about; DINOv3 says what is where. (It makes
     one more point for the whole image, which the drawing leaves out.) The
     drawing's one claim is that patches of the same thing land together, so
     the book's gather. Where they gather means nothing. The green one is the
     patch outlined on the book, followed across. -->
