<!-- _class: full -->

![bg fit](figs/calib-patches.png)

## Patch Work

<!-- build: figs/calib-patches.build1.png -->

<!-- build: figs/calib-patches.build2.png -->

<!-- build: figs/calib-patches.build3.png -->

<!-- What sits under a box. Region voting runs on a different model from the
     rest of the talk, and the next slide takes a maximum over what it makes. -->

<!-- **a** — One photograph: Extreme Measures' doll with a book under her
     arm. -->

<!-- **b** — DINOv3, a vision model from Meta, does not look at the photo as
     one thing. It cuts a 224-pixel square into a 14 by 14 grid of 16-pixel
     patches: 196 of them. -->

<!-- **c** — And it describes each patch on its own, in context: 768 numbers a
     patch, plus one more vector for the whole image. SigLIP, which the talk
     has used so far, makes only that last kind: one vector a photo, saying
     what the photo is about. DINOv3's say what is where. -->

<!-- **d** — What region voting does with them. A Good box trains on the
     patches inside it, so the vote says *this part*. And a photo is scored by
     its best row, the maximum over all 197. That maximum is the next slide. -->
