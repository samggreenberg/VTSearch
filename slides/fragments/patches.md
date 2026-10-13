<!-- _class: full -->

![bg fit](figs/calib-patches.png)

## DINO Might

<!-- build: figs/calib-patches.build1.png -->

<!-- build: figs/calib-patches.build2.png -->

<!-- **a** — Region voting runs on a different model from the rest of the talk:
     DINOv3, from Meta. Here's a doll with a book under her arm. Like SigLIP
     earlier, DINOv3 sends a picture into a 768-dimensional space, the same
     kind of cube. But SigLIP turns the whole photo into one point. -->

<!-- **b** — DINOv3 cuts it up first, into a 14 by 14 grid: 196 little patches. -->

<!-- **c** — And each patch gets its own point. SigLIP tells you what a photo's
     about; DINOv3 tells you what's where. Patches of the same thing land near
     each other, so the book's patches gather together; where in the cube
     doesn't mean anything. The green one is the patch outlined on the book. A
     detector scores every one of these, and the photo gets its best patch's
     score. -->
