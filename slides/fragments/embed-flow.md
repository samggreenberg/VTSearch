<!-- _class: full -->

![bg fit](figs/embed-flow.png)

## Embed-time<br>Stories

<!-- build: figs/embed-flow.build1.png -->

<!-- build: figs/embed-flow.build2.png -->

<!-- build: figs/embed-flow.build3.png -->

<!-- **a** — Quick detour: how does typing one word find pictures? Start with
     the pile of photos. -->

<!-- **b** — When we import them, every photo goes through a big pretrained
     network, SigLIP, once, and comes out as a point in a 768-dimensional
     space. The cube's standing in for those 768 dimensions. Similar photos end
     up near each other. That's also why retraining is so fast: the big network
     never runs again, and the detector is a tiny model on top of these points.
     -->

<!-- **c** — Now the word. "book" goes through the same network. One model, two
     kinds of input. -->

<!-- **d** — So it lands in the same space as the photos, and "which photos are
     near the word book?" is a question with an answer. That's our first
     ranking, and that's where the first question comes from. After that, the
     votes take over and the word stops mattering. -->
