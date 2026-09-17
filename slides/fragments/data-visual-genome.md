<!-- _class: full -->

![bg fit](figs/dataset-card-visual-genome.webp)

## Visual Genome

<!-- The set almost everything in this talk was measured on, and the one most
     people in the room will not have opened. Twelve real frames, drawn at
     random from the id space and then filtered only to drop close-up
     portraits — nothing here is picked for being clean. A laptop screen shot
     at an angle, two street signs, a plate of pasta: that is what the pile
     looks like. -->

<!-- What makes it worth building on is the *density*. A hundred and eight
     thousand photographs, two and a half million objects, and every object
     carries a pixel box and a name. One frame is a dozen labelled things
     rather than one, which is what lets a single dataset answer questions
     about small objects and large ones without changing dataset. -->

<!-- The catch, and it is the one the next four slides are all about: the
     names are **free text**. Whoever annotated the image typed a word.
     Nothing reconciles `bike` with `bicycle`, and every object's name list
     has exactly one entry, so there is no synonym to look up. -->

<!-- And it is not exhaustive. Measured against COCO on the twenty-five
     classes we care about, VG names **0.61** of the objects COCO does. So
     an image where VG says nothing is not an image with nothing in it — the
     single fact that costs the most to work around. -->

<!-- In the app it is `visual_genome_s/m/l/a` on the Demo tab; the download
     is about 14.5 GB, which is why the cards further on are built on top of
     it rather than beside it. -->
