# 2026-08-17 — I rendered the evidence and did not look at it (#3156)

**Cost:** a false claim in a merged report, caught by the owner rather than by me;
one correction PR, and a Visual Genome example that has to be retired.

**What broke.** The overview-bench report shows contact sheets of the images
behind each error claim — I built that machinery specifically so "is this a model
error or a label error?" could be judged by eye. Under the `visual_genome_m` /
`bus` sheet I wrote: *"No buses, and no argument that the labels are wrong
either."* The owner opened the page and found buses: `498326.jpg` is an
articulated trolleybus filling the frame (annotated `car, clouds`), `3078.jpg` is
a red double-decker with `Metroline` written across it (annotated
`door, tire, window`), and `286074.jpg` has several more. Three of the eight
images I was pointing at contradicted the sentence under them.

**The mechanism is not "VG is noisy" — that was already the report's finding.**
It is that I wrote a caption from the *aggregate* (a 60 % false-positive rate that
looked like a threshold collapse) and from the *entailment test* (which found
nothing for `bus`, because no category in the vocabulary entails a bus), and then
rendered images and did not read them. The entailment test can only find misses
that have a co-occurring label; it is silent exactly where a whole object is
simply absent from the annotation. Silence from that test is not evidence of
clean labels, and I treated it as if it were.

**Worse, it was a negative claim.** "The labels are wrong here" is a claim the
sheet can support. "The labels are *not* wrong here" needs every image checked,
and I had eight of them on screen.

**Prevented?** *Advice only, and deliberately so* — no script can tell whether I
looked. The concrete practice: **a sheet is not a decoration, it is a check you
have to actually run.** Before writing any sentence about what a set of images
does or does not contain, read every thumbnail on the sheet, and prefer the
positive form ("these three contain the target") over the negative one. If a
claim rests on a test that can be silent — the entailment test with no entailing
category, a grep with no hits — say which test was silent rather than reporting
its silence as a result.

**Related:** the same reading is why #3156 exists (correcting VG's annotations
rather than working around them), and why the 250-vote `vg_box_*` grid was
cancelled mid-flight: those datasets are being redefined, so their numbers were
going to be superseded either way.
