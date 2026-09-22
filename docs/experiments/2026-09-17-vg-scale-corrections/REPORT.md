# What `vg_scale` corrects in Visual Genome

**A running record, kept so the corrections can be described later.** `vg_scale` is built from VG
boxes plus a file of human corrections, and by 2026-09-17 that file holds **4,709 rows** against
VG's own annotation. This is the log of *what kind* of thing was wrong, how often, and a literal
example of each — the part that a table of counts cannot carry.

It is a log, not a study: each row is produced by work recorded elsewhere (the issue is cited), and
the numbers here are copied from those. New kinds get appended as review finds them.

## Why VG needs correcting at all

VG is **not exhaustive** and it is **not adjudicated**. Every entry below follows from one or the
other. Two consequences shape the whole benchmark:

* A class that is present and unnamed makes the image a false negative. Measured silence rate:
  **1.0%** [0.90%, 1.1%], bound 1.7% (#3696).
* A box is not only a label, it is a **size**. `vg_scale` puts an image in a band from the union of
  a class's boxes, so a box on the wrong object, or on the smaller of two instances, moves the image
  to the wrong band — a quieter error than a wrong label, and one nothing downstream can detect.

## The corrections, by kind

| kind | how often | what it does to the benchmark |
|---|---|---|
| class present, VG silent | 3,837 images added as boxed positives (#3940) | were false negatives |
| box on a smaller instance than the frame's main one | **8.3%** of stored-box positives (#3924, #3925) | one band too small |
| box on the wrong object | **1.2%** (#3924, #3925) | wrong label |
| box on no object at all | seen in review; rate not yet measured | wrong label and wrong band |
| one object annotated many times | 130 of 2,896 seat images carry >1 box | no data effect; it defeats review |
| the class does not hold the object | 21 `vase` positives retired (#3784) | definitional, not perceptual |

### Literal examples

* **Box on empty background.** `bicycle` on image **2388483** (500x375): VG's box is
  `[274, 136, 341, 212]`, the empty road to the right of the cyclist, while the bicycle being ridden
  is unboxed. Traced end to end 2026-09-17: VG's recorded dimensions match the file, and the value
  reaching the render equals the annotation, so the box is VG's error and not a coordinate fault.
* **Smaller instance boxed, larger one ignored.** `bowl` on **4299**: the box is on a small green
  bowl on a shelf while a large bowl sits on the kitchen island. This is the 8.3% case, and it is
  why the band moves: of 102 redraws in the `car`/`cup` triage, **63 moved the image up a band** and
  **55 left the small band** (#3939).
* **A can is not a cup.** `cup` on **4363**: the only candidate object is a drinks can, so the image
  stays a negative rather than becoming a positive (#3940).
* **One object, twenty-one boxes.** `bench` on **2320173** carries 21 VG bench boxes, and `bird` on
  **2335647** carries 23, all of one object. They do not change the band (the union is the same), but
  a render outlining each one is unreadable, so review renders collapse near-duplicates (#3976).
* **Definitional, not perceptual.** A kettle is not a `bottle`; a spreader is not a `knife`; a car
  seat is not a `chair`, and by the same principle a rowboat's thwart is not a `bench` (#3756);
  planters are not `vase` (21 positives retired, #3784). VG annotators were not wrong about the
  pixels — the class does not admit the object.

## How the corrections are made

1. **A human answers a rendered question.** The renders come from
   [`scripts/experiments/pile/slate_render.py`](../../../scripts/experiments/pile/slate_render.py):
   the photo with the box drawn on it, and the same box magnified beside it on black padding, so no
   part of the scene is covered. Each answer is Good, a redrawn box, or Bad.
2. **The answers are banked** as labelsets under
   [`scripts/experiments/pile/human_record/`](../../../scripts/experiments/pile/human_record), each
   stamped with the rule wording its reviewer read (#3814).
3. **They become corrections** through
   [`scripts/experiments/pile/pass_verdicts.py`](../../../scripts/experiments/pile/pass_verdicts.py)
   and `verdicts_to_corrections.py`, then `apply_recheck.py`. A redrawn box designates one instance
   and re-bands the image; a Good with no redraw confirms the stored box; a Bad leaves the image a
   negative.

## Open, for whoever tells the story

* **The rate of "box on no object" is not measured.** The seat check running now (2,896 images, of
  which 1,790 are positives no human has seen) is what would measure it.
* **VG's redundant boxes are not deduplicated in the data**, only in the renders. Nothing depends on
  it today because the band uses the union.
* **The corrections are not symmetric.** Review has looked hardest where a defect was suspected
  (`bowl`, `car`, `chair`, `cup`), so the per-class rates above are not comparable across classes.
