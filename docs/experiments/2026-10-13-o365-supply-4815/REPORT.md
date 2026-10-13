# Objects365 v2 val as a second source for `coco_better`'s cells (#4815)

**2026-10-13.** Every object-class method comparison in this repo is read off
`coco_better`. Its per-cell voting/test split guards a detector against its own
votes, but not a *method* against COCO: the curation, the annotators' habits and
the rulings built on them (#4056, #4119, #3985, #4179) are the same on both halves
of every split. #4815 asked for a second bench from a different source, to check
for over-tuning to COCO and for domain shift. This measures whether the
Objects365 v2 validation split can supply one under the same construction —
same *C*, same three bands, same shared-pool rule — before anything is staged or
embedded.

**It can, at reduced width.** 110 of the 147 cells reach the shipped
`SCALE_N_POS` of 100; 107 of them pair with a built `coco_better` cell. Five
classes go entirely, and they are the sports gear COCO was collected around. The
shared pool falls short: 7,356 images hold none of *C* against the 10,900 the
draw takes, so a cell here is scored against a median 16,693 negatives where a
`coco_better` cell carries about 23,900. The measurement takes under a minute
and needs no images, no pile and no GPU.

```bash
python scripts/experiments/pile/o365_supply.py --annotations <zhiyuan_objv2_val.json> --out supply.json
```

`supply.json` beside this report is that run's output.

| | |
|---|---:|
| Objects365 v2 val images | **80,000** |
| boxes, over 365 categories | 1,240,587 |
| cells reaching `SCALE_N_POS` = 100 | **110 of 147** |
| … of which pair with a built `coco_better` cell | 107 |
| classes with no cell at all | 5 |
| shared-pool candidates (hold none of *C*, not even doubtfully) | **7,356** (draw takes 10,900) |
| images a build emits | 17,403 |
| negatives each built cell is scored against (min / median / max) | 12,722 / **16,693** / 17,301 |
| same, `coco_better` (at its #4056 rebuild) | 23,891 |

## How the census reads Objects365

The census is not a restatement of the build. It calls the loader's own reader
(`read_o365_labels`) and banding (`banded`, the shipped `band_candidates` on the
largest instance), and computes each cell's negatives from the same label fields
the build writes. Three choices in that reader decide the numbers.

**The class map goes through the rules, not through COCO's names.**
`pile_config.O365_NAMES` maps each class of *C* to the Objects365 categories its
`SCALE_CLASS_RULES` entry describes. Most are one-to-one. The rest follow the
rule text: `enclosed road vehicle` takes Car, SUV, Van, Pickup Truck, Sports Car,
Truck, Heavy Truck, Fire Truck and Ambulance; `bowl` takes Plate; `bottle` takes
Jug; `chair` takes Stool; `tie` takes Bow Tie; `bag or luggage` takes Briefcase;
`bird` takes seven bird categories. The reader refuses a name its file does not
carry, because Objects365 spells some names wrong (`Dinning Table`,
`Moniter/TV`), and a "corrected" spelling would map nothing without saying so.

**A box makes a positive only if it is ONE real object.** The release flags
three kinds of box: `iscrowd` (a group nobody separated), `isfake` (a depiction)
and `isreflected` (a mirror image). A flagged box, a box with no area, or a box
under an ambiguous name makes its (image, class) pair *doubtful*. The image then
holds the class, so it is never a negative for it, and the pair is never a
positive either. An unflagged box of the same class in the same image still
bands it.

**Ambiguous names are withheld, not guessed.** `pile_config.O365_AMBIGUOUS` lists
the 18 categories that may or may not be a class under its rule, such as Desk for
`dining table` (the rule is function, not furniture) or Scooter for `motorcycle`
(motor scooters are Good, kick scooters are not). Withholding costs supply and
never contaminates a label, which is the right direction to err in before
anyone has ruled.

## The cells

Bold is under 100; those 37 cells are `pile_config.O365_DROPPED_CELLS`.
"Unflagged" counts the images where the class has at least one box that can
band.

| class | small | medium | large | images holding it | of which unflagged |
|---|---:|---:|---:|---:|---:|
| enclosed road vehicle | 1,521 | 2,617 | 1,695 | 7,126 | 5,833 |
| clock | 729 | 570 | **22** | 1,341 | 1,321 |
| bird | 1,096 | 1,843 | 601 | 4,521 | 3,540 |
| boat | 222 | 626 | 1,017 | 1,997 | 1,865 |
| umbrella | 310 | 584 | 343 | 1,357 | 1,237 |
| kite | **27** | 115 | **45** | 190 | 187 |
| book | 579 | 1,524 | 286 | 3,993 | 2,389 |
| dog | 107 | 379 | 351 | 956 | 837 |
| bag or luggage | 1,475 | 3,206 | 697 | 5,886 | 5,378 |
| knife | 850 | 3,026 | 618 | 5,189 | 4,494 |
| bicycle | 255 | 671 | 314 | 1,831 | 1,240 |
| bus | 229 | 272 | 224 | 770 | 725 |
| stop sign | 358 | **87** | **8** | 457 | 453 |
| fork | 511 | 3,006 | 794 | 4,457 | 4,311 |
| spoon | 621 | 2,410 | 620 | 3,844 | 3,651 |
| single serving drinking vessel | 4,815 | 6,948 | 1,362 | 14,056 | 13,125 |
| bowl | 2,573 | 4,768 | 10,329 | 23,934 | 17,670 |
| bottle | 5,127 | 5,189 | 551 | 17,306 | 10,867 |
| vase or potted plant | 1,211 | 2,023 | 373 | 4,562 | 3,607 |
| bench | 375 | 810 | 429 | 1,821 | 1,614 |
| chair | 1,086 | 6,440 | 3,008 | 12,058 | 10,534 |
| sink | 1,879 | 6,151 | 1,223 | 9,332 | 9,253 |
| cell phone | 1,707 | 1,092 | **39** | 2,937 | 2,838 |
| fire hydrant | 176 | **42** | **11** | 234 | 229 |
| airplane | 111 | 438 | 683 | 1,295 | 1,232 |
| apple | 846 | 725 | **95** | 3,640 | 1,666 |
| banana | 695 | 1,042 | 356 | 2,985 | 2,093 |
| baseball bat | **12** | **73** | **25** | 123 | 110 |
| dining table | **8** | **58** | 2,563 | 17,309 | 2,629 |
| frisbee | **56** | **84** | **5** | 146 | 145 |
| keyboard | 587 | 3,072 | 443 | 4,121 | 4,102 |
| laptop | 132 | 1,841 | 1,073 | 3,076 | 3,046 |
| microwave | 178 | 1,870 | 229 | 2,291 | 2,277 |
| motorcycle | 136 | 471 | 224 | 1,385 | 831 |
| mouse | 3,163 | 946 | **16** | 4,142 | 4,125 |
| orange | 470 | 621 | 116 | 2,855 | 1,207 |
| parking meter | 161 | 114 | **38** | 313 | 313 |
| person | 3,412 | 8,579 | 18,334 | 33,217 | 30,325 |
| remote | 1,031 | 368 | **32** | 1,451 | 1,431 |
| scissors | 658 | 511 | **89** | 1,327 | 1,258 |
| skateboard | **13** | **56** | **17** | 92 | 86 |
| skis | **18** | **36** | **13** | 69 | 67 |
| snowboard | **16** | **41** | **20** | 80 | 77 |
| surfboard | **40** | 139 | **73** | 259 | 252 |
| tennis racket | **13** | 114 | **26** | 159 | 153 |
| tie | 646 | 589 | **36** | 1,335 | 1,271 |
| toothbrush | 927 | 469 | **51** | 1,623 | 1,447 |
| traffic light | 743 | 240 | **8** | 1,014 | 991 |
| tv | 379 | 3,395 | 1,832 | 5,740 | 5,606 |

**The shortfall is COCO's curation showing, not noise.** COCO was collected
around its eighty classes, so it over-samples the things they name. #3983 found
235, 253 and 444 `small` images of skateboard, snowboard and surfboard in COCO;
Objects365 val has 13, 16 and 40. `baseball bat`, `frisbee`, `skateboard`,
`skis` and `snowboard` have no cell here at all. Most of the other dropped cells
are `large` cells of small objects (a mouse, a remote, a traffic light filling
the frame), a close-up a general photo pool rarely takes. That is itself the
shift #4815 was after: a method tuned where every class is plentiful at every
size meets a source where some are not.

**Objects365 fills three cells `coco_better` cannot.** `apple@small`,
`banana@small` and `orange@small` (846, 695, 470) are dropped from `coco_better`
because COCO draws one box round a pile of fruit and LVIS rarely boxes a small
one (#3985). Objects365 flags those piles `iscrowd`, so the single fruit it does
box individually band honestly, with no second annotation source needed.

**`dining table` is mostly withheld.** 17,309 images hold it and only 2,629
unflagged: Desk alone makes 14,246 pairs doubtful. Under the class rule a desk
with a meal on it is Good and a bare one is not, and Objects365's furniture
labels cannot tell the two apart. The `large` cell fills anyway (2,563).

## The shared pool

7,356 images hold none of *C*, not even doubtfully, against the 9,900 negatives
plus 1,000 spares the draw takes. `draw_negatives` takes the whole pool and
draws no spares. Objects365 photographs are busier than COCO's (15.5 boxes per
image against COCO's ~7), so fewer of them are empty of all 49 classes.

Each built cell is also scored against the other cells' positives that do not
hold its class (#3667), so a cell's haystack is the pool plus those: a median of
**16,693** negatives, against 23,891 on `coco_better`. At 100 positives that is
a prevalence of about 0.60% against 0.42%. #4056 measured what haystack size
does to AP: −0.030 when a `coco_better` cell grew from 16,535 to 23,891
negatives. **Absolute AP is therefore not comparable across the two
benches.** A method contrast measured within each, A against B on the same
source, is, and that is the use.

## What makes a pair doubtful

46,264 (image, class) pairs are held but doubtful. A pair can have more than one
reason, so the counts overlap:

| reason | pairs |
|---|---:|
| `iscrowd` | 15,073 |
| name: Desk | 14,246 |
| name: Bakset (sic) | 4,413 |
| name: Toiletry | 3,516 |
| `isreflected` | 2,788 |
| name: Pot | 1,438 |
| name: Cleaning Products | 1,214 |
| `isfake` | 1,195 |
| name: Tea pot | 761 |
| name: Flower | 747 |
| name: Chicken | 686 |
| name: Machinery Vehicle | 637 |
| name: Scooter | 492 |
| name: Rickshaw | 413 |
| name: Side Table | 333 |
| name: Wallet/Purse | 325 |
| name: Coffee Table | 304 |
| name: Tricycle | 216 |
| name: Flask | 134 |
| name: Wheelchair | 124 |
| name: Formula 1 | 61 |

**Crowds needed a different rule from `coco_better`'s.** `coco_better` drops
`iscrowd` regions outright, so a class held *only* by a crowd box would read as
absent and could land in that class's negatives. Checked with `coco_better`'s own
reader over COCO 2017 train+val: no image holds a class of *C* only as a crowd
region, because COCO boxes the individual instances beside a crowd. The
shortcut costs COCO nothing. On Objects365 it would have put 15,073 crowds of a
class into that class's negatives, so this loader treats a crowd as held.

## Rulings owed

Each ambiguous name stays withheld until an owner rules. A ruling either moves
the name to `O365_NAMES` (its boxes become positive candidates) or deletes it
(its images become plain negatives). What each name alone withholds:

| name | for | pairs kept out of negatives | clean-pool images withheld |
|---|---|---:|---:|
| Desk | dining table | 14,074 | 468 |
| Bakset (sic) | bowl | 4,130 | 321 |
| Toiletry | bottle | 2,968 | 197 |
| Pot | bowl | 1,123 | 98 |
| Flower | vase or potted plant | 715 | 33 |
| Cleaning Products | bottle | 693 | 78 |
| Chicken | bird | 650 | 177 |
| Tea pot | bottle | 634 | 16 |
| Machinery Vehicle | enclosed road vehicle | 588 | 115 |
| Scooter | motorcycle | 478 | 21 |
| Rickshaw | enclosed road vehicle, bicycle | 399 | 5 |
| Wallet/Purse | bag or luggage | 319 | 6 |
| Tricycle | bicycle | 199 | 5 |
| Side Table | dining table | 159 | 7 |
| Coffee Table | dining table | 129 | 5 |
| Flask | bottle | 125 | 3 |
| Wheelchair | chair | 118 | 3 |
| Formula 1 | enclosed road vehicle | 53 | 12 |

Ruling every name out of the table would return 1,800 images to the pool, for
9,156 in all, still short of 9,900. No ruling closes the pool gap; the rulings
decide which pairs score.

**Some of what the rules count has no Objects365 category at all,** and can
therefore sit unlabelled in a negative: parachutes and parasails (`kite`), jars
and shakers (`bottle`), saucers and food containers (`bowl`), magazines
(`book`, if not boxed as Book). No count can show these without looking at
pixels. If a confirmation result turns on one of these classes, review its
negatives before reading it.

## What it takes to build

- **Stage** the annotation file (269 MB) and the 44 image tarballs (17.5 GB in
  all), each extracted under `val/images/v1/` or `val/images/v2/` of
  `pile_config.O365_ROOT`, so every annotation's `file_name` resolves as written.
  `build_pile.py --rebuildable` checks the annotation file and all 44 directories.
- **Build** `o365_better`, which is `on_request`, by naming it. A designated
  build emits 17,403 images, fewer than `coco_better`'s.
- **Licence.** The file's own `licenses` table lists Creative Commons
  non-commercial licences, the Flickr pattern. Cells are vectors of these images
  and stay inside the group, like COCO's.

**The train split would probably fill every cell, but that is unmeasured.** Its
annotation file is 20 times val's (5.5 GB against 269 MB), so by proportion even
`dining table@small` (8) would clear 100. That file is a single JSON that does
not parse in memory on a 15 GB machine, and its images ship as 51 patches; the
three sizes checked run from 3.9 to 9.2 GB each. The census above was run on val
alone, and so is the loader.

## Using it

`CALIB_DATASETS=o365_better`, with the same two arms as `coco_better` (`siglip`,
`siglip+dinov3_patch`) under their own variable, `CALIB_O365_BETTER_EMBEDDERS`,
and the same typed queries.

**It is a confirmation bench.** Its value is that nothing has been tuned on it.
Read a method contrast decided on `coco_better` here before it ships, over the
107 paired cells. If a study tunes on `o365_better` instead, the repo has two
tuned benches and no check.
