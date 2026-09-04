# Annotation guide — the 13 candidate classes (#3588)

One paragraph per class: what counts, what does not, and the boundary case that
is measurably there. **Every rule below is measured, not drafted.** For each
class, `coco_folds.py` asked which VG names land on that COCO class's boxes over
the 51,411-image VG∩COCO overlap, which enumerates the boundary cases before a
reviewer meets one. Run against `book` it prints `magazine` (79 boxes) and
`magazines` (30) — so this is the check that would have caught the split that
cost us the `book` pass.

## The protocol, once

- **Good = the object is present.** Drag a box on it. **Bad = not present.**
- VTSearch has **no skip**: Autopilot re-serves an image until it is voted. So
  "I can't tell" has to be voted Bad — which means **Bad on a boxed positive
  reads as _not confirmed_, never as _absent_.** That distinction is what keeps
  the small band from being deleted by the review protocol rather than by the
  data (the reviewer rejected 43% of small-band positives in #3156, a clean
  function of object size, i.e. a property of the protocol).
- **Judge by COCO's reading, not by narrow English.** About a fifth of what you
  see has a known answer — COCO annotated those images exhaustively over exactly
  these classes — and they are indistinguishable from the rest at voting time
  (every file is named by image id alone). They correct nothing; they score the
  reviewer, which is what turns the open half's residual error into a bounded
  number instead of a hope.
- **The rule travels in the dataset name.** The name in the app is the whole
  rule you are voting under. If the name and this document ever disagree,
  the name is what you actually voted, and this document is wrong. The name is
  *display only* — a detector's `text_query` is seeded with the bare COCO class
  (`import_slates.py`), so a long rule never reaches SigLIP and never drags the
  text sort.
- **On an image with no box, when several objects qualify, box the single
  biggest one — not the collection.** A box around six plates measures the
  group, and a band is a claim about the size of *an object*. One plate, the
  largest, is the answer.
- **On an image that arrives WITH a box, judge the object in the box.** The
  question is "is *this* a bowl?", not "is there a bowl here?" — that image was
  drawn into a specific size band *because of* that box. Redraw only to correct
  the extent of the **same** object; if a different, more prominent one is a
  better example, that is not a reason to move the box, because moving it moves
  the image into another band and out of the cell it was sampled to fill. Across
  the first four classes, 13 boxed positives were redrawn and **6 changed
  band** (#3616). If the boxed object is not a member, the answer is Bad —
  remembering that Bad on a boxed positive reads as *not confirmed*.
- **Vote on the object, not a depiction of it.** A car on a billboard, cutlery
  printed on a menu, a bottle in a logo — all Bad, for every class here. So is
  a *pictogram*: the bicycle on a BIKE ROUTE sign is a sign, not a bicycle. Note
  that COCO's annotators do not always agree — three of the ten
  `bicycle@small` positives in #3156 are road-sign pictograms COCO boxed as
  `bicycle`. Where the two rules collide, **this one wins**: a depiction is Bad
  even when COCO boxed it (#3614).
- **A reflection is evidence of the object; an illustration is not.** A phone
  seen in a mirror, a bike in a shop window, a bird on the surface of a pond —
  the object is physically in the scene, and the reflection is how you can tell.
  Good. The line is whether the thing itself is there, not whether you are
  looking straight at it.
- **Obvious toys and models are Bad.** A toddler's plastic Cinderella phone is
  not a cell phone; a die-cast car is not a car. The reason is not English but
  consistency: the shared negative pool was drawn as *images COCO says hold none
  of these classes*, and COCO does not annotate toys — 41 toy-ish boxes in
  49,579 across the shipped twelve, which is noise. Voting a toy Good makes the
  same content a positive here and a negative there. **"Obvious" is the
  operative word**: if you have to squint to decide whether it is a toy, it is
  not obvious, so judge it as the object.

## Risk order

`(no COCO class)` is the share of a VG box of this name that lands on **no**
COCO class at all on an image COCO annotated exhaustively. Since COCO is
exhaustive over these 80 classes, a high share means the VG name covers objects
COCO does not have — i.e. the name means something other than the class. The
mechanical floor is ~7–15% (box slop, occlusion, crowd regions).

**`book`, the class that actually broke, scores 43.3%.** That is the calibration
for reading this column.

| class | unmatched | read as |
|---|---:|---|
| `cell phone` (VG `phone`) | **46.2%** | worse than `book`. The riskiest class here. |
| *(`book`, for reference)* | *43.3%* | *the known failure* |
| `sink` | 34.2% | box-extent risk, not membership |
| `bench` | 29.4% | |
| `bowl` | 26.4% | |
| `bottle` | 25.8% | |
| `vase` | 23.5% | |
| `cup` | 22.9% | but the largest fold-in of any class |
| `chair` | 22.6% | |
| `car` | 20.4% | |
| `spoon` | 16.2% | |
| `truck` | 14.8% | at the mechanical floor |
| `fork` | 13.3% | at the mechanical floor |
| `fire hydrant` | **7.1%** | the cleanest class measured |

---

## Tier A — habitat partners of a class already in *C*

### `truck incl vans not SUVs`

Pickups, box trucks, semis and tractor units, flatbeds, tow trucks, fire trucks
and food trucks all count, as do full-size cargo and panel vans. SUVs,
crossovers and passenger minivans do **not** — those are `car`. The van boundary
is the one COCO is itself inconsistent about, and the inconsistency is
measurable: 261 VG `van` boxes sit on COCO `truck` boxes and 318 sit on COCO
`car` boxes, so COCO's annotators split vans roughly evenly. Use the body, not
the badge: a separate cargo box, or no rear side windows, makes it a truck; a
passenger minivan with seats and windows all round is a car. A cab-only tractor
unit counts; a detached trailer with no cab does not (VG names 63 of those
`trailer`, and COCO boxes them as truck only when the cab is attached). Expect
disagreement with COCO on vans specifically — that is a known cost of this
class, not a mistake.

### `car incl SUVs and minivans`

Sedans, hatchbacks, coupes, estates, SUVs, crossovers, passenger minivans,
taxis and cabs all count — COCO folds all of them into `car` (222 VG `suv`,
318 `van`, 62 `taxi`, 60 `sedan`, 51 `minivan` and 45 `cab` boxes land on COCO
car boxes). Pickups and cargo vans do not; those are `truck`. Parked, partly
occluded, and distant cars all count as long as you can tell it is a car. Note
this is the most prevalent class in the set — annotated on 8.5% of VG — which
means its negative pool is the thinnest and the least trustworthy of the
thirteen, so the boundary stratum will be dense with genuine misses rather than
near-misses. That is the class doing its job: it is here to widen the hard end.

### `fork incl plastic`

Metal, plastic and disposable forks all count, as do serving forks, carving
forks and fondue forks. Spatulas, tongs, whisks and skewers do not. The
measured confusion is not with another object but with **generic names**: VG
labels 56 of COCO's fork boxes `silverware`, 27 `utensil` and 24 `utensils`,
and those boxes often cover a whole place setting rather than one implement —
so vote Good only when the object in the box is a fork, not when a fork is
merely somewhere inside it. Forks resting on a plate, held in a hand, or
sticking out of food all count. `fork` is one of the two cleanest classes here
(13.3% unmatched, at the mechanical floor), so a genuinely hard call is rare
and usually means the object is a spoon or a knife.

### `spoon incl plastic not spatulas`

Teaspoons, tablespoons, soup spoons, wooden spoons, plastic and disposable
spoons and serving spoons all count, and so do **ladles** — a ladle is a spoon
with a deep bowl, and COCO boxes 14 of VG's `ladle` boxes as spoon. Spatulas,
slotted turners, scoops, whisks and tongs do **not**: 17 VG `spatula` boxes
land on COCO spoon boxes, and those are COCO's errors rather than its rule.
As with `fork`, the common trap is a generic box — 73 `utensil` and 34
`silverware` boxes sit on COCO spoons — so judge the object, not the drawer. A
spoon standing in a cup, bowl or jar counts if any part of it is visible; a
spoon-shaped handle on something that is not a spoon does not.

---

## Tier C — objects whose surroundings *are* the negative pool

### `cup incl mugs and glasses not stemware`

**A plain drinking glass is a cup.** This is the largest fold-in measured
anywhere in this set: 1,136 VG `glass` boxes — 13.8% of every COCO `cup` box on
the overlap — are COCO cups, more than ten times the size of the `magazine`
fold-in that broke `book`. Mugs (238 `mug`, 67 `coffee mug`), teacups, paper and
disposable coffee cups (120 `coffee cup`), plastic cups, tumblers and beer
glasses or pints all count. **Stemware does not**: COCO has a separate
`wine glass` class, so anything with a stem and a foot — wine glass, champagne
flute, martini glass, snifter — is Bad here. Measuring cups and trophy cups
count. When in doubt, ask: does it hold a drink and lack a stem? Then Good.

### `bowl incl plates and food containers not wrappers`

**COCO has no `plate` class, so its annotators put plates in `bowl`** — 212 VG
`plate` boxes and 146 `dish` boxes land on COCO bowl boxes. But the name `bowl`
undersells the class badly, which is why the dataset is not called that: the
**fourth-largest fold-in is `container` (143 boxes)**, ahead of `pot` (79) and
`basket` (69). The class is really *rigid open vessels that hold food*.

Count bowls, plates (a **paper plate is a plate**), saucers, shallow serving
dishes, serving pots, baskets that hold food, and **disposable food containers**
— a yogurt pot counts.

A **dog's water bowl is a bowl** — the vessel is what is being judged, and it
holds food or drink for an animal that eats from it.

Do not count:

- **Flat paper wrappers.** A sleeve or a sheet folded round a hotdog is a
  wrapper. But a **paper food boat with turned-up sides is a bowl**, flimsy or
  not — the test is whether it has walls that contain, not what it is made of.
  (`tray`, 19 boxes, splits across this line.)
- **A cup or mug** — that is `cup`, and it is the single largest thing excluded
  here (120 boxes), so expect to reject it often.
- **A toilet bowl.** The word is not the object; nothing about it holds food.
- **A feed trough.** Built for animals, but it is a fixture rather than a
  vessel, and COCO does not fold troughs in.
- **A plant bowl or planter** — that is `vase`, and a plant is not food.
- **A sink basin** (`sink`), an **ashtray**, and a **blender or coffee carafe**.
  None is a systematic fold-in, so excluding them costs almost nothing.

**Open:** a 5-gallon bucket of apples. `bucket` is a real fold-in (19 boxes) and
a bucket does hold food, but it stretches "vessel" a long way. Unruled.

Judge the **vessel, not the food**: an empty plate counts, and a pile of food on
a bare table does not — which matters because VG names 163 of these boxes `food`
and 51 `salad`. Note that `plate` cannot be a class of its own here because it
is polysemous (dinner plate / licence plate), and that is precisely why it lives
inside `bowl`.

### `bottle incl jars`

Water, wine, beer, soda and spirit bottles count; so do **jars — always, and
whatever is in them** (120 VG `jar` boxes are COCO bottles), soap and shampoo
dispensers (47 `soap`), jugs (28), shakers (21), spray bottles, baby bottles,
condiment bottles, and vacuum flasks.

**A jar of cut flowers is still a bottle.** `jar` has no COCO class of its own,
so COCO's annotators sent it both ways — 120 boxes to `bottle` and 41 to `vase`
— and a reviewer meeting the same jar in two slates would otherwise record two
incompatible truths (87 of the 300 bottle images are also in the vase slate).
The rule is decided on manufacture, not use: a jar is a storage vessel, so it is
a bottle, and `vase` is reserved for vessels made as vases. This is what "judge
the container, not its contents" actually buys — it was doing no work while the
contents could still move a jar into another class. Cans, cartons and boxes do not. Bottles behind fridge glass or ranked
on a bar shelf count. Judge the container, not its contents: VG names 124 of
these boxes `wine`, 90 `water` and 87 `beer`, and the box is on a bottle in
every case. A stemmed glass of wine is not a bottle (it is `wine glass`, and
not a class here).

**A fuel tank is not a bottle**, a motorcycle's included, and the reason is not
its shape. Two hold:

- *An integral component of a larger object is not an instance of a container
  class.* **A mouth is not a food container; a stomach is not a bottle.** Both
  hold their contents, and neither is the thing. A fuel tank is part of the
  machine in exactly that way. This is the general form of the test that keeps
  a feed trough out of `bench` and a toilet bowl out of `bowl`, and it is the
  one to reach for first, because it decides without appeal to shape or size.
- *The pool would contradict itself.* COCO has no fuel-tank class, so an image
  whose only vessel-like object is a tank is annotated as holding no bottle and
  sits in the shared negative pool. Voting it Good makes the same content a
  positive here and a negative there, exactly as with toys.

The measurement agrees, emphatically: across **9,169 COCO bottle boxes and 515
distinct VG names, `tank`, `fuel tank`, `gas tank`, `propane`, `barrel`, `drum`
and `keg` appear zero times.** The only near neighbours are `canister` (7) and
`cylinder` (2), and `soap canister` (1) says what kind those are.

Do **not** reach for a neck-and-cap test instead. `jar` (120), `jug` (28) and
`dispenser` (20) all fold in and barely have a neck, so shape would throw out
more than it saves.

**Open:** a *standalone* fuel container — a jerry can, a propane cylinder off
the barbecue. The component test does not exclude it (nothing owns it) and the
pool test does (COCO has no class for it). Unruled until someone meets one.

### `vase incl pots and planters`

**Only a vessel made as one.** Vases, flower pots, planters, urns and pottery
count — 105 VG `pot`, 19 `planter`, 19 `flower vase` and 18 `urn` boxes land on
COCO vase boxes. Vote on the **vessel, not the plant**: a potted plant's pot is
a vase here, even though COCO separately has a `potted plant` class for the
greenery above it. A cooking pot on a stove does not count, and neither does a
plain bowl.

**A borrowed vessel is not a vase**, however it is being used in the picture. A
jar of cut flowers is a `bottle`; a drinking glass of them is a `cup`; a pitcher
of them is neither. This is the ruling that keeps `jar` from meaning two things
at once — see `bottle` — and it is applied on *intent of manufacture*, the same
test that keeps a planter wall out of `bench`.

**Its price is measured, and it is the largest narrowing in this guide.** COCO's
annotators do use vase for a borrowed vessel: `glass` 42, `jar` 41, `bottle` 28,
`container` 20, `pitcher` 18, `bowl` 16, `cup` 11, `jug` 9, and a few more —
**192 boxes, 8.2% of all COCO vase boxes**, against ~3% for the bench narrowing.
Expect to reject a vase COCO annotated roughly one time in twelve.

### `bench not chairs`

Park benches, bus-stop and station benches, picnic-table benches, church pews
and any backed or backless seat **built as seating for two or more people**
count. A single chair does not, and a sofa or couch does not — VG puts 53
`chair` boxes on COCO benches. For a picnic table the bench is the seating plank
rather than the table top, but COCO's boxes here are loose (23 VG `picnic table`
boxes land on bench boxes), so if the whole picnic table is boxed and it has
integral benches, vote Good. A bench with people sitting on it still counts; the
reviewer's job is the bench, not whether it is occupied.

**The two largest confusions are not `chair`.** `seat` (64 boxes) and `table`
(58) both outrank it, and the class's rule turns on two questions COCO's
annotators do not ask.

*Is it seating, or is it a surface?* A **judge's bench is a table** — the
seating is the individual chairs behind it — and so is a low console or coffee
table of bench proportions. COCO boxes 58 VG `table` boxes as bench; reject them
on what the object is *for*, not on its silhouette.

*Was it built as seating, or is it merely sittable?* A **concrete planter wall
in a courtyard is not a bench** even though people sit on it, and neither is a
ledge, a low wall, a kerb or a platform (`wall` 16, `concrete` 6, `platform` 5,
`rail`/`railing` 10). The test is intent of manufacture, not affordance.

Together these narrow the class by roughly 3% of COCO's bench boxes, which the
supply absorbs. A rowboat's **thwart** — the crosswise plank built to sit on —
**counts**: it is seating, and it was built as seating, so it passes both tests
even though it is part of a boat.

### `chair incl stools not couches`

Dining chairs, office chairs, folding and deck chairs, armchairs, high chairs,
and **stools and bar stools** all count — 98 VG `stool` and 42 `armchair` boxes
land on COCO chair boxes. Couches and sofas do **not**: COCO has a separate
`couch` class, and the 37 `couch` boxes sitting on COCO chairs are its errors,
not its rule. Benches do not count either. A single seat within a row of
stadium, theatre or waiting-room seating counts as a chair. Like `car`, this
class is prevalent — 6.3% of VG — so its negative pool is thin and the boundary
stratum will hold many genuine chairs rather than near-misses.

### `sink basin not counter`

Kitchen sinks, bathroom sinks, pedestal basins, utility sinks and vessel basins
all count; bathtubs, showers, toilets and urinals do not. **This class's rule is
about the box rather than about membership**, which is why its name says so:
box the **basin** — the bowl and its rim, plus the tap where the tap is mounted
on the sink itself — and not the whole vanity or the run of counter it sits in.
That is the measured error to avoid: 43 VG `counter` and 8 `countertop` boxes
land on COCO sink boxes. A double sink with two bowls in one unit is one sink,
not two. Its 34.2% unmatched share is high for the same reason: VG's `sink`
boxes are often really counter boxes.

### `cell phone not landlines`

**This is the riskiest class in the set — it scores worse than `book` did.**
VG's `phone` boxes land on a COCO `cell phone` only 53.6% of the time; the other
46.2% land on no COCO class at all, on images COCO annotated exhaustively.
That gap is landlines: COCO has no class for them, so they vanish. Mobile phones
and smartphones count, including one held to an ear, lying face-down on a table,
or in a hand turned away. Landline handsets, desk phones, payphones, wall phones
and intercoms do **not**. Tablets, cameras, remotes, calculators and music
players do not either — COCO's own annotators put 39 `camera` and 12 `ipod`
boxes on cell-phone boxes, and those are their errors. The test: **Bad if the
handset needs the base to work.** A mobile phone resting in a charging dock or
cradle is still Good.

The first wording was *"anything with a cord or a base station is Bad"*, which
discriminates on a base being **present** when what it means is that the handset
is not itself the whole device — and it rejected 2387021, a mobile in a charging
dock (#3612). `pile_config.SCALE_CLASS_RULES` carried the correction before this
paragraph did; where the two ever disagree again, the config is the one the
slate builder reads.

### `fire hydrant not standpipes`

Street fire hydrants count, in any colour or design, including ones half-buried
in snow, wrapped, repainted, or partly hidden behind a post — this class needs
the least judgement of the thirteen. Building standpipes and siamese
fire-department connections mounted on a wall do not count, and neither do
bollards, water valves, parking meters or utility posts. Note the dataset is
built from **both** VG spellings, `fire hydrant` and `hydrant`, which are one
object under two names (box IoU 0.77/0.74); `hydrant` alone accounts for 266 of
the 835 COCO hydrant boxes, so taking either spelling by itself would throw away
a third of the class. At 7.1% unmatched and 81% of COCO boxes carrying a VG box,
this is the cleanest class measured — so a call that feels hard here is a good
sign the object is something else.
