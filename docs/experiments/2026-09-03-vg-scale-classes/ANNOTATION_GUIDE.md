# Annotation guide — the 13 candidate classes (#3588)

One paragraph per class: what counts, what does not, and the boundary case that
is measurably there. **Every rule below is measured, not drafted.** For each
class, `coco_folds.py` asked which VG names land on that COCO class's boxes over
the 51,411-image VG∩COCO overlap, which enumerates the boundary cases before a
reviewer meets one. Run against `book` it prints `magazine` (79 boxes) and
`magazines` (30) — so this is the check that would have caught the split that
cost us the `book` pass.

## What we are optimising

**The goal is the best final dataset, not fidelity to COCO.** Merging, splitting,
renaming and redefining are all allowed. We build on COCO because starting from
200k annotated images is easier than starting from nothing, not because its
taxonomy is right — it has no `plate` class, files magazines under `book`, puts
bicycle pictograms in `bicycle`, and splits vans between `truck` and `car`.
Where a different boundary makes a more coherent class, take it.

So *"judge by COCO's reading"*, below, is a **default with a reason**, not the
objective. The reason is the scored subset: a fifth of every slate carries a COCO
answer, and that is the only thing turning a reviewer's residual error into a
number instead of a hope. What matters is not agreeing with COCO but staying
**expressible** against it. Three cases, and they cost very differently:

| kind of change | example | what happens to the reference |
|---|---|---|
| **Union** of COCO classes | `cup` ∪ `wine glass` | Derivable — "COCO annotated a cup or a wine glass here". Calibration intact. **Free.** |
| **Narrowing** inside one | a judge's bench is not a `bench` | COCO is a superset, so disagreement is visible and priced. Calibration works, and shows a known rate. **Cheap, if recorded.** |
| **Extending beyond** COCO | a jerry can is a `bottle` | No reference for the new part; those images sit in the shared negative pool and the calibration cannot see them. **Expensive — and the only one that can silently corrupt the pool.** |

Every ruling in this guide is one of those three, and each carries its cost in
the text. That is the discipline the goal requires: not "does COCO agree" but
"if it does not, do we know what that costs us".

## A note on names

**A capitalised name means the class; a lowercase one means the English word.**
A Chair is whatever this guide says it is; a chair is what you would call a
chair. The two come apart constantly — a judge's bench is not a Bench, a toilet
bowl is not a Bowl, a jar of flowers is a Bottle — and most of the rulings here
exist precisely at that gap. Code keeps the lowercase form, because `"chair"` is
the COCO key and a string is not prose.

## The protocol, once

- **Good = the object is present.** Drag a box on it. **Bad = not present.**
- VTSearch has **no skip**: Autopilot re-serves an image until it is voted. So
  "I can't tell" has to be voted Bad — which means **Bad on a boxed positive
  reads as _not confirmed_, never as _absent_.** That distinction is what keeps
  the small band from being deleted by the review protocol rather than by the
  data (the reviewer rejected 43% of small-band positives in #3156, a clean
  function of object size, i.e. a property of the protocol).
- **Judge by COCO's reading, not by narrow English** — *unless a ruling above
  says otherwise, and several do.* About a fifth of what you
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
- **Even when the box is plainly wrong and a real one is elsewhere, the answer
  is still Bad.** This is the case the rule above did not spell out. Image
  2334634 arrives as a `cup@large` positive with its box on a *windowpane*, and
  the photo does contain two real drinking glasses lower down. Re-boxing one of
  them keeps a positive, but it moves the image from `large` to `medium`: the
  `large` cell stays short — correctly, because its supply was overstated by an
  annotation error — while `medium` gains an image nothing sampled it for. **A
  wrong box is a finding about the band, not an inconvenience to route around.**
  Reject it, and the real glass is simply not recruited from this stratum; the
  cost is one positive, against a band structure that stays honest.
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
- **Containing a thing does not make it a container *for* that thing.** A
  5-gallon bucket of apples is not a food container; nor is a shopping cart, a
  grocery store, or a car boot with the shopping in it. Each holds food and none
  was made to. The vessel has to be *for* the contents — which is the same
  intent-of-manufacture test as the one above, pointed at contents instead of at
  ownership, and it is what keeps "judge the vessel, not the food" from
  swallowing the whole scene.
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

**A held fork is often gripped to stab; a spoon never is.** When only the handle
shows and the food gives nothing away, the grip does: a fist closed over the
handle with the business end pointing down and away is a fork. This is the
second rule here that reads the surroundings rather than the object — see the
handle-in-the-food rule under `spoon` — and it arrived after `spoon` was
reviewed, so it applies from `fork` onward rather than retroactively.

**The literal error to expect: image 2322780 is a steam locomotive, and its
cow-catcher is a boxed `fork` positive.** COCO annotated the slatted triangular
pilot at the front of the engine as cutlery — tine-like, at a glance, and
somebody clicked it. It is worth knowing this is in the slate, because it is a
`fork@medium` box, which puts it *above* the small-band guard: rejecting it
actually removes it, unlike the `bicycle@small` pictograms that needed #3614.
The `positive_boxed` stratum exists exactly to catch this.

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

**When only the handle shows, read the food.** A utensil buried in a dish with
its business end out of sight is the common case here, and it has a good answer:
what it is in tells you what it is. A handle out of cereal is a spoon; a handle
out of a salad is a fork, so Bad.

This is the one rule in the guide that licenses **inference from the
surroundings rather than from the object**, and it is worth being explicit that
it does, because everything else here insists on judging the thing itself. The
justification is the alternative: with no skip, an unreadable utensil has to be
voted Bad, so a blanket "can't see the end, say no" would delete every partly
buried spoon in the class — the review deciding the data rather than the data
deciding, which is exactly the failure the small-band guard exists for (#3156's
43%). Context is weaker evidence than sight, and it beats discarding the image.

---

## The vessel ladder: an empty vase against an ornamental bowl

Four of these classes are open vessels and the words run out fast — an empty
large vase and an ornamental bowl are both decorative, both made as themselves,
and neither is holding anything to judge. **Their boxes separate almost
completely on one number.** Measured over every non-crowd COCO box:

| class | n | h/w p25 | median | p75 | taller than wide | median area |
|---|---:|---:|---:|---:|---:|---:|
| `bottle` | 25,081 | 1.84 | **2.52** | 3.22 | 94% | 0.0036 |
| `vase` | 6,849 | 1.16 | **1.58** | 2.16 | 84% | 0.0095 |
| `cup` | 21,458 | 0.95 | **1.26** | 1.61 | 71% | 0.0054 |
| `bowl` | 14,944 | 0.49 | **0.66** | 0.86 | 13% | 0.0139 |

**A vase is taller than it is wide; a bowl is wider than it is tall.** 84%
against 13%, and vase's p25 (1.16) sits above bowl's p75 (0.86) — the middle
halves do not overlap at all. That is the discriminator, and size is not: bowl's
median box is *larger* than vase's, so "large" does not push a vessel towards
`vase`.

The ladder is worth carrying whole, because the four classes sit on it in order
and the near-misses are always neighbours: **bowl 0.66 → cup 1.26 → vase 1.58 →
bottle 2.52.** Cup and vase are the closest pair, which is why a tall tumbler and
a squat bud vase are genuinely hard, and why `glass` folds 42 times into vase.

**This does not license shape tests generally.** A neck-and-cap test for `bottle`
was rejected two sections down precisely because the data refused it — `jar`
(120) and `jug` (28) fold in without necks. The difference is that this shape
test *is* the measurement rather than a guess about it. Reach for geometry only
where it has been checked.

## Tier C — objects whose surroundings *are* the negative pool

### `cup incl mugs and glasses not stemware`

**A plain drinking glass is a cup.** This is the largest fold-in measured
anywhere in this set: 1,136 VG `glass` boxes — 13.8% of every COCO `cup` box on
the overlap — are COCO cups, more than ten times the size of the `magazine`
fold-in that broke `book`. Mugs (238 `mug`, 67 `coffee mug`), teacups, paper and
disposable coffee cups (120 `coffee cup`), plastic cups, tumblers and beer
glasses or pints all count, and so do measuring cups (9). **A cup is hand-held
and a single serving** — that, not shape, is the test, and it is what separates
this class from `bottle` on the pouring vessels below.

**A drinking glass holding cut flowers is still a cup**, not a vase — `vase` is
reserved for vessels made as vases, so a borrowed one stays with whatever it was
made as.

Bad here:

- ~~**Stemware.**~~ **Stemware now COUNTS** — `cup` was merged with COCO's
  `wine glass` class on 2026-09-04, so a wine glass, champagne flute, martini
  glass or snifter is Good here. The dataset name says so:
  `cup incl mugs glasses and stemware`. Everything below is the argument that
  was made for keeping them apart, kept because it is the measurement, not the
  decision.

  *Is stemware not a kind of glass?* In English yes, and that is the trap. The
  class is not "glass": VG's `glass` folds **1,136** boxes into `cup` because
  most glasses are tumblers, while every stemware word together —
  `wine glass` 6, `wine` 6, `cocktail` 2, `wine glasses`, `goblet`, `flute`,
  `champagne` — folds in **18 times in 8,242, 0.22%**. A 63:1 ratio. COCO cuts
  exactly where this rule cuts, and does it more cleanly than any other boundary
  measured in this guide.

  **Merging them IS available** — this guide said otherwise for one commit and
  was wrong. `wine glass` is itself one of COCO's exhaustively annotated 80, so
  the reference for a merged class is simply *"COCO annotated a cup or a wine
  glass here"*: well defined, and the scored subset survives untouched. A union
  of two COCO classes is not the same thing as a class COCO does not have, which
  is what the toy and fuel-tank arguments turn on.

  What a merge would actually buy and cost, measured: **+8,180 boxes (+38%),
  +1,469 images, and +35% in the small band** — which is the binding constraint
  on class supply everywhere in this project (#3603). Against that, 1,469 images
  holding stemware but no cup would have to leave the shared negative pool at
  build time, and the class would be the first here that is not a plain COCO
  class. Deliberately left as a decision rather than a rule, because it is one.
- **A jar**, however it is being drunk from. A jar is a `bottle` unconditionally
  (see there), and COCO does put 25 VG `jar` boxes on cups, so this one will
  come up.
- **A can** (21 boxes), a **tin**, a **carton**. As in `bottle`, the container
  is judged, and none of these is a cup.
- **Anything that serves more than one.** A **pitcher** (30), a **jug**, a
  carafe, a teapot, a thermos (5), a dispenser (2) — these are `bottle`.
  Together with the jars, cans and bottles above, the whole not-a-cup family is
  **146 boxes, 1.8% of COCO cup**, so the narrowing is cheap.
- **A bucket** (13). Not a cup, and not a `bottle` or a `bowl` either — a bucket
  is a general-purpose container that was made for nothing in particular. COCO
  puts buckets on bowls 19 times and cups 13; both are rejected.
- **A trophy cup.** Named a cup and nothing else about it is one: it serves no
  drink, so it is not a single serving of anything. Flagged because an earlier
  draft of this guide counted it, and that claim was never measured — VG uses
  the word `trophy` on **zero** of COCO's 8,242 cup boxes.

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
  here (120 boxes), so expect to reject it often. **A ramekin is a bowl**, not a
  cup: it is hand-held and single-serving, but it holds *food*, and the ladder
  agrees — a ramekin is wider than tall. COCO splits it 5 to bowl and 3 to cup,
  too thin to settle alone, which is why the drink test and the geometry carry
  it.
- **A toilet bowl.** The word is not the object; nothing about it holds food.
- **A feed trough.** Built for animals, but it is a fixture rather than a
  vessel, and COCO does not fold troughs in.
- **A plant bowl or planter** — that is `vase`, and a plant is not food.
- **A sink basin** (`sink`), an **ashtray**, and a **blender or coffee carafe**.
  None is a systematic fold-in, so excluding them costs almost nothing.

**A 5-gallon bucket of apples is not a bowl**, and neither is a shopping cart, a
grocery store, or a car boot with the shopping in it. Holding food is not the
test; being made to hold food is. `bucket` is a genuine fold-in — 19 boxes COCO
called a bowl — so this one is a real disagreement rather than a hypothetical,
but it is a cheap one, and the alternative has no floor: each of those four
holds food, and nothing in "it contains food" stops at the first.

Judge the **vessel, not the food**: an empty plate counts, and a pile of food on
a bare table does not — which matters because VG names 163 of these boxes `food`
and 51 `salad`. Note that `plate` cannot be a class of its own here because it
is polysemous (dinner plate / licence plate), and that is precisely why it lives
inside `bowl`.

### `bottle incl jars`

Water, wine, beer, soda and spirit bottles count; so do **jars — always, and
whatever is in them** (120 VG `jar` boxes are COCO bottles), soap and shampoo
dispensers (47 `soap`), jugs (28), shakers (21) — **salt (16) and pepper (11) shakers included** — spray bottles,
baby bottles, condiment bottles, vacuum flasks, carafes (6 — `bowl` sends them
here), and **pitchers and jugs**. A pouring vessel that serves more than one person is a
bottle: `cup` is reserved for what is hand-held and a single serving.

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

The seasoning and condiment shelf belongs here as a whole: shakers, `condiment`
(17), `ketchup` (17), `mustard` (17), `dispenser` (20), `oil` (9), `spice` (8),
`salt` (7) and `pepper` (9) all land on COCO bottle boxes — **181 boxes for the
family, against 42 on cup and 51 on bowl.**

**A squeezable tube is a bottle** — toothpaste, suntan lotion, hand cream,
shower gel, ointment. It is made for what is in it, nothing owns it, and it is
not a single serving, so every test in this guide sends it here. Note also that
*material and rigidity are already rejected as tests*: `bowl` counts a flimsy
paper boat on the strength of its walls, so a tube cannot be excluded for being
soft.

**This one is reasoned, not measured, and it is the only bottle rule that is.**
The toiletries family as a whole is emphatically bottle's — `soap` 47,
`lotion` 12, `dish soap` 11, `hand soap` 10, `shampoo` 7, `spray bottle` 6,
`spray` 4, `conditioner` 3, `detergent` 3, **110 boxes, and not one of them on
`cup`, `bowl` or `vase`**. But the *tube shape specifically* barely appears:
`tube` 1, `toothpaste tube` 1, `tooth paste tube` 1, `toothpaste` 1,
`caulking tube` 1, `shower gel` 1. Six boxes cannot tell you whether COCO's
annotators declined to call a tube a bottle or simply never met one, and the
fold-in cannot separate those — the same blindness that made the depiction count
useless (#3614). Treat this as the rule that would be cheapest to reverse.

It is the same shape as the jerry can: pool consistency says Bad, since COCO has
no tube class and such an image sits in the negative pool, while manufacture
says Good. That was already settled in favour of manufacture.

**The exception proves the rule about contents.** `sauce` goes the other way —
38 boxes on `bowl`, 18 on bottle, 16 on cup — because it names what is *in* the
vessel, not the vessel. A squeeze bottle of sauce is a bottle; a dipping dish of
it is a `bowl`. That is "judge the container, not its contents" doing visible
work: the same word, three classes, decided every time by the vessel.

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

**A standalone fuel container is a bottle** — a jerry can, a propane cylinder
off the barbecue. Nothing owns it, so the component test that excludes a
motorcycle's tank does not reach it, and what is left is a free-standing vessel
made to store and pour a liquid. The tank and the can are the two sides of that
line, and they are the clearest illustration of it in this guide.

Worth knowing what this costs, because it is the first ruling here where the
component test **beats** pool consistency rather than agreeing with it: COCO has
no class for a jerry can, so an image holding only one sits in the shared
negative pool, and a Good vote contradicts it exactly as a toy would. The saving
grace is frequency — `tank`, `fuel`, `gas`, `propane`, `barrel`, `drum` and
`keg` appear **zero** times in 9,169 bottle boxes, so you will rarely be asked.

### `vase incl pots and planters`

**Only a vessel made as one.** Vases, flower pots, planters, urns and pottery
count — 105 VG `pot`, 19 `planter`, 19 `flower vase` and 18 `urn` boxes land on
COCO vase boxes. Vote on the **vessel, not the plant**: a potted plant's pot is
a vase here, even though COCO separately has a `potted plant` class for the
greenery above it. A cooking pot on a stove does not count, and neither does a
plain bowl.

**"Vote the vessel, not the plant" does not mean "find whatever holds the
flowers."** A table with a pile of flowers on it is not a vase. A basket of them
is not a vase. Neither is the **florist's bucket** holding three dozen
individually wrapped roses — a bucket is made as a bucket, which is the same
call `bowl` makes about a bucket of apples (`bucket` lands on vase twice).

**A planter built into the pavement is not a planter.** A freestanding pot,
planter or urn is a vase; a concrete bed cast into the sidewalk and holding
trees or bushes is part of the street, and *an integral component of a larger
object is not an instance of a container class* — the fuel-tank rule from
`bottle`, arriving here. This narrows the plain word "planters" above, so read
the two together: **freestanding** planter yes, **built-in** no.

The ladder agrees independently, which is worth noticing because it was measured
for a different purpose: a sidewalk bed is low and broad, so its box is wider
than tall and lands in `bowl` territory rather than vase's. Two unrelated tests
giving one answer is the strongest signal this guide offers. Both hold flowers; neither was made to, which is the
containing-is-not-being rule from the protocol arriving here. Unlike the
borrowed-vessel narrowing below, this one costs nothing measured — COCO's
annotators never do it, putting `table` on 1 vase box and `basket` on 1, out of
2,328. It is written down not to correct COCO but to stop a reviewer
over-reading our own rule.

**A borrowed vessel is not a vase**, however it is being used in the picture. A
jar of cut flowers is a `bottle`; a drinking glass of them is a `cup`. This is
the ruling that keeps `jar` from meaning two things at once — see `bottle` — and
it is applied on *intent of manufacture*, the same test that keeps a planter
wall out of `bench`.

**A pitcher or jug of flowers is a `bottle`.** COCO could not say which class
either belongs to — `pitcher` lands on `cup` 30 times and `vase` 18, while `jug`
lands on `bottle` 28 and `cup` 0, two names for nearly one object sent to two
classes — so the split is made on portion instead: a cup is hand-held and a
single serving, and a pouring vessel that serves several is a bottle. It costs
the 30 `pitcher` boxes COCO called a cup and the 18 it called a vase.

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
land on COCO chair boxes, with 8 more under `arm chair` and 7 under `recliner`.
Benches do not count. A single seat within a row of stadium, theatre or
waiting-room seating counts, and `seat` (206) plus `seats` (63) is the third
largest fold-in here, so that case is common.

**One seat, one Chair; two or more, a couch.** COCO has a separate `couch`
class, and the 37 `couch` and 16 `sofa` boxes on COCO chairs are its errors, not
its rule. But *upholstered* is not the test — a single-seat upholstered piece,
club chair or recliner, is a Chair, which is what `armchair` (50 across both
spellings) and `recliner` (7) already say. A true two-seat loveseat is a couch by
the same head-count that separates Chair from Bench. `love seat` appears once in
15,868 boxes, so this is a rule for the reviewer's benefit rather than a
measured boundary.

**A lifeguard station is a Chair** — built as seating for one, which is the test
`bench` already uses. Measured at exactly 1 box, so it is decided by the
principle, not the data.

**A toilet is not a Chair.** COCO has a separate `toilet` class and the two are
never confused: `toilet`, `commode` and `urinal` appear **zero** times in 15,868
chair boxes. Free.

**A car seat is not a Chair, and that is the important one.** Sam's argument is
the reductio: *every* car almost certainly has seats inside, so counting them
makes Chair a class that fires on every street scene, stops discriminating, and
poisons the shared negative pool everywhere a car appears. Three things agree:

- **COCO never does it** — `car seat` 2, `child seat` 1, `car` 1, out of 15,868.
- **A component is not an instance.** A car seat is part of the car, the same
  test that keeps a concrete bed cast into the pavement out of `vase` and a fuel
  tank out of `bottle`.
- **The pool.** Cars are everywhere in VG, so this is the one exclusion here
  whose absence would be catastrophic rather than merely wrong.

The corollary, as with the jerry can: **a car seat removed from the car** —
sitting in a garage, a scrapyard, a skip — is free-standing, nothing owns it,
and it is a Chair.

**A headrest is not a Chair either — and the reason is not that it is small.**
Seeing part of a thing is normally good evidence of the thing: a chair back over
a table, a leg under it, and you box the Chair. VG names COCO chair boxes by a
part often enough to matter — `back` 57, `cushion` 34, `legs` 13, `backrest` 7.
So the rule is **a part inherits the ruling of its whole**: a headrest on a
dining chair is evidence of a Chair and you box the Chair; a headrest in a car is
part of a car seat, which is part of a car, and neither of those is a Chair, so
neither is the headrest. (`headrest` itself: 2 boxes in 15,868.) Like `car`, this
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
