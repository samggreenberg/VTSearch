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

### And when two definitions are both defensible, take the one that labels clean

The table above prices a change against the *reference*. It says nothing about
the other cost, which is usually the larger one: **a definition that is hard to
apply produces noisy labels, and noisy labels are worse than an odd boundary.**
So where the taxonomy leaves a choice — and it usually does — the tie-break is
*which definition can a reviewer apply quickly and repeatably at three in the
afternoon on the two-hundredth image*.

That is the thread through most of what is here, and it is worth seeing as one
idea rather than nine separate rulings:

- **A car seat is not a Chair** partly because otherwise every street scene
  becomes an agonised judgement call.
- **A part inherits the ruling of its whole** replaces "is this fragment enough?"
  with a lookup.
- **The vessel ladder** turns an unanswerable semantic question — empty Vase or
  ornamental Bowl? — into a glance at proportions.
- **Judge the object in the box** removes the standing invitation to hunt for a
  better example.
- **"Neither", for a handle with no evidence either way**, is cleaner than a
  coin-flip carrying a 4:1 prior.

**The scored subset is the measurement of this.** Agreement against COCO on the
fifth of each slate that has an answer is not only a check on the reviewer; it is
a check on the *definition*, because a rule that is hard to apply shows up as
disagreement. A class that scores 98% is telling you its rule is mechanical. One
that scores 89% is telling you something is still being decided image by image —
which is exactly what `bottle` was doing while its boundaries were being widened.
Read that column as a labelability score and it earns its keep twice.

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

### Van against SUV — the only hard boundary left

**An SUV is not the problem. `van` is.** The measurement separates them cleanly:

| VG name | → `truck` | → `car` | reading |
|---|---:|---:|---|
| `van` | **261** | **318** | **45/55 — a coin flip** |
| `suv` | 62 | **222** | 78% Car |
| `minivan` | 5 | **51** | 91% Car |
| `mini van` | 1 | **17** | 94% Car |
| `jeep` | 25 | 44 | 64% Car |
| `pickup` / `pickup truck` | **51** | 15 | 77% Truck |

`suv` is not ambiguous, just noisy — an SUV is a **Car**, and so is a minivan.
Nothing in the data supports agonising over them.

**`van` splits because the word names two different vehicles.** A passenger
minivan and a panel cargo van share a noun and nothing else, so COCO's
annotators land 45/55 not because the boundary is unknowable but because they
were not applying one test. We are.

> **Was it built to carry goods, or to carry people?**
> Goods → Truck. People → Car.

**Cargo space is the cue, not the definition** — a distinction that matters,
because it fails at both ends. A **bobtail tractor unit**, cab with no trailer,
has no cargo compartment at all and is plainly a Truck; that is what it was
built to do, and the fifth-wheel coupling behind the cab says so. A **car with a
tow hitch** can haul and is still a Car; a hitch is an accessory, not a purpose.
The cargo test as first written got both backwards, and it contradicted a rule
sitting six lines below it — *"a cab-only tractor unit counts"* — which should
have been the tell.

**This is intent of manufacture, the same test as everywhere else in this
guide** — a Vase is a vessel *made as one*, a Bench is *built as seating*, a Bowl
is *made to hold food*. Vehicles are no different, and reaching for a visible
proxy instead of the principle is what produced two wrong drafts in a row.

What to look for, still from outside and without squinting:

| built for goods → **Truck** | built for people → **Car** |
|---|---|
| open bed, box body, panelled sides | glazed passenger body all the way back |
| fifth-wheel coupling behind the cab | body that simply ends behind the cabin |
| tipper, tanker, flatbed, ambulance | seats and windows, any number of rows |

**Accessories change nothing either way.** A tow hitch, a roof rack, a bike
carrier or a loaded roof box on an estate leave it a Car. A tractor unit with
its trailer dropped is still a Truck.

An earlier draft of this said *"windows all round **with seats behind the
driver**"*, and that was wrong twice over. It threw out every two-seater — a
sportscar has no back seat and is obviously a Car — and it asked the reviewer to
squint through glass to count rows, which is the opposite of a usable test. The
data is unanimous on the first point: `sports car` 0 → Truck and 3 → Car,
`convertible` 0/4, `coupe` 0/2, `hatchback` 0/6, `limousine` 0/1. **Not one
two-seater body in the whole overlap is a Truck.**

So the question is not seats, it is **cargo**, and it is answered from outside
the vehicle:

- **Sportscar, coupe, convertible** — built to carry people; the body ends
  behind the cabin. **Car.**
- **Bobtail tractor unit** — no trailer, no cargo space, built for nothing but
  hauling. **Truck** (`semi` 9 → Truck, 0 → Car).
- **SUV, estate, passenger minivan** — glazed passenger body all the way back.
  No cargo compartment. **Car.**
- **Cargo or panel van** — blank metal where the rear side windows would be.
  **Truck.** (`cargo van` 1/0, `delivery van` 2/0.)
- **Pickup** — separate open bed. **Truck.** (`pickup` + `pickup truck`, 51/15.)
- **Ambulance** — a box body behind the cab. **Truck**, and emphatically so:
  19 → Truck, 0 → Car.
- **Motorcycle** — no enclosed body at all, and COCO has its own `motorcycle`
  class. Neither.

### Car against Bus — barely a boundary

`bus` is one of COCO's 80 and one of the shipped twelve, so it is a live class,
but Car and Bus hardly touch: `car` lands on a COCO bus box **15** times out of
3,074, and `bus` lands on a COCO car box **19** out of 21,945. Around half a
percent each way. **Do not spend time here.**

Bus's real confusions are elsewhere, and one of them matters to us: `truck`
lands on bus 29 times and `bus` on truck **60**, while **`van` lands on bus 37**.

**Which makes `van` a three-way word, not a two-way one.** It splits 261 → Truck,
318 → Car, and 37 → Bus, because it names a cargo van, a passenger minivan *and*
a minibus shuttle. `cab` (22 Truck / 45 Car) and `van` are the two words in this
whole vocabulary that name genuinely different vehicles, and `van` names three.

**A taxi is not the hard case either, despite feeling like one.** `taxi` lands on
a COCO bus box **once**, against 62 on car; `cab` once against 45; `taxi cab`
never, against 11. Being for hire does not move a saloon anywhere.

**Because service is not a property of the object — and this is the jar rule
again.** A jar holding cut flowers is a Bottle, because `Vase` is reserved for a
vessel *made as one* and a borrowed vessel keeps the class it was built as. A
minibus running as a shared taxi is a **Bus** for exactly the same reason, and a
saloon with a roof sign and a meter is a **Car**. Livery, fare and route are
borrowings; **built-as beats used-as**, here as everywhere else in this guide.

So the taxi question collapses into the minibus one: it is the *body* that was
ever in doubt, and the taxi livery was never the variable.

The case that needs a call is therefore the **minibus or shuttle**, not the
saloon. The visible cue, consistent with everything else here: a Bus is boarded —
**a passenger door separate from the driver's position, and rows of side windows
running down a body longer than any car's** — where a Car is entered, with a door
per seating row. Unmeasured beyond the 37 boxes above, so treat a genuinely
borderline shuttle as the coin-flip it is rather than agonising.

**One more two-vehicles-one-word case, for the same reason as `van`:** `cab`
splits 22 → Truck and 45 → Car, because it names both a taxi and the tractor
half of an articulated lorry. Read the body, as ever.

**Yes: the same shell is a Car with glass and a Truck with panels.** That is
exactly what we are saying about a minivan, and it is not a quirk of our rule —
it is why COCO's own `van` splits 261/318. A passenger minivan and a panel van
are two vehicles built for two jobs, and the rear side panel is the one place
that difference is visible. `minivan` 5/51 and `cargo van` 1/0 are the same fact
read from both ends.

**The word was never the question — the body is.** This is the vessel ladder in
another guise: replace an unanswerable naming argument with one thing you can
see, from outside, without squinting.

### The cargo test assumes a vehicle. Check that first.

A bike trailer has a cargo compartment and is not a Truck; so does a handcart, a
skip, a wheelie bin. The cargo test only runs **after** a prior one:

> **Is it a self-propelled road vehicle, or the powered unit of one?**
> No → neither Car nor Truck, whatever it is carrying.

That covers a bike trailer, a detached semi-trailer, a handcart, a caravan under
tow. It also keeps the existing ruling that a **cab-only tractor unit counts** —
it is the powered half — while the trailer alone does not.

**Not "does it have a cab", though, and the difference matters.** A cab is a
proxy that fails on a forklift (COCO: 2 → Truck) and on anything with an open
driving position. Self-propulsion is the property actually being tracked.

**A tuk-tuk is a vehicle, so the cargo test decides it** — passenger cabin →
Car, flatbed or box → Truck. Note this is entirely unmeasured: `tuk tuk`,
`rickshaw`, `three wheeler` and `moped` appear **nowhere** in the overlap, and
COCO's own `motorcycle` class may well be where its annotators would put one.
Treat it as a principled call on a case you will probably never meet.

**What this narrowing costs, honestly.** COCO is looser here than we are:
`trailer` lands on a COCO `truck` box **63** times and `cart` **35**, plus
`tractor` 24, `forklift` 2, `dolly` 1. So excluding towed and pushed things is a
real disagreement, not a free one — of the same order as the vase narrowing.

And a correction to this guide: it previously said COCO boxes trailers as truck
*"only when the cab is attached"*. **Those 63 boxes do not support that** — it
was an interpretation, and the measurement cannot tell an attached trailer from
a detached one. The ruling stands on the vehicle test above; the claim about
COCO's reasons has been withdrawn.

Expect to disagree with COCO on vans regardless, because half its own `van`
boxes went the other way. That is a known, priced cost of this class, not a
mistake — and it is why the disagreement column for `truck` will read worse than
`fire hydrant`'s without meaning the reviewer did worse.

### `truck incl vans not SUVs`

Pickups, box trucks, semis and tractor units, flatbeds, tow trucks, fire trucks
and food trucks all count, as do full-size cargo and panel vans. SUVs,
crossovers and passenger minivans do **not** — those are `car`. The van boundary
is the one COCO is itself inconsistent about, and the inconsistency is
measurable: 261 VG `van` boxes sit on COCO `truck` boxes and 318 sit on COCO
`car` boxes, so COCO's annotators split vans roughly evenly. Use the body, not
the badge: a separate cargo box, or no rear side windows, makes it a truck; a
passenger minivan with seats and windows all round is a car. A cab-only tractor
unit counts, as the powered half; a detached trailer does not, and neither does
a handcart or a bike trailer — see the vehicle test above, and note it costs us
the 63 `trailer` and 35 `cart` boxes COCO does call trucks. Expect
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

### Cup against Bowl

**No single object is both.** An image can hold one of each — and two do, in the
completed slates, with the boxes disjoint at IoU 0.000 — but a given vessel is a
Cup or a Bowl and never both.

**If it is full of soup, it is a Bowl**, whatever its silhouette. **If it is
empty, the geometry decides**: the ladder puts Bowl at a median h/w of 0.66 and
Cup at 1.26, with the middle halves not overlapping, so "best judgement" on an
empty vessel is really "is it wider than tall".

**This does not contradict "judge the vessel, not the food" — the two answer
different questions.** *Judge the vessel* answers **what to box**: the vessel,
never the contents, so an empty plate counts and a pile of food on a bare table
does not. *Soup means Bowl* answers **which class**, in the case where the
vessel alone underdetermines it. Contents never make something a member; they
only break a tie between two classes it could belong to.

It is the same move as `spoon`'s handle-in-the-cereal and `fork`'s stab-grip:
**when the object's own features run out, read what is around it.** Three rules,
one idea, and all three exist because the alternative is a coin-flip.

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

**The same goes for a motorcycle's seat, and for a saddle.** A motorcycle
almost always shows a little single-person seat, so admitting them would do to
motorcycles what car seats would do to cars; and a saddle is tack, part of the
horse's kit rather than free-standing furniture. Free, as with the toilet: the
whole family — `saddle`, `motorcycle seat`, `bike seat`, `motorcycle` — appears
**zero** times in 15,868 chair boxes, with a single stray `scooter`.

The corollary, as with the jerry can: **a car seat removed from the car** —
sitting in a garage, a scrapyard, a skip — is free-standing, nothing owns it,
and it is a Chair. The same for a saddle on a rack in a tack room, which is a
harder call and much rarer.

### Someone is clearly sitting, and the Chair is invisible

Vote **Good, and draw no box** — on an image that arrived **without** one.

**The stratum decides what "no box drawn" means, and the two meanings are
opposite.** On a `positive_boxed` image a box already arrived, so a Good that
draws nothing *confirms that box* (`positive_confirmed`) — which is the normal
action, and what happened for 27 of every 30 positives confirmed so far.
Drawing on a pre-boxed image is the exceptional case (`positive_reboxed`), and
it is the one that moves bands. Everything below is about a **bare** image, from
the `boundary` or `random` stratum, where no box arrived at all.

There, it is the one place the three-valued design is reachable from the voting
UI, and it is exactly right. A Good *with* a box says "present, and this is its
size". A Good *without* one says **"present, but no size was measured"** — and
`verdicts_to_corrections` files it as `negative_excluded`: `present: True`,
`boxes: []`. The image is taken **out of the shared negative pool**, because a
Chair really is there and scoring a detector wrong for finding it would be
absurd, and it is **not made a positive**, because a band is a claim about size
and you did not measure one. Neither, precisely.

Do **not** box where you think the Chair is. An invented box is an invented
size, in a study whose entire subject is size.

**One caveat, and it is a real one.** A boxless Good cannot distinguish *"I
could not see it"* from *"I forgot to draw"*. Both land in the same state. On
2026-09-03 eight boxless Goods in `cell phone` were the second kind and had to
be re-issued to recover them (#3616 neighbourhood); here they are the first kind
and want no recovery at all. The pipeline cannot tell, so the reviewer has to —
which is the argument in #3643 for making the third state something you can say
out loud.

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

**A fire truck is not a Hydrant.** Obvious said aloud, easy to slip when the
whole slate is red street furniture, and a fire truck is a `Truck` — one of the
thirteen — so the image is very likely a positive for a *different* class here.

**Busted street plumbing, gushing water, and you cannot see what broke, is not a
Hydrant.** This is the third distinct kind of "cannot tell" in the guide, and
they resolve differently. The rule that separates them:

> **Good-with-no-box requires certainty of PRESENCE.** It says *"it is there, I
> cannot measure it"*. It never means *"something might be there"*.

- **Sure it is present, cannot measure it** — a person clearly sitting on an
  unseen Chair. → **Good, no box.** Excluded from the class: not a positive, and
  out of the negative pool.
- **Cannot tell whether it is present at all** — a sheared pipe under a plume of
  water. → **Bad.** You cannot confirm what you cannot see, and Bad is what the
  protocol has.
- **Sure something is present, cannot tell which class** — a buried utensil
  handle with no cereal or salad to read. → **Bad for both**, which is Sam's
  "neither".

- **Sure something was there, but it has no locatable extent** — a long-exposure
  night shot where traffic is nothing but streaks of headlight. → **Good, no
  box.** New in kind: the object is not occluded and not ambiguous, it is
  *smeared across the frame*, so no box could be correct even in principle. A
  band is a claim about an object's size and a three-second trail has none.

That last one looks like it should be Bad — you cannot say whether any given
streak was a Car or a Truck, which is the third case above. It goes the other way
on **asymmetric regret**, and the principle is worth stating on its own:

> **When the two errors cost very different amounts, take the third state.**
> Wrongly excluding an image costs one image of supply. Wrongly filing a
> photograph of traffic as *confirmed no-Car* puts a falsehood into the shared
> pool that will score every model built on it afterwards — and a detector that
> fires on a night traffic shot is not obviously wrong.

The second and third file the image as a confirmed negative, which is stronger
than the reviewer means, and that is the cost recorded in #3643. A sheared
hydrant under a plume is a real thing and the pool will now say there was none —
but no vote available today says otherwise honestly.
