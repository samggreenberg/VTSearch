# VG's silence rate, off the exhaustive pass's own answers (#3696)

**2026-09-13, 9 of 25 classes finished.** How often VG fails to name a class
that is really in the picture — the rate that justified `vg_scale`'s negative
pool composition (#3670) and that the composition then made unmeasurable, by
drawing all 9,900 shared negatives from the COCO-scored half.

`scripts/experiments/pile/silence_rate.py` reads it off the exhaustive pass
instead, at no extra labour: every class VG did **not** name in a queue image is
a measurement of its silence against a human reference.

```bash
python silence_rate.py --deep-unprovable 6264 --out silence_rate.json   # ~20s on a login node
```

| | |
|---|---|
| silent `(image, class)` pairs in the queue | **81,363** over 3,397 images |
| …over the 9 finished, rule-current classes | **29,414** |
| silence errors a human confirmed | **815** |
| **measured rate** | **2.8%** [2.6%, 3.0%] |
| **upper bound** | **3.5%** |
| …if the below-cut samples are not pooled | 6.0% |
| `vg_scale_deep`: of its 6,264 unprovable negatives (#3723) | **at most 219 contaminated** |

For comparison, #3666 measured the shipped twelve's pool error at 1.40%
[0.68, 2.86]. This sitting ~2x above it is the expected direction, not a
disagreement — see the conditioning below.

## Per class

`cand` is what cleared the class's screening cut; `seen` is what a human has
voted on (it can exceed `cand`, because the earlier `pass25` review reached
below-cut images too).

| class | silent | cand | seen | found | rate | 95% CI | bound | |
|---|---:|---:|---:|---:|---:|---|---:|---|
| bottle | 3,243 | 822 | 0 | 0 | 0.0% | [0.0%, 0.1%] | 25.9% | not started |
| chair | 3,212 | 841 | 300 | 219 | 6.8% | [6.0%, 7.7%] | 25.1% | 541 unreviewed |
| cell phone | 3,248 | 687 | 0 | 0 | 0.0% | [0.0%, 0.1%] | 21.8% | not started |
| cup | 3,246 | 676 | 0 | 0 | 0.0% | [0.0%, 0.1%] | 21.4% | not started |
| truck | 3,255 | 674 | 0 | 0 | 0.0% | [0.0%, 0.1%] | 21.3% | not started |
| bird | 3,254 | 645 | 0 | 0 | 0.0% | [0.0%, 0.1%] | 20.5% | not started |
| clock | 3,240 | 632 | 0 | 0 | 0.0% | [0.0%, 0.1%] | 20.1% | not started |
| car | 3,206 | 588 | 0 | 0 | 0.0% | [0.0%, 0.1%] | 19.0% | not started |
| bowl | 3,238 | 592 | 0 | 0 | 0.0% | [0.0%, 0.1%] | 18.9% | not started |
| umbrella | 3,243 | 571 | 0 | 0 | 0.0% | [0.0%, 0.1%] | 18.3% | not started |
| spoon | 3,272 | 541 | 0 | 0 | 0.0% | [0.0%, 0.1%] | 17.2% | not started |
| knife | 3,279 | 465 | 0 | 0 | 0.0% | [0.0%, 0.1%] | 14.8% | not started |
| kite | 3,247 | 423 | 0 | 0 | 0.0% | [0.0%, 0.1%] | 13.7% | not started |
| **bench** | 3,272 | 1,231 | 1,289 | 178 | **5.4%** | [4.7%, 6.3%] | 6.7% | |
| fork | 3,251 | 396 | 396 | 134 | 4.1% | [3.5%, 4.9%] | 5.4% | stale rule |
| **book** | 3,271 | 397 | 397 | 131 | **4.0%** | [3.4%, 4.7%] | 5.3% | |
| vase | 3,252 | 394 | 394 | 116 | 3.6% | [3.0%, 4.3%] | 4.8% | stale rule |
| **bicycle** | 3,273 | 169 | 886 | 102 | **3.1%** | [2.6%, 3.8%] | 4.2% | |
| **bus** | 3,249 | 327 | 327 | 89 | **2.7%** | [2.2%, 3.4%] | 3.9% | |
| **backpack** | 3,285 | 979 | 979 | 90 | **2.7%** | [2.2%, 3.4%] | 3.8% | |
| **sink** | 3,256 | 168 | 268 | 77 | **2.4%** | [1.9%, 3.0%] | 3.5% | |
| **boat** | 3,237 | 141 | 241 | 65 | **2.0%** | [1.6%, 2.6%] | 3.1% | |
| dog | 3,263 | 73 | 173 | 59 | 1.8% | [1.4%, 2.3%] | 2.9% | stale rule |
| **fire hydrant** | 3,247 | 166 | 266 | 47 | **1.5%** | [1.1%, 1.9%] | 2.5% | |
| **stop sign** | 3,324 | 308 | 408 | 36 | **1.1%** | [0.8%, 1.5%] | 2.1% | |
| **POOLED** | **29,414** | | | **815** | **2.8%** | **[2.6%, 3.0%]** | **3.5%** | 9 of 25 |

Bold rows are the nine in the pooled figure. **Silence varies 5x across the
nine** — `bench` 5.4% against `stop sign` 1.1% — so a single pooled number is a
budget, not a description of any one class.

## What the number is, and is not

**An upper bound on the *uniform* off-COCO rate, not an estimate of it.** The
queue's images are selected for holding a class of *C*, so they are cluttered
scenes, and clutter correlates with holding more classes (#3667 excluded 3,247
images for holding a *different* class; #3679 measured the scene-clutter
shortcut at 2.5x `@small`). Silence measured on cluttered scenes is at least as
high as silence on a uniform draw.

Three things push the other way, and the script counts rather than argues each:

- **Screening.** The pass is screened (#3760), so a positive below a class's cut
  was never shown to anyone. #3768's 600 random below-cut images found **zero**,
  which bounds that mass at 0.64% pooled — that is the gap between the 3.0% CI
  top and the 3.5% bound. The pooling is an assumption (six sampled classes
  speaking for nineteen) and `bound_unpooled` prices it at **6.0%**.
- **The question asked.** A slate asks *is this box one?* of the screen's best
  box, so a second instance the detector never boxed reads as absent. Only
  #3768's below-cut slates ask the wider image question.
- **Coverage.** `chair` is banked at 300 of 841, so its 541 unreviewed
  candidates enter its bound at their worst case — which is why it reads 25.1%
  and stays out of the pool. The twelve unstarted classes read the same way, and
  that is the point: a class nobody has looked at must never read as clean.

**Three classes were voted under a rule that has since moved** — `dog`
(`dog` → `dog not wolves`, #3771), `fork` (`fork incl plastic` → `fork incl
sporks not strainers`), `vase` (`vase incl pots and planters` → `vase not
planters`, #3784). They are reported and kept out of the pool rather than
dropped. `vase` is the one to watch: the planter recheck retired **20 of 31** of
the *old* positives, and it does not overlap these 394 at all.

## What was given up

The original ask was a designated off-COCO stratum, which would have measured the
uniform rate rather than bounding it. It was dropped because it needs hand
annotation to be a reference and the pass supplies a bound free. If a decision
ever turns on the uniform rate specifically, this cannot supply it, and drawing
the stratum afterwards means a different annotator cohort under a different
protocol.

## Re-running

The number moves as the pass finishes classes, so re-run it rather than quoting
this file. `measurements/silence_rate.json` is this run's output; the script's
docstring carries the full rationale.
