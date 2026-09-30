# Does diversifying Autopilot's opening walk cure the sibling fixation?

**No, not by enough to ship.** The sibling fixation #4197 names is real, but it lives in
the **opening's text-sort walk**, not the learned Hard phase the issue proposed to
change. There, 17–38% of the opening's Bad clicks hold the class's sibling for bowl,
laptop and knife, against 0–3% in the Hard phase. The query itself surfaces the
sibling: "laptop" ranks keyboards high.

A knob that makes the walk pass over near neighbours of the Bads voted so far does
move those clicks, but only a little. Bowl's toilet share falls from 38% to 30%; the
other classes move by a point or two. Its effect on the session is small:

- **Cost at vote 150:** −0.0008 ± 0.0017 at the 0.44% pool (COCO Better's default)
  and −0.0063 ± 0.0031 at 0.1%.
- **AP at vote 150:** +0.0033 ± 0.0014 and +0.0027 ± 0.0016.
- **Positives found:** unchanged.

That passes the rule pre-registered on #4197, which had no minimum effect size. It is
about a third of the 0.01 cost change the issue set as its target, and shipping it
would need new app plumbing. **Owner's call (2026-09-29): don't ship.** The knob stays
an experiment-only harness option.

What did move these classes is the opening #4288 shipped. Positives found by vote 150
at 0.44%:

| class | old opening | shipped opening |
|---|---|---|
| laptop | 2.9 | 12 |
| tv | 3.1 | 10 |
| person | 3.1 | 8.6 |
| spoon | 3.6 | 7.9 |
| knife | 3.0 | 5.6 |
| bowl | 2.9 | 4.1 |

Pools are scenarios, not claims about users. Binary voting, SigLIP, COCO Better.

## Where the sibling Bads are

`analyze_siblings_4197.py` splits each session's Bad clicks by phase and counts those
whose largest COCO object is the class's sibling, the measurement `why.py` used to
name the siblings. Old opening (`g3@top,b4@mid`) against the shipped one, 720 sessions
each at 0.44% (`sibling_share_old_vs_shipped.csv`):

| class | sibling | share of opening Bads (old / shipped) | share of Hard-phase Bads (old / shipped) | opening Bad clicks per session (old / shipped) |
|---|---|---|---|---|
| bowl | toilet | 38% / 38% | 2.5% / 2.3% | 76 / 94 |
| laptop | keyboard | 25% / 25% | 1.2% / 1.4% | 60 / 84 |
| knife | scissors | 17% / 20% | 0.7% / 0.9% | 58 / 80 |
| tv | remote | 6.1% / 8.1% | 0.1% / 0.3% | 54 / 75 |
| spoon | knife | 4.8% / 4.8% | 0.4% / 0% | 65 / 96 |

These classes spend most of their clicks walking the text sort; the Hard phase barely
sees the sibling. (The "×49" figures in the issue are lift over the corpus, not
counts.)

## The knob

`CALIB_OPENING_DIVERSITY="<tau>/<k>"`, experiment-only
(`vtscore.eval.al_strategies._diverse_top`). While the opening walks the top of the
text sort (`good` / `more`), it passes over a candidate at SigLIP cosine ≥ tau to at
least k of the Bads voted so far, and takes the next one down.

The two settings were read off the embedding's own scale. A random pair of images has
median cosine 0.50, and 1% of pairs reach 0.78. Measured against the nearest earlier
Bad, the next opening pick sits at:

| next pick | median cosine to the nearest earlier Bad |
|---|---|
| a sibling Bad | 0.72–0.81 |
| another Bad | 0.69–0.82 |
| a Good | 0.65–0.75 |

- **`0.85/1` (strict):** pass over anything very close to any Bad.
- **`0.80/2` (cluster):** pass over only what sits close to two or more Bads, a
  repeated sibling rather than a one-off.

Off by default, so the default arm, and therefore the app, is unchanged.

## What was run

- Today's app on COCO Better, SigLIP, binary voting. 144 class@band cells × 5 seeds,
  150 votes each, at 0.44% and at 0.1% (positives thinned).
- Three arms, all on the same commit (cdf77d973): the default opening (control),
  `0.85/1` and `0.80/2`. 4,320 sessions.
- Read with `analyze_drystop_4222.py`: cost (the harness's weighted FNR/FPR at the
  shipped threshold), AP, and a per-class table. A session with no detector scores
  AP 0.
- **Rule, fixed on #4197 before the run:** an arm wins if, in both pools, its mean
  final cost at vote 150 is lower than the control's and neither cost nor AP is worse
  by more than 2 paired SE; nor may it raise the share of sessions with no detector by
  more than 2 SE. Among winners, take the larger mean cost gain.

## Results

Vote 150, all 720 cells per arm, paired against the control (`quality_paired.csv`).
The control's cost is 0.46 (0.44%) and 0.48 (0.1%), and its AP 0.51 and 0.36:

| arm | cost, 0.44% | cost, 0.1% | AP, 0.44% | AP, 0.1% | no detector (0.44% / 0.1%) |
|---|---|---|---|---|---|
| control | – | – | – | – | 2.5% / 13% |
| `0.85/1` | −0.0007 ± 0.0015 | −0.0059 ± 0.0021 | +0.0030 ± 0.0012 | +0.0016 ± 0.0014 | 2.1% / 12% |
| `0.80/2` | −0.0008 ± 0.0017 | −0.0063 ± 0.0031 | +0.0033 ± 0.0014 | +0.0027 ± 0.0016 | 2.6% / 12% |

- **By the rule, both arms pass.** `0.80/2` has the larger mean cost gain,
  −0.0035 against −0.0033, a tie in practice.
- **Positives found at vote 150** are unchanged: 19.7 at 0.44% in all three arms, and
  5.5 at 0.1%.
- **At vote 25** the arms are within 0.002 AP of the control.

**Sibling clicks in the opening** (`sibling_share.csv`), control → `0.80/2`:

| class | 0.44% | 0.1% |
|---|---|---|
| bowl | 38% → 30% | 38% → 26% |
| laptop | 25% → 24% | 26% → 24% |
| knife | 20% → 19% | 15% → 14% |
| tv | 8.1% → 7.4% | 7.4% → 6.5% |
| spoon | 4.8% → 4.4% | 3.5% → 3.5% |

Opening Bad clicks per session barely change (bowl: 94 → 92). The knob mostly swaps
one near-miss for another.

**Per class** at 0.44% (`class_paired.csv`), `0.80/2` against the control:

| class | positives found (control → arm) | cost | AP |
|---|---|---|---|
| laptop | 18 → 18 | −0.035 ± 0.022 | +0.048 ± 0.024 |
| bowl | 6.6 → 6.4 | −0.028 ± 0.019 | +0.014 ± 0.013 |
| person | 13 → 12 | −0.030 ± 0.027 | +0.024 ± 0.027 |
| knife | 8.5 → 8.0 | +0.011 ± 0.026 | +0.0007 ± 0.0084 |
| spoon | 13 → 14 | +0.016 ± 0.016 | −0.0053 ± 0.016 |
| tv | 12 → 12 | +0.0097 ± 0.0071 | −0.0026 ± 0.0021 |
| skis (best) | 40 → 38 | +0.014 ± 0.010 | +0.0030 ± 0.0053 |
| snowboard (best) | 38 → 38 | +0.0032 ± 0.012 | −0.0039 ± 0.0060 |

Only laptop comes near 2 SE, and the best classes wobble the other way (skis finds 2
fewer positives). With `0.85/1`, snowboard's cost rises by 0.031 ± 0.018.

## What this means

- **The fixation is the text query's, not the loop's.** A query like "laptop" puts its
  sibling class high in the text sort. Skipping the Bads' neighbours only reaches the
  next similar item.
- **What helped these classes was walking longer,** which found more positives in
  spite of the siblings (#4288).
- Anything further would have to change the *query*, for example suggesting a
  negative ("laptop, not keyboard") once a sibling cluster has been voted Bad. That is
  a different experiment, not filed here.

## Reproduce

```
# the #4222 launcher; arm <world>-div<TT>k<K> sets CALIB_OPENING_DIVERSITY=0.TT/K
VTS_REPO=<checkout> TEXTGOOD_BASE=<out> CALIB_MEM=2G \
  bash scripts/experiments/calibration/launch_textgood_4222.sh prepare natural-new natural-div85k1 ...
VTS_REPO=<checkout> TEXTGOOD_BASE=<out> CALIB_MEM=2G \
  bash scripts/experiments/calibration/launch_textgood_4222.sh arms natural-new natural-div85k1 ...
python scripts/experiments/calibration/analyze_drystop_4222.py \
  --arm 0.44%/g3=<out>/natural-new/results --arm 0.44%/g85=<out>/natural-div85k1/results ... --out <analysis>
python scripts/experiments/calibration/analyze_siblings_4197.py \
  --arm natural-new=<out>/natural-new/results ... --out <analysis>/sib
```

The analyzer's opening-ending table (`opening.csv`) reads the knob arms' labels as a
Good target and is not meaningful for them. The knob changes what the walk picks, not
when it stops.

Files here:
- `quality_paired.csv`, `class_paired.csv`, `harvest.csv`, `starved.csv`, `opening.csv`: the analyzer's tables.
- `sibling_share.csv`: the knob arms.
- `sibling_share_old_vs_shipped.csv`: old and shipped openings.
