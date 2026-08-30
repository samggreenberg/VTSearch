# Pre-registration — does the loop keep improving past 150 votes?

**Scored in [`RESULT-horizon-250.md`](RESULT-horizon-250.md)** — one expectation
right, one wrong, one half wrong.

Written before the 250-vote grid drained. The report's curves stop at 150 votes
because that is where the original grid stopped, not because anything about the
data or the method ends there. This run extends the horizon to 250.

## The question

**Does a user who keeps clicking past 150 votes get a better detector, and
where?** 100 extra clicks costs a person a couple of minutes, so the bar for
"worth doing" is low — but the report's own figures suggest the answer is
regime-dependent, and a flat answer either way would be worth knowing.

## What is *not* the constraint

Votes are drawn from the simulation half of each haystack: **419** medias on
`caltech101_m`, **2,096** on `visual_genome_m`, **2,476** on `coco_val`, **6,000**
on each `vg_box_*` band. At 250 votes the loop has seen 4 % of a box band's pool.
Running out of *images to click* is not a limit at this horizon.

Running out of **positives** can be, for the rarest categories: the simulation
half holds roughly half of a category's positive images — ~15 for
`visual_genome_m`/`cat`, ~25 for `ball`, ~24 for `coco_val`/`bear`, ~10 for
`caltech101_m`/`cougar_face`. Runs currently end on a median of 4–11 positives,
so there is headroom, but for those categories the ceiling is 10–25 positives
however long anyone clicks.

## Design

The same cells carried further, not a new draw: the run reuses each source
study's `prepare_info.json` and exemplar crops by symlink, so categories, seeds,
splits and the startup exemplar are identical, and only `CALIB_MAX_STEPS` moves
(150 → 250).

**Verified, not assumed.** On the sizing cell (`vg_box_small` × `siglip` ×
`glasses` × seed 0) all 148 overlapping steps are bit-identical to the original
run — `n_good`, `n_bad`, `threshold`, `cost`, `fpr`, `fnr`, `AP`, `AUROC` all
match to 0. Steps 151–250 are therefore a continuation of the published curves,
and any difference in the deep regime is the horizon, not a re-draw.

## What I expect

Stated before the grid drained, so the result can contradict it:

1. **Little on the box bands.** Their ramp is over by t≈60 — `vg_box_small ×
   siglip` improves 0.89 → 0.71 over the first 60 votes and 0.71 → 0.64 over the
   next 90 — so votes 150–250 should buy under ~0.02 cost. The one sizing cell
   available agrees: 10 → 12 positives, cost 0.769 → 0.783, AP 0.342 → 0.343.
   Nothing.
2. **Something on `visual_genome_m` and `coco_val`**, whose curves were still
   drifting down at 150.
3. **Most of it on the slow starters.** 6 % of runs reach vote 150 with two or
   fewer positives, and one found its first positive at vote 144 — those runs are
   currently reported as failures of the method when they are really failures of
   the horizon. If the gain concentrates there, the conclusion changes from
   "clicking plateaus" to "clicking plateaus *for runs that got started*", which
   is a different design brief: it makes the fix cold-start acquisition rather
   than more clicks.

## What would change the report

- **Cost at 250 lower than at 150 by more than 2 SE on a haystack** → that
  haystack's numbers get restated at the longer horizon, and "the ramp is over by
  t≈60" is wrong for it.
- **The stuck-run count falls** (deep-regime cost > 0.9) → the "a quarter of
  whole-image runs on sub-patch targets never work" finding is horizon-dependent
  and must say so.
- **Neither** → the report gains a sentence that the plateau is real out to 250
  votes, which is worth more than the 150-vote silence it replaces.

## Cost

Per-step cost rises with the vote count, because each Good vote adds patch rows
to the training set: a patch cell measures 3.7 s/step at votes 2–25 against
7.3 s/step at 126–150, so total cell time grows roughly quadratically with the
horizon. Measured on the sizing cells: a whole-image cell goes 1.9 min → 3.5 min,
a patch cell 15 min → ~1 h. The grid is ~500 cells, of which ~150 are patch
cells — a few hours of wall time at the usual concurrency, and the pile is
already embedded, so nothing is re-encoded.
