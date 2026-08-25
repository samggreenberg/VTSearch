# 2026-08-24 — a rebuild silently retired the images a human had just reviewed

**Cost:** ~1 hour of the owner's labelling and a full model triage pass (2,698
tiles) were inert for a day, and were only recovered because the selection was
reproducible. Two rebuilds' worth of confusion before the cause was found.

## What happened

`vg_scale` designates each cell explicitly: a fixed number of positives per
`(class, band)` plus one shared pool of negatives, drawn with
`rng.sample(candidates, k)`. That is deterministic — same list, same seed, same
draw — but **not stable**: `rng.sample` re-derives the whole selection from the
list it is given, so *any* edit to the candidate list reshuffles every cell.

Between the slates being generated and the review being ingested, the candidate
list changed three times for good reasons (a coordinate-space fix, an
aspect-ratio guard, then the corrections themselves). Each rebuild silently
drew a different 3,900 negatives. By the time anyone checked, **577 of 743
reviewed images were no longer in the dataset at all** — not corrected, not
excluded, simply not drawn this time.

Nothing announced it. Every check still passed: 36 cells at exactly 100
positives, prevalence 0.0250 everywhere, boxes agreeing with their bands,
patch grids present, media counts consistent across embedders. A dataset whose
review covers 20% of it looks exactly like one whose review covers all of it.

## Why the first fix was not enough

Switching to hash-stable ranking (`sha1(cell:image_id)`, take the first N) is
the right long-term fix: adding or removing one candidate then changes only that
candidate's membership. But it was adopted *after* the drift, and adopting it
would itself have been a fourth reshuffle — the hash draw and the random draw it
replaced share ~228 of 3,900 negatives.

A roster was then captured to pin membership — but from the *current* pool,
which was already the post-drift one. Freezing the wrong state faithfully.

## What actually recovered it

The selection is a pure function of (labels, seed), and the labels of any past
moment are recoverable from git. Checking out the commit that generated the
slates (`797edfc94`), pointing `VTS_CORRECTIONS` at a nonexistent path so the
later corrections could not perturb the labels, and re-running the load
regenerated the original pool **exactly**: 100% of the 743 reviewed images and
100% of the 1,742 triaged ones.

## Prevented vs. still advice

* **Prevented:** selection is now hash-stable, and a roster file pins membership
  and is rewritten on every build, so a review keeps covering what it reviewed.
* **Still advice:** *before* a rebuild, check what fraction of reviewed images
  the new pool still contains, and treat a drop as a failure rather than a
  detail. Coverage is not implied by any of the structural checks, and human
  review is the most expensive input in the pipeline — it deserves an assertion
  of its own.
* **Worth knowing:** deterministic and stable are different properties. If work
  is keyed to a sample, the sample needs to be stable under edits to its
  candidate list, not merely reproducible from a seed.
