# DocMarks v1 — the same corpus, with the marks drawn correctly

**2026-08-31.** Companion to [`report.html`](report.html) (open it in a browser —
GitHub renders raw HTML as source). Same sources, same 1,088 SPODS pages, same
generator as [v0](../2026-08-31-docmarks-v0/REPORT.md); what changed is the
extractor that decides where a mark *is*.

Regenerate with:

```
python scripts/experiments/docmarks/make_report.py \
    --corpus <corpus> --out docs/experiments/2026-08-31-docmarks-v1/report.html
```

**Question.** v0's verdict was "the data is right and the labels are not."
Issue [#3361](https://github.com/samggreenberg/VTSearch/issues/3361) named why:
not the identities, the *geometry*. Does fixing the geometry fix the inventory?

**Verdict.** Yes, and it moved a number nobody expected it to move.

## What was wrong

The area floor ran *inside* `mask_to_boxes`, before `merge_overlapping` ever
saw the components. A stamp's mask is not one connected component — it is a
ring, the text inside it, a broken arc where the ink did not take, one component
per pen stroke — and each of those is individually below the floor. So eleven
fragments of the dozen were deleted as speckle, the merge that exists to
reassemble them had nothing left to work with, and the one or two chunkiest
survivors were promoted to classes in their own right.

The clearest case is the class v0 shipped as `spods/stamp_00129_1`: **38
instances, median 77 px, every one of them the word "New"** — clipped out of a
three-line `Dy.Manager / NewEastZone / ☏-000666` rubber stamp that was never
boxed at all.

The fix is the order: **decompose, merge, then filter**, with the floor applied
to the merged group's *ink* rather than to each raw fragment. See
`scripts/experiments/docmarks/README.md` § "Masks to marks".

## What changed in the corpus

| | v0 | v1 |
|---|---:|---:|
| labelled marks | 2,096 | 2,054 |
| median mark, longest side | 276 px | **376 px** |
| p10 / p90 | 74 / 420 px | **267 / 509 px** |
| smallest labelled mark | — | **195 px** |
| median share of page | 0.76% | 1.23% |
| largest mark on any page | **45.95%** | **3.47%** |
| `text` pseudo-marks | 1,196 | **0** |
| classes with ≥10 instances | 36 | **44** |
| largest single component | 60 (2.9%) | 31 (1.5%) |

Three things to read off that table:

**The bimodal size distribution was an artefact.** v0 reported "a spike near
0.05% and a broad hump from 0.5% to 2%", and warned that any per-size analysis
should band rather than average. The spike was the fragments. v1's distribution
is single-humped: **84% of marks fall in one 256–512 px bucket**, nothing below
128 px, and the p10 moves from 74 px to 267 px. The banding advice was sound
advice about bad data, not a property of document marks.

**The 46% "largest mark" is gone, and not because of the size guard.** It was a
`text` mark — a ruled table whose borders welded the whole grid into one
component. SPODS' `text` mask is the page body, which is a property of the page
rather than a thing on it, so it no longer emits marks at all; it is recorded as
`meta["text_frac"]` (median 3.62% of the page, range 1.45–7.05%) and
`meta["text_components"]`. A `MAX_MARK_AREA_FRAC` tripwire went in alongside,
but nothing in SPODS trips it — the largest surviving mark is 3.47%. It is there
for the next source.

**`signature` is now the whole negative control**, and it is a better one: 1,088
marks, one per page, every page signed, never promoted to a query class. That
was always the honest half of what v0 called its "documented negative control".

## The number nobody expected to move: the clustering threshold

v0 set `CLUSTER_THRESHOLD = 0.16` off a sweep, with the largest component pinned
at 60 marks (2.9%) from 0.08 through 0.16. That reading was correct for the
marks it was taken on. It is now wrong, and badly:

| threshold | classes | largest component | share | classes with ≥10 |
|---:|---:|---:|---:|---:|
| 0.02 | 1,261 | 31 | 1.5% | 31 |
| 0.08 | 800 | 31 | 1.5% | 40 |
| **0.10** | **672** | **31** | **1.5%** | **44** |
| 0.12 | 523 | 166 | 8.1% | 47 |
| 0.14 | 428 | 354 | 17.2% | 43 |
| 0.16 | 310 | **653** | **31.8%** | 36 |
| 0.22 | 138 | 1,288 | 62.7% | 22 |

At the inherited 0.16, single linkage now chains **653 of 2,054 marks — 31.8% of
the corpus — into one class**, while reporting a perfectly plausible 310
classes. This is v0's own lesson recurring one level up: *read the largest
component share, never the class count.* Had the geometry been fixed without
re-reading the sweep, the corpus would have shipped with a third of its stamps
in a single class and nothing on the surface saying so.

The mechanism is simple once stated. **The threshold is a property of the
marks.** The descriptor now hashes whole stamps rather than the one chunkiest
fragment of each, and whole stamps are a different object: they are larger, they
share more structure with each other (every rubber stamp has a border and a
block of text), and their pairwise distance distribution is correspondingly
tighter. The new sweep is cleanly bimodal — a within-class mode below 0.06, the
between-class bulk above 0.18, a real valley between — and 0.10 is the top of
the flat region by the same rule that picked 0.16 before.

That the class count moved the *right* way (36 usable classes → 44) is not what
tells you this; it would have read as an improvement either way.

## What this does not fix

Nothing here is verified. All 44 classes are clustering proposals with
`membership_verified = false`, exactly as in v0 — the human audit pass has still
not run, which is the point of doing this first: no adjudication has been
recorded yet, so none was invalidated.

`spods/stamp_00129_1` still exists as a class *id* — ids are derived from the
anchor page number, so the name recurs. It is now 12 instances at median 330 px,
and it is the complete three-line stamp.
