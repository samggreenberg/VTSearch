# FullMarks v3 — the roster corpus

**#3343 stages 3–5.** v2 built 200,000 pages and 65 candidate classes and said,
correctly, that none of it was ground truth yet: every class in it was a
clustering *proposal*. This is the pass that makes it ground truth — a
hand-picked roster of 24 classes, every instance of every one adjudicated in or
out, every pair of them adjudicated same or different, and the embedding cells
an eval runs on.

**The headline is not the roster.** It is that the audit the corpus already had
would not have survived being rebuilt, and stage 3 *is* a rebuild. Three
defects, each of which turns a hand decision into silent corruption on the next
build, were found by running that rebuild as a control before trusting it. All
three are fixed, and the control now reproduces the audited partition exactly:
**58 classes with byte-identical membership, 0 changed.**

A fourth came out of stage 5, and it is the same shape: `embed_corpus.py` could
never write a cell, and failed on the *last line* after 2h16m of embedding. The
common thread is that none of these paths had ever been run end to end, and each
of them fails in the direction that looks like success until much later.

## What was owed, and what this delivers

| stage | before | after |
|---|---|---|
| 3 — roster | 65 candidates, no roster | **24 classes**, 613 instances, no drift warning |
| 4 — membership | nothing examined | **613 crops examined, 0 rejected** |
| 4 — confusable | nothing recorded | **276 of 276 pairs adjudicated**, all `different` |
| durability | 2 rows, both ambiguous | **586 must-link + 291 cannot-link**, all mark-keyed |
| 5 — cells | none, and `embed_corpus.py` could not write one | tier `s` built and **verified**; `m` running; `l` priced out (#3842) |

> **Superseded counts (2026-09-17).** The completeness pass (#3927) grew the
> roster from 613 to **721** instances on the same pages (corpus v3.1). Counts
> here describe v3. Current composition:
> [`scripts/experiments/fullmarks/DATASHEET.md`](../../../scripts/experiments/fullmarks/DATASHEET.md).

## The roster is not the top of the ranking, and that is the point

`shortlist.py` ranks candidates on instances, size, tightness and separation.
Its top 24 are **21 SPODS logos and 3 Tobacco800** — because SPODS logos are
pasted digital artwork, so they are big, numerous and sharply clustered, and the
candidate pool is 71% SPODS to begin with. An eval drawn from that ranking would
substantially be measuring SPODS.

So the roster is picked by hand against the ranking:

| source | candidates | roster | share |
|---|---:|---:|---:|
| SPODS | 46 | 11 (5 logos, 6 rubber stamps) | 46% |
| Tobacco800 | 14 | 11 | 46% |
| StaVer | 5 | 2 | 8% |

Everything on it clears **n ≥ 8 instances**; the largest is 65. Three decisions
worth stating because they are judgement rather than measurement:

- **Plain shapes are left off.** `spods/logo_00001_0` is a red cross and
  `spods/logo_00020_0` an orange `H+` sign. "Find this rectangle" is not a
  well-posed retrieval query, and the prior study's worst classes
  (`warning_diamond`, `hospital_cross`) were exactly these. They stay in the
  corpus as candidates; they do not carry an eval.
- **The hard pairs are kept, not avoided.** Three Tobacco800 leaf marks
  (`ald41a00` / `azb11c00` / `asg54f00`), two chief engravings (`afm90c00` and
  `ciy01a00`, one a near-mirror of the other), and four SPODS script stamps that
  all read `…Secretary`. Telling those apart is the thing the benchmark exists
  to measure. By `siglip2_l` centroid the two leaf marks sit at **0.070** and the
  two chiefs at **0.136**, against a 24-class range topping out at 0.447.
- **Tobacco800 is over-weighted on purpose**, taking 11 of its 14 candidates
  against 11 of SPODS's 46.

## Three defects that would have deleted the audit on the next build

`build_corpus.py` re-clusters from the sources and writes `classes.json` from
scratch, replaying exactly one file: `adjudications.json`. So "applied" means
"applied until the next build" for anything that does not reach that file — and
stage 3 rebuilds the corpus on purpose, to stamp the roster.

### 1. A page id is not a mark, and every source here puts several marks on a page

`resolve_pairs` expanded an adjudicated `(page_id, page_id)` pair into *every*
row pair across the two pages' marks. A SPODS page carries a logo, a stamp and a
signature; a StaVer page carries two stamps.

The corpus carried one such row — the `DY.Secretary` merge answered in the
#3561 audit:

```
spods/00546  →  logo_00016_0 (27 instances) + stamp_00551_1
spods/00551  →  logo_00017_0 (29 instances) + stamp_00551_1
```

Replayed as a must-link it resolves to all four crossings, so the next build
fuses **two 30-instance logo classes and a stamp class into one ~76-instance
class**. That is an over-merge, which this corpus's own README says is the error
nothing downstream can see — written by the mechanism that exists to prevent
over-merges, with nothing warning.

Endpoints are now `(page_id, mark_index)`. A bare page id is still accepted
where it is unambiguous and **refused** where it is not, rather than guessed at.
`--migrate-adjudications` recovers the index for an existing store from the
class ids the rows already carry: on the real corpus, 4 endpoints resolved, 0
left ambiguous.

### 2. A `split` and a `membership` verdict never reached the store at all

`resplit_classes` and `apply_membership` wrote `classes.json` and the manifest
and nothing else. The two splits the #3561 audit applied — a five-mark StaVer
class and a four-mark SPODS one — would have come back fused, re-proposing the
exact over-merge a reviewer had just taken apart. Every membership rejection
would have been handed straight back.

Both now record the partition they assert: a must-link star inside each piece or
verified class, one cannot-link between each pair of pieces and one per
rejection. The asymmetry is sound rather than lazy — `single_linkage` applies
every must-link before any distance, so by the time a separation is registered
each group has already formed and a single row pins all of it.

The regression test is the argument in miniature: re-cluster the fixture at a
threshold measured to fuse its two marks, with the recorded rows replayed, and
the partition comes back; without them the control returns one class of six.

### 3. A hand-merged class is renamed by the next build

Found by the control rather than by reading: `spods/stamp_00551_1` came back as
`spods/stamp_00546_1`. A merge keeps the *larger* class's id, so the name goes
on meaning what it meant; `assign_class_ids` derives an id from the group's
*smallest page id*. Two writers, no agreement, and the disagreement is invisible
until a rebuild renames the class under a roster that names it. `apply_confusable`
now says which id the rebuild will use, at the moment of merging.

The roster-drift warning would have caught this one — that safety net works —
but only three steps downstream, after a roster had been written wrong.

## The membership pass

Every instance of every roster class, on 29 numbered contact sheets: **613
crops, 0 rejected.** That is the expected outcome for a corpus whose threshold
runs deliberately strict — it over-splits rather than over-merging, so the error
this pass catches is the rarer one — and the two over-merges that did exist were
found and split by the #3561 audit before this roster was picked.

What the pass is worth is not the count. It is that no positive in the eval is
now unexamined, so a miss is the detector's fault and not possibly the label's.

Two things seen while looking, neither a rejection:

- **A wide crop is not a second mark.** Two of the 16 StaVer routing-box
  instances (`stampds-00218`, `-00228`) include the `EINGEGANGEN AM 05. JAN.
  2011` date stamp sitting above the box, and `spods/00546` catches an order
  line above the `DY.Secretary` stamp. Same mark, extra paper — the box
  geometry is loose, not the label.
- **`tobacco800/logo_afm90c00-first_1_0` is flagged `mixed`** (spread 0.417,
  the only class over the 0.40 bar) and proposes a 16/8 split. At full size all
  65 instances are one engraving; the spread is binarisation varying between
  scans, which is the false proposal the README warns is obvious on a sheet and
  invisible in the number.

## The confusable pass

**All 276 pairs of the 24-class matrix, adjudicated in full rather than
sampled.** Every one is `different`; no merges. The closest pairs were checked
at full size, the rest side by side on composite sheets.

That the answer is uniform does not make the pass empty — it is the half of the
ground truth that usually goes missing, and its absence is invisible. Before
this, the only thing keeping those 24 marks in 24 classes was where a distance
threshold happened to land. Now each class records the 23 it is distinct from,
and 291 cannot-link rows enforce it on every future re-cluster.

The ranking the sheets were ordered by is worth one note. `phash` puts the
StaVer routing box and `DFKI Empfang` first at d=0.031 — two marks nobody would
confuse — while the pair a reviewer actually has to think about, the two
Tobacco800 leaf marks, sits far down it. That is #3600's finding reproduced: a
perceptual hash of ink on paper measures ink layout. Adjudicating the whole
matrix is what makes the ordering harmless.

## Provenance of the labels

`audit.reviewed_by` now records who worked the sheets, because
`membership_verified` is a boolean and "checked by the person who owns this
benchmark" and "checked by whoever ran the script" are different standards of
evidence that a boolean cannot tell apart. Every class on this roster carries:

```
claude-opus-5 (#3343 stage 4); not countersigned by the corpus owner
```

Two of the calls behind it are the owner's, from the #3561 round-1 and round-2
audits: that the three B&W leaf marks are different, and that all 65 instances
of the chief engraving are one mark. The sheets are all on disk at
`/expscratch/sgreenberg/fullmarks/corpus/audit/`, so countersigning is looking,
not redoing.

## The cells

Tier `s` is built and verified. Tier `m` is running. **Tier `l` should not be
started until the cell can be streamed to disk**, and that is the finding.

| cell | size | embed | medias | verify |
|---|---:|---:|---:|---|
| `fullmarks_s__siglip.pkl` | 17 MB | 230 s | 5,000 | ok |
| `fullmarks_s__sift_vlad.pkl` | 847 MB | 3,840 s | 5,000 | ok |

Both carry 2,222 labelled medias and every media has a vector.

**Nothing had ever run this code.** Stage 5 had not been reached, and no
`sift_vlad` cell exists anywhere in the shared pile — so this is the first time
the shipped structural embedder has been priced on documents, and the first time
`embed_corpus.py` has been asked to write anything. It could not: it loads the
calibration harness's `_cells_io` by path, and that module imports its own
sibling `_cells_paths` by bare name, so `dump_medias` was unreachable (#3843).

The defect is trivial. Where it fires is not — `dump_medias` is the *last line*
of a cell, so `fullmarks_s × sift_vlad` embedded all 5,000 pages, ran **2h 16m at
27.7 GB**, and then died on `ModuleNotFoundError`. The fix is therefore two
fixes: the import resolves, and `main` now resolves the writer *before* embedding
anything, so a broken write path costs seconds instead of hours.

### Tier `l` is priced out as written (#3842)

Two walls, and the softer one is the one that gets quoted:

- **Time.** `sift_vlad` measured **1.30 pages/s** on 8 CPUs (0.61 on 4). Tier
  `m` is ~11 h, tier `l` ~43 h. Slow, but affordable. `siglip` is not the
  problem at 21.7 pages/s.
- **Memory, which binds harder.** `load_medias` reads every page in the tier
  into one dict before embedding starts and holds `media_bytes` for all of them;
  the medias then accumulate `local_features` until the whole dict is pickled at
  the end. **34.5 GB peak for 5,000 pages.** The added UCSF pages are cheaper
  than the anchors — 150 dpi against SPODS's A4 at 300 dpi, so far fewer
  keypoints — but not forty times cheaper, and there is no node here that holds
  the extrapolation.

The cell *on disk* is fine: 847 MB for 5,000 pages is 169 KB/page, so tier `l`
lands near the ~34 GB #3343 estimated. It is the assembly that does not fit.
Streaming it — load-embed-drop in chunks, appending rather than building whole —
is the fix, and it is a change to the cell format's writer rather than a bigger
`--mem`. A job that OOMs at hour 40 costs the same as one that OOMs at hour 1
and tells you less.

## What is still owed

- **A countersignature.** Every label here is `reviewed_by: claude-opus-5`. The
  sheets are on disk; agreeing with them is looking, not redoing.
- **#3837** — a separation written *before* the membership pass pins two marks
  rather than two classes. All 291 of this corpus's separations are the strong
  kind, but the `merge` slate, which runs before membership by design, is where
  the weak kind gets written.
- **Box geometry on wide crops.** Three roster instances include page furniture
  the mark does not own. The label is right and the box is loose; nothing
  measures how often that happens across the corpus.
- **The `letterhead` pass on UCSF's 197k pages**, which decides whether that
  pool can ever contribute classes, and the `siglip` band descriptor that would
  let it. Both were already open from v2.
- **`eligible_distractor_sources` still has no consumer.** The contamination
  data is correct; nothing scores with it yet. This is the caveat #3343 opens
  with and it survives the issue.
