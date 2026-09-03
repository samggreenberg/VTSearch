# DocMarks v2 — the full-scale corpus, and what it can actually measure

**2026-09-02.** Companion to [`report.html`](report.html) (open it in a browser —
GitHub renders raw HTML as source). First build at the size
[#3343](https://github.com/samggreenberg/VTSearch/issues/3343) asks for: all four
sources, 200,000 pages, three nested tiers. Previous:
[v1](../2026-08-31-docmarks-v1/REPORT.md), 1,520 pages.

Regenerate with:

```
python scripts/experiments/docmarks/make_report.py \
    --corpus <corpus> --out docs/experiments/2026-09-02-docmarks-v2/report.html
```

**Question.** v1 fixed *where a mark is*. At 130× the scale, is the corpus one an
eval can be trusted on?

**Verdict.** Yes for SPODS and Tobacco800; no for UCSF, which contributes 197k
distractors and **zero** classes on purpose. Getting there cost four findings
that a build reporting `exit 0` had hidden.

## The corpus

| | v1 | v2 |
|---|---:|---:|
| pages | 1,520 | **200,000** |
| tiers `s` / `m` / `l` | — | **5,000 / 50,000 / 200,000** (exact) |
| admitted classes | 52 | **60** |
| warnings | 2 kinds | 2 kinds |

Pages by source: SPODS 1,088 · StaVer 400 · Tobacco800 1,290 · UCSF 197,222.

## Four things a green build was hiding

**StaVer contributed nothing.** Its scans are `stampDS-00001.png` and its masks
`stampDS-00001-`**`px`**`.png`, so every lookup missed on the suffix: 427
warnings and **zero pages**. It read as a source that had been skipped rather
than one that was broken. Fixed, it gives 400 pages and 374 marks — and
`count-mismatch = 0` across all 400, meaning StaVer's own recorded stamp counts
agree with the decomposition. That check had never run on real StaVer data.

**Tobacco800 contributed no classes.** It was hardcoded out of the clustering
loop, because "Tobacco800 ships GEDI ground truth" is true of its *signatures*
and false of its *logos* — its 432 logo zones carry no identity attribute at
all. The one source with a published logo protocol was supplying 1,290
distractor pages and nothing else. Nothing warned: an absent class is not an
error, and the survival curve counted the 130 signature classes it would never
admit, so the printed numbers looked plausible.

**The contamination guard was inert.** `industry` is *indexed but not stored* in
UCSF's Solr — `industry:Tobacco` filters correctly while `fl=industry` returns
nothing — so all 117,028 UCSF pages recorded `industry: null` and the Tobacco800
exclusion could never fire. Tobacco800 and UCSF Tobacco are the same archive
(IIT-CDIP), where a *correct* retrieval is scored as a false positive. The build
claimed 118,516 eligible distractors for Tobacco800 when ~47,000 of them were
its own archive.

**The budget was unreachable.** Distractors were split evenly across six
industries, three of which are nearly empty — **Fossil Fuel has 311 documents in
total**, against an even share of 33,333. Ceiling: 105,031. Now water-filled,
and ordered so Tobacco is drawn last; at 200k it is drawn not at all.

## One threshold did not fit four sources

`CLUSTER_THRESHOLD`'s docstring ends *"this number is a property of the data, and
it does not travel."* It does not travel between **sources** either. Swept
separately, largest-component share at each source's chosen value:

| source | threshold | largest share | classes ≥ bar |
|---|---:|---:|---:|
| StaVer | 0.04 | 7.4% | 1 |
| SPODS | 0.10 | 1.5% | 44 |
| Tobacco800 | 0.18 | 14.7% | **15** |

Tobacco800 wants nearly double SPODS's threshold and yields **5× the classes**
there (15 against 3). A single 0.10 was serving SPODS and quietly costing the
other two.

## UCSF letterhead bands are not classes

**No threshold works, and the cause is the descriptor.** The band is a
fixed-geometry crop — the top 22% of every page — so a perceptual hash of it
describes page layout, not the logo inside it, and two unrelated companies'
letterheads at the same position hash alike. Swept on 3,000 marks:

| threshold | 0.02 | 0.04 | 0.06 | 0.10 | 0.15 | 0.22 |
|---|---:|---:|---:|---:|---:|---:|
| largest share | **12.4%** | 36.3% | 53.4% | 80.7% | 94.5% | 98.8% |
| singletons | 2,565 | 1,835 | 1,326 | 552 | 138 | 23 |

SPODS sits pinned at 1.5% across that whole range. UCSF is percolated at the
*lowest* threshold on the grid — 85% dust plus a 12% blob, with nothing in
between — so there is no transition to find and no number to choose.

The literal artifact: at 0.10 this produced a single class of **12,706
instances**. Being made of admitted-class pages, it pinned 13,874 pages into
tier `s` against a 5,000 budget, and measured nothing.

README's own audit list already gated this — the `letterhead` pass exists to
"count how many carry a printed mark at all — decides whether that pool is worth
clustering", and it has never been run. So UCSF stays as 197k distractors, which
is what 92% of its pages were for, and the bands keep `class_id=None` for that
pass. Making them usable needs a descriptor that looks at the mark rather than
the strip; `siglip` is the candidate and wants its own sweep.

## What each class can be scored against

The number that decides whether an eval means anything is not the page count but
the contamination-safe haystack:

| class source | eligible haystack | share | classes |
|---|---:|---:|---:|
| SPODS | 198,912 | 99.5% | 44 |
| StaVer | 199,600 | 99.8% | 1 |
| Tobacco800 | 184,708 | 92.4% | 15 |

Tobacco800's 92.4% is the guard working: 14,002 UCSF Tobacco pages and its own
1,290 are excluded. Before the fix it read 98.9%, and the difference was
unlabelled positives.

## Honest limits

- **StaVer contributes one class.** Its sweep gives 1 at `≥5` and 12 only at
  `≥2`; most StaVer stamps appear once. `≥2` would manufacture classes that
  cannot support train-and-search, which is the bar README rejects for
  Tobacco800's published protocol. One is the honest answer, not an unturned
  knob.
- **The roster is 73% SPODS** (44 of 60). An eval on it substantially measures
  SPODS.
- **No instance is adjudicated yet.** Every class here is a *proposal* from
  clustering. The `membership` and `confusable` passes are what turn them into
  ground truth, and they must precede embedding — the cells carry the labels.
- **16,791 pages were dropped past tier `l`'s budget.** Anchor pages are now
  pinned into every tier and none of them are among the dropped.
