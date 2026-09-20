# DocMarks — datasheet and use register

**Corpus version v3.1** (`docmarks_config.CORPUS_VERSION`), 2026-09-17. DocMarks
is a benchmark for **finding a given stamp or printed logo in a pile of scanned
pages**, from one query crop. It exists so that ideas about that task (matchers,
shortlists, embedders, query handling) can be tested against labels someone has
checked. It is not a model and it does not come with a recommended method.

How the corpus is built and audited is in [`README.md`](README.md). This page
covers what the data *is* and what a study may *conclude* from it.

## What it is

| | |
|---|---|
| pages | **200,000**, in three nested tiers: `s` = 5,000 ⊂ `m` = 50,000 ⊂ `l` = 200,000 |
| roster | **23 classes**, **721 instances**, almost always one instance per page — one class repeats, see [Repeated marks on a page](#repeated-marks-on-a-page) |
| instances per class | 8 to 82, median 31 |
| other labels | 695 must-link and 789 cannot-link rows in `adjudications.json`, keyed on `(page_id, mark_index)` |
| query | one crop per class (`query_crop`). Extra hand-chosen crops are pending (#3949) |
| cells | tiers `s` and `m`: `siglip` and `sift_vlad`. Tier `l`: `siglip` only (`sift_vlad` was cancelled because its keypoint budget starves it, #3911) |
| lives at | `/expscratch/sgreenberg/docmarks/corpus/` (`corpus.jsonl`, `classes.json`, `adjudications.json`, `added_marks.json`, `roster.json`); cells under `vts-cache/datadir/embeddings/docmarks_<tier>__<embedder>.pkl` |

### Repeated marks on a page

One roster class breaks the otherwise one-instance-per-page rule.
`tobacco800/logo_asg54f00_1` is a letterhead mark printed **three times on each
of four pages** — `asg54f00`, `bjn43c00-page02_1`, `dqn43c00` and
`nrg54f00-page02_1` — and once on six others: 18 instances over 10 pages.

This is the data, not a labelling error; the letterhead really does carry the
logo three times. A study that assumes one instance per page will undercount
this class, so **rank by mark, not by page**, and count a page-level hit as
correct when it matches any instance on that page.

Every other roster class has at most one instance per page.

### Sources

| source | what it is | pages | roster classes | roster instances |
|---|---|---:|---:|---:|
| SPODS | pseudo-official documents made for the dataset, carrying logos, stamps and signatures | 1,088 | 11 (5 logos, 6 stamps) | 347 |
| Tobacco800 | 1980s–90s tobacco-litigation scans (IIT-CDIP), binarised; boxed logos and signatures | 1,290 | 10 logos | 348 |
| StaVer | German scanned invoices carrying rubber stamps | 400 | 2 stamps | 26 |
| UCSF Industry Documents | real scanned pages from six industries, **distractors only** | 197,222 | 0 | 0 |

**Every anchor page is in tier `s`.** The 2,778 SPODS, Tobacco800 and StaVer
pages all sit in the smallest tier, next to 2,222 UCSF pages. Tiers `m` and `l`
add **only UCSF pages**. So a class's same-source hard negatives are the same
pages in every tier, and going from `s` to `l` adds only UCSF-style distractors.

| UCSF industry | in `s` | in `m` | in `l` |
|---|---:|---:|---:|
| Opioids | 1,250 | 26,954 | 112,403 |
| Food | 744 | 15,796 | 66,198 |
| Tobacco | 158 | 3,340 | 14,002 |
| Chemical | 52 | 826 | 3,355 |
| Drug | 12 | 234 | 979 |
| Fossil Fuel | 6 | 72 | 285 |

The table is cumulative, since tiers are nested. `meta.tier` on a page records
the smallest tier it is in, not its cumulative membership.

UCSF pages carry 14,002 `letterhead_author` band boxes. None of them belongs to
a class: they are raw material for the proposed
UCSF classes (#3921, #3922). Until those pass the same audits, **UCSF holds no
positives for anything.**

### The roster

| class | kind | instances | median box (px) |
|---|---|---:|---:|
| `spods/logo_00003_0` | logo | 31 | 331 × 329 |
| `spods/logo_00011_0` | logo | 31 | 494 × 270 |
| `spods/logo_00014_0` | logo | 31 | 393 × 379 |
| `spods/logo_00023_0` | logo | 31 | 377 × 391 |
| `spods/logo_00029_0` | logo | 31 | 294 × 372 |
| `spods/stamp_00293_1` | stamp | 33 | 263 × 263 |
| `spods/stamp_00546_1` | stamp | 32 | 389 × 168 |
| `spods/stamp_00577_1` | stamp | 32 | 444 × 243 |
| `spods/stamp_00612_1` | stamp | 32 | 531 × 242 |
| `spods/stamp_00769_1` | stamp | 32 | 448 × 214 |
| `spods/stamp_00931_1` | stamp | 31 | 376 × 82 |
| `staver/stamp_stampds-00213_1` | stamp | 18 | 560 × 300 |
| `staver/stamp_stampds-00230_0` | stamp | 8 | 317 × 183 |
| `tobacco800/logo_aah97e00-page02_1_0` | logo | 63 | 316 × 193 |
| `tobacco800/logo_aeq93a00_1` | logo | 14 | 217 × 105 |
| `tobacco800/logo_afm90c00-first_1_0` | logo | 82 | 114 × 94 |
| `tobacco800/logo_ajj10e00_1` | logo | 50 | 720 × 216 |
| `tobacco800/logo_ald41a00-ernest_1` | logo | 51 | 237 × 99 |
| `tobacco800/logo_asg54f00_1` | logo | 14 | 174 × 173 |
| `tobacco800/logo_azb11c00_1` | logo | 33 | 69 × 78 |
| `tobacco800/logo_bqz95d00_1` | logo | 10 | 173 × 214 |
| `tobacco800/logo_cgr96c00_1` | logo | 9 | 170 × 177 |
| `tobacco800/logo_ciy01a00-page02_1_0` | logo | 22 | 192 × 184 |

The roster was picked by hand, and deliberately not from the top of the
shortlist ranking, whose top 24 were 21 SPODS logos. Plain shapes (a red cross,
an `H+` sign) are left off. The hard pairs are kept on purpose:
- three Tobacco800 leaf marks: `ald41a00`, `azb11c00`, `asg54f00`;
- two chief engravings, one a near-mirror of the other: `afm90c00`, `ciy01a00`;
- four SPODS stamps that all read `…Secretary`.

See [`2026-09-13-docmarks-v3`](../../../docs/experiments/2026-09-13-docmarks-v3/REPORT.md).

## Versions

| version | date | what changed | instances |
|---|---|---|---:|
| v3 | 2026-09-14 | roster countersigned; every instance and all 276 pairs adjudicated | 613 |
| **v3.1** | 2026-09-17 | completeness pass applied (#3927); cells relabelled, no pages added or removed | **721** |

The pages and tiers are identical between v3 and v3.1. Only labels moved: 108
pages that v3 scored as **negatives** for a class are positives in v3.1.

- **A number measured on v3 is not a v3.1 number.** That covers every study up
  to and including #3904, #3911, #3912, #3914 and #3928.
- It is comparable only after re-scoring against the relabelled cells.

Versioning rule: a label-only change to the same page set bumps the minor
version; a new page set, tier cut or roster bumps the major.
`build_report.json` records `corpus_version` from builds after v3.1. The
on-disk v3.1 corpus predates that field, and is recognisable by
`added_marks.json` holding 7 boxes and 721 roster instances.

## Where the labels come from

**Boxes come from the sources and identities come from people.**
- SPODS ships a per-mark mask and StaVer a per-stamp mask; boxes are cut from
  those.
- Tobacco800 ships logo boxes.
- None of the three ships "these two marks are the same". Identity started as
  greyscale perceptual-hash clustering and was then settled by hand in the
  passes below.

| pass | question | result on this roster |
|---|---|---|
| 1 `merge` | which classes are one mark? | the v3 merges, plus one owner overturn: the "100 Years of Achievement" panel *is* the `afm90c00` chief engraving, so the roster went 24 → 23 classes |
| 2 `membership` | is every instance this mark? | **613 of 613** v3 instances examined, 0 rejected |
| 3 `confusable` | is each pair of classes different? | **276 of 276** pairs adjudicated; the one pair the owner ruled the same is the merge in row 1 |
| 4 `cluster` | is a class one mark at all? | recorded per class in `audit.cluster_ok` |
| 5 `distinctive` | mark or plain shape? | recorded per class in `audit.distinctive`; plain shapes were kept off the roster when it was picked |
| 6 `letterhead` | do UCSF bands carry a mark? | 157 of 320 sampled bands (49%) do (#3901); no UCSF class yet |
| 7 `completeness` | which pages carry a class's mark but aren't members? | **+108 instances**: 613 → 721 (#3927) |
| 8 `query_crops` | which members make good extra queries? | proposed, awaiting owner review (#3949) |

**Completeness (#3927).**
- **Proposals:** SIFT at 8,192 keypoints compared every roster query with
  every page of tier `s`. The best-matching non-members were drawn on
  one-screen sheets.
- **Decisions:** the owner confirmed every accepted candidate.
- **What it found:** mostly marks that clustering had filed under other
  classes. The SPODS elephant stamp was spread over ten classes. A few pages
  had never been boxed; 7 boxes were added by hand, with
  `provenance="completeness"`.
- **Durable records:** accepted candidates became must-links and rejected ones
  cannot-links. These are permanent hard negatives.
- **Owner rulings:**
  - the leafless B&W monogram counts as `ald41a00-ernest_1`;
  - Sam Greenberg, 2026-09-17: in `tobacco800/logo_aah97e00-page02_1_0`
    (Philip Morris) the globe crest with *VENI·VIDI·VICI* and the *PM*
    monogram crest with *PHILIP MORRIS* are the **same mark, one class**; both
    crests are positives for each other. Adjudications name mark pairs, not
    class-level rulings, so this lives here and in the class's members.

**Everything decided by hand survives a rebuild.** Three stores are replayed,
in this order:
- `adjudications.json` (must-link and cannot-link rows, keyed on the mark
  rather than the class id);
- `added_marks.json` (hand-drawn boxes, replayed **before** clustering so the
  must-links can name them);
- `box_overrides.json` (hand-accepted tighter boxes, replaced in place by mark
  index right after the added marks, so every adjudication still names the
  same mark);
- `query_crops.json` (extra query boxes, replayed after the primary crops,
  once #3949 is applied).

A rebuild that cannot place a stored item warns rather than dropping it.

## Scoring pools and the contamination rule

A class is scored against its positives plus a pool of pages that are safe to
call negatives. `eval_retrieval.py` offers three pools:

| pool | negatives | use |
|---|---|---|
| **`own_verified`** (the headline, `HEADLINE_POOL`, #3913) | the class's own source as **known negatives**, plus every page the contamination rule allows | quote numbers from this one |
| `eligible` | only pages the contamination rule allows, so **the class's own source is excluded** | diagnostic only |
| `naive` | every page in the tier | records the size of the exclusion |

The query crop's own page is dropped from every pool.

**The contamination rule** (`CONTAMINATES` in `docmarks_config.py`) is applied
**per page** (#3904), with UCSF's industry read for each page:

- SPODS and StaVer marks exist nowhere else, so every other source is a safe
  distractor.
- Tobacco800 and UCSF's **Tobacco** industry are the same IIT-CDIP archive, so a
  Tobacco800 class is never scored against a UCSF Tobacco page. The other five
  UCSF industries are admitted. Before the per-page fix, all 3,182 UCSF
  Tobacco pages first placed in tier `m` counted as negatives, and SigLIP's
  top-ranked ones carried the Lorillard mark unlabelled.
- The Food industry was checked for leaks through corporate ownership
  (Philip Morris owned Kraft, RJR owned Nabisco): **0 of 80** top-ranked pages
  and **0 of 49** company-named pages carry a roster mark (#3914). That check
  was not exhaustive over Food's 15,796 tier-`m` pages, and tier `l` is
  unchecked.

**Why own-source pages are negatives and not exclusions.**
- Every mark on a SPODS, Tobacco800 or StaVer page is boxed and clustered. A
  same-source page that is not a member has therefore been checked not to
  carry the mark.
- Excluding those pages leaves the positives as the only pages in their
  source's style. A `source_prior` control that ranks pages by source and
  ignores the mark then scores **AP 1.00** under `eligible`.
- Under `own_verified` the same control scores **0.029** (#3904, #3913).

## What this dataset can and cannot be asked

The sections above say what the data *is*. This one says what a study may
**conclude** from it. Each verdict points at the measurement behind it. A
question shape that is not listed has not been thought about.

The central fact: **the labels were completed by one matcher.** The
completeness pass could only propose pages that SIFT at 8,192 keypoints ranked
highly. A copy that SIFT cannot see (faint, heavily degraded, or tiny) is the
copy most likely to still be sitting in the negatives. A method that finds
those copies is scored down for it, and SIFT is not.

| question shape | verdict | what decides it |
|---|---|---|
| **Method A vs method B**, `own_verified`, one tier, v3.1 | **supported, with a stated bias** | Same pages and labels for both arms. Residual missed positives are not symmetric between methods: they are the ones SIFT ranked low, so a SIFT-family method is favoured on exactly the pages still unlabelled. Say so beside any SIFT-vs-other result, and inspect the top-ranked "false positives" of the non-SIFT arm before claiming it lost. |
| How does a method degrade as the haystack grows (`s` → `m` → `l`)? | **supported for UCSF distractors only** | Tiers `m` and `l` add only UCSF pages; the same-source hard negatives are all already in `s`. The drop measures robustness to real, unrelated scans, not to harder same-style pages. Tier `l` has only a `siglip` cell. |
| Absolute AP under the `eligible` pool | **not supported as a headline** | The source shortcut: `source_prior` scores AP 1.00 there (#3904). |
| Anything on the `naive` pool for a Tobacco800 class | **not supported** | UCSF Tobacco pages carry roster letterheads unlabelled; `naive` scores a correct retrieval as a false positive. |
| Are **stamps** harder than **logos**? Is SPODS easier than Tobacco800? | **not supported** | Kind is confounded with source. Every Tobacco800 class is a logo; 6 of 8 stamp classes are SPODS. The source also sets the imaging: SPODS is clean colour, Tobacco800 is binarised 1990s scans. A kind or source contrast measures all of that at once. |
| Does a method find **real rubber-stamp impressions**? | **weakly supported: 2 classes, 26 instances** | StaVer is the only source of real hand-stamped impressions on the roster. The SPODS stamp classes vary across copies (median pixel correlation with a reference copy 0.15–0.52: rotation, ink and scale), but SPODS documents were made for the dataset, and nothing here records how their stamps were applied. |
| Invariance claims on the **SPODS logos** | **not supported** | The five SPODS logo classes are **pixel copies of one artwork**: 8 members each correlate **0.96–0.99** with a reference copy at 128 × 128 greyscale, and box area has a coefficient of variation of only 0.06–0.20. Within-class variation is print/scan noise, so finding them is close to template matching and says little about appearance change. |
| **Per-class** AP or recall | **supported with n beside it** | Classes run from 8 to 82 instances. One miss moves recall by 1/8 on `staver/stamp_stampds-00230_0` and by 1/82 on `afm90c00`. |
| Is class X **harder** than class Y? | **read with the known-gaps list** | Label residuals differ by class (below). `spods/stamp_00931_1` (OUTWARD-) had the weakest SIFT evidence of any class, so its completeness is least certain. |
| Telling **near-identical marks** apart | **supported** | All 276 roster pairs adjudicated; the three leaf marks, two chiefs and four Secretary stamps are separate classes by ruling, and cannot-links are permanent. |
| **Localisation** (box IoU, detection mAP) | **not supported for StaVer, unmeasured elsewhere** | Boxes come from source masks and were never audited for tightness, except the query crops. StaVer boxes on `stampds-00213_1` are wide enough to take in the separate EINGEGANGEN AM date stamp. |
| **Query sensitivity**: how much does the crop matter? | **not yet** | One crop per class until #3949 is applied. `tobacco800/logo_aeq93a00_1`'s crop has 211 SIFT keypoints and sits at the bottom of every ranking, so a per-class result mixes the method with that crop. |
| Retrieval **of UCSF letterheads**, or anything with a UCSF positive | **not supported** | UCSF holds no classes. The band classes proposed in #3902 are audit candidates (#3921, #3922). |
| Compare against a number measured **before 2026-09-17 07:40** | **not comparable** | That is v3: 108 of today's positives were negatives then. Re-score against the relabelled cells. |

### Known gaps, by class

- **`spods/stamp_00931_1` (OUTWARD-).** Faint red impressions. Its members'
  median was 19 SIFT inliers, against 43–160 for the other SPODS classes, so
  faint copies are the most likely positives still unlabelled.
- **`tobacco800/logo_aah97e00-page02_1_0` (Philip Morris).** The members
  mix the globe crest (the query) and the *PM* monogram crest. By owner ruling
  (2026-09-17, above) they are one mark, so a method that ranks only globe
  copies high is *missing positives*, not being strict. A per-class number
  here still depends on how a method bridges the two crests.
- **`staver/stamp_stampds-00213_1`.** Some boxes are wide enough to take in
  the EINGEGANGEN AM date stamp, which is page furniture and not part of the
  mark (recorded in `roster.json` at v3). The owner chose to tighten them by
  review: `box_tighten.py` (README audit item 9) proposes the boxes, and
  nothing changes until the verdicts are applied.
- **`staver/stamp_stampds-00230_0`.** 8 instances, the minimum the roster
  admits.
- **`tobacco800/logo_azb11c00_1`.** The copies are tiny: median box 69 × 78 px,
  and only two members are large. It sits near the scale floor at which the
  matcher accepts a fit (0.03 of page width, #3912).
- **`spods/stamp_00612_1`** is a piece of the #3561 split of
  `spods/stamp_00489_1`. It depends on that split being replayed from
  `adjudications.json`.
- **UCSF distractors are not exhaustively clean.** The per-page rule and the
  Food check (#3914) cover the known routes by which a roster mark could leak.
  No hand pass covers 197,222 pages.

### Reference points, not targets

These are recorded so a new idea has something to stand next to. **All were
measured on v3 labels**, so re-score before comparing:

| ranker | tier `s` AP | tier `m` AP | source |
|---|---:|---:|---|
| SIFT, 8,192 keypoints, every page verified | 0.78 | 0.75 | #3911 |
| tiled VLAD top-1,000 → SIFT | 0.71 | 0.59 | #3928 |
| tiled VLAD alone | 0.40 | 0.25 | #3928 |
| SuperPoint + LightGlue | 0.39 | — | #3911 |
| SigLIP | 0.12 | 0.076 | #3904 |
| page-level VLAD (`sift_vlad` cell) | 0.029 | 0.003 | #3911 |

The `source_prior` control, which ignores the mark, scores 0.029 on
`own_verified`: within-source chance (#3904).

The completeness pass drew its proposals from the top row. That row is the one
most likely to have risen with the v3.1 labels.
