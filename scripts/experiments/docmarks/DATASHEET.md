# DocMarks — datasheet and use register

**Corpus version v4.3** (`docmarks_config.CORPUS_VERSION`), 2026-09-23. DocMarks
is a benchmark for **finding a given stamp or printed logo in a pile of scanned
pages**, from one query crop. It exists so that ideas about that task (matchers,
shortlists, embedders, query handling) can be tested against labels someone has
checked. It is not a model and it does not come with a recommended method.

How the corpus is built and audited is in [`README.md`](README.md). This page
covers what the data *is* and what a study may *conclude* from it.

## What it is

| | |
|---|---|
| pages | **199,855**, in three nested tiers: `s` = 4,999 ⊂ `m` = 49,969 ⊂ `l` = 199,855 |
| roster | **27 classes**, **2,024 instances**, almost always one instance per page — one class repeats, see [Repeated marks on a page](#repeated-marks-on-a-page) |
| instances per class | 8 to 399, median 32 |
| other labels | 733 must-link and 1,159 cannot-link rows in `adjudications.json`, keyed on `(page_id, mark_index)` |
| query | one primary crop per class (`query_crop`) plus up to four hand-chosen alternates (`query_crops`), **119 in all**, so a study can average over queries (#3949) |
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
| SPODS | pseudo-official documents made for the dataset, carrying logos, stamps and signatures | 1,088 | 11 (5 logos, 6 stamps) | 349 |
| Tobacco800 | 1980s–90s tobacco-litigation scans (IIT-CDIP), binarised; boxed logos and signatures | 1,290 | 10 logos | 1,312 |
| StaVer | German scanned invoices carrying rubber stamps | 400 | 2 stamps | 28 |
| UCSF Industry Documents | real scanned pages from six industries; **distractors, and since v4.0 four roster classes of its own** | 197,077 | 4 logos | 335 |

**Every anchor page is in tier `s`.** The 2,778 SPODS, Tobacco800 and StaVer
pages all sit in the smallest tier, next to 2,221 UCSF pages. Tiers `m` and `l`
add **only UCSF pages**, so a class's same-source hard negatives are the same
pages in every tier.

**But since v4.0 a larger tier adds positives too, not only distractors.** The
four UCSF roster classes put **10 instances in `s`, 256 more in `m` and 55 more
in `l`**. For those four classes the positive set grows with the tier, so their
tier-`s` and tier-`l` numbers are not measured over the same ground truth and
must not be compared as though they were. The other 23 classes are unaffected:
every one of their instances is on an anchor page, and every anchor page is in
`s`.

| UCSF industry | in `s` | in `m` | in `l` |
|---|---:|---:|---:|
| Opioids | 1,250 | 26,954 | 112,403 |
| Food | 744 | 15,796 | 66,198 |
| Tobacco | 157 | 3,309 | 13,857 |
| Chemical | 52 | 826 | 3,355 |
| Drug | 12 | 234 | 979 |
| Fossil Fuel | 6 | 72 | 285 |

The table is cumulative, since tiers are nested. `meta.tier` on a page records
the smallest tier it is in, not its cumulative membership.

UCSF pages carry 13,857 `letterhead_author` band boxes, none of which belongs
to a class: they were the raw material for the proposed UCSF classes (#3921,
#3922).

**Those classes passed their audit, so as of v4.0 UCSF does hold positives.**
Four of them — `bat_leaf`, `bw_oval_emblem`, `p_lorillard_crest` and
`rjr_script` — carry 335 instances over 335 UCSF pages (315 at v4.0; the
v4.2 review of banded pages found 20 more). Anything that scored
every UCSF page as a negative on the strength of the older claim is wrong for
those four classes and needs re-checking (#4050).

### The roster

| class | kind | instances | median box (px) |
|---|---|---:|---:|
| `spods/logo_00003_0` | logo | 31 | 331 × 329 |
| `spods/logo_00011_0` | logo | 31 | 494 × 270 |
| `spods/logo_00014_0` | logo | 31 | 393 × 379 |
| `spods/logo_00023_0` | logo | 31 | 377 × 391 |
| `spods/logo_00029_0` | logo | 31 | 294 × 372 |
| `spods/stamp_00293_1` | stamp | 34 | 263 × 263 |
| `spods/stamp_00546_1` | stamp | 32 | 389 × 168 |
| `spods/stamp_00577_1` | stamp | 32 | 444 × 243 |
| `spods/stamp_00612_1` | stamp | 32 | 531 × 242 |
| `spods/stamp_00769_1` | stamp | 32 | 448 × 214 |
| `spods/stamp_00931_1` | stamp | 32 | 376 × 82 |
| `staver/stamp_stampds-00213_1` | stamp | 20 | 558 × 292 |
| `staver/stamp_stampds-00230_0` | stamp | 8 | 317 × 184 |
| `tobacco800/logo_aah97e00-page02_1_0` | logo | 370 | 120 × 72 |
| `tobacco800/logo_aeq93a00_1` | logo | 130 | 94 × 39 |
| `tobacco800/logo_afm90c00-first_1_0` | logo | 112 | 95 × 95 |
| `tobacco800/logo_ajj10e00_1` | logo | 399 | 312 × 79 |
| `tobacco800/logo_ald41a00-ernest_1` | logo | 206 | 156 × 45 |
| `tobacco800/logo_asg54f00_1` | logo | 18 | 147 × 147 |
| `tobacco800/logo_azb11c00_1` | logo | 33 | 69 × 78 |
| `tobacco800/logo_bqz95d00_1` | logo | 10 | 173 × 214 |
| `tobacco800/logo_cgr96c00_1` | logo | 10 | 172 × 179 |
| `tobacco800/logo_ciy01a00-page02_1_0` | logo | 24 | 192 × 184 |
| `ucsf/logo_bat_leaf` | logo | 213 | 55 × 41 |
| `ucsf/logo_bw_oval_emblem` | logo | 30 | 100 × 33 |
| `ucsf/logo_p_lorillard_crest` | logo | 24 | 209 × 73 |
| `ucsf/logo_rjr_script` | logo | 68 | 280 × 59 |

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
| v3.1 | 2026-09-17 | completeness pass applied (#3927); cells relabelled, no pages added or removed | 721 |
| v4.0 | 2026-09-20 | UCSF classes admitted (#3953), second completeness pass, query-crop alternates; duplicate page records removed (#4054) | 2,004 |
| v4.1 | 2026-09-22 | the four UCSF classes score against UCSF's un-banded industries (#3922); no page, label or roster change | 2,004 |
| v4.2 | 2026-09-22 | 1,610 banded pages reviewed for the UCSF classes (#4088): 20 new positives, 1,551 reviewed negatives; SigLIP's top presumed negatives checked for all 27 classes (#4089): 0 of 516 carry the mark | 2,024 |
| **v4.3** | 2026-09-23 | every UCSF-class box reviewed (#4109, #4125): 303 proposals accepted, 30 drawn by hand, 3 set to their query crop's extent; band-located marks 64 → 0; no page, positive or negative changed | **2,024** |

The pages and tiers are identical between v3 and v3.1. Only labels moved: 108
pages that v3 scored as **negatives** for a class are positives in v3.1.

Versioning rule: a label-only change to the same page set bumps the minor
version; a new page set, tier cut or roster bumps the major.

**v4.0 is a major bump, and by that rule it had to be.** The roster went from 23
classes to 27, which is the clause that decides it; the page set moved as well,
from 200,000 records to 199,855, when 137 UCSF pages that had been ingested more
than once were collapsed to one record each (#4054).

- **A number measured on v3 or v3.1 is not a v4.0 number.** That covers every
  study up to and including #3904, #3911, #3912, #3914 and #3928.
- A minor bump is comparable after re-scoring against the relabelled cells. A
  major one is not: at v4.0 the baselines are **re-run**, because the roster, the
  positive sets and the page list all moved.

**v4.1 is a minor bump.** Pages, tiers, labels and roster are unchanged; only
the four UCSF classes' pools grew, when the contamination rule for UCSF classes
narrowed from all of UCSF to its Tobacco industry (below). A v4.0 number for
any of the other 23 classes is a v4.1 number. For the four UCSF classes,
**re-score**. The cells need no relabel, because pools are computed at scoring
time.

**v4.2 is a minor bump too:** labels only, on the same pages. Two hand passes
moved them. The banded review (#4088) added 20 positives to the UCSF classes
and 1,551 reviewed negatives. The top-hit review (#4089) turned 516 presumed
negatives, spread over all 27 classes, into reviewed ones, and found no
unlabelled positive among them. Every class's pool changed, so **re-score**
every class. The cells need no relabel.

**v4.3 changes boxes only.** Every box on the four UCSF classes was reviewed
(#4109, #4125): 303 proposals accepted, 30 drawn by hand, and 3 query-page
boxes set to their crop's extent. Retrieval scores are page-level, so every v4.2
retrieval number is a v4.3 number. A number that reads boxes (localisation,
crops cut from members) is not.

`build_report.json` records `corpus_version` from builds after v3.1. The
on-disk v3.1 corpus predates that field, and is recognisable by
`added_marks.json` holding 7 boxes and 721 roster instances; v4.0 holds 1,268
boxes and 2,004 roster instances.

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
| `eligible` | only pages the contamination rule allows, so **the class's own source is excluded** (for a UCSF class, UCSF's Tobacco industry) | diagnostic only |
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
- A **UCSF** class never uses UCSF's **Tobacco** industry. Those 13,857 pages
  are exactly the ones the letterhead pull banded, and the band classes miss
  most of their own mark there (#3922). The five un-banded industries were
  admitted at v4.1, on corporate lineage (these are tobacco marks) and a hand
  check. That check drew 200 pages at random, three-quarters from Food, and
  took SigLIP's top 30 per class; no page in either sample carried the mark,
  and all 6 planted controls were found. At tier `m` this took each class's
  pool from ~2,900 pages to ~46,700
  ([report](../../../docs/experiments/2026-09-22-docmarks-ucsf-contamination/REPORT.md)).

**Why own-source pages are negatives and not exclusions.**
- Every mark on a SPODS, Tobacco800 or StaVer page is boxed and clustered. A
  same-source page that is not a member has therefore been checked not to
  carry the mark.
- Excluding those pages leaves the positives as the only pages in their
  source's style. A `source_prior` control that ranks pages by source and
  ignores the mark then scores **AP 1.00** under `eligible`.
- Under `own_verified` the same control scores **0.029** (#3904, #3913).
- **The UCSF classes' style shortcut is closed at tier `m` (v4.2).** Every
  positive is on a banded Tobacco-industry letter. Until v4.2 the only such pages
  among their negatives were the 1–13 per class a person had reviewed. So a
  control that ranks UCSF Tobacco pages first, ignoring the mark, scored **AP
  0.50–0.97** under `own_verified` at tier `m`. The #4088 review added 65–1,041
  reviewed Tobacco negatives per class, and the control now scores **0.15–0.17**
  at `m`. Two residuals remain. At tier `l` it is 0.17–0.27, because
  `p_lorillard_crest` gains positives whose banded neighbours were not sampled.
  At tier `s` it is 0.14–0.38, because 2–5 positives make the control noisy.
  Ranking by source alone scored 0.20–0.97 at v4.0 and 0.00 since v4.1.

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
| **Localisation** (box IoU, detection mAP) | **supported for the four UCSF classes; not supported for StaVer; unmeasured elsewhere** | UCSF (v4.3, #4109): 303 boxes were proposed by fitting the query crop's ink outline with SIFT, padded 10%, and each was accepted or redrawn by hand. **The tolerance:** a box counts as tight if it clips only the tips of descending loops, and not if it cuts off substantial strokes such as the tops of letters (owner's rule, 2026-09-23). A loose-IoU criterion suits that tolerance; a strict one does not. The 30 marks no proposal fitted were boxed by hand (#4125), and the 3 query pages carry their crop's own extent. The rjr_script logo is the script. The "Tobacco Company" line is printed under it on some letters and not others, and does not change what the mark is (owner, 2026-09-23). So a box with or without that line is correct: 4 hand-drawn boxes take it in, because the script's loops hug it, and the query crop and the other 64 do not. Other sources' boxes come from source masks and were never audited for tightness, except the query crops. StaVer boxes on `stampds-00213_1` are wide enough to take in the separate EINGEGANGEN AM date stamp. |
| **Query sensitivity**: how much does the crop matter? | **not yet** | One crop per class until #3949 is applied. `tobacco800/logo_aeq93a00_1`'s crop has 211 SIFT keypoints and sits at the bottom of every ranking, so a per-class result mixes the method with that crop. |
| Retrieval **of UCSF letterheads** (the four roster classes) | **supported at tier `m`, with the control beside it** | At v4.2 a mark-blind Tobacco-industry control scores AP 0.15–0.17 at `m` (above). Quote it beside any UCSF-class number, and read tiers `s` and `l` with their residuals. The other band classes proposed in #3902 are still audit candidates (#3921, #3922). |
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
  No hand pass covers 197,222 pages. Two measured bounds exist, both at tier `m`:
  six anchor classes, 0 of 239 random pages (≤1.3%, 2026-09-20); and the four
  UCSF classes, 0 of 152 Food and 0 of 48 other random pages (≤2.0% and ≤6.3%).
  A rate bound is not a count bound: 1% of a 46,700-page pool is ~470 pages, and
  a UCSF class has 9–181 positives there. What the bounds show is that nothing
  is common, and the ranked arms show nothing sits at the top of a SigLIP
  ranking. The top-hit review (#4089, v4.2) then checked each class's
  highest-ranked presumed negatives from a real SigLIP run at tiers `s` and `m`:
  **0 of 516** carried the mark. Each run can queue its own
  (`eval_retrieval.py` writes `surprise_hits.json`; `surprise_review.py`).

### Scoring a new idea

`score_ranker.py` scores any ranker the way the reference numbers below were
scored (#4108). Write a CSV (optionally gzipped) of `class_id,page_id,score`,
where higher means "more likely to carry this class's mark", then:

```
python score_ranker.py score --scores my_idea.csv.gz --tiers s,m --name my_idea --out <dir>
```

- It uses the headline `own_verified` pool and the same ranking and tie-break
  as `eval_retrieval.py` and `eval_sift_rank.py`. It reproduces their SIFT and
  SigLIP APs exactly, class for class.
- Every number sits beside two **mark-blind controls**: `source_prior` and
  `provenance_prior` (same source *and* industry as the positives). A result
  that doesn't clear the provenance control is not evidence the method sees the
  mark.
- It writes `surprise_hits.json`, the method's top-ranked presumed negatives.
  Queue them with `surprise_review.py`: if one carries the mark, the method was
  right and the labels were not (#4089).
- Every result is stamped with the corpus version, and with whether the corpus
  on disk still matches that version's frozen manifest in `versions/`. v4.3 is
  the first frozen version: `versions/v4.3.json` checksums the six files that
  define it and all 27 query crops. A version bump runs `score_ranker.py freeze`.

To start from a built-in method, `score_ranker.py export --method siglip --tier m`
writes SigLIP's scores in that format. Export from the tier you score, because
each tier's cell was embedded separately.

### Reference points, not targets

These are recorded so a new idea has something to stand next to. `own_verified`
pool, mean over the roster. The first rows were **re-run on v4.2** (27 classes,
#4087); the rest are **v3** numbers (23 classes), to be re-scored before
comparing:

| ranker | tier `s` AP | tier `m` AP | labels | source |
|---|---:|---:|---|---|
| SIFT, 8,192 keypoints, every page verified | 0.79 | 0.83 | v4.2 | #4087 |
| SigLIP top-1,000 → SIFT | 0.52 | 0.30 | v4.2 | #4087 |
| page-level VLAD top-1,000 → SIFT | 0.39 | 0.041 | v4.2 | #4087 |
| SigLIP | 0.12 | 0.083 | v4.2 | #4087 |
| page-level VLAD (`sift_vlad` cell) | 0.022 | 0.004 | v4.2 | #4087 |
| tiled VLAD top-1,000 → SIFT | 0.71 | 0.59 | v3 | #3928 |
| tiled VLAD alone | 0.40 | 0.25 | v3 | #3928 |
| SuperPoint + LightGlue | 0.39 | — | v3 | #3911 |

At tier `s`, `ucsf/logo_p_lorillard_crest` has no positive and scores 0 for
every ranker; the tier-`s` means include it. The four UCSF classes, tier `m`,
v4.2: SIFT 0.75–0.91, SigLIP 0.00–0.16, against the mark-blind Tobacco-first
control at 0.15–0.17 ([report](../../../docs/experiments/2026-09-22-docmarks-review-4088-4089/REPORT.md)).

The `source_prior` control, which ignores the mark, scores 0.029 on
`own_verified`: within-source chance (#3904).

The completeness pass drew its proposals from the top row. That row is the one
most likely to have risen with the v3.1 labels.
