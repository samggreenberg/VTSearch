# DocMarks — stamps and logos in scanned documents

One instance-retrieval corpus, assembled from four sources in three strata, with
nested size tiers. Built for structural-search experiments: the 2026-07-13
screenshot/scanned-document study found the **first configuration where
structural search beats the deep embedder on a real corpus**
(SuperPoint+LightGlue, AP 0.395/0.481 vs SigLIP's 0.204/0.235), and then could
not push on that result because the two document corpora it used are far too
small — 259 and 1,088 pages, with as few as 9 instances per class.

```bash
python build_corpus.py --probe                        # can I reach every source?
python build_corpus.py --sources spods --limit 40     # small real build
python build_corpus.py --survival                     # class counts vs threshold
python build_corpus.py                                # the whole thing

python make_audit_slate.py --task cluster             # the human passes
python audit_to_corrections.py --task cluster --apply

source ../pile/pile_env.sh
python embed_corpus.py --tier s --embedders sift_vlad,siglip
```

## The three strata

| stratum | source | what it gives | what it costs |
|---|---|---|---|
| anchor | SPODS, StaVer, Tobacco800 | real marks, real scans, real boxes | small; ~2,700 pages, and two of the three ship no identities |
| haystack | UCSF Industry Documents Library | millions of real scanned pages; weak letterhead classes at 10⁵ instances | labels are metadata-derived and unverified |
| synth | LogoDet-3K artwork on held-out real scans | exact ground truth, sweepable size/rotation/count | pasted marks are not printed marks |

The rule that comes with the third one: **synthetic numbers quantify a
mechanism, real numbers size the effect.** A finding that appears only in
`synth` is a hypothesis about the pipeline, not a claim about documents.

## What each source actually ships

**SPODS** — 1,088 scanned pseudo-official pages, direct download, no
registration. Confirmed by walking the RAR headers:

```
SPODS_Dataset/image (1..1088).png
Ground truth (GT1)/{logo,signature,stamp,text}/image (1..1088).png
```

Four **binary pixel masks per page**, one per category. Note what is *not*
there: any notion of *which* logo. A previous study reported "64 logo/stamp
classes" for SPODS with names like `logo_14` — those identities were derived by
that study, not read off the dataset, and nothing verified them. Since class
identity is the entire ground truth of an instance benchmark, that inventory was
a hypothesis with unmeasured error bars. Here the derivation is explicit
(`cluster_marks.py`), flagged (`provenance="clustered"`), and audited.

SPODS is **not offline**, despite having been recorded that way. Its own page
still advertises `www.facweb.iitkgp.ernet.in`, a decommissioned host that 503s;
`facweb.iitkgp.ac.in` serves the same 2.94 GB file. The authors' Scanned
Document Degradation Tool is beside it (`sddt.zip`, 689 MB).

**StaVer** — 400 scanned dummy bills with pixel-accurate stamp GT plus per-file
`info` text (stamp count, colour, overlap, signature presence). Kaggle mirror;
the DFKI original was unreachable when this was written. Again: locations, no
identities. The recorded stamp count is used as an independent check on the mask
decomposition — a page the dataset says has one stamp that decomposes to four
means the merge gap is wrong, which is otherwise invisible.

**Tobacco800** — 1,290 real scanned business documents, 412 carrying a logo,
GEDI XML ground truth from UMD's LAMP. The published logo protocol keeps the 21
categories with **≥2 occurrences**, which cannot support a train-and-search
eval: at two instances, using one as the query leaves one thing to retrieve. Use
`--min-instances` and read the survival curve.

**UCSF IDL** — an open Solr index. Measured live on 2026-08-25:

| query | count |
|---|---:|
| tobacco, 1–2 pages, has `collection` | 13,216,456 |
| tobacco, 1 page, `type:letter` | 1,802,100 |
| `author:"RJR"` 1-page letters | 162,197 |
| `author:"PHILIP MORRIS"` | 73,320 |
| `author:"LOR, LORILLARD"` | 21,120 |

Prefer `author` over `collection`. `collection` is provenance — whose filing
cabinet the page sat in — so a letter *in* the Philip Morris collection is about
as likely to be incoming mail on someone else's letterhead.

## Contamination is prevented by construction, not by annotation

The trap: RVL-CDIP, Tobacco800 and UCSF's Tobacco industry all descend from
IIT-CDIP. An American Tobacco letterhead is *certain* to appear in an RVL-CDIP
"distractor" pool. Unlabelled positives in a distractor set do not make a
benchmark slightly noisy — they make a correct retrieval count as a false
positive, so the metric punishes the model for being right.

No amount of hand annotation fixes that at 200k pages. `CONTAMINATES` in
`docmarks_config.py` encodes which sources may serve as distractors for which
classes, `classes.json` records the resolved list per class, and eval code must
score a class against its `eligible_distractor_sources` rather than the whole
corpus. Tobacco800 classes therefore get UCSF's *non-tobacco* industries; SPODS
and StaVer, whose marks exist nowhere else, get everything.

Note one residual: companies span industries (Philip Morris reaches Food through
Kraft), so industry exclusion is a strong filter, not a proof.

## Tiers

`s`=5k, `m`=50k, `l`=200k pages, **nested**: every page in `s` is in `m` is in
`l`, and all three share class ids, so a result on one is comparable to a result
on another. Positives are in every tier — a tier keeping 3 of a class's 30
instances measures a different and much harder problem, not the same one more
cheaply. Distractors get a stable hash rank and tiers are prefixes of it.

Two stability promises are on offer and they genuinely conflict:

* **within a build** (default): exact budgets, nested. Run on `s`, then on `l`,
  no rebuild.
* **across builds** (`--pin-tiers <earlier build_report.json>`): membership fixed
  by absolute rank cutoff, so a grown source pool cannot evict a page from a
  tier it was already in. Budgets drift instead.

Without pinning, a build over a different page set is a **new corpus version**
and should be named as one. `tests_lib/datasets/test_docmarks_corpus.py` pins
both behaviours, including the negative one.

## The three human passes — and the one that isn't

Each exists because a specific number is otherwise *unknowable*, not merely
unverified. In value order:

1. **`letterhead`** — sample ~100 pages per weak-label author and count how many
   really carry the mark. This is the single highest-value check in the corpus:
   at 90% the UCSF stratum is usable with a noise model, at 40% it is not usable
   at all, and nothing except looking can tell you which. Everything else in the
   haystack layer is downstream of this number.
2. **`cluster`** — confirm the derived SPODS/StaVer classes. One contact sheet
   per class, all instances; single linkage's failure mode (two classes bridged
   by one ambiguous crop) is obvious on sight. Verdicts: `ok`, `split`,
   `merge_into:<id>`, `drop`.
3. **`distinctive`** — one sheet of every class's exemplar, marked `distinctive`
   or `generic`. A plain warning triangle or a ruled box is a *shape*, not an
   *instance*: "find this rectangle" is not a well-posed retrieval query. The
   prior study's worst classes (`warning_diamond` at 17 keypoints,
   `hospital_cross`) are exactly this, and averaging them into a headline AP
   measures the dataset's junk. Generic classes are **kept and labelled**, never
   deleted, so both numbers stay reportable.

Query crops are generated automatically from each class's largest boxed instance
(the prior study measured a 2.2× AP advantage for a clean query over a small
in-scene crop). Weak-label classes have no box, so they are listed in
`build_report.json` under `needs_hand_crop` — one hand-drawn crop each.

**Not a pass:** exhaustively checking the distractor pool for unlabelled
positives. Unfixable by hand at this scale; prevented by construction above.
What *is* worth doing after a run is adjudicating the top-k false positives per
query — a few hundred thumbnails that separate a model error from a missing
label, which no aggregate can do.

## Output

```
corpus.jsonl        one record per page: path, size, marks, provenance, tier
classes.json        class inventory: instances, median px, eligible distractors, audit slots
queries/            one query crop per admitted class
cluster_report.json what the identity clustering did
build_report.json   counts, survival curve, tier cutoffs, rejections with reasons, warnings
```

Every mark carries a `provenance`: `gt` (shipped by the source), `clustered`
(derived here, audit pending), `weak` (metadata-implied, unverified) or
`synthetic` (true by construction). Do not aggregate across them without saying
so.

## Embedding cells

`embed_corpus.py` writes `docmarks_<tier>__<embedder>.pkl` into the shared
pile's `embeddings/` dir, in the pile's own format, via the pile's own pickle
IO. It is deliberately *not* a `pile_config.DATASETS` entry: the pile builds the
full dataset × embedder cross-product, so adding DocMarks and `sift_vlad` there
would silently schedule `sift_vlad` cells for all six existing datasets — a
dozen-odd cells nobody asked for, on a mount the playbook already calls
chronically full.

## Before running this on the grid

`python build_corpus.py --probe` first. Every source fails differently — a
decommissioned hostname, a missing Kaggle token, an absent RAR extractor — and
finding out which costs seconds now and a queue slot later. SPODS needs one of
`bsdtar` / `7z` / `unar` / `unrar`; StaVer and Tobacco800 need a Kaggle token at
`~/.kaggle/kaggle.json` or in `KAGGLE_USERNAME`/`KAGGLE_KEY`.

Then `bash ../preflight.sh` as usual, and size from a real cell rather than a
guess: build tier `s` first and read its actual seconds.
