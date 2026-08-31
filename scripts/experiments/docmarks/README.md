# DocMarks — eval data for stamp detection

Labelled data for evaluating **structural similarity search on small marks in
scanned documents** — finding a given stamp, seal or letterhead logo in a pile
of pages. Built for the feature the structural embedder is heading toward, and
for the experiments that will decide how it should work.

The 2026-07-13 study found the first configuration where structural search beats
the deep embedder on a real corpus (SuperPoint+LightGlue, AP 0.395/0.481 vs
SigLIP's 0.204/0.235), then ran out of road: its two document corpora are 259
and 1,088 pages, with as few as 9 instances per class, and their class
identities were derived by that study rather than verified by anyone.

```bash
python build_corpus.py --probe                       # can I reach every source?
python build_corpus.py --sources spods               # cluster into candidates
python shortlist.py --corpus <dir> --write-roster     # rank them, draft a roster
$EDITOR <dir>/roster.json                             # pick your two dozen
python build_corpus.py --sources spods --roster <dir>/roster.json

python make_audit_slate.py --task membership          # verify every instance
python make_audit_slate.py --task confusable          # adjudicate every pair
python audit_to_corrections.py --task membership --apply

source ../pile/pile_env.sh
python embed_corpus.py --tier s --embedders sift_vlad,siglip
```

## The design in one idea: a curated roster, not an inventory

The corpus holds two populations with completely different standards of
evidence, and keeping them apart is the whole point:

- **Roster classes** — a small, named, checked-in set of about two dozen. Every
  instance is adjudicated in or out by hand; every confusable pair is
  adjudicated same or different. Nothing enters by heuristic.
- **Distractors** — everything else, unlabelled and unexamined, in whatever
  quantity the tier budget allows. They need no labels, only to be *safe to
  score against*.

That split is what buys a trustworthy eval at a cost a person can pay.
Verifying 24 classes exhaustively is an afternoon; verifying 400 is not, and a
benchmark whose labels nobody checked is a benchmark whose numbers nobody should
quote. Without a `--roster`, the builder emits candidate classes for
`shortlist.py` to rank — those are *proposals*, and the build says so.

## Both directions of the ground truth

An eval for "find this mark" needs two kinds of label, and clustering can only
ever propose one:

- **same** — a shared `class_id`. These instances must all be found.
- **different** — a recorded separation. These must be told apart.

The second is what usually goes missing, and its absence is invisible: without
it, the only thing keeping two similar marks in separate classes is where a
distance threshold happened to land, so nudging the threshold silently rewrites
the ground truth. Measured on the fixture corpus at a loose threshold, three
distinct marks collapse into one class — unless the separations are on disk, in
which case all three survive.

So a `different` verdict is stored permanently in `separations.json`, keyed on
**page ids** (which survive a re-cluster; class ids do not) and enforced as a
cannot-link constraint on every future run. The constraint propagates, so two
separated marks cannot be reunited through some third ambiguous crop.

## What each source ships

**SPODS** — 1,088 scanned pseudo-official pages, direct download, no
registration. Confirmed by walking the RAR headers:

```
SPODS_Dataset/image (1..1088).png
Ground truth (GT1)/{logo,signature,stamp,text}/image (1..1088).png
```

Four **binary pixel masks per page**, one per category. Note what is *not*
there: any notion of *which* logo. A previous study reported "64 logo/stamp
classes" for SPODS with names like `logo_14` — those identities were derived by
that study and never verified. Since class identity is the entire ground truth
of an instance benchmark, that inventory was a hypothesis with unmeasured error
bars. Here the derivation is explicit (`cluster_marks.py`), flagged
(`provenance="clustered"`), and only becomes ground truth after the membership
audit.

SPODS is **not offline**, despite having been recorded that way: its own page
advertises `www.facweb.iitkgp.ernet.in`, a decommissioned host that 503s, while
`facweb.iitkgp.ac.in` serves the same 2.94 GB file. The authors' Scanned
Document Degradation Tool sits beside it (`sddt.zip`, 689 MB).

**StaVer** — 400 scanned dummy bills, pixel-accurate stamp GT, per-file `info`
text (stamp count, colour, overlap). Kaggle mirror; DFKI's original was
unreachable. Locations, no identities. The recorded stamp count is used as an
independent check on the mask decomposition — a page the dataset says has one
stamp that decomposes to four means the merge gap is wrong.

**Tobacco800** — 1,290 real scanned business documents, 412 with a logo, GEDI
XML ground truth from UMD's LAMP. Its published protocol keeps the 21 logo
categories with ≥2 occurrences, which cannot support a train-and-search eval.

**UCSF IDL** — an open Solr index; measured live 2026-08-25:

| query | count |
|---|---:|
| tobacco, 1–2 pages, has `collection` | 13,216,456 |
| tobacco, 1 page, `type:letter` | 1,802,100 |
| `author:"RJR"` 1-page letters | 162,197 |
| `author:"PHILIP MORRIS"` | 73,320 |

**`author` is a candidate pool, not a class.** The field asserts a page is
*from* a company; it has never looked at the mark. Making it a class id writes
two guaranteed errors into the ground truth: a company that redesigned its
letterhead yields one class holding two artworks (so a detector is punished for
telling them apart), and two subsidiaries sharing artwork yield two classes
holding one mark (so it is punished for recognising it). Those are exactly the
errors the eval exists to measure. So the author narrows millions of pages to a
high-yield pool, each candidate gets a coarse top-of-page band to locate the
mark, and identity is settled by clustering plus adjudication like everything
else. `documentdate` is recorded and never enters a class id: era is a fact
about the calendar, not about the mark.

Distractors only: `--ucsf-letterhead-per-author 0`.

**Synth** — real artwork (LogoDet-3K, or any `--synth-pool-dir`) pasted onto
held-out real scans at known `(x, y, scale, rotation)` with scanner-style
degradation. Exact ground truth, and the only stratum that can be *swept*: size,
rotation and count are inputs, so an experiment can locate the ~32px floor or
the inlier-count working point instead of straddling it. The rule that comes
with it: **synthetic numbers quantify a mechanism, real numbers size the
effect.** A finding that appears only in `synth` is a hypothesis about the
pipeline, not a claim about documents.

## Three kinds of negative

Not all distractors are equal, and the manifest keeps them distinct:

- **known negative** — a page from a source exhaustively checked for this class,
  so its *absence* of the mark is verified. These are the valuable ones: same
  scanner, same paper, same era, known clean. A SPODS page carrying a different
  mark is the hardest possible negative for a SPODS class, and the membership
  audit is what makes it usable instead of a contamination risk.
- **presumed negative** — from a contamination-safe source nobody checked
  individually. Fine in bulk, and the only way to reach 200k.
- **excluded** — a contamination risk, never scored.

The trap that last category exists for: RVL-CDIP, Tobacco800 and UCSF's Tobacco
industry all descend from IIT-CDIP, so an American Tobacco letterhead is
*certain* to appear in an RVL-CDIP "distractor" pool. Unlabelled positives don't
make a benchmark slightly noisy — they make a correct retrieval count as a false
positive, so the metric punishes the model for being right. No hand pass fixes
that at 200k pages; `CONTAMINATES` in `docmarks_config.py` fixes it by
construction, and each class records its resolved
`eligible_distractor_sources`.

## Tiers

`s`=5k, `m`=50k, `l`=200k pages, **nested**: every page in `s` is in `m` is in
`l`, sharing class ids, so a result on one is comparable to a result on another.
Roster positives are in every tier — a tier keeping 3 of a class's 30 instances
measures a different and harder problem, not the same one more cheaply.
Distractors get a stable hash rank and tiers are prefixes of it.

Two stability promises are on offer and they genuinely conflict:

- **within a build** (default): exact budgets, nested. Run on `s`, then `l`, no
  rebuild.
- **across builds** (`--pin-tiers <earlier build_report.json>`): membership fixed
  by absolute rank cutoff, so a grown source pool cannot evict a page from a
  tier it was already in. Budgets drift instead.

Without pinning, a build over a different page set is a **new corpus version**.
Both behaviours are pinned by tests, including the negative one.

## The human passes

In the order you run them. Only the first two are needed for a first eval.

1. **`membership`** — every instance of every roster class, numbered on contact
   sheets. Verdict is `ok` or the indices that are *not* this mark (`3,17`), so
   a 30-crop class is one line. Afterwards no positive is unexamined, which is
   what lets a miss be blamed on the detector rather than the label. A rejected
   crop keeps its box and stays on its page — it becomes a known negative.
2. **`confusable`** — every roster pair side by side, ranked by distance. 24
   classes is 276 pairs, so the full matrix is adjudicated rather than sampled.
   `same` sends you to `merge_into:` on the cluster task; `different` writes a
   permanent separation.
3. **`cluster`** — is a class one mark at all? Mostly useful while choosing a
   roster. `split` is productive: it re-clusters that class alone at half the
   threshold and re-sheets the pieces, disturbing nothing else.
4. **`distinctive`** — mark vs shape. A plain warning triangle or ruled box is a
   *shape*: "find this rectangle" is not a well-posed retrieval query. The prior
   study's worst classes (`warning_diamond` at 17 keypoints, `hospital_cross`)
   are exactly this. Generic classes are kept and labelled, never deleted.
5. **`letterhead`** — for the later UCSF expansion: sample bands per candidate
   author and count how many carry a printed mark at all. Decides whether that
   pool is worth clustering.

Query crops come from each class's largest boxed instance automatically (the
prior study measured a 2.2× AP advantage for a clean query over a small in-scene
crop). Band-located classes get none — auto-cropping the strip would hand the
query a banner of letterhead plus address plus rule line and call it a logo,
which is worse than no crop because it looks like ground truth. They are listed
in `build_report.json` under `needs_hand_crop`.

## Output

```
corpus.jsonl        one record per page: path, size, marks, provenance, tier
classes.json        per class: instances, distinct_from, caveats, eligible distractors, audit state
roster.json         the hand-picked classes an eval runs on
separations.json    adjudicated "different mark" pairs, keyed on page ids
queries/            one query crop per box-located roster class
shortlist.png/json  ranked candidates for choosing a roster
cluster_report.json what clustering did, and how many separations it honoured
build_report.json   counts, survival curve, tier cutoffs, rejections, warnings
```

Every mark carries a `provenance`: `gt` (a box shipped by the source),
`clustered` (identity derived here), `clustered_band` (identity derived from a
coarse strip, so the box locates a region and not the mark), `candidate` (pool
member, no identity yet) or `synthetic` (true by construction). A class also
carries `audit.membership_verified` — **false means it is still a proposal**.
Do not aggregate across provenances without saying so.

## Embedding cells

`embed_corpus.py` writes `docmarks_<tier>__<embedder>.pkl` into the shared
pile's `embeddings/` dir, in the pile's format, via its pickle IO. It is
deliberately *not* a `pile_config.DATASETS` entry: the pile builds the full
dataset × embedder cross-product, so adding DocMarks and `sift_vlad` there would
silently schedule `sift_vlad` cells for all six existing datasets, on a mount
the playbook already calls chronically full.

## The clustering threshold is measured, not chosen

Single linkage does not degrade gracefully. Measured on SPODS's 2,096 real
marks:

| threshold | classes | largest component | share | classes with ≥10 |
|---:|---:|---:|---:|---:|
| 0.005–0.030 | 1,153 | 31 | 1.5% | 32 |
| **0.040–0.060** | **728** | **60** | **2.9%** | **41** |
| 0.080 | 458 | 382 | 18.2% | 43 |
| 0.100 | 230 | 1,021 | 48.7% | 28 |
| 0.180 | 34 | 1,818 | 86.7% | 9 |

Read the **share** column, not the class count: at 0.18 the corpus reports 34
classes, which sounds like an inventory and is actually one blob of 1,818 marks
with a tail. An earlier default of 0.18 — validated on a three-class fixture —
produced exactly that, and it did not look like an error.

The plateaus are wide because pHash distances are quantised to multiples of
1/64, so every threshold between two steps gives an identical partition; 0.04,
0.05 and 0.06 are the same corpus. That makes 0.05 a safe working point rather
than a knife edge.

Re-run `tune_clustering.py` whenever the source set or the descriptor changes.
The number is a property of the data and does not travel.

## Looking at what you built

```bash
python make_report.py --corpus <dir> --out docs/reports/<date>-docmarks.html
```

One self-contained HTML page: counts per source and provenance, whole pages with
marks boxed in situ (the only way to see how small the target is), every class
as a strip of its own instances, the distractor pool, and the mark-size
distribution against the 32px structural floor. Images are inlined, so the file
survives being archived or opened on a machine with no access to `/expscratch`.

## Running it at full scale

A tier-`s` SPODS-only build fits on a laptop. Tiers `m` and `l` need the
cluster — see **[`GRID-RUNBOOK.md`](GRID-RUNBOOK.md)** for sizing, staging, the
resume story and what to check afterwards.

`python build_corpus.py --probe` first, wherever you run. Every source fails
differently — a decommissioned hostname, a missing Kaggle token, an absent RAR
extractor — and finding out which costs seconds now and a queue slot later.
SPODS needs one of `bsdtar` / `7z` / `unar` / `unrar`; StaVer and Tobacco800
need a Kaggle token.
