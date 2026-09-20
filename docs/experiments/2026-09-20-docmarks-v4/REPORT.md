# DocMarks v4.0: the UCSF classes remove a degenerate shortcut

**2026-09-20.** DocMarks moved from v3.1 to v4.0: the roster grew from 23
classes to 27, instances from 721 to 2,004, and the page set lost 145 duplicate
records. This is the re-measure a major bump requires (#4050), plus the three
defects it had to clear first.

## The finding

At v3.1, `source_prior` scored a **perfect 1.00 mean AP** on the eligible pool.
That is not a result, it is a shortcut: every roster class lived on a source of
400–1,290 pages, so "rank pages from this class's source first" answered the
benchmark without looking at a single pixel.

The four UCSF classes end it. Their source is 197,077 pages, so the prior
carries almost no information about which page holds the mark.

| tier `s`, eligible pool | v3.1 (23 cls) | v4.0 (27 cls) |
|---|---:|---:|
| siglip | 0.20 | 0.18 |
| vlad | 0.01 | 0.01 |
| vlad_rerank | 0.01 | 0.01 |
| rerank_all | 0.05 | 0.07 |
| **source_prior** | **1.00** | **0.92** |

`source_prior` on the UCSF classes alone is **0.58** at tier `s` and **0.54** at
`m`, against 1.00 for every anchor-source class at v3.1. The benchmark is
harder in the way a benchmark should be: a method now has to look at the mark.

## SigLIP is unmoved, and likes the new classes

Tripling the instance count and adding a fourth source left real retrieval
where it was:

| pool | v3.1 tier `s` | v4.0 tier `s` |
|---|---:|---:|
| eligible | 0.20 | 0.18 |
| own_verified | 0.12 | 0.12 |
| naive | 0.11 | 0.11 |

Tier `m` eligible siglip moved 0.10 → 0.13. The lift is the new classes, which
SigLIP handles markedly better than anything already in the roster:

| tier `m`, own_verified, siglip | classes | mean AP |
|---|---:|---:|
| **ucsf** | 4 | **0.26** |
| tobacco800 | 10 | 0.10 |
| spods | 11 | 0.08 |
| staver | 2 | 0.07 |

The UCSF marks are printed letterhead logos on clean scans — `bat_leaf` at a
median 55 × 41 px, `rjr_script` at 280 × 59 — rather than the rubber-stamp
impressions and binarised litigation scans that dominate the rest.

`vlad` and `vlad_rerank` stay at 0.00–0.01 across every tier and pool,
consistent with #3911: the keypoint budget starves it.

## Comparability

**These two columns are adjacent, not comparable**, which is why v4.0 is a major
bump rather than v3.2. The roster changed, the positive sets changed, and the
page list changed. The datasheet states the rule and the consequence: at a major
bump the baselines are re-run rather than re-scored.

One consequence is specific and easy to trip over. For the four UCSF classes the
positive set **grows with the tier** — 10 instances in `s`, 256 more in `m`, 55
more in `l` — so their tier-`s` and tier-`l` numbers are not measured over the
same ground truth. The other 23 classes are unaffected: every instance sits on
an anchor page, and every anchor page is in `s`.

## Three defects cleared first

The re-measure would not have been defensible without these.

1. **A held vote was applied as its opposite** (#4052, PR #4053). A Good vote
   parked for a hand-drawn box stayed out of `verdict`, and the applier reads
   every index outside `verdict` as a rejection — so it became a cannot-link or
   an unboxed rejection, which `completeness_multi.decided()` then reads back as
   settled. #4041 had narrowed the trigger without closing the hole. Caught on a
   real vote: `tobacco800/logo_ciy01a00-page02_1_0` candidate 23.

2. **145 duplicate page records** (#4054, PR #4055). `corpus.jsonl` held 200,000
   records over 199,855 distinct ids, all duplicates UCSF. **13 groups disagreed
   about the marks**, all of them roster pages: one copy carried the roster mark
   and the other did not, so a dict-building reader lost the mark and a
   streaming scorer saw one page as a positive *and* a negative for the same
   class. It landed on `logo_aah97e00-page02_1_0` and the new `ucsf/logo_bat_leaf`.

3. **The cells inherited those duplicates** (PR #4057). 1 extra row at tier `s`,
   31 at `m`, 145 at `l`.

## What the repair cost, and what it could have cost

Fixing (3) looked like a cell rebuild: roughly **15 h of GPU**, since tier `m`
`sift_vlad` alone is documented at 11 h. It was not, because **no page's pixels
changed**. A duplicate is the same image at the same path, so both its rows
carry the same vector.

`embed_corpus.py --prune` drops the rows by streaming the cell through
`_rewrite_cell`, the same machinery `--repair` uses. Prune plus relabel:

| step | result | time |
|---|---|---|
| `--prune --force` | 209 rows dropped; cells now 4,999 / 49,969 / 199,855 | ~1 min |
| `--relabel --force` | 3,765 medias relabelled across five cells | ~2 min |

The general rule, and the reason all three repairs exist: **a cell is labels, a
row list and vectors, and only the vectors are expensive.** Before re-embedding,
ask what actually changed.

## Reproduce

```
# corpus is at v4.0 already; this re-runs the measurement
source scripts/experiments/pile/pile_env.sh
export OMP_NUM_THREADS=1
cd scripts/experiments/docmarks
python eval_retrieval.py --tiers s --rerank-all --out <out>/s   # ~35 min, 8 CPUs
python eval_retrieval.py --tiers m --out <out>/m                # ~3 min
```

Results: `/expscratch/sgreenberg/docmarks/remeasure-v4/{s,m}/` (job 670431,
38 min). Corpus numbers and the versioning rule are in
[`DATASHEET.md`](../../../scripts/experiments/docmarks/DATASHEET.md).

## Follow-ups

- Tier `l` was not re-measured; only `siglip` has a cell there and the run is
  hours. Worth doing before any tier-`l` claim is published.
- Stage 1 (#3928) was not re-measured: its dimensionality-reduced cells were
  built from the pre-dedupe corpus and need rebuilding first.
- `--relabel` reports **2,927 labels on candidate classes not on the roster** in
  every cell. Expected — pages carry cluster-level class ids that never reached
  the roster — but nothing asserts it, so a real drift would look the same.
