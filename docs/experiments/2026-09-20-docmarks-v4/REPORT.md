# DocMarks v4.0: the completeness passes removed a degenerate shortcut,
# and the new classes are not yet scored on the same footing

**2026-09-20.** DocMarks moved from v3.1 to v4.0: the roster grew from 23
classes to 27, instances from 721 to 2,004, and the page set lost 145 duplicate
records. This is the re-measure a major bump requires (#4050), plus the three
defects it had to clear first.

## The finding

At v3.1, `source_prior` scored a **perfect 1.00 mean AP** on the eligible pool.
That is not a result, it is a shortcut: "rank pages from this class's source
first" answered the benchmark without looking at a single pixel.

| tier `s`, eligible pool | v3.1 (23 cls) | v4.0 (27 cls) |
|---|---:|---:|
| siglip | 0.20 | 0.18 |
| vlad | 0.01 | 0.01 |
| vlad_rerank | 0.01 | 0.01 |
| rerank_all | 0.05 | 0.07 |
| **source_prior** | **1.00** | **0.92** |

**The completeness passes broke it, not the new classes.** Per class at tier
`m`, the five Tobacco800 classes whose instance count grew are exactly the five
whose prior fell below 1.000, and the five that stayed flat kept it:

| tobacco800 class | v3.1 | v4.0 | `source_prior` |
|---|---:|---:|---:|
| `logo_ajj10e00_1` | 50 | **399** | 0.148 |
| `logo_aah97e00-page02_1_0` | 63 | **370** | 0.186 |
| `logo_aeq93a00_1` | 14 | **130** | 0.212 |
| `logo_ald41a00-ernest_1` | 51 | **206** | 0.272 |
| `logo_afm90c00-first_1_0` | 82 | 112 | 0.869 |
| `logo_asg54f00_1` | 14 | 18 | 1.000 |
| `logo_azb11c00_1` | 33 | 33 | 1.000 |
| `logo_bqz95d00_1` | 10 | 10 | 1.000 |
| `logo_cgr96c00_1` | 9 | 10 | 1.000 |
| `logo_ciy01a00-page02_1_0` | 22 | 24 | 1.000 |

All 11 SPODS and both StaVer classes are still at 1.000, and none of them grew.
Densifying a class inside a 1,290-page source is what stops "rank my own source
first" from isolating it; admitting a class on a 197,077-page source is a
separate change whose effect is confounded below.

**An earlier draft of this report credited the shortcut's collapse to the UCSF
classes. The per-class numbers do not support that.**

## SigLIP is unmoved

Tripling the instance count and adding a fourth source left real retrieval
where it was:

| pool | v3.1 tier `s` | v4.0 tier `s` |
|---|---:|---:|
| eligible | 0.20 | 0.18 |
| own_verified | 0.12 | 0.12 |
| naive | 0.11 | 0.11 |

`vlad` and `vlad_rerank` stay at 0.00–0.01 across every tier and pool,
consistent with #3911: the keypoint budget starves it.

## The four new classes are not scored on the same footing

Tier `m` eligible siglip moved 0.10 → 0.13, and the UCSF classes score far
above everything else — `bat_leaf` 0.42, `rjr_script` 0.47, against
`aah97e00` 0.02. **That is the pool, not the retrieval.**

| source | eligible pool at tier `m` |
|---|---:|
| staver | 49,576 – 49,588 |
| spods | 48,911 – 48,914 |
| tobacco800 | 45,379 – 45,780 |
| **ucsf** | **2,813 – 2,965** |

`eligible_distractor_sources` for each UCSF class is
`[spods, staver, synth, tobacco800]` — **its own source is excluded**, and
correctly so: nobody has verified that the other 197,077 UCSF pages lack the
mark, which is the per-page rule #3921 states. So a UCSF class is ranked
against roughly **2,900** pages while every other class faces **45,000–49,600**,
a 16× difference, with a comparable number of positives (`bat_leaf` 181,
`aah97e00` 354).

Two consequences:

- **Any mean over all 27 classes averages incompatible pools.** The aggregates
  in this report are reported for continuity with v3.1, not because a 27-class
  mean is a meaningful number.
- Their AP *rises* from tier `s` to `m` (0.05 → 0.26 own_verified) because the
  tier adds positives to them but almost no eligible distractors — the mirror
  of the comparability note below.

The fix is to make UCSF negatives knowable for those four marks, so `ucsf`
can join their eligible distractor sources (#3922). Until then they are a
separate measurement that happens to be printed in the same table.

**An earlier draft attributed their high AP to the marks being printed
letterhead logos on clean scans. That explanation is unsupported; the pool size
accounts for it.**

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

- **The four UCSF classes need their own source searched and reviewed** before
  any 27-class aggregate means anything (#3922). Until `ucsf` is an eligible
  distractor for them, v4.0 is two measurements in one table.
- Tier `l` was not re-measured; only `siglip` has a cell there and the run is
  hours. Worth doing before any tier-`l` claim is published.
- Stage 1 (#3928) was not re-measured: its dimensionality-reduced cells were
  built from the pre-dedupe corpus and need rebuilding first.
- `--relabel` reports **2,927 labels on candidate classes not on the roster** in
  every cell. Expected — pages carry cluster-level class ids that never reached
  the roster — but nothing asserts it, so a real drift would look the same.
