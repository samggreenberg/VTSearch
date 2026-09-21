# How many of the pages DocMarks calls negatives actually carry the mark?

**2026-09-20.** Zero, as far as 439 human judgements can see: **0 of 239** in a
detector-free random draw (95% upper bound **≤ 1.26%**) and **0 of 200** at the
top of a SigLIP ranking. Nine planted positives were found nine times out of
nine, so the zeros are evidence that someone looked rather than evidence that
someone clicked.

## Why it was worth asking

`roster.eligible_pages` splits every page into `positive`, `known_negative` (a
source exhaustively checked for this class, so absence is a verified fact) and
`presumed_negative` (a contamination-safe source nobody checked individually).
Censused across the v4.0 roster at tier `m`:

| | page-class pairs |
|---|---:|
| positive | 1,845 |
| known negative | 25,251 |
| **presumed negative** | **1,101,461** |

**97.6% of everything DocMarks scores is a page nobody looked at.** That is not
a flaw — it is the only way to reach 200,000 — and `CONTAMINATES` already
removes the *systematic* risk by construction, per page rather than per source:
a Tobacco800 class is never scored against UCSF's Tobacco industry, the same
IIT-CDIP archive where its letterhead recurs unlabelled. What had never been
measured is the **residual**.

It matters because of the asymmetry the README states: an unlabelled positive
does not make a benchmark slightly noisy, it makes a **correct retrieval score
as a false positive**, so the metric punishes a model for being right.

## Design

Two samples per class, **reported separately and never pooled**:

- **`uniform`** — a random draw from the class's unchecked UCSF pages. No ranker
  touches it, so the bound owes nothing to any detector. With zero hits in *n*,
  the 95% upper bound is 3/*n*.
- **`ranked`** — the top 50 of a SigLIP ranking over the same frame, scored
  exactly as `eval_retrieval` does it. Biased by construction, and that is the
  point: an unlabelled positive at rank 40,000 cannot move a number; one at
  rank 5 moves it a lot.

**No automated pass may create a negative here.** A detector's misses sit
exactly on the faint, small, low-keypoint marks an advanced method should win
on, so clearing a pool with one puts the benchmark's ceiling at the annotation
tool's recall and penalises the next good detector for finding a hard catch.
Exhaustive SIFT keeps its job as an annotation aid proposing *positives*; only a
human or a provenance rule creates a negative.

**Planted attention controls.** Each queue hides one instance the corpus already
knows carries the mark, a third of them two, drawn from the middle of the
class's size distribution, never a page already shown as a reference, shuffled
in before numbering. Nothing on the sheet says which arm a question came from —
not the filename, not the footer — because a footer reading `spods/00882` among
twenty `ucsf/...` pages answers the question for the reviewer, and knowing
SigLIP ranked a page highly is a reason to look harder at it.

Risk is concentrated by source, so the sample is too: a SPODS mark is a logo
invented for the dataset and a StaVer mark is on a German invoice, neither of
which can plausibly appear on a real UCSF scan. The four Tobacco800 classes are
the live case — same era, same American corporate letterhead, and the
Kraft/Philip Morris path is documented in `CONTAMINATES`. The other two are
spot checks confirming a prior, not measurements.

## Result

| arm | n | hits | 95% upper bound |
|---|---:|---:|---|
| uniform (detector-free) | 239 | **0** | **≤ 1.26%** |
| ranked (SigLIP top-50) | 200 | **0** | ≤ 1.50% |

Uniform arm, per class:

| class | n | hits |
|---|---:|---:|
| `tobacco800/logo_aah97e00-page02_1_0` | 50 | 0 |
| `tobacco800/logo_ald41a00-ernest_1` | 50 | 0 |
| `tobacco800/logo_cgr96c00_1` | 50 | 0 |
| `tobacco800/logo_ajj10e00_1` | 49 | 0 |
| `spods/logo_00003_0` | 20 | 0 |
| `staver/stamp_stampds-00213_1` | 20 | 0 |

**Controls: 9 of 9 found.** **Test–retest: 100%** — 84 pages from an earlier
partial pass and 75 from another were judged identically the third time, over
the same images.

## What it licenses, and what it does not

A study may now say *"AP against a pool carrying at most ~1.3% unlabelled
positives"* instead of implying zero. The ranked arm is the more useful half: if
contamination existed at a rate that could distort AP, the top of a similarity
ranking over ~44,000 candidates is where it would surface, and it did not.

It does **not** license treating the whole corpus as clean:

- Six classes were sampled, not 27. Per class the uniform *n* is ~50, so a
  single class's own bound is only ~6%; the ≤1.26% is pooled and assumes the
  sampled classes are representative of the rest.
- The frame is each class's *unchecked UCSF* pages. Anchor-source pages are
  `presumed` too, but they are exhaustively decomposed into marks and clustered,
  so their absence of a mark is near-verified and they were not sampled.
- Nothing here says anything about the four UCSF roster classes, whose own
  source is excluded from their distractors entirely (#3922).

## Reproduce

```
cd scripts/experiments/docmarks
python contamination_sample.py --out <dir>          # renders the queues
python binary_review.py load --queue <dir>/*__contam  # onto the dashboard
```

Verdicts, joined to class, page and arm:
`/expscratch/sgreenberg/docmarks/contamination/verdicts.final.jsonl` (448 rows).
Detector JSONs archived under `keep/detectors-cleared-20260920/`. Seed 20260920,
pinned in the source so the draw is reproducible and cannot be re-rolled to a
nicer answer.

## Follow-ups

- Widen to the remaining 17 anchor classes before any claim that the *corpus*
  is bounded rather than these six.
- The sheet the reviewer sees is fetched whole for every vote, so its weight is
  the labelling cadence: keep it near **124 KB median**. Greyscale at q80 is the
  lever. The reference column took **35% of the canvas** for four crops that
  exist for recognition rather than scrutiny, and narrowing it buys more
  magnification (~1.25×) than trimming page margins ever did (1.03–1.14×).
- #3922 still stands for the UCSF classes, and rung 0 of its ladder — the band
  detector's recall — is the cheapest thing left in that direction.
