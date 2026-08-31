# DocMarks v0 — what the stamp-detection corpus looks like

> **Superseded in part by [v1](../2026-08-31-docmarks-v1/REPORT.md).** The look-at-it
> pass below stands, but two of its numbers were artefacts of a mask-decomposition bug
> ([#3361](https://github.com/samggreenberg/VTSearch/issues/3361)): the bimodal size
> distribution and the 45.95% "largest mark". Fixing it also invalidated the 0.16
> clustering threshold read off these marks. Kept as written — it is the evidence.

**2026-08-31.** The pictures are the point, and they are in the self-contained
page beside this file: [`report.html`](report.html) (open it in a browser —
GitHub renders raw HTML as source). 356 real images: whole pages with their
marks boxed in situ, every class as a strip of its own instances, the distractor
pool, and the size distributions.

Regenerate with the committed generator, which reads a built corpus and types no
number twice:

```
python scripts/experiments/docmarks/make_report.py \
    --corpus <corpus> --out docs/experiments/2026-08-31-docmarks-v0/report.html
```

**Question.** Not "how good is the detector" — nothing is being scored here.
This is the look-at-it pass on a corpus nobody had seen: is it the right data,
and are its labels worth anything yet?

**Verdict.** The data is right and the labels are not, in a way that only
looking could have shown.

## What is in it

| | count |
|---|---:|
| SPODS pages | 1,088 |
| UCSF distractor pages | 432 |
| logo marks | 1,103 |
| stamp marks | 993 |
| signature marks (recorded, never queryable) | 975 |
| pages carrying ≥1 logo or stamp | 1,062 |

Median mark **276 px** on the longest side (p10 74, p90 420), covering **0.76%**
of its page. **Zero** marks fall below the 32 px floor the 2026-07-13 study
measured, so this corpus sits entirely inside the regime where structural
matching has a chance.

The page-area distribution is **bimodal** — a spike near 0.05% and a broad hump
from 0.5% to 2% — so "median 0.76%" describes no actual mark. Any per-size
analysis should band rather than average.

## Three things the pictures caught that the counts did not

**The masks were inverted.** SPODS ships 1-bit masks with the mark *black* on
white paper. Read as "non-zero is foreground" this selects the paper: 2,176
page-sized "marks", two per page, median longest side 3,480 px — the full page
width. It did not crash. It produced a `classes.json` that read as a working
corpus with one class in it. Fixed by detecting polarity from the minority
phase; the resulting 0.2–1.1% marked fraction matches the prior study's "median
mark ≈ 1.3% of the page", which is the independent check that the fix is right.

**The threshold was tuned on a fixture.** At the inherited 0.18, single linkage
put 87% of all marks in one component while reporting 34 classes — a number that
reads like an inventory. Read the largest-component share, never the class
count.

**64 bits could not tell two round stamps apart.** A red *book* stamp (5
instances) and a blue *elephant* stamp (27) merged into one class, and no
threshold separated them: one pair at Hamming 2/64 bridged the set, so it was
one group at 0.04 and 21 fragments at 0.03. A stamp's border ring is big, smooth
and low-frequency; the interior that says *which* stamp is not, so an 8×8 DCT
block encodes "is a round stamp". At 16×16 with a radial taper the same class
resolves to exactly {27 elephants} + {5 books}.

One thing that already worked: the descriptor is greyscale, and the elephant
appears in blue on 26 pages and red on one. It lands in a single class, which is
correct — ink colour is not identity.

## Status of the labels

**Unverified, and the report says so at the top of the page.** Every class here
was proposed by clustering. The corpus now runs a deliberately strict partition
and expects hand-merging, with both directions of every human decision recorded
in `adjudications.json` and replayed on each re-cluster — but that pass has not
been done, so nothing in this report is ground truth.

Encouraging sign, not evidence: 27 of 41 classes hold exactly 31 instances,
consistent with SPODS placing each mark on ~30 pages by design.

Cited from [`docs/plans/structural-embedder.md`](../../plans/structural-embedder.md).
