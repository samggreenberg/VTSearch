# DocMarks — a cached Stage 1, and what compressing it costs (#3928)

**2026-09-18.** #3928's probe established the shape of DocMarks' Stage 1:
max-over-tiles VLAD ranks the roster where a page-level vector ranks at chance,
and the top 1,000 it hands SIFT keep most of exhaustive verification. It did
that by recomputing every tile from the page images on each run, which is what
made it a probe rather than a stage — tier `m` cost 39 minutes of 40 CPUs to
answer 23 queries, and tier `l`, the 200,000-page tier the corpus was built for,
could not be searched at all.

This builds the stage: a cell stores the tiles once, and a query becomes a
matmul.

**Verdict: the stage is affordable, and the compression that makes it affordable
also makes it better.** A whitened PCA to 256 dimensions stores 26 KiB per page
instead of 818 KiB — **31x smaller** — and ranks *above* the raw 8,192-dimensional
tile it replaces at every shortlist size (AP 0.84 against 0.77 at K = 1,000).
Tier `l` becomes a **5.0 GiB** cell, against 156 GiB raw.

![Stage 1 by stored width](fig_width.png)

## The obstacle is storage, and the arithmetic says so before anything is built

A tile VLAD is 64 x 128 = 8,192 dimensions and ~64 tiles survive a page, so at
fp16 a page costs 818 KiB (measured, tier `s`). Projected to `D` dimensions it
costs `D` x 2 x 64 bytes:

| stored width | KiB/page (measured) | tier `s` (5,000) | tier `m` (50,000) | tier `l` (200,000) |
|---|---:|---:|---:|---:|
| 8,192 (raw) | 818 | 3.90 GiB | 39 GiB | **156 GiB** |
| 512 | 51.6 | 0.25 GiB | 2.5 GiB | 9.8 GiB |
| 256 | 26.1 | 0.12 GiB | 1.2 GiB | **5.0 GiB** |
| 128 | 13.3 | 0.06 GiB | 0.6 GiB | 2.5 GiB |

156 GiB was never available: `/expscratch` had 249 GB free on the night this
ran, with a concurrent `coco_quarry` embed wanting ~36 GB of it. So the question
was never whether to compress, only how far — and that is a measurement.

## What was built

`stage1_cell.py` writes one cell per width in a single pass over the pages:

- **Tiles** are the layout #3928 measured as the better of two: 0.25 x 0.18 of a
  page in normalised coordinates, stride half a tile, keeping tiles with at
  least 20 keypoints. The geometry is pinned against the probe by a test, because
  the probe owns the published numbers.
- **The projection** is a PCA with whitening, fitted on ~20,000 tiles sampled
  from 400 pages and applied to every tile. Whitening scales each component by
  its own standard deviation, so the first `D` rows of a wider fit *are* the
  `D`-wide fit — which is why one pass writes every width, and a test pins it.
- **Rows are L2-normalised after projection**, so a stored dot product is a
  cosine and a page score is a segment max (`numpy.maximum.reduceat`) over one
  matmul.
- **Stage 2 is replayed, not re-run**: each shortlist is re-ordered by the SIFT
  inliers `eval_sift_rank.py` saved at the same tier and budget, which is exactly
  what verifying that shortlist would compute.

## Results, tier `s` (5,000 pages, 23 roster classes, v3.1 corpus)

| stored width | alone | K = 100 | K = 500 | K = 1,000 | K = 2,000 | recall@1,000 |
|---|---:|---:|---:|---:|---:|---:|
| 8,192 (raw) | 0.43 | 0.61 | 0.73 | 0.77 | 0.82 | 0.83 |
| **512** | **0.54** | **0.70** | **0.80** | **0.85** | 0.86 | **0.93** |
| **256** | 0.48 | 0.65 | 0.80 | 0.84 | **0.88** | 0.92 |
| 128 | 0.44 | 0.61 | 0.76 | 0.81 | 0.86 | 0.89 |
| *SIFT, every page* | | | | *0.88* | *0.88* | |

**The projection is not a compromise; it is an improvement.** Every projected
width beats the raw tile it compresses, at every K. At 512 dimensions the gain is
+0.08 AP at K = 1,000 (0.85 against 0.77) for 16x less storage. This is the
known effect of whitening on VLAD — the raw descriptor is dominated by
co-occurrence directions shared by every document page, and dividing each
component by its own spread stops those directions deciding the cosine.

**At K = 2,000, 256 dimensions matches verifying every page** (0.879 against
0.876), while looking at 40% of the tier.

### Where it fails, literally

- **`tobacco800/logo_cgr96c00_1`**: exhaustive SIFT 0.89, shortlisted 0.25,
  Stage-1 recall@1,000 **0.25**, Stage 1 alone **0.00**. Its query crop is a
  small binarised device with 117 SIFT keypoints — too few for a tile vector to
  describe. The probe found the same class at 0.00; nothing here fixes it.
- **`tobacco800/logo_aah97e00-page02_1_0`**: exhaustive 1.00, shortlisted 0.50,
  recall **0.48**. This is the Philip Morris class whose members carry two
  different crests — the globe and the PM monogram — which the owner ruled one
  mark on 2026-09-17. One query crop can only match one of them, so half the
  members never enter the shortlist. This is the case multiple query crops
  (#3949) exist for, and it is measurable evidence for that pass rather than a
  Stage-1 defect.
- **`tobacco800/logo_ald41a00-ernest_1`**: exhaustive 0.97 against 0.88.

### Where the projection rescues a class the raw tile lost

- **`spods/stamp_00612_1`**: raw **0.01** -> 256 dims **0.62**, against 0.54 for
  exhaustive SIFT. The raw tile vectors ranked this class at chance; whitening
  makes it findable.
- **`spods/stamp_00293_1`**: 0.57 -> 0.94. **`tobacco800/logo_bqz95d00_1`**:
  0.89 -> 1.00. **`logo_ajj10e00_1`**: 0.82 -> 0.98.

### Where a shortlist beats verifying everything

Four classes score higher behind the shortlist than under exhaustive SIFT:
`staver/stamp_stampds-00213_1` (0.53 -> 0.65), `spods/stamp_00612_1` (0.54 ->
0.62), `tobacco800/logo_aeq93a00_1` (0.34 -> 0.44) and `logo_asg54f00_1` (0.66
-> 0.76). The probe saw the same effect: printed confusers that match SIFT do
not match the tile ranking, so the shortlist keeps them out.

## Cost

![storage and search cost](fig_cost.png)

| stored width | cell size | load | search |
|---|---:|---:|---:|
| 8,192 (raw) | 3.90 GiB | 8.2 s | **7,790 ms/query** |
| 512 | 0.25 GiB | 0.6 s | 509 ms/query |
| 256 | 0.12 GiB | 0.3 s | **288 ms/query** |
| 128 | 0.06 GiB | 0.2 s | 179 ms/query |

Search is one matmul plus a segment max, single-threaded, over the whole tier.
Against it, exhaustive SIFT verification is ~100 s per query at 50,000 pages
(#3911), so the cached stage answers in under a third of a second what
verification needs minutes for — and the raw cell is 27x slower than the
projected one for a worse ranking.

**Build cost is dominated by SIFT extraction and paid once.** Tier `s` (5,000
pages, 40 CPUs) took 50 minutes *because the raw width was built alongside*: a
raw tile matrix is 818 KiB per page and shipping it back from every worker made
the build IPC-bound at 1.7 pages/s, against the probe's 21 pages/s. **Build only
projected widths and that cost disappears** — the projected payload is 26 KiB per
page. At the probe's rate, tier `l` is ~2.7 h on 40 CPUs.

**The PCA fit is a serial bottleneck**: the SVD of ~20,000 x 8,192 float32 took
~20 minutes before any page was tiled. It is paid once per tier, and a randomised
SVD or an eigendecomposition of the 8,192 x 8,192 covariance would cut it;
neither was tried here.

## Is tier `l` searchable now?

**On storage and search, yes.** At 256 dimensions the cell is **5.0 GiB** and a
query is a matmul over 200,000 pages: the tier-`s` search cost scales linearly
with pages, so ~12 s per query single-threaded, or well under a second sharded
across the cores a node already has.

**On labels, not yet.** Nothing can report end-to-end AP at tier `l`, because
Stage 2 has never been run exhaustively there — that is the cost this stage
exists to avoid. What *is* measurable without it is Stage-1 recall@K against the
known members, and `eval_stage1_cell.py` reports exactly that when `--inliers` is
omitted. Building and measuring it is the natural next step, at ~2.7 h of 40 CPUs
and 5 GiB.

## Caveats

- **The corpus moved under the comparison.** These numbers are on the **v3.1**
  corpus (721 instances; the completeness pass added 108 members on 2026-09-17),
  where the probe ran on the pre-completeness corpus. Exhaustive SIFT scores 0.88
  here against 0.78 then, and the raw arm scores 0.43 alone against the probe's
  0.40. **Compare within this report, not across to #3928's table.**
- **One query crop per class**, 23 classes. The reviewed extra crops (#3949) are
  not countersigned yet, so averaging over them would report a number nobody has
  approved.
- **The headline pool still counts Tobacco800 pages carrying a class's mark as
  negatives** (#3927), so Tobacco800 rows are lower bounds for every ranker.
- **Tile coverage stops half a tile short of the bottom edge** (the last window
  ends at 0.99 of the page height). A mark in that strip is seen only through the
  window overlapping it. This is the probe's geometry, kept deliberately so the
  numbers stay comparable, and pinned by a test.

## Reproduce

```
source scripts/experiments/pile/pile_env.sh
cd scripts/experiments/docmarks
python stage1_cell.py build --tier s --dims 0,512,256,128 --workers 40 --out <dir>/tier-s
python eval_stage1_cell.py --tier s \
    --cells <dir>/tier-s/raw,<dir>/tier-s/d512,<dir>/tier-s/d256,<dir>/tier-s/d128 \
    --inliers <sift run>/inliers.json --out <dir>/eval-s
python docs/experiments/2026-09-18-docmarks-stage1-3928/figures.py
```
