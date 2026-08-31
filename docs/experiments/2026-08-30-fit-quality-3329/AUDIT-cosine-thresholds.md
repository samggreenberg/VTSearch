# Cosine-magnitude thresholds, audited for patch embedders (issue #3347)

Action item 5 of the [#3329 fit-quality report](REPORT.md). That report found
`dinov3_patch`'s embedding space to be far less concentrated than the four
single-vector spaces, four independent ways, and closed with a standing warning:
the coverage atlas's fixed-α typicality guard is *an* instance of a constant
fitted on single-vector cosine magnitudes, **not necessarily the only one**.

This is the inventory. Every place in `vtscore/` and `vtsearch/` where a
constant meets a cosine similarity or a cosine distance, what it was fitted on,
whether it runs on a patch embedder's vectors, and whether it still means there
what it meant where it was fitted.

**Headline: one live-path hit, and it holds.** `_CALIBRATION_MIN_RBAR` in the
coverage atlas is the only fixed cosine-magnitude constant on the click loop,
and #3329's own r̄ table clears it by ~4× at the worst measured decile of the
worst embedder. Two other real hits are inert — one has no consumer, one is an
experiment arm. Everything else in the loop draws its line from the data it is
handed, which is why the sweep came back this quiet.

---

## Method, and what would have been missed

The grep the issue asked for is `constant [<>] cosine`, and on its own it finds
**nothing** — no line in the production tree compares a similarity to a literal.
That result is misleading, and worth recording so the next pass does not stop
there. The two real code hits are a **multiplication** (`sims * 4.0`) and a
**convex blend** (`alpha * cos_d + …`); neither is a comparison. The sweep that
found them looks for a numeric literal anywhere in a statement that also names a
cosine-valued quantity, then reads each hit.

Scope is production code: `vtscore/` and `vtsearch/`, plus `frontend/src/`.
`scripts/experiments/` is deliberately out — those are per-study arms whose
constants are the study's subject. `vtscore/eval/` is **in**, because the
default arm is the shipped algorithm (see `CLAUDE.md`); its named arms are
called out as such where they appear.

A "hit" is a constant whose meaning changes with the absolute scale of the
cosines it meets. A rank, a quantile, a ratio's indifference point, or a
threshold refit on the observed distribution is scale-free and is **not** a hit,
however many literals sit near it.

---

## The hits

### 1. `_CALIBRATION_MIN_RBAR = 0.1` — the only one on the live path

`vtscore/state/coverage_atlas.py`. A floor on each node's mean resultant length
r̄ = ‖Σ unit vectors‖ / n, which is a cosine magnitude: the average cosine of a
node's members to their own mean direction. Two consumers, and they are not
equally exposed:

| consumer | reachability |
|---|---|
| node calibration → `typicality_pvalues` → `domain_shift_report` | one on-demand HTTP endpoint, and #3351 now refuses it outright on patch embedders |
| **`CoverageAtlas.next_sample`** | **the click loop** — autopilot's `new` phase and the diversity probe |

The second is the exposure that matters, and #3329 traced it explicitly:
`next_sample` "reads `node["ids"]`, which is sorted by raw `mu · x`, and gates
on `node["rbar"]`". The gate decides whether to probe a node's **typical half**
or the **whole node**. Below the floor the typicality ordering is treated as
noise and the whole node is probed.

Was the constant fitted? No — it is a degeneracy guard, aimed at the root, whose
resultant vanishes by construction after centering. That provenance is what
saves it: it is not asserting "this space is concentrated to degree 0.1", it is
asserting "this node has *a* direction at all".

Checked against measurement rather than assumed. #3329's part-2 grid reports
node r̄ per embedder:

| embedder | node r̄ (median) | r̄ 10th pct |
|---|---|---|
| `clip` | 0.70 | 0.53 |
| `siglip` | 0.69 | 0.52 |
| `siglip2_l` | 0.67 | 0.49 |
| `clip_l` | 0.66 | 0.50 |
| **`dinov3_patch`** | **0.61** | **0.38** |

The worst decile of the least concentrated space sits at 3.8× the floor.
**Verdict: correct on patch spaces, with real margin.** The residual is that the
grid reports a median and a 10th percentile, not a full distribution, and
`dinov3_patch`'s atlas splits to the `max_depth` cap of 10 — deeper nodes are
smaller and less concentrated. Nothing below the 10th percentile was measured.
The margin recorded above is now in the constant's comment, so the next reader
inherits the check instead of redoing it.

### 2. `torch.softmax(sims * 4.0)` — real, and inert

`vtscore/media/patch_embed.py`, the CLS-cosine saliency proxy. An inverse
temperature applied directly to per-patch cosines against the CLS vector. How
peaked the resulting map is depends entirely on the spread of those cosines, so
4.0 means something different in every embedding space.

Provenance is the giveaway: the comment said it was "empirically tunable in the
caltech101_s sweep alongside K and α" — K and α of the HAC region tree, which
**#2886 deleted**. Nothing has read `patch_saliency` since:
`_attach_patch_grid_to_media` in `vtscore/datasets/stages/embedding.py` says so
in its docstring ("nothing downstream reads it now that leaf pooling is gone"),
and a repo-wide search for the attribute finds only its own construction and
three test assertions.

**Verdict: a genuine cosine-magnitude constant with zero blast radius.** Left in
place — re-tuning a number with no consumer is not an improvement — but the
comment now records that it is inert, why it is embedder-dependent, and that
anything reviving saliency-weighted pooling must re-derive it per embedder
rather than inherit 4.0. This is the item most likely to bite later, precisely
because it looks tuned and is sitting in live code.

### 3. `alpha = 0.5` in the HAC merge affinity — experiment arm, not comparable across embedders

`vtscore/eval/patch_styles.py`, `build_patch_hac_tree`:
`blended = alpha * cos_d + (1 - alpha) * spatial`.

Both terms are nominally in [0, 1]. Only one of them fills that range. The
spatial term is a normalised grid distance and spans its range on every image by
construction; `cos_d = (1 − cos)/2` spans only what the embedder's within-image
patch cosines span. In a concentrated space `cos_d` stays in a narrow band near
0 and the merge order comes out mostly **spatial**; in a less concentrated one
the cosine term carries more of the decision at the same nominal α. The
*effective* α is therefore per-embedder.

This is an arm (`max_patch_hac`, `max_patch_pca_hac`), never the default — the
shipped path has had no tree since #2886 — so it is out of scope for the
"default arm is the app" rule. It matters anyway, because cross-embedder
comparison is exactly what that arm exists to do: an α-sweep read across
embedders is not reading the same knob twice.

**Verdict: a caveat, not a bug.** Whitening `cos_d` per image would fix it and
would also change the arm's definition, so this is recorded at the call site
rather than edited. One honest limit: #3329 measured **media-level** cosines
(atlas nodes, browse regions, Shepard distances), never within-image
patch-to-patch cosines. The mechanism above is arithmetic; its magnitude on
`dinov3_patch` specifically is a prediction, not a measurement.

### 4. `return 0.5` in `calculate_gmm_threshold` — degenerate branch

`vtscore/training/thresholds.py`. The function is the app's threshold for every
cosine and text sort, and it is scale-adaptive by construction (a 2-component
GMM refit on the scores it is handed). The single literal is the fewer-than-two-
scores return: a sigmoid-scale sentinel handed back on a cosine scale.

**Verdict: harmless, documented.** With at most one item ranked there is nothing
for a threshold to separate. Noted in the docstring so it is not mistaken for a
fitted cosine cutoff.

---

## Checked, and fine — with the reason

The useful half of the answer. Each of these looks like a candidate to a grep
and is immune for a stated reason, so a future pass can skip it or re-check the
reason rather than re-derive it.

**Scale-free by construction — the loop's whole defence.**

| site | why the scale cannot reach it |
|---|---|
| `calculate_gmm_threshold`, the conformal rule, `fold_anchored_gmm_threshold` (`vtscore/training/thresholds.py`) | every threshold on the sort/vote path is **refit on the observed distribution**. This is why a patch embedder's compressed cosines produce a compressed threshold automatically, and it is the single biggest reason this audit is short |
| `support_pvalues` (`vtscore/detectors/evidence_coverage.py`) | a rank-based inductive-conformal p-value — the query's k-NN distance against the class's own leave-one-out distances. Uniform under exchangeability **whatever the geometry**. Same *shape* as the atlas guard, none of its defect: no path averaging, no vMF model, no α fitted on a magnitude |
| `trust_scores` … `TS < 1.0` (same file) | 1.0 is the exact indifference point of a **ratio** of distances (other class ÷ predicted class), not a fitted magnitude. It survives any monotone rescaling of the distances |
| `@q0.05` startup cuts (`vtscore/eval/startup_schedule.py`) | an explicit rank quantile of the live sort's own scores |
| centroid and query rankings (`vtscore/eval/al_strategies.py`) | cosines feed `min`/`max` only; the cut is the sort's own GMM line |
| vocabulary tagging (`vtscore/projection/signpost_texts.py`) | top-*k* by dot product. No magnitude floor for a term to qualify |

**Not embedder cosines at all** — the category a naive grep flags hardest.

| site | what the constant actually meets |
|---|---|
| `_THRESHOLDS = {"image": 4, "text": 4}` (`vtscore/state/near_dupes.py`) | Hamming distance out of 64 bits, over pHash/SimHash of **pixels and text**. The module docstring draws this line itself: a perceptual hash answers "these *are* the same thing", an embedding answers "these *mean* the same thing" |
| `_LOWE_RATIO = 0.75` (`vtscore/media/structural.py`) | ratio of two SIFT descriptor **L2** distances — and a ratio at that |
| `_LG_FILTER_THRESHOLD = 0.1` (`vtscore/media/structural_splg.py`) | LightGlue match confidence |
| `_RANSAC_REPROJ_THRESHOLD = 0.02`, `_MIN_SANE_SCALE`/`_MAX_SANE_SCALE` (`vtscore/media/structural_geometry.py`) | normalised pixel reprojection error and a scale sanity band |
| `STRUCTURAL_DECISION_THRESHOLD = 0.5` (`vtscore/training/structural_similarity.py`) | a calibrated match **probability**; also the structural slot, a different embedder from the patch slot |
| scene-cut threshold (`vtscore/media/video/clipper.py`) | Pearson correlation of hue×saturation **histograms** |
| VAD threshold, `top_db` (`vtscore/media/audio/clipper.py`) | Silero speech probability; amplitude in dB |
| detection threshold (`vtscore/media/image/clipper.py`) | object-detector confidence |
| `PROJECTION_MIN_DIST = 0.1` (`vtscore/config.py`) | a UMAP **output-space** parameter. The metric is cosine; the constant is not |
| compaction radius (`vtscore/projection/compaction.py`) | a 90th percentile of 2-D layout coordinates — data-driven, and off by default |
| `_LOGISTIC_K = 8.0` (`vtscore/training/blend_schedules.py`) | warps a vote-count ramp in [0, 1] |
| `SMART_FLAT_THRESHOLD`, `STABLE_*` (`vtscore/eval/autopilot_flow.py`) | a relative cost slope and a class-flip rate. Both dimensionless |
| `NO_GOOD_THRESHOLD = 2.0` (`vtscore/training/thresholds.py`) | a sentinel chosen to exceed **any** score; still above the cosine maximum of 1 |

**No cosine constant to find.** The production MaxPatch path —
`pool_box_from_media`, `bad_negative_vecs`, `media_score_rows` — selects rows
spatially and floods all of them; there is no similarity threshold anywhere in
it. `frontend/src/` contains no similarity constant at all: the SPA reads
`score`/`similarity` for display and takes its cutoff from the server.

---

## Prior art: where the codebase already got this right

Two patterns worth copying rather than reinventing, both already in the tree:

- **Key on the capability, not the space's numbers.**
  `PRODUCTION_SPLIT_BY_SPACE` (`vtscore/training/thresholds.py`) is a dict on
  `single_vector` vs `patch`, because #3287 measured the two wanting opposite
  train/calibrate splits. It carries the measurement in its comment.
- **Refuse rather than answer.** #3351 gates the domain-shift endpoint on the
  declared `supports_patch_regions` capability — not a name list — so every
  present and future patch embedder is covered the day it registers.

The failure mode this audit was looking for is the inverse of both: a plausible
number, no runtime check that the data resembles what it was fitted on, and a
wrong answer that arrives as confident output rather than an error. Item 1 has
the shape and survives on measurement; item 2 has the shape and has no reader;
item 3 has the shape inside an arm. Nothing else in the tree has the shape.

---

## What this does not cover

- **Within-image patch-to-patch cosine spread is unmeasured.** #3329 read
  media-level vectors throughout. Item 3's magnitude, and any future
  saliency-weighted pooling, both turn on that distribution.
- **`typicality_pvalues` remains callable on a patch atlas** as library API.
  #3351 gates the *endpoint*; a direct library caller still gets the
  miscalibrated values. Fixing the model rather than the caller is #3348.
- **Learned constants are out of scope by construction.** MLP weights, GMM
  parameters, and conformal quantiles are refit per detector per session; they
  are not tuned numbers and cannot go stale against a new embedder.
