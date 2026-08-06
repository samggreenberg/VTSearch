# Machine Learning Details

VTSearch learns a binary classifier from user votes ("good" vs "bad") using a **linear (logistic) head** — a single `Linear(input_dim, 1)` trained with balanced binary cross-entropy, i.e. logistic regression. The model operates on embeddings produced by pretrained feature extractors (LAION-CLAP for audio, SigLIP for images, X-CLIP for video, E5-base-v2 for text) and outputs a score in [0, 1] for each item in the dataset.

Alongside the head, each dataset carries a **[Coverage Atlas](#coverage-atlas)** — a hierarchical partition of the embedding space that guides the autopilot's diversity sampling and provides calibrated typicality scores for domain-shift detection.

## Architecture

The head is defined in `vtscore/training/mlp.py` via `build_model()`. Production always builds the linear head, selected by the `hidden_dim=LINEAR_HEAD` (`0`) sentinel:

```
Linear(input_dim, 1)
```

- **Input dimension**: Dynamic, depends on the embedding model for the current media type (see [Embedding Models](#embedding-models) below).
- **No hidden layer**: hence no ReLU and no dropout — a bare linear map has nothing to regularize with dropout, matching plain logistic regression.
- **Output**: A single logit. `torch.sigmoid` is applied at inference time to produce a probability in [0, 1].

The model outputs raw logits (not probabilities) during training. This allows the use of `BCEWithLogitsLoss`, which fuses the sigmoid and binary cross-entropy computation using the log-sum-exp trick for better numerical stability. At inference time, `torch.sigmoid()` is applied explicitly to convert logits to probabilities.

### Why linear, and where the MLP survives

The head used to be a small MLP (`Linear(input_dim, hidden_dim) -> ReLU -> Dropout(p) -> Linear(hidden_dim, 1)`, width auto-sized by `_auto_hidden_dim(n_train)`). With only ~3–5 labeled positives that MLP is under-determined: each retrain wobbles the scores, and the calibrated cut lurches over the unlabeled positives — the threshold-stability spikes tracked in issue #2790. A linear boundary has no such freedom, and measurements on COCO/SigLIP2 and VG/SigLIP1+2 put the deep-spike rate roughly 55–73% lower at equal or better FNR and cost.

`build_model(input_dim, hidden_dim=N)` with `N > 0` still builds the MLP, and `_auto_hidden_dim` still sizes it. That path exists only for the eval harness's MLP sweep arm (see [`docs/EVAL.md`](EVAL.md)) and unit tests; it is not reachable from the app. The Stage-2 structural-verification classifier (`vtscore/training/structural_similarity.py`) is a separate feature and keeps its own MLP.

## Training Configuration

| Setting | Value | Notes |
|---------|-------|-------|
| **Loss function** | `BCEWithLogitsLoss` | Per-sample (unreduced), with manual class weighting |
| **Optimizer** | Adam | `lr=0.001`, `weight_decay=1e-4` |
| **Epochs (cap)** | 200 | Configurable via `TRAIN_EPOCHS` in `config.py` or the `VTSEARCH_TRAIN_EPOCHS` env var |
| **Early-stop patience** | 10 | Training halts when the loss fails to improve for this many consecutive epochs (configurable via `TRAIN_PATIENCE` / `VTSEARCH_TRAIN_PATIENCE`; set 0 to disable) |
| **Head** | `Linear(input_dim, 1)` | The `LINEAR_HEAD` sentinel (`hidden_dim=0`); no hidden layer, no dropout |
| **Batching** | Full-batch | All labeled data in every forward pass |
| **Reproducibility** | Local `torch.Generator` | Per-model seed (default 42) for thread-safe deterministic init |
| **Gradient scoping** | `torch.enable_grad()` | Explicitly enabled during training loop |

### Class Weighting

Training balances classes by **inverse-frequency weighting** by default (`mlp.py`). The one exception is region flooding on patch datasets, where the caller supplies explicit per-bag `sample_weights` instead (see [Region-aware training](#region-aware-training-on-patch-region-datasets) below). Either way, inclusion does **not** enter training — the trained model, and therefore every item's score, is independent of inclusion. Inclusion is applied later as a pure threshold knob in `conformal_threshold` (see [Threshold Calibration](#threshold-calibration) below).

- **Weights**: `weight_true = num_false / num_true`, `weight_false = 1.0`

Keeping the model inclusion-independent is what lets the calibration cache reuse fold scores across cutoff slides: when the user changes inclusion, the labels are unchanged, the fold models are unchanged, and only the cheap quantile rule re-runs.

### Label Smoothing

Targets are label-smoothed with ε = 0.05 (`MLP_LABEL_SMOOTHING` in `vtscore/config.py`): Good examples train toward 0.95, Bad toward 0.05, with class weights still derived from the hard labels. This is **not** a knob-mover — it exists as tie insurance for the conformal threshold rule below, which takes quantiles of the calibration scores and therefore needs distinct score values. Smoothing bounds the optimal logit (≈ ±2.9 at ε = 0.05), so a strongly-fit model cannot saturate every score to exact 0.0/1.0 sigmoids, where all quantiles would collapse to the same cut.

### Threshold Calibration

A decision threshold separating "good" from "bad" predictions is computed via **cross-calibration**:

1. Split labeled data into Train (1 − `calibration_fraction`) and Calibrate (`calibration_fraction`).
2. Train a fold model on Train **using the same head as the final model**, score the held-out Calibrate portion.
3. Repeat for `calibrate_count` independent random splits.
4. Pool every fold's held-out (score, label) pairs and apply the **conformal inclusion rule** (`conformal_threshold`) once. Pooling — rather than averaging per-fold thresholds — is deliberate: the knob's resolution is bounded by how many calibration scores the quantiles are taken over.

The `calibrate_count` setting defaults to `2` (`DEFAULT_CALIBRATE_COUNT` in `vtscore/config.py`) and can be raised or lowered (`VTSEARCH_CALIBRATE_COUNT`) to trade calibration quality (and Inclusion-knob resolution — more folds means more pooled calibration scores) for latency. (The eval runner uses its own default of `2` for a separate, non-interactive path — see [`docs/EVAL.md`](EVAL.md).) The folds are trained at every label count: the fold *models* are an input to the fold-anchored estimator below — it anchors on their held-out scores — so there is nothing to skip. (Before the population-anchored adoption, the calibration was skipped wherever the mix-in schedule gave the cross-cal cut zero weight, at 6 labels or fewer; the schedule is no longer what combines the two estimators, so that skip is gone.) Every training entry point (vote-driven Train, labelset re-derivation, Find, and detector-load-from-origins) cross-calibrates below 6 labels rather than falling back to 0.5.

**Every trained threshold fuses the haystack into the cut.** There is no setting for this. It used to be the `safe_thresholds` per-user toggle, on by default since the #2799 A/B; the population-anchored adoption made the fused estimator uniformly better than the alternative at every label count, so the toggle was deleted rather than left as a way to opt into a worse threshold. The historical A/B is still worth reading for *why* the population distribution belongs in the cut: [`docs/experiments/safe-thresholds/REPORT.md`](experiments/safe-thresholds/REPORT.md).

**What it computes: the fold-anchored mixture, not the old blend.** The blend-vs-fusion question was measured by the #2852 deep-regime study and fusion won decisively, so the schedule was retired as the *combiner* (`fold_anchored_gmm_threshold`, adopting the study's recommendation). Per calibration fold *k*:

1. Score the whole haystack with fold model *k* and fit a **semi-supervised** 2-component mixture to those scores, with fold *k*'s **held-out** labeled scores clamped to their labeled component (Good → high, Bad → low). The anchors are honest — those items were not in fold *k*'s training set — and they share one score scale with the population they anchor.
2. Cut fold *k*'s fit with the **rate-optimal** rule at the Inclusion knob's cost weights, and read that cut's empirical quantile in fold *k*'s own haystack distribution.
3. Average the fold quantiles and realize the result on the **final** model's haystack distribution (rank transfer: two models scoring the same haystack are related by an approximately monotone map, and quantiles survive monotone maps). No raw score ever crosses scales.

The shipped anchor mass is **κ = 1** — each vote counts as *one* haystack point among the ~50k the mixture is fitted on — and the shipped cut rule is the rate crossing. **The #2861 follow-up says both should change.** The first sweep only went *down* to κ=1, so "performance degrades monotonically as κ grows" described one side of an interior optimum; extending the grid to 0.01 across six environments puts the optimum at **κ = 0.3 with the *midpoint* cut**, which beats the shipped pair in 6 of 6 environments (pooled −0.0045 paired regret, p<1e-4). The rule flips with the mass for a reason: `mid` ignores the mixture weights, `rate` reads them, and anchor mass is what lets the votes' acquisition-biased prevalence into those weights. Labels and haystack do now feed one estimator rather than being averaged as rivals on a label-count schedule; the labels' authority grows with the data instead of on a hand-tuned ramp — though #2861 also finds it grows *too fast*, with the best κ falling like 1/n, so a fixed *total* anchor mass is the better parameterisation.

Measured at 100–300 votes on Visual Genome region voting, this beats pure cross-calibration by **−0.085** paired regret (winning 71–74 % of ~36k paired steps), beats the 6→20 ramp everywhere they differ, holds *both* error rates below x-cal's until 200 votes, and is **2.8×** steadier step to step. **Both of that study's open threads are now closed, and one of them is a regression.** #2861 scored the schedules that actually ship (`slow_cap50` region / `cap50` binary) as controls: against those rather than the old ramp, fusion wins clearly on **region voting** (−0.026, p≤3e-8) but is a **dead heat on COCO binary voting and a loss on an 838-image set**, where `cap50` beats every fusion arm. The gain tracks how many *positive* anchors the regime supplies (24 → −0.093, 8 → −0.019, 3 → −0.002), so binary-voting detectors — which the fused path also covers, unconditionally since #2863 — currently get a slightly worse threshold than the blend gave them, and there is no longer a way back to it. The fold-count half is closed too: `qmean` beats `qmedian` at every κ (indistinguishably at κ=0.3, significantly from κ=1 up), and four folds are nominally −0.008 better than two at every grid point with no significant cell. See [`docs/experiments/population-anchored-calibration/REPORT.md`](experiments/population-anchored-calibration/REPORT.md).

Degeneracy policy, per fold and then globally: a fold whose anchored fit degenerates (inverted means, collapsed component) falls back to that fold's **unanchored** GMM fit; if no fold yields a fit at all, the threshold falls back to the **blend** (`calculate_safe_threshold`), which is what still runs for label sets too small to form calibration folds — never to 0.5. A training path with no haystack at all (detector-load-from-origins, which re-derives from saved labels with no dataset in hand) ships the plain cross-calibration cut. The blend schedules therefore remain in the tree as that fallback and as harness arms, but they no longer decide the shipped cut.

Because the estimator reads Inclusion as the rate weights it optimizes, the fitted mixtures are cached on the detector context and **re-cut** when the user slides Inclusion — no refit, no re-scoring — so a slide lands on exactly what a retrain at that inclusion would have stored. The rate cut is taken as the *highest* score at which the Bad component still out-densities the Good one under the cost tilt, which keeps it monotone in Inclusion (and so keeps the included sets nested) even on fits where the density crossing leaves the inter-mean interval.

**Why fold models share the final model's architecture:** a different architecture produces a different score distribution, so a threshold found on fold models would not transfer faithfully to the final model. The training code threads one `hidden_dim` through both the final fit and every fold fit, so the two can't drift apart. With the linear head that value is a constant (`LINEAR_HEAD`), which makes the property automatic — the head has no capacity to size. It mattered more under the old MLP, whose width was auto-sized from the training-set size: left alone, each fold would have trained on fewer examples and got a narrower hidden layer than the final model.

The **Calibration Fraction** setting (0–1, default 0.5) controls how much data is reserved for threshold calibration vs. model training in each split. For example, a value of 0.2 means 80% Train / 20% Calibrate. If the fraction is so extreme that a valid Train/Calibrate split cannot be formed (fewer than 2 training examples or fewer than 1 calibration example), the system returns a maximum threshold so that nothing is predicted as Good.

The threshold is a **split-conformal quantile rule** over the pooled held-out scores, governed by the `inclusion_value` parameter (integer in range [-10, +10]). This is where inclusion biases the result toward recall or precision — at calibration/threshold time, **not** at training time. For `k = inclusion_value` (with `BASE = 0.25`, `QPOS_MAX = 0.75` — `CONFORMAL_BASE_BUDGET` / `CONFORMAL_QPOS_MAX` in `vtscore/training/thresholds.py`):

- A **false-negative cap** `α(k) = min(1, BASE·2⁻ᵏ)`: the threshold never exceeds the α-quantile of the held-out *positive* scores, so an estimated at-most-`α` fraction of true matches falls below the cut. `+k` therefore has a portable meaning — "the fraction of true matches I'm willing to miss, halving per step" — independent of dataset or detector (e.g. `+3` ≈ miss at most ~3%, `+10` ≈ miss at most ~0.02%). The cap is an upper bound, not a target: when the classes separate cleanly the cut drops into the gap below the calibration positives and the budget goes unspent.
- A **false-positive guard** for `k ≤ 0`: the threshold stays at or above the `1 − BASE·2ᵏ` quantile of the held-out *negative* scores (so overlap-heavy tasks keep FPR control) and above a walk *up* toward the positive score distribution. The walk interpolates linearly in score space from the **gap midpoint** at `k = 0` to the `QPOS_MAX` quantile of positives at `k = −10` ("just the surest matches").

The **gap midpoint** is the cut's default anchor. When the classes separate cleanly there is an empty band between the top of the negatives and the lowest calibration positive, and every cut inside it has identical empirical error on the calibration set. The band's top edge — the lowest calibration positive — is therefore an arbitrary choice among equals, and it is the worst one: it is an extreme order statistic over a handful of held-out votes (so it lurches from vote to vote), and it is measured on the *fold* models' score scale while being applied to the *final* model's scores. The fold models train on half the votes and saturate, so their lowest held-out positive routinely lands above every score the final model produces — a cut that admits nothing at all, not even the items the user voted Good, self-healing on the next click when the fold split is redrawn (issue #2781). Sitting in the middle of the band is the max-margin choice among cuts the calibration data cannot distinguish, and it spends no FN budget: the midpoint is strictly below every calibration positive. Under class overlap there is no band, the midpoint collapses to the false-positive guard, and the rule is unchanged.

The rule is **monotone in `k` by construction**: raising inclusion can only lower the threshold, so included sets are *nested* — everything included at Inclusion 1 stays included at Inclusion 4. That makes "cut off at Inclusion 1, then verify the extra band up to Inclusion 4" a well-defined workflow. (The previous min-cost argmin over observed cuts had exactly as many distinct optima as the calibration folds had ranking errors, so on well-separated votes the knob provably never moved; see `docs/experiments/inclusion-knob/REPORT.md` and issue #2693.)

Because the fold models are inclusion-independent, the pooled held-out scores can be cached once and re-thresholded at any inclusion (this powers the Find Stats sweep across all inclusion values).

For semantic (text/example) sorts, a **GMM-based threshold** is used instead: a 2-component Gaussian Mixture Model is fitted to the score distribution and the cut is placed at the **midpoint between the two fitted component means**. The same cut is the GMM half of the safe-threshold blend, which is now only the fallback for label sets too small to form calibration folds.

**Why not the equal-density crossing?** Issue #2798 replaced the midpoint with the crossing of the two weighted components — solving `w_lo·N(x; μ_lo, σ²_lo) = w_hi·N(x; μ_hi, σ²_hi)`, the Bayes boundary between them — on the argument that under **region voting** (a media's score is the max over ~24 region-node scores) the Bad mode is an extreme-value statistic: right-skewed, wider and much heavier than the Good mode, which puts the crossing *above* the midpoint and means the midpoint cuts inside Bad mass. The #2799 study measured the two rules as paired within-step variants (each re-cutting the same model on the same votes) and the crossing lost on cost in every max-pooled window: +0.0036 at 6–20 votes, +0.0059 at 2–5 (`docs/experiments/safe-thresholds/REPORT.md`). The geometry argument holds in *direction* — the crossing does cut higher and does buy FPR — but the exchange rate is ~1.3 FNR per 1 FPR, and for a needle-finding tool the missed match is the worse error. So issue #2833 reverted production to the midpoint. The crossing solver stays in the tree as an eval variant, and issue #2836 is the open question of why the midpoint wins (leading hypothesis: the crossing is the count-optimal cut while we score a *rate* loss, so its prior-odds term is a bias, and the right cut is a third point rather than either of these two).

## PyTorch Environment Settings

| Setting | Where | Value |
|---------|-------|-------|
| `OMP_NUM_THREADS` | `app.py` | `1` |
| `MKL_NUM_THREADS` | `app.py` | `1` |
| `torch.set_num_threads` | `vtscore/embedding/loader.py` | `1` |
| dtype | `training.py` | `torch.float32` |
| Device | default | CPU (GPU supported, see tests) |

Threading is restricted to 1 to minimize memory overhead; the real cost is the embedding models, not the head.

## Embedding Models

Each media type uses a different pretrained model to produce fixed-size embedding vectors:

| Media type (`type_id`) | Embedder | Model | Embedding dim |
|------------------------|----------|-------|--------------|
| Audio (`audio`) | `clap` (default) | LAION CLAP (`laion/clap-htsat-unfused`) | 512 |
| Audio (`audio`) | `clap_music` | CLAP Music & Speech (`laion/larger_clap_music_and_speech`) | 512 |
| Audio (`audio`) | `clap_general` | CLAP General 2024 (`laion/larger_clap_general`) | 512 |
| Audio (`audio`) | `paraspeechclap` | ParaSpeechCLAP speech-style (WavLM + Granite, `ajd12342/paraspeechclap-combined`) | 768 |
| Audio (`audio`) | `ast` | AST audio spectrogram (`MIT/ast-finetuned-audioset-10-10-0.4593`, audio-only) | 768 |
| Audio (`audio`) | `whisper_encoder` | Whisper-base encoder (`openai/whisper-base`, audio-only) | 512 |
| Image (`image`) | `siglip` (default) | SigLIP (`google/siglip-base-patch16-224`) | 768 |
| Image (`image`) | `siglip2` | SigLIP 2 (`google/siglip2-base-patch16-224`) | 768 |
| Image (`image`) | `clip` | CLIP (`openai/clip-vit-base-patch32`) | 512 |
| Image (`image`) | `dinov2_single` / `dinov2_patch` | DINOv2 ViT-B/14 (`facebook/dinov2-base`) | 768 |
| Image (`image`) | `dinov3_single` / `dinov3_patch` | DINOv3 ViT-B/16 (`facebook/dinov3-vitb16-pretrain-lvd1689m`, gated) | 768 |
| Image (`image`) | `eupe_single` / `eupe_patch` | EUPE ViT-B/16 (`facebookresearch/EUPE`, FAIR Noncommercial) | 768 |
| Image (`image`) | `sift_vlad` | SIFT/VLAD instance matching (classical, no text encoder) | 8192 (64 centroids × 128-dim SIFT) |
| Image (`image`) | `face` | FaceNet identity (`InceptionResnetV1`, face crops, no text encoder) | 512 |
| Video (`video`) | `xclip` (default) | Microsoft X-CLIP (`microsoft/xclip-base-patch32`) | 768 |
| Text (`text`) | `e5` (default) | E5 (`intfloat/e5-base-v2`) | 768 |
| Text (`text`) | `bge` | BGE (`BAAI/bge-base-en-v1.5`) | 768 |
| Document (`document`) | (none) | None (no embedder) | N/A |

Each embedder lives in its own `embedder_<name>.py` file inside the media-type package and exposes a module-level `EMBEDDER` sentinel; the default for a given media type is whichever embedder overrides `is_default` to return `True` (exactly one per media type).

Audio, image, and text media types each ship alternative embedders alongside the default. The image variants come in **single/patch pairs**: `_single` embedders produce one CLS-pooled vector per image (cheap, same shape as SigLIP); `_patch` embedders additionally produce a hierarchical HAC region tree (~24 region vectors per image) and the raw patch grid, enabling region-level similarity, region-aware detector scoring, and region voting on yes-votes.  See [`docs/plans/patch-embedder.md`](plans/patch-embedder.md) for the full design.

Embedders carry capability flags consumed by the routes layer and the frontend:

- `supports_text: bool`: whether the embedder can embed text queries. Text-sort returns HTTP 400 + `supports_text: false` when this is false.
- `supports_patch_regions: bool`: set on the `_patch` variants. Loaders that see this flag populate `media["patch_regions"]` (HAC tree) and `media["patch_grid"]` (raw `H × W × D` fp16) in addition to the embedder's vector in `media["embeddings"]`.
- `license_notice: Optional[str]`: non-None for embedders with usage restrictions (e.g. EUPE's FAIR Noncommercial Research Licence). Surfaced as a warning chip on the embedder picker.

The **document** media type has no embedding model of its own. Documents (PDF, DOC, PPT) are intended to be converted to other media types (images or text) via media converters in `vtscore/converters/` before embedding.

Embeddings are computed once when a dataset is loaded. The full-image vector lands in each clip's `"embeddings"` dict, keyed by embedder name (`numpy.ndarray` values; read it through the `media_embedding` accessor); patch embedders additionally populate `"patch_regions"` (list of `RegionVector`s, fp16-on-disk / fp32-in-RAM) and `"patch_grid"` (`H × W × D` ndarray, fp16). The detector head trains on these pre-computed vectors, so training is fast (typically < 1 second for 200 epochs on a few hundred labeled examples).

### Region-aware training on patch-region datasets

Inference max-pools the head over each image's `patch_regions` (an image scores by its **best** region — see `score_media`). Training is shaped to match that scorer, and it is deliberately asymmetric between Good and Bad votes — the multiple-instance-learning treatment of a max-pool bag:

- **Good vote** — a positive bag needs only *one* good region, and the user tells us which via an optional `region_box` (drawn by Shift-drag on the focus pane). The box is **snapped to the nearest `patch_regions` node** (max box-IoU, `snap_box_to_region`), so the positive is one of the exact candidates the max-pool will score — not a fresh uniform pool that matches no node. A Good vote with no box falls back to the full-image CLS node.
- **Bad vote** — a negative bag asserts that *no* region is good, so a Bad vote **floods the image's CLS + HAC-leaf nodes** (the disjoint covering set) as negatives. This trains every leaf down, so the max-pool can't surface a look-alike sub-region of a rejected image.

  **The internal HAC nodes are scored but never flooded**, and that gap is a measured exception rather than a redundancy claim. `build_hac_tree` renormalises each merged vector (`_l2_normalize(sum_a + sum_b)`), so an internal node is the convex-hull point of its descendants *projected back onto the unit sphere* — a ~1.53× gain on real trees. Under the linear head that scales the logit by the same factor, so an internal node out-scores every one of its own leaves on ~4.7% of node/direction pairs: training the leaves down does **not** pull the internal down with them. Flooding internals anyway was A/B'd over 24 synthetic patch detectors (`scripts/probe_hac_internal_flood.py`) and it *hurts* — paired AP −0.058 ± 0.036 (95% CI, excludes zero; leaves-only wins 19/24 seeds), FPR +0.037 ± 0.045 and FNR −0.010 ± 0.049 both straddling zero. It does close part of the gap (negatives whose winning row is an internal node: 4.6% → 2.7%), but suppressing a rejected image's renormalised mean directions also suppresses the geometry a *positive* image's concept-blob internal lives in, and that costs more than it buys. So the flood stays leaves-only, with the gap pinned by tests rather than papered over (see #2731).

Because flooding turns one Bad vote into many correlated leaf rows, class balance and calibration are **per-bag, not per-row**:

- The final fit is `train_model(..., sample_weights=...)` where each Bad image's leaves share one image's worth of negative mass (`_per_bag_fit_weights`), so a rejected image counts once regardless of leaf count. Good votes weigh `n_bad_bags / n_good`, matching the default inverse-frequency balance but with the *bag* as the unit.
- Cross-calibration (`compute_fold_orderings(groups=...)`) splits Train/Calibrate **by bag** (a Bad image's leaves never straddle the boundary), sizes fold counts over votes not rows, weights fold fits per-bag, and **max-pools each calibration group to one score** — so the threshold is placed on the per-image score scale the detector actually deploys. Hidden-layer width and the fallback blend's ramp likewise size on vote count.

Flooding applies only where scoring is region-aware max-pool: the Learned-sort vote path (`train_and_score`) and the saved-detector labelset path (`labelset_train_and_score` / `train_from_labelset`). Paths that score each image by a single vector — Find cold-detector scoring, label-file sort — score image-level and are intentionally *not* flooded (flooding leaf negatives while scoring one image vector would be a train/score space mismatch). On any dataset whose embedder produces no regions, every bag holds one row and the whole path collapses byte-for-byte to the historical single-vector BCE — fully backward-compatible.

## Coverage Atlas

The Coverage Atlas (`vtscore/state/coverage_atlas.py`, class `CoverageAtlas`) is a hierarchical k-means partition of a dataset's embedding space that remembers, per region, how much labeled evidence of each class the user has provided. It serves two jobs:

1. **Diversity sampling** — the Training autopilot's "Explore Diversity" phase asks it for the next item to label, so a handful of clicks covers the whole collection and stress-tests the model where it is most likely to be wrong.
2. **Domain-shift detection** — it answers "how typical is this item of the data this atlas was built on?" with a calibrated p-value, so a detector trained on dataset A can be sanity-checked against dataset B before anyone trusts its scores there.

One atlas exists per dataset (`DatasetContext.coverage_atlas`). It replaced the earlier Diversity Tree, which kept only a boolean "seen" flag per region; the atlas keeps the geometry and statistics the tree threw away. The full design study (including the not-yet-built portable artifact, blob scan, and active auditor) lives in [`docs/plans/coverage-atlas.md`](plans/coverage-atlas.md).

### Geometry: center, then normalize

All stored embeddings are unit vectors (L2-normalized at ingest), and contrastive embedders concentrate them in a narrow cone — raw cosines between any two items are uniformly high, which makes raw directions nearly useless for partitioning or typicality. The atlas therefore works in a **centered spherical frame**: it subtracts the dataset's mean vector and re-normalizes to the unit sphere. The centering vector is part of the structure and every query is mapped into the same frame.

One consequence worth remembering: the **root node is directionally degenerate by construction**. Centering makes the sum of all vectors (the "resultant") vanish, so the root has no preferred direction. Everything below the root — the k-means cells — is cohesive and directional. Several behaviors key off this via the resultant length `rbar` (see the calibration gate below).

### Build

Built automatically at dataset load for datasets up to 50 000 items (`COVERAGE_ATLAS_AUTO_THRESHOLD`), on demand via `POST /api/datasets/registry/<id>/coverage-atlas` for larger ones, and cached inside the dataset pickle (key `"coverage_atlas"`, format `"coverage-atlas/1"`) so reloads skip the k-means. The build is recursive k-means (k = 3) over the centered vectors, splitting until a node has fewer than 20 items (`min_node_size`) or the depth cap is hit (`auto_max_depth` bounds the leaf count at ~4 000 for very large datasets). K-means runs on cuML when a usable GPU is present, sklearn otherwise, with restart counts scaled down for large nodes.

Each node stores:

| Field | Contents |
|-------|----------|
| `ids` | The node's item IDs, sorted **most-typical-first** (descending `mu . x`) — `ids[0]` is the region's representative |
| `children` | Child node names, stored **largest-first** so breadth-first traversal reaches big unexplored regions before small ones |
| `n` | Item count |
| `mu`, `rbar` | Mean direction and resultant length — the sufficient statistics of a von Mises–Fisher component, so reading the tree at any depth gives a multiresolution mixture model of the dataset |
| `t_quantiles` | A 21-point quantile grid of the node's own points' **leave-one-out** typicality scores, used to calibrate query p-values |

Node records are **immutable** once built. Labeled evidence lives in a separate per-atlas **overlay** — `n_pos` / `n_neg` counts keyed by node name (session state, not serialized), plus the labeled-ID set — so two atlases can share one node table by reference while keeping independent labels. `structural_clone()` exploits this: an atlas over the same id set (e.g. the labeling-progress per-step atlas mirroring the dataset context's) is cloned with the node table shared and a fresh overlay, skipping the hierarchical-k-means re-fit.

Evidence flows in from votes: every good/bad vote calls `label(id, good=...)`, which increments the class counter in the item's leaf and every ancestor; un-voting decrements; clearing votes or swapping detectors resets and replays (`resync_coverage_atlas_to_detector`, via `reset_labeled()`). A node is **covered** when `n_pos + n_neg > 0` (read through `atlas.n_pos(name)` / `atlas.n_neg(name)`).

### Diversity sampling (`next_sample`)

`GET|POST /api/coverage-atlas/next` returns the next item the autopilot should show. The walk is breadth-first from the root: the first node carrying **no evidence** is the next region to explore. Because siblings are stored largest-first, ties break toward the biggest unexplored region — best coverage gain per click.

Within the chosen node, the pick is a **surprise probe** when sort scores are supplied (the autopilot always supplies the current learned-sort scores and threshold):

- Node's median score ≥ threshold (**presumed good**) → return the **lowest**-scored element: the item most likely to be a hidden bad in a region the model calls good.
- Otherwise (**presumed bad**) → return the **highest**-scored element: the item most likely to be a hidden good.

The extremum probe is informative in both outcomes. If the probe *flips* (the greenest item of a presumed-red region is actually good), the user just found a hidden pocket the model was wrong about — maximum training value for one click. If it *doesn't* flip, the region's presumption has been stress-tested at its weakest point: nothing else in the node was more likely to surprise.

Two refinements:

- **Typicality tempering.** In nodes with a concentrated direction (`rbar ≥ 0.1`), the extremum is taken over the node's **typical half** (`ids` is typicality-sorted, so this is just the first half). An extreme score on an *atypical* item is disproportionately often a lone oddball — a corrupt file, a weird crop — whose flip says nothing about the region; a flip on a typical item is evidence of a real pocket. Degenerate nodes (the root) probe the whole node, since their typicality ordering is noise.
- **Regional median.** The median that decides the probe direction always spans the whole node, not the pool — the presumption being tested is about the region.

Without scores, the pick is the node's most typical element (`ids[0]`), a representative of the unexplored region.

The response's `coverage_level` — the number of consecutive covered nodes in breadth-first order — is the autopilot's **Span** indicator: it turns green at `autopilot_goal_diversity` (default 40) covered nodes, ending the diversity phase. `exhausted: true` means every node carries evidence.

### Typicality and domain shift

`CoverageAtlas.typicality_pvalues(matrix)` answers, per query vector: *what fraction of the data this atlas was built on looks less typical than this?* Small p-value = the atlas has essentially never seen anything like it.

How a query is scored:

1. Map the query into the atlas frame (subtract `center`, renormalize).
2. Route it down the tree, at each node descending into the cosine-nearest child.
3. At every **calibrated** node along the path — at least 20 points *and* `rbar ≥ 0.1` — compute the alignment `t = mu . x` and read a p-value off the node's stored quantile grid.
4. Average the p-values along the path.

Three details make the p-values honest rather than merely monotone:

- **Leave-one-out calibration.** Each node's quantile grid is built from scores of its own points against the mean direction of the *other* points (closed form on the sphere: `(R.x - 1) / ||R - x||`). Scoring a point against a mean it helped shape is optimistic, and without the correction fresh in-domain queries systematically read as atypical.
- **The `rbar` gate.** A node with no concentrated direction has a meaningless `mu` and pathological leave-one-out scores; the gate excludes it — notably the always-degenerate root. Sparse branches terminate shallow, which is the adaptive bandwidth: dense regions are judged at fine scale, sparse ones at coarse scale.
- **Path averaging.** A hard partition has boundary artifacts (a fresh in-domain query near a k-means cell edge looks atypical at leaf scale); averaging across scales smooths them the way a tree ensemble would, at zero extra build cost.

`domain_shift_report(atlas, matrix, alpha=0.05)` aggregates the p-values into a dataset-level verdict. Under no shift, about `alpha` of items fall below `alpha`; the report gives the observed fraction (`frac_atypical` — roughly the shifted proportion), a binomial z-score for the excess, the median p-value, and a headline `shifted` boolean (excess both statistically clear, z > 3, and practically large, ≥ 2×`alpha`).

### Tutorial: how a diversity session works

What actually happens when the autopilot enters its "Explore Diversity" phase, click by click:

1. **The atlas already exists.** It was built (or restored from the pickle cache) when the dataset loaded, and every vote cast during the earlier good/bad/refine phases has already been counted into its evidence channels.
2. **The frontend asks for a sample.** `POST /api/coverage-atlas/next` with the current learned-sort scores and decision threshold in the body.
3. **The atlas walks breadth-first** to the first evidence-free node — say a 900-item region of the collection no vote has ever touched — preferring the largest such region among siblings.
4. **It probes for a surprise.** Suppose the node's median score is 0.81 against a threshold of 0.5: the model presumes the whole region is good. The atlas returns the *lowest*-scored item from the region's typical half — the most plausible hidden bad that is still representative of the region.
5. **The user votes.** The vote lands in the detector's labels *and* increments `n_neg` (or `n_pos`) in the item's leaf and all its ancestors — the region is now covered, and the next `next_sample` call moves on to the next uncovered region.
6. **The Span indicator advances.** Each labeling-status poll reads `span_info()`; when 40 consecutive breadth-first nodes carry evidence, Span turns green and the autopilot declares the collection covered.

Either outcome of step 5 helped: a flip hands the head a training example from a region it was confidently wrong about (the next retrain bends the boundary there); a non-flip certifies the region at its weakest point for one click.

### Tutorial: checking for domain shift before reusing a detector

You trained a detector on dataset A and want to run it on dataset B. Should you trust it? Ask the atlas:

```
# Both datasets loaded; A's coverage atlas built (automatic ≤ 50k items).
# The X-Dataset-Id header names the ACTIVE dataset (B); the URL names the
# REFERENCE dataset (A, the training domain).
GET /api/datasets/registry/<dataset_A_id>/domain-shift
X-Dataset-Id: <dataset_B_id>
```

```json
{
  "reference_dataset_id": "…",
  "n_items": 40000,
  "alpha": 0.05,
  "frac_atypical": 0.31,
  "expected_atypical": 0.05,
  "z_score": 24.1,
  "median_pvalue": 0.18,
  "shifted": true
}
```

Reading this: 31% of dataset B sits in regions of embedding space where dataset A had essentially no mass (against the 5% that chance would produce), so the verdict is `shifted` — the detector will be *extrapolating* on a third of B, and its scores there are unfalsified guesswork. Verify by hand before trusting it. A same-domain report instead shows `frac_atypical` near `alpha`, a median p-value near 0.5, and `shifted: false`.

The endpoint refuses (HTTP 400) when the two datasets use different embedders — typicality across embedding spaces would be confident nonsense — or when the reference has no atlas yet (build it via the endpoint above).

The same machinery is available in the library tier:

```python
import numpy as np
from vtscore.state.coverage_atlas import CoverageAtlas, domain_shift_report

atlas = CoverageAtlas({mid: vec for mid, vec in train_vectors.items()}, k=3)

atlas.label(42, good=True)          # count labeled evidence
print(atlas.next_sample(scores, threshold))  # next diversity probe

pvals = atlas.typicality_pvalues(np.stack(list(other_vectors.values())))
print(domain_shift_report(atlas, np.stack(list(other_vectors.values()))))
```

### Costs

Build is the same order as the embedding-matrix work a dataset load already does — seconds on CPU for 50k items, with progress reported to the load bar. Queries are microseconds per item (`O(depth × k × dim)`); a full domain-shift sweep over a 40k-item dataset is well under a second. Nothing needs a GPU.

## Key Files

- `vtscore/training/mlp.py`: `build_model`, `train_model`, `build_model_from_weights`
- `vtscore/state/coverage_atlas.py`: `CoverageAtlas`, `domain_shift_report`
- `vtscore/state/coverage.py`: atlas build/restore/resync helpers, vote wiring
- `vtscore/training/thresholds.py`: `calculate_cross_calibration_threshold`, `fold_anchored_gmm_threshold`, `calculate_safe_threshold`, `calculate_gmm_threshold`, `conformal_threshold`
- `vtscore/detectors/training.py`: `train_and_score`, `train_and_threshold`, origin-based detector training
- `vtscore/detectors/labeling_progress.py`: Cached per-step training and stability analysis
- `vtscore/embedding/loader.py`: Model initialization and thread configuration
- `vtscore/eval/voting_iterations.py`: Voting simulation evaluation
- `vtscore/config.py`: `TRAIN_EPOCHS` and model IDs
