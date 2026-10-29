# Machine Learning Details

VTSearch learns a binary classifier from user votes ("good" vs "bad") using a **linear SVM head** — a single `Linear(input_dim, 1)` fitted to the maximum-margin boundary between the two vote classes (L2-regularised squared hinge, class-balanced), **started from the Goods' centroid** (#4785, [below](#the-svm-started-from-the-goods-centroid-4785)). The model operates on embeddings produced by pretrained feature extractors (LAION-CLAP for audio, SigLIP for images, X-CLIP for video, E5-base-v2 for text) and outputs a score in [0, 1] for each item in the dataset. Below a minimum label count the detector is the Goods' centroid instead ([label quota](#below-the-label-quota-the-goods-centroid-not-a-head)), and a trained head's Good/Bad line is drawn from the labels at the detector's F-beta balance ([Threshold Calibration](#threshold-calibration)).

Alongside the head, each dataset carries a **[Coverage Atlas](#coverage-atlas)** — a hierarchical partition of the embedding space that guides the autopilot's diversity sampling. [Domain shift](#domain-shift) — is this dataset unlike the one a detector was trained on? — is a separate kNN report against the training dataset's own vectors.

Each section below says what ships, the constant that sets it, and in a line why. The measurements behind a decision live in its `docs/experiments/*/REPORT.md`, linked beside it; this document does not restate them.

## Architecture

The head's *architecture* is defined in `vtscore/training/mlp.py` via `build_model()`, and its *fit* in `vtscore/training/svm_prior.py` via `fit_linear_svm_prior_head()`, the centroid-started fit (`vtscore/training/svm.py`'s `fit_linear_svm_head()` fits the plain SVM). Production always builds the linear head, selected by the `hidden_dim=LINEAR_SVM_PRIOR_HEAD` (`-3`) sentinel, the centroid-started SVM (`LINEAR_SVM_HEAD`, `-1`, is the plain SVM it replaced):

```
Linear(input_dim, 1)
```

- **Input dimension**: Dynamic, depends on the embedding model for the current media type (see [Embedding Models](#embedding-models) below).
- **No hidden layer**: hence no ReLU and no dropout — a bare linear map has nothing to regularize with dropout.
- **Output**: A single number — the SVM's decision function, `w·x + b`. `torch.sigmoid` is applied at inference time to squash it into [0, 1].

The head is fitted by liblinear (scikit-learn's `LinearSVC`, `C=1.0`, `class_weight="balanced"`) rather than by a gradient loop, and its hyperplane is then copied into the `Linear(input_dim, 1)` module. Keeping the result a torch module is what lets the rest of the pipeline — calibration folds, max-pooled region scoring, weight serialization, the ONNX exporter, threshold fusion — stay exactly as it was.

**The sigmoid is not a calibrated probability**, and never was under the logistic head either: the decision point is the separately calibrated threshold below, not 0.5. The sigmoid is a monotone squash that keeps every downstream consumer (sorting, the score column, the quantile rules) working in [0, 1].

### The heads: which one is shipped, and why

| Sentinel | Architecture | Fitted by | Status |
|---|---|---|---|
| `LINEAR_SVM_PRIOR_HEAD` (`-3`) | `Linear(D, 1)` | liblinear (primal), squared hinge + L2, class-balanced, the Goods' centroid a cheap direction (and a typed-query detector's query, #4859) | **shipped** (#4785) |
| `LINEAR_SVM_HEAD` (`-1`) | `Linear(D, 1)` | liblinear, squared hinge + L2, class-balanced | shipped until #4785; eval arm |
| `LINEAR_HEAD` (`0`) | `Linear(D, 1)` | balanced BCE gradient loop = logistic regression | eval arm, tests |
| `LINEAR_LOGREG_HEAD` (`-2`) | `Linear(D, 1)` | scikit-learn logistic regression to convergence (balanced, L2, `C=1`; #4114) | eval arm |
| `hidden_dim > 0` | `Linear(D, H) -> ReLU -> Dropout -> Linear(H, 1)` | balanced BCE gradient loop | eval arm, tests |

**Why linear.** The head used to be a small MLP, its width auto-sized by `_auto_hidden_dim(n_train)`. With 3–5 labelled positives that MLP is under-determined: each retrain wobbled the scores and the calibrated cut lurched (the threshold-stability spikes of #2790). A linear boundary has no such freedom, and it cut the deep-spike rate by roughly 55–73% at equal or better FNR and cost.

**Why an SVM.** It keeps the linear boundary and changes only the objective that places it: a maximum-margin fit is decided by the votes nearest the boundary, where logistic regression keeps paying attention to every example. The SVM's ranking measured clearly ahead of the logistic head's; *why* it wins is still open.

The other heads remain reachable **by name** as eval arms (`head="linear_svm"`, `head="linear"`, `head="linear_logreg"`, `head="mlp"`; see [`docs/EVAL.md`](EVAL.md#trainer-vs-head-which-knob-picks-what)) and in unit tests; none is reachable from the app. Structural search's Stage 2 (`vtscore/training/structural_similarity.py`) is a separate feature and trains no scorer: since #4169 it ranks a shortlisted page by the inliers its best template verifies.

### The SVM started from the Goods' centroid (#4785)

`LINEAR_SVM_PRIOR_HEAD` fits the plain SVM's objective on the votes plus one feature: 10 times each vector's dot with the unit Goods' centroid `ĉ` of the rows being fitted. With weight `w'` on the original features and `v` on the new one, the head is `w = w' + 10·v·ĉ` with penalty `½(‖w'‖² + v²)`, so moving the hyperplane along `ĉ` costs 1/100 of moving it any other way — a prior on `w` stretched along the centroid. With few labels the cheapest separation is nearly the centroid; as labels accumulate the head leaves it wherever the Bads pay for it. It folds back into one `Linear(D, 1)`, so folds, the line, scoring, max-pooling and export take it unchanged, and each fit (every calibration fold included) computes `ĉ` from its own Goods. It is solved in liblinear's primal (the scaled feature stalls the dual solver at the same optimum).

**Why.** At the label quota the votes pin down almost none of the space, so the plain SVM's `w` is roughly the Goods' mean minus the Bads' mean, and the Bads' share of that is noise: on FHIBE's face crops the first trained head ranked a person's withheld photos worse than the centroid it replaced (AP 0.88 → 0.70, #4762). Priced as full sessions in [`REPORT.md`](experiments/2026-10-11-face-handover-4776/REPORT.md) (#4776) and on the Region path in [`REPORT.md`](experiments/2026-10-11-head-start-region-4785/REPORT.md); the owner chose one head everywhere.

### A typed-query detector also starts from its typed query (#4859)

On the query's own single-vector space (`query_head.typed_query_prior_applies`: a known non-patch embedder with a text tower, SigLIP on the Binary photo path), every fit of a training pass adds a second cheap feature beside the centroid's: the dot with the typed query's unit embedding at `svm_prior.QUERY_PRIOR_SCALE` (10). The head starts from the text sort as well as from the Goods, leaves both as far as the labels pay for, and folds into the same `Linear(D, 1)`. The training entry points enter `svm_prior.typed_query_prior` with the query embedded as the text sort embeds it (`query_head.typed_query_prior_vector`): `labelset_train_and_score` and `train_from_labelset` (Find, AutoFind, a load, an export, the CLI), and the Train view's learned sort, which passes its detector's registry query. A head fitted that way carries its query (`svm_prior.fitted_query_prior_text`); `cached_head_is_current` refuses a trained head fitted without the detector's query, and the calibration folds' cache keys on it. Region, example openings, faces and documents are unchanged.

**Why.** Three Goods taken off the top of the query's own sort say less about the class than the words that found them, and the centroid-started head ranked worse than the query it replaced for its first ~25 votes (#4384), while Autopilot asks its Hard questions from vote 8. Priced in [`REPORT.md`](experiments/2026-10-13-query-prior-4384/REPORT.md); the owner chose scale 10 over 3.

### Below the label quota: the Goods' centroid, not a head

A head fitted to three Goods and one Bad is a detector in name only, and every
vote is saved as it is cast, so Tune, AutoFind, a load or an import of the
labels could hand one out at any click. So what a labelset gives depends on its
counts (#4643, `vtscore/detectors/label_quota.py`):

| Labels | Detector | Detector made from a typed query (#4758) |
|---|---|---|
| no Good | none: Tune is refused, there is nothing to sort toward | **the typed query's own set** |
| a Good, but under both quotas below | **the Goods' centroid** | **the typed query's own set**; on a patch space, once the query has missed (a Bad before the third Good), the Goods' centroid (#4800) |
| 3 Goods and 4 Bads | the trained head above | the trained head above; on a single-vector space the typed query's own set until 30 labels (#4736) |
| a Good and 16 Bads | the trained head above (#4731) | the same as at 3 Goods and 4 Bads |

The first quota is Autopilot's own quorum (`goodToStart` / `badToStart`). The second is what Autopilot's Good walk leaves when it runs dry (`moreDryRun`, 16 picks in a row without a Good, with a Good in hand), so a target with one or two Goods the sort can reach still gets a head ([`REPORT.md`](experiments/2026-10-09-good-dry-4731/REPORT.md)). The counts are of labels that resolved to a vector.

**The centroid** (`vtscore/detectors/centroid_head.py`) is the example sort the app draws for several uploaded examples, built from the Goods: the unit mean of the L2-normalised Good vectors, every item ranked by its cosine to it (max-pooled over a patch item's rows). Its line is a **count** over those cosines on the corpus being scored, at the detector's balance: it keeps `beta ** 0.708` times the larger of the typed query's count (#4603, the items standing out of the cosines' bulk) and the items above the widest drop between neighbours in the top 11, and never fewer than that drop's count (`count_gap_floor`, #4759). From beta 1 up, when the widest drop is the first one, the line looks past that standout to the widest drop below the second item, if it is at least a quarter as wide (`count_gap_floor_skip25_b1`, #4783). Face crops read the count at z 5.5 instead of 4 (`CENTROID_LINE_RULE_BY_MEDIA_TYPE`, `centroid_line_rule`, #5325): ArcFace's cosines carry about 6 look-alikes per session above z 4 where a Gaussian bulk puts 0.3, and the count read them as the person (one photo, β 1: 6.4 faces kept for 3.8 at click 0, F 0.73; at z 5.5 3.5 and 0.85). The same bar cut COCO Better's click 0 by 0.06–0.15, so it is the face crops' alone; the app's example sort, its centroid head and the eval harness's default all read it ([report](experiments/2026-10-27-face-lines-5325/REPORT.md)). The line counts the unvoted media only (#4779, `fit_centroid`'s `line_mask`; an example sort leaves out the voted media and the example itself), since the Goods the centroid is built from sit at the top of their own ranking. It ships as a `Linear(D, 1)` whose weight is the centroid scaled by `CENTROID_LOGIT_SCALE` and whose bias puts the cut at logit 0, so its threshold is 0.5 and every scorer, the weight export and an ONNX consumer take it as they take the SVM. A balance change redraws the line on the cosines it was drawn on, and Autopilot's Hard select still samples at their midpoint. The labels' line needs held-out Bads the centroid does not have.

**Why a count.** The two-Gaussian midpoint it replaced (#4732) splits a rare target's own negative bulk: on FHIBE's face crops it kept a median of 2,300 of 5,439 images. The gap floor keeps the drop just below a person's last photo, and the standout skip helps from beta 1 up, while below it the lone standout is the better set. Reports: [count line](experiments/2026-10-10-centroid-line-4732/REPORT.md), [gap floor](experiments/2026-10-10-centroid-count-4759/REPORT.md), [standout skip](experiments/2026-10-11-centroid-drop-4783/REPORT.md).

**A detector made from a typed query gives that query's own set instead (#4758).** Through Autopilot's opening the screen already shows the typed query's sort at its line, and that set beats the centroid's on average at every preset ([`REPORT.md`](experiments/2026-10-12-find-typed-query-4758/REPORT.md)). So the detector's `text_query` (which its file and registry entry already hold) decides: from the first vote until the trained head, Tune, AutoFind, a load, a cold Find, an export and the CLI give the typed query's set (`vtscore/detectors/query_head.py`). It is the text sort as a head: the query's unit vector as a `Linear(D, 1)` in the dataset's **text** space, which on a dual-embedder dataset is not the space the detector learns in (every scorer asks `head_score_embedder`), cut where `text_sort_cuts` draws the display line at the detector's balance, over every item scored, voted ones included, as the sort counts them. A balance change redraws it as it redraws the centroid's. The query lives on the detector, not the labelset: an exported, an imported and a CLI-made labelset carry none, and a detector with no query, or a dataset with no text slot, gives the centroid as before.

The rule is on the labels, wherever they came from: `train_from_labelset`
(Tune, AutoFind, a detector load, the CLI) and a cold Find
(`labelset_train_and_score(label_quota=True)`) apply it, each handed the
detector's typed query (`detector_text_query`), and a trained head the
Train view's learned sort cached is not reused under it. The Train view's own
learned sort is not a detector anyone takes away and still trains from the
first Good and Bad. The eval harness's default arm scores the withheld half the
same way (`detector_tier`, see [`EVAL.md`](EVAL.md)).

**On a single-vector space the typed query holds past the quota, until 30 labels (#4736).** A head trained on the quota's 3 Goods and 4 Bads asks good questions, and Autopilot asks them from vote 8 (#4742), but its set stays worse than the typed query's until around vote 27 at beta 1/4 and 1, and 45 at 4. So a typed-query detector keeps giving its query's set until its labels number `QUERY_HOLD_LABELS` (30; `label_quota.query_holds`). That covers Tune, Find, AutoFind, a load, an export, the CLI and the Train view, where the text sort stays on screen while Autopilot asks from the learned ranking held behind it (the learned-sort payload's `query_hold`). 30 is one number for all three presets ([`REPORT.md`](experiments/2026-10-12-handover-4736/REPORT.md)); judges that read the votes did not beat a fixed count, because the Hard picks sit at the head's own line, where its calls are coin flips (#4816). The Region path's head (DINOv3 patches) passes the query by about vote 10, so a patch space, or an embedder the registry does not know (`training._patch_space_for`), hands over at the quota.

**On a patch space the Goods' centroid answers once the typed query has missed (#4800).** A typed query that turned up a Bad before its third Good has missed, and on the Region path the centroid of the Goods' own boxes is then the better set; before any Bad, and once the query has given 3 Goods, the query's set is. So a typed-query detector on a patch space gives the centroid only from that Bad to the quota (`label_quota.query_missed`: a Good but fewer than 3, and 1 to 15 Bads, `QUERY_MISS_BADS`; [`REPORT.md`](experiments/2026-10-19-region-missed-query-4800/REPORT.md)). A single-vector space keeps the query (#4758). Like the hold, it changes what Tune, Find, AutoFind, a load, an export and the CLI give, never what Autopilot asks.

## Training Configuration

| Setting | Value | Notes |
|---------|-------|-------|
| **Loss function** | Squared hinge + L2 | scikit-learn `LinearSVC`, solved by liblinear |
| **Regularization** | `C = 1.0` | `SVM_HEAD_C` in `vtscore/config/runtime.py`, or the `VTSEARCH_SVM_HEAD_C` env var |
| **Solver iteration cap** | 5000 | liblinear's `max_iter`; the fit is milliseconds at any real vote count |
| **Head** | `Linear(input_dim, 1)` | The `LINEAR_SVM_PRIOR_HEAD` sentinel (`hidden_dim=-3`); no hidden layer, no dropout |
| **Batching** | Full-batch | All labeled data in one solve |
| **Reproducibility** | deterministic | liblinear's primal solve reads no seed (`random_state=seed`, default 42, is passed and has no effect); the fit runs on CPU |

`TRAIN_EPOCHS`, `TRAIN_PATIENCE`, `MLP_DROPOUT` and `MLP_LABEL_SMOOTHING` no longer touch a production fit — they configure the BCE gradient loop, which now only the eval arms run.

### Class Weighting

Training balances classes by **inverse-frequency weighting** by default (`class_weight="balanced"`, liblinear's equivalent of the gradient loop's per-sample weights). The one exception is region flooding on patch datasets, where the caller supplies explicit per-bag `sample_weights` instead — those replace the class balance rather than stacking on it (see [Region-aware training](#region-aware-training-on-patch-datasets) below). Either way, inclusion does **not** enter training — the trained model, and therefore every item's score, is independent of inclusion. Inclusion is the unit cuts are measured in, applied later at threshold time in `conformal_threshold` (see [Threshold Calibration](#threshold-calibration) below); it is no longer a user preference (#4269).

- **Weights**: `weight_true = num_false / num_true`, `weight_false = 1.0` (the same ratio `class_weight="balanced"` derives)

Keeping the model inclusion-independent is what lets the calibration cache reuse fold scores across re-cuts: when the balance moves, or the line is re-cut at another inclusion (the acquisition cut, Smart's pricing), the labels are unchanged, the fold models are unchanged, and only the cheap cut re-runs.

### Label Smoothing (BCE arms only)

The BCE gradient loop label-smooths its targets with ε = 0.05 (`MLP_LABEL_SMOOTHING` in `vtscore/config/runtime.py`): Good examples train toward 0.95, Bad toward 0.05, with class weights still derived from the hard labels. This is **not** a knob-mover — it exists as tie insurance for the conformal threshold rule below, which takes quantiles of the calibration scores and therefore needs distinct score values. Smoothing bounds the optimal logit (≈ ±2.9 at ε = 0.05), so a strongly-fit model cannot saturate every score to exact 0.0/1.0 sigmoids, where all quantiles would collapse to the same cut.

The SVM head needs no such insurance and does not use it: the margin objective has no incentive to run the decision function off to infinity, so its scores stay in a narrow band around the boundary and quantiles over them are naturally distinct.

### Threshold Calibration

**At a glance — what draws the line a user sees**, and its fallbacks:

- **Under the label quota:** the Goods' centroid's count line, or a typed-query detector's own text-sort line, which on a single-vector space holds until 30 labels ([above](#below-the-label-quota-the-goods-centroid-not-a-head)).
- **A trained head:** the **labels line** at the detector's balance *beta* (`fit_labels_line`; [below](#the-labels-line)). It is fitted from the calibration folds' held-out scores of the votes, so the folds described first are its input. While the class model holds fewer than 6 held-out Goods it keeps the head's own count instead (#4786, #4798).
- **No class model even in-sample** (one class), or no balance (a library caller): the fold-anchored mixture cut at Inclusion 0, else the blend.
- **A document (`sift_vlad_doc`) dataset:** the structural inlier gate ([below](#document-datasets)).
- **A typed-query sort** (no detector trained yet): the guarded or count line ([below](#typed-query-sorts)).

Autopilot's Hard and New picks read a separate **acquisition cut**: the target pick precision (0.5), with the reporting line's inclusion − 4 as its fallback ([below](#the-acquisition-cut)). The sections below build the pieces up in the order they compose, each with the report that set it.

The calibration folds come from **cross-calibration**:

1. Split labeled data into Train (1 − `calibration_fraction`) and Calibrate (`calibration_fraction`), **class by class**: each class holds out its own share rounded to the nearest whole vote, halves up, keeping at least one vote of the class on each side (`nearest_holdout`, #4864). There is no coin, so the held-out count is a fixed function of the class's own count and never falls as the class grows. That is what makes the labels line take over once and stay: the line stands when the folds hold out six Goods between them (`HEAD_COUNT_GOODS`), which is 9 Goods on a single-vector space (0.3 held out) and 5 on a patch space (0.5). The stochastic rounding it replaced (#3288) redrew the line by coin over a band of Goods counts, switching it back and forth ([`REPORT.md`](experiments/2026-10-14-split-rounding-4864/REPORT.md)); it is kept as an eval arm, `CALIB_SPLIT_ROUNDING=dither`.
2. Train a fold model on Train **using the same head as the final model**, score the held-out Calibrate portion.
3. Repeat for `calibrate_count` independent random splits.
4. Pool every fold's held-out (score, label) pairs and apply the **conformal inclusion rule** (`conformal_threshold`) once. Pooling — rather than averaging per-fold thresholds — is deliberate: the rule's resolution is bounded by how many calibration scores the quantiles are taken over.

The `calibration_fraction` setting is the Calibrate share of each split, and its default is **per-embedder** (issue #3287): with no explicit user setting, a detector that learns in a single-vector space splits 70% Train / 30% Calibrate (`0.3`), while a patch-grid embedder keeps `0.5` — in *both* its voting styles, including the boxless whole-image fallback, because the measured optimum follows the embedder's capability rather than the voting mode. The table lives in `PRODUCTION_SPLIT_BY_SPACE` (`vtscore/training/thresholds/knobs.py`), resolved with an unknown-space fallback of `0.5` by `resolve_calibration_fraction` (`vtscore/detectors/training_calibration.py`); an explicit `calibration_fraction` user setting always wins, and clearing it returns to the automatic per-embedder default. Evidence: [`docs/experiments/2026-08-27-calibration-fraction-3287/REPORT.md`](experiments/2026-08-27-calibration-fraction-3287/REPORT.md).

The `calibrate_count` setting defaults to `2` (`DEFAULT_CALIBRATE_COUNT` in `vtscore/config/runtime.py`) and can be raised or lowered (`VTSEARCH_CALIBRATE_COUNT`) to trade calibration quality (and cut resolution — more folds means more pooled calibration scores) for latency. (The eval runner uses its own default of `2` for a separate, non-interactive path — see [`docs/EVAL.md`](EVAL.md).) The folds are trained at every label count: the fold *models* are an input to the fold-anchored estimator below — it anchors on their held-out scores — so there is nothing to skip. Every training entry point (vote-driven Train, labelset re-derivation, Find, and detector-load-from-origins) cross-calibrates below 6 labels rather than falling back to 0.5.

**Every trained threshold fuses the haystack into the cut.** There is no setting for this: the `safe_thresholds` toggle was deleted once the fused estimator measured better at every label count (why the population distribution belongs in the cut: [`REPORT.md`](experiments/2026-08-03-safe-thresholds/REPORT.md)). The combiner is the **fold-anchored mixture** (`fold_anchored_gmm_threshold`), which won the #2852 deep-regime study against the old blend. Per calibration fold *k*:

1. Score the whole haystack with fold model *k* and fit a **semi-supervised** 2-component mixture to those scores, with fold *k*'s **held-out** labeled scores clamped to their labeled component (Good → high, Bad → low). The anchors are honest — those items were not in fold *k*'s training set — and they share one score scale with the population they anchor.
2. Cut fold *k*'s fit at the **midpoint between its two component means**, and read that cut's empirical quantile in fold *k*'s own haystack distribution.
3. Average the fold quantiles and realize the result on the **final** model's haystack distribution (rank transfer: two models scoring the same haystack are related by an approximately monotone map, and quantiles survive monotone maps). No raw score ever crosses scales.

**The voted media are not part of any of those haystacks** (#3308). Every model in the chain was trained on some or all of the votes, so a voted item's score under it is optimistically shifted, and the calibration votes would sit in the fold haystack twice. All voted media are therefore dropped from the fold haystacks *and* from the final model's realization sample, which keeps every distribution in the quantile transfer over one population: the unlabeled remainder, the only population the threshold still has to decide. The exclusion switches off entirely when it would leave fewer than `EXCLUSION_MIN_REMAINDER` (60) scores, because a drained remainder is too coarse to read quantiles from and, after deep Autopilot voting, a selection-biased sample; the switch is all-or-nothing so the fold and final populations never diverge. The blend *fallback* below keeps the full distribution: it only runs when there are no usable folds.

**The anchor mass is κ = 0.3, with the midpoint cut.** Each vote counts as three tenths of a haystack point among the ~50k the mixture is fitted on. κ and the cut rule are one interior optimum: `mid` peaks at low κ and `rate` at high. It is the best single global setting measured over six environments in the deep regime (votes 51–300), and it beat the previously shipped `κ=1, rate` in 6 of 6; `qmean` beat `qmedian`, and four folds were no significantly better than two ([`REPORT.md`](experiments/2026-08-05-population-anchored-calibration/REPORT.md)). **One regression is on record**, and it bears only on the fallback, since a trained head's line is the labels line (#4452): against the `cap50` blend schedule it replaced, fusion is a dead heat on COCO binary voting and a loss on an 838-image set, because its gain tracks how many positive anchors the regime supplies. No fix is owed on the app's path: no recorded run since #4878 has a click where a trained head drew no labels line, so the fused cut never decided a line there (#5261, [`REPORT.md`](experiments/2026-10-26-fallback-gate-5261/REPORT.md)); a voting-mode split or positive-count gate would matter only to a library caller that sets no balance.

**Media that cannot be scored are not part of the population.** A head that emits a non-finite logit for a media has that media recorded at the `-1.0` sentinel (`NON_FINITE_SCORE_SENTINEL`, see [`vtscore/docs/packages/utils.md`](../vtscore/docs/packages/utils.md)), outside the sigmoid range so that `score >= threshold` is always false for it. Every estimator here drops those entries before fitting — the fold haystacks, the held-out anchors, the pooled conformal orderings, and the blend's GMM (`scored_only` in `vtscore/utils/scores.py`). Handed to a fit, a spike a full unit below the range pulls the cut below zero, and the detector calls the entire dataset a match (#3180). When *no* media is scorable there is no population at all, and the cross-calibration cut ships alone.

Degeneracy policy, per fold and then globally: a fold whose anchored fit degenerates (inverted means, collapsed component) falls back to that fold's **unanchored** GMM fit; if no fold yields a fit at all, the threshold falls back to the **blend** (`calculate_safe_threshold`), which is what still runs for label sets too small to form calibration folds — never to 0.5. A training path with no haystack at all (detector-load-from-origins, which re-derives from saved labels with no dataset in hand) ships the plain cross-calibration cut. The blend schedules therefore remain in the tree as that fallback and as harness arms, but they no longer decide the shipped cut.

The fitted mixtures are cached on the detector context and **re-cut** at another inclusion — the acquisition cut — with no refit and no re-scoring, so a re-cut lands on exactly what a retrain at that inclusion would have stored.

<a id="the-acquisition-cut"></a>**Under the balance Autopilot samples at a target pick precision (#3546).** `detector_acquisition_threshold` reads `ctx.labels_line` and returns the score where the labels line's corpus posterior — the same 3-part fit the line is cut on — falls below `ACQUISITION_TARGET_PRECISION` (0.5; `target_precision_threshold`): the items the line's own model thinks are even odds. The line − 4 re-cut below is its fallback for a detector with no labels line. **Why:** under the balance that re-cut had saturated into a rank pin nobody chose (near the 98.5th pool percentile at every preset); the target's Hard picks were 38% Good against the re-cut's 19–24%, and it found more Goods at every preset ([`REPORT.md`](experiments/2026-10-07-acquisition-cut-3546/REPORT.md)). A target that moves with the preset was level with 0.5 (#4746, [`REPORT.md`](experiments/2026-10-10-acq-target-4746/REPORT.md)). The rank cut at a share of the mixture's F-beta argmax (`ACQUISITION_ARGMAX_FACTOR`, #4409) shipped and was reverted the next day (#4427), because it harvested the user's unvoted top; the factor is `None`, and the harness keeps it as the `acq_p_crossing` arm. The harness's default arm resolves to the same constant (`resolve_acquisition_target`, the `acquisition.target_precision` mirror); `acq_target_p="off"` restores the offset cut.

<a id="the-labels-line"></a>**The line is drawn at the balance, from the labels alone (#4413, #4452).** Each detector carries a **balance** *beta* (F-beta's beta) — per detector, seeded from the user's `beta` setting, `1` until they change it. The presets are 1/4 (precision-leaning), 1 (balanced) and 4 (recall-leaning), the range's own ends because they give the most distinct choices (#4448); any beta in [0.25, 4] is accepted ([`REPORT.md`](experiments/2026-10-01-fbeta-line-4411/REPORT.md)). The line is **derived from the labels**, so an exported labelset is enough to draw it again later, on another corpus and with another embedder (`vtscore.training.thresholds.labels_line`):

- **The class model.** The calibration folds hold every vote out and score it with a head that never saw it; their held-out scores of the Goods and of the Bads are modelled as two normals of one spread on the logit scale (`class_score_model`). One spread keeps the chance a score is a match rising with the score, so the line is a single cut. Every fit is seeded, so the same labels and embedder re-derive the same model; a new embedder derives its own from the same labels. With one Good the folds hold nothing out, and the head's in-sample scores of the labelled items stand in, so a session's first Good already draws a line. **The spread floor follows the corpus (#4492):** once a corpus is known, the floor is `RELATIVE_SIGMA_FLOOR` (0.5) × its robust logit spread, on the class model and the bulk alike, so the line is invariant to a rescaling of the logit scale. An absolute floor held an early head's compressed negative bulk 2.5× wider than the data, so it covered the positives and the line kept one image for the first ~15 clicks.
- **The corpus side, fitted on the corpus being decided** — the Train dataset in Train, the searched one in Find, AutoFind and the CLI. Two fits of that corpus's unvoted scores (Saerens, Latinne & Decaestecker 2002). A **2-part** fit — the labels' Good component held fixed, a normal for the corpus's own bulk of negatives fitted with its share — gives how many positives the corpus holds, its Good votes counted as they are (`fit_corpus`, `estimate_positives`, `corpus_prevalence`). A **counted bound** guards it: a cut that returns *R* items holds at most *R* positives, and the labels say a share *S1* of all positives clear it, so the unvoted share is held to four times the tightest *R* / (*N* · *S1*) over the cuts that hold at least 5% of the Good component, and a heavy upper tail of negatives is not read as positives. A **3-part** fit — the labels' Good component with its shape fixed, their Bad component with its spread fixed and its mean fitted on the corpus (the held-out Bads counted as labelled points of it), the bulk fitted — gives each unvoted item's chance of being a positive (`corpus_posteriors`). The Bad mean is fitted on the corpus because Autopilot picks the Bads near the line, where their own mean would underrate the positives (#5040, [`REPORT.md`](experiments/2026-10-17-chances-5040/REPORT.md)). Only the labels travel — the labelset is the evidence, never a number derived from it (owner, 2026-10-02). The working assumption is that a Find corpus has the Train corpus's properties; on a corpus with no positives the estimate falls to about zero.
- **The cut** maximises the expected F-beta of what it would really return from that corpus (`corpus_cut`): items best first, the cut after the *k*-th returns *R* = *k* (counted), holds *TP* = the sum of their chances, out of *P* = the 2-part total (or the chances' sum, if larger), and `F = (1 + β²) TP / (β² P + R)`. Each piece is the model that is right about it: the 3-part chances are honest about precision near the top, where the Bads sit; the 2-part total is honest about the positives deeper down, which the Goods a session found — its easiest — under-represent. The cut sits halfway between the last kept item and the first dropped one, so an image re-scored by another path, differing in the last bits, does not flip sides. A cut that returns nothing scores 0; with nothing worth keeping the threshold sits above every item. (A line built with no corpus scores falls back to the parametric cut at a prevalence *p*: with `S1` and `S0` the shares of positives and negatives at or above a cut, `F = (1 + β²) p S1 / (p S1 + β² p + (1 − p) S0)`, `labels_line_threshold`.)
- **When the Bads are a random sample, the negatives take their shape (#4490).** When the class model holds at least `RANDOM_BADS_MIN` (100) Bads and they are at most `RANDOM_BADS_MAX_ENRICHMENT` (2) times over-represented in the corpus's top `RANDOM_BADS_TOP_SHARE` (5%), the negatives' density is a kernel density of the Bads' held-out logit scores, the positives' share is fitted by EM on the corpus, and the cut is the same counted cut over those chances (`LabelsLine.bads_shape`). One normal fitted to every label of an exhaustively labelled dataset under-reads the negatives' upper tail many times over, and the line cut deep there. A session's Bads, picked from the top, are many times over-represented there, so no session within 150 clicks changes ([`REPORT.md`](experiments/2026-10-07-ceiling-line-4490/REPORT.md)). The model keeps the Bads' scores (`ClassScoreModel.neg_logits`) only from 100 and never from the in-sample fallback, whose head saw them.
- **While the class model is thin, the line is the head's count (#4786, #4798, #5039, #5084).** Until the class model holds `HEAD_COUNT_GOODS` (6) held-out Goods (9 in hand on a single-vector space, 5 on a patch one, at two folds; step 1 above), however many Bads (`head_count_serves`; `HEAD_COUNT_BADS` is unbounded since #4798), `LabelsLine.threshold` keeps instead, at every balance, what the Goods' centroid's count rule (`CENTROID_LINE_RULE`) finds standing out in the head's own logit scores of the corpus being decided (`head_count_cut`): the line Tune showed until the hand-over, drawn on the head. Up to β 1 the count is 0.71·β times what stands out (`HEAD_COUNT_PRECISION_SCALE`, `HEAD_COUNT_PRECISION_EXPONENT`; #5084), because a weak head's bulk is heavier than a Gaussian and #4603's count reads its tail as positives. Above β 1 it grows as β^0.189 (`HEAD_COUNT_RECALL_EXPONENT`), and where a held-out Good (`ClassScoreModel.pos_logits`) ranks below the count at β 1 the line keeps at least as deep as the head's excess over a Gaussian bulk is expected to score best at the balance, from 2.75 bulk spreads up (`HEAD_COUNT_RECALL_Z`, `_excess_depth`; #5039): the scores alone cannot tell a weak head's deep positives from a heavy-tailed bulk, but a held-out Good below the count can. **Why a count:** at the hand-over the class model stands on four Bads and two to four Goods, and its cut went wrong both ways (at β 4, one face kept in a fifth of FHIBE sessions, thousands in the weakly separated ones). #4786's gate also handed over at 32 held-out Bads, but past that point, with few Goods, the class model's chances collapse while its 2-part total stays several times the positives, and no fix to `corpus_cut`'s positives total won on COCO, so the Goods alone close the gate (#4798). It is the one place the labels line draws a count on a corpus, so a corpus with none of the target keeps a median of 8 images at β 4; a session that never reaches 6 held-out Goods keeps the count for the whole session. Reports: [#4786](experiments/2026-10-12-beta4-line-4786/REPORT.md), [#4798](experiments/2026-10-13-cut-total-4798/REPORT.md), [#5039](experiments/2026-10-18-head-count-5039/REPORT.md), [#5084](experiments/2026-10-24-head-bulk-5084/REPORT.md). Eval arms restore each earlier line: `CALIB_HEAD_COUNT_BADS=0` is the line before #4786 and `=32` #4786's gate; `CALIB_HEAD_COUNT_RECALL_EXPONENT=off CALIB_HEAD_COUNT_RECALL_Z=off` is the line before #5039; `CALIB_HEAD_COUNT_PRECISION_SCALE=off CALIB_HEAD_COUNT_PRECISION_EXPONENT=off` is #4603's β^0.708.

Past the head's count line's window, no count is drawn on any corpus and nothing caps the line: it keeps every unvoted item at or above the threshold, possibly none. A head whose folds support no class model — they held nothing out (one Good), or their held-out Goods score no higher than the Bads on average — takes the head's own in-sample scores of the labels as its class model, which leans precise; in the second case the labels' separation is still read from the folds, so it reads as weak (#4899). Only labels with no class model even in-sample (one class) take the retrain's fallbacks — the fold-anchored cut at Inclusion 0, else the blend, which admits nothing when the folds never split — never a count. **Known weakness:** the votes are not a random sample (active learning picks near the line and from the top). The 3-part fit corrects the Bads' position (#5040), but not the Goods' selection (the easy positives), so at β 4 the line still keeps short of the best set early, and weak sessions (1–8 Goods) can over-return (#4466). End to end on COCO Better the line beat the count line it replaced at every preset and held up on Find corpora from 200 to 10,900 images (#4452).

Every retrain fits the line and parks it on the detector (`DetectorContext.labels_line`), so a balance change re-cuts it without a retrain, and a Find over a new dataset re-fits only the corpus side there (`LabelsLine.on_corpus`), exactly as a cold Find over the same corpus computes it. The ranking the last retrain or Find pass scored is still parked (`line_ranking`), for the line's state — how many it keeps there — and for the spot check's bands. Every detector has a balance: the Inclusion knob is retired as a user preference (#4269) and survives only as the internal unit the acquisition fallback and Smart's pricing are measured in. Which line a balance draws is one function, `fit_labels_line` (through the retrain's `_labels_line`), shared by the retrain, the no-refit re-cut and the eval harness's default arm. `vtscore.state.line_knobs()` — `{"beta": b}` — is what every retrain and re-cut passes on, so one setting moves every line.

**Why a balance and not a floor (#4413).** The preference was briefly a precision floor, which #4421 removed. Everyone would take perfect precision *and* perfect recall if they could, so the question is which balance of the two the line should maximise; and no single beta lands on a precision target (an F1 cut sits at 91% precision on large objects and 22% on small ones), so nothing maps a floor to a beta ([`REPORT.md`](experiments/2026-10-01-fbeta-line-4411/REPORT.md)). Three things are given up on purpose: the checkable promise "at least *P* of what you see is right" (a check now says what it estimated and how sure it is, and nothing is met or missed); about a third more votes for a check; and a recall estimate that leans on the model's count of the positives, so its range is wider than the precision one.

<a id="document-datasets"></a>**A document (tiled structural) dataset draws its own line (#4367, #4440, #4458).** The labels line above belongs to the SVM path. On `sift_vlad_doc` the returned set is verified pages, by inliers (`maybe_structural_rerank`):
- before the first Bad, at least 8 inliers and a tight fit (#4440);
- after it, more inliers than any Bad reached (the Bad ceiling, #4367).

The rerank drops the ranking, so the balance state on this line is `gate` (#4505): its `count` is the unvoted pages the gate passes (`gate_balance_state`, over the passed set the rerank leaves on the detector context), and no spot check applies.

**The balance moves this line only at its ends.** From beta 2 up (`RECALL_BETA`, nearer the preset 4 than 1), the set also takes every verified page with at least max(10, ⌈0.25 × the Goods' median leave-one-out inliers⌉) inliers, whatever the ceiling and the geometry cuts say: a superset of beta 1's set. At beta 0.5 and below (`PRECISION_BETA`, nearer the preset 1/4) it is the mirror image: beta 1's set intersected with the verified pages holding at least max(16, ⌈0.5 × the Goods' median leave-one-out inliers⌉, the Bad ceiling + 5) inliers, a subset of beta 1's set, and click 0's line rises to 16. So the control stays monotone. Class-split cross-validation on FullMarks chose both floors; the precision end missed its pre-registered per-click limit by 0.003 at one click, and the owner chose to ship it ([`REPORT.md`](experiments/2026-10-03-balance-line-4458/REPORT.md), [`REPORT.md`](experiments/2026-10-04-precision-end-4479/REPORT.md)).

**A document detector stops on the dry run** (#4488): 16 of its best matches in a row that are not Good. Smart, Stable and Span would read the page-VLAD head, which ranks documents near chance, against the structural line, so a tiled dataset's labeling status says `stop_rule: "dry_run"` instead (`dry_run_status`, `document_labeling_status`). Autopilot walks the detector's own ranking with no Good target until the run of misses is complete, then Toasty says the detector is trained. Known limit: at 200k pages 7–14% of positives are still unfound at the stop, which fails the pre-registered 5%; they are the positives outside the 2,000-page Stage-1 shortlist (#4493), and the owner chose to ship ([`REPORT.md`](experiments/2026-10-04-doc-stop-4488/REPORT.md)).

*The spot check* (`vtscore.training.thresholds.spot_check`; #4267, a **band walk** since #4383, stopped at the F-beta peak since #4413). The unvoted ranking is cut into bands from the top - the top 8, the next 8, then 16, 32, 64, ... to its end - and each band is audited with 5 picks drawn uniformly from it (a census when it holds no more), never twice. The set under test is the top *b* bands; its positives are estimated band-stratified, each band's share of right picks times the band's size, which is unbiased for the set whatever the picks per band. The walk (`SpotCheck.start_balance`) fixes *n_pos*, the mixture's count of positives over the unvoted ranking (`mixture_positives`), at its start, and scores the set under test as F-beta = (1 + β²) tp / (β² n_pos + k). It starts at the bands that hold the balance's cap (32 at beta ≤ 1, 128 above), auditing each; then it goes one band deeper while the estimate rises and ends on the peak the first time it falls; from a start whose first deeper step falls, it goes one band shallower while the estimate does not fall (no new picks: a shallower set is a subset of an audited one) and ends on the peak; a tie keeps the smaller set. The ranking's ids are fixed when the check starts, so every band samples one list while the model retrains behind it.

- **The check never moves the line** (`check_shape` is `advisory` at every preset since #4452). Its picks are ordinary votes (provenance `check`) the next retrain learns from — uniform picks within bands, the least biased evidence the labels hold — and the line stays where the labels put it, since a walk's end is a count on one corpus that an exported labelset could not reproduce. The harness keeps the older shapes as forced `walk_shape` arms. The state carries `shape`, `audited` (the set the walk ended on) and `kept_at_check`: how many unvoted items the line kept when the check started, the set the ranges describe (#4483).
- **Two likely ranges** for that set (#4483; `SpotCheck.range_over`): the precision range, each band's part of the set bounded exactly for its known size from the picks that fell in it (hypergeometric, `hypergeometric_lower` / `hypergeometric_upper`, each tail at α; exact where the part was censused, 0–100% where no pick reached it), weighted by size; and the recall range, the same bounds times the part sizes over *n_pos*, the rougher of the two because the picks cannot measure its denominator. The walk decides on the point estimate (the owner's "do your best", #4267); the ranges say how close. The exact bound replaced a union-bounded Clopper–Pearson interval at 0.55–0.76 of its width while still holding the truth in at least 98.9% of checks ([`REPORT.md`](experiments/2026-10-15-check-range-4483/REPORT.md)).
- **A check is due when the labels separate weakly** (#4496): `weak_check_due` reads the labels line's `separation`, d' = (Good mean - Bad mean) / the spread the line was cut with, and calls for a check below `WEAK_SEPARATION_D` (1.5) from `WEAK_CHECK_MIN_VOTES` (10) votes, once a session: never after any check has ended (`WEAK_CHECK_REPEAT`, `None`; #4909), and never on face crops (`WEAK_CHECK_OFF_MEDIA_TYPES`, #4778, #4799: a face opening has found the person's few photos before a check could start). Such labels mark the sessions that over-return, and the check's band-uniform picks are better training votes for them than the acquisition picks they replace ([`REPORT.md`](experiments/2026-10-05-weak-check-4496/REPORT.md), [`REPORT.md`](experiments/2026-10-15-weak-repeat-4909/REPORT.md)). At today's head it pays on Region (+0.0090 over votes 1-150 at beta 1) and is level on Binary, where the query hold hides its early votes (#4797, [`REPORT.md`](experiments/2026-10-28-weak-check-coco-4797/REPORT.md)).
- **Only Autopilot runs a check**, as its Spot Check step: the weak-separation check past its text-sort opening, where no detector is trained; and once a run before it reports Done, the first time every quality light is green (or the labels line expects `STOP_LEVEL` of F-beta at the balance, #4878: `stop_level_met`; or the run holds `STOP_CAP` votes, 200, #4903), unless a check ended in the last `WEAK_CHECK_COOLDOWN` votes (`done_check_due`). It runs on every media type, face crops included, and there is no Check button. A finished result belongs to its beta and is kept on the detector for its ranges: later votes retrain the model and the labels' line with it, and the ranges are reported `stale` once the set under them has moved. The headless paths (AutoFind, the CLI, the cold Find path), which have nobody to vote, export the labels' line unchecked.
- **Why a check:** every vote a model chose is a biased sample of that model's scores — learned-sort evidence alone broke 83% of the #4220 estimator's promises of 50% precision (#4256) — and a uniform pick has no such bias. The walk's count follows the corpus, where a fixed top 32 was right on one bench only ([`REPORT.md`](experiments/2026-09-30-line-estimate-4383/REPORT.md)).

*The #4220 estimator stays as library API, and the app no longer builds it* (#4362). `precision_floor_cut` and `PrecisionFloorEstimate` are public library API the calibration studies still call: each fold's held-out votes ranked in that fold's own haystack, a logistic `P(positive | percentile)`, a 10th-percentile bootstrap lower bound on the precision of every top *k*, and an EM re-estimate of the corpus prior. `precision_floor_cache` stays `None`, and the names of the filter that cut its evidence down to the votes drawn off the learned sort (`calibrates_precision`, `eligible_fold_orderings`, `vote_calibrating_groups`, `labelset_calibrating_groups`, `calibration_rows_for`) are deprecated.

**How Inclusion reaches the cut (`mid_tilt`, #2865).** Inclusion arrives at the fold-anchored estimator only as the rate weights a cut rule optimizes, and the bare midpoint ignores them, so shipping it verbatim made the knob a no-op. The shipped rule anchors at the midpoint and tilts by the rate rule, in fold-quantile space: `q(k) = q_mid + (q_rate(k) − q_rate(0))`. At inclusion 0 the shift is zero, so the threshold is bit-for-bit the measured `κ=0.3, mid` arm; away from 0, raising Inclusion can only lower the cutoff, and the included sets stay nested. The rate cut is the *highest* score at which the Bad component still out-densities the Good one under the cost tilt, and once that crossing leaves the inter-mean interval it continues past the mode at the rule's own first-order slope (`var/d` per nat of log-cost) rather than pinning to the edge, which had deadened the knob over whole bands (#2896). A fold too degenerate for a rate cut contributes a zero shift. `mid_tilt` was not beaten across four environments and thirteen stops of the knob, and the eval-only `cross_tilt` and `q_tilt` arms do not ship ([`REPORT.md`](experiments/2026-08-21-inclusion-cut-rule/REPORT.md)). **Why `mid` beats `rate` at inclusion 0:** not because `rate` reads the acquisition-biased mixture weights, as the anchor-mass report says (its prior-odds factor cancels them exactly), but the **variance asymmetry**: `rate` solves the equal-density crossing, which sits off the midpoint whenever the components differ in width.

**The acquisition cut is not the decision line, and its fallback is an offset (`ACQUISITION_INCLUSION_OFFSET`, −4; PR #2876, #3319).** The threshold does two unrelated jobs. *Reporting* is the line the user sees, what Find calls a match. *Acquisition* is what Autopilot's **Hard** and **New** picks consume as a *sampling position*, not a decision boundary: Hard takes the nearest unlabeled item to it **in rank space**, and New steers the atlas probe by a node's median score. A **negative** inclusion raises the cut, moves it up the ranking, and so samples where more positives are. Where there is no labels line for the target-precision rule above, production re-cuts the same fitted estimator at `inclusion − 4` for the selector and leaves reporting where it is (`acquisition_inclusion` in `vtscore/training/thresholds/knobs.py`; `detector_acquisition_threshold` in `vtscore/state/core.py` derives it per request from the cached mixtures, a re-cut with no refit). The `inclusion` there is wherever the reporting cut sits, never a value the user sets: `detector_acquisition_threshold` recovers it from the reporting threshold with `FoldAnchoredCut.inclusion_for_threshold`. **Why −4:** against no offset it reaches the answer `k=0` ends its session with in about half the clicks. The decision endpoint is a plateau from −2 to −5, so −4 rests on labelling efficiency (hard-pick precision 24% → 35%), a product judgement; deeper is bounded by the deep-spike guardrail and by region voting, where −4's margin is thin — **the number to re-measure first if this is revisited.** It is not gated by voting mode (the two binary environments disagree with each other more than the modes do), and it is an offset, not an absolute cut. The knob under-delivers its own steps (log₂-odds of pick precision on `k` has slope −0.65 binary / −0.78 region against a calibrated −1.00), which is why its successor targets a pick precision: #3546 shipped that rule under the balance (above), and the deep regime it could not settle is **#3547**. Evidence: [`REPORT.md`](experiments/2026-08-07-acquisition-inclusion/REPORT.md) (COCO), [`REPORT_SECOND_ENVIRONMENT.md`](experiments/2026-08-07-acquisition-inclusion/REPORT_SECOND_ENVIRONMENT.md) (VG binary), [`REPORT_REGION_VOTING.md`](experiments/2026-08-07-acquisition-inclusion/REPORT_REGION_VOTING.md) (VG region — **voided by #2943, read its banner before citing anything in it**) and [`REPORT_PILE_2877.md`](experiments/2026-08-07-acquisition-inclusion/REPORT_PILE_2877.md) (three environments, both voting modes).

Everything else still reads the reporting cut: the green/red line, the above-threshold count, Find verdicts and the Find boundary walk, exports, and the labeling-progress indicators. Only the two Autopilot picks moved. The learned sort's response carries an `acq_threshold` for them, and since #4136 so does the text sort's (next paragraph).

<a id="typed-query-sorts"></a>**A typed-query sort carries the same two lines, and they are drawn by different rules (#3826, #4136; `text_sort_cuts` in `vtscore/training/thresholds/text_sort.py`).** A cosine sort of a typed query is usually *one* broad mode with the matches as a thin right shoulder, so the two-Gaussian midpoint every cosine sort used to draw split the mode in its densest region and admitted a median 43% of the haystack ([`REPORT.md`](experiments/2026-09-22-text-cut-3826/REPORT.md)). The display line is the **guarded rule** (`guarded_text_sort_threshold`): keep the mixture, continued to convergence, when its components are separated (Ashman's D ≥ 2, about 9% of sorts, nearly all single-object images), else cut the bulk's median + 3 robust sigmas (the sigma read from the half below the median, which a heavy shoulder of matches does not reach); re-scored on the objective it wins at beta 1/4 and 1 and ties at 4 ([`REPORT.md`](experiments/2026-10-06-cost-fbeta-rescore-4582/REPORT.md)). Its known failure is a majority-class query the embedder does not separate ("a person" on COCO): the median sits inside the matches. Nothing label-free tells that sort from a query the embedder barely separates, so it is documented and pinned by a test.

**That line is the display line only.** At the guarded line Autopilot's Bad phase sampled near the top of the ranking, voted true matches while hunting negatives, and trained the first detectors on near-miss negatives only ([`REPORT.md`](experiments/2026-09-23-text-cut-ab-3826/REPORT.md)). So the text route sends `threshold` = the guarded line and `acq_threshold` = the mixture midpoint (`text_sort_acquisition_threshold`), and the Hard select reads `acq_threshold ?? threshold` as it does on a learned sort: the opening's picks are the midpoint arm's by construction, which `tests_lib/sorting/test_text_sort_threshold.py` pins by running the harness's opening under both rules. `VTSEARCH_TEXT_SORT_CUT=gmm_midpoint` restores the pre-#3826 line for display (the acquisition cut is unchanged by the flag); the eval harness's Bad phase and `@mid` schedule cut call the acquisition function, and `scripts/experiments/calibration/text_baseline.py`'s click-0 anchor scores the display line, since that is what the user sees.

**At a balance of beta 1 or below the display line is a count instead** (#4603, `_count_line`): it keeps the top `round(c(beta) · n_hat)` of the sort, where `n_hat` is the excess over a Gaussian bulk above median + 4 robust sigmas (`TEXT_SORT_COUNT_Z`) and `c(beta) = beta^(ln(3/8)/ln(1/4))` (3/8 of the estimate at beta 1/4, all of it at beta 1); with fewer than 50 scores, or above beta 1 (`TEXT_SORT_COUNT_MAX_BETA`), the guarded line stays. It beat the guarded line at beta 1/4 and 1 on COCO Better. The acquisition cut does not move with beta. The guarded and count rules were measured on typed queries only: an example sort's display line is the Goods' centroid's line over the unvoted media ([above](#below-the-label-quota-the-goods-centroid-not-a-head), #4732, #4779; `query_sort.cosine_sort_cuts`) with the same midpoint as its acquisition cut, and a label-file sort trains a head on the file's labels (`train_and_score_active`) and sends that head's line alone.

**Why fold models share the final model's head:** a different head produces a different score distribution, so a threshold found on fold models would not transfer faithfully to the final model. The training code threads one `hidden_dim` sentinel through both the final fit and every fold fit, so the two can't drift apart. With a linear head that value is a constant (`LINEAR_SVM_PRIOR_HEAD`), which makes the property automatic — the head has no capacity to size.

The **Calibration Fraction** setting (0–1; unset by default, which selects the per-embedder value above) controls how much data is reserved for threshold calibration vs. model training in each split. For example, a value of 0.2 means 80% Train / 20% Calibrate. If the fraction is so extreme that a valid Train/Calibrate split cannot be formed (fewer than 2 training examples or fewer than 1 calibration example), the system returns a maximum threshold so that nothing is predicted as Good.

The threshold is a **split-conformal quantile rule** over the pooled held-out scores, governed by the `inclusion_value` parameter (integer in range [-10, +10]; an internal unit since #4269, which no user sets: the fallback line and Smart's pricing cut at 0, the acquisition fallback four steps stricter than the line). This is where inclusion biases the result toward recall or precision — at calibration/threshold time, **not** at training time. For `k = inclusion_value` (with `BASE = 0.25`, `QPOS_MAX = 0.75` — `CONFORMAL_BASE_BUDGET` / `CONFORMAL_QPOS_MAX` in `vtscore/training/thresholds/conformal.py`):

- A **false-negative cap** `α(k) = min(1, BASE·2⁻ᵏ)`: the threshold never exceeds the α-quantile of the held-out *positive* scores, so an estimated at-most-`α` fraction of true matches falls below the cut. `+k` therefore has a portable meaning — "the fraction of true matches I'm willing to miss, halving per step" — independent of dataset or detector (e.g. `+3` ≈ miss at most ~3%, `+10` ≈ miss at most ~0.02%). The cap is an upper bound, not a target: when the classes separate cleanly the cut drops into the gap below the calibration positives and the budget goes unspent.
- A **false-positive guard** for `k ≤ 0`: the threshold stays at or above the `1 − BASE·2ᵏ` quantile of the held-out *negative* scores (so overlap-heavy tasks keep FPR control) and above a walk *up* toward the positive score distribution. The walk interpolates linearly in score space from the **gap midpoint** at `k = 0` to the `QPOS_MAX` quantile of positives at `k = −10` ("just the surest matches").

The **gap midpoint** is the cut's default anchor. When the classes separate cleanly there is an empty band between the top of the negatives and the lowest calibration positive, and every cut inside it has identical empirical error on the calibration set. The band's top edge is the worst of those equals: an extreme order statistic over a handful of held-out votes, measured on the saturated *fold* models' score scale but applied to the *final* model's, so it routinely landed above every score the final model produced and admitted nothing, not even the user's Goods (#2781). The midpoint is the max-margin choice among cuts the calibration data cannot distinguish, and it spends no FN budget. Under class overlap there is no band, the midpoint collapses to the false-positive guard, and the rule is unchanged.

The rule is **monotone in `k` by construction**: raising inclusion can only lower the threshold, so included sets are *nested* — everything included at Inclusion 1 stays included at Inclusion 4. (The previous min-cost argmin over observed cuts had exactly as many distinct optima as the calibration folds had ranking errors, so on well-separated votes the knob provably never moved; see `docs/experiments/2026-07-27-inclusion-knob/REPORT.md` and issue #2693.)

Because the fold models are inclusion-independent, the pooled held-out scores can be cached once and re-thresholded at any inclusion (this is what lets the line move with the balance, and the acquisition cut sit at another inclusion, without a retrain).

The **GMM-based threshold** fits a 2-component Gaussian Mixture Model to a score distribution and places the cut at the **midpoint between the two fitted component means**. It is the acquisition cut of typed-query and example sorts, whose display lines are drawn by the rules [above](#typed-query-sorts), and the GMM half of the safe-threshold blend, which is now only the fallback for label sets too small to form calibration folds.

**Why not the equal-density crossing?** The crossing of the two weighted components (#2798), the Bayes boundary between them, sits above the midpoint under region voting, whose Bad mode is right-skewed and wide. It does cut higher and does buy FPR, but at ~1.3 FNR per FPR, and it lost on cost in every max-pooled window (`docs/experiments/2026-08-03-safe-thresholds/REPORT.md`); for a needle-finding tool the missed match is the worse error, so #2833 reverted production to the midpoint. The crossing solver stays in the tree as an eval variant, and why the midpoint wins is open (#2836).

**The pinned seeds: which are real draws, and which are decorative.** Several `42`s appear in this pipeline, and they are not equivalent — only one of them is a coin flip whose outcome a user can feel:

| Site | What the seed decides | Does it move the detector? |
|---|---|---|
| `CALIBRATION_SPLIT_SEED` (`vtscore/training/thresholds/conformal.py`) | **which labels land on the Calibrate side** of each fold | **Yes.** Different splits leave different held-out scores, so the conformal quantile cuts somewhere else. This is the pipeline's one genuine draw. |
| `fit_linear_svm_prior_head(seed=…)` → `LinearSVC(random_state=…)` | nothing: the shipped head is solved in liblinear's primal, which reads no seed | No. A convex solve with a unique optimum. (The plain SVM arm, `fit_linear_svm_head`, lets scikit-learn pick, which is the dual while votes are fewer than dimensions; there the seed permutes the path to the optimum, not the answer.) |
| the `gmm_fit_array` subsample | which scores are kept above `_GMM_MAX_SAMPLES` | Rarely. A large uniform draw from a distribution whose quantiles are the only thing read off it. (The mixture fit itself has no seed since #3585: a deterministic 2-means init and our own EM, not `GaussianMixture(random_state=42)`.) |

All of them are *pinned* rather than drawn fresh, and that is the point: issue #2934 fixed exactly the opposite state of affairs, where the uncached threshold paths drew their fold splits from the unseeded global `np.random` and a detector's verdicts moved between two runs on identical votes. Reproducibility is the feature.

What the pin costs is that every user gets the *same* arbitrary draw and no study reports the spread around it. That is a measurement gap, not a reason to unpin: the eval harness's `calibration_seed` knob (issue #3794) holds the data fixed and sweeps the split instead, so the spread can be quoted as an error bar on single-seed numbers. See [`docs/EVAL.md`](EVAL.md#the-calibration-draw-calibration_seed-default-the-apps-own-pin).

### "All my Goods rank above all my Bads" — is the head over-training?

No, and the symptom cannot tell you either way. Embeddings are 512–768-dimensional, so a few hundred labeled points are almost always linearly separable for **any** labeling — including a pure coin flip. A head that fits its own training labels perfectly is therefore doing what a correctly-specified model does on separable data, not memorizing; shrinking the head would not change the symptom, because the symptom is a property of the geometry rather than of the model's capacity.

Over-training is only visible on items the head did **not** train on. The shuffled-label control makes that concrete: run the same dataset, the same train/held-out split, and the same head twice, changing only the labels.

- **REAL** — label each item by its true category (a signal that exists).
- **NOISE** — label each item by a fixed coin flip (a signal that does not).

Both arms reach ≈1.0 training AUC. That is the whole point: perfect training separation happens even for pure noise, so it proves nothing. The held-out numbers are what separate them:

| Arm | Train AUC | Held-out AUC | Reading |
|---|---|---|---|
| REAL | ≈ 1.0 | well above 0.5 | The head is genuinely learning; the perfect training separation is *good behavior*. |
| NOISE | ≈ 1.0 | ≈ 0.5 | The head correctly fails to generalize noise — it is not a memorizer faking held-out signal. |

A high held-out AUC on the NOISE arm would be the real alarm: that is label leakage or a broken split, not over-training.

The same control answers the neighbouring question — "is the capacity right?" — by sweeping the hidden width (or the head sentinel) with the split, votes and seeds held fixed, and reading held-out AUC per width. Note that production ships the **linear SVM head**, which has no width to sweep: capacity questions apply to the BCE eval arms in [The heads](#the-heads-which-one-is-shipped-and-why) above.

## PyTorch Environment Settings

| Setting | Where | Value |
|---------|-------|-------|
| `OMP_NUM_THREADS` | `app.py` | the resolved thread count |
| `MKL_NUM_THREADS` | `app.py` | the resolved thread count |
| `torch.set_num_threads` | `vtscore/media/torch_setup.py` (`ensure_torch_configured`) | the resolved thread count |
| dtype | `vtscore/training/mlp.py` | `torch.float32` |
| Device | default | CPU (GPU supported, see tests) |

All three thread settings read one value, `VTSEARCH_TORCH_THREADS`. The **server** (`python app.py`) resolves it in `vtsearch/torch_threads.py` — the env var if set, otherwise the process's CPU allocation (`os.sched_getaffinity`), because a person waits on every vote — and publishes it to the OMP/MKL variables before torch is imported. Anything that imports `vtscore` without going through `app.py` (library use, batch scripts, tests) gets `vtscore.config.TORCH_THREADS` (`vtscore/config/runtime.py`), which defaults to `1` to keep memory overhead low: each thread allocates its own scratch buffers. See [`docs/DEPLOYMENT.md`](DEPLOYMENT.md) for the operator knob. The real cost is the embedding models, not the head. (The head's own fit is liblinear on the CPU and is milliseconds either way.)

## Embedding Models

Each embedder produces fixed-size embedding vectors from a pretrained model. The full roster is generated from the live registry (the `document` media type has no embedder of its own, so it is absent here — documents are converted to other media types before embedding):

<!-- BEGIN GENERATED: embedders -->
<!-- Generated by scripts/gates/gen-docs-inventories.py; do not edit by hand. Refresh with: python scripts/gates/gen-docs-inventories.py -->

| Media type | Embedder | Display name | Model | Dim | Notes |
|---|---|---|---|---|---|
| `audio` | `clap_general` | CLAP (general, larger) | `laion/larger_clap_general` | 512 | **default** for its media type |
| `audio` | `ast` | AST (audio spectrogram) | `MIT/ast-finetuned-audioset-10-10-0.4593` | 768 | no text queries |
| `audio` | `beats` | BEATs (audio events) | `lpepino/beats_ckpts` | 768 | no text queries |
| `audio` | `clap` | CLAP (general, faster) | `laion/clap-htsat-unfused` | 512 | — |
| `audio` | `clap_music` | CLAP (music) | `laion/larger_clap_music_and_speech` | 512 | — |
| `audio` | `embeddinggemma2_audio` | EmbeddingGemma 2 (audio) | `google/embeddinggemma-2` | 768 | — |
| `audio` | `paraspeechclap` | ParaSpeechCLAP (speech style) | `ajd12342/paraspeechclap-combined` | 768 | — |
| `audio` | `whisper_encoder` | Whisper encoder (speech) | `openai/whisper-base` | 512 | no text queries |
| `face` | `face` | Face identity (512d) | — | 512 | **default** for its media type; no text queries; restricted model license |
| `image` | `siglip` | SigLIP (general images) | `google/siglip-base-patch16-224` | 768 | **default** for its media type |
| `image` | `clip` | CLIP (general images) | `openai/clip-vit-base-patch32` | 512 | — |
| `image` | `clip_l` | CLIP ViT-L/14 (evaluation only) | `openai/clip-vit-large-patch14` | 768 | **evaluation only** — registered, but withheld from the app's pickers |
| `image` | `dinov2_patch` | DINOv2 patch (region-aware images) | `facebook/dinov2-base` | 768 | no text queries; patch grid (region-aware) |
| `image` | `dinov2_single` | DINOv2 single (image vector) | `facebook/dinov2-base` | 768 | no text queries |
| `image` | `dinov3_patch` | DINOv3 patch (region-aware images) | `facebook/dinov3-vitb16-pretrain-lvd1689m` | 768 | no text queries; patch grid (region-aware) |
| `image` | `dinov3_single` | DINOv3 single (image vector) | `facebook/dinov3-vitb16-pretrain-lvd1689m` | 768 | no text queries |
| `image` | `embeddinggemma2` | EmbeddingGemma 2 (images) | `google/embeddinggemma-2` | 768 | **evaluation only** — registered, but withheld from the app's pickers |
| `image` | `eupe_patch` | EUPE patch (region-aware images) | `https://huggingface.co/facebook/EUPE-ViT-B/resolve/main/EUPE-ViT-B.pt` | 768 | no text queries; patch grid (region-aware); restricted model license |
| `image` | `eupe_single` | EUPE single (image vector) | `https://huggingface.co/facebook/EUPE-ViT-B/resolve/main/EUPE-ViT-B.pt` | 768 | no text queries; restricted model license |
| `image` | `sift_vlad` | SIFT/VLAD (instance matching) | — | 8192 | no text queries; geometric verification |
| `image` | `sift_vlad_doc` | SIFT/VLAD (document scans) | — | 8192 | no text queries; geometric verification |
| `image` | `siglip2` | SigLIP 2 (general images) | `google/siglip2-base-patch16-224` | 768 | — |
| `image` | `siglip2_l` | SigLIP2-L (SO400M/384) | `google/siglip2-so400m-patch14-384` | 1152 | — |
| `image` | `siglip_l` | SigLIP-L (SO400M/384) | `ViT-SO400M-14-SigLIP-384` | 1152 | — |
| `text` | `e5` | E5 (text) | `intfloat/e5-base-v2` | 768 | **default** for its media type |
| `text` | `bge` | BGE (text) | `BAAI/bge-base-en-v1.5` | 768 | — |
| `video` | `xclip` | X-CLIP (video) | `microsoft/xclip-base-patch32` | 512 | **default** for its media type |
| `video` | `languagebind` | LanguageBind (video) | `LanguageBind/LanguageBind_Video_V1.5_FT` | 768 | — |
| `video` | `videomae` | VideoMAE v2 (action features) | `OpenGVLab/VideoMAEv2-Base` | 768 | no text queries |

<!-- END GENERATED: embedders -->

Each embedder lives in its own `embedder_<name>.py` file inside the media-type package and exposes a module-level `EMBEDDER` sentinel; the default for a given media type is whichever embedder overrides `is_default` to return `True` (exactly one per media type). Each embedder declares its output dimensionality via the `embedding_dim` descriptor property (the "Dim" column above).

Most media types ship alternative embedders alongside the default. The image variants come in **single/patch pairs**: `_single` embedders produce one CLS-pooled vector per image (cheap, same shape as SigLIP); `_patch` embedders additionally produce the raw `H × W × D` patch grid (196 vectors on a DINOv3 14×14), enabling region-level similarity, region-aware detector scoring, and region voting on yes-votes.  See [`ARCHITECTURE.md`](ARCHITECTURE.md#embedder-binding-and-routing) for the full design.

Embedders carry capability flags consumed by the routes layer and the frontend:

- `supports_text: bool`: whether the embedder can embed text queries. Text-sort returns HTTP 400 + `supports_text: false` when this is false.
- `supports_patch_regions: bool`: set on the `_patch` variants. Loaders that see this flag populate `media["patch_grid"]` (raw `H × W × D` fp16) in addition to the embedder's vector in `media["embeddings"]`.
- `license_notice: Optional[str]`: non-None for embedders with usage restrictions (e.g. EUPE's FAIR Noncommercial Research Licence). Surfaced as a warning chip on the embedder picker.

The **document** media type has no embedding model of its own. Documents (PDF, DOC, PPT) are intended to be converted to other media types (images or text) via media converters in `vtscore/converters/` before embedding.

The **face** media type is never imported either: the `image2face` converter hands each image to the server's face provider (the `imagekit` package: an SCRFD detector and an ArcFace-contract embedder on ONNX Runtime, or a proprietary backend behind the same interface), which finds every face and embeds it from the full photo, aligned on its five landmarks; the crop leaves with its vector already filed under the `face` embedder. On the FHIBE face bench this path beat the MTCNN + FaceNet one it replaced in every cell, +0.047 to +0.11 session mean, and closed most of FaceNet's gaps between people ([`REPORT.md`](experiments/2026-10-26-imagekit-face-bench-5321/REPORT.md), #5320). The app downloads no face models, so Face is on only where a deployment supplies them ([`DEPLOYMENT.md`](DEPLOYMENT.md#face-models)). **The stored crop is turned upright** (#4777): the photo is turned about the box's centre by the angle that takes the landmarks onto the ArcFace template before the same box is cropped (`align="roll"`, the default; `none` is the plain box). This changes only what a person looks at; the vector does not depend on it. On the MTCNN path, roll alignment raised the face review in every session set and cut failed one-photo sessions from 12 to 7 ([`REPORT.md`](experiments/2026-10-10-face-align-4777/REPORT.md)).

Embeddings are computed once when a dataset is loaded. The full-image vector lands in each clip's `"embeddings"` dict, keyed by embedder name (`numpy.ndarray` values; read it through the `media_embedding` accessor); patch embedders additionally populate `"patch_grid"` (`H × W × D` fp16 ndarray, re-derived at load — never persisted). The detector head trains on these pre-computed vectors, so training is fast (the liblinear head fit is milliseconds on a few hundred labeled examples; scoring the dataset dominates a retrain).

### Description enrichment

The Settings toggle **Enrich descriptions** (`enrich_descriptions`, off by default) changes one thing: the *query* vector. With it on, `embed_text_query` replaces the embedding of what you typed with the L2-normalised mean over the embedder's `description_wrappers` applied to it — `clap_general`'s are `"the sound of {text}"`, `"a recording of {text}"`, `"{text}"`, `"audio of {text}"`, `"the noise of {text}"` — via `MediaEmbedder.embed_text_enriched`. Media vectors are untouched. Its consumers are the Text Sort box (`vtsearch/routes/sorting.py`) and a typed-query detector, which embeds its query as the text sort does (`query_head.embed_detector_query`: the query's head under the label quota, and the trained head's typed-query prior). It costs 5 text-encoder passes instead of 1 on the query alone (~20-30 ms), cached per `(embedder, media type, enrich, text)`.

**It is not a general improvement, and the default stays off** (#3127, [`REPORT.md`](experiments/2026-08-31-enrich-descriptions-3127/REPORT.md): every media-type default, 22 eval datasets, 560 paired queries). The effect belongs to the embedder-and-corpus pair, not to enrichment: only `clap_general` (+0.014 text-sort AP, inside 2 SE) and `xclip` have a positive point estimate on their own model, it is inert on `siglip`, and it costs `e5` and `bge` about 0.06 AP on every category. So the wrapper list is **per-embedder** (#3341): `siglip`, `e5`, `bge` and `clap` declare no wrappers at all, so `embed_text_enriched` falls back to `embed_text` and the toggle is a no-op for them rather than a small silent cost, and only `clap_general` and `xclip` keep theirs. `{text}` is itself one of the five templates, so where an embedder has wrappers, enrichment always averages the plain query back in.

Two general CLAP checkpoints disagree on the *identical* five templates, so **a sibling model's prompts are not evidence for yours**. A new embedder should leave `description_wrappers` empty unless it has measured otherwise — see [EXTENDING-media.md](EXTENDING-media.md#adding-a-media-embedder).

`face` (no text tower) and `document` (no embedder of its own) cannot text-sort, so the setting is inert for them.

### Region-aware training on patch datasets

Every patch media has one **score-row stack** — `media_score_rows` in `vtscore/embedding/matrix.py` — and it is the single definition of the geometry:

| row | vector | box |
|---|---|---|
| `0` | the image-level (CLS) vector | whole image |
| `1 .. H·W` | every raw patch, row-major (`1 + r·W + c` is grid cell `(r, c)`) | that one cell |

Inference max-pools the head over that stack (an image scores by its **best** row — see `score_rows_with_model`). Training is shaped to match, and it is deliberately asymmetric between Good and Bad votes — the multiple-instance-learning treatment of a max-pool bag:

- **Good vote** — a positive bag needs only *one* good region, and the user tells us which via an optional `region_box` (drawn by Shift-drag on the focus pane). The vote trains on **the raw patch nearest the box** (`nearest_patch_to_box`), which is by construction one of the rows the max-pool will score. A Good vote with no box falls back to the image-level vector — row 0 of the same stack.
- **Bad vote** — a negative bag asserts that *no* region is good, so a Bad vote **floods the entire stack** (image-level vector + every raw patch) as negatives. This trains every row down, so the max-pool can't surface a look-alike sub-region of a rejected image.

**The invariant tying the two together: every vector a vote can train on must also be a row that is scored.** Both bullets above call the same `media_score_rows`, so it holds by construction rather than by two implementations agreeing. It is not cosmetic — an early MaxPatch prototype scored raw patches only, so a *boxless* Good vote trained on the image-level vector, a vector nothing ever scored. The classifier learned to separate "full-image-like" from "raw-patch-like" (every Bad vote floods raw patches), and the calibrated threshold landed in a gap the score distribution never reaches: perfect ranking, zero FPR, catastrophic FNR.

This geometry (**MaxPatch**) replaced a HAC region tree in #2886. Tree-free MaxPatch beat the tree on ErrorCost in every scale band, most on small objects, where a raw patch is a near-pure object sample while the tree's smallest pooled leaf already blends object with context; and having no internal nodes, it has no flood/score gap (#2731). See [`docs/experiments/2026-07-29-max-patch/REPORT.md`](experiments/2026-07-29-max-patch/REPORT.md).

**Its cost** (DINOv3 14×14, D = 768, CPU) is in scoring: 197 scored rows per image, a ~3.0 GB float16 flattened score matrix per 10k images, and ~110 µs per image per scoring pass. A retrain runs three scoring passes (the final model plus one per calibration fold), so a 10k-image collection pays roughly 3 s per vote. Ingest is only the fp16 cast of the grid (~0.5 ms per image).

Two things keep that bounded. The flattened matrix is kept **float16** (the grid's own dtype) and upcast chunk-wise by both consumers (`_forward_sigmoid_chunked`, `chunked_row_scores`), so peak float32 memory is `ROW_CHUNK × D × 4` regardless of dataset size; and `_build_region_arrays` allocates the matrix once and fills it in place rather than concatenating per-media blocks, which would hold 2× the matrix at peak. The matrix itself is cached on the `DatasetContext` and rebuilt only when the media-id set changes — never per vote.

Because flooding turns one Bad vote into ~197 correlated rows, class balance and calibration are **per-bag, not per-row**:

- The final fit is `train_model(..., sample_weights=...)` where each Bad image's rows share one image's worth of negative mass (`_per_bag_fit_weights`), so a rejected image counts once regardless of row count. Good votes weigh `n_bad_bags / n_good`, matching the default inverse-frequency balance but with the *bag* as the unit.
- Cross-calibration (`compute_fold_orderings(groups=...)`) splits Train/Calibrate **by bag** (a Bad image's rows never straddle the boundary), sizes fold counts over votes not rows, weights fold fits per-bag, and **max-pools each calibration group to one score** — so the threshold is placed on the per-image score scale the detector actually deploys. The fallback blend's ramp (and an MLP eval arm's hidden width) likewise sizes on vote count.

Flooding applies wherever a detector is trained from labels the app will then score by region-aware max-pool: the Learned-sort vote path (`train_and_score`), the saved-detector labelset path (`labelset_train_and_score` / `train_from_labelset`), and — since #3525 and #3544 — **both on-demand cold trains, which delegate to that same labelset path** rather than re-deriving a head of their own: cross-dataset Find's (`_score_with_cold_detector`) and the active-dataset one behind `/api/find-label`, `/api/auto-detect` and the portable export (`resolve_or_train_detector`). That delegation is what makes one labelset mean one detector: a detector trained by drawing boxes stays a region detector when Find points it at a dataset it was not loaded against. Both of Find's scorers therefore read `scoring_rows_for_snap` at the ordinary max-pooled geometry, the *live* one for a head that was flooded and whose `det_ctx.threshold` was cut on the max-pooled distribution, the *cold* one for a head just built the same way. Reading the image-level row alone for either would compare a systematically lower score against a cut placed on a higher one, and the detector would under-return on every patch dataset it is pointed at.

The remaining unflooded path is label-file sort, which scores each image by one vector and is intentionally *not* flooded (flooding patch negatives while scoring one image vector would be a train/score space mismatch). `scoring_rows_for_snap(..., region_pooling=False)` exists for exactly that shape of caller — a head that never saw a patch as a negative must not have a distractor patch promoted into its media score — and has no shipped caller now that neither cold path builds a whole-image head. On any dataset whose embedder produces no patch grid, every bag holds one row and the whole path collapses byte-for-byte to the plain single-vector fit.

## Coverage Atlas

The Coverage Atlas (`vtscore/coverage/atlas.py`, class `CoverageAtlas`) is a hierarchical k-means partition of a dataset's embedding space that remembers, per region, how much labeled evidence of each class the user has provided. Its job is **diversity sampling**: the Training autopilot's "Explore Diversity" phase asks it for the next item to label, so a handful of clicks covers the whole collection and stress-tests the model where it is most likely to be wrong.

It used to have a second job, domain-shift detection, through calibrated per-node typicality p-values. #3348 measured that calibration as wrong on every embedder and replaced it with a split-conformal kNN report that does not use the atlas at all. See [Domain shift](#domain-shift).

One atlas exists per dataset (`DatasetContext.coverage_atlas`). It replaced the earlier Diversity Tree, which kept only a boolean "seen" flag per region; the atlas keeps the geometry and statistics the tree threw away. The full design study (including the not-yet-built portable artifact, blob scan, and active auditor) lives in [`docs/plans/coverage-atlas.md`](plans/coverage-atlas.md).

### Geometry: center, then normalize

All stored embeddings are unit vectors (L2-normalized at ingest), and contrastive embedders concentrate them in a narrow cone — raw cosines between any two items are uniformly high, which makes raw directions nearly useless for partitioning or typicality. The atlas therefore works in a **centered spherical frame**: it subtracts the dataset's mean vector and re-normalizes to the unit sphere. The centering vector is part of the structure and every query is mapped into the same frame.

One consequence worth remembering: the **root node is directionally degenerate by construction**. Centering makes the sum of all vectors (the "resultant") vanish, so the root has no preferred direction. Everything below the root — the k-means cells — is cohesive and directional. `next_sample` keys off this via the resultant length `rbar` (see typicality tempering below).

### Build

Built automatically at dataset load for datasets up to 50 000 items (`COVERAGE_ATLAS_AUTO_THRESHOLD`), on demand via `POST /api/datasets/registry/<id>/coverage-atlas` for larger ones, and cached inside the dataset pickle (key `"coverage_atlas"`, format `"coverage-atlas/1"`) so reloads skip the k-means. A cache written before #3348 also carries per-node `t_quantiles`, which are ignored. The build is recursive k-means (k = 3) over the centered vectors, splitting until a node has fewer than 20 items (`min_node_size`) or the depth cap is hit (`auto_max_depth` bounds the leaf count at ~4 000 for very large datasets). K-means runs on cuML when a usable GPU is present, sklearn otherwise, with restart counts scaled down for large nodes.

Each node stores:

| Field | Contents |
|-------|----------|
| `ids` | The node's item IDs, sorted **most-typical-first** (descending `mu . x`) — `ids[0]` is the region's representative |
| `children` | Child node names, stored **largest-first** so breadth-first traversal reaches big unexplored regions before small ones |
| `n` | Item count |
| `mu`, `rbar` | Mean direction and resultant length — the sufficient statistics of a von Mises–Fisher component, so reading the tree at any depth gives a multiresolution mixture model of the dataset |

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

### Tutorial: how a diversity session works

What actually happens when the autopilot enters its "Explore Diversity" phase, click by click:

1. **The atlas already exists.** It was built (or restored from the pickle cache) when the dataset loaded, and every vote cast during the earlier good/bad/refine phases has already been counted into its evidence channels.
2. **The frontend asks for a sample.** `POST /api/coverage-atlas/next` with the current learned-sort scores and decision threshold in the body.
3. **The atlas walks breadth-first** to the first evidence-free node — say a 900-item region of the collection no vote has ever touched — preferring the largest such region among siblings.
4. **It probes for a surprise.** Suppose the node's median score is 0.81 against a threshold of 0.5: the model presumes the whole region is good. The atlas returns the *lowest*-scored item from the region's typical half — the most plausible hidden bad that is still representative of the region.
5. **The user votes.** The vote lands in the detector's labels *and* increments `n_neg` (or `n_pos`) in the item's leaf and all its ancestors — the region is now covered, and the next `next_sample` call moves on to the next uncovered region.
6. **The Span indicator advances.** Each labeling-status poll reads `span_info()`; when 40 consecutive breadth-first nodes carry evidence, Span turns green and the autopilot declares the collection covered.

Either outcome of step 5 helped: a flip hands the head a training example from a region it was confidently wrong about (the next retrain bends the boundary there); a non-flip certifies the region at its weakest point for one click.

### Costs

Build is the same order as the embedding-matrix work a dataset load already does — seconds on CPU for 50k items, with progress reported to the load bar. A `next_sample` call is a breadth-first walk over node records. Nothing needs a GPU.

## Domain shift

You trained a detector on dataset A and want to run it on dataset B. Should you trust it there? `vtscore.coverage.domain_shift.domain_shift_report(reference, matrix)` answers with a **split-conformal kNN** over A's own vectors (the reference):

1. A's vectors are split in half by a fixed seed: a **fit** half and a **calibration** half.
2. An item's score is the mean cosine to its 5 nearest fit vectors. Each calibration vector is scored the same way.
3. An item's p-value is its rank among the calibration scores, `(1 + #{cal ≤ s}) / (n_cal + 1)`.

For an item drawn from A, the calibration vectors and the item are exchangeable, so the p-value is **uniform by construction**: about `alpha` of A's own items fall below `alpha`, on every embedder. The report counts how many of B's items fall below `alpha = 0.05` (`frac_atypical`, roughly the shifted share). It gives a binomial z-score for the excess, the median p-value, and a `shifted` verdict that needs the excess to be both statistically clear (z > 3) and practically large (≥ 2×`alpha`). To keep the check cheap enough to run automatically, at most 4 000 of B's items are scored (sampled), and the fit and calibration halves are capped at 20 000 and 10 000 items. A reference under 40 items is refused (`ReferenceTooSmall`): the smallest attainable p-value would not reach below `alpha`.

#### Why not the atlas

Until #3348 this was the atlas's job: each node calibrated a quantile grid of leave-one-out `mu . x`, and a query's p-value was the mean over its root-to-leaf path. That ranked a query against the very points the k-means cells had been drawn around, which sit more centrally in their cells than any fresh item does, so a dataset's own held-out items read atypical: KS 0.11 from uniform on every embedder, and on `dinov3_patch` false alarms on 60% of its own held-out splits, so the route refused patch embedders. Calibrating on a held-back half fixed every per-node statistic, and the tree-free kNN rule then beat every atlas variant ([`REPORT.md`](experiments/2026-10-23-atlas-node-model-3348/REPORT.md)). The atlas's typicality *ordering* is untouched by any of this, and it is the only part `next_sample` ever read.

### Tutorial: checking for domain shift before reusing a detector

```
# Both datasets loaded. The X-Dataset-Id header names the ACTIVE dataset (B);
# the URL names the REFERENCE dataset (A, the training domain).
GET /api/datasets/registry/<dataset_A_id>/domain-shift
X-Dataset-Id: <dataset_B_id>
```

```json
{
  "reference_dataset_id": "…",
  "n_items": 4000,
  "alpha": 0.05,
  "frac_atypical": 0.31,
  "expected_atypical": 0.05,
  "z_score": 75.4,
  "median_pvalue": 0.18,
  "shifted": true
}
```

Reading this: 31 % of the sampled items of B sit farther from A's vectors than almost anything of A's own does (against the 5 % that chance produces), so the verdict is `shifted`. The detector will be *extrapolating* on about a third of B, and its scores there are unfalsified guesswork; verify by hand before trusting it. A same-domain report shows `frac_atypical` near `alpha`, a median p-value near 0.5, and `shifted: false`.

The app asks this for you. [`GET /api/find/training-domain`](api/find.md#training-domain-overlap) takes A to be the loaded dataset holding most of the detector's labelled examples and returns the same report, and the SPA calls it the moment a detector meets a dataset, warning on a `shifted` verdict before anything is run.

The endpoint refuses (HTTP 400) when the two datasets use different embedders (typicality across embedding spaces would be confident nonsense), when A is not loaded, or when A has fewer than 40 embeddings. It needs no coverage atlas, and it answers on patch embedders.

In the library tier:

```python
import numpy as np
from vtscore.coverage import domain_shift_report

report = domain_shift_report(np.stack(list(train_vectors.values())), np.stack(list(other_vectors.values())))
```

## Key Files

- `vtscore/training/mlp.py`: `build_model`, `train_model`, `build_model_from_weights`, the head sentinels
- `vtscore/training/svm_prior.py`: `fit_linear_svm_prior_head` (the shipped head's fit); `vtscore/training/svm.py`: `fit_linear_svm_head` (the plain SVM arm); `vtscore/training/logreg.py`: `fit_linear_logreg_head`
- `vtscore/coverage/atlas.py`: `CoverageAtlas`
- `vtscore/coverage/domain_shift.py`: `domain_shift_report`, `typicality_pvalues` (split-conformal kNN)
- `vtscore/state/coverage.py`: atlas build/restore/resync helpers, vote wiring
- `vtscore/training/thresholds/`: `calculate_cross_calibration_threshold`, `fold_anchored_gmm_threshold` (`anchored.py`), `calculate_safe_threshold` (`blend.py`), `calculate_gmm_threshold` (`gmm_threshold.py`), `conformal_threshold` (`conformal.py`)
- `vtscore/detectors/training.py`: `train_and_score`, `train_and_threshold`, origin-based detector training; its stages are `training_snap.py` (the embedding space), `training_scoring.py` (scoring a snapshot), `training_votes.py` (a vote's training rows) and `training_calibration.py` (the fold context, the split and the line)
- `vtscore/detectors/label_quota.py`, `centroid_head.py`, `query_head.py`: the label quota, and the Goods' centroid and typed query's heads under it
- `vtscore/training/thresholds/labels_line.py`, `spot_check.py`, `line_test.py`: the labels line, the spot check and Tune's test sample (reference: [`vtscore/docs/packages/training.md`](../vtscore/docs/packages/training.md)); the labels line draws on `class_model.py` (the labels' class model), `corpus_model.py` (the corpus fits and the cut) and `head_count.py` (the head's count line and Autopilot's cues)
- `vtscore/detectors/labeling_progress.py`: Per-step cache of the detectors the app trained, and their stability analysis
- `vtscore/embedding/loader.py`: Model initialization; `vtscore/media/torch_setup.py`: torch thread configuration
- `vtscore/eval/voting_iterations.py`: Voting simulation evaluation
- `vtscore/config/runtime.py`: `SVM_HEAD_C`, `DEFAULT_CALIBRATE_COUNT`, `TORCH_THREADS`, and the BCE-arm knobs (`TRAIN_EPOCHS`, …); `vtscore/config/models.py`: the model IDs
