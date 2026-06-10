# Coverage Atlas: domain-shift and evidence-aware verification for transferred detectors

**Status:** design / research writeup only — nothing implemented. This document is the
output of a first-principles design discussion ("invent the perfect density structure,
don't adapt the existing one"). It scopes the science; an implementation plan would be a
follow-up document.

**Relationship to shipped work:** this design layers on top of the shipped
[Find verification workflow](find-verification-workflow.md). That feature built the
verify loop (frozen `find_scores`, `verified_ids`, the marginal-positive work queue, the
Stats modal). Its Stats section explicitly documents a **false-confidence caveat**:
unverified items are flood-filled with the detector's own call, so precision "agreement"
is inflated by everything the human never looked at. The active-auditor portion of this
design (§6) is the principled replacement for that flood-fill: stratified estimates with
confidence intervals instead of adopted self-agreement.

---

## 1. Problem statement

UserA trains a detector for concept X on haystackA (the Train interface: vote on a few
items, an MLP learns to rank). UserA hands the detector to userB. UserB knows neither
what haystackA looked like nor what userA's operational definition of X was. UserB runs
the detector on haystackB (the Find interface) and wants to verify enough items to (a)
catch the detector's mistakes on his data and (b) measure its accuracy, with bounded
effort.

The question: **which items in haystackB are worth re-evaluating?** Obviously every
predicted positive. But also anything the detector is *silently unqualified* to judge:
items unlike anything in haystackA (the detector extrapolates), and items in regions
userA never labeled (the detector interpolates without supervision). Today VTSearch
gives userB no signal for either; the work queue is ordered purely by detector score.

We want equipment that, given an artifact produced during (or after) userA's training,
tells userB:

- per item: "how typical is this of what the detector was trained on?" and "how much
  labeled evidence backs the detector's call here?" — both **calibrated**, not raw
  distances;
- per region: "here are the coherent blobs of haystackB that haystackA never
  considered" — browsable units, not a per-item score he must threshold and cluster
  himself;
- per session: "verify these next" — an ordering that finds mistakes fast and yields
  honest accuracy estimates with confidence intervals;
- per detector: a small, portable, mergeable artifact carrying all of the above across
  the user boundary, where (crucially) haystackA's files may not resolve.

## 2. Design requirements

The artifact must answer four queries:

| # | Query | Output |
|---|-------|--------|
| Q1 | **Typicality** of a point x of haystackB | calibrated p-value: "what fraction of haystackA looks less typical than x?" |
| Q2 | **Decision support** at x | labeled-evidence mass of the *predicted class* near x, by scale |
| Q3 | **Blob enumeration** | maximal coherent regions of haystackB with B-mass and ~no A-mass (or no A-*evidence*) |
| Q4 | **Merge / update** | combine artifacts across sessions and datasets; absorb userB's verifications online |

Plus four non-functional requirements:

- **Portable**: small enough to ride in (or beside) the detector JSON; usable when
  haystackA's origins do **not** resolve on userB's machine.
- **Calibrated by construction**: every score is a p-value or a count, never a unitless
  distance with a magic threshold.
- **Multi-resolution**: haystacks are multi-scale blob soup (near-duplicate clumps plus
  diffuse regions); a single bandwidth is wrong everywhere.
- **Mergeable**: every stored statistic must come from the mergeable family (counts,
  moment sums, mergeable quantile sketches), or Q4 dies.

Q4 quietly dictates the implementation. It is also what makes the structure *live*:
userB's verifications stream back in as a new channel, so the recommender sharpens as he
works (§6).

## 3. Conceptual foundation: two densities, four quadrants

"Out of domain" conflates two different reference distributions:

1. **Data coverage** — p_data: where haystackA had mass at all. Outside it the detector
   *extrapolates*; its score is unfalsified guesswork.
2. **Evidence coverage** — p_evidence, split by class: where userA's votes actually
   landed. The diversity sampler tries to spread labels over p_data, but sessions end
   early, samplers skip, and users fixate; large in-domain regions routinely carry zero
   supervision. There the detector *interpolates*: better than extrapolation, but
   unverified — and userA's **definition of X was never exercised there**, which is
   precisely userB's "I don't know what their X means" problem.

Tracking both over one shared geometry yields a quadrant picture with genuinely
different verification semantics:

| | near labeled evidence | far from evidence |
|---|---|---|
| **dense in haystackA** | trust the detector | userA chose not to label here — interpolation; medium priority, item-level checks |
| **sparse in haystackA** | rare but supported | pure extrapolation — top-priority **blobs**; browse-level review |

**Why decision support is the primary score, not generic OOD distance.** What userB
actually fears is disagreement between userA's concept and the detector's output at x.
That is unknowable directly, but for a smooth ranker it is bounded by a Lipschitz
argument: if f has Lipschitz constant L on the (normalized) embedding sphere and is
consistent with a labeled example x_i of class y, then |f(x) − f(x_i)| ≤ L·d(x, x_i) —
the call at x can only be badly wrong if either x is far from all evidence of its
predicted class, or the concept itself bends faster than L (a concept conflict, which
only verification can reveal). Distance-to-evidence-of-the-predicted-class is the tight,
decision-aware quantity; distance-to-all-of-haystackA is its loose upper bound. A
predicted positive with no positive evidence within any reasonable radius is suspect
*even in a dense region of A*. This is the same intuition as the **trust score** of
Jiang et al. [11] (ratio of distance-to-nearest-other-class over
distance-to-predicted-class), enriched with scale and density information.

**Why not the model's own confidence.** The MLP's sigmoid margin is only meaningful
*inside* supported regions. Neural networks are systematically overconfident away from
training data — ReLU networks provably so in the far-field [10] — and a plain softmax
baseline [9] is the weakest OOD detector in every comparison. The margin earns a slot as
a *third* signal (boundary fragility, |s − τ| small), never as the novelty signal.

**Why not "just learn the density" with a neural density model.** Normalizing flows and
friends famously assign *higher* likelihood to out-of-distribution data than to their
training data [8]. Density for OOD purposes is better served by simple geometry over
good embeddings — the consistent empirical finding of the OOD literature, where
k-nearest-neighbor distance on normalized deep features [3] and (relative) Mahalanobis
distance [1][2] beat learned densities.

## 4. The Coverage Atlas structure

A hierarchical partition over **unit-normalized, mean-centered** embeddings, with
mergeable sufficient statistics, conformal calibration, and labeled-evidence channels at
every node. Each ingredient is classical and boring on purpose; the design's novelty
budget is spent on the *query semantics* (evidence-aware, multi-resolution, mergeable),
not on exotic estimators.

### 4.1 Geometry

- **Normalize and center.** CLAP/SigLIP/E5 embeddings live in cosine geometry, and
  contrastive multimodal embeddings concentrate in a narrow cone [19] — raw cosines are
  uniformly high and uninformative. So: subtract haystackA's mean direction, renormalize
  to the unit sphere, and do everything in that frame. (The centering vector is part of
  the artifact's geometry payload.) Optionally whiten with a low-rank estimate of A's
  covariance — the same correction that turns Mahalanobis into *relative* Mahalanobis
  and fixes its near-OOD failures [2].
- **Partition tree.** Recursive spherical k-means (k ≈ 3, min leaf ≈ 20–50, depth ≈
  10–12) — deliberately the same *shape* of structure as the existing diversity tree,
  but the resemblance ends there: this tree **keeps** what the diversity tree throws
  away. The split rule is not load-bearing; PCA splits or 2-means work too.
- **Small ensemble.** Hard partitions have boundary artifacts: a point near a split
  plane gets a noisy node assignment. Standard fix: build 3–5 trees with different
  seeds and average the per-tree p-values — the density-forest / Mondrian-forest move
  [7][16][21]. Mondrian trees are worth a look specifically because their projectivity
  gives clean online-insertion semantics, but plain re-seeded k-means trees suffice.

### 4.2 Per-node record (everything mergeable)

| Field | Contents | Why |
|---|---|---|
| counts | `n_data`, `n_pos`, `n_neg`, `n_viewed` | the multi-channel density: data mass, evidence mass per class, and "userA looked but didn't vote" (VTSearch knows this from the seen/vote history; weak coverage evidence no off-the-shelf method has) |
| moments | resultant vector Σx (quantized), yielding mean direction μ̂ = Σx/‖Σx‖ and resultant length R̄ = ‖Σx‖/n | on the sphere (Σx, n) is the **complete sufficient statistic** for a von Mises–Fisher component [13]; concentration κ̂ ≈ R̄(d − R̄²)/(1 − R̄²) falls out for free. Reading the tree at any depth gives a vMF mixture at that scale — a multiresolution mixture model in two floats per dimension per node |
| calibration | quantile summary (deciles, or a KLL sketch [14] / t-digest [15] if exactness matters) of the node's own points' typicality scores t(x) = μ̂ᵀx | bakes calibration into the structure: a query returns a p-value natively, no held-out split, no bandwidth knob |
| spine | 1–3 medoid **origins** per leaf (origin dicts, not vectors) | rehydration seed for embedder drift (§7.3) and human-browsable blob summaries |

Counts and moment sums merge exactly (vector addition; pairwise-merge variance algebra
is classical [17]); KLL/t-digest merge by design; deciles merge approximately by
weighted interpolation. Nothing in the record breaks Q4.

### 4.3 Queries

- **Q1 Typicality.** Route x down each tree (cosine-nearest child); at the deepest node
  with n_data ≥ n_min (≈ 30; back off to the parent otherwise — sparse regions
  terminate shallow, which *is* the adaptive bandwidth), read the one-sided p-value of
  t(x) = μ̂ᵀx against the stored quantiles. Average across the ensemble.
  *Calibration honesty:* using a node's own points to calibrate its quantiles is
  mildly optimistic (those points shaped μ̂). The rigorous variant is a split build —
  fit geometry on half of A, fill quantiles with the other half — the standard
  inductive-conformal construction [12]. With n ≥ 30 per node the optimism is small;
  the split build is the default for the portable artifact since it costs nothing.
- **Q2 Decision support.** Along the root-to-leaf path of x, accumulate depth-weighted
  evidence rates for the predicted class:
  E_y(x) = Σ_levels w_level · n_y(node) / (n_data(node) + α), with w increasing with
  depth (evidence at finer scale is worth more). Complement it with the direct kNN
  form — distance from x to the k-th nearest labeled item of class y, conformally
  calibrated against leave-one-out same-class distances *within* the labelset (exact and
  cheap at labelset sizes of dozens). The two agree in the bulk; the kNN form is sharper
  when the labelset is tiny, the path form is cheaper at scale and falls out of the
  artifact without needing the labeled vectors themselves.
- **Q3 Blob enumeration.** Pour haystackB into the tree as counts and run a scan
  statistic over the hierarchy (§5).
- **Q4 Merge / update.** Counts add; resultants add; sketches merge; userB's
  verifications append as new channels (§6, §7.5).

### 4.4 Build cost

Recursive spherical k-means is O(N · k · d · iters · depth) — the same order as the
existing diversity-tree build that already runs at dataset load, times a small ensemble
constant. For N = 100k, d ≈ 10³: seconds on CPU. Queries are O(depth · k · d) per item.
Nothing here needs a GPU.

---

## 5. Thread 2 in detail: the blob scan

*(Ordered before thread 1's scoring because the scan defines the regions the scores are
aggregated over.)*

### 5.1 Null model and per-node statistic

Pool haystackA's mass (anchor-weighted if A is represented by its atlas rather than raw
points) and haystackB's items; label each point "is-B". Under the null "B is drawn from
A's distribution", the B-fraction in every node matches the global B-fraction. For node
v with A-mass a_v (of N_A total) and B-count b_v (of N_B), the Kulldorff scan likelihood
ratio [4] in its two-population Bernoulli form, with MLEs p̂_in = b_v/(a_v+b_v),
p̂_out = (N_B−b_v)/(N_A+N_B−a_v−b_v), p̂ = N_B/(N_A+N_B):

```
LLR_v = b_v·log(p̂_in/p̂) + a_v·log((1−p̂_in)/(1−p̂))
      + (N_B−b_v)·log(p̂_out/p̂) + (N_A+N_B−a_v−b_v)·log((1−p̂_out)/(1−p̂))
```

evaluated one-sided (only when p̂_in > p̂ — we hunt B-excess, not B-deficit). This is a
multi-resolution two-sample test in the spirit of classifier two-sample tests [22] and
MMD [18], but with *localization built in*: the rejected unit is a region, not a global
verdict.

Two flavors of excess, reported separately because they answer different questions:

- **Data-novelty nodes**: b_v ≫ expected from a_v — haystackB has mass where haystackA
  had (relatively) none. These are the "domain shift" blobs.
- **Evidence-vacuum nodes**: b_v ≥ m (say 10) and n_pos + n_neg = 0 in v and its
  ancestors below some scale — regardless of a_v. These are "userA never exercised the
  definition here" blobs. No significance machinery needed; this is a coverage query
  (a count threshold), not a test.

### 5.2 Calibration of the scan: Monte Carlo on the tree

Nodes are nested and their statistics are strongly dependent; per-node χ² thresholds are
meaningless. The standard scan-statistics answer [4][5] is Monte Carlo calibration of
the **max** statistic: for each replicate, drop N_B simulated points into the tree by a
multinomial draw over A's leaf distribution, aggregate counts up the tree, record
max_v LLR_v. The 95th percentile of the max-distribution is the family-wise threshold.
Each replicate is a multinomial draw plus a tree aggregation — O(#nodes) ≈ microseconds
to milliseconds — so 999 replicates are free. When A is represented by weighted anchors,
resample the anchors with their weights inside the replicate so the subsampling variance
is included.

An alternative with more reporting resolution is hierarchical FDR testing: test
top-down, descend only into rejected nodes, control FDR within each family [6][20]. It
naturally yields the "maximal anomalous subtree" report and spends test budget only
where discrepancy lives. Recommended shape: MC-calibrated max-scan as the headline
yes/no gate, hierarchical descent for the report.

*Prior-art note:* Bayesian nonparametrics arrived at the same shape from the other
direction — coupled optional Pólya trees [36] and probabilistic multi-resolution
scanning [37] both test *and localize* two-sample differences over recursive partition
trees. The atlas scan is the frequentist, scan-statistic rendering of that idea, with
the partition reused from the density structure instead of integrated over.

**Honest caveat — the null is usually false.** Under genuine domain shift, *hundreds* of
nodes reject; a strict test answers a question nobody asked. The practical procedure:
rank nodes by LLR, group into maximal subtrees, and use the MC threshold only as the
noise floor that cuts the tail of the ranking. The report is "here are the top blobs, in
order, with their sizes", not "the null is rejected".

### 5.3 Maximal-subtree reporting rule

Descend from the root. At a flagged node v, examine its children: if a single child
carries ≥ θ (≈ 80%) of v's excess (excess(v) = b_v − e_v, e_v = N_B·a_v/N_A), descend
into it; otherwise report v itself — the excess is *homogeneous* at v's scale and v is
the natural unit. Each reported blob is summarized for the UI by its medoid exemplars
(from haystackB — always resolvable, they're userB's own files) and its quadrant
classification (data-novel vs evidence-vacuum vs both).

### 5.4 Power: minimum detectable blob

For an evidence-vacuum or near-empty node (smoothed expectation e_v = α ≈ 0.5), the
one-sided LLR is ≈ b·log(b/e) − (b − e). Significance against the max over ~10⁴ nodes
needs roughly LLR ≳ log(#nodes/0.05) ≈ 12. With e = 0.5: b = 8 gives 8·log(16) ≈ 22 —
comfortably detectable. So **novel blobs of ~8–15 items are findable** even in a
100k-item haystackB; blob detection is not the bottleneck, review bandwidth is.

The weak spot is the opposite regime: **diffuse shift**, where every node is slightly
enriched and no subtree concentrates the excess. Scan statistics are blind there by
design. That is what the cheap *global* alarms are for, run alongside the scan:

- **Score-distribution drift**: a two-sample test (KS) on the detector's score histogram
  over B vs the stored histogram over A — black-box shift detection, i.e. two-sample
  testing on the model's own outputs (the BBSE/BBSD line [23][24]), which the empirical
  shoot-out of Rabanser et al. [24] found to be among the most effective practical shift
  detectors. Costs one histogram in the artifact.
- **Domain discriminator (C2ST)** [22]: train a small MLP (machinery VTSearch already
  has) to classify A-anchors vs B-items. Its held-out AUC is a one-number shift alarm
  (0.5 = same domain); its confident-B outputs localize diffuse shift softly; and its
  output odds are the density-ratio importance weights [25] that §6's estimators can
  reuse.

## 6. Threads 1 + 3 in detail: scoring and the active auditor

### 6.1 The per-item scores

For each item x in haystackB, the atlas yields:

- **T(x)** — typicality p-value (Q1).
- **D(x)** — decision support for the predicted class: E_ŷ(x) (Q2), plus the trust-style
  ratio TS(x) = r_¬ŷ(x)/r_ŷ(x) (distance to k-th nearest evidence of the other class
  over same for the predicted class) [11].
- **M(x)** — boundary margin |s(x) − τ| over the frozen `find_scores` (already shipped).
- **quadrant(x)** — from thresholding T (conformal p < 0.05) and D (zero evidence at the
  finest two scales).

Pitfalls handled at this layer:

- **Class imbalance in evidence.** Labelsets are typically few positives, many
  negatives; raw E_pos is systematically sparse. Calibrate each class against its own
  leave-one-out distance distribution, never against a shared scale.
- **Near-duplicate evidence inflation.** Five labeled near-identical frames are one
  piece of evidence, not five. Count *distinct leaves* containing evidence, or dedup
  evidence by a tight cosine radius before counting.
- **Cone effect.** Without mean-centering, every cosine is ≈ 0.8 and every p-value is
  garbage [19]. Centering is not optional.

### 6.2 Should decision support dominate typicality? Tiers, then a unified score

They stay **separate tiers** at the UI level because they trigger different verification
*actions* — and a single scalar would erase that:

| Tier | Trigger | Action |
|---|---|---|
| 1. Predicted positives | s ≥ τ | item-level verify (already the shipped work queue; keep marginal-first order within the tier) |
| 2. Novel blobs | scan report (§5.3) | **browse-level** review in VTSBrowse — a whole region at a glance, colored by novelty |
| 3. Unsupported calls | D low, T high | item-level verify — detector interpolating without evidence |
| 4. Boundary band | M small | item-level verify — fragile decisions |
| 5. Audit stream | random, stratified | the unbiased estimation backbone (§6.3); interleaved ~1-in-k |

When a single ranking is forced (one work queue), unify as **expected flip value**:
priority(x) = π(x) · c(ŷ(x)), where π(x) = P(verification contradicts the detector) and
c encodes asymmetric error costs (a missed positive usually costs more than a verified
true positive's wasted click). π is not hand-designed: it is a tiny online model
(logistic regression or per-feature isotonic stacking) over features (M, T, D, TS,
node-level disagreement-so-far), **trained on userB's own verifications as they
accumulate**. Cold-start π is a fixed monotone blend dominated by D (per §3's argument);
after a few dozen verifications the learned π takes over. This converts the philosophical
"which signal dominates?" question into an empirical one the session answers for itself.

### 6.3 Estimation mode: honest accuracy numbers with budgets

UserB's analysis goal is precision/recall/accuracy of the detector *on haystackB*, where
truth = his verifications. The shipped Stats modal flood-fills unverified items with the
detector's own call — the documented false-confidence caveat. The replacement:

- **Design**: stratified sampling, strata = quadrant tier × predicted label (≈ 8
  strata; finer per-node post-stratification at report time). Per-stratum error rates
  with Beta posteriors; global metrics by post-stratified combination; intervals by
  Wilson per stratum and delta method (or a 5-line bootstrap) for composites.
- **Allocation**: Neyman (n_h ∝ N_h·σ̂_h) with plug-in σ̂; or, anytime and adaptive,
  Thompson sampling over per-stratum Beta posteriors — which approximates Neyman while
  staying valid under continuous monitoring. This is the practical face of optimal
  active risk estimation [26][27].
- **Recall is the hard one.** False negatives hide among the huge predicted-negative
  stratum. Uniform sampling there is hopeless; sample predicted negatives with proposal
  q(x) ∝ s(x)^β · (1 + γ·novelty(x)) mixed with an ε-uniform floor (bounded weights,
  full support), and reweight by Horvitz–Thompson. Score-proportional negative sampling
  is the classic web-scale evaluation trick [27]; the novelty boost is this design's
  addition (extrapolating regions have a higher FN prior).
- **Adaptive-sampling bias.** If acquisition adapts item-by-item (as the bug-hunt mode
  does), naive averages are biased; apply the LURE-style weighting that removes it
  [28], or keep the estimation stream a separately-randomized 1-in-k interleave (the
  simpler engineering answer — recommended first).
- **Zero-label priors, for context.** A family of methods predicts target-domain
  accuracy from unlabeled data alone — average thresholded confidence [38],
  difference of confidences [39], agreement-on-the-line [40], and
  disagreement-discrepancy bounds [41]. Worth surfacing as a *pre-verification prior*
  in the Stats modal, but they lean on calibration and accuracy-on-the-line phenomena
  that a small MLP under concept transfer cannot honestly claim; verification-based
  estimates remain the ground truth here.

### 6.4 Bug-hunt mode and the disagreement taxonomy

Greedy mode maximizes flips found per click: rank by π(x), with per-node pick caps (or
node-stratified Thompson sampling) so one bad blob doesn't monopolize the queue — the
explore/exploit structure of Bayesian active search [29].

Every verified flip is classified by the quadrant it occurred in:

- **Coverage gap** (D low or T low): the detector had no evidence there. Fixable with
  data: userB's verifications in that region become Train votes for a forked detector —
  VTSearch's labelset-merge machinery already supports exactly this.
- **Concept conflict** (D high *and* T high — the detector was well-supported and still
  wrong by userB's lights): userA's X ≠ userB's X *here*. No amount of in-domain data
  fixes this; it needs relabeling, and the *map of where definitions diverge* is itself
  a first-class deliverable ("you and the detector's author disagree about marching
  bands; here are the regions"). This localization of concept drift vs covariate shift
  — usually a global taxonomy [30] — into embedding-space regions is, with the
  evidence channels, the most novel piece of the design; the closest published
  relative is drift-aware classification in security pipelines [31].

**Stopping rules.** Estimation mode: stop at target CI half-width on the headline
metric. Bug-hunt mode: track the discovery curve (flips per verification); stop when the
marginal flip rate drops below the per-click value, with a capture-recapture /
Good–Turing-style estimate of remaining undiscovered error mass [32] as the formal
option — the rate of *singleton* discoveries (errors whose region produced exactly one
flip so far) estimates the unexplored remainder.

### 6.5 Concrete integration with the shipped Find workflow

- The left work queue gains tier headers (§6.2) in place of pure score order; tier 1
  keeps the shipped marginal-first auto-advance within itself.
- Blob review rides the existing VTSBrowse canvas: color hex tiles by median T(x) (or
  scan-LLR of the tile's items), one click sends a blob to the verify flow. The shipped
  browse-canvas Verified Good/Bad buttons (find-verification Phase 4) already do bulk
  verification of a canvas selection — blob review is that feature pointed at
  scan-reported regions instead of hand-drawn ones.
- The Stats modal gains: a **domain-overlap chip** ("31% of this dataset is outside the
  training domain at the 95% level; 12% sits in evidence vacuums"), and
  coverage-corrected precision/recall **with intervals** from §6.3 alongside (or
  replacing) the flood-filled numbers.
- The detector's score histogram on its training haystack (a few dozen floats) joins the
  artifact so the score-drift alarm [23][24] can fire before any verification happens.

## 7. Thread 4 in detail: the portable artifact

### 7.1 Schema sketch

```jsonc
{
  "format": "coverage-atlas/1",
  "embedder": { "name": "siglip", "dim": 1152, "weights_id": "…", "centering": "<b64 f16[d]>" },
  "build": { "split": "spherical-kmeans", "k": 3, "min_leaf": 50, "max_depth": 12,
              "n_trees": 3, "seeds": [7, 11, 13], "calibration": "split-half" },
  "totals": { "n_data": 100000, "n_pos": 41, "n_neg": 73, "n_viewed": 350 },
  "score_hist": { "edges": [...], "counts": [...] },          // for the BBSD alarm
  "trees": [ { "nodes": [
      { "id": 0, "parent": null, "children": [1, 2, 3],
        "n": 100000, "n_pos": 41, "n_neg": 73, "n_viewed": 350,
        "mu_q8": "<b64 i8[d'] + scale>", "rbar": 0.42,
        "t_deciles": [/* 11 f16 */],
        "exemplars": [ { /* origin dict, leaf nodes only */ } ] }
  ] } ],
  "privacy_level": 2
}
```

### 7.2 Size analysis and the knobs

Per tree with min_leaf = 50 on N = 100k: ~2k leaves, ~3k nodes. Per node, the dominant
cost is μ: at full d = 1152 in int8, ~1.2 KB/node → ~3.5 MB/tree, ~10 MB for three
trees (×1.33 for base64). Acceptable as a sidecar but heavier than a detector JSON.
Knobs, in order of preference:

1. **Store μ in a PCA-compressed frame**: one d×128 projection (~0.6 MB once), then 128
   int8 per node → ~0.15 MB/tree of moments; total artifact ≈ 1–2 MB. Routing and
   typicality cosines in PCA-128 approximate full-d well in practice (this is the same
   compression regime as product-quantized ANN indexes); validate empirically.
2. Store μ only for nodes with n ≥ some floor; route by exemplar medoids below it.
3. Drop to a single tree (lose ensemble smoothing) — last resort.

Deciles (11 × f16) and counts are noise. The exemplar spine is origin dicts — small.

### 7.3 Embedder drift and the no-persisted-vectors rule

The repo rule "no persisted vectors" exists to make embedder drift impossible by
construction: origins are canonical, geometry is rederived. The atlas *must* persist
geometry (moments are vectors in embedding space), so it must answer the drift problem
explicitly rather than by construction:

- The artifact is **stamped** with embedder identity (name, weights id, dim,
  normalization). On mismatch, atlas queries refuse.
- On refusal, if the **exemplar spine resolves** (userA's origin files are reachable),
  rebuild: re-embed ~(leaves × m) exemplars under the active embedder and refit the
  atlas from the spine (counts carry over as weights). Cost: one moderate import-sized
  embedding pass, once.
- If the spine does **not** resolve, degrade to the signals that never break: the
  labelset (which always re-embeds, because labeled origins must resolve for the
  detector to retrain at all) gives D(x) and the kNN evidence signals; only T(x) and the
  blob scan against A are lost.

**Why the rule's strict form cannot hold here, argued honestly:** the rule's premise —
"rederive origin → embedding on demand" — assumes origins resolve. The cross-user
handoff this feature exists for is *exactly* where they don't: userB has the detector
JSON, not userA's filesystem. The atlas is the first artifact designed to travel where
origins can't. Proposed amendment to the rule: *derived geometric summaries may be
persisted iff (a) stamped with embedder identity and refused on mismatch, and (b)
accompanied by an origins spine sufficient to rebuild them where origins resolve.* Item
embeddings and MLP weights stay banned; nothing about this amendment weakens the
original rule's protection against silent drift.

### 7.4 Privacy: what the artifact leaks about haystackA

Handing userB a density model of haystackA is handing him information about its
contents. Embedding inversion is real — text embeddings can be inverted nearly verbatim
[33], image embeddings to recognizable reconstructions — so quantized mean directions of
small leaves are not anonymous. The artifact therefore carries an explicit
**privacy level**, set at export:

- **L0 — counts only**: no geometry. Usable only for merging statistics; no Q1–Q3.
- **L1 — moments, coarse nodes only** (n ≥ 200, say): blob-scale geometry; leaf-level
  typicality lost; inversion yields only broad gist.
- **L2 — full moments + deciles** (default for trusted handoff): full Q1–Q3.
- **L3 — + exemplar spine**: adds origin *references* (paths/URLs, not media). Note the
  detector labelset already leaks exactly this for labeled items, so L3's marginal leak
  is the unlabeled exemplars.

### 7.5 Merge semantics

- **Same-lineage updates** (userB's verifications, more A-data under the same tree):
  channel counts and resultants add in place; sketches merge. Exact.
- **Different trees** (two sessions, two datasets): trees don't align node-for-node.
  Strategy: **forest-of-atlases** — keep both, query = average of p-values, scan = run
  per atlas and union reports. Trivially correct, size grows linearly. **Compaction**
  when the forest gets fat: treat each atlas's leaves as weighted pseudo-points (μ, n,
  channels) and rebuild one tree over the pseudo-points; counts exact, quantiles
  approximated by weighted decile interpolation. This is BIRCH's CF-tree trick [16] —
  the structure was designed in 1996 to be built from summaries of summaries.

## 8. Alternatives considered and rejected

| Candidate | Pros | Why rejected (here) |
|---|---|---|
| **Gaussian KDE** in embedding space | the textbook answer | hopeless at d ≈ 10³: sample complexity exponential in d, no bandwidth answer, distance concentration flattens the estimate. Even PCA-50 KDE is marginal. kNN distance *is* adaptive-bandwidth KDE up to a monotone transform, minus the failure modes |
| **Plain global kNN to A** [3] | strong OOD baseline; nearly what §4 reduces to pointwise | no blobs (threshold-then-cluster afterthought), no evidence channels, no calibration story, no merge, and requires shipping raw A vectors (size + privacy). Kept as the *pointwise* gold standard the atlas should match |
| **Single GMM / Mahalanobis** [1][2] | smooth, cheap, parametric | unimodality (or fixed K) is the wrong prior for blob-soup haystacks. The atlas read at any depth *is* a GMM; the hierarchy is what a flat GMM gives up. Relative-Mahalanobis whitening absorbed as the optional preprocessing step |
| **Normalizing flows / neural density** | best density in principle | systematically wrong for OOD [8]; opaque; no blobs; no merge; persists a trained network |
| **LSH count sketches (RACE)** [34] | KDE in kilobytes; mergeable; streaming; leaks no geometry | fixed implicit bandwidth; no hierarchy; no blob enumeration; no evidence channels. Its privacy framing is stolen for §7.4 |
| **KDE coresets** [35] | provable ε-approximation of the kernel density with few points | single-scale, point-based; calibration/blobs/channels still need separate machinery. Demoted to an ingredient: a principled selector for the exemplar spine |
| **Density estimation trees / forests** [21][7] | the closest published relative — partition trees as density estimators | adopted in spirit; what they lack is everything problem-specific: evidence channels, conformal node calibration, scan-based two-sample reporting, mergeable portability |
| **Topological methods (Mapper, persistence)** | capture shape | uncalibrated, parameter-fragile, expensive, no p-values, no counts. Wrong tool |

## 9. Feasibility summary

| Operation | Cost | Notes |
|---|---|---|
| Atlas build (train side) | O(N·k·d·iters·depth) × trees ≈ seconds CPU at N=100k | same order as the existing diversity-tree build that already runs at load |
| Artifact size | ~1–2 MB (PCA-compressed moments) to ~10 MB (full-d) | §7.2 knobs |
| Per-item query (T, D, quadrant) | O(depth·k·d) ≈ µs | batch over haystackB ≈ ms–s |
| Blob scan + 999 MC replicates | O(N_B·depth·k·d) + O(replicates·nodes) | well under a second |
| Discriminator / flip model | tiny MLP / logistic on ≤ thousands of points | existing training machinery |
| Active-auditor bookkeeping | Beta posteriors + counts | trivial |

No GPU anywhere; nothing exceeds what a dataset load already costs.

## 10. Phasing (sketch, not a commitment)

1. **v0 — zero infrastructure**: labelset-kNN evidence signals (D, TS) + conformal
   calibration via leave-one-out labelset distances. Needs nothing persisted that isn't
   already in the detector JSON; works today, cross-user, by construction.
2. **v1 — in-session atlas**: build the atlas at Find time over haystackA when both
   haystacks are local (same user, two datasets) — no artifact yet; tiered work queue +
   blob scan + VTSBrowse coloring.
3. **v2 — portable artifact**: schema of §7, export at Train time, the rule amendment of
   §7.3, Stats-modal domain chip.
4. **v3 — active auditor**: flip model, stratified estimation replacing the flood-fill
   Stats numbers, disagreement taxonomy and the concept-conflict map.

## 11. Open questions

- Final blend for cold-start π(x) (before userB's verifications can train it): how heavy
  should D weigh against T and M? Needs an empirical pass on a synthetic shift bench
  (e.g., train on GTZAN-half, find on the other half plus an injected foreign genre).
- Is the `n_viewed` channel signal or noise? "Saw it and didn't vote" is ambiguous
  between "irrelevant", "ambiguous", and "fatigued".
- PCA-128 compression of μ: how much does typicality ranking degrade at leaf scale?
  (Cheap experiment: Spearman correlation of full-d vs compressed p-values.)
- Ensemble size: do 3 trees suffice to kill boundary artifacts at the p < 0.05 flag
  level, or does the flag rate need 5?
- Should the evidence-vacuum threshold scale with labelset size? (A 30-label detector
  leaves most of any haystack "vacuum" by raw count.)
- UI question for blob review: scan-reported subtrees don't map 1:1 to VTSBrowse hex
  tiles (different projections); is tile-coloring by per-item T(x) good enough, or do
  blobs need their own overlay?

## 12. References

1. K. Lee, K. Lee, H. Lee, J. Shin. *A Simple Unified Framework for Detecting
   Out-of-Distribution Samples and Adversarial Attacks.* NeurIPS 2018.
2. J. Ren, S. Fort, J. Liu, A. G. Roy, S. Padhy, B. Lakshminarayanan. *A Simple Fix to
   Mahalanobis Distance for Improving Near-OOD Detection.* arXiv:2106.09022 (ICML 2021
   UDL workshop).
3. Y. Sun, Y. Ming, X. Zhu, Y. Li. *Out-of-Distribution Detection with Deep Nearest
   Neighbors.* ICML 2022.
4. M. Kulldorff. *A Spatial Scan Statistic.* Communications in Statistics — Theory and
   Methods, 26(6), 1997.
5. D. B. Neill. *Fast Subset Scan for Spatial Pattern Detection.* Journal of the Royal
   Statistical Society B, 74(2), 2012.
6. D. Yekutieli. *Hierarchical False Discovery Rate–Controlling Methodology.* JASA
   103(481), 2008.
7. B. Lakshminarayanan, D. M. Roy, Y. W. Teh. *Mondrian Forests: Efficient Online
   Random Forests.* NeurIPS 2014.
8. E. Nalisnick, A. Matsukawa, Y. W. Teh, D. Görür, B. Lakshminarayanan. *Do Deep
   Generative Models Know What They Don't Know?* ICLR 2019.
9. D. Hendrycks, K. Gimpel. *A Baseline for Detecting Misclassified and
   Out-of-Distribution Examples in Neural Networks.* ICLR 2017.
10. M. Hein, M. Andriushchenko, J. Bitterwolf. *Why ReLU Networks Yield High-Confidence
    Predictions Far Away from the Training Data and How to Mitigate the Problem.* CVPR
    2019.
11. H. Jiang, B. Kim, M. Guan, M. Gupta. *To Trust or Not to Trust a Classifier.*
    NeurIPS 2018.
12. V. Vovk, A. Gammerman, G. Shafer. *Algorithmic Learning in a Random World.*
    Springer, 2005. See also A. N. Angelopoulos, S. Bates, *A Gentle Introduction to
    Conformal Prediction and Distribution-Free Uncertainty Quantification*,
    arXiv:2107.07511, 2021.
13. A. Banerjee, I. S. Dhillon, J. Ghosh, S. Sra. *Clustering on the Unit Hypersphere
    Using von Mises–Fisher Distributions.* JMLR 6, 2005.
14. Z. Karnin, K. Lang, E. Liberty. *Optimal Quantile Approximation in Streams.* FOCS
    2016. (KLL sketch.)
15. T. Dunning, O. Ertl. *Computing Extremely Accurate Quantiles Using t-Digests.*
    arXiv:1902.04023, 2019.
16. T. Zhang, R. Ramakrishnan, M. Livny. *BIRCH: An Efficient Data Clustering Method
    for Very Large Databases.* SIGMOD 1996.
17. T. F. Chan, G. H. Golub, R. J. LeVeque. *Algorithms for Computing the Sample
    Variance: Analysis and Recommendations.* The American Statistician 37(3), 1983.
18. A. Gretton, K. Borgwardt, M. Rasch, B. Schölkopf, A. Smola. *A Kernel Two-Sample
    Test.* JMLR 13, 2012.
19. W. Liang, Y. Zhang, Y. Kwon, S. Yeung, J. Zou. *Mind the Gap: Understanding the
    Modality Gap in Multi-modal Contrastive Representation Learning.* NeurIPS 2022.
20. M. Bogomolov, C. B. Peterson, Y. Benjamini, C. Sabatti. *Hypotheses on a Tree: New
    Error Rates and Controlling Strategies.* Biometrika 108(3), 2021.
21. P. Ram, A. G. Gray. *Density Estimation Trees.* KDD 2011. See also A. Criminisi,
    J. Shotton (eds.), *Decision Forests for Computer Vision and Medical Image
    Analysis*, Springer 2013, ch. on density forests.
22. D. Lopez-Paz, M. Oquab. *Revisiting Classifier Two-Sample Tests.* ICLR 2017.
23. Z. C. Lipton, Y.-X. Wang, A. Smola. *Detecting and Correcting for Label Shift with
    Black Box Predictors.* ICML 2018.
24. S. Rabanser, S. Günnemann, Z. C. Lipton. *Failing Loudly: An Empirical Study of
    Methods for Detecting Dataset Shift.* NeurIPS 2019.
25. M. Sugiyama, T. Suzuki, T. Kanamori. *Density Ratio Estimation in Machine
    Learning.* Cambridge University Press, 2012.
26. C. Sawade, N. Landwehr, S. Bickel, T. Scheffer. *Active Risk Estimation.* ICML
    2010.
27. P. N. Bennett, V. R. Carvalho. *Online Stratified Sampling: Evaluating Classifiers
    at Web-Scale.* CIKM 2010.
28. S. Farquhar, Y. Gal, T. Rainforth. *On Statistical Bias in Active Learning: How and
    When to Fix It.* ICLR 2021. See also J. Kossen, S. Farquhar, Y. Gal, T. Rainforth,
    *Active Testing: Sample-Efficient Model Evaluation*, ICML 2021.
29. R. Garnett, Y. Krishnamurthy, X. Xiong, J. Schneider, R. Mann. *Bayesian Optimal
    Active Search and Surveying.* ICML 2012.
30. J. Gama, I. Žliobaitė, A. Bifet, M. Pechenizkiy, A. Bouchachia. *A Survey on
    Concept Drift Adaptation.* ACM Computing Surveys 46(4), 2014.
31. L. Yang et al. *CADE: Detecting and Explaining Concept Drift Samples for Security
    Applications.* USENIX Security 2021.
32. I. J. Good. *The Population Frequencies of Species and the Estimation of Population
    Parameters.* Biometrika 40, 1953.
33. J. X. Morris, V. Kuleshov, V. Shmatikov, A. M. Rush. *Text Embeddings Reveal
    (Almost) As Much As Text.* EMNLP 2023.
34. B. Coleman, A. Shrivastava. *Sub-linear RACE Sketches for Approximate Kernel
    Density Estimation on Streaming Data.* WWW 2020.
35. J. M. Phillips, W. M. Tai. *Near-Optimal Coresets of Kernel Density Estimates.*
    Discrete & Computational Geometry 63, 2020 (SoCG 2018).
36. L. Ma, W. H. Wong. *Coupling Optional Pólya Trees and the Two Sample Problem.* JASA
    106(496), 2011.
37. J. Soriano, L. Ma. *Probabilistic Multi-Resolution Scanning for Two-Sample
    Differences.* JRSS-B 79(2), 2017.
38. S. Garg, S. Balakrishnan, Z. C. Lipton, B. Neyshabur, H. Sedghi. *Leveraging
    Unlabeled Data to Predict Out-of-Distribution Performance.* ICLR 2022. (ATC.)
39. D. Guillory, V. Shankar, S. Ebrahimi, T. Darrell, L. Schmidt. *Predicting with
    Confidence on Unseen Distributions.* ICCV 2021. (Difference of confidences.)
40. C. Baek, Y. Jiang, A. Raghunathan, J. Z. Kolter. *Agreement-on-the-Line: Predicting
    the Performance of Neural Networks under Distribution Shift.* NeurIPS 2022.
41. S. Garg, S. Balakrishnan, J. Z. Kolter. *(Almost) Provable Error Bounds Under
    Distribution Shift via Disagreement Discrepancy.* NeurIPS 2023.
