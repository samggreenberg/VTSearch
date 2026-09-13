# The threshold's mixture fit is ours now, not sklearn's (issue #3585)

[#3585](https://github.com/samggreenberg/VTSearch/issues/3585) asked for one
thing before any code shipped: *"the deliverable is a **measurement**, not a
diff."* `fit_score_gmm` fits two Gaussians over one dimension and it was
sklearn's `GaussianMixture(n_components=2, random_state=42)` — full-covariance
machinery and a k-means init on a problem whose covariance is a scalar, next
door to an EM loop this repo already owns. Replacing it is easy. Establishing
that the replacement is safe is the whole task, because **it moves the fit**: a
different starting point lands EM somewhere else inside its tolerance, and
sometimes in a different basin, and that propagates through the per-fold
quantile, the combine, `np.quantile` on the final haystack and
`snap_cut_to_sample` to the green/red line a user sees.

So: a corpus of real fit inputs captured from the shipped path, every candidate
replayed against it, and the two numbers the issue named — the distribution of
`|Δthreshold|` and **the count of steps where the admitted set changes at all**
— plus the trajectory A/B that the issue's own decision rule calls for once the
answer to the second one is "not zero".

<!-- SHORT VERSION -->

<!-- BODY -->

---

## What was measured

**The corpus is real fit inputs, captured from the shipped path.** A synthetic
sweep over (n, prevalence, separation) was already run on the issue's own
thread, and it is not the gate: what decides whether a re-initialised EM lands
somewhere else is the *shape* of the sample, and the shapes that matter are the
ones the app actually produces — saturated at click 80, barely bimodal at click
5, max-pooled and right-skewed under region voting, and — the one that turned
out to matter most — a cosine sort, where the query's matches are a shoulder on
one broad mode rather than a second mode.

| | grid | what a case is |
|---|---|---|
| **fold path** | 84 cells: 3 datasets × 2 embedders × their categories × 2 seeds, 100 clicks, every 5th threshold fit captured | one call to `fit_fold_anchored_cut` — two fold haystacks, their held-out votes, and the final model's haystack |
| **sort path** | 62 queries: 4 datasets × 3 text embedders, 838 to 18,050 medias | one whole cosine/text sort's score array, as `calculate_gmm_threshold` receives it |

The three datasets span the shapes on purpose: `caltech101_m`/`siglip` is
**saturated** (the #3166 regime, where the snap is load-bearing),
`visual_genome_m` and `coco_val` are ordinary bimodal in two environments, and
`dinov3_patch`/`max_patch` is **max-pooled** — the heavy right-skewed Bad mode
region voting produces. Shipped defaults everywhere else; the contrast under
test is the fit.

**Each case is replayed through the whole production chain, not through the
fit.** A fold case runs `fit_fold_anchored_cut` → per-fold anchored EM → per-fold
quantile → combine → `np.quantile` on the final haystack → `snap_cut_to_sample`,
and the admitted set is read off the haystack the threshold is applied to. A
sort case runs `calculate_gmm_threshold` and reads the admitted set off the
sorted scores. Pairing is exact — every arm sees the identical captured input —
so there is nothing to match approximately.

### The arms

Every arm fits the same model — two Gaussians over one dimension — to the same
sample. They differ only in where EM starts and where it stops, which is what
makes this a gate and not a benchmark.

| arm | what it is |
|---|---|
| `baseline` | `GaussianMixture(n_components=2, random_state=42)` — what shipped before #3585 |
| **`native`** | a deterministic 2-means init, the anchored EM loop with no anchors, stopped where sklearn stops: an iteration improving the mean log-likelihood by less than 1e-3 |
| `native_ll1e-4`, `native_ll1e-5` | the same, converged further |
| `native_param1e-8` | the same loop stopped on the *parameters* instead — the anchored path's rule, and what this branch tried first |
| `native_iter50` | the parameter rule, capped at 50 iterations |
| `sklearn_kmeanspp` | **the control**: keep sklearn, change only `init_params` to `"k-means++"`. A three-character re-initialisation of the incumbent, and the yardstick for how much a fit *of the same estimator* moves when nothing but its starting point changes |
| `sklearn_spherical` | the issue's cheap option: keep sklearn, pass `covariance_type="spherical"` |
| `native_10k` | `native` fitted on at most 10k scores — the `_GMM_MAX_SAMPLES` lever, measured but not a candidate here |

`sklearn_kmeanspp` is the most important row in that table and it is not a
candidate. Without it the gate can only say "the new fit moves the admitted set
by X" with nothing to compare X against. With it, X has a scale: *this is what
re-initialising the estimator we already ship does.*

