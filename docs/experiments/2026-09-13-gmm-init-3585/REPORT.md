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

