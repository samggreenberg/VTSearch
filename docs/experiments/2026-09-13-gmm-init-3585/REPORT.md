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
