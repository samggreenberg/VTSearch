"""Every candidate line for a typed-query sort (#3826), as ``scores -> cut``.

A rule takes the whole score array of one text sort and returns a threshold;
the admitted set is ``scores >= cut``, exactly as the app paints it.  Nothing
here reads a label: labels are for scoring a rule, never for drawing one.

Four families, in the order the issue lists its options:

* **The mixture, as shipped and stabilised.**  ``gmm_shipped`` is
  :func:`calculate_gmm_threshold` itself.  ``gmm_converged`` is the same EM run
  until the likelihood stops moving (so *where the optimiser stopped* is no
  longer a parameter), and ``gmm_multistart`` also removes *where it started*,
  keeping the best of several deterministic starts.  ``gmm_priorfree`` reads
  #2836's rate-optimal crossing off the shipped fit instead of the midpoint.
* **Rank / quantile.**  ``quantile{q}`` admits a fixed fraction of the
  haystack: identifiable by construction, blind to how many matches exist.
* **Unimodal-tail models.**  A text sort is one broad mode with the query's
  matches as a right shoulder, so model the *bulk* and admit what it cannot
  explain.  The bulk is located at the median and scaled by the **left**
  half-MAD - the side the matches do not reach - so positives cannot inflate it.
  ``tail_z{k}`` cuts ``k`` of those scales above the median; ``tail_fdr{a}``
  cuts where the Gaussian bulk would explain at most a fraction ``a`` of what
  is admitted (a Benjamini-Hochberg-style empirical-null cut).
* **Prior-free / exact.**  ``otsu`` is the exact global 2-means split (the
  hard-assignment, equal-variance limit of the mixture), solved by a scan over
  the sorted sample - the mixture's question with no optimiser in it.

``gmm_guarded_*`` is the issue's option 3: keep the shipped mixture when its two
components are separated (Ashman's D >= 2, the standard bimodality criterion
for a two-Gaussian mixture) and fall back to a tail rule when they are not.

Every rule that runs an optimiser also has a *perturbed* form (``perturb``),
used by the gate to ask what re-initialising or re-tolerancing it does - the
issue's own experiment, generalised.  A closed-form rule has none: its answer
cannot depend on an optimiser it does not have.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from vtscore.training.thresholds import gmm as G
from vtscore.training.thresholds.gmm import GmmFit1D

#: Tolerance and budget for the "run it until it stops moving" arms.  1e-10 in
#: mean log-likelihood is seven orders below the shipped 1e-3; the cap exists so
#: a flat ridge cannot hang a job, and whether it bound is recorded per fit.
CONVERGED_TOL = 1e-10
CONVERGED_MAX_ITER = 5000

#: The deterministic start set of ``gmm_multistart``: the shipped 2-means start,
#: plus splits that put the top ``1 - q`` of the sample in the high component.
#: The high quantiles are the starts a *shoulder* needs - a 2-means start on a
#: unimodal sample splits the mode near its middle, and the flat ridge between
#: that split and a small right component is exactly where EM stalls.
MULTISTART_QUANTILES = (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)

#: Ashman's D above which the two fitted components count as separated.
ASHMAN_D_SEPARATED = 2.0


# --------------------------------------------------------------------------- EM


def _ordered(fit: "GmmFit1D | None") -> "GmmFit1D | None":
    if fit is None or fit.mu_lo <= fit.mu_hi:
        return fit
    return GmmFit1D(fit.w_hi, fit.mu_hi, fit.var_hi, fit.w_lo, fit.mu_lo, fit.var_lo)


def run_em(x: np.ndarray, init: GmmFit1D, loglik_tol: float, max_iter: int) -> "tuple[GmmFit1D | None, dict]":
    """The shipped EM loop (``_plain_em``'s body) at a chosen tolerance and budget."""
    stats: dict = {}
    fit = G._anchored_em(x, G._NO_ANCHORS, G._NO_ANCHORS, init, 1.0, max_iter, G._EM_TOL, loglik_tol, stats)
    return _ordered(fit), stats


def split_init(xs: np.ndarray, k: int) -> GmmFit1D:
    """Components = the sorted sample's first *k* and the rest, by moments."""
    n = xs.size
    k = min(max(k, 1), n - 1)
    floor = max(1e-12, G._ANCHOR_VAR_FLOOR_FRAC * float(np.var(xs)))
    lo, hi = xs[:k], xs[k:]
    return GmmFit1D(
        k / n, float(lo.mean()), max(float(lo.var()), floor), (n - k) / n, float(hi.mean()), max(float(hi.var()), floor)
    )


def random_init(xs: np.ndarray, rng: np.random.Generator) -> GmmFit1D:
    """Two distinct data points as means, the pooled variance for both, equal weights.

    The textbook random start - what a re-initialisation of the same estimator
    looks like when nothing about the init is chosen for the data.
    """
    i, j = rng.choice(xs.size, size=2, replace=False)
    a, b = sorted((float(xs[i]), float(xs[j])))
    if b <= a:
        b = a + 1e-6
    v = max(float(np.var(xs)), 1e-12)
    return GmmFit1D(0.5, a, v, 0.5, b, v)


def mean_loglik(fit: "GmmFit1D | None", x: np.ndarray) -> float:
    if fit is None:
        return float("nan")
    out = np.zeros_like(x)
    parts = []
    for w, mu, var in ((fit.w_lo, fit.mu_lo, fit.var_lo), (fit.w_hi, fit.mu_hi, fit.var_hi)):
        if w <= 0 or var <= 0:
            continue
        parts.append(math.log(w) - 0.5 * math.log(2 * math.pi * var) - 0.5 * (x - mu) ** 2 / var)
    if not parts:
        return float("nan")
    out = np.logaddexp.reduce(np.vstack(parts), axis=0)
    return float(out.mean())


def ashman_d(fit: "GmmFit1D | None") -> float:
    """``|mu_hi - mu_lo| * sqrt(2 / (var_lo + var_hi))``; > 2 means clean separation."""
    if fit is None:
        return float("nan")
    return abs(fit.mu_hi - fit.mu_lo) * math.sqrt(2.0 / max(fit.var_lo + fit.var_hi, 1e-300))


# ---------------------------------------------------------------- fitted rules


@dataclass
class FitResult:
    cut: float
    fit: "GmmFit1D | None"
    n_iter: float = float("nan")
    converged: float = float("nan")


def _mid(fit: "GmmFit1D | None", x: np.ndarray) -> float:
    return float(np.median(x)) if fit is None else fit.midpoint()


def fit_shipped(x: np.ndarray, *, init: "GmmFit1D | None" = None, loglik_tol: float = G._EM_LOGLIK_TOL) -> FitResult:
    """The shipped fit; with defaults this IS ``fit_score_gmm`` (asserted by the gate)."""
    xs = np.sort(x)
    start = init if init is not None else G._two_means_init(xs)
    if start is None:
        return FitResult(float(np.median(x)), None)
    fit, st = run_em(x, start, loglik_tol, G._EM_MAX_ITER)
    return FitResult(_mid(fit, x), fit, st.get("n_iter", float("nan")), st.get("converged", float("nan")))


def fit_converged(x: np.ndarray, *, init: "GmmFit1D | None" = None) -> FitResult:
    xs = np.sort(x)
    start = init if init is not None else G._two_means_init(xs)
    fit, st = run_em(x, start, CONVERGED_TOL, CONVERGED_MAX_ITER)
    return FitResult(_mid(fit, x), fit, st.get("n_iter", float("nan")), st.get("converged", float("nan")))


def fit_multistart(x: np.ndarray, *, starts: "list[GmmFit1D] | None" = None) -> FitResult:
    """Best-likelihood of several fully converged EM runs."""
    xs = np.sort(x)
    if starts is None:
        starts = [G._two_means_init(xs)] + [split_init(xs, int(round(q * xs.size))) for q in MULTISTART_QUANTILES]
    best: "tuple[float, GmmFit1D | None, dict]" = (-math.inf, None, {})
    total_iter = 0.0
    all_conv = 1.0
    for s in starts:
        if s is None:
            continue
        fit, st = run_em(x, s, CONVERGED_TOL, CONVERGED_MAX_ITER)
        total_iter += st.get("n_iter", 0.0)
        all_conv = min(all_conv, st.get("converged", 0.0))
        ll = mean_loglik(fit, x)
        if fit is not None and ll > best[0]:
            best = (ll, fit, st)
    return FitResult(_mid(best[1], x), best[1], total_iter, all_conv)


# ------------------------------------------------------------ closed-form rules


def quantile_cut(x: np.ndarray, q: float) -> float:
    """Admit the top ``q`` fraction (at least one item)."""
    xs = np.sort(x)
    k = max(1, int(round(q * xs.size)))
    return float(xs[xs.size - k])


def otsu_cut(x: np.ndarray) -> float:
    """Exact 2-means boundary: the split of the sorted sample maximising between-class variance."""
    xs = np.sort(x)
    n = xs.size
    c = np.cumsum(xs)
    k = np.arange(1, n)
    m0 = c[:-1] / k
    m1 = (c[-1] - c[:-1]) / (n - k)
    between = (k / n) * ((n - k) / n) * (m1 - m0) ** 2
    i = int(np.argmax(between))  # low class = xs[: i + 1]
    return float(0.5 * (xs[i] + xs[i + 1]))


def bulk_location_scale(x: np.ndarray) -> "tuple[float, float]":
    """Median, and the left half-MAD scaled to a Gaussian sigma.

    The left half is the side of the bulk the query's matches do not reach, so
    a heavy right shoulder of positives moves neither number much.
    """
    mu = float(np.median(x))
    left = mu - x[x <= mu]
    s = 1.4826 * float(np.median(left)) if left.size else 0.0
    if not s > 0:
        s = max(float(np.std(x)), 1e-12)
    return mu, s


def tail_z_cut(x: np.ndarray, k: float) -> float:
    mu, s = bulk_location_scale(x)
    return mu + k * s


def _norm_sf(z: np.ndarray) -> np.ndarray:
    from scipy.special import ndtr  # noqa: PLC0415

    return ndtr(-z)


def tail_fdr_cut(x: np.ndarray, alpha: float) -> float:
    """Lowest observed score whose admitted set the Gaussian bulk explains at most *alpha* of.

    ``FDR(t) = n * P_bulk(X >= t) / #{x >= t}`` with the bulk from
    :func:`bulk_location_scale` and ``pi0`` taken as 1 (conservative: a typed
    query's matches are a small fraction).  Step-up, as in Benjamini-Hochberg:
    the deepest rank at which the estimate clears *alpha*.  When no rank does,
    the cut sits above every score - "nothing here stands out" is an answer.
    """
    mu, s = bulk_location_scale(x)
    xs = np.sort(x)[::-1]
    n = xs.size
    fdr = n * _norm_sf((xs - mu) / s) / np.arange(1, n + 1)
    ok = np.nonzero(fdr <= alpha)[0]
    if ok.size == 0:
        return float(xs[0] + 1e-9)
    return float(xs[ok[-1]])


# ------------------------------------------------------------------- registry


@dataclass(frozen=True)
class Rule:
    name: str
    family: str
    cut: Callable[[np.ndarray], float]
    #: ``(label, fn)`` pairs: the same rule with its optimiser perturbed.
    perturb: "tuple[tuple[str, Callable[[np.ndarray], float]], ...]" = ()
    headline: bool = False


def _random_perturbs(maker: Callable[[np.ndarray, np.random.Generator], float], n: int = 5):
    out = []
    for r in range(n):

        def fn(x, r=r):
            return maker(x, np.random.default_rng(1000 + r))

        out.append((f"rinit{r}", fn))
    return tuple(out)


def _guarded(fallback: Callable[[np.ndarray], float]) -> Callable[[np.ndarray], float]:
    def cut(x: np.ndarray) -> float:
        r = fit_shipped(x)
        if r.fit is not None and ashman_d(r.fit) >= ASHMAN_D_SEPARATED:
            return r.cut
        return fallback(x)

    return cut


def _priorfree(fit: "GmmFit1D | None", x: np.ndarray) -> float:
    if fit is None:
        return float(np.median(x))
    c = fit.rate_crossing(1.0, 1.0)
    return fit.midpoint() if c is None else c


def build_rules() -> "dict[str, Rule]":
    rules: list[Rule] = [
        Rule(
            "gmm_shipped",
            "mixture",
            lambda x: fit_shipped(x).cut,
            _random_perturbs(lambda x, g: fit_shipped(x, init=random_init(np.sort(x), g)).cut)
            + (
                ("tol1e-4", lambda x: fit_shipped(x, loglik_tol=1e-4).cut),
                ("tol1e-5", lambda x: fit_shipped(x, loglik_tol=1e-5).cut),
            ),
            headline=True,
        ),
        Rule(
            "gmm_priorfree",
            "mixture",
            lambda x: _priorfree(fit_shipped(x).fit, x),
            _random_perturbs(lambda x, g: _priorfree(fit_shipped(x, init=random_init(np.sort(x), g)).fit, x))
            + (("tol1e-5", lambda x: _priorfree(fit_shipped(x, loglik_tol=1e-5).fit, x)),),
            headline=True,
        ),
        Rule(
            "gmm_converged",
            "mixture",
            lambda x: fit_converged(x).cut,
            _random_perturbs(lambda x, g: fit_converged(x, init=random_init(np.sort(x), g)).cut),
            headline=True,
        ),
        Rule(
            "gmm_multistart",
            "mixture",
            lambda x: fit_multistart(x).cut,
            _random_perturbs(
                lambda x, g: (
                    fit_multistart(
                        x, starts=[random_init(np.sort(x), g) for _ in range(len(MULTISTART_QUANTILES) + 1)]
                    ).cut
                ),
                n=3,
            ),
            headline=True,
        ),
        Rule("otsu", "exact", otsu_cut, headline=True),
    ]
    for q in (0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20):
        rules.append(Rule(f"quantile{q:g}", "quantile", lambda x, q=q: quantile_cut(x, q), headline=q in (0.02, 0.05)))
    for k in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0):
        rules.append(Rule(f"tail_z{k:g}", "tail", lambda x, k=k: tail_z_cut(x, k), headline=k in (2.5, 3.0)))
    for a in (0.05, 0.1, 0.2, 0.3, 0.5):
        rules.append(Rule(f"tail_fdr{a:g}", "tail", lambda x, a=a: tail_fdr_cut(x, a), headline=a in (0.2, 0.3)))
    rules.append(Rule("gmm_guarded_z3", "guarded", _guarded(lambda x: tail_z_cut(x, 3.0)), headline=True))
    rules.append(Rule("gmm_guarded_fdr0.2", "guarded", _guarded(lambda x: tail_fdr_cut(x, 0.2)), headline=True))
    return {r.name: r for r in rules}


RULES = build_rules()
