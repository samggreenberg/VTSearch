"""Active-learning acquisition strategies for the voting-iterations harness.

The voting-iterations evaluator (:mod:`vtscore.eval.voting_iterations`) simulates
a user labelling one item at a time and measures how fast the detector's cost
falls.  *Which* item the simulated user is shown next is an
**acquisition strategy**: a rule that, given the current model, its scores over
the unlabelled pool, the embeddings, and the labels gathered so far, picks the
next item to query.  A good strategy reaches a low-cost detector in fewer votes
than random labelling.

An :class:`ALContext` bundles that per-step state; a strategy is a callable
``ALContext -> int`` returning the chosen pool id.  The :data:`STRATEGIES`
registry maps a name to that callable:

- ``random``   — uniform pick from the pool (the baseline every strategy is
  measured against).
- ``margin``   — smallest ``|p - t|``: the item whose score sits closest to the
  decision threshold, i.e. the model is most on-the-fence about.
- ``entropy``  — largest binary entropy ``H(p)``: maximal predictive
  uncertainty (for a threshold at 0.5 this and ``margin`` coincide; away from
  0.5 they diverge).
- ``bald``     — largest MC-dropout mutual information: runs the trained net
  with dropout left on for several stochastic passes and picks the item the
  ensemble *disagrees* with itself about most (epistemic, not aleatoric,
  uncertainty).
- ``ensemble_std`` — like ``bald`` but the disagreement comes from an explicit
  N-member *deep* ensemble: train :data:`_ENSEMBLE_MEMBERS` seed-varied MLPs on
  the votes so far, score the pool with each, and pick the item with the
  greatest member-to-member sigmoid std.  Costs N retrains per step (no dropout
  needed at inference) where ``bald`` costs one train plus N cheap dropout
  passes.
- ``eig``      — largest expected pool-entropy reduction, estimated by
  counterfactually retraining the model with each candidate labelled both ways
  (``2 x K`` retrains) and scoring how much the pool's total entropy is expected
  to drop.
- ``coreset``  — greedy k-centre: the pool item **farthest** (in embedding
  space) from everything labelled so far, spreading coverage rather than
  chasing uncertainty.
- ``balanced`` — an **oracle** ordering (not label-blind): peeks at the pool's
  ground-truth labels to always query the currently under-represented class, so
  the running Good/Bad counts stay balanced.  It is a diagnostic baseline
  isolating "does class balance during voting help?" from vote *content*, only
  usable in the simulation harness where pool labels are known; it reads
  :attr:`ALContext.pool_labels` and degrades to ``random`` when that is absent.

Each strategy above also has a ``density_<name>`` variant that multiplies its
per-item acquisition utility by ``1 / sqrt(cell_count)`` — the reciprocal root
of the population of the item's :class:`~vtscore.state.coverage_atlas.CoverageAtlas`
leaf — so items in sparsely-populated regions of the dataset are up-weighted.
This is *information density* sampling: prefer items that are both informative
and representative of an under-explored pocket.  ``density_*`` variants exist for
every base strategy except ``random`` and ``eig`` (``random`` is the baseline;
``eig`` is already the most expensive sampler and its density variant buys
little).

All acquisition utilities are non-negative and higher-is-better, so the density
multiply only ever *re-ranks* by rarity, never flips the base preference.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    import torch

    from vtscore.state.coverage_atlas import CoverageAtlas


# Number of stochastic forward passes for the MC-dropout BALD estimate.  More
# passes shrink the Monte-Carlo error on the mutual information at linear cost;
# 20 is the usual sweet spot in the deep-AL literature.
_BALD_PASSES = 20

# Number of seed-varied MLPs trained per step for the ``ensemble_std`` deep
# ensemble.  Members use fixed seeds ``42 + k`` so the pick is a deterministic
# function of the votes cast so far (reproducible for a given eval seed).
_ENSEMBLE_MEMBERS = 5

# Ceiling on the number of pool candidates ``eig`` counterfactually retrains
# for.  A retrain is expensive, so on a large pool we only evaluate the most
# uncertain items (top-K by entropy) — the low-entropy tail is almost never the
# information-maximising pick anyway.  ``2 x K`` retrains happen per step.
_EIG_MAX_CANDIDATES = 32

_EPS = 1e-9


@dataclass
class ALContext:
    """Per-step state an acquisition strategy needs to pick the next item.

    Attributes:
        pool_ids: Candidate ids the strategy chooses among (unlabelled so far),
            in a stable order; every utility array a scorer returns is aligned
            to this list.
        embeddings: ``{id: vector}`` for every id (pool and labelled) — the
            whole-image embedding, used by ``coreset``/``eig`` and by the
            density atlas.
        labeled: ``{id: 1.0 | 0.0}`` for every item labelled so far
            ("labelled-so-far").  Empty before the first vote.
        scores: ``{pool_id: p}`` sigmoid probabilities from the current model,
            used by ``margin``/``entropy``/``eig``.  Empty before the first
            model exists.
        model: The current trained MLP (``None`` before the first trainable
            step, i.e. before one Good and one Bad vote coexist).
        threshold: The current decision threshold ``t`` (``margin`` measures
            ``|p - t|`` against it).
        input_dim: Embedding dimensionality, for ``eig`` counterfactual retrains.
        hidden_dim: Hidden width the counterfactual retrains should use (matches
            the live model's width); ``None`` lets training auto-size.
        atlas: The dataset's coverage atlas, for ``density_*`` variants (``None``
            disables density weighting — every weight is 1).
        rng: Seeded RNG driving ``random`` and the MC-dropout seed, so a run is
            reproducible from its seed.
        pool_labels: Ground-truth ``{pool_id: 1.0 | 0.0}`` for the pool, used
            **only** by the ``balanced`` oracle strategy to interleave Good/Bad
            queries.  This deliberately breaks the label-blind contract every
            other strategy honours; it is meaningful only in the simulation
            harness (where every pool label is known ground truth).  ``None``
            (the default) makes ``balanced`` degrade to a random pick.
    """

    pool_ids: list[int]
    embeddings: dict[int, np.ndarray]
    labeled: dict[int, float]
    scores: dict[int, float]
    model: Optional["torch.nn.Sequential"]
    threshold: float
    input_dim: int
    hidden_dim: Optional[int]
    atlas: Optional["CoverageAtlas"]
    rng: np.random.RandomState = field(default_factory=lambda: np.random.RandomState(0))
    pool_labels: Optional[dict[int, float]] = None


# ------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    """Return the binary entropy ``H(p) = -p log p - (1-p) log(1-p)`` (nats).

    Clamps *p* off ``{0, 1}`` so the ``log`` stays finite; the entropy of a
    saturated prediction is 0, which the clamp reproduces to machine precision.
    """
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)
    return -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)


def _pool_matrix(ctx: ALContext) -> np.ndarray:
    """Stack the pool embeddings into an ``(n_pool, dim)`` float32 matrix."""
    return np.array([ctx.embeddings[i] for i in ctx.pool_ids], dtype=np.float32)


def _pool_scores(ctx: ALContext) -> np.ndarray:
    """Return the current model's pool probabilities aligned to ``pool_ids``.

    Falls back to ``0.5`` (maximal uncertainty) for any pool id without a
    recorded score, so a scorer never trips on a missing key.
    """
    return np.array([ctx.scores.get(i, 0.5) for i in ctx.pool_ids], dtype=np.float64)


# ------------------------------------------------------------------
# Base acquisition scorers: ALContext -> per-pool-item utility (higher = pick)
# ------------------------------------------------------------------


def _score_random(ctx: ALContext) -> np.ndarray:
    """Uniform-random utility — argmax of it is a uniform pick from the pool."""
    return ctx.rng.random(len(ctx.pool_ids))


def _score_margin(ctx: ALContext) -> np.ndarray:
    """Margin sampling: high utility for items whose score is near the threshold.

    Utility ``1 - |p - t|`` is maximal (1) when ``p == t`` and non-negative
    everywhere (``p, t`` both in ``[0, 1]``), so its argmax is the minimal-margin
    item and the density multiply keeps its sign.
    """
    p = _pool_scores(ctx)
    return 1.0 - np.abs(p - ctx.threshold)


def _score_entropy(ctx: ALContext) -> np.ndarray:
    """Predictive entropy ``H(p)`` — maximal at ``p = 0.5``, non-negative."""
    return _binary_entropy(_pool_scores(ctx))


def _score_bald(ctx: ALContext) -> np.ndarray:
    """MC-dropout BALD mutual information ``H(E[p]) - E[H(p)]`` per pool item.

    Runs the trained net with dropout **on** for :data:`_BALD_PASSES` stochastic
    passes; the gap between the entropy of the averaged prediction and the
    average of the per-pass entropies is the model's epistemic uncertainty
    (disagreement across dropout masks), which is non-negative by Jensen.  The
    dropout RNG is forked and seeded from ``ctx.rng`` so the estimate is
    reproducible.
    """
    import torch  # noqa: PLC0415

    model = ctx.model
    assert model is not None
    X = torch.tensor(_pool_matrix(ctx), dtype=torch.float32)
    device = next(model.parameters()).device
    X = X.to(device)

    seed = int(ctx.rng.randint(0, 2**31 - 1))
    was_training = model.training
    probs: list[np.ndarray] = []
    model.train()  # enable Dropout layers
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(seed)
        for _ in range(_BALD_PASSES):
            p = torch.sigmoid(model(X)).squeeze(1).cpu().numpy()
            probs.append(p.astype(np.float64))
    if not was_training:
        model.eval()

    stacked = np.stack(probs, axis=0)  # (passes, n_pool)
    mean_p = stacked.mean(axis=0)
    entropy_of_mean = _binary_entropy(mean_p)
    mean_of_entropy = _binary_entropy(stacked).mean(axis=0)
    return np.maximum(entropy_of_mean - mean_of_entropy, 0.0)


def _score_coreset(ctx: ALContext) -> np.ndarray:
    """Greedy k-centre utility: distance from each pool item to the labelled set.

    The utility of a pool item is its Euclidean distance to the **nearest**
    already-labelled embedding; its argmax is the farthest-from-covered item,
    the greedy k-centre pick.  With nothing labelled yet the notion is undefined,
    so it degrades to a random utility (the very first pick is arbitrary anyway).
    """
    if not ctx.labeled:
        return _score_random(ctx)
    pool = _pool_matrix(ctx)
    labelled = np.array([ctx.embeddings[i] for i in ctx.labeled], dtype=np.float32)
    # Pairwise distances (n_pool, n_labelled); nearest labelled per pool item.
    dists = np.linalg.norm(pool[:, None, :] - labelled[None, :, :], axis=2)
    return dists.min(axis=1)


def _score_ensemble_std(ctx: ALContext) -> np.ndarray:
    """Deep-ensemble disagreement: per-pool member-to-member sigmoid std.

    Trains :data:`_ENSEMBLE_MEMBERS` seed-varied MLPs on the votes gathered so
    far (``ctx.labeled``), scores every pool item's embedding with each member,
    and returns the per-item std across members — high where the ensemble
    disagrees, i.e. maximal epistemic uncertainty.  Fixed member seeds keep the
    utility a deterministic function of the current votes.  Requires at least
    one Good and one Bad vote to be trainable; the cold-start guard
    (``needs_model``) routes the pre-trainable steps to a random pick.
    """
    import torch  # noqa: PLC0415

    from vtscore.training.mlp import train_model  # noqa: PLC0415

    X = torch.tensor(np.array([ctx.embeddings[i] for i in ctx.labeled]), dtype=torch.float32)
    y = torch.tensor([float(v) for v in ctx.labeled.values()], dtype=torch.float32).unsqueeze(1)
    X_cand = torch.tensor(_pool_matrix(ctx), dtype=torch.float32)

    member_scores: list[np.ndarray] = []
    for k in range(_ENSEMBLE_MEMBERS):
        model = train_model(X, y, ctx.input_dim, seed=42 + k, hidden_dim=ctx.hidden_dim)
        with torch.no_grad():
            Xk = X_cand.to(next(model.parameters()).device)
            member_scores.append(torch.sigmoid(model(Xk)).squeeze(1).cpu().numpy())
    return np.stack(member_scores, axis=0).std(axis=0).astype(np.float64)


def _score_balanced(ctx: ALContext) -> np.ndarray:
    """Oracle class-balancing utility: prefer the under-represented class.

    Reads the pool's ground-truth labels (:attr:`ALContext.pool_labels`) — the
    one strategy that is *not* label-blind — and gives every pool item of the
    currently under-represented class a utility that dominates the other class,
    with a small seeded jitter to break ties randomly.  Argmax therefore queries
    a random item of the needed class, keeping the running Good/Bad counts within
    one of each other.  Falls back to a plain random utility when no pool labels
    are supplied (the harness always supplies them; a direct caller may not).
    """
    if ctx.pool_labels is None:
        return _score_random(ctx)
    labels = np.array([ctx.pool_labels.get(i, 0.0) for i in ctx.pool_ids], dtype=np.float64)
    n_good = sum(1 for v in ctx.labeled.values() if v == 1.0)
    n_bad = len(ctx.labeled) - n_good
    needed = 1.0 if n_good <= n_bad else 0.0
    # Matched items score in [1, 1.5); unmatched in [0, 0.5) — matched always
    # wins, and the jitter randomises the pick within the needed class.  When the
    # needed class is exhausted every item is "unmatched" and one is picked at
    # random from whatever remains.
    match = (labels == needed).astype(np.float64)
    return match + 0.5 * ctx.rng.random(len(labels))


def _score_eig(ctx: ALContext) -> np.ndarray:
    """Expected pool-entropy reduction via ``2 x K`` counterfactual retrains.

    For each evaluated candidate, retrain the model twice — once assuming the
    candidate is labelled Good, once Bad — and score the *rest* of the pool with
    each retrained model.  The candidate's utility is the current summed pool
    entropy (over that same rest) minus the label-probability-weighted expected
    entropy after the vote, i.e. how much labelling it is expected to sharpen the
    model over everything else.  Only the top-:data:`_EIG_MAX_CANDIDATES` pool
    items by current entropy are evaluated (a retrain is costly); the rest get
    ``-inf`` so they are never chosen over an evaluated candidate.
    """
    import torch  # noqa: PLC0415

    from vtscore.training.mlp import train_model  # noqa: PLC0415

    pool = ctx.pool_ids
    n = len(pool)
    util = np.full(n, -np.inf, dtype=np.float64)
    if n <= 1:
        # Nothing to reduce entropy over; fall back to raw entropy so a valid
        # pick still comes out.
        return _score_entropy(ctx)

    cur_p = _pool_scores(ctx)
    # Evaluate the most uncertain candidates first, capped for cost.
    order = np.argsort(-_binary_entropy(cur_p))
    candidate_idx = order[: min(_EIG_MAX_CANDIDATES, n)]

    base_X = [ctx.embeddings[i] for i in ctx.labeled]
    base_y = [float(v) for v in ctx.labeled.values()]

    for j in candidate_idx:
        cand = pool[j]
        rest_idx = [k for k in range(n) if k != j]
        rest = np.array([ctx.embeddings[pool[k]] for k in rest_idx], dtype=np.float32)
        cur_rest_entropy = float(_binary_entropy(cur_p[rest_idx]).sum())

        p_good = float(cur_p[j])
        expected_entropy = 0.0
        for label, weight in ((1.0, p_good), (0.0, 1.0 - p_good)):
            X = torch.tensor(np.array(base_X + [ctx.embeddings[cand]]), dtype=torch.float32)
            y = torch.tensor(base_y + [label], dtype=torch.float32).unsqueeze(1)
            model = train_model(X, y, ctx.input_dim, hidden_dim=ctx.hidden_dim)
            with torch.no_grad():
                Xr = torch.tensor(rest, dtype=torch.float32).to(next(model.parameters()).device)
                pr = torch.sigmoid(model(Xr)).squeeze(1).cpu().numpy()
            expected_entropy += weight * float(_binary_entropy(pr).sum())
        util[j] = cur_rest_entropy - expected_entropy

    return util


# Base scorers and whether they require a trained model to run.  When the model
# is absent (cold start) a model-requiring scorer degrades to a random pick.
_BASE_SCORERS: dict[str, tuple[Callable[[ALContext], np.ndarray], bool]] = {
    "random": (_score_random, False),
    "margin": (_score_margin, True),
    "entropy": (_score_entropy, True),
    "bald": (_score_bald, True),
    "ensemble_std": (_score_ensemble_std, True),
    "eig": (_score_eig, True),
    "coreset": (_score_coreset, False),
    "balanced": (_score_balanced, False),
}

# Which base strategies get a ``density_<name>`` companion.  ``random`` is the
# baseline and ``eig`` is already the costliest sampler, so neither is reweighted.
_DENSITY_BASES = ["margin", "entropy", "bald", "coreset"]


# ------------------------------------------------------------------
# Density weighting
# ------------------------------------------------------------------


def density_weights(ctx: ALContext) -> np.ndarray:
    """Return ``1 / sqrt(cell_count)`` per pool item from the coverage atlas.

    ``cell_count`` is the population of the item's leaf node in the dataset's
    :class:`~vtscore.state.coverage_atlas.CoverageAtlas`, so items in sparse
    regions score higher.  Items with no atlas, or an id the atlas never saw,
    get weight 1 (no reweighting).  Weights are strictly positive, so a density
    multiply re-ranks the base utility by rarity without changing its sign.
    """
    n = len(ctx.pool_ids)
    if ctx.atlas is None:
        return np.ones(n, dtype=np.float64)
    weights = np.ones(n, dtype=np.float64)
    for i, cid in enumerate(ctx.pool_ids):
        leaf = ctx.atlas.vector_to_leaf.get(cid)
        if leaf is None:
            continue
        count = ctx.atlas.nodes[leaf]["n"]
        if count > 0:
            weights[i] = 1.0 / np.sqrt(count)
    return weights


def _make_selector(
    scorer: Callable[[ALContext], np.ndarray],
    *,
    needs_model: bool,
    density: bool,
) -> Callable[[ALContext], int]:
    """Wrap a base scorer into a full ``ALContext -> chosen id`` selector.

    Applies the cold-start fallback (random utility when the scorer needs a
    model that doesn't exist yet) and the optional density reweight, then
    returns the argmax pool id.
    """

    def select(ctx: ALContext) -> int:
        if needs_model and ctx.model is None:
            util = _score_random(ctx)
        else:
            util = scorer(ctx)
        if density:
            util = util * density_weights(ctx)
        return ctx.pool_ids[int(np.argmax(util))]

    return select


def _build_registry() -> dict[str, Callable[[ALContext], int]]:
    """Assemble the name -> selector registry from the base scorers."""
    registry: dict[str, Callable[[ALContext], int]] = {}
    for name, (scorer, needs_model) in _BASE_SCORERS.items():
        registry[name] = _make_selector(scorer, needs_model=needs_model, density=False)
    for name in _DENSITY_BASES:
        scorer, needs_model = _BASE_SCORERS[name]
        registry[f"density_{name}"] = _make_selector(scorer, needs_model=needs_model, density=True)
    return registry


#: Registry of acquisition strategies: ``name -> (ALContext -> chosen pool id)``.
STRATEGIES: dict[str, Callable[[ALContext], int]] = _build_registry()


def select_next(strategy: str, ctx: ALContext) -> int:
    """Pick the next pool id to query using the named *strategy*.

    Raises ``KeyError`` (via :func:`available_strategies` message) for an unknown
    name so a typo fails loudly instead of silently defaulting.
    """
    try:
        selector = STRATEGIES[strategy]
    except KeyError:
        raise KeyError(
            f"Unknown acquisition strategy {strategy!r}; available: {', '.join(available_strategies())}"
        ) from None
    if not ctx.pool_ids:
        raise ValueError("select_next called with an empty pool")
    return selector(ctx)


def available_strategies() -> list[str]:
    """Return the sorted list of registered acquisition strategy names."""
    return sorted(STRATEGIES)


def acquisition_utilities(strategy: str, ctx: ALContext) -> np.ndarray:
    """Return the (density-weighted) per-pool-item utilities for *strategy*.

    Exposes the raw utility vector the selector maximises, aligned to
    ``ctx.pool_ids`` — useful for tests and for inspecting *why* a pick was made.
    Mirrors the cold-start and density handling of the selector.
    """
    base = strategy[len("density_") :] if strategy.startswith("density_") else strategy
    if base not in _BASE_SCORERS:
        raise KeyError(f"Unknown acquisition strategy {strategy!r}")
    scorer, needs_model = _BASE_SCORERS[base]
    if needs_model and ctx.model is None:
        util = _score_random(ctx)
    else:
        util = scorer(ctx)
    if strategy.startswith("density_"):
        util = util * density_weights(ctx)
    return util
