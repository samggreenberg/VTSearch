"""The example opening for the voting-iterations harness (#4699): a session that starts from K photos.

The app has two real openings. One is a typed query's text sort. The other is an **example sort**:
the user loads one or more photos of what they want (New Detector's examples, or right-click "Sort by
this"), each is voted Good without a click
(:func:`vtscore.detectors.media_seeding.seed_good_votes_from_examples`), and the Good phase walks
down the ranking by cosine to the examples' centroid
(:func:`vtscore.training.query_sort.example_sort_from_paths`).

The harness's no-text start, three random known-goods, stands in for that flow where every class has
hundreds of positives. It breaks where a class has six: an identity in a face benchmark has about
three photos in the voting half, so the opening takes all of them and leaves nothing to find, and
in 42% of FHIBE cells it cannot finish at all (#4699).

Two pieces, both opt-in:

* :func:`stratified_split` splits the target's positives apart from the negatives, so every class
  with two or more positives keeps at least one on each side. A plain random 50/50 split leaves the
  withheld half empty for 4% of six-photo identities, and those cells score nothing.
* :func:`choose_examples` and :func:`example_sort` take *K* examples from the voting half's
  positives and rank every media by cosine to their centroid. The examples are nested in *K* under
  one seed (the first example of a K=4 run is the K=1 run's example), and the withheld half does
  not depend on *K*, so arms that differ only in *K* pair cell for cell.

The examples are cast as the run's first *K* votes, logged with phase :data:`EXAMPLE_PHASE`, so every
row and curve keeps counting votes. A reader who wants clicks after the examples reads ``t - K``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from vtscore.embedding.media_vectors import media_embedding
from vtscore.eval.labels import media_is_positive

if TYPE_CHECKING:
    import numpy as np

#: The phase an example vote is logged under in the pick log.
EXAMPLE_PHASE = "example"

#: The example draw's own stream, so choosing examples takes nothing from the run's RNG.
_EXAMPLE_STREAM = 4699


def positive_split_sizes(n_pos: int, sim_fraction: float) -> tuple[int, int]:
    """``(voting, withheld)`` counts for a class with *n_pos* positives under :func:`stratified_split`.

    The withheld half takes ``floor(n_pos * (1 - sim_fraction))`` of them, but never fewer than one
    nor all of them, so a class with two or more positives has one on each side. At the default
    ``sim_fraction=0.5`` that is ``ceil(n/2)`` to vote on and ``floor(n/2)`` withheld. A single
    positive can only sit on one side; it is voted on, and the run's empty-test guard skips the cell.
    """
    if n_pos < 2:
        return n_pos, 0
    # The epsilon keeps a product like 10 * (1 - 0.7) = 2.9999999999999996 from flooring a whole
    # number away.
    n_test = min(max(1, int(n_pos * (1.0 - sim_fraction) + 1e-9)), n_pos - 1)
    return n_pos - n_test, n_test


def stratified_split(
    clips_dict: dict[int, dict[str, Any]],
    target_category: str,
    sim_fraction: float,
    rng: np.random.RandomState,
) -> tuple[list[int], list[int]]:
    """Split *clips_dict* into ``(sim_ids, test_ids)`` with the target's positives split on their own.

    Positives are shuffled and divided by :func:`positive_split_sizes`. Negatives are shuffled and
    divided at *sim_fraction*, as the plain split divides everything. Both shuffles draw from *rng*,
    so the split is fixed by the run's seed alone.
    """
    ids = sorted(clips_dict)
    pos = [cid for cid in ids if media_is_positive(clips_dict[cid], target_category)]
    neg = [cid for cid in ids if not media_is_positive(clips_dict[cid], target_category)]
    pos_order = [int(c) for c in rng.permutation(pos)] if pos else []
    neg_order = [int(c) for c in rng.permutation(neg)] if neg else []
    n_sim_pos, _ = positive_split_sizes(len(pos), sim_fraction)
    n_sim_neg = int(len(neg) * sim_fraction)
    return pos_order[:n_sim_pos] + neg_order[:n_sim_neg], pos_order[n_sim_pos:] + neg_order[n_sim_neg:]


def check_example_opening(
    seed_examples: Optional[int],
    *,
    stratify_target: bool,
    seed_scores: Optional[dict[int, float]],
    startup_schedule: Optional[str],
    train_mix: Any,
    target_prevalence: Optional[float],
) -> None:
    """Refuse an example opening the run cannot honour, before any work.

    Raises:
        ValueError: If *seed_examples* is not a positive integer, or is combined with a text sort
            (*seed_scores*), a startup schedule, a mixed-band cell or a thinned prevalence; or if
            *stratify_target* is combined with a mixed-band cell, whose split is its own.
    """
    if stratify_target and train_mix is not None:
        raise ValueError("stratify_target and train_mix both define the split; a mixed-band cell keeps its own")
    if seed_examples is None:
        return
    if isinstance(seed_examples, bool) or not isinstance(seed_examples, int) or seed_examples < 1:
        raise ValueError(f"seed_examples must be an integer >= 1 or None; got {seed_examples!r}")
    if seed_scores is not None:
        raise ValueError("seed_examples IS the opening's sort; it cannot also take a text sort (seed_scores)")
    if startup_schedule is not None:
        raise ValueError("seed_examples opens on the app's Good phase; drop startup_schedule")
    if train_mix is not None or target_prevalence is not None:
        raise ValueError("seed_examples draws from the natural voting half; drop train_mix and target_prevalence")


def choose_examples(
    sim_ids: list[int],
    clips_dict: dict[int, dict[str, Any]],
    target_category: str,
    k: int,
    seed: int,
) -> list[int]:
    """The *k* positives of the voting half the session starts from, in the order they are voted.

    Drawn from a stream of the run's *seed* and nothing else, so for one seed the examples are
    nested in *k*.

    Raises:
        ValueError: If the voting half holds fewer than *k* positives. Under :func:`stratified_split`
            that count is fixed by the class's size (:func:`positive_split_sizes`), so a study can
            select classes that never reach here.
    """
    import numpy as np  # noqa: PLC0415

    pos = sorted(cid for cid in sim_ids if media_is_positive(clips_dict[cid], target_category))
    if len(pos) < k:
        raise ValueError(
            f"{target_category!r} has {len(pos)} positive(s) in the voting half and the opening needs {k} examples"
        )
    order = np.random.default_rng([int(seed), _EXAMPLE_STREAM]).permutation(np.array(pos, dtype=np.int64))
    return [int(c) for c in order[:k]]


def _unit(vec: Any) -> np.ndarray:
    import numpy as np  # noqa: PLC0415

    v = np.asarray(vec, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def example_sort(clips_dict: dict[int, dict[str, Any]], examples: list[int]) -> dict[int, float]:
    """``{media_id: cosine}`` to the centroid of *examples*, as the app's example sort ranks.

    One example ranks by cosine to its own vector. Several rank by cosine to the mean of their
    L2-normalised vectors, so each counts equally whatever its norm
    (:func:`~vtscore.training.query_sort.example_sort_from_paths`).

    Raises:
        ValueError: On a patch dataset. The app's example sort there pools regions
            (:mod:`vtscore.training.region_similarity`), which a whole-image cosine does not
            reproduce.
    """
    import numpy as np  # noqa: PLC0415

    if not examples:
        raise ValueError("example_sort needs at least one example")
    if any(m.get("patch_grid") is not None for m in clips_dict.values()):
        raise ValueError("the example opening ranks whole-image vectors; a patch dataset's example sort pools regions")
    ids = list(clips_dict)
    vectors = {cid: media_embedding(clips_dict[cid]) for cid in ids}
    missing = [cid for cid, v in vectors.items() if v is None]
    if missing:
        raise ValueError(f"{len(missing)} media(s) carry no vector to rank, e.g. {missing[:5]}")
    query = _unit(np.mean(np.stack([_unit(vectors[c]) for c in examples]), axis=0))
    matrix = np.stack([_unit(vectors[c]) for c in ids])
    cos = matrix @ query
    return {cid: float(cos[k]) for k, cid in enumerate(ids)}
