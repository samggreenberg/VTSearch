"""#4960: Find on a second corpus at every click, with the app's two domain-shift flags beside the objective.

The test side already models Find (#4452): at every click the withheld half is
scored with the detector Test gives and cut where a Find there would draw the
line, the labels' class model with the corpus side refitted on that half
(``LabelsLine.on_corpus``).  The withheld half is the corpus the votes came
from, so it cannot show what a detector does on a corpus unlike it, which is
what #4739 prices the Coverage Atlas walk on.  ``find_corpus`` gives the
simulator a second corpus to Find on, beside the in-corpus read:

* **The Find corpus** is a second dataset's cell of the same name, in the same
  embedder (:func:`find_cell`: ``bus@medium`` in ``o365_better`` for a
  ``coco_better`` run, #4815), or that dataset whole when it has no such cell,
  where the target has no positives.  A **mix** (:func:`parse_find_mix`) adds it
  to the withheld half as extra haystack instead: ``0`` is the withheld half
  alone, ``equal`` adds as many foreign medias as the withheld half holds (a
  seeded draw), any other share that many times it, and ``only`` - the default
  - is the foreign cell alone.  The medias are re-keyed past the cell's own ids
  (:func:`build_find_corpus`), so nothing collides, and they are only ever
  scored: they never enter the pool, the votes, the atlas or the folds.
* **At every click** the simulator scores the Find corpus with the detector the
  click's Test gives, refits the line's corpus side there, and emits
  :data:`~vtscore.eval.voting_columns.FIND_CORPUS_COLUMNS` beside the in-corpus
  columns (:func:`find_columns`).  The in-corpus columns do not move.
* **At the snapshot clicks** (``find_flag_steps``) it also reads the app's two
  domain-shift flags against that corpus, as delegated calls into the library
  functions the app's own routes call: the Training-domain overlap
  (:func:`training_domain_flags`, ``domain_shift_report`` against the atlas the
  harness builds over the sim pool for the New phase) and the Evidence
  coverage (:func:`evidence_flags`, ``evidence_coverage_report`` from the votes'
  vectors, :func:`vote_vectors`, and the Find cut).  Routing a whole corpus
  through the atlas is the costly step, so it runs at those clicks only.

What is ported rather than called is how the app's routes call the two reports
- which vectors, which threshold, which refusals - and the eval/app sync gate
pins both (``find.training_domain``, ``find.evidence_coverage``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vtscore.embedding.media_vectors import media_embedding
from vtscore.eval.labels import evaluable_pool, region_box_for_category
from vtscore.eval.step_model import good_training_vec
from vtscore.eval.voting_columns import FBETA_COLUMNS, FIND_CORPUS_COLUMNS, FIND_FLAG_COLUMNS

#: The mix that is the foreign cell alone, the default.
FIND_ONLY = "only"

#: Salt for the seeded draw of a mixed corpus's foreign medias: combined with the
#: run's seed on a generator of its own, so the draw pairs across arms of one
#: (cell, seed) and cannot move the trajectory's RandomState.
_MIX_SALT = 4960


def parse_find_mix(mix: "float | str | None") -> "float | str":
    """The Find corpus's mix: :data:`FIND_ONLY`, or the foreign medias' share of the withheld half's size.

    ``None`` and ``"only"`` are the foreign cell alone; ``"equal"`` is a share
    of 1, as many foreign medias as the withheld half holds; ``0`` is the
    withheld half alone; any other finite number ``>= 0`` is that share.
    Refused otherwise.
    """
    if mix is None:
        return FIND_ONLY
    if isinstance(mix, bool):
        raise ValueError(f"find_mix must be 'only', 'equal' or a share >= 0; got {mix!r}")
    if isinstance(mix, str):
        spelled = mix.strip().lower()
        if spelled == FIND_ONLY:
            return FIND_ONLY
        if spelled == "equal":
            return 1.0
        try:
            share = float(spelled)
        except ValueError:
            raise ValueError(f"find_mix must be 'only', 'equal' or a share >= 0; got {mix!r}") from None
    else:
        share = float(mix)
    if not math.isfinite(share) or share < 0:
        raise ValueError(f"find_mix must be 'only', 'equal' or a share >= 0; got {mix!r}")
    return share


def find_cell(medias: Mapping[int, dict[str, Any]], target_category: str) -> dict[int, dict[str, Any]]:
    """*medias*' cell of the same name as *target_category*, or *medias* whole when it has none.

    The cell is :func:`~vtscore.eval.labels.evaluable_pool`'s: on a dataset
    that designates its cells (a scale-banded one) the medias scorable for the
    target, every media elsewhere.  A designated dataset with no such cell -
    no media lists the target - is the whole dataset, where the target has no
    positives.
    """
    cell = evaluable_pool(dict(medias), target_category)
    return cell if cell else dict(medias)


@dataclass
class FindCorpus:
    """The corpus a run Finds on beside the withheld half, keyed as the run scores it.

    *clips* is the cell's own medias plus the foreign ones under their new
    ids, so the scorers can read the votes and the corpus from one dict;
    *ids* is the corpus, the withheld half's share first, in scoring order;
    *query_scores* the typed query's cosine of every media in it, on a
    typed-query run.
    """

    clips: dict[int, dict[str, Any]]
    ids: list[int]
    query_scores: "dict[int, float] | None"
    n_withheld: int
    n_foreign: int
    _matrix: "np.ndarray | None" = field(default=None, repr=False)

    def matrix(self) -> "np.ndarray":
        """The corpus's whole-image vectors, in :attr:`ids` order: what both flags read.

        The app's two routes stack ``media_embedding`` of every media of the
        dataset they ask about, so the flags do here.  Built once, on the first
        snapshot click.
        """
        if self._matrix is None:
            self._matrix = np.stack(
                [np.asarray(media_embedding(self.clips[cid]), dtype=np.float32) for cid in self.ids]
            )
        return self._matrix


def _draw(foreign_ids: list[int], mix: "float | str", n_withheld: int, seed: int) -> list[int]:
    """The foreign medias a mix takes: all of them under ``only``, else a seeded draw of the share's count."""
    if mix == FIND_ONLY:
        return foreign_ids
    n_take = min(len(foreign_ids), int(round(float(mix) * n_withheld)))
    if n_take >= len(foreign_ids):
        return foreign_ids
    rng = np.random.default_rng([int(seed), _MIX_SALT])
    pick = np.sort(rng.choice(len(foreign_ids), size=n_take, replace=False))
    return [foreign_ids[int(i)] for i in pick]


def _check_embedder(
    clips_dict: Mapping[int, dict[str, Any]], taken: list[dict[str, Any]], *, region_aware: bool
) -> None:
    """Refuse foreign medias that are not in the run's embedder: another vector width, or patch grids on one side."""
    if not taken:
        return
    width = np.asarray(media_embedding(next(iter(clips_dict.values())))).shape[-1]
    foreign_width = np.asarray(media_embedding(taken[0])).shape[-1]
    if foreign_width != width:
        raise ValueError(
            f"the Find corpus is not in the run's embedder: its vectors are {foreign_width} wide, the run's {width}"
        )
    if any(m.get("patch_grid") is not None for m in taken) != region_aware:
        raise ValueError(
            "the Find corpus is not in the run's embedder: patch grids on "
            + ("the run's medias only" if region_aware else "the Find corpus's medias only")
        )


def _corpus_query_scores(
    withheld: list[int],
    taken: list[int],
    base: int,
    query_scores: "Mapping[int, float] | None",
    foreign_query_scores: "Mapping[int, float] | None",
) -> "dict[int, float] | None":
    """The typed query's cosine of every media in the corpus, under the ids it is scored by; ``None`` off a query run."""
    if query_scores is None:
        return None
    out = {cid: float(query_scores[cid]) for cid in withheld}
    if not taken:
        return out
    if foreign_query_scores is None:
        raise ValueError(
            "a typed-query run's Test gives the typed query's set, so a Find corpus with foreign medias needs "
            "find_corpus_scores: the typed query's cosine over them"
        )
    missing = [cid for cid in taken if cid not in foreign_query_scores]
    if missing:
        raise ValueError(f"find_corpus_scores lacks {len(missing)} of the Find corpus's medias, e.g. {missing[:5]}")
    out.update({base + k: float(foreign_query_scores[cid]) for k, cid in enumerate(taken)})
    return out


def build_find_corpus(
    clips_dict: Mapping[int, dict[str, Any]],
    test_ids: Sequence[int],
    foreign: Mapping[int, dict[str, Any]],
    target_category: str,
    *,
    mix: "float | str",
    seed: int,
    region_aware: bool,
    query_scores: "Mapping[int, float] | None" = None,
    foreign_query_scores: "Mapping[int, float] | None" = None,
) -> FindCorpus:
    """The Find corpus of a run on *clips_dict* whose withheld half is *test_ids*.

    *foreign* is the second dataset's medias, in the run's embedder; its cell
    of the same name (:func:`find_cell`) is what the mix (already parsed,
    :func:`parse_find_mix`) draws from.  The draw is seeded off *seed* alone,
    on a generator of its own.  The foreign medias are copied under ids past
    the cell's, their ``"id"`` field with them (the HAC styles key a cache on
    it).  *query_scores* is the run's typed-query sort, when its Test can give
    the typed query's set, and then *foreign_query_scores* - the same query's
    cosine over *foreign* - is required for a corpus holding foreign medias.

    Refused when the corpus would be empty, or *foreign* is not in the run's
    embedder: another vector width, or patch grids on one side only.
    """
    cell = find_cell(foreign, target_category)
    withheld = [] if mix == FIND_ONLY else list(test_ids)
    taken = _draw(sorted(cell), mix, len(withheld), seed)
    if not withheld and not taken:
        raise ValueError(f"the Find corpus holds no media (mix {mix!r}, {len(cell)} in the foreign cell)")
    _check_embedder(clips_dict, [cell[cid] for cid in taken], region_aware=region_aware)
    base = max(int(cid) for cid in clips_dict) + 1
    corpus_query = _corpus_query_scores(withheld, taken, base, query_scores, foreign_query_scores)
    rekeyed = {base + k: {**cell[cid], "id": base + k} for k, cid in enumerate(taken)}
    return FindCorpus(
        clips={**clips_dict, **rekeyed},
        ids=[*withheld, *rekeyed],
        query_scores=corpus_query,
        n_withheld=len(withheld),
        n_foreign=len(taken),
    )


def blank_find_columns() -> dict[str, float]:
    """Every Find and flag column, NaN: a row with no Find corpus, or off a snapshot click for the flags."""
    return dict.fromkeys((*FIND_CORPUS_COLUMNS, *FIND_FLAG_COLUMNS), math.nan)


def find_columns(metrics: Mapping[str, Any]) -> dict[str, float]:
    """The Find corpus's metric row, as :data:`~vtscore.eval.voting_columns.FIND_CORPUS_COLUMNS`.

    *metrics* is the row the in-corpus scorer (``_evaluate_on_test``) returns
    for the Find corpus, so a column means here what its in-corpus namesake
    means on the withheld half.
    """
    out: dict[str, float] = {f"find_{c}": metrics[c] for c in FBETA_COLUMNS}
    out.update(
        {
            "find_precision": metrics["precision"],
            "find_recall": metrics["recall"],
            "find_n_flagged": metrics["n_flagged"],
            "find_average_precision": metrics["average_precision"],
            "find_n_pos": metrics["n_test_pos"],
            "find_n_neg": metrics["n_test_neg"],
        }
    )
    return out


def training_domain_flags(atlas: Any, matrix: "np.ndarray", *, patch_space: bool) -> dict[str, float]:
    """The Training-domain overlap chip's reading of *matrix* (#4961's warning), NaN where the app refuses it.

    The app's route (``vtsearch.training_domain.domain_shift_against``) runs
    :func:`~vtscore.coverage.atlas.domain_shift_report` at its default alpha
    against the reference dataset's coverage atlas, over the whole-image
    vectors of every media of the dataset searched.  The harness's reference
    is the sim pool and its atlas the one it builds there for the New phase.
    Refused, as the app refuses, with no atlas, no vectors, or on a patch
    embedder, where the guard does not separate in-domain from out (#3329).
    """
    nan = math.nan
    if atlas is None or patch_space or matrix.shape[0] == 0:
        return {"shift_frac_atypical": nan, "shift_z": nan, "shift_shifted": nan}
    from vtscore.coverage.atlas import domain_shift_report  # noqa: PLC0415

    report = domain_shift_report(atlas, matrix)
    return {
        "shift_frac_atypical": float(report["frac_atypical"]),
        "shift_z": float(report["z_score"]),
        "shift_shifted": 1.0 if report["shifted"] else 0.0,
    }


def vote_vectors(
    good_votes: Sequence[int],
    bad_votes: Sequence[int],
    clips_dict: Mapping[int, dict[str, Any]],
    target_category: str,
    *,
    region_voting: bool,
    style_obj: Any,
    dim: int,
) -> tuple["np.ndarray", "np.ndarray"]:
    """The votes' training rows, Goods and Bads, as the app's evidence route reads its labelset.

    The route builds them with ``build_xy_from_labelset``: a Good's one vector
    and a Bad's flooded image-level + raw-patch negatives on a patch dataset,
    one image-level vector elsewhere.  The harness's trainers assemble the same
    rows, so these are theirs: the style's ``good_vec`` (given the ground-truth
    box under *region_voting*) and ``bad_vecs`` on a style run,
    :func:`~vtscore.eval.step_model.good_training_vec` and the whole-image
    vector on a single-vector one.  An empty class is a ``(0, dim)`` matrix.
    """
    goods: list[np.ndarray] = []
    bads: list[np.ndarray] = []
    for vid in good_votes:
        media = clips_dict[vid]
        if style_obj is not None:
            box = region_box_for_category(media, target_category) if region_voting else None
            goods.append(np.asarray(style_obj.good_vec(media, box), dtype=np.float32))
        else:
            goods.append(np.asarray(good_training_vec(media, target_category, region_voting), dtype=np.float32))
    for vid in bad_votes:
        media = clips_dict[vid]
        if style_obj is not None:
            bads.extend(np.asarray(v, dtype=np.float32) for v in style_obj.bad_vecs(media))
        else:
            bads.append(np.asarray(media_embedding(media), dtype=np.float32))

    def _stack(rows: list[np.ndarray]) -> np.ndarray:
        return np.stack(rows) if rows else np.empty((0, dim), dtype=np.float32)

    return _stack(goods), _stack(bads)


def evidence_flags(
    pos: "np.ndarray", neg: "np.ndarray", matrix: "np.ndarray", predicted_positive: "np.ndarray"
) -> dict[str, float]:
    """The Evidence coverage chip's reading of *matrix* under the votes, NaN where the app has none.

    The app's route (``find_evidence_coverage``) runs
    :func:`~vtscore.detectors.evidence_coverage.evidence_coverage_report` at its
    defaults over the whole-image vector of every scored media, each predicted
    positive where its Find score clears the Find threshold; *pos* / *neg* are
    the labelset's rows (:func:`vote_vectors`).  It reports nothing without
    labelled rows or scored media.
    """
    nan = math.nan
    if pos.shape[0] + neg.shape[0] == 0 or matrix.shape[0] == 0:
        return {
            "evidence_frac_unsupported": nan,
            "evidence_z": nan,
            "evidence_unsupported": nan,
            "evidence_frac_low_trust": nan,
        }
    from vtscore.detectors.evidence_coverage import evidence_coverage_report  # noqa: PLC0415

    report = evidence_coverage_report(pos, neg, matrix, np.asarray(predicted_positive, dtype=bool))
    return {
        "evidence_frac_unsupported": float(report["frac_unsupported"]),
        "evidence_z": float(report["z_score"]),
        "evidence_unsupported": 1.0 if report["unsupported"] else 0.0,
        "evidence_frac_low_trust": float(report["frac_low_trust"]),
    }
