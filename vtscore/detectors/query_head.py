"""The typed query as a detector head: what a typed-query detector gives until it has a trained head (#4758).

Through Autopilot's opening the screen shows the typed query's own set, the
text sort at its line, until the label quota gives a trained head
(:mod:`~vtscore.detectors.label_quota`).  A Find there gave something else:
nothing before the first Good, then the Goods' centroid
(:mod:`~vtscore.detectors.centroid_head`), whose set is worse than the one on
screen.  On the State of the App's Binary opening (COCO Better, 720 sessions),
on the votes under the quota, the typed query's set scores 0.56 / 0.44 / 0.54
F-beta at beta 1/4, 1 and 4 against the centroid's 0.43 / 0.28 / 0.35; over
votes 1-10, giving it from vote 0 is worth +0.08 / +0.09 / +0.14
(``docs/experiments/2026-10-12-find-typed-query-4758``).

So a detector made from a typed query keeps it (its ``text_query``, which the
detector file and the registry already hold), and until its labels meet the
quota, Test, AutoFind, a load, a cold Find, an export and the CLI give the
typed query's set: this head.  The labelset still carries no query (an
exported, imported or CLI-made labelset has none), so a detector without one
gives the centroid and nothing as before.  The query lives on the detector,
not on the labels.

**It is the text sort, as a head.**  A ``Linear(D, 1)`` over the query's unit
vector, built and cut as the centroid head is (:func:`fit_query_head`), with
the line the text sort draws: :func:`~vtscore.training.thresholds.text_sort_cuts`'
display line at the detector's balance (``centroid_cut(rule="text")``), over
every media scored, voted ones included, as the sort counts them.  It scores in
the dataset's **text** space (:func:`query_head_embedder`), which on a
dual-embedder dataset is not the space the detector learns in: every scorer
asks :func:`head_score_embedder` which rows to build.  A dataset with no text
slot cannot embed the query, and gives the centroid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

if TYPE_CHECKING:
    import torch.nn as nn

    from vtscore.detectors.centroid_head import CentroidLine

#: The attribute that marks a ``Linear(D, 1)`` as a typed query's head.
_MARK = "vtsearch_typed_query"
#: The text embedder the head scores in.
_EMBEDDER = "vtsearch_query_embedder"
#: The typed query the head was built from.
_TEXT = "vtsearch_query_text"


def detector_text_query(det_data: "dict[str, Any] | None") -> str:
    """The typed query a detector was made from: its ``text_query``, else its first text example; ``""`` for none."""
    if not det_data:
        return ""
    text = str(det_data.get("text_query") or "").strip()
    if text:
        return text
    for ex in det_data.get("examples") or []:
        if isinstance(ex, dict) and ex.get("type") == "text" and str(ex.get("value") or "").strip():
            return str(ex["value"]).strip()
    return ""


def is_query_head(model: object) -> bool:
    """Whether *model* is a typed query's head."""
    return bool(getattr(model, _MARK, False))


def query_head_embedder(model: object) -> str | None:
    """The text embedder a typed query's head scores in, or ``None`` for any other model."""
    return getattr(model, _EMBEDDER, None) if is_query_head(model) else None


def query_head_text(model: object) -> str | None:
    """The typed query a head was built from, or ``None`` for any other model."""
    return getattr(model, _TEXT, None) if is_query_head(model) else None


def is_quota_head(model: object) -> bool:
    """Whether *model* is a head the label quota gives (the Goods' centroid or a typed query), not a trained one.

    Both carry their own line (``DetectorContext.centroid_line``), which a
    balance change redraws, and neither has a labels line or a fitted cut.
    """
    from vtscore.detectors.centroid_head import is_centroid_head  # noqa: PLC0415

    return is_centroid_head(model) or is_query_head(model)


def head_score_embedder(model: object, default: str | None) -> str | None:
    """The embedder *model* is scored in: a typed query's text embedder, else *default* (the detector's own)."""
    return query_head_embedder(model) or default


def text_embedder_for_snap(snap: "dict[int, dict[str, Any]] | None") -> str | None:
    """The text-slot embedder *snap*'s medias bind, which a typed query is embedded and scored in; ``None`` for none."""
    from vtscore.embedding.binding import slot_embedders_for_snap  # noqa: PLC0415

    return slot_embedders_for_snap(snap)[0] or None


def embed_detector_query(text: str, media_type: str, embedder: str) -> "np.ndarray | None":
    """*text* embedded as the text sort embeds it: by *embedder*, enriched when the user's setting says so."""
    from vtscore.embedding.helpers import embed_text_query  # noqa: PLC0415

    try:
        from vtscore.config import CoreConfig  # noqa: PLC0415

        enrich = bool(CoreConfig.from_settings().enrich_descriptions)
    except Exception:  # noqa: BLE001 - no settings provider (a library caller): the shipped default
        enrich = False
    return embed_text_query(text, media_type, enrich=enrich, embedder_name=embedder)


def query_head(
    query_vec: np.ndarray, cut: float, *, embedder: str, text: str, scale: float | None = None
) -> "nn.Sequential":
    """A ``Linear(D, 1)`` scoring ``scale * (cosine - cut)`` against *query_vec*, marked as a typed query's head."""
    from vtscore.detectors.centroid_head import CENTROID_LOGIT_SCALE, _MARK as _CENTROID_MARK  # noqa: PLC0415
    from vtscore.detectors.centroid_head import centroid_head, goods_centroid  # noqa: PLC0415

    head = centroid_head(goods_centroid([query_vec]), cut, scale=CENTROID_LOGIT_SCALE if scale is None else scale)
    delattr(head, _CENTROID_MARK)
    setattr(head, _MARK, True)
    setattr(head, _EMBEDDER, embedder)
    setattr(head, _TEXT, text)
    return head


def fit_query_head(
    query_vec: np.ndarray,
    score: Callable[["nn.Sequential"], Sequence[float]],
    *,
    beta: float | None,
    embedder: str,
    text: str,
) -> tuple["nn.Sequential", float, "CentroidLine"]:
    """A typed query's head, its threshold, and its line, cut on the corpus *score* scores.

    As :func:`~vtscore.detectors.centroid_head.fit_centroid` fits the Goods'
    centroid, with the query's vector for the centroid and the text sort's own
    rule (``"text"``) for the line, drawn at *beta* over every media *score*
    scores: the sort counts the voted ones too, and so does this.  *score* runs
    a head over the corpus in the text space (*embedder*'s rows).
    """
    from vtscore.detectors.centroid_head import (  # noqa: PLC0415
        CENTROID_THRESHOLD,
        CentroidLine,
        _gap_centre,
        centroid_cut,
    )

    probe = query_head(query_vec, 0.0, embedder=embedder, text=text, scale=1.0)
    s = np.asarray(score(probe), dtype=np.float64)
    s = s[(s > 0.0) & (s < 1.0)]
    cosines = np.log(s) - np.log1p(-s)
    cut = _gap_centre(cosines, centroid_cut(cosines, rule="text", beta=beta))
    head = query_head(query_vec, cut, embedder=embedder, text=text)
    return head, CENTROID_THRESHOLD, CentroidLine(cosines, cut, "text")
