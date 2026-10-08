"""The label quota: how many Goods and Bads a labelset needs before it gets a trained head (#4643).

A linear SVM fitted to three Goods and one Bad is a detector in name only, and
nothing stopped the app handing one out: votes save on every click, Test,
AutoFind and an export all retrain from the labelset, so the opening's first
head was a click away (the click-4 dip on the State of the App viewer, #4640).
So what a labelset gives depends on its counts, and on nothing else - not on
where it came from, not on whether an Autopilot session made it:

======================================  =========================================
Labels                                   What Test, AutoFind and a load give
======================================  =========================================
no Goods                                 nothing to sort toward (:data:`TIER_NONE`)
a Good, but under either quota           the Goods' centroid (:data:`TIER_CENTROID`)
:data:`GOOD_QUOTA` Goods and             the trained head (:data:`TIER_TRAINED`)
:data:`BAD_QUOTA` Bads
======================================  =========================================

The centroid is :mod:`vtscore.detectors.centroid_head`.  There is no tier keyed
on the typed query: a labelset does not carry it (an exported, an imported and
a CLI-made labelset have none), so a tier that read it would not travel with
the labels.

**Why 3 and 4.**  They are Autopilot's own quorum, ``goodToStart`` /
``badToStart`` in ``frontend/src/app/services/autopilot-state.service.ts``
(``GOOD_TARGET`` / ``BAD_TARGET`` in :mod:`vtscore.eval.autopilot_flow`): the
counts at which the opening has always judged the labels enough to train on.
Until #4282 it handed over to the learned sort right there.  The "more" walk
#4282 added after them mines positives because the typed query's sort is still
paying, not because a head at 3 and 4 is unready, so it is no reason to raise
the quota: its 20 Goods would leave a rare class (around 11 positives at 0.1%
prevalence, #4222) with no trained head at all.  The labelset cannot see the
opening's dry stop, so the rule only approximates the session's own hand-off
(``app_trained`` in the eval harness).  ``tests_lib`` pins the two constants to
Autopilot's.

The counts are of labels the detector can actually use: a Good or Bad that
resolves to a vector.  On the active dataset that is every one of them; an
element whose origin cannot be resolved does not count toward the quota, so a
head is never fitted to fewer labels than the quota names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Literal

if TYPE_CHECKING:
    from vtscore.datasets.labelset import LabelSet

#: Goods a labelset needs before it gets a trained head: Autopilot's ``goodToStart``.
GOOD_QUOTA = 3

#: Bads a labelset needs before it gets a trained head: Autopilot's ``badToStart``.
BAD_QUOTA = 4

#: No Goods: there is nothing to sort toward, so Test is refused.
TIER_NONE = "none"

#: At least one Good, under either quota: the Goods' centroid, cut with a GMM.
TIER_CENTROID = "centroid"

#: Both quotas met: the trained head (the linear SVM).
TIER_TRAINED = "trained"

DetectorTier = Literal["none", "centroid", "trained"]


@dataclass(frozen=True)
class LabelQuota:
    """A labelset's Good and Bad counts, the tier they give, and what is still owed."""

    n_good: int
    n_bad: int

    @property
    def tier(self) -> DetectorTier:
        """Which detector these counts give: :data:`TIER_NONE`, :data:`TIER_CENTROID` or :data:`TIER_TRAINED`."""
        if self.n_good <= 0:
            return TIER_NONE
        if self.n_good >= GOOD_QUOTA and self.n_bad >= BAD_QUOTA:
            return TIER_TRAINED
        return TIER_CENTROID

    @property
    def goods_owed(self) -> int:
        """Goods still needed before the trained head (0 once met)."""
        return max(0, GOOD_QUOTA - self.n_good)

    @property
    def bads_owed(self) -> int:
        """Bads still needed before the trained head (0 once met)."""
        return max(0, BAD_QUOTA - self.n_bad)

    def as_dict(self) -> dict[str, Any]:
        """The JSON shape every response that reports a tier carries."""
        return {
            "tier": self.tier,
            "n_good": int(self.n_good),
            "n_bad": int(self.n_bad),
            "goods_owed": self.goods_owed,
            "bads_owed": self.bads_owed,
            "good_quota": GOOD_QUOTA,
            "bad_quota": BAD_QUOTA,
        }


def label_quota(n_good: int, n_bad: int) -> LabelQuota:
    """The :class:`LabelQuota` for *n_good* Goods and *n_bad* Bads."""
    return LabelQuota(int(n_good), int(n_bad))


def labelset_quota(labelset: "LabelSet | None") -> LabelQuota:
    """The quota a labelset's elements make, counting every Good and Bad it holds.

    The count before resolution: what the user labelled.  Training counts what
    resolved (:func:`quota_from_groups`), which is the same on the active
    dataset and can be fewer when an origin is out of reach.
    """
    if labelset is None:
        return LabelQuota(0, 0)
    n_good = sum(1 for el in labelset.elements if el.label == "good")
    n_bad = sum(1 for el in labelset.elements if el.label == "bad")
    return LabelQuota(n_good, n_bad)


def served_quota(model: object, labelset: "LabelSet | None") -> dict[str, Any]:
    """What a response reports about the detector it served: :meth:`LabelQuota.as_dict` of *labelset*.

    ``tier`` is read off *model* rather than the counts, because it is what was
    served: a Good or Bad whose origin did not resolve counts in *labelset* but
    not toward the quota, so the two can differ.  ``None`` model, ``none`` tier.
    """
    from vtscore.detectors.centroid_head import is_centroid_head  # noqa: PLC0415

    out = labelset_quota(labelset).as_dict()
    if model is None:
        out["tier"] = TIER_NONE
    else:
        out["tier"] = TIER_CENTROID if is_centroid_head(model) else TIER_TRAINED
    return out


def quota_from_groups(groups: Iterable[tuple[str, Any]]) -> LabelQuota:
    """The quota of the labels a training set actually holds, from its per-row bag ids.

    *groups* is :func:`~vtscore.detectors.labelset_training.build_xy_from_labelset`'s
    ``("g" | "b", element id)`` per row.  A Bad that floods its patch rows is
    one bag of many rows, so the count is of distinct bags: labels, not rows.
    """
    bags = set(groups)
    return LabelQuota(
        sum(1 for kind, _ in bags if kind == "g"),
        sum(1 for kind, _ in bags if kind == "b"),
    )
