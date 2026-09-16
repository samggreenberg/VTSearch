"""The Stable indicator's arithmetic, shared by the app and the eval harness.

Stable asks one question: *would more labels still change which items the
detector calls positive?*  It is answered by comparing successive detectors'
predictions over the still-unlabeled pool and counting the items that changed
class - a "flip".  This module holds every constant and every line of that
arithmetic, and both consumers call it rather than copying it:

* :func:`vtscore.detectors.labeling_progress._compute_stable_status` (the app's
  ``/api/labeling-status`` indicator, which gates Autopilot's ``hard`` phase),
* :func:`vtscore.eval.autopilot_flow.stable_status` (the harness's simulated
  user, which has to stop where the app stops for a study to mean anything).

Two properties of the rule are deliberate, and both come from issue #3831.

**Only confident flips count against stability.**  When a category is not
separable in the embedding - a pile of dice labeled by whether they rolled at
least 5, given only the number of sides - every detector will put the d8s on
one side of the cut or the other, and a single new label will move them all
back.  Those items flip forever, and a rule that counts every flip never fires:
the app sat in ``hard`` for 127 of 150 clicks on exactly that pool.  A flip is
counted as *confident* only when the item sat clear of the cut - further than
:data:`STABLE_BAND_STD_FRACTION` of the pool's score spread - under **both**
detectors.  An item that wobbles across the cut from just above to just below
is boundary noise, not the detector changing its mind; an item that goes from
clearly-in to clearly-out is.  The band is a fraction of the score *standard
deviation* rather than a fixed width because the shipped head's sigmoid scores
are compressed (a spread of ~0.15 is typical), so any absolute width either
swallows most of the pool or none of it.

**The denominator is the whole pool, not the unlabeled remainder.**  The old
rate divided by the unlabeled count, which shrinks as a haystack is labeled: on
the last fifty items one flip was 2% and blocked green, and on the last one it
was either 0% or 100%.  Dividing by the pool size keeps the thresholds meaning
the same thing at the start of a session and at its end.

Green therefore needs both of: the confident flip rate has settled (average
under :data:`STABLE_RATE_THRESHOLD`, no single step above
:data:`STABLE_MAX_THRESHOLD`), and the *raw* flip rate has stopped falling
(the later half of the window is not below :data:`STABLE_FALLING_RATIO` of the
earlier half) - otherwise the boundary is still sharpening and more labels are
still buying something.  When green is reached with the raw rate still above
the settled threshold over the later half of the window, the status carries
``plateau=True``: the detector has
stopped improving but a fraction of the pool sits in an ambiguity the
embedding cannot resolve, and the UI says so instead of pretending the pool
converged.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

Status = Literal["red", "yellow", "green"]

#: Both Smart and Stable stay red until the labelset has this many of each
#: class - below it there is no detector worth judging.
MIN_PER_CLASS = 5

#: How many stability entries the rule looks back over, and how many it needs
#: before it will say anything but "not enough history".
STABLE_WINDOW = 10
STABLE_MIN_ENTRIES = 5

#: Confident-flip rate over the pool: the window average must be under the
#: first and no single entry at or above the second.
STABLE_RATE_THRESHOLD = 0.005
STABLE_MAX_THRESHOLD = 0.01

#: Half-width of the ambiguity band around the cut, as a fraction of the
#: standard deviation of the scores being compared.  Chosen on the #3831 dice
#: pool: at 0.25 the irreducibly ambiguous groups sit inside the band while a
#: genuinely learning detector's early reclassifications (over a hundred items
#: moving clear across the cut) still count.
STABLE_BAND_STD_FRACTION = 0.25

#: The raw flip rate is "still falling" when the later half of the window
#: averages below this fraction of the earlier half (and is not already under
#: :data:`STABLE_RATE_THRESHOLD`).
STABLE_FALLING_RATIO = 0.5


def ambiguity_band(scores: Iterable[float]) -> float:
    """Half-width of the ambiguity band for one detector's *scores*.

    :data:`STABLE_BAND_STD_FRACTION` of their standard deviation; ``0.0`` for
    an empty or constant set, where every flip is a confident one by
    definition (a pool at one score has no "near the cut" to hide in).
    """
    arr = np.fromiter((float(s) for s in scores), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return STABLE_BAND_STD_FRACTION * float(arr.std())


@dataclass(frozen=True)
class ScoredSnapshot:
    """One detector's scores over the pool it is compared on, with its cut.

    *scores* map each still-unlabeled item to its served score, *threshold* is
    the cut the app applied to them, and *band* the ambiguity half-width
    :func:`ambiguity_band` derived from those scores.  Built once per step and
    held as the baseline the next step's detector is compared against.
    """

    scores: dict[int, float]
    threshold: float
    band: float

    @classmethod
    def from_scores(cls, scores: Mapping[int, float], threshold: float) -> "ScoredSnapshot":
        scores = {int(cid): float(s) for cid, s in scores.items()}
        return cls(scores=scores, threshold=float(threshold), band=ambiguity_band(scores.values()))

    def predicted(self, cid: int) -> int:
        """The class this detector assigned *cid*: ``1`` at or above the cut."""
        return 1 if self.scores[cid] >= self.threshold else 0

    def confident(self, cid: int) -> bool:
        """Whether *cid* sat clear of the cut under this detector."""
        return abs(self.scores[cid] - self.threshold) >= self.band


def count_flips(prev: ScoredSnapshot, cur: ScoredSnapshot) -> tuple[int, int]:
    """``(num_flips, num_confident_flips)`` between two consecutive detectors.

    Counted over the items both snapshots score - an item labeled in between
    has left the pool and is nobody's flip.  A flip is confident when the item
    was clear of the cut under *both* detectors (see the module docstring).
    """
    flips = 0
    confident = 0
    for cid in cur.scores.keys() & prev.scores.keys():
        if cur.predicted(cid) != prev.predicted(cid):
            flips += 1
            if cur.confident(cid) and prev.confident(cid):
                confident += 1
    return flips, confident


def stability_entry(prev: ScoredSnapshot, cur: ScoredSnapshot, num_pool: int) -> dict[str, Any]:
    """The per-step record :func:`stable_status_from_entries` consumes.

    ``num_pool`` is the size of the whole pool the detector scores (labeled
    items included) - the denominator every rate is taken over - and
    ``num_unlabeled`` the number of items actually compared, kept for the
    progress chart.
    """
    num_flips, num_confident = count_flips(prev, cur)
    return {
        "num_flips": num_flips,
        "num_confident_flips": num_confident,
        "num_unlabeled": len(cur.scores),
        "num_pool": int(num_pool),
    }


def _rates(entries: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    return [e[key] / e["num_pool"] if e["num_pool"] > 0 else 0.0 for e in entries]


def stable_status_from_entries(entries: Sequence[Mapping[str, Any]], good: int, bad: int) -> dict[str, Any]:
    """The Stable indicator over a history of :func:`stability_entry` records.

    Returns the ``/api/labeling-status`` sub-object: ``status`` (red / yellow /
    green) and ``reason``, plus ``avg_flip_rate`` (raw), ``avg_confident_flip_rate``
    and ``plateau`` once there is enough history to compute them.  ``plateau``
    is true only on green, and means the raw flip rate over the later half of
    the window is still above :data:`STABLE_RATE_THRESHOLD` - the pool has an
    ambiguous fringe the detector cannot settle - which the UI reports beside
    the green light.

    Three further numbers ride along, and they are the *other* quantities green
    turns on: ``max_confident_flip_rate`` (the worst single step in the window,
    which :data:`STABLE_MAX_THRESHOLD` caps) and ``flip_rate_early`` /
    ``flip_rate_late`` (the two halves of the raw-rate window, which the
    still-falling test compares).  Without them a reader has the status and one
    of the three gates behind it, and cannot say how close a yellow was to
    green or which condition was binding - which is what issue #3560 asked the
    eval harness to record per step.  All five rates are rounded to six places
    rather than four: on a haystack of tens of thousands a single flip is finer
    than 1e-4, and these are reporting numbers that no rule reads back.
    """
    if good < MIN_PER_CLASS or bad < MIN_PER_CLASS:
        return {
            "status": "red",
            "reason": f"Need at least {MIN_PER_CLASS} good and {MIN_PER_CLASS} bad. Currently {good}g, {bad}b.",
        }
    if len(entries) < STABLE_MIN_ENTRIES:
        return {"status": "yellow", "reason": "Not enough history to assess prediction stability."}

    recent = list(entries[-STABLE_WINDOW:])
    raw = _rates(recent, "num_flips")
    confident = _rates(recent, "num_confident_flips")
    avg_raw = sum(raw) / len(raw)
    avg_confident = sum(confident) / len(confident)
    max_confident = max(confident)
    half = len(raw) // 2
    early = sum(raw[:half]) / half
    late = sum(raw[half:]) / (len(raw) - half)

    extras = {
        "avg_flip_rate": round(avg_raw, 6),
        "avg_confident_flip_rate": round(avg_confident, 6),
        "max_confident_flip_rate": round(max_confident, 6),
        "flip_rate_early": round(early, 6),
        "flip_rate_late": round(late, 6),
        "plateau": False,
    }
    if not (avg_confident < STABLE_RATE_THRESHOLD and max_confident < STABLE_MAX_THRESHOLD):
        return {
            "status": "yellow",
            "reason": f"Average {avg_confident:.1%} of items confidently changing class in recent steps.",
            **extras,
        }
    if late >= STABLE_RATE_THRESHOLD and late < STABLE_FALLING_RATIO * early:
        return {
            "status": "yellow",
            "reason": (
                f"Predictions near the cut are still settling: flips fell from {early:.1%} to {late:.1%} "
                "of items over recent steps."
            ),
            **extras,
        }
    if late >= STABLE_RATE_THRESHOLD:
        # Judged on the later half of the window - where the run *is* - rather
        # than the window average, so wobble that has since died away reads as
        # convergence and not as a plateau.
        extras["plateau"] = True
        return {
            "status": "green",
            "reason": (
                f"Confident predictions have stabilized. {late:.1%} of items near the cut still change "
                "class between retrains, but the rate has stopped falling, so the remaining ambiguity "
                "looks irreducible in this embedding."
            ),
            **extras,
        }
    return {"status": "green", "reason": "Predictions have stabilized.", **extras}
