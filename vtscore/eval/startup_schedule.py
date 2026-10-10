"""Parameterised Autopilot **openings** for the voting-iterations harness (#3267).

Autopilot's opening is what decides how many positives a run has when its first
learned sort happens, and a Good-starved run is the one that fails.  Today the
opening is fixed: vote the **top** of the seed sort until three positives exist,
then vote that sort's **cutoff** until four negatives do, then hand over to the
learned Hard sort forever.  This module makes that opening a *parameter* so a
study can ask whether a different one mines better positives.

**The unifying observation** (issue #3267) is that both of today's opening
phases are the same operation - the app's rank-space ``hard`` select against a
cut drawn on the seed sort - at two different cuts:

* the Good phase's ``top`` select is a ``hard`` select against a cut placed
  *above every score*, so the first at-or-below-cut rank is 0 and the pick is the
  top of the sort;
* the Bad phase's cut is that sort's own fitted 2-component GMM, split at the
  production midpoint.

So "Text-Good" really is "Text-Hard at a very exclusive cut", and the whole
opening collapses to a list of rounds, each *how many clicks* to spend and
*where on the seed sort* to spend them.

**Where** is named the way the rest of the app names it: an **Inclusion**.  The
seed sort ships an inclusion-independent cut (:func:`~vtscore.training.
thresholds.calculate_gmm_threshold`, the midpoint of the fitted components), but
the fit underneath it is the same object every inclusion-aware cut rule reads,
so the sort *can* be split at an inclusion - the prior-agnostic split at ``k=0``
and an inclusion-biased split either side of it (:func:`~vtscore.training.
thresholds.gmm_cut_from_fit`'s ``rate`` rule at
:func:`~vtscore.training.thresholds.inclusion_cost_weights`).  Negative ``k``
prices a false alarm higher, raises the cut, and moves the pick **up** the
ranking toward the positives - the same direction, and the same reason, as
:data:`~vtscore.training.thresholds.ACQUISITION_INCLUSION_OFFSET` on the learned
sort.  ``k`` far enough below zero converges on ``top``, which is the issue's
"Text-Good is Text-Hard(-100)" made literal.

**The spec.**  A schedule is a comma-separated list of
``<stop><n>[+dry<m>/<w>]@<cut>`` rounds:

============  ==================================================================
``g3``        stay until **3 goods** exist (a global count, as in the app)
``b4``        stay until **4 bads** exist (likewise)
``n8``        stay for **8 clicks**, whatever they turn out to be
``+dry1/8``   ...or until this round's last **8** picks yielded fewer than
              **1** good - the walk has run dry (``g`` and ``n`` rounds only)
``@top``      cut above every score - the top of the sort (today's Good phase)
``@mid``      the shipped GMM midpoint (today's Bad phase, and every cosine sort)
``@k-3``      the fitted GMM split at inclusion ``-3``; ``@k0`` is prior-agnostic
``@q0.05``    cut at the sort's own 5th percentile, by rank
============  ==================================================================

**Why ``q`` exists next to ``k``.**  ``k`` is the arm that could *ship* - the app
has an Inclusion knob and no rank-position knob - but how far a given ``k`` moves
the pick is a property of the fitted mixture, not of ``k``.  On a steep sort the
whole usable inclusion range can land inside a couple of rank percent, which
makes an arm grid that looks well spread in ``k`` almost inert in the space the
picks actually live in.  ``q`` names the rank position directly, so a study can
establish whether *position* is the mechanism before asking whether ``k`` is a
usable handle on it.  Read :data:`~vtscore.eval.voting_columns.PICK_COLUMNS`'
``startup_cut_percentile`` to see where each round's cut really landed; a ``k``
family that does not separate there has not been tested, whatever its spec says.

**Why ``+dry`` exists.**  A fixed Good target is the wrong length in one
pool or the other (#4222): at 0.44% prevalence walking the text sort until 20
goods beat today's 3, but at 0.1% there are ~11 positives to find, so ``g10``
and up never met their target, walked the text sort for the whole session, and
never handed over to a learned sort.  ``g20+dry1/8@top`` keeps the long walk
where the text sort is still paying and stops it where it has stopped paying:
the round ends at the 20th good *or* once its last 8 picks held none.  The
window counts this round's own picks only, and cannot fire until it is full,
so a round always spends at least ``w`` clicks before it can call itself dry.
It counts goods, which is why a ``b`` round refuses it: a round hunting
negatives would call itself dry on exactly the picks it wants.

:data:`PRODUCTION_STARTUP` spells today's opening in that grammar, and
``tests_lib/detectors/test_startup_schedule.py`` pins it against the ported
phase machine: running it must reproduce the default flow pick for pick.  That
pinning is the point - the default arm has to *be* the app (see "The Eval
Default Arm IS the App" in ``docs/EVAL.md``), so a study measures deviations
from what users get rather than from a second, drifting opening.
"""

from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass, replace
from typing import Literal, Optional, Sequence

#: Today's opening, in the grammar above: the top of the seed sort until three
#: positives, then that sort's midpoint cut until four negatives, then Hard.
#: Equal by construction to ``GOOD_TARGET`` / ``BAD_TARGET`` / ``MORE_TARGET``
#: and the Sort+Select pairing in
#: :func:`vtscore.eval.al_strategies._select_phase_faithful`; pinned against all
#: of them by ``tests_lib/detectors/test_startup_schedule.py`` and by
#: ``scripts/check-eval-app-sync.py``'s ``autopilot.startup_default`` mirror.
#: From #4282 to #4740 it was ``g3@top,b4@mid,g20+dry1/16@top``, the ``more``
#: walk; #4671 found it lost after vote 23 on the objective.
PRODUCTION_STARTUP = "g3@top,b4@mid"

StopKind = Literal["good", "bad", "clicks"]
CutKind = Literal["top", "mid", "rate", "quantile"]

_ROUND_RE = re.compile(r"^([gbn])(\d+)(?:\+dry(\d+)/(\d+))?@(top|mid|k-?\d+|q0?\.\d+|q1\.0+|q0|q1)$")

#: Phase names a schedule produces: ``s0``, ``s1``, ...  Deliberately outside
#: :data:`~vtscore.eval.autopilot_flow.TRAINED_PHASES` - a startup round is on
#: the seed sort, so the app would have no detector on screen.
_PHASE_PREFIX = "s"


def is_startup_phase(phase: str) -> bool:
    """Whether *phase* is one of a schedule's rounds (``s0``, ``s1``, ...)."""
    return len(phase) > 1 and phase[0] == _PHASE_PREFIX and phase[1:].isdigit()


@dataclass(frozen=True)
class StartupRound:
    """One round of an opening: how long to stay, and where on the seed sort."""

    stop: StopKind
    """What ends the round - ``good``/``bad`` count a **global** vote total (the
    app's own rule: its Good phase ends on the third positive however many
    negatives the top of the sort happened to hand back), ``clicks`` counts
    votes spent **in this round**."""

    n: int
    """The count *stop* is compared against."""

    cut: CutKind
    """Which cut the round's rank-space ``hard`` select measures against."""

    k: int = 0
    """Inclusion for ``cut == "rate"``; ignored otherwise."""

    q: float = 0.0
    """Rank quantile of the seed sort for ``cut == "quantile"`` (0 = the top);
    ignored otherwise."""

    dry_goods: int = 0
    """The ``m`` of a ``+dry<m>/<w>`` stop: the round also ends once its last
    :attr:`dry_window` picks held **fewer** than this many goods.  Meaningless
    while :attr:`dry_window` is 0."""

    dry_window: int = 0
    """The ``w`` of a ``+dry<m>/<w>`` stop, counted in this round's own picks;
    ``0`` means the round has no dry stop, which is every round but #4222's."""

    def spec(self) -> str:
        """Round-trip this round back to its spec string."""
        stop = {"good": "g", "bad": "b", "clicks": "n"}[self.stop]
        if self.cut in ("top", "mid"):
            cut = self.cut
        elif self.cut == "rate":
            cut = f"k{self.k}"
        else:
            cut = f"q{self.q}"
        dry = f"+dry{self.dry_goods}/{self.dry_window}" if self.dry_window else ""
        return f"{stop}{self.n}{dry}@{cut}"


def parse_startup_schedule(spec: str) -> tuple[StartupRound, ...]:
    """Parse a schedule *spec* into its rounds; raises ``ValueError`` on junk.

    Deliberately strict.  A silently-misparsed arm is a study that measures
    something other than what its launch script says it measures, and the
    schedule is the only thing that distinguishes one arm from the next here.
    """
    rounds = tuple(_parse_round(raw.strip(), spec) for raw in spec.split(",") if raw.strip())
    if not rounds:
        raise ValueError(f"empty startup schedule {spec!r}")
    return rounds


def _parse_round(token: str, spec: str) -> StartupRound:
    m = _ROUND_RE.match(token)
    if m is None:
        raise ValueError(
            f"bad startup round {token!r} in schedule {spec!r}; "
            "expected <g|b|n><count>[+dry<m>/<w>]@<top|mid|k[-]N|q<frac>>, "
            "e.g. 'g3@top', 'n8@k-3', 'n8@q0.05' or 'g20+dry1/8@top'"
        )
    stop_letter, count, cut = m.group(1), int(m.group(2)), m.group(5)
    if count <= 0:
        raise ValueError(f"round {token!r} has a non-positive count")
    stop: StopKind = {"g": "good", "b": "bad", "n": "clicks"}[stop_letter]  # type: ignore[assignment]
    if cut.startswith("k"):
        rnd = StartupRound(stop=stop, n=count, cut="rate", k=int(cut[1:]))
    elif cut.startswith("q"):
        rnd = StartupRound(stop=stop, n=count, cut="quantile", q=float(cut[1:]))
    else:
        rnd = StartupRound(stop=stop, n=count, cut=cut)  # type: ignore[arg-type]
    if m.group(3) is None:
        return rnd
    dry_goods, dry_window = int(m.group(3)), int(m.group(4))
    if stop == "bad":
        raise ValueError(
            f"round {token!r} puts a dry stop on a bad round; the dry stop counts "
            "goods, so a round hunting negatives would end on the picks it wants"
        )
    if not 0 < dry_goods <= dry_window:
        raise ValueError(
            f"round {token!r} needs 0 < m <= w in +dry<m>/<w>: with m = 0 the round "
            "could never run dry, and with m > w it is dry the moment its window fills"
        )
    return replace(rnd, dry_goods=dry_goods, dry_window=dry_window)


def round_cut(scores: Sequence[float], rnd: StartupRound) -> float:
    """The cut *rnd* samples against, over the seed sort's score distribution.

    *scores* is the **whole** sort, labelled rows included: the app fits its
    cosine-sort GMM over every score on screen and does not refit as votes
    accumulate, so a schedule's cuts are constants of the run.

    ``top`` returns ``+inf``, which is not a sentinel but the literal statement
    of the rule: with the cut above every score, the first rank at-or-below it
    is 0 and :func:`~vtscore.eval.al_strategies._hard_pick_by_index` returns the
    top unlabelled item - exactly what the Good phase's ``top`` select does.
    """
    if rnd.cut == "top":
        return math.inf
    from vtscore.training.thresholds import (  # noqa: PLC0415
        calculate_gmm_threshold,
        fit_score_gmm,
        gmm_cut_from_fit,
        gmm_fit_array,
        inclusion_cost_weights,
        text_sort_acquisition_threshold,
    )

    values = list(scores)
    if rnd.cut == "quantile":
        # A rank position, named directly.  Descending, so q=0 is the top of the
        # sort and the cut is the score at that depth.
        import numpy as np  # noqa: PLC0415

        return float(np.quantile(np.asarray(values, dtype=np.float64), 1.0 - rnd.q))
    if rnd.cut == "mid":
        # The app's typed-query *acquisition* cut, called rather than re-derived:
        # a schedule always runs on the text sort (``seed_scores``), and the Bad
        # phase it parameterises samples at the cut the text route sends as
        # ``acq_threshold`` - the GMM midpoint under every display rule (#4136),
        # not the guarded line the route paints green (#3826).
        return float(text_sort_acquisition_threshold(values))
    fit = fit_score_gmm(gmm_fit_array(values))
    if fit is None:
        # Same fallback the shipped cut takes on an unfittable distribution -
        # there is no mixture to tilt, so an inclusion has nothing to price.
        return float(calculate_gmm_threshold(values))
    wf, wn = inclusion_cost_weights(rnd.k)
    cut, _kind = gmm_cut_from_fit(fit, "rate", wf, wn)
    return float(cut) if math.isfinite(cut) else float(calculate_gmm_threshold(values))


class StartupState:
    """Where a trajectory is in its opening, and what ends the current round.

    Advanced once per vote by :class:`~vtscore.eval.autopilot_flow.AutopilotFlow`,
    which owns the phase after the schedule runs out.
    """

    def __init__(self, rounds: Sequence[StartupRound]):
        if not rounds:
            raise ValueError("a startup schedule needs at least one round")
        self.rounds: tuple[StartupRound, ...] = tuple(rounds)
        self.index = 0
        #: Clicks spent in the current round, for its ``clicks`` stop rule.
        self.clicks_in_round = 0
        #: Clicks spent past the end of the schedule because one vote class was
        #: still empty (see :meth:`advance`).  A non-zero value on an arm is a
        #: finding, not noise: the opening as written did not produce a
        #: trainable pair and the harness had to keep voting to get one.
        self.extended_clicks = 0
        self._held_for_quorum = False
        #: The current round's last ``dry_window`` outcomes (``True`` = good),
        #: for its dry stop.  Rebuilt on entering each round, so the window
        #: never reaches back into another round's picks.
        self._recent: deque[bool] = deque()
        self._enter_round()

    @property
    def done(self) -> bool:
        return self.index >= len(self.rounds)

    @property
    def held_for_quorum(self) -> bool:
        """Whether the schedule is spent but one vote class is still empty.

        Public because it has to reach the **rows**.  ``extended_clicks``
        already counted these, but it lived only on this object, so nothing
        downstream could tell "16 opening clicks, as written" from "16 as
        written and 184 more waiting for a first positive" - the two look
        identical in a pick log, and the second is this study's headline
        failure mode rather than a footnote.  It also silently broke the
        length-matched control: an arm that spends 16 clicks by design is only
        comparable to `flat_mid` if `flat_mid` really spent 16.
        """
        return self._held_for_quorum

    def current(self) -> Optional[StartupRound]:
        """The round now being voted, or ``None`` once the schedule is spent."""
        return None if self.done else self.rounds[self.index]

    def phase_name(self) -> str:
        """``s0``, ``s1``, ... - the phase label rows carry for this round."""
        return f"{_PHASE_PREFIX}{min(self.index, len(self.rounds) - 1)}"

    def _enter_round(self) -> None:
        self.clicks_in_round = 0
        rnd = self.current()
        self._recent = deque(maxlen=rnd.dry_window if rnd is not None and rnd.dry_window else None)

    def ran_dry(self) -> bool:
        """Whether the current round's dry stop has fired.

        Only on a full window: a round that has cast fewer than ``w`` picks has
        not yet seen enough of the sort to call it dry, however those picks went.
        """
        rnd = self.current()
        if rnd is None or not rnd.dry_window or len(self._recent) < rnd.dry_window:
            return False
        return sum(self._recent) < rnd.dry_goods

    def on_click(self, good: Optional[bool] = None) -> None:
        """Record one vote against the current round's click budget.

        *good* is the vote's outcome.  Only a round with a dry stop reads it,
        and that round refuses a click without one: a dry stop that silently
        counted nothing would never fire, and the arm would run as the plain
        ``g``/``n`` round its spec says it is not.
        """
        if self.done or self._held_for_quorum:
            self.extended_clicks += 1
            return
        self.clicks_in_round += 1
        rnd = self.rounds[self.index]
        if rnd.dry_window:
            if good is None:
                raise ValueError(f"round {rnd.spec()!r} has a dry stop, so every click needs its outcome")
            self._recent.append(bool(good))

    def advance(self, good_count: int, bad_count: int, remaining_unlabeled: float) -> None:
        """Finish any rounds whose stop condition is now met.

        Each stop is capped by what the pool could still supply, mirroring
        :func:`~vtscore.eval.autopilot_flow.next_phase`'s target capping, so a
        round can never strand a trajectory on a collection too small to satisfy
        it.

        The schedule as a whole will **not** finish while one vote class is
        still empty and the pool could still fill it: the app's opening exists
        to produce a trainable pair, and handing a learned Hard sort a
        one-class labelset would leave the selector picking at random and make
        the arm uninterpretable.  Those extra clicks stay on the last round and
        are counted in :attr:`extended_clicks`.

        A round with a dry stop also ends when :meth:`ran_dry`, whichever of the
        two comes first; the quorum hold applies to it all the same.
        """
        while not self.done:
            rnd = self.rounds[self.index]
            if not (self._target_met(rnd, good_count, bad_count, remaining_unlabeled) or self.ran_dry()):
                return
            if self.index == len(self.rounds) - 1 and remaining_unlabeled > 0 and (good_count == 0 or bad_count == 0):
                self._held_for_quorum = True
                return
            self._held_for_quorum = False
            self.index += 1
            self._enter_round()

    def _target_met(self, rnd: StartupRound, good_count: int, bad_count: int, remaining_unlabeled: float) -> bool:
        if rnd.stop == "good":
            return good_count >= min(rnd.n, good_count + remaining_unlabeled)
        if rnd.stop == "bad":
            return bad_count >= min(rnd.n, bad_count + remaining_unlabeled)
        return self.clicks_in_round >= rnd.n or remaining_unlabeled <= 0
