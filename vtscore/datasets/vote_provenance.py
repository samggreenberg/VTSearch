"""Vote surfacing provenance - how a label came to be in front of the user.

Every vote is a *sample*, and the sampler is the detector's own sort: the
most biased one available.  The selection-bias study
(``docs/experiments/2026-07-27-inclusion-knob/SELECTION-BIAS.md``) measured
what that costs and found it splits by *how the item was drawn*, not by who
was labeling - a margin-sampled draw calibrates safely while a
top-of-the-list draw poisons the conformal threshold.  Deciding which votes
a future calibration may trust therefore needs to know each vote's surfacing
context, and that context is **not re-derivable**: the ranking that surfaced
the item is client-side ephemeral state and the model that produced its score
is overwritten by the next retrain.  So it is recorded at click time, here.

This module owns the vocabulary and the validation.  Recording only - nothing
in the codebase reads these values back to change behaviour yet; consuming
them is gated on the experiment pre-registered in
``docs/plans/provenance-partitioned-calibration.md``.

Storage
-------

A vote's provenance rides in :attr:`~vtscore.datasets.labelset.LabeledElement.metadata`
under the :data:`METADATA_KEY` namespace, so it round-trips through labelset
JSON export/import with no format change.  Scalars only: the
no-persisted-vectors rule is untouched.

Axes
----

The three questions "which ranking?", "which draw off that ranking?" and
"who was driving?" are **independent**, and each is recorded in its own
field rather than fused into one enum:

* :data:`FLOWS` - which UI flow drove the vote (``autopilot``, ``manual``, ...).
* :data:`PHASES` - the autopilot phase, when autopilot was driving.
* :data:`SELECT_MODES` - how the item was drawn off the ranking
  (``top`` / ``hard`` / ``new``).
* :data:`SORT_KINDS` - which ranking the user was looking at.

Keeping them separate matters because the bias lives on the *selection*
axis while the flow axis is what a user notices.  The two come apart at both
edges: a user can set the left panel's select mode to ``hard`` by hand and
get exactly the margin-sampled draw autopilot's Hard phase uses, and
autopilot's own ``good`` phase is mechanically a top-of-list draw.  A single
fused ``autopilot:hard`` / ``list_review`` enum would label both of those
cases by their flow and discard the variable that actually predicts the bias.
"""

from __future__ import annotations

import math
from typing import Any

#: Namespace key under which provenance is stored in ``LabeledElement.metadata``.
#: Prefixed so it can never collide with an importer's ``custom_metadata``,
#: which shares the same dict.
METADATA_KEY = "vt:provenance"

#: Schema version of the recorded payload, stored as ``"v"``.  Bump when the
#: field set changes in a way a reader must branch on.
SCHEMA_VERSION = 1

#: Which UI flow drove the vote.
#:
#: ``autopilot``       - the guided Train loop was driving item selection.
#: ``list_review``     - the user was voting down a sorted result list they
#:                       steered themselves.  This is the flow the
#:                       selection-bias study's ``toplist`` arm models and
#:                       found unsafe for calibration.
#: ``find_verify``     - verifying a Find pass's scored results.
#: ``labelset_review`` - correcting an already-saved label from the detector's
#:                       own labelset, not a fresh draw off any ranking.
#: ``seed_example``    - a user-supplied exemplar: example media seeded on
#:                       detector load, or an add-to-pile upload.  Never
#:                       surfaced by a sort at all.
#: ``import``          - applied by a label import that carried no provenance
#:                       of its own.
#: ``bulk``            - a batch action over a set (Browser verify-good/bad,
#:                       fill-from-sort).
#: ``undo``            - replayed from the undo/redo stack; not a fresh
#:                       surfacing event.
#: ``unknown``         - unattributed; the default for legacy votes.
FLOWS = frozenset(
    {
        "autopilot",
        "list_review",
        "find_verify",
        "labelset_review",
        "seed_example",
        "import",
        "bulk",
        "undo",
        "unknown",
    }
)

#: Autopilot phase at click time; ``None`` outside autopilot.  Mirrors the
#: frontend's ``AutopilotPhase`` minus its non-labeling states (``idle``,
#: ``done``, ``exhausted`` never surface an item to vote on).
PHASES = frozenset({"good", "bad", "hard", "new"})

#: How the item was drawn off the ranking.  Mirrors the frontend's
#: ``SelectMode``: ``top`` is the head of the sort, ``hard`` is margin
#: sampling near the decision boundary, ``new`` is atlas-diverse exploration.
SELECT_MODES = frozenset({"top", "hard", "new"})

#: Which ranking the user was looking at.  Mirrors the frontend's ``SortMode``.
SORT_KINDS = frozenset({"learned", "text", "load"})

_ENUM_FIELDS: dict[str, frozenset[str]] = {
    "flow": FLOWS,
    "phase": PHASES,
    "select_mode": SELECT_MODES,
    "sort_kind": SORT_KINDS,
}

#: Every key a recorded payload may carry, in serialisation order.
FIELDS = ("v", "flow", "phase", "select_mode", "sort_kind", "rank_at_vote", "score_at_vote")


def _clean_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be an integer")
    ivalue = int(value)
    if ivalue != value or ivalue < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return ivalue


def _clean_float(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    fvalue = float(value)
    if not math.isfinite(fvalue):
        raise ValueError(f"{field} must be finite")
    return fvalue


def _clean_enums(raw: dict[str, Any]) -> dict[str, str]:
    """Validate every enum-valued field present in *raw*, dropping absent ones."""
    out: dict[str, str] = {}
    for field, allowed in _ENUM_FIELDS.items():
        value = raw.get(field)
        if value is None:
            continue
        if not isinstance(value, str) or value not in allowed:
            raise ValueError(f"{field} must be one of: {', '.join(sorted(allowed))}")
        out[field] = value
    return out


def normalize_provenance(raw: Any) -> dict[str, Any] | None:
    """Validate *raw* into a canonical provenance payload.

    Returns ``None`` for ``None`` or for a payload that carries no
    information (no ``flow`` and no context scalars) - there is nothing worth
    persisting in an empty record, and storing one would inflate every
    labelset with a dict that says only "unknown".

    Absent optional fields are omitted rather than stored as ``None``, so the
    serialised form stays small; a reader treats a missing key as unknown.

    Raises:
        ValueError: on an unrecognised enum value, an unknown key, or a
            malformed scalar.  Callers at a request boundary let this become
            a 400; callers reading from disk should use
            :func:`coerce_provenance` instead.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("provenance must be an object")

    unknown_keys = set(raw) - set(FIELDS)
    if unknown_keys:
        raise ValueError(f"unknown provenance field(s): {', '.join(sorted(unknown_keys))}")

    out: dict[str, Any] = {"v": SCHEMA_VERSION}
    out.update(_clean_enums(raw))

    rank = _clean_int(raw.get("rank_at_vote"), "rank_at_vote")
    if rank is not None:
        out["rank_at_vote"] = rank
    score = _clean_float(raw.get("score_at_vote"), "score_at_vote")
    if score is not None:
        out["score_at_vote"] = score

    # A phase only means something under autopilot, and a bare "unknown" flow
    # says exactly what an absent record says.
    if out.get("flow") not in (None, "autopilot"):
        out.pop("phase", None)
    if len(out) == 1 or (len(out) == 2 and out.get("flow") == "unknown"):
        return None
    return out


def coerce_provenance(raw: Any) -> dict[str, Any] | None:
    """Best-effort :func:`normalize_provenance` that never raises.

    For payloads arriving from outside our own request validation - an
    imported labelset, a detector JSON hand-edited or written by an older
    build.  A record we cannot make sense of is dropped, not repaired: a
    half-understood provenance is worse than none, because a future
    calibration partition would trust it.
    """
    try:
        return normalize_provenance(raw)
    except (ValueError, TypeError):
        return None


def read_provenance(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    """Extract the provenance payload from a ``LabeledElement.metadata`` dict."""
    if not isinstance(metadata, dict):
        return None
    return coerce_provenance(metadata.get(METADATA_KEY))


def attach_provenance(
    metadata: dict[str, Any] | None,
    provenance: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return *metadata* with *provenance* written into :data:`METADATA_KEY`.

    Copies rather than mutating: the caller's dict is typically an importer's
    ``custom_metadata``, shared across every element built from that media.
    Returns *metadata* unchanged when there is no provenance to attach, so a
    labelset from a build that never recorded any is byte-identical to before.
    """
    cleaned = coerce_provenance(provenance)
    if cleaned is None:
        return metadata
    merged = dict(metadata) if metadata else {}
    merged[METADATA_KEY] = cleaned
    return merged
