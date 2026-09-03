"""Deployment-policy guards applied at the route boundary.

Today that is the Semantic-lock (``semantic_only``): a server configured to
offer only Semantic embedders must refuse a hand-rolled request that names a
patch or structural one, rather than binding a type the rest of the UI hides.
"""

from __future__ import annotations

from flask_smorest import abort


#: Message used by both Semantic-lock guards below, so the API surfaces one
#: consistent explanation whichever route the request hit.
SEMANTIC_ONLY_MESSAGE = (
    "This server is locked to Semantic embedders (semantic_only). "
    "Patch Semantic and Structural embedders are unavailable here."
)


def abort_if_semantic_only_type(embedder_type: str) -> None:
    """Reject a non-Semantic detector *embedder_type* on a Semantic-locked server.

    The type is the detector's declared intent, so this is the one gate that
    keeps a hand-rolled ``POST /api/detectors`` (or a portable bundle carrying a
    Structural detector) from creating a detector this deployment can never
    run. Empty / ``"semantic"`` pass through untouched, as does every request
    when the lock is off.
    """
    if not embedder_type or embedder_type == "semantic":
        return
    from vtsearch.settings import get_effective_semantic_only  # noqa: PLC0415 - avoid import cycle

    if get_effective_semantic_only():
        abort(400, message=SEMANTIC_ONLY_MESSAGE)


def abort_if_semantic_only_embedders(embedder_names) -> None:
    """Reject patch / structural *embedder_names* on a Semantic-locked server.

    Guards the dataset-load routes, whose ``embedders`` trio arrives straight
    from the client: the pickers never offer a prototype embedder under the
    lock (``GET /api/embedders`` filters them out), so a request that names one
    is either stale or hand-rolled and should fail loudly rather than quietly
    binding a type the rest of the UI hides. Unknown names are left alone --
    they fail their own validation downstream.
    """
    names = [n for n in (embedder_names or ()) if n]
    if not names:
        return
    from vtsearch.settings import get_effective_semantic_only  # noqa: PLC0415 - avoid import cycle

    if not get_effective_semantic_only():
        return

    from vtscore.embedding.binding import embedder_type as _classify  # noqa: PLC0415

    offenders = sorted({n for n in names if _classify(n) in ("patch_semantic", "structural")})
    if offenders:
        abort(400, message=f"{SEMANTIC_ONLY_MESSAGE} Rejected: {', '.join(offenders)}.")
