"""Shared test helpers for the VTSearch test suite."""


def train_detector_from_votes():
    """Train a detector from current good/bad votes and return the payload.

    Replacement for the removed ``POST /api/detector/export`` endpoint.
    Returns a dict with ``weights``, ``threshold``, ``good_origins``,
    ``bad_origins``, ``inclusion``, and ``media_type``.
    """
    from vtsearch.models import collect_media_origins
    from vtsearch.routes.detectors_helpers import serialize_weights, train_and_threshold
    from vtsearch.utils import bad_votes, get_inclusion, good_votes, snapshot_medias

    if not good_votes or not bad_votes:
        raise ValueError("Need at least one good and one bad vote")

    snap = snapshot_medias()

    good_origins = collect_media_origins(good_votes, snap)
    bad_origins = collect_media_origins(bad_votes, snap)

    X_list, y_list = [], []
    for cid in good_votes:
        if cid in snap and "embedding" in snap[cid]:
            X_list.append(snap[cid]["embedding"])
            y_list.append(1.0)
    for cid in bad_votes:
        if cid in snap and "embedding" in snap[cid]:
            X_list.append(snap[cid]["embedding"])
            y_list.append(0.0)

    model, threshold = train_and_threshold(X_list, y_list, snap)
    weights = serialize_weights(model)

    media_type = "audio"
    if snap:
        media_type = next(iter(snap.values())).get("type", "audio")

    return {
        "weights": weights,
        "threshold": threshold,
        "good_origins": good_origins,
        "bad_origins": bad_origins,
        "inclusion": get_inclusion(),
        "media_type": media_type,
    }
