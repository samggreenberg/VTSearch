"""Liveness (``/healthz``) and readiness (``/readyz``) probes.

These are the standard two-endpoint pattern popularised by Kubernetes:

* ``/healthz`` answers *"is the process up?"* — cheap, dependency-free,
  always 200 while Flask is serving. A failure means the orchestrator
  should restart the container.
* ``/readyz`` answers *"can this process actually serve traffic right now?"*
  It returns 200 when every readiness sub-check passes and 503 otherwise.
  A failure means the orchestrator should pull this instance out of the
  load-balancer rotation but leave it running (it may recover on its own,
  e.g. once embedders finish warming up).

The endpoints live at the root of the URL space (not under ``/api/``) to
follow the convention orchestrators expect.
"""

from __future__ import annotations

import os

from flask import jsonify
from flask_smorest import Blueprint

from vtscore.config import DATA_DIR
from vtscore.embedding.loader import predict_embedders_to_preload
from vtscore.media import get_embedder
from vtsearch.schemas.health import HealthSchema, ReadinessSchema

health_bp = Blueprint(
    "health",
    __name__,
    description="Liveness and readiness probes for orchestrators (Kubernetes, ECS, etc.).",
)


@health_bp.route("/healthz", methods=["GET"])
@health_bp.response(200, HealthSchema)
def healthz() -> dict:
    """Liveness probe — 200 while the Flask process is serving.

    Intentionally does no work beyond returning a constant. Anything more
    expensive risks turning a slow dependency into a restart loop.
    """
    return {"status": "ok"}


def _check_data_dir() -> tuple[bool, str]:
    """Return ``(ok, detail)`` for the data-directory readiness check."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return False, f"cannot create data dir {DATA_DIR}: {exc}"
    if not os.access(DATA_DIR, os.W_OK):
        return False, f"data dir {DATA_DIR} is not writable"
    return True, str(DATA_DIR)


def _check_models() -> tuple[bool, str]:
    """Return ``(ok, detail)`` for the embedder-warmup readiness check.

    "Ready" means every embedder predicted from the registered datasets
    and detectors has its model in memory. With an empty registry (fresh
    process, no datasets loaded), nothing is expected and the check
    passes trivially.
    """
    expected = predict_embedders_to_preload()
    if not expected:
        return True, "no embedders required (empty registry)"

    pending: list[str] = []
    for name in expected:
        try:
            emb = get_embedder(name)
        except KeyError:
            pending.append(name)
            continue
        if getattr(emb, "_model", None) is None:
            pending.append(name)

    if pending:
        return False, f"loading: {', '.join(pending)}"
    return True, f"loaded: {', '.join(expected)}"


@health_bp.route("/readyz", methods=["GET"])
@health_bp.response(200, ReadinessSchema)
@health_bp.alt_response(503, schema=ReadinessSchema, description="One or more readiness checks failed.")
def readyz():
    """Readiness probe — 200 when every sub-check passes, 503 otherwise.

    Sub-checks:

    * ``data_dir`` — :data:`vtscore.config.DATA_DIR` exists and is
      writable (the app needs this for embeddings, model cache, settings).
    * ``models`` — every embedder implied by the dataset/detector
      registries is warm (``_model is not None``).
    """
    checks: dict[str, dict] = {}

    data_ok, data_detail = _check_data_dir()
    checks["data_dir"] = {"ok": data_ok, "detail": data_detail}

    models_ok, models_detail = _check_models()
    checks["models"] = {"ok": models_ok, "detail": models_detail}

    all_ok = all(c["ok"] for c in checks.values())
    body = {"status": "ready" if all_ok else "not_ready", "checks": checks}
    if all_ok:
        return body
    return jsonify(body), 503


__all__ = ["health_bp"]
