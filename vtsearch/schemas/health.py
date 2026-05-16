"""Schemas for the ``/healthz`` (liveness) and ``/readyz`` (readiness) probes.

Liveness is a fixed-shape ``{"status": "ok"}``; readiness reports per-check
detail so an operator can tell *why* the pod is not yet receiving traffic.
"""

from __future__ import annotations

from marshmallow import Schema, fields


class HealthSchema(Schema):
    """Response for ``GET /healthz`` (always 200 while the process is alive)."""

    status = fields.String(
        required=True,
        metadata={"description": "Always 'ok' — the process is running and serving requests."},
    )


class ReadinessCheckSchema(Schema):
    """One readiness sub-check (data dir, models, etc.)."""

    ok = fields.Boolean(required=True, metadata={"description": "Whether this individual check passed."})
    detail = fields.String(
        required=False,
        metadata={"description": "Optional human-readable note (failure reason or extra info)."},
    )


class ReadinessSchema(Schema):
    """Response for ``GET /readyz`` (200 when ready, 503 when not).

    The HTTP status carries the verdict; the body lets operators see which
    sub-check tripped without grepping logs.
    """

    status = fields.String(
        required=True,
        metadata={"description": "'ready' on 200, 'not_ready' on 503."},
    )
    checks = fields.Dict(
        keys=fields.String(),
        values=fields.Nested(ReadinessCheckSchema),
        required=True,
        metadata={"description": "Per-check results keyed by check name (data_dir, models, …)."},
    )


__all__ = ["HealthSchema", "ReadinessCheckSchema", "ReadinessSchema"]
