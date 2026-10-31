"""Flask route for the usage-metrics API.

Endpoints
---------
POST /api/metrics
    Append a batch of SPA usage events to this month's metrics file.
    Body: ``{"events": [{"group", "attributes", "client_ts"}], "session"}``.

The sink, the file layout and the ``enable_metrics`` switch live in
:mod:`vtsearch.metrics`.
"""

from __future__ import annotations

from flask_smorest import Blueprint

from vtsearch import metrics
from vtsearch.auth import get_current_user
from vtsearch.schemas.metrics import MetricsBatchResponseSchema, MetricsBatchSchema

metrics_bp = Blueprint(
    "metrics",
    __name__,
    description="Usage metrics posted by the SPA.",
)


@metrics_bp.route("/api/metrics", methods=["POST"])
@metrics_bp.arguments(MetricsBatchSchema)
@metrics_bp.response(200, MetricsBatchResponseSchema)
def record_metrics(body: dict):
    """Append a batch of usage events for the current user.

    Records nothing when the server-tier ``enable_metrics`` setting is off,
    and says so in ``enabled`` so the SPA can stop sending.
    """
    enabled = metrics.is_enabled()
    recorded = metrics.record_events(body["events"], user=get_current_user(), session=body["session"]) if enabled else 0
    return {"recorded": recorded, "enabled": enabled}
