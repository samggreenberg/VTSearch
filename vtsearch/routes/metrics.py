"""Prometheus ``/metrics`` endpoint.

Renders the application's metric registry in the standard Prometheus
text exposition format (``text/plain; version=0.0.4``) so any
Prometheus server, OpenTelemetry collector, or Grafana Agent can scrape
it without further configuration.

Metric definitions live in :mod:`vtsearch.metrics`; this module is a
thin Flask wrapper. ``/metrics`` is registered on the plain ``Flask``
app — not the ``flask_smorest.Api`` — because the response body is
opaque to OpenAPI and would only clutter the spec.
"""

from __future__ import annotations

from flask import Blueprint, Response

from vtsearch import metrics

metrics_bp = Blueprint("metrics", __name__)


@metrics_bp.route("/metrics")
def prometheus_metrics() -> Response:
    """Return the Prometheus exposition payload."""
    return Response(metrics.render(), mimetype=metrics.CONTENT_TYPE_LATEST)
