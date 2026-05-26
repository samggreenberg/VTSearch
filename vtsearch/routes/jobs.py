"""Blueprint for background-job introspection.

Currently exposes a single endpoint, ``GET /api/jobs/active``, which lists
every ``(dataset_id, detector_id)`` pair with a running or pending job on
any :class:`vtscore.concurrency.async_jobs.JobManager`. The top-bar
context-pulldown polls this endpoint and renders a spinner glyph on rows
whose pair has work in flight.
"""

from __future__ import annotations

from flask_smorest import Blueprint

from vtscore.concurrency.async_jobs import list_active_pairs
from vtsearch.schemas.jobs import ActiveJobsResponseSchema

jobs_bp = Blueprint(
    "jobs",
    __name__,
    description="Introspection endpoints for the JobManager singletons.",
)


@jobs_bp.route("/api/jobs/active", methods=["GET"])
@jobs_bp.response(200, ActiveJobsResponseSchema)
def active_jobs():
    """Return every ``(dataset, detector)`` pair with a running or pending job.

    Used by the top-bar pulldown to render a spinner glyph on rows whose
    pair has background work in flight (learned-sort, eval, …). The endpoint
    is intentionally cheap - it walks the in-memory ``JOB_MANAGERS`` map and
    reads each manager's current/pending slot under the manager's own lock,
    nothing else. Safe to poll on a 2–3 second interval.
    """
    return {"busy_pairs": list_active_pairs()}


__all__ = ["jobs_bp"]
