"""Schemas for the background-job introspection endpoints.

The frontend top-bar pulldown calls :func:`vtsearch.routes.jobs.active_jobs`
to learn which ``(dataset_id, detector_id)`` pairs currently have work in
flight on a :class:`vtscore.concurrency.async_jobs.JobManager`. Each busy
pair gets a small spinner glyph on the row that completes the pair.
"""

from __future__ import annotations

from marshmallow import Schema, fields


class ActiveJobPairSchema(Schema):
    """One (dataset, detector) pair with at least one background job in flight."""

    dataset_id = fields.String(
        required=True,
        metadata={"description": "Registry id of the dataset the job is running against."},
    )
    detector_id = fields.String(
        required=True,
        metadata={"description": "Registry id of the detector the job is running against."},
    )
    job_types = fields.List(
        fields.String(),
        required=True,
        metadata={
            "description": (
                "Logical job-type names with a running or pending slot on this pair "
                "(e.g. ``learned-sort``, ``eval``). Stable across releases."
            ),
        },
    )


class ActiveJobsResponseSchema(Schema):
    """Response for ``GET /api/jobs/active`` — one entry per busy pair."""

    busy_pairs = fields.List(
        fields.Nested(ActiveJobPairSchema),
        required=True,
        metadata={"description": "Every pair with at least one active job, in arbitrary order."},
    )


__all__ = ["ActiveJobPairSchema", "ActiveJobsResponseSchema"]
