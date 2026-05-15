"""Marshmallow schemas for VTSearch's HTTP API.

The sub-modules in this package declare the request/response shapes
consumed by ``vtsearch/routes/`` and exported via the OpenAPI spec at
``/api/openapi.json``. The hand-maintained DTOs in
``frontend/src/app/models/api.models.ts`` are being replaced by
TypeScript types generated from this spec — see
``docs/plans/openapi-schema.md``.
"""
