"""OpenAPI 3.0 schema generation for VTSearch's Flask routes.

VTSearch's HTTP API today is documented manually in ``docs/API.md`` and
duplicated in the Angular frontend's DTOs.  This module walks Flask's
``url_map`` and produces a minimal OpenAPI 3.0 document — enough for a
Swagger UI and an auto-generated TS client to stop drifting from the
backend.

Limitations
-----------

We don't introspect handler bodies, so request/response *schemas* are
left empty (``content`` defaults to JSON with no shape).  Path
parameters declared in the rule (``<int:id>``) are surfaced; query and
JSON body parameters would need either decorators or runtime
introspection — both deferred.  Even so, the route inventory + methods +
docstrings already prevents the most common kind of drift (a route exists
or it doesn't).
"""

from __future__ import annotations

import re
from typing import Any

from flask import Flask

#: HTTP methods Flask adds to every route, which aren't real verbs.
_INTERNAL_METHODS: frozenset[str] = frozenset({"OPTIONS", "HEAD"})

#: Maps Werkzeug rule converters to OpenAPI ``schema.type`` values.
_TYPE_MAP: dict[str, dict[str, str]] = {
    "int": {"type": "integer"},
    "float": {"type": "number"},
    "uuid": {"type": "string", "format": "uuid"},
    "path": {"type": "string"},
    "string": {"type": "string"},
    "default": {"type": "string"},
}

_RULE_PARAM_RE = re.compile(r"<(?:(?P<converter>[^:>]+):)?(?P<name>[^>]+)>")


def _flask_rule_to_openapi_path(rule: str) -> tuple[str, list[dict[str, Any]]]:
    """Convert a Flask rule like ``/foo/<int:id>`` to OpenAPI path + params.

    Returns ``(openapi_path, [parameter_dict, ...])`` where each parameter
    dict is an OpenAPI Parameter Object with ``in: path``.
    """
    params: list[dict[str, Any]] = []

    def _replace(match: re.Match[str]) -> str:
        converter = match.group("converter") or "default"
        name = match.group("name")
        params.append(
            {
                "name": name,
                "in": "path",
                "required": True,
                "schema": _TYPE_MAP.get(converter, _TYPE_MAP["default"]),
            }
        )
        return "{" + name + "}"

    return _RULE_PARAM_RE.sub(_replace, rule), params


def _operation_for(view: Any, methods: list[str], path: str) -> dict[str, Any]:
    """Build an OpenAPI Operation Object from a Flask view function."""
    doc = (getattr(view, "__doc__", "") or "").strip()
    summary = doc.splitlines()[0] if doc else ""
    description = doc

    op: dict[str, Any] = {
        "operationId": getattr(view, "__name__", "unknown"),
        "responses": {
            "200": {"description": "Successful response"},
        },
    }
    if summary:
        op["summary"] = summary
    if description:
        op["description"] = description

    # Tag operations by URL family ("/api/datasets/..." → "datasets").
    tag = _infer_tag(path)
    if tag:
        op["tags"] = [tag]

    if any(m in methods for m in ("POST", "PUT", "PATCH")):
        op["requestBody"] = {
            "required": False,
            "content": {"application/json": {"schema": {"type": "object"}}},
        }

    return op


def _infer_tag(path: str) -> str:
    """Pick an OpenAPI tag from a URL path.

    Uses the segment after ``/api/`` when present, otherwise the first
    non-empty segment.  Returns ``""`` for paths like ``/``.
    """
    parts = [p for p in path.strip("/").split("/") if p]
    if not parts:
        return ""
    if parts[0] == "api" and len(parts) > 1:
        return parts[1]
    return parts[0]


def generate_openapi_spec(app: Flask, *, title: str = "VTSearch API", version: str = "1.0.0") -> dict[str, Any]:
    """Return an OpenAPI 3.0 dict describing every Flask route on *app*.

    The result is JSON-serialisable and ready to be served from a route or
    written to a file.
    """
    paths: dict[str, dict[str, Any]] = {}
    tags: dict[str, bool] = {}

    for rule in sorted(app.url_map.iter_rules(), key=lambda r: r.rule):
        # Skip Flask's static file serving rule — it has no useful API surface.
        if rule.endpoint == "static":
            continue

        view = app.view_functions.get(rule.endpoint)
        if view is None:
            continue

        openapi_path, path_params = _flask_rule_to_openapi_path(rule.rule)
        path_entry = paths.setdefault(openapi_path, {})
        if path_params and "parameters" not in path_entry:
            path_entry["parameters"] = path_params

        methods = sorted(m for m in (rule.methods or set()) if m not in _INTERNAL_METHODS)
        for method in methods:
            op = _operation_for(view, methods, openapi_path)
            path_entry[method.lower()] = op
            for tag in op.get("tags", []):
                tags[tag] = True

    spec: dict[str, Any] = {
        "openapi": "3.0.3",
        "info": {
            "title": title,
            "version": version,
            "description": (
                "Auto-generated from VTSearch's Flask routes. "
                "Request/response schemas are intentionally permissive — "
                "see docs/API.md for the canonical reference."
            ),
        },
        "paths": paths,
    }
    if tags:
        spec["tags"] = [{"name": t} for t in sorted(tags)]
    return spec


__all__ = ["generate_openapi_spec"]
