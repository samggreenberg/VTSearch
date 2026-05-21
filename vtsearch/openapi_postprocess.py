"""Post-processing for the flask-smorest OpenAPI spec.

flask-smorest / apispec do not populate ``operationId`` by default, so
``ng-openapi-gen`` has to synthesize method names from the path + method
when generating the TypeScript client. That produces names like
``apiDetectorsRegistryDatasetIdLabelsetSourceMoveFilePost`` that churn
on every URL rename.

:func:`assign_operation_ids` walks the live Flask URL map, matches each
rule to its view function, and writes ``operationId = view_func.__name__``
into the spec. It is wired up in ``app.py`` by wrapping
``api.spec.to_dict`` so both the live ``/api/openapi.json`` endpoint and
the ``scripts/dump_openapi.py`` snapshot dumper see the post-processed
result.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from flask import Flask


_FLASK_PARAM_RE = re.compile(r"<(?:[^:>]+:)?([^>]+)>")


def _flask_to_openapi_path(rule: str) -> str:
    """Convert a Flask rule (``/foo/<int:bar>``) to OpenAPI (``/foo/{bar}``)."""
    return _FLASK_PARAM_RE.sub(r"{\1}", rule)


def _map_spec_ops_to_view_names(app: Flask, paths: dict[str, Any]) -> dict[tuple[str, str], str]:
    """Return ``{(openapi_path, method_lower): view_func.__name__}``."""
    op_to_view: dict[tuple[str, str], str] = {}
    for rule in app.url_map.iter_rules():
        view_func = app.view_functions.get(rule.endpoint)
        if view_func is None:
            continue
        op_path = _flask_to_openapi_path(rule.rule)
        if op_path not in paths:
            continue
        rule_methods = (rule.methods or set()) - {"HEAD", "OPTIONS"}
        for method in rule_methods:
            op_method = method.lower()
            if op_method in paths[op_path]:
                op_to_view[(op_path, op_method)] = view_func.__name__
    return op_to_view


def _operation_ids_for_view(name: str, op_keys: list[tuple[str, str]]) -> dict[tuple[str, str], str]:
    """Assign deterministic operationIds for a set of ops sharing a view.

    Single-operation views keep the bare ``name``. Collisions get a
    ``_{method}`` suffix; if multiple ops share both view and method,
    they additionally get a 1-based index in sorted order.
    """
    if len(op_keys) == 1:
        return {op_keys[0]: name}

    op_keys_sorted = sorted(op_keys)
    method_counts: dict[str, int] = defaultdict(int)
    for _, m in op_keys_sorted:
        method_counts[m] += 1

    assigned: dict[tuple[str, str], str] = {}
    per_method_index: dict[str, int] = defaultdict(int)
    for op_path, op_method in op_keys_sorted:
        if method_counts[op_method] == 1:
            assigned[(op_path, op_method)] = f"{name}_{op_method}"
        else:
            per_method_index[op_method] += 1
            assigned[(op_path, op_method)] = f"{name}_{op_method}_{per_method_index[op_method]}"
    return assigned


def assign_operation_ids(app: Flask, spec: dict[str, Any]) -> None:
    """Mutate ``spec`` to assign an ``operationId`` to every operation.

    The id is the Flask view function's ``__name__``. When two operations
    in the spec map to the same view function (a single view registered
    under several routes, like ``/dashboard`` and ``/label`` sharing
    ``angular_routes``), we disambiguate by appending the HTTP method and,
    if that is still not unique, a 1-based index in sorted (path, method)
    order — so the assignment is deterministic and snapshot-stable across
    runs.
    """
    paths = spec.get("paths")
    if not paths:
        return

    op_to_view = _map_spec_ops_to_view_names(app, paths)

    by_view: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for op_key, name in op_to_view.items():
        by_view[name].append(op_key)

    for name, op_keys in by_view.items():
        for (op_path, op_method), op_id in _operation_ids_for_view(name, op_keys).items():
            paths[op_path][op_method]["operationId"] = op_id
