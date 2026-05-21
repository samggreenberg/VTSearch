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

    # Map every (openapi_path, method) operation in the spec to its
    # Flask view function name.
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

    # Group operations by view function so we can detect collisions and
    # disambiguate deterministically.
    by_view: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for op_key, name in op_to_view.items():
        by_view[name].append(op_key)

    for name, op_keys in by_view.items():
        if len(op_keys) == 1:
            op_path, op_method = op_keys[0]
            paths[op_path][op_method]["operationId"] = name
            continue

        # Multiple operations share this view function. First try
        # appending the HTTP method; if methods alone don't disambiguate
        # (same view bound to several paths under the same method), fall
        # back to a 1-based index in sorted order.
        op_keys_sorted = sorted(op_keys)
        method_counts: dict[str, int] = defaultdict(int)
        for _, m in op_keys_sorted:
            method_counts[m] += 1

        per_method_index: dict[str, int] = defaultdict(int)
        for op_path, op_method in op_keys_sorted:
            if method_counts[op_method] == 1:
                op_id = f"{name}_{op_method}"
            else:
                per_method_index[op_method] += 1
                op_id = f"{name}_{op_method}_{per_method_index[op_method]}"
            paths[op_path][op_method]["operationId"] = op_id
