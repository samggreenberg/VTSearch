"""Every error code a route can return is declared on its OpenAPI operation (#5255).

flask-smorest documents an error code only when the route says so with
``@blp.alt_response(code, ...)`` (422 aside, which ``@blp.arguments`` adds).
A route that ``abort()``s with a code it never declares still works, but the
spec, the ``frontend/openapi.json`` snapshot built from it and the client
generated from that all omit the code.  Nothing noticed: the gap showed up
only when a documentation audit compared ``docs/api/*.md`` against the spec.

So this test reads the code.  For each route in the spec it walks the view,
everything wrapped around it (the ``__wrapped__`` chain, so
``require_detector_header``'s 400 counts) and every app-tier function the
body calls by name, transitively, and collects the status codes of:

* ``abort(<code>, ...)``,
* ``error_response(<message>, <code>, ...)``,
* ``return jsonify(...), <code>`` (any call or string as the body),

with ``<code>`` an integer literal, or a parameter that a caller fills with
one (``_abort_find(404, ...)``).  Any code of 400 or above missing from the
operation's ``responses`` fails the test.

What it does not cover, on purpose: the codes every route shares.  Those
are the global handlers in ``vtsearch/errors.py``, which turn exceptions
into 409 / 404 for an unloaded or deleted context, 400 for a missing one and
500 for anything uncaught, and the request hooks in ``vtsearch/runtime/request_hooks.py``,
whose 401 refuses a request the login provider rejects.  They are documented
once, by the ``default`` response every operation carries.

The walk is static, so a call it cannot resolve (a method on an object, a
function passed in as a value) is a place an undeclared code could hide.
``test_scanner_reaches_every_error_site`` closes that hole: every
``abort()`` / ``error_response()`` call in ``vtsearch/`` must be reached
from some route, so a call the walk cannot see fails there instead of
passing here by omission.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import textwrap
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, TypeGuard

import app as app_module
import vtsearch
from vtsearch.openapi.postprocess import _flask_to_openapi_path

_VTSEARCH_DIR = Path(vtsearch.__file__).resolve().parent
#: Modules whose codes apply to every route: the ``default`` response's (see above).
_GLOBAL_MODULES = {_VTSEARCH_DIR / "errors.py", _VTSEARCH_DIR / "runtime" / "request_hooks.py"}
#: Callee name -> (position, keyword) of its status argument.
_ERROR_CALLS = {"abort": (0, None), "error_response": (1, "status")}
_FUNCTION_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef)
_SCOPES = (*_FUNCTION_DEFS, ast.Lambda, ast.ClassDef)

Codes = dict[int, set[str]]  # status code -> where it is emitted


def _source_path(code: types.CodeType) -> Path:
    return Path(code.co_filename).resolve()


def _is_app_function(obj) -> TypeGuard[types.FunctionType]:
    """True for a plain function defined in ``vtsearch/``, outside the global modules."""
    if not inspect.isfunction(obj):
        return False
    path = _source_path(obj.__code__)
    return path.is_relative_to(_VTSEARCH_DIR) and path not in _GLOBAL_MODULES


def _int_literal(node: ast.AST | None) -> int | None:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    return None


def _error_status_arg(call: ast.Call) -> tuple[bool, ast.AST | None]:
    """``(is an abort() / error_response() call, its status argument)``."""
    if not isinstance(call.func, ast.Name) or call.func.id not in _ERROR_CALLS:
        return False, None
    position, keyword = _ERROR_CALLS[call.func.id]
    if len(call.args) > position:
        return True, call.args[position]
    return True, next((k.value for k in call.keywords if keyword and k.arg == keyword), None)


def _response_tuple_status(node: ast.Return) -> ast.AST | None:
    """The status of ``return <body>, <status>`` when ``<body>`` is a call or a string."""
    value = node.value
    if not isinstance(value, ast.Tuple) or len(value.elts) < 2:
        return None
    body = value.elts[0]
    if isinstance(body, ast.Call) or (isinstance(body, ast.Constant) and isinstance(body.value, str)):
        return value.elts[1]
    return None


def _own_nodes(fnode: ast.AST):
    """The nodes of *fnode*'s body, not descending into nested defs, lambdas or classes."""
    stack = list(ast.iter_child_nodes(fnode))
    while stack:
        node = stack.pop()
        if isinstance(node, _SCOPES):
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def _params(fnode: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    args = fnode.args
    return [a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]]


def _argument(call: ast.Call, params: list[str], name: str) -> ast.AST | None:
    """The node *call* passes for parameter *name*, or ``None``."""
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    index = params.index(name)
    return call.args[index] if index < len(call.args) else None


def _local_imports(fnode: ast.AST) -> dict[str, tuple[str, str | None]]:
    """``{bound name: (module, attribute or None)}`` for the imports inside *fnode*."""
    bound: dict[str, tuple[str, str | None]] = {}
    for node in ast.walk(fnode):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                bound[alias.asname or alias.name] = (node.module, alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    bound[alias.asname] = (alias.name, None)
                else:
                    top = alias.name.split(".")[0]
                    bound[top] = (top, None)
    return bound


def _resolve_name(name: str, func, imports):
    if name in func.__code__.co_freevars:
        return None  # a closure cell: the view a decorator wraps, walked via ``__wrapped__``
    if name in imports:
        module, attr = imports[name]
        resolved = importlib.import_module(module)
        return getattr(resolved, attr, None) if attr else resolved
    return func.__globals__.get(name)


def _resolve_callee(callee: ast.AST, func, imports):
    """The object a ``name(...)`` or ``module.name(...)`` call refers to, or ``None``."""
    if isinstance(callee, ast.Name):
        return _resolve_name(callee.id, func, imports)
    if isinstance(callee, ast.Attribute) and isinstance(callee.value, ast.Name):
        base = _resolve_name(callee.value.id, func, imports)
        if isinstance(base, types.ModuleType):
            return getattr(base, callee.attr, None)
    return None


class _Scanner:
    """Collects the error codes a function can emit, following app-tier calls."""

    def __init__(self) -> None:
        #: ``(file, line)`` of every abort() / error_response() call walked so far.
        self.sites_seen: set[tuple[Path, int]] = set()
        self._cache: dict[types.CodeType, tuple[Codes, set[str], list[str]]] = {}

    def codes_of(self, view) -> Codes:
        """The codes *view* and every layer it wraps can emit."""
        found: Codes = defaultdict(set)
        layer = view
        while layer is not None:
            if _is_app_function(layer):
                for code, origins in self._analyze(layer)[0].items():
                    found[code] |= origins
            layer = getattr(layer, "__wrapped__", None)
        return found

    def _analyze(self, func, stack: frozenset = frozenset()) -> tuple[Codes, set[str], list[str]]:
        """``(codes, forwarded, params)`` for *func*'s own body; see :meth:`_walk`."""
        code = func.__code__
        if code in self._cache:
            return self._cache[code]
        if code in stack:
            return {}, set(), []
        lines, first_line = inspect.getsourcelines(code)
        fnode = ast.parse(textwrap.dedent("".join(lines))).body[0]
        assert isinstance(fnode, _FUNCTION_DEFS), fnode
        where = f"{func.__globals__['__name__']}.{code.co_qualname}"
        result = (*self._walk(func, fnode, first_line - 1, where, stack | {code}), _params(fnode))
        self._cache[code] = result
        return result

    def _walk(self, func, fnode, line_offset: int, where: str, stack: frozenset) -> tuple[Codes, set[str]]:
        """``(codes, forwarded)``: what *fnode* emits, and which of its parameters reach a status argument."""
        params = _params(fnode)
        nested = {n.name: n for n in ast.walk(fnode) if isinstance(n, _FUNCTION_DEFS) and n is not fnode}
        imports = _local_imports(fnode)
        codes: Codes = defaultdict(set)
        forwarded: set[str] = set()

        def take(status: ast.AST | None, origin: str, *, may_forward: bool = True) -> None:
            value = _int_literal(status)
            if value is not None:
                codes[value].add(origin)
            elif may_forward and isinstance(status, ast.Name) and status.id in params:
                forwarded.add(status.id)

        for node in _own_nodes(fnode):
            if isinstance(node, ast.Return):
                take(_response_tuple_status(node), where, may_forward=False)
                continue
            if not isinstance(node, ast.Call):
                continue
            is_error_call, status = _error_status_arg(node)
            if is_error_call:
                self.sites_seen.add((_source_path(func.__code__), node.lineno + line_offset))
                take(status, where)
                continue
            if isinstance(node.func, ast.Name) and node.func.id in nested:
                local = nested[node.func.id]
                label = f"{where}.<locals>.{local.name}"
                sub_codes, sub_forwarded = self._walk(func, local, line_offset, label, stack)
                sub_params = _params(local)
            else:
                target = _resolve_callee(node.func, func, imports)
                if not _is_app_function(target):
                    continue
                sub_codes, sub_forwarded, sub_params = self._analyze(target, stack)
                label = f"{target.__module__}.{target.__qualname__}"
            for value, origins in sub_codes.items():
                codes[value] |= origins
            for name in sub_forwarded:
                take(_argument(node, sub_params, name), f"{where} -> {label}")
        return codes, forwarded


def _operations(client):
    """Yield ``(method, openapi path, view, declared codes or None)`` for every URL rule.

    ``declared`` is ``None`` for a rule the spec does not describe (a plain
    Flask blueprint); those are still walked, for the reach check.
    """
    app = app_module.app
    paths: dict[str, dict[str, Any]] = client.get("/api/openapi.json").get_json()["paths"]
    for rule in app.url_map.iter_rules():
        view = app.view_functions[rule.endpoint]
        path = _flask_to_openapi_path(rule.rule)
        for method in sorted((rule.methods or set()) - {"HEAD", "OPTIONS"}):
            operation = paths[path].get(method.lower()) if path in paths else None
            declared = None if operation is None else {int(c) for c in operation["responses"] if c.isdigit()}
            yield method, path, view, declared


def test_every_error_code_a_route_emits_is_declared(client):
    scanner = _Scanner()
    missing = []
    for method, path, view, declared in _operations(client):
        if declared is None:
            continue
        for code, origins in sorted(scanner.codes_of(view).items()):
            if code >= 400 and code not in declared:
                missing.append(f"{method} {path}: {code} (from {', '.join(sorted(origins))})")
    assert not missing, (
        "These routes can return an error code their OpenAPI operation does not declare.\n"
        "Add `@<blueprint>.alt_response(<code>, description=...)` to each route, then regenerate\n"
        "the snapshot (`cd frontend && npm run regenerate-openapi-snapshot`):\n  " + "\n  ".join(missing)
    )


def test_scanner_reaches_every_error_site(client):
    scanner = _Scanner()
    for _method, _path, view, _declared in _operations(client):
        scanner.codes_of(view)

    unreached = []
    for path in sorted(_VTSEARCH_DIR.rglob("*.py")):
        if path in _GLOBAL_MODULES:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and _error_status_arg(node)[0]
                and (path, node.lineno) not in scanner.sites_seen
            ):
                unreached.append(f"{path.relative_to(_VTSEARCH_DIR.parent)}:{node.lineno}")
    assert not unreached, (
        "No route reaches these abort() / error_response() calls through a call the scanner in this\n"
        "file can follow, so it cannot check that their codes are declared. Call them by name from\n"
        "the route (or from a helper it calls by name), or teach the scanner the new call shape:\n  "
        + "\n  ".join(unreached)
    )
