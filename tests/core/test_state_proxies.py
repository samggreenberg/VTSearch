"""Gate tests for the app-tier state proxies.

``_ProxyDict`` / ``_ProxyList`` inherit from ``dict`` / ``list`` so that
``isinstance(medias, dict)`` holds, but they keep their own built-in storage
permanently empty and forward a hand-enumerated method list to the active
context's real container.  That design has one failure mode, and it is the
worst kind: a method that is *not* forwarded executes against the empty own
storage and returns a confidently wrong answer instead of raising.

These tests make that failure mode impossible to reintroduce silently:

* :class:`TestForwardingCoverage` fails when any public ``dict`` / ``list``
  method is neither forwarded nor explicitly blacklisted, so a new Python
  version growing a container method cannot regress the proxies quietly.
* :class:`TestForwardedBehaviour` pins the behaviour of the methods that were
  previously missing.
* :class:`TestCFastPathBypasses` pins the operations that *cannot* be fixed by
  forwarding, so the boundary stays documented rather than surprising.
* :class:`TestReExportParity` fails when ``vtsearch.state`` stops re-exporting
  a public ``vtscore.state`` name.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from vtsearch.state_proxies import _ProxyDict, _ProxyList

# Dunders inherited from ``object`` that carry no container semantics: they are
# about construction, reflection, or pickling, not about the proxied contents.
_OBJECT_DUNDERS = {
    "__class__",
    "__class_getitem__",
    "__delattr__",
    "__dir__",
    "__doc__",
    "__format__",
    "__getattribute__",
    "__getstate__",
    "__hash__",
    "__init__",
    "__init_subclass__",
    "__new__",
    "__reduce__",
    "__reduce_ex__",
    "__setattr__",
    "__sizeof__",
    "__str__",
    "__subclasshook__",
}

# Public members deliberately NOT forwarded, each with the reason it is safe.
# The bar for this list is that the un-forwarded member must fail *loudly*
# (raise) rather than answer from the proxy's empty own storage.
_ALLOWED_UNFORWARDED = {
    _ProxyDict: {
        # ``dict`` has no ordering; these come from ``object`` and return
        # NotImplemented, so any use raises TypeError instead of comparing the
        # empty own storage.  Verified by ``test_blacklisted_members_raise``.
        "__lt__": "dict has no ordering; raises TypeError",
        "__le__": "dict has no ordering; raises TypeError",
        "__gt__": "dict has no ordering; raises TypeError",
        "__ge__": "dict has no ordering; raises TypeError",
        # A classmethod-style constructor: it builds a new mapping from the
        # supplied keys and never reads instance storage.  Called on a proxy it
        # raises TypeError (the proxy constructor takes a target attribute).
        "fromkeys": "constructor, not a read of instance storage; raises TypeError",
    },
    _ProxyList: {},
}


def _public_members(base: type) -> set[str]:
    """Public API of *base*: plain names plus dunders, minus object plumbing."""
    return {
        name
        for name in dir(base)
        if (not name.startswith("_") or (name.startswith("__") and name.endswith("__"))) and name not in _OBJECT_DUNDERS
    }


def _forwarded(proxy_cls: type) -> set[str]:
    """Names *proxy_cls* defines in its own body (i.e. actually forwards)."""
    return set(vars(proxy_cls))


class _Ctx:
    """Stand-in for a DatasetContext / DetectorContext."""

    def __init__(self) -> None:
        self.d: dict[int, str] = {1: "a", 2: "b"}
        self.lst: list[int] = [10, 20]


@pytest.fixture
def ctx() -> _Ctx:
    return _Ctx()


@pytest.fixture
def pdict(ctx: _Ctx) -> _ProxyDict:
    return _ProxyDict("d", lambda: ctx)


@pytest.fixture
def plist(ctx: _Ctx) -> _ProxyList:
    return _ProxyList("lst", lambda: ctx)


class TestForwardingCoverage:
    """Every public dict/list method is forwarded or explicitly blacklisted."""

    @pytest.mark.parametrize(("proxy_cls", "base"), [(_ProxyDict, dict), (_ProxyList, list)])
    def test_no_silently_unforwarded_members(self, proxy_cls: type, base: type) -> None:
        unforwarded = _public_members(base) - _forwarded(proxy_cls)
        unexplained = unforwarded - set(_ALLOWED_UNFORWARDED[proxy_cls])
        assert not unexplained, (
            f"{proxy_cls.__name__} does not forward {sorted(unexplained)}, and they are not in "
            f"_ALLOWED_UNFORWARDED. An unforwarded member runs against the proxy's permanently "
            f"empty own storage and returns a wrong answer instead of raising. Either forward it "
            f"in vtsearch/state_proxies.py, or add it to _ALLOWED_UNFORWARDED with the reason it "
            f"fails loudly."
        )

    @pytest.mark.parametrize(("proxy_cls", "base"), [(_ProxyDict, dict), (_ProxyList, list)])
    def test_blacklist_has_no_stale_entries(self, proxy_cls: type, base: type) -> None:
        """A blacklisted name must still exist on the base and still be unforwarded."""
        for name in _ALLOWED_UNFORWARDED[proxy_cls]:
            assert name in _public_members(base), f"{name} is no longer public on {base.__name__}"
            assert name not in _forwarded(proxy_cls), (
                f"{name} is now forwarded by {proxy_cls.__name__}; drop its _ALLOWED_UNFORWARDED entry"
            )

    def test_blacklisted_members_raise(self, pdict: _ProxyDict) -> None:
        """The blacklist's premise: these fail loudly rather than answering wrongly.

        ``pdict`` is rebound through an ``Any`` alias because pyright statically
        rejects the very unsupported-operator case this test exists to exercise
        at runtime.
        """
        untyped: Any = pdict
        for op in (
            lambda: untyped < {},
            lambda: untyped <= {},
            lambda: untyped > {},
            lambda: untyped >= {},
            lambda: untyped.fromkeys([1]),
        ):
            with pytest.raises(TypeError):
                op()


class TestForwardedBehaviour:
    """The previously-missing forwards read through to the live container."""

    def test_dict_popitem(self, pdict: _ProxyDict, ctx: _Ctx) -> None:
        assert pdict.popitem() == (2, "b")
        assert ctx.d == {1: "a"}

    def test_dict_reflected_or(self, pdict: _ProxyDict) -> None:
        assert {0: "z"} | pdict == {0: "z", 1: "a", 2: "b"}

    def test_list_ne(self, plist: _ProxyList) -> None:
        assert not (plist != [10, 20])
        assert plist != [99]

    def test_list_reversed(self, plist: _ProxyList) -> None:
        assert list(reversed(plist)) == [20, 10]

    def test_list_ordering(self, plist: _ProxyList) -> None:
        assert plist < [99]
        assert plist > []
        assert plist <= [10, 20]
        assert plist >= [10, 20]

    def test_list_multiplication(self, plist: _ProxyList) -> None:
        assert plist * 2 == [10, 20, 10, 20]
        assert 2 * plist == [10, 20, 10, 20]

    def test_list_imul_mutates_target(self, plist: _ProxyList, ctx: _Ctx) -> None:
        plist *= 2
        assert ctx.lst == [10, 20, 10, 20]

    def test_list_reflected_add(self, plist: _ProxyList) -> None:
        assert [1] + plist == [1, 10, 20]


class TestCFastPathBypasses:
    """Operations no Python-level override can intercept.

    ``json.dumps`` and unbound ``dict``/``list`` calls read a subclass's
    internal table directly, so they see the proxy's empty own storage. These
    assertions pin the *wrong* answers deliberately: they document the boundary
    and will fail loudly if a future Python makes these interceptable, which is
    the moment to revisit the warning in ``state_proxies``' module docstring.
    """

    def test_json_dumps_sees_empty_storage(self, pdict: _ProxyDict) -> None:
        assert json.loads(json.dumps(pdict)) == {}

    def test_unbound_dict_calls_bypass_forwarding(self, pdict: _ProxyDict) -> None:
        assert list(dict.keys(pdict)) == []
        assert list(dict.items(pdict)) == []
        assert dict.get(pdict, 1) is None
        assert dict.copy(pdict) == {}

    def test_documented_workarounds_are_correct(self, pdict: _ProxyDict, plist: _ProxyList) -> None:
        """What production code actually does, and why it is safe."""
        assert dict(pdict) == {1: "a", 2: "b"}
        assert {**pdict} == {1: "a", 2: "b"}
        assert sorted(pdict) == [1, 2]
        assert {k: v for k, v in pdict.items()} == {1: "a", 2: "b"}
        assert list(plist) == [10, 20]
        assert sorted(plist) == [10, 20]


class TestReExportParity:
    """``vtsearch.state`` must re-export every public ``vtscore.state`` name."""

    @staticmethod
    def _library_api() -> set[str]:
        """Public names ``vtscore/state/__init__.py`` deliberately exposes.

        Read statically from the source rather than from ``vars()``: the live
        module namespace also carries incidentally-imported submodules
        (``core``, ``votes``, …) and typing helpers (``Any``, ``Callable``),
        which are not API and are not the shim's job to re-export.
        """
        import ast

        import vtscore.state as lib

        tree = ast.parse(pathlib.Path(lib.__file__).read_text())
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("vtscore.state"):
                names |= {alias.asname or alias.name for alias in node.names}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        return {n for n in names if not n.startswith("_")}

    def test_no_dropped_public_names(self) -> None:
        import vtsearch.state as app

        dropped = {name for name in self._library_api() if not hasattr(app, name)}
        assert not dropped, (
            f"vtsearch.state drops {sorted(dropped)} from vtscore.state. The shim's explicit "
            f"import list has to grow whenever the library gains a public name, or app code "
            f"importing it from vtsearch.state fails with ImportError."
        )

    def test_library_api_is_non_trivial(self) -> None:
        """Guard the guard: a parsing regression must not silently pass the check."""
        assert len(self._library_api()) > 50
