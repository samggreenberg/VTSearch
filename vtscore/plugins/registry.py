"""The generic auto-discovering plugin registry.

:class:`PluginRegistry` scans a package for sub-packages (and optionally flat
modules) exposing a sentinel attribute, then an optional
:mod:`importlib.metadata` entry-point group for third-party plugins;
:func:`make_plugin_registry` wraps one in the ``(get, list)`` accessor pair
every family re-exports.  :class:`EntryPointTombstone` stands in for an
entry-point plugin whose own import failed.

Import these from :mod:`vtscore.plugins`, which re-exports them.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import sys
import threading
import warnings
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar


# ---------------------------------------------------------------------------
# Entry-point tombstone: defer a plugin's own ImportError to first use
# ---------------------------------------------------------------------------


class EntryPointTombstone:
    """Placeholder for an entry-point plugin whose *own* import raised.

    When a third-party entry point fails to load (e.g. a missing optional
    dependency such as ``open_clip`` for a ``siglip_l`` embedder), we don't
    want that single failure to crash discovery for every other plugin.
    Instead the registry registers one of these under the entry point's name
    so the plugin still *resolves* via :meth:`PluginRegistry.get`, and the
    original error is re-raised only when the plugin is actually invoked
    (any attribute access other than :attr:`name` / private / dunder names).

    Tombstones are deliberately kept out of :meth:`PluginRegistry.list` so
    listing/serialising the plugin family (which touches ``to_dict``,
    ``display_name``, ``ui_mode``, ...) never trips over them; the broken
    plugin only bites the caller who deliberately asks for it by name.

    The object is a plain, picklable value: :attr:`name`, the entry-point
    ``value`` string, and the captured exception all live in ``__dict__``,
    so ``pickle.loads(pickle.dumps(tombstone))`` round-trips and the restored
    copy re-raises the same error on use.
    """

    def __init__(self, name: str, value: str, error: BaseException) -> None:
        self.name = name
        self._ep_value = value
        self._error = error

    def __getattr__(self, attr: str) -> Any:
        # Private / dunder lookups (``__reduce_ex__``, ``__getstate__``,
        # ``__deepcopy__``, ...) must fall through with AttributeError so
        # pickle and copy keep working; only "real" plugin attribute/method
        # access re-raises the deferred import error.
        if attr.startswith("_"):
            raise AttributeError(attr)
        raise ImportError(
            f"Plugin {self.name!r} (entry point {self._ep_value!r}) is unavailable "
            f"because its import failed: {self._error}"
        ) from self._error


# ---------------------------------------------------------------------------
# PluginRegistry: generic auto-discovery registry
# ---------------------------------------------------------------------------

T = TypeVar("T")


class PluginRegistry(Generic[T]):
    """Auto-discovering plugin registry.

    Parameters
    ----------
    package:
        Fully-qualified dotted name of the package whose sub-packages will be
        scanned, e.g. ``"vtscore.exporters"``.
    sentinel:
        Module-level attribute name to look for in each sub-package, e.g.
        ``"EXPORTER"``.
    label:
        Human-readable noun used in warning messages, e.g. ``"exporter"``.
    discover_modules:
        When ``True``, also scan flat ``.py`` files (not just sub-packages)
        for the sentinel.  Useful for plugin families where each plugin is
        a single module rather than a sub-package (e.g. converters, sources).
    entry_point_group:
        Optional :mod:`importlib.metadata` entry-point group to scan after
        the local package scan, e.g. ``"vtscore.importers"``.  Third-party
        packages can register a plugin by adding an entry to this group in
        their own ``pyproject.toml`` / ``setup.cfg``::

            [project.entry-points."vtscore.importers"]
            my_importer = "my_pkg.my_module:IMPORTER"

        The entry point must resolve to an already-instantiated plugin
        object (same shape as a sentinel attribute); typically you point
        directly at the module's ``IMPORTER`` / ``EXPORTER`` / ... sentinel.
    eager:
        When ``True`` (the default) discovery runs at construction time so
        the registry is populated by the time the constructor returns.
        Set ``False`` to defer discovery until the first :meth:`get` /
        :meth:`list` call; useful in tests that want to inspect the
        pre-discovery state or simulate concurrent first access.
    """

    def __init__(
        self,
        package: str,
        sentinel: str,
        label: str,
        *,
        discover_modules: bool = False,
        entry_point_group: str | None = None,
        eager: bool = True,
    ) -> None:
        self._package = package
        self._sentinel = sentinel
        self._label = label
        self._discover_modules = discover_modules
        self._entry_point_group = entry_point_group
        self._items: dict[str, T] = {}
        #: Entry-point plugins whose own import raised, keyed by entry-point
        #: name.  Resolvable via :meth:`get` (re-raising the original error on
        #: use) but deliberately excluded from :meth:`list`.
        self._tombstones: dict[str, EntryPointTombstone] = {}
        self._discovered = False
        self._discovering = False
        #: Re-entrant by design: a module scanned during :meth:`_discover` may
        #: call :meth:`get` / :meth:`list` on this same registry at import
        #: time, on the discovering thread.  An ``RLock`` lets that call back
        #: in so the ``_discovering`` guard in :meth:`_ensure_discovered` can
        #: hand it the partial registry; a plain ``Lock`` would deadlock the
        #: process before the guard was ever reached.
        self._lock = threading.RLock()
        if eager:
            self._ensure_discovered()

    # -- Discovery ----------------------------------------------------------

    def _discover(self) -> None:
        """Scan sub-packages (and optionally flat modules) for sentinel objects.

        Uses direct filesystem scanning so that symlinked directories are
        reliably discovered.  A symlink to a package directory (containing
        ``__init__.py``) is treated identically to a regular sub-package.

        When :attr:`_discover_modules` is ``True``, also scans ``.py`` files
        (excluding ``__init__.py`` and ``base.py``) for the sentinel.
        """
        parent = importlib.import_module(self._package)
        if parent.__file__ is None:
            raise RuntimeError(f"Cannot discover plugins under namespace package {self._package!r}")
        package_dir = Path(parent.__file__).parent
        for entry in sorted(package_dir.iterdir()):
            if entry.name.startswith((".", "_")):
                continue

            # Sub-packages (directories with __init__.py)
            if entry.is_dir():
                # Skip names containing dots; they aren't valid Python
                # identifiers and would be misinterpreted as nested module
                # paths by importlib (e.g. "foo.symbolic_link" would try to
                # import package "foo" first).  This commonly happens with
                # symlinks whose names include an extension or suffix.
                if "." in entry.name:
                    continue
                init_path = entry / "__init__.py"
                if not init_path.exists():
                    continue
                self._try_load(entry.name, file_path=init_path if entry.is_symlink() else None)
            # Flat modules (.py files)
            elif self._discover_modules and entry.is_file() and entry.suffix == ".py":
                if entry.name in ("__init__.py", "base.py"):
                    continue
                self._try_load(entry.stem, file_path=entry if entry.is_symlink() else None)

        if self._entry_point_group:
            self._discover_entry_points()

    def _discover_entry_points(self) -> None:
        """Load third-party plugins registered via :mod:`importlib.metadata`.

        Each entry point in :attr:`_entry_point_group` is resolved to an
        object that's treated like a sentinel value; its ``.name`` is the
        registry key.  When an entry point's *own* import raises (e.g. a
        missing optional dependency), the failure is surfaced as a warning
        and a :class:`EntryPointTombstone` is registered under the entry-point
        name so the plugin still resolves via :meth:`get` and the original
        error is deferred to first use — a single bad third-party plugin can't
        break discovery for the rest of the registry.

        Built-in plugins (discovered by the package scan above) take
        precedence: an entry point whose name clashes with a built-in is
        skipped.  This prevents an installed third-party package from
        accidentally shadowing a core plugin.
        """
        group = self._entry_point_group
        if group is None:
            return
        try:
            eps = importlib.metadata.entry_points(group=group)
        except Exception as exc:  # pragma: no cover
            warnings.warn(
                f"Failed to read entry-point group {self._entry_point_group!r}: {exc}",
                stacklevel=2,
            )
            return
        for ep in eps:
            try:
                plugin = ep.load()
            except Exception as exc:
                # The plugin's *own* import raised (e.g. a missing optional
                # dependency).  Don't crash discovery: register a tombstone
                # under the entry-point name so the plugin still resolves and
                # the original error is deferred to first use.  A built-in of
                # the same name still wins (it's already in ``_items``).
                warnings.warn(
                    f"Failed to load {self._label} entry point {ep.name!r} from {ep.value!r}: {exc}; "
                    f"deferring the error to first use of {ep.name!r}",
                    stacklevel=2,
                )
                if ep.name not in self._items:
                    self._tombstones[ep.name] = EntryPointTombstone(ep.name, ep.value, exc)
                continue
            plugin_name = getattr(plugin, "name", None)
            if not plugin_name:
                warnings.warn(
                    f"{self._label} entry point {ep.name!r} from {ep.value!r} has no 'name' attribute; skipped",
                    stacklevel=2,
                )
                continue
            if plugin_name in self._items:
                warnings.warn(
                    f"{self._label} entry point {ep.name!r} from "
                    f"{ep.value!r} clashes with built-in plugin "
                    f"{plugin_name!r}; skipped",
                    stacklevel=2,
                )
                continue
            self._items[plugin_name] = plugin

    def _try_load(self, module_name: str, *, file_path: Path | None = None) -> None:
        """Import *module_name* under this registry's package and register its sentinel.

        When *file_path* is given (symlinked entries), uses
        :func:`importlib.util.spec_from_file_location` to load the module
        directly from the resolved path.  Python's default ``FileFinder`` can
        miss symlinked packages on some platforms because its directory cache
        may not follow symlinks consistently.
        """
        full_name = f"{self._package}.{module_name}"
        try:
            if file_path is not None:
                resolved = file_path.resolve()
                is_package = resolved.name == "__init__.py"
                spec = importlib.util.spec_from_file_location(
                    full_name,
                    str(resolved),
                    submodule_search_locations=[str(resolved.parent)] if is_package else None,
                )
                if spec is None or spec.loader is None:  # pragma: no cover
                    return
                mod = importlib.util.module_from_spec(spec)
                sys.modules[full_name] = mod
                spec.loader.exec_module(mod)
            else:
                mod = importlib.import_module(full_name)
            plugin = getattr(mod, self._sentinel, None)
            if plugin is not None:
                if plugin.name in self._items:
                    warnings.warn(
                        f"{self._label} module {module_name!r} declares name "
                        f"{plugin.name!r} which is already registered; skipped",
                        stacklevel=2,
                    )
                    return
                self._items[plugin.name] = plugin
        except Exception as exc:  # pragma: no cover
            # Clean up partially-registered module on failure.
            sys.modules.pop(full_name, None)
            warnings.warn(
                f"Failed to load {self._label} '{module_name}': {exc}",
                stacklevel=2,
            )

    def _ensure_discovered(self) -> None:
        if self._discovered:
            return
        with self._lock:
            if not self._discovered:
                # Guard against re-entrant discovery on the discovering
                # thread.  When discover_modules is True, importing a sibling
                # module may trigger get()/list() on this registry before
                # discovery finishes (e.g. runner.py importing from its own
                # package's __init__).  ``self._lock`` is an RLock precisely
                # so that call re-enters here instead of deadlocking; we
                # return early with a partial registry, and the ongoing
                # discovery completes shortly and fills in the rest.  Other
                # threads never see _discovering True: they block on the
                # RLock until discovery is done and _discovered is set.
                if self._discovering:
                    return
                self._discovering = True
                try:
                    self._discover()
                finally:
                    self._discovering = False
                self._discovered = True

    # -- Public API ---------------------------------------------------------

    def get(self, name: str) -> T | None:
        """Return the registered plugin with *name*, or ``None``.

        Falls back to a tombstone for an entry-point plugin whose own import
        failed (see :class:`EntryPointTombstone`); the returned object
        re-raises the original error the moment it is actually invoked.
        """
        self._ensure_discovered()
        item = self._items.get(name)
        if item is not None:
            return item
        return self._tombstones.get(name)  # type: ignore[return-value]

    def list(self) -> list[T]:
        """Return all registered plugins in discovery order.

        Tombstones for failed entry-point imports are excluded so that
        listing/serialising a plugin family never trips over a broken plugin;
        such plugins are reachable only by an explicit :meth:`get`.
        """
        self._ensure_discovered()
        return list(self._items.values())

    @property
    def package(self) -> str:
        """Dotted name of the package scanned for built-in plugins."""
        return self._package

    @property
    def sentinel(self) -> str:
        """Module-level attribute a built-in plugin exposes, e.g. ``"EXPORTER"``."""
        return self._sentinel

    @property
    def entry_point_group(self) -> str | None:
        """Entry-point group third-party plugins register into, or ``None``."""
        return self._entry_point_group


# ---------------------------------------------------------------------------
# Factory helper: collapses the per-package boilerplate into one call
# ---------------------------------------------------------------------------


def make_plugin_registry(
    package: str,
    sentinel: str,
    label: str,
    *,
    discover_modules: bool = False,
    entry_point_group: str | None = None,
    eager: bool = True,
) -> tuple[Callable[[str], Any], Callable[[], list[Any]]]:
    """Create a :class:`PluginRegistry` and return its ``(get, list)`` accessors.

    Shorthand for the boilerplate repeated across every plugin ``__init__.py``::

        from vtscore.plugins import make_plugin_registry

        get_importer, list_importers = make_plugin_registry(
            package=__name__,
            sentinel="IMPORTER",
            label="dataset importer",
            entry_point_group="vtscore.importers",
        )

    Parameters are forwarded to :class:`PluginRegistry`.
    """
    registry: PluginRegistry = PluginRegistry(
        package=package,
        sentinel=sentinel,
        label=label,
        discover_modules=discover_modules,
        entry_point_group=entry_point_group,
        eager=eager,
    )
    return registry.get, registry.list
