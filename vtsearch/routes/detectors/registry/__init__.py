"""Blueprint for detector-registry routes (the in-memory detector catalog).

Every registered detector is backed by a labelset file on disk and a head
that lives only in :class:`~vtsearch.state.DetectorContext` once the user
loads the detector.

The routes share one blueprint, ``detectors_registry_bp``, and are split by
what they do to a detector:

* :mod:`~vtsearch.routes.detectors.registry.listing` - list the registry,
  register a new detector, seed one from a label importer.
* :mod:`~vtsearch.routes.detectors.registry.loading` - load, unload and delete
  a detector, and cancel a load.
* :mod:`~vtsearch.routes.detectors.registry.editing` - rename a detector,
  toggle its AutoFind flag, set its readers.
* :mod:`~vtsearch.routes.detectors.registry.stats` - the Stats window and
  Browse Positives.

The blueprint and the helpers the create and rename routes share live in
:mod:`~vtsearch.routes.detectors.registry._common`, so no route module imports
this package.  A route registers when its module is imported, which is what
the import below is for: a module left out of it loses its endpoints.

Built on ``flask_smorest`` so the routes are described in
``/api/openapi.json``, except for ``POST /from-labelset/<importer>``, which
takes plugin-dependent fields and declares no schema (see
"Routes with no typed schema" in ``docs/API.md``).

**Patch targets** follow the route module that looks a name up: stubbing the
listing's ``_read_detector`` means patching
:mod:`~vtsearch.routes.detectors.registry.listing`, not this package.
"""

from vtsearch.routes.detectors.registry import editing, listing, loading, stats  # noqa: F401
from vtsearch.routes.detectors.registry._common import detectors_registry_bp

__all__ = ["detectors_registry_bp"]
