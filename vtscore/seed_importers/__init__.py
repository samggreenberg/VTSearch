"""Seed Importer plugin registry.

Seed importers produce a *batch of unlabeled seed media* for a new blank
detector — media that are "close but not quite" what the user is hunting
for, carrying no ``good`` / ``bad`` verdict.  See
:mod:`vtscore.seed_importers.base` for the plugin contract and for why a
seed is a query rather than a label.

No seed importers ship in-tree: the family exists so a third-party package
can add one through the ``vtscore.seed_importers`` entry-point group,
exposing an already-instantiated ``SEED_IMPORTER``.  An in-tree plugin
would live in a sub-package of this package exposing the same sentinel.
"""

from vtscore.plugins import make_plugin_registry
from vtscore.seed_importers.base import SeedImporter, SeedMediaItem

get_seed_importer, list_seed_importers = make_plugin_registry(
    package=__name__,
    sentinel="SEED_IMPORTER",
    label="seed importer",
    entry_point_group="vtscore.seed_importers",
)

__all__ = [
    "SeedImporter",
    "SeedMediaItem",
    "get_seed_importer",
    "list_seed_importers",
]
