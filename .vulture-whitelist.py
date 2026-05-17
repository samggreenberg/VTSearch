"""Whitelist for vulture's dead-code detector.

Vulture finds defined-but-never-referenced names. This file lists symbols
that VTSearch DOES use, but only reflectively — so static analysis can't
see the reference. Running vulture with this file as a second argument
suppresses the corresponding false positives:

    vulture vtsearch .vulture-whitelist.py --min-confidence 80

Add a new entry whenever you confirm that a vulture finding is reflective
or framework-managed rather than actually dead.
"""

# Plugin sentinels. Each `<FAMILY> = SomeClass` line at the bottom of a
# plugin module is discovered by the `PluginRegistry` scanner via the
# matching attribute name. Vulture sees the assignment but no reference;
# discovery happens at import time through `getattr`.
IMPORTER  # noqa: F821
EXPORTER  # noqa: F821
CONVERTER  # noqa: F821
SOURCE  # noqa: F821
PROCESSOR  # noqa: F821
PICKER  # noqa: F821
MEDIA_TYPE  # noqa: F821
SETTINGS_IMPORTER  # noqa: F821
SETTINGS_EXPORTER  # noqa: F821
SETTINGS_SOURCE  # noqa: F821
LABEL_IMPORTER  # noqa: F821
LABELSET_SOURCE  # noqa: F821
PickerView  # noqa: F821

# argparse.Action.__call__ requires the `option_string` parameter even
# when the action ignores it.
option_string  # noqa: F821
