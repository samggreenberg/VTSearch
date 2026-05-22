"""Dataset configurations — built from the media type registry.

``DEMO_DATASETS`` is assembled at import time from every registered
:class:`~vtscore.media.base.MediaType`'s
:attr:`~vtscore.media.base.MediaType.demo_datasets` list.  Adding a new
media type (and registering it in ``vtsearch/media/__init__.py``)
automatically makes its demo datasets appear here with no further edits.
"""

from vtscore.media import all_demo_datasets

DEMO_DATASETS: dict = all_demo_datasets()
