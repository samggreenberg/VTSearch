"""Clipping machinery shared by every media type.

The :class:`~vtscore.media.clipping.clipper.MediaClipper` and
:class:`~vtscore.media.clipping.cleaner.MediaCleaner` ABCs (both re-exported
from :mod:`vtscore.media`), the ``origin.params`` clip-recipe parser, and the
lazy replay that derives a clip's bytes from its source on demand.  Each media
type's own ``clipper.py`` / ``cleaner.py`` subclasses these.
"""
