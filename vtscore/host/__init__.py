"""The slots a host application fills: what ``vtscore`` asks of the app running it.

``vtscore`` runs without ``vtsearch`` and without Flask, so wherever the library
needs something only a host can answer, it holds a module-level slot with a
library default and a ``register_*`` function for the host to fill it.  Most of
those slots are declared next to the code that calls them; this package holds
the two that are about the host as a whole:

* :mod:`~vtscore.host.achievements` - the achievement-recording seam.  The
  library raises a handful of events (a vote, a dataset load, a detector import,
  a find run) and the host decides what crediting one means; with no host it is
  a no-op.
* :mod:`~vtscore.host.seams` - the complete list of host seams, and the
  snapshot / restore pair tests use to put them back after one installs its
  own.

``vtscore.achievements_hooks`` and ``vtscore.host_seams`` are deprecated aliases
of these two modules - the same module objects, so a recorder registered
through either name is the one the library calls.
"""
