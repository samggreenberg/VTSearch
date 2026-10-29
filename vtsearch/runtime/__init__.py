"""What the running server does outside any one route.

* :mod:`~vtsearch.runtime.request_hooks` - the Flask ``before_request`` /
  ``after_request`` / ``teardown_request`` handlers: request id, user and
  dataset/detector binding, auth enforcement, response shaping.
* :mod:`~vtsearch.runtime.context_threads` -
  :func:`~vtsearch.runtime.context_threads.spawn`, a background thread that
  carries the spawning request's user and contexts.  (Not ``threading``: that
  name is the standard library's.)
* :mod:`~vtsearch.runtime.autofind` - running a user's AutoFind detectors on a
  dataset, inline or as a background task.
* :mod:`~vtsearch.runtime.import_hooks` - the functions a server admin names to
  run when a user's dataset import finishes.
* :mod:`~vtsearch.runtime.logging_config` - structured logging tagged with the
  request's user, dataset, detector and request id.
* :mod:`~vtsearch.runtime.diagnose` - the ``VTSEARCH_DIAGNOSE`` preset and the
  startup line that records which bars a session ran at.
* :mod:`~vtsearch.runtime.torch_threads` - how many native-math threads the
  server uses.

This ``__init__`` imports nothing, and must keep it that way:
``app.py`` imports :mod:`~vtsearch.runtime.torch_threads` before anything
loads torch or numpy, and importing a submodule runs this file first.
"""
