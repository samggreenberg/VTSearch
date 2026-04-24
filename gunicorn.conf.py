"""Gunicorn configuration for VTSearch.

VTSearch keeps all dataset/model state in-process (see CLAUDE.md: multi-dataset
context, global registries, RLock-protected mutable state). Multiple worker
processes would each hold their own independent copy, so we run a single
worker and rely on threads for concurrency — matching the Flask dev server's
``threaded=True`` behaviour.

Environment overrides:
  VTSEARCH_BIND          — host:port (default 0.0.0.0:5000)
  VTSEARCH_THREADS       — threads per worker (default 8)
  VTSEARCH_TIMEOUT       — worker timeout in seconds, 0 disables (default 120)
"""

import os

bind = os.environ.get("VTSEARCH_BIND", "0.0.0.0:5000")

workers = 1
worker_class = "gthread"
threads = int(os.environ.get("VTSEARCH_THREADS", "8"))

timeout = int(os.environ.get("VTSEARCH_TIMEOUT", "120"))
graceful_timeout = 30
keepalive = 5

accesslog = None
errorlog = "-"
loglevel = os.environ.get("VTSEARCH_LOG_LEVEL", "warning").lower()

proc_name = "vtsearch"
