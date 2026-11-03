"""``python app.py``: the command line, apart from the WSGI ``app`` gunicorn imports.

* :mod:`~vtsearch.cli.main` - the argparse definition, the early-exit
  informational flags, the ``--pipeline`` / ``--autodetect`` dispatch and the
  dev-server launch.
* :mod:`~vtsearch.cli.port_preflight` - the single-instance lock and the check
  for an old server still holding the port, run before the dev server binds.

The autodetect pipeline the CLI dispatches to is library code, in
:mod:`vtscore.cli`.
"""
