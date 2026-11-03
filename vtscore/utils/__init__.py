"""Small leftover helpers.

State, plugins, sync, concurrency, security, and audio helpers have moved
to their own top-level modules - see :mod:`vtscore.state`, :mod:`vtscore.plugins`,
:mod:`vtscore.sync`, :mod:`vtscore.concurrency`, :mod:`vtscore.security`, and
:mod:`vtscore.media.audio`.

Remaining here:
- :mod:`vtscore.utils.hashing` - ``content_md5``/``content_sha1``/``file_md5``
  content fingerprints. Use these instead of ``hashlib`` directly so the
  non-security declaration stays in one place (keeps MD5 working on
  FIPS-enabled hosts).
- :mod:`vtscore.utils.io` - server-side file I/O for plugins: JSON reads,
  atomic and resumable writes, CSV-cell escaping, a cross-process file lock.
  It was the top-level ``vtscore.io`` (#5398), now a deprecated alias of it.
- :mod:`vtscore.utils.optional_deps` - actionable messages for the opt-out AGPL
  packages (``ultralytics``, ``PyMuPDF``), which a default install has but an
  ``VTSEARCH_NO_AGPL=1`` install deliberately lacks.
- :mod:`vtscore.utils.synthetic` - Synthetic media generators for the offline demo importer.

Deprecated aliases (#5399), each handing back the module its helpers were
folded into:

- :mod:`vtscore.utils.hits` - now :mod:`vtscore.datasets.labelset`.
- :mod:`vtscore.utils.import_metadata` - now :mod:`vtscore.media.torch_utils.torch_setup`.
- :mod:`vtscore.utils.scores` - now :mod:`vtscore.training.thresholds.knobs`.
"""
