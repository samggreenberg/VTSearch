"""Small leftover helpers.

State, plugins, sync, concurrency, security, and audio helpers have moved
to their own top-level packages - see ``vtscore/state``, ``vtscore/plugins``,
``vtscore/sync``, ``vtscore/concurrency``, ``vtscore/security``, and
``vtscore/media/audio``.

Remaining here:
- :mod:`vtscore.utils.hashing` - ``content_md5``/``content_sha1``/``file_md5``
  content fingerprints. Use these instead of ``hashlib`` directly so the
  non-security declaration stays in one place (keeps MD5 working on
  FIPS-enabled hosts).
- :mod:`vtscore.utils.optional_deps` - actionable messages for the opt-out AGPL
  packages (``ultralytics``, ``PyMuPDF``), which a default install has but an
  ``VTSEARCH_NO_AGPL=1`` install deliberately lacks.
- :mod:`vtscore.utils.synthetic` - Synthetic media generators for the offline demo importer.

Deprecated aliases (#5399), each handing back the module its helpers were
folded into:

- :mod:`vtscore.utils.hits` - now :mod:`vtscore.datasets.labelset`.
- :mod:`vtscore.utils.import_metadata` - now :mod:`vtscore.media.torch_setup`.
- :mod:`vtscore.utils.scores` - now :mod:`vtscore.training.thresholds.knobs`.
"""
