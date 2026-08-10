"""Small leftover helpers.

State, plugins, sync, concurrency, security, and audio helpers have moved
to their own top-level packages - see ``vtsearch/state``, ``vtsearch/plugins``,
``vtsearch/sync``, ``vtsearch/concurrency``, ``vtsearch/security``, and
``vtsearch/media/audio``.

Remaining here:
- :mod:`vtscore.utils.hashing` - ``content_md5``/``content_sha1``/``file_md5``
  content fingerprints. Use these instead of ``hashlib`` directly so the
  non-security declaration stays in one place (keeps MD5 working on
  FIPS-enabled hosts).
- :mod:`vtscore.utils.hits` - ``build_media_hit`` helper for scoring/route hit dicts.
- :mod:`vtscore.utils.optional_deps` - actionable messages for the opt-out AGPL
  packages (``ultralytics``, ``PyMuPDF``), which a default install has but an
  ``VTSEARCH_NO_AGPL=1`` install deliberately lacks.
- :mod:`vtscore.utils.scores` - ``sigmoid_to_finite_scores``/``finite_or`` for
  JSON-safe MLP scoring (no NaN/Infinity leaks to strict clients).
- :mod:`vtscore.utils.synthetic` - Synthetic media generators for the offline demo importer.
"""
