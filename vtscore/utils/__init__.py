"""Small leftover helpers.

State, plugins, sync, concurrency, security, and audio helpers have moved
to their own top-level packages - see ``vtsearch/state``, ``vtsearch/plugins``,
``vtsearch/sync``, ``vtsearch/concurrency``, ``vtsearch/security``, and
``vtsearch/media/audio``.

Remaining here:
- :mod:`vtscore.utils.hits` - ``build_media_hit`` helper for scoring/route hit dicts.
- :mod:`vtscore.utils.scores` - ``sigmoid_to_finite_scores``/``finite_or`` for
  JSON-safe MLP scoring (no NaN/Infinity leaks to strict clients).
- :mod:`vtscore.utils.synthetic` - Synthetic media generators for the offline demo importer.
"""
