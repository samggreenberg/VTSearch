"""Settings import/export plugin system.

Three families live here: one-shot ``importers`` and ``exporters``, and
``sources`` - bidirectional sync targets that stay bound to a user and are
consulted on every settings read and write.

Sources are the subtle one. The engine behind them (freshness probing, the
dirty-key contract that decides whether an upstream value may overwrite a
local edit, the cross-process dedup marker, and the lock ordering a source
method runs under) is documented in
``docs/EXTENDING-plugins.md`` under "Adding a Settings Source" ->
"How the sync engine works". Read it before writing a new source; the
contract is not discoverable from :class:`SettingsSource` alone, and a
source that violates it fails by losing user edits under load rather than
by raising.
"""
