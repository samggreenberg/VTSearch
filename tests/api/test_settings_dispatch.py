"""Drift guards for the settings declarations.

A single setting is currently declared in several parallel places:

* a Pydantic field on :class:`ServerSettings` / :class:`UserSettings`
  (``vtsearch.settings_models``) - the source of truth for type / default
  / range / enum, and the source the ``get_<key>`` / ``set_<key>``
  accessors in :mod:`vtsearch.settings` are generated from;
* a marshmallow field on ``SettingsUpdateSchema`` (loadable) and
  ``AppSettingsSchema`` (dumpable) in ``vtsearch.schemas.settings``;
* a dispatch entry in :mod:`vtsearch.routes.settings.api`
  (``_SCALAR_SETTERS`` / ``_CUSTOM_SETTERS``), or an explicit exemption.

These used to be kept in sync by hand. The route ``_SCALAR_SETTERS`` table
is now *generated* from the update schema, and these tests assert the rest
of the declarations cannot silently diverge: a new settable field added to
the schema must be dispatchable (or explicitly exempted), and the schema
fields must stay backed by real Pydantic fields and accessors.
"""

from __future__ import annotations

from vtsearch import settings
from vtsearch.routes.settings import api
from vtsearch.schemas.settings import AppSettingsSchema, SettingsUpdateSchema
from vtsearch.settings_models import ServerSettings, UserSettings


def _loadable_update_keys() -> set[str]:
    return {name for name, field in SettingsUpdateSchema().fields.items() if not field.dump_only}


def _pydantic_fields() -> set[str]:
    return set(ServerSettings.model_fields) | set(UserSettings.model_fields)


class TestDispatchCoversSchema:
    """Every settable schema field must be wired or explicitly exempted."""

    def test_every_loadable_key_is_dispatchable_or_exempt(self):
        dispatchable = set(api._SCALAR_SETTERS) | set(api._CUSTOM_SETTERS)
        exempt = api._READ_ONLY_KEYS | api._NON_PUT_KEYS
        unrouted = _loadable_update_keys() - dispatchable - exempt
        assert not unrouted, (
            f"Settable settings with no dispatch entry: {sorted(unrouted)}. "
            "Add a setter to _SCALAR_SETTERS (or _CUSTOM_SETTERS), or list the "
            "key in _NON_PUT_KEYS / _READ_ONLY_KEYS if it is intentionally not "
            "settable via PUT /api/settings."
        )

    def test_no_orphan_dispatch_entries(self):
        loadable = _loadable_update_keys()
        scalar_orphans = set(api._SCALAR_SETTERS) - loadable
        custom_orphans = set(api._CUSTOM_SETTERS) - loadable
        assert not scalar_orphans, f"_SCALAR_SETTERS keys absent from the update schema: {sorted(scalar_orphans)}"
        assert not custom_orphans, f"_CUSTOM_SETTERS keys absent from the update schema: {sorted(custom_orphans)}"

    def test_scalar_and_custom_are_disjoint(self):
        overlap = set(api._SCALAR_SETTERS) & set(api._CUSTOM_SETTERS)
        assert not overlap, f"Keys in both dispatch tables (ambiguous): {sorted(overlap)}"

    def test_non_put_keys_are_loadable_but_unrouted(self):
        """Each _NON_PUT_KEYS entry is a real, settable-looking schema field
        deliberately excluded from dispatch - not a typo."""
        loadable = _loadable_update_keys()
        dispatchable = set(api._SCALAR_SETTERS) | set(api._CUSTOM_SETTERS)
        for key in api._NON_PUT_KEYS:
            assert key in loadable, f"_NON_PUT_KEYS entry {key!r} is not a loadable schema field"
            assert key not in dispatchable, f"_NON_PUT_KEYS entry {key!r} is also dispatched (contradiction)"


class TestSchemaBackedByModels:
    """Marshmallow fields must stay backed by Pydantic fields + accessors."""

    def test_loadable_keys_are_pydantic_fields(self):
        missing = _loadable_update_keys() - _pydantic_fields()
        assert not missing, f"Update-schema fields with no Pydantic field: {sorted(missing)}"

    def test_app_schema_extras_are_only_computed_views(self):
        # AppSettingsSchema may carry read-only resolver views that have no
        # stored Pydantic field; everything else must be model-backed.
        extras = set(AppSettingsSchema().fields) - _pydantic_fields()
        assert extras == {
            "effective_solo_embedder_per_media_type",
        }, f"Unexpected non-model AppSettingsSchema fields: {sorted(extras)}"

    def test_scalar_setters_have_real_accessors(self):
        # Generated scalar entries (everything but the state-tier overrides)
        # must resolve to an actual settings.set_<key> callable.
        for key in api._SCALAR_SETTERS:
            if key in api._STATE_TIER_SETTERS:
                continue
            assert getattr(settings, f"set_{key}", None) is api._SCALAR_SETTERS[key], (
                f"_SCALAR_SETTERS[{key!r}] is not settings.set_{key}"
            )
