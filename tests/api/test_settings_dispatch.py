"""Drift guards for the settings declarations.

A single setting is currently declared in several parallel places:

* a Pydantic field on :class:`ServerSettings` / :class:`UserSettings`
  (``vtsearch.settings_models``) - the source of truth for type / default
  / range / enum, and the source the ``get_<key>`` / ``set_<key>``
  accessors in :mod:`vtsearch.settings` are generated from;
* a marshmallow field on ``SettingsUpdateSchema`` (loadable) and
  ``AppSettingsSchema`` (dumpable) in ``vtsearch.schemas.settings``;
* a dispatch entry in :mod:`vtsearch.routes.settings.api`
  (``_SCALAR_SETTERS`` / ``_CUSTOM_SETTERS``), or an explicit exemption;
* a ``TYPE_CHECKING`` stub in :mod:`vtsearch.settings`, which is what makes
  the dynamically-installed ``get_<key>`` / ``set_<key>`` accessors visible
  to pyright and ruff.

These used to be kept in sync by hand. The route ``_SCALAR_SETTERS`` table
is now *generated* from the update schema, and these tests assert the rest
of the declarations cannot silently diverge, in **both** directions:

* schema -> model: a new settable field must be dispatchable (or explicitly
  exempted), and every schema field must be backed by a real Pydantic field
  and accessor;
* model -> schema/stubs: every model field must have a stub, and must either
  be dumped by ``AppSettingsSchema`` or be listed as a deliberate omission
  with its reason. Marshmallow drops undeclared keys from ``dump()`` without
  erroring, so nothing else would ever report a field missing from
  ``GET /api/settings``.
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


def _stubbed_accessor_keys() -> set[str]:
    """Keys declared in :mod:`vtsearch.settings`'s ``TYPE_CHECKING`` block.

    Parsed from source rather than imported: the block is erased at runtime,
    so the stubs exist only as text. ``ast`` is used instead of a regex so a
    ``get_``-prefixed name in a comment or docstring can't be miscounted.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(settings))
    for node in tree.body:
        if isinstance(node, ast.If) and ast.unparse(node.test) == "TYPE_CHECKING":
            return {
                child.name[len("get_") :]
                for child in node.body
                if isinstance(child, ast.FunctionDef) and child.name.startswith("get_")
            }
    raise AssertionError("no `if TYPE_CHECKING:` block found in vtsearch.settings")


class TestStubsMirrorTheModels:
    """The hand-written ``TYPE_CHECKING`` stubs must mirror the Pydantic fields.

    The stubs are the only thing making the generated ``get_<key>`` /
    ``set_<key>`` accessors visible to pyright and ruff. A model field with no
    stub is not a cosmetic gap: every call site needs a
    ``# type: ignore[name-defined]  # noqa: F821`` pair to lint, and those
    suppressions then outlive the gap that caused them (13 stale ones had
    accumulated by the time this guard was written).
    """

    #: Fields whose accessors are hand-written module-level ``def``s rather
    #: than generated, so type-checkers already see them. Mirrors
    #: ``settings._SKIP_AUTOGEN``.
    HAND_WRITTEN = frozenset({"autofind_detectors", "saved_datasets_dir", "detectors_dir"})

    def test_hand_written_set_matches_skip_autogen(self):
        # Guards the exemption below against drifting from the real skip list.
        assert self.HAND_WRITTEN == settings._SKIP_AUTOGEN

    def test_every_generated_accessor_has_a_stub(self):
        missing = _pydantic_fields() - _stubbed_accessor_keys() - self.HAND_WRITTEN
        assert not missing, (
            f"Pydantic fields with no TYPE_CHECKING stub in vtsearch/settings.py: {sorted(missing)}. "
            "Add `def get_<key>() -> T: ...` / `def set_<key>(value: T) -> None: ...` to that block; "
            "do not silence the call sites with `# type: ignore[name-defined]`."
        )

    def test_no_orphan_stubs(self):
        per_side = settings._PER_SIDE_KEYS
        orphans = _stubbed_accessor_keys() - _pydantic_fields() - per_side
        assert not orphans, (
            f"TYPE_CHECKING stubs naming keys that are not Pydantic fields: {sorted(orphans)}. "
            "Delete the stub, or add the field to ServerSettings / UserSettings."
        )

    def test_hand_written_accessors_are_not_stubbed(self):
        # A stub for a real `def` would shadow the true signature for
        # type-checkers (e.g. `get_saved_datasets_dir` returns Path, not str).
        redundant = _stubbed_accessor_keys() & self.HAND_WRITTEN
        assert not redundant, f"Keys with both a real def and a stub: {sorted(redundant)}"


class TestAppSchemaCoversTheModels:
    """Every stored field must be dumped by ``GET /api/settings``, or be
    listed as a deliberate omission.

    ``AppSettingsSchema`` is a marshmallow ``Schema``, so ``dump()`` drops any
    key it does not declare -- silently, with no error anywhere. Nothing
    previously checked this direction (the guards above only assert schema
    fields are model-backed), which is how four fields came to be absent from
    the response without that ever being written down as intentional.
    """

    #: Model fields deliberately absent from ``AppSettingsSchema``, each with
    #: the reason it is not part of the settings DTO. Removing a field from
    #: the schema without adding it here fails ``test_no_undeclared_omissions``.
    DELIBERATE_OMISSIONS = {
        # Served hydrated by ``GET /api/sessions/recent`` (see
        # ``vtsearch.routes.sessions``), which resolves the stored
        # ``{dataset_id, detector_id}`` pairs into display records. The raw
        # stored list is storage, not the API shape.
        "recent_sessions",
        # Deployment-wide settings-sync target. Already excluded from the
        # defaults endpoint and from source export (``_EXCLUDE_FROM_DEFAULTS``
        # / ``_EXCLUDE_FROM_SOURCE_EXPORT``); echoing it to every client would
        # publish an operator's sync config.
        "default_settings_source",
        # Server-tier UMAP knobs, consumed into ``CoreConfig`` by
        # ``vtsearch.shim``. Operator-tuned via the settings file; no frontend
        # code reads them.
        "projection_n_neighbors",
        "projection_min_dist",
    }

    def test_no_undeclared_omissions(self):
        omitted = _pydantic_fields() - set(AppSettingsSchema().fields)
        undeclared = omitted - self.DELIBERATE_OMISSIONS
        assert not undeclared, (
            f"Model fields silently dropped by GET /api/settings: {sorted(undeclared)}. "
            "Marshmallow drops undeclared keys without erroring, so the frontend would "
            "never see these. Add a field to AppSettingsSchema, or list the key in "
            "DELIBERATE_OMISSIONS with the reason it is not part of the settings DTO."
        )

    def test_declared_omissions_are_still_real_fields(self):
        stale = self.DELIBERATE_OMISSIONS - _pydantic_fields()
        assert not stale, f"DELIBERATE_OMISSIONS names non-existent fields: {sorted(stale)}"

    def test_declared_omissions_are_actually_omitted(self):
        # If one gets added to the schema, drop it from the list rather than
        # leaving a note that contradicts the code.
        contradicted = self.DELIBERATE_OMISSIONS & set(AppSettingsSchema().fields)
        assert not contradicted, f"DELIBERATE_OMISSIONS names fields the schema does declare: {sorted(contradicted)}"
