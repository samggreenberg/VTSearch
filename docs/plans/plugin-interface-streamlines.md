# Plugin author interface streamlines

Status: **All five phases shipped (A → E).** P0/P1/P2/P3 candidates
landed; the only deferred items are the generator-based `RawMedia`
flow (#2's ambitious form, paired with #12) and a library-tier
`vtscore.threading.spawn` for the remaining non-app-tier background
spawn sites. See **What shipped** at the bottom for the per-phase
summary and **Open follow-ups** for what's parked.

## Background

We just split **embedding** and **converting** out of `DatasetImporter`.
Before: every importer had to call the embedder, run converters, and
populate a single combined media dict. After: importers yield raw
source-type media records; the framework owns the rest. The win wasn't
"less typing"; it was that *importers stopped mixing what they do with
what the framework does*.

This document is the result of asking "where else does that pattern still
exist?" across every plugin family in the tree:

| Family | Base class | Concrete impls |
|---|---|---|
| Dataset importers | `vtscore/datasets/importers/base.py` | 10 |
| Labelset exporters | `vtscore/exporters/base.py` | 6 |
| Label importers | `vtscore/labels/importers/base.py` | 3 |
| Labelset sources | `vtscore/labels/sources/base.py` | 1 |
| Settings importers | `vtsearch/settings_io/importers/base.py` | 2 |
| Settings exporters | `vtsearch/settings_io/exporters/base.py` | 2 |
| Settings sources | `vtsearch/settings_io/sources/base.py` | 1 |
| Media converters | `vtscore/converters/base.py` | 7 |
| Media sources | `vtscore/datasets/sources/base.py` | 3 |

## The recurring pattern: what counts as a candidate

A simplification belongs here when it satisfies all three of:

1. The behavior is **identical across implementations** (or at least
   uniformly defaulted).
2. The behavior is a **framework concern, not a plugin concern**; the
   plugin author has no business knowing about it.
3. Forgetting it is a **silent footgun** (security gap, wrong path
   produced, missing user context, etc.) rather than a loud error.

The embedding/converting extraction hit all three. The candidates below
do too.

## Candidates

Each candidate names the **leak**, where it shows up, the **fix**, and a
short evaluation. Numbering is for cross-reference only; see "Priority"
at the bottom for ordering.

### 1. Origin construction (`DatasetImporter.build_origin`)

**Leak.** `DatasetImporter.build_origin()` is overridden in roughly half
the importers (`server_folder`, `server_files`, `combine_datasets`,
`demo`, `http_archive`, `recaller`) and the default already handles
non-file, non-checkbox fields by serializing field_values verbatim. Most
overrides exist only to *exclude* a field or rename one.

**Fix.** Make field exclusion declarative on `PluginField`
(e.g. `omit_from_origin=True`, defaulting to `True` for `field_type="file"`
and `False` otherwise). Then every importer can drop `build_origin()`
entirely; the few with renames live as one-line `_RENAMES = {"old":
"new"}` class attrs the base class consults.

**Eval.** High win. ~6 method overrides deleted. Same shape as the
embedding extraction: a framework-uniform serialization currently
expressed as plugin code. Low risk; the default already does the right
thing for new fields; the overrides exist only because the data model
didn't have an "omit me" knob.

**Shim.** Trivial; `build_origin()` stays a hook on the base class. If a
subclass overrides it, the override wins (called as today). If a subclass
doesn't override it, the new default consults `omit_from_origin` /
`_RENAMES` and produces the same dict the old hand-written overrides
produced. Third-party importers that still ship a `build_origin` override
keep working unchanged.

### 2. Content optimization dicts (`content_vectors` / `content_md5s` / `custom_metadata_map`)

**Leak.** Importers populate three parallel `dict[str, Any]` instance
attributes during `run()`: `content_vectors` (precomputed embeddings),
`content_md5s` (precomputed hashes), `custom_metadata_map` (extra
metadata per file). Then they hand all three to
`load_dataset_from_folder()` / similar. `server_folder`,
`server_files`, and `http_archive` all repeat this pattern verbatim
(grep finds 9 distinct write sites and 6 distinct read sites).

**Fix.** Replace the three dicts with one `yield_precomputed(filename,
*, embedding=None, md5=None, metadata=None)` helper on the base class,
or, cleaner: let importers yield a `RawMedia(filename, bytes_or_path,
precomputed=...)` dataclass that carries it all together. The framework
unpacks it.

**Eval.** High win. The three-dict pattern is exactly the kind of
parallel-keyed-by-filename state that's easy to get wrong (one entry in
`content_md5s` but not in `content_vectors`, off-by-one filename casing,
etc.). Folding into a single record makes the contract obvious.

**Shim.** Keep the three instance dicts (`content_vectors`,
`content_md5s`, `custom_metadata_map`) initialized on the base class and
keep accepting them in the loader entry points. The framework's
post-`run()` step inspects whichever surface the importer used: if `run()`
returned/yielded `RawMedia` records, those win; otherwise it falls back to
the three dicts (which is exactly today's behavior). Third-party
importers that write to the dicts keep working; new code can yield
`RawMedia` instead. Document the dicts as deprecated-but-supported in
EXTENDING.md.

### 3. Declarative path / URL / template validation

**Leak.** Plugins individually call `validate_url()`,
`validate_server_filepath()`, and `sanitize_template_value()` from
`vtscore/security/`. Grep finds 14+ scattered call sites across
importers, exporters, sources, and even `vtsearch/routes/_shared.py`
(`validate_filepath_field`; which is hard-coded to the literal key
`"filepath"` rather than driven by the field schema). The webhook
exporter calls `validate_url(url)` in its `export()` body; the
labelset/settings sources sprinkle `sanitize_template_value` over
`{detector_name}` / `{username}` substitutions one by one.

**Fix.** Make the field schema carry the validator. We already have
`field_type="server_path"` and `field_type="url"` as declared types but
they're cosmetic; the framework doesn't enforce anything based on
them. Promote them to *enforced* types: before `run()` / `export()` is
called, the framework walks the field schema and applies
`validate_server_filepath` to every `server_path`, `validate_url` to
every `url`, and `sanitize_template_value` to any field with
`template_vars=["detector_name", "username", ...]` declared.

**Eval.** Highest-leverage candidate. Same shape as the post-processing
extraction: a uniform safety/security concern currently expressed as
plugin-author imperative code. Forgetting a validator is a silent
security gap, which is exactly the "silent footgun" trigger from the
pattern definition. Also: it would let `routes/_shared.py` stop
special-casing the literal key `"filepath"`; every field carries its
own validation rule.

**Shim.** Purely additive; the existing `validate_url`,
`validate_server_filepath`, and `sanitize_template_value` helpers stay
exported from `vtscore.security` with the same signatures. A third-party
plugin that calls them by hand inside `run()` keeps working: by the time
its code runs, the framework has already validated the value, so the
plugin's re-validation is an idempotent no-op (sanitize is idempotent on
already-sanitized strings; validate is read-only). The "promotion" is in
*what the framework does before calling `run()`*, not in removing the
plugin-side API.

### 4. Required / empty-string handling

**Leak.** Every file/path plugin has this snippet:

```python
filepath = (field_values.get("filepath") or "").strip()
if not filepath:
    raise ValueError("A file path is required.")
```

Counted 6+ times verbatim. `PluginField` already has a `required: bool`
flag but the framework doesn't enforce it consistently; plugins still
check by hand.

**Fix.** When `required=True`, the framework treats `""`, whitespace,
and missing-key the same way `marshmallow` would: rejects with a 422
before `run()` is called. The plugin body trusts `field_values["foo"]`
is a non-empty string.

**Eval.** Medium win. Removes a line per field per plugin and produces
consistent 422-vs-500 error responses (today a missing required field
becomes a 500 in some plugins, a 400 in others).

**Shim.** Purely additive at the plugin-source level: a third-party
plugin's manual `if not foo: raise ValueError(...)` check still compiles
and runs; it just never fires, because the framework rejected the empty
value before `run()` was called. The plugin's bespoke error message is
preempted by the framework's 422, which is a *behavior* change in error
text but not a *code* change the author has to make. No source edits
required to keep an old plugin working.

### 5. File-read boilerplate for JSON-on-disk plugins

**Leak.** Settings importer, label importer, settings exporter, and
labelset exporter all do their own `Path.exists()` / `is_file()` /
`read_bytes()` / `json.loads()` / `isinstance(data, dict)` dance, with
slightly different error messages each time.

**Fix.** Either:
- (a) introduce `field_type="server_json"` that resolves to *parsed
  JSON* before `run()` is called (so importer bodies receive a dict, not
  a path string), OR
- (b) move the whole read+parse helper to one place
  (`vtscore.io.read_server_json(path) -> dict`) and have each plugin
  call it.

**Eval.** Medium win. (a) is more invasive but follows the
"framework-prepares-the-input" pattern. (b) is a smaller refactor.
Probably (b) first, (a) later if it pulls weight.

**Shim.** Option (b) is intrinsically backwards-compatible; it's a new
helper that old plugins can ignore. Option (a) needs more care: a new
`field_type="server_json"` is a *new* field type, so existing plugins
that declare `field_type="server_path"` keep receiving a path string
exactly as before. Only plugins that opt into the new type receive a
parsed dict. Old `server_path`-based JSON readers continue to work
unchanged; we encourage migration but don't force it.

### 6. Atomic write helper

**Leak.** `_atomic_write_text` is implemented in
`vtscore/exporters/server_json_file/` *and* re-implemented inline in
`vtsearch/settings_io/exporters/server_json_file/`. Any future
file-writing exporter has to remember the tmp+rename+fsync ritual or
the file will be truncated on a crash.

**Fix.** Promote `vtscore.io.atomic_write_text(path, text)` and
`atomic_write_json(path, obj)` to a single library helper. Better: a
`field_type="server_path"` field could carry an `atomic_writer` so the
plugin just calls `writer.write(text)` and the writer handles tmp +
fsync + rename + parent-mkdir.

**Eval.** Small/medium win, but it's a real bug risk every time someone
writes a new file-writing plugin and forgets.

**Shim.** Purely additive; `vtscore.io.atomic_write_text` /
`atomic_write_json` are new public helpers. Existing in-tree call sites
(the two `server_json_file` exporters) get migrated when the helper
lands; third-party plugins that ship their own atomic-write logic keep
working unchanged. Re-export the helpers from the original locations
(`vtscore/exporters/server_json_file/`) as well so any third-party that
imported them directly from there doesn't break.

### 7. Template variable resolution

**Leak.** Exporters and sources hand-substitute `{YYYYMMDD-HHMMSS}`,
`{detector_name}`, `{detector_id}`, `{username}` into their path
fields. Each one re-implements roughly the same `str.replace` chain
through `sanitize_template_value`. `vtscore/exporters/_template.py` has
the canonical one; `vtscore/labels/sources/server_json_file/` and
`vtsearch/settings_io/sources/server_json_file/` reinvent the wheel.

**Fix.** Couple this with #3: a field declares `template_vars=
["detector_name", "username", "YYYYMMDD-HHMMSS"]` and the framework
substitutes them (via `sanitize_template_value`) before `run()`
receives the field value. Plugin body trusts the string is already
resolved.

**Eval.** Medium win. Removes a class of bug where one plugin
substitutes `{detector_name}` but forgets `{username}` (or vice versa).
Same shape as #3; declarative replaces imperative.

**Shim.** Default `template_vars=None` on `PluginField` means "framework
does no substitution"; i.e. existing fields keep receiving the raw
template string and the plugin keeps doing its own substitution as
today. Only fields that explicitly declare `template_vars=[...]` opt
into framework substitution. Migration is per-field and per-plugin, on
the plugin's own schedule. If a third-party plugin later opts in and
*also* leaves a `sanitize_template_value` call in its body, that's a
no-op on an already-resolved string (sanitize is idempotent).

### 8. Background-thread context propagation

**Leak.** CLAUDE.md flags this explicitly: "Background threads spawned
from a request handler must scope `vtsearch.auth.thread_user(name)` so
per-user writes resolve correctly." Same applies to
`thread_dataset_context()` / `thread_detector_context()` for the
`medias` / `good_votes` / etc. proxies. (Audit M22, now closed, added
those context-manager scopes and migrated every production call site
off the bare `set_thread_*` setters.) `JobManager` and the dataset-load
thread spawn sites do this; ad-hoc background threads in plugins are
on their own.

**Fix.** Provide `vtsearch.threading.spawn(target, ...)` that snapshots
the current `(user, dataset_ctx, detector_ctx)` and enters the matching
`thread_user` / `thread_dataset_context` / `thread_detector_context`
scopes inside the new thread. Plugin / route code uses `spawn()` instead
of `threading.Thread(target=...).start()`; the context plumbing
disappears from sight.

**Eval.** Medium win. Lower volume than #1–#3 but high consequence;
when this leak fires it silently writes per-user settings to the wrong
user or crashes on a missing context. The "logical-bug-audit.md" plan
already calls out context-propagation gaps as a recurring root-cause
pattern; this would close most of them.

**Shim.** Purely additive; `vtsearch.threading.spawn(...)` is a new
helper. The `thread_user()` / `thread_dataset_context()` /
`thread_detector_context()` context managers and the bare
`set_thread_*` setters stay exported with the same signatures. A
third-party plugin that builds its own `threading.Thread` and calls
the setters by hand keeps working exactly as today; new code is just
recommended to use `spawn()` instead.

### 9. Converter param normalization

**Leak.** `MediaConverter.convert(media, params)` accepts
`params: dict | None`, and the base class provides `get_param(params,
key)` so converter authors can fall back to the declared
`PluginField.default`. Today: every converter has to read params
through `get_param()` (and some forget, going `params["foo"]` directly
and crashing on `None`).

**Fix.** Normalize `params` *before* `convert()` is called: the
framework expands defaults and validates ranges (`validate_params()`),
then hands `convert()` a fully-populated, plain dict. `params` becomes
non-`None`, every key declared in `fields` is present, every value is
range-valid. `get_param()` and `validate_params()` move from the
converter API to internal helpers the framework owns.

**Eval.** Solid win. Same as #4; converts an opt-in helper plugin
authors might forget into a framework-guaranteed input shape.

**Shim.** Keep `get_param()` as a method on `MediaConverter`; once the
framework has pre-populated `params`, `get_param(params, key)` collapses
to a plain `params[key]` lookup, so old converters that route through it
keep working. Converters that did `params["key"]` directly *start*
working (they previously crashed on `None`), which is a strict
improvement. Keep `validate_params()` as a hook the framework calls
before normalizing; if a third-party converter overrode it for custom
checks, the override still runs. The `params: dict | None` type
annotation stays `dict | None` for source compatibility even though
the framework now always passes a non-None dict.

### 10. Converter metadata inconsistency

**Leak.** `MediaConverter` declares `display_name` and
`converter_description` as class attrs *instead of* the `description`
that every other plugin family uses. The name dodge (`description` →
`converter_description`) was to avoid colliding with `PluginBase.description`,
but the rest of the codebase doesn't have that collision.

**Fix.** Rename `converter_description` → `description` and resolve
the collision (it appears `PluginBase` doesn't actually claim
`description`; check). One-line change per converter.

**Eval.** Tiny win, but it's the kind of inconsistency that creates a
small mental tax every time someone writes a new converter and has to
look up which name applies.

**Shim.** The framework's description getter checks `description` first
and falls back to `converter_description` if the former is unset. Old
converters that only declare `converter_description` keep working; new
converters declare `description` like every other plugin family does.
After a long-enough soak period we can warn (or remove) the fallback,
but the initial change is non-breaking.

### 11. Plugin metadata defaults

**Leak.** Every plugin class declares `name`, `display_name`,
`description`, `icon`, `fields`. The first three are essentially the
class name, a title-cased variant, and the docstring's first line.

**Fix.** Derive defaults: `name = snake_case(cls.__name__.replace(
"DatasetImporter", "").replace("LabelsetExporter", ""))`, `display_name
= title_case(name)`, `description = first_line(cls.__doc__ or "")`.
Override remains explicit and wins.

**Eval.** Modest win. Saves boilerplate but doesn't remove a *concern*
:  just typing. Lowest priority of the candidates.

**Shim.** Purely additive; the defaults only fire when the class attr
is missing. Existing plugins that explicitly declare `name` /
`display_name` / `description` keep using their explicit values
verbatim. Use a sentinel (or `getattr(cls, "name", None)`) rather than
overwriting attrs on the class so subclassed plugins still see their own
declarations.

### 12. Cooperative cancellation in importers

**Leak.** Some importers sprinkle `check_dataset_cancelled()` /
`vtscore.concurrency.progress.check_cancelled()` calls in their loops.
Forgetting one means an in-flight load can't be cancelled.

**Fix.** Bundle cancellation into the iteration shape. If we land #2
(importer yields `RawMedia` records via a generator), the framework can
check cancellation between yields. The importer body never imports the
cancellation API at all.

**Eval.** Solid win if #2 lands; not really separable from it.

**Shim.** Tracks #2; if the importer uses the old non-generator `run()`
shape, the framework can't insert cancellation points between yields, so
those importers keep relying on their hand-placed
`check_dataset_cancelled()` calls (which stay exported and functional).
Only importers that opt into the generator/`RawMedia` shape get free
between-yield cancellation. Third-party importers that mix both
(generator yields + explicit `check_dataset_cancelled()`) are fine;
the explicit call is a no-op redundant check.

### 13. `field_type="file"` returns Werkzeug `FileStorage`

**Leak.** Library-tier plugin bases (`vtscore/labels/importers/base.py`,
`vtscore/datasets/importers/base.py`) document that
`field_type="file"` *receives a Werkzeug FileStorage*; a Flask/app-tier
type; but the module is library-tier and supposedly Flask-clean.
`run_cli()` exists explicitly to paper over this by handing in a path
string instead.

**Fix.** Pick one normalized representation (a `BinaryIO`-like with a
`.filename` attr) that both Flask and the CLI hand in. The plugin body
trusts that representation and never mentions Werkzeug. `run_cli` /
`run` collapse into a single `run`.

**Eval.** Medium-high win. Closes the cleanest tier-leak in the
codebase and removes an entire CLI-vs-API method pair from every
file-accepting plugin.

**Shim.** Pick the normalized type to be *structurally compatible* with
the FileStorage attrs old plugins already use; `.filename` (str) and
`.read()` / `.stream` (binary IO). Werkzeug's `FileStorage` already
satisfies this shape, so on the Flask side we can pass `FileStorage`
through unchanged (or wrap it in a thin adapter that preserves those
attrs); on the CLI side we wrap the path string in an adapter that
opens the file lazily and exposes the same surface. Old plugin bodies
that read `.filename` and call `.read()` keep working in both code
paths. Keep `run_cli` callable on the base class: if a subclass
overrides it, the framework still routes CLI invocations through the
override; if not, CLI invocations go through `run()` with the adapter.
That lets third-party plugins migrate at their own pace.

## Cross-cutting observation

Six of the candidates above (#3, #4, #5, #6, #7, #9) collapse into one
larger move: **make `PluginField` carry richer behavior, and make the
framework normalize/validate `field_values` *before* `run()` /
`export()` is called.** Right now the field schema is mostly used for
UI rendering and a thin layer of validation; everything else is
plugin-author responsibility. If we re-frame the schema as "the
framework's contract for what the plugin receives", then path
validation, URL validation, template substitution, empty-string
handling, JSON parsing, param normalization, and atomic-writer
provisioning all fall out of the same mechanism.

That's the unifying insight from the embedding extraction too:
**plugins should describe what they need, and receive it in a form
that's already ready to use.**

## Priority

**P0; uniform safety + the field-schema collapse**
- #3 Declarative path/URL/template validation (security; highest
  consequence; unblocks #7)
- #4 Framework-enforced `required` (consistency; low risk)
- #7 Declarative template variables (couples to #3)

**P1; uniform shape**
- #1 Origin construction default (deletes overrides)
- #2 Content optimization dicts → `RawMedia` (eliminates parallel-dict
  bug class; enables #12)
- #9 Converter param normalization (closes the `get_param` hole)
- #13 Drop Werkzeug from library-tier plugin bases

**P2; quality of life**
- #5 Server-JSON read helper
- #6 Promote `atomic_write_text` / `atomic_write_json`
- #8 `spawn()` for thread context propagation (touches a different
  axis; can land independently)
- #12 Implicit cancellation (depends on #2)

**P3; boilerplate trim**
- #10 Converter metadata naming
- #11 Plugin metadata defaults

## Non-goals

- Not changing the *capability* of any plugin family. Every existing
  plugin continues to express the same behavior, just with the
  framework-uniform parts hoisted out.
- Not breaking external (third-party) plugin subclasses. Even though
  the CLAUDE.md backwards-compat policy says breaking is fine in
  general, *written plugin extensions* are an exception: the goal is
  that every candidate here can ship without forcing third-party plugin
  authors to edit their code. Each candidate above has a **Shim** note
  describing the specific backwards-compatibility hook (override
  fallback, additive-only API, structural type compatibility, etc.).
  Serialized objects (old detector JSON, cached pickles, etc.) are
  *not* under this protection; those can change shape as needed.
- Not unifying base classes across families into a single
  super-base; they have legitimately different return shapes
  (`run` vs `export` vs `convert`). The unification proposed here is
  *behind* the bases, in the framework that calls them.

## Next: scheduled work

### Phase A: Candidate #1: declarative origin construction

**Status:** **shipped.** All six in-tree `build_origin` overrides
deleted; new declarative knobs (`include_in_origin`, `origin_serializer`,
`extra_origin_keys`, `origin_suppressed`) live on `PluginField` /
`DatasetImporter`. Coverage in `tests_lib/datasets/test_build_origin.py`;
all 4232 tests pass.
**Sequence:** first PR in the streamline series. Phase B (P0 field-schema
collapse: #3 + finish #4 + #7) follows next.
**Why first:** smallest blast radius (one base class + 6 in-tree
importers, ~30-50 LOC deleted), zero required edits for external
plugins, no CLI / HTTP-schema surface touched. It validates the
"declarative replaces imperative override" pattern end-to-end before
the more invasive validation work.

#### What the existing overrides actually do

A grep + read of every override exposes four distinct patterns, not
just one. The Shim note in §1 above understated this; `_RENAMES` alone
doesn't cover any of them. The patterns are:

| Importer | Override pattern | Phase A handles via |
|---|---|---|
| `server_folder` | Add the non-`PluginField` key `source_specs` to params, JSON-encoding it if it arrived as a list | New `extra_origin_keys` class attr + framework-side list→JSON coercion |
| `http_archive` | Same as `server_folder` | Same |
| `server_files` | Include only `paths_file` and `media_type`; nothing else from `fields` | `include_in_origin=False` on the unwanted fields |
| `demo` | Include only `name` and `converter`; emit `name` even when empty (the default skips empties) | `include_in_origin=False` on the unwanted fields; accept losing the empty-`name` edge case |
| `combine_datasets` | Custom serializer for the `datasets` field (list-or-string → comma-joined string) | `origin_serializer` callable on `PluginField` |
| `recaller` | Return empty params: dataset-level origin is meaningless; per-media origins are built in `_build_media` | `origin_suppressed = True` class attr |

A latent footgun the override-by-override approach is hiding: **the
current default `build_origin` does not exclude `field_type="password"`
fields.** An importer like the SFTP skeleton in the base-class docstring
(which declares a `password` field) would silently land that password
in the persisted origin dict. Phase A fixes this by defaulting
`include_in_origin=False` for `password` and `file` field types.

#### Design

Two additions to `PluginField` (in `vtscore/plugins/__init__.py`) and
two to `DatasetImporter` (in `vtscore/datasets/importers/base.py`):

```python
# PluginField
include_in_origin: bool | None = None    # None → field_type-driven default
origin_serializer: Callable[[Any], str] | None = None

# DatasetImporter
extra_origin_keys: tuple[str, ...] = ()  # non-PluginField keys to copy
origin_suppressed: bool = False          # short-circuit to {importer, params={}}
```

Default for `include_in_origin` resolves at runtime:
- `field_type in ("file", "password")` → `False`
- everything else → `True`

The framework's new `build_origin` body becomes:

1. If `origin_suppressed`, return `{"importer": self.name, "params": {}}`.
2. For each `PluginField` where `include_in_origin` resolves true: pull
   the value, run `origin_serializer` if set, fall back to the existing
   checkbox / `str(val) if val else skip` rules.
3. For each key in `extra_origin_keys`: copy from `field_values`,
   JSON-encoding lists/dicts on the way out.
4. Return `{"importer": self.name, "params": params}`.

The list→JSON coercion for `source_specs` moves from the importer to
the framework because every multi-media importer needs it. New
multi-media importers (third-party `recaller`-style) get this for free
by declaring `extra_origin_keys = ("source_specs",)`; or, even better,
the framework auto-adds `"source_specs"` to `extra_origin_keys` when
`multi_media = True` on the class.

#### Migration impact: in-tree

All six overrides get deleted. Concrete diffs:

- `server_folder`: delete override (auto-added via `multi_media=True`).
- `http_archive`: delete override (auto-added via `multi_media=True`).
- `server_files`: delete override; add `include_in_origin=False` to any
  field other than `paths_file` / `media_type` (today the field list is
  already just those two plus `source_specs` for multi-media, so the
  override is largely vestigial; deleting it is a near-no-op).
- `demo`: delete override; add `include_in_origin=False` to whatever
  fields aren't `name`/`converter`. Loses the "emit empty `name`" edge
  case; verify no test asserts on the empty-name behavior.
- `combine_datasets`: delete override; declare
  `origin_serializer=lambda v: ",".join(v) if isinstance(v, list) else v`
  on the `datasets` field.
- `recaller`: delete override; set `origin_suppressed = True` on the
  class.

LOC delta: roughly -50 (six overrides averaging ~8 lines) +12 (new
class attrs and PluginField args).

#### Migration impact: external plugins (the four families you named)

**`DatasetImporter` (external; the in-repo `recaller` scaffold is the
template).** Migration cost: **zero required edits.** External
importers that override `build_origin` keep working unchanged; the
new framework default only fires when the subclass doesn't override.
Optional cleanup: delete the override after declaring the equivalent
class attrs. Concretely for a recaller-style importer, the migration
is a one-line diff:

```python
# Before
def build_origin(self, field_values):
    return {"importer": self.name, "params": {}}

# After
origin_suppressed = True
```

**`MediaSource` (external; `pullwrest`).** Migration cost: **none.**
`MediaSource` doesn't participate in field-schema-driven origin
construction at all; sources are factoried from an existing origin
dict via `create_from_origin(origin)`, they don't build origins.
Phase A doesn't touch sources.

**`LabelImporter` (external; `holder`).** Migration cost: **none.**
Label importers don't have a `build_origin` concept; they return label
dicts keyed by md5 / origin. Phase A doesn't touch label importers.

**`LabelsetExporter` (external; `holder`).** Migration cost: **none.**
Exporters consume labelsets, they don't produce origins. Phase A
doesn't touch exporters.

In summary: Phase A only affects DatasetImporters, and even there the
override-still-wins shim means zero required edits for external code.
This is the gentlest possible first move and is a clean test of the
"declarative knobs on PluginField replace plugin-side overrides"
hypothesis before Phase B raises the stakes.

#### Implementation steps

1. Add `include_in_origin` and `origin_serializer` to `PluginField` in
   `vtscore/plugins/__init__.py`; thread them through `to_dict()` /
   `from_dict()` so the registry's JSON snapshot stays round-trip safe.
2. Add `extra_origin_keys` and `origin_suppressed` to `DatasetImporter`.
3. Rewrite `DatasetImporter.build_origin` per the design above; add a
   `_resolve_include_in_origin(field)` helper that applies the
   field-type defaults.
4. Auto-add `"source_specs"` to `extra_origin_keys` when
   `multi_media = True`.
5. Delete the six in-tree overrides; add the equivalent declarative
   attrs.
6. Tests: extend `tests_lib/datasets/` with cases for each pattern
   (suppressed origin, omitted password field, list-typed field via
   `origin_serializer`, `extra_origin_keys` JSON-encoded). Verify the
   "override still wins" shim with a fake third-party-style importer
   that overrides `build_origin` and asserts the override's return
   value reaches `media["origin"]`.
7. Run `./run-tests.sh datasets io detectors` first (origin info is
   read by labelset resolution and label export, so the blast radius
   touches those groups), then the full `./run-tests.sh`.

#### Open questions

- **`origin_suppressed` vs an empty `params` dict by default for
  recaller.** Suppression is more explicit and is the only pattern that
  emits the literal `{"importer": name, "params": {}}` short-circuit.
  Alternative: declare all fields `include_in_origin=False` and let the
  loop produce empty params. Cleaner per-field, but loses the "I have
  no useful dataset-level origin" intent. Recommend keeping the
  explicit class attr.
- **Should `origin_serializer` get the full `field_values` dict** in
  addition to the field's value, so a serializer can reach across to
  another field? Only `combine_datasets` would care, and it doesn't
  need cross-field access. Keep the API single-value to start; widen
  later if a real use case appears.
- **Phase A or Phase B owns the `validate_filepath_field` deletion.**
  Phase A doesn't strictly need it; it's a separate route-layer
  concern; but if Phase B drags, it might be worth pulling the
  declarative-`server_path`-validation slice forward into Phase A. Park
  this until Phase A is in flight.

### Phase C: Candidates #2 + #9 + #13: shape unification (P1)

**Status:** **shipped.** Three independent surfaces landed in one PR:

- **#9: converter param normalization.**
  :meth:`MediaConverter.convert_normalized` is the new framework entry
  point.  It pre-strips empty-string values whose field declares a
  default, runs the per-converter marshmallow schema (validating
  declared :attr:`~PluginField.min` / :attr:`max` / :attr:`options`
  constraints), fills missing or empty-string keys with the field's
  declared default, then dispatches to the subclass's
  :meth:`~MediaConverter.convert`.  Every in-tree call site
  (:meth:`DatasetImporter._ingest_spec_stream`,
  :func:`vtscore.converters.runner._run_converter_on_source`,
  :func:`vtscore.converters.runner.run_converters_on_folder`,
  :func:`vtscore.datasets.clipper_chain._run_converter_step`, the
  clipper-chain replay loop) routes through it.  Validation failures
  raise :class:`ValueError` (matching the rest of the plugin-arg error
  contract) rather than :class:`marshmallow.ValidationError`.
  :meth:`~MediaConverter.get_param` stays as a back-compat shim for
  third-party converters whose call sites bypass the framework
  wrapper.
- **#13: drop Werkzeug from library tier.**  New module
  :mod:`vtscore.plugins.uploads` defines :class:`UploadedFile` (a
  ``runtime_checkable`` :class:`typing.Protocol` with ``filename`` /
  ``read`` / ``save``), :class:`CliUploadedFile` (adapter wrapping a
  filesystem path string), and :class:`BytesIOUploadedFile` (adapter
  holding upload bytes in memory for background-thread reads).
  Werkzeug's :class:`~werkzeug.datastructures.FileStorage` satisfies
  the protocol natively and is passed through unchanged on the Flask
  request path.  The default :meth:`DatasetImporter.run_cli` and
  :meth:`LabelImporter.run_cli` now wrap any ``field_type="file"``
  path-string argument in :class:`CliUploadedFile` before delegating
  to :meth:`run`, so plugin bodies see one shape regardless of
  ingress.  :func:`vtscore.datasets.load_pipeline._run_importer_in_background`
  and :func:`_stage_importer_in_background` apply the same wrapping,
  so the reload-from-origin path (which supplies a server-path string)
  reaches :meth:`run` as an :class:`UploadedFile` too.  The bytesio
  branch of :func:`vtsearch.routes._shared.validate_plugin_args` now
  produces a :class:`BytesIOUploadedFile` instead of a bare
  :class:`io.BytesIO` with ``.name`` taped on (``.name`` still exposed
  as a back-compat shim).  Settled on ``.filename`` as the canonical
  attr per the open follow-up.  The pickle importer's ``run_cli``
  override was deleted (its custom logic collapsed into the now-shared
  base path via :meth:`UploadedFile.save`).  Library-tier base-class
  docstrings (:mod:`vtscore.datasets.importers.base`,
  :mod:`vtscore.labels.importers.base`, :class:`PluginField`'s
  ``"file"`` description) stopped referencing
  :class:`~werkzeug.datastructures.FileStorage` and point at
  :class:`UploadedFile`.
- **#2: yield_precomputed helper.**  New method
  :meth:`DatasetImporter.yield_precomputed(filename, *, embedding,
  md5, metadata)` routes to the three legacy precomputed dicts
  (:attr:`content_vectors`, :attr:`content_md5s`,
  :attr:`custom_metadata_map`).  Plugin authors that previously wrote
  to all three dicts in parallel can now collapse to a single helper
  call so a misspelled key can't land in only one or two of the
  parallel maps.  The three dicts stay public and continue to work for
  third-party importers that write to them directly.  The full
  generator-based :class:`RawMedia` shape proposed in the original
  candidate is deferred; it isn't separable from #12 (implicit
  between-yield cancellation), which has no scheduled work yet.

Coverage in :file:`tests_lib/datasets/test_phase_c.py` (22 tests).
4277 in-suite tests pass.

#### Migration impact: external plugins

| Family | Cost |
|---|---|
| `DatasetImporter` (e.g. `recaller`) | None required.  CLI invocations of plugins with ``file`` fields now receive an :class:`UploadedFile` instead of a raw path string: plugin bodies that read ``.filename`` / ``.read()`` / ``.save(dst)`` continue to work, and the pre-existing ``isinstance(value, str)`` fallback in :meth:`PickleDatasetImporter.default_display_name` is no longer needed.  Multi-source-type importers can call :meth:`yield_precomputed` per file instead of writing to the three dicts. |
| `MediaSource` (e.g. `pullwrest`) | None.  Sources do not participate in this candidate. |
| `LabelImporter` (e.g. `holder`) | None required.  Same UploadedFile change as DatasetImporter: ``run`` now receives a wrapped path on the CLI path. |
| `LabelsetExporter` (e.g. `holder`) | None.  Exporters do not participate. |
| Third-party converters | None required.  Plugins called via the framework path now receive validated + default-filled ``params``; subclasses that route reads through :meth:`get_param` keep working; subclasses that index ``params[key]`` directly start working for the cases that previously crashed on missing keys.  Third-party call sites that invoke ``convert()`` directly (rather than ``convert_normalized()``) keep getting raw, un-validated params: the shim is intentional. |

#### Open follow-ups

- The full generator-based :class:`RawMedia` flow (candidate #2's
  ambitious form) plus implicit cancellation (#12) remain unscheduled.
  Worth revisiting only when a concrete importer wants both; the
  cancellation win doesn't materialise without it.
- :meth:`PickleDatasetImporter.run_chunked_cli` still expects a bare
  path string (its own override, untouched by Phase C); consolidating
  it with the wrapping default is a minor cleanup worth folding into
  any future chunked-load refactor.
- The :meth:`MediaConverter.get_param` shim can be removed once a
  soak period confirms no third-party converters rely on it.

### Phase B: Candidates #3 + #4 + #7: declarative validation & templates

**Status:** **shipped.** One central
`normalize_field_values(plugin, field_values)` pass runs after
schema/CLI validation and before `run()` / `export()`, applying
whitespace strip, framework-enforced `required`, declarative
`template_vars` substitution (via `sanitize_template_value`), and
field-type-driven security validators (`validate_url` for
`field_type="url"`, `validate_server_filepath` for
`field_type="server_path"`). In-tree plugins (webhook, the two
`server_json_file` exporters, the two `server_json_file` importers,
the labelset + settings `server_json_file` sources, `server_csv_file`
exporter, `server_csv_file` label importer, `email_smtp`, `http_archive`)
shed their manual strip+`raise ValueError` checks and their hand-rolled
template + validator calls. The hard-coded `validate_filepath_field` in
`vtsearch/routes/_shared.py` is gone; `server_path` fields validate
themselves regardless of key name. 4227 tests pass.

#### What changed

Two new public surfaces in `vtscore/plugins`:

- `PluginField.template_vars: tuple[str, ...] = ()`: opt-in list of
  variable names the framework should substitute before the plugin
  receives the value. Currently recognised vars: `YYYYMMDD-HHMMSS`,
  `detector_name`, `detector_id`, `username`. Unknown names raise at
  normalize-time, so a typo in a plugin schema fails fast.
- `vtscore.plugins.normalize.normalize_field_values(plugin, field_values)`
 ; the central pass. For each field, strips text-like values,
  substitutes any declared template vars (sanitised through
  `sanitize_template_value`), then runs the field-type-driven security
  validator if the value is non-empty. Returns the mutated dict;
  re-raises `validate_url` / `validate_server_filepath`'s `ValueError`.

Wired in two places; these are the only ingress points for plugin
field values:

- `vtsearch/routes/_shared.py:validate_plugin_args`: after marshmallow
  loads and file uploads are populated, the normalize pass runs. A
  validation `ValueError` becomes a 400 with the standard envelope.
- `vtscore/plugins/__init__.py:PluginBase.validate_cli_field_values`:
  the CLI presence check now also runs the normalize pass, so CLI
  invocations get the same validation guarantees the HTTP path does.

The exporter routes (`vtsearch/routes/labels/exporters.py`,
`vtsearch/routes/settings/io.py:run_settings_export`) receive
`field_values` via a different shape (nested inside a marshmallow
schema) and now route through a small helper
`validate_exporter_field_values(plugin, field_values)` that runs the
plugin-arg schema + normalize on the dict and aborts 422/400 as
appropriate. Their hand-written "missing required" loops and inline
`validate_server_filepath` calls are deleted.

#### In-tree plugin migrations

| Plugin | Removed |
|---|---|
| `vtscore/exporters/webhook` | `url.strip()`, `if not url: raise ValueError`, `validate_url(url)`, the now-unused `validate_url` import |
| `vtscore/exporters/server_json_file` | `filepath_str.strip()`, `if not filepath_str: raise ValueError`, `resolve_export_filepath(filepath_str)` and its import; field declares `template_vars=("YYYYMMDD-HHMMSS", "detector_name", "username")` |
| `vtscore/exporters/server_csv_file` | Same shape as `server_json_file` |
| `vtscore/exporters/email_smtp` | `from_addr.strip()`, `to_addr.strip()`, and the two "X is required" branches; the `@` invariant remains as plugin-specific validation |
| `vtscore/labels/sources/server_json_file` | The `_resolve_filepath()` helper; field declares `template_vars=("detector_id", "detector_name")`. The plugin renames `load`/`load_full`/`save` → `_do_load`/`_do_load_full`/`_do_save` template methods; the `SyncSource` base class's public wrappers run `normalize_field_values` on a copy of `field_values` before dispatching. `resolve_filepath_for()` retained for the rename code path that resolves a path for a *different* detector than the active context. |
| `vtsearch/settings_io/sources/server_json_file` | Same template-method migration as the labelset source; field declares `template_vars=("username",)`. `peek_version` renamed to `_do_peek_version` |
| `vtscore/labels/importers/server_json_file` | `filepath.strip()` and the `if not filepath: raise ValueError` |
| `vtscore/labels/importers/server_csv_file` | Same |
| `vtsearch/settings_io/importers/server_json_file` | Same |
| `vtsearch/settings_io/exporters/server_json_file` | Same |
| `vtscore/datasets/importers/http_archive` | `validate_url(url)` in both `run()` and `_download_and_extract()`; the `run_cli` URL-prefix check (subsumed by `validate_url`); the now-unused `validate_url` import |
| `vtscore/datasets/importers/server_folder` | No body changes (display-name strip kept: still needed for the UI label); already accessed values directly |
| `vtscore/datasets/importers/server_files` | No body changes (already accessed values directly) |

The `resolve_export_filepath` helper in
`vtscore/exporters/_template.py` is kept as a no-op compatibility
shim; any third-party exporter that still imports it sees its
templates resolved twice (once by the framework, once by this helper),
which is idempotent.

#### Migration impact: external plugins

| Family | Cost |
|---|---|
| `DatasetImporter` (e.g. `recaller`) | None required. Plugins that hand-call `validate_url` / `validate_server_filepath` keep working: the framework validates first, the plugin's call is an idempotent no-op. To clean up: declare `template_vars=[...]` on templated fields and delete the manual call. |
| `MediaSource` (e.g. `pullwrest`) | None. Sources don't participate in field-driven invocation. |
| `LabelImporter` (e.g. `holder`) | None required. Plugins that strip+check `field_values["filepath"]` keep working; the framework rejects empty values earlier with a 422. |
| `LabelsetExporter` (e.g. `holder`) | None required. Same shim as label importers. |

#### Open follow-ups

- The CLI `--filepath` flag short-form in
  `vtscore/labels/importers/server_json_file/__init__.py:add_cli_arguments`
  is now somewhat redundant with the field-driven `add_cli_arguments`
  default; consider deleting the override in a future cleanup.
- `vtscore/exporters/_template.py:resolve_export_filepath` is a shim
  for now. Once a soak period confirms no third-party imports remain,
  delete it (the in-tree migration removes all the in-tree call sites).
- ~~Sync sources still call `_normalized(source, field_values)` at the
  top of each method body~~; **shipped as part of Phase B**.
  `SyncSource` now wraps `load` / `save` / `load_full` / `peek_version`
  to normalize *field_values* before dispatching to the new
  underscored template methods (`_do_load` / `_do_save` /
  `_do_load_full` / `_do_peek_version`). This is a breaking change
  for any third-party `SyncSource` subclass: rename your overrides to
  the `_do_*` form.

### Phase D: Candidates #5 + #6 + #8 (P2 quality of life)

**Status:** **shipped.** Three small but recurring footguns closed:

- **#5: shared JSON read helper.** New :func:`vtscore.io.read_server_json`
  collapses the
  ``Path.exists()`` / ``is_file()`` / ``read_bytes()`` / ``json.loads()``
  dance into one call.  ``missing_ok=True`` lets sync sources opt into
  "no file yet → return None" without sprinkling their own
  ``if not path.exists(): return []`` shim.  Migrated:
  :class:`~vtscore.labels.importers.server_json_file.ServerJsonLabelImporter`,
  :class:`~vtsearch.settings_io.importers.server_json_file.ServerFileSettingsImporter`,
  the labelset source
  :class:`~vtscore.labels.sources.server_json_file.ServerFileLabelsetSource`,
  and the settings source
  :class:`~vtsearch.settings_io.sources.server_json_file.ServerFileSettingsSource`.
- **#6: shared atomic-write helpers.** New
  :func:`vtscore.io.atomic_write_text` and
  :func:`vtscore.io.atomic_write_json` hoist the tmp + ``fsync`` +
  :func:`os.replace` ritual into one place.  Per-writer unique tmp
  filenames (``<dest>.<pid>.<uuid>.tmp``) keep concurrent writers from
  fighting over the same in-flight tmp.  ``newline=""`` on the writer
  preserves already-formatted CRLF (so the CSV exporter doesn't double
  its line endings on Windows after routing through the helper).
  Migrated:
  :mod:`vtscore.exporters.server_json_file`,
  :mod:`vtscore.exporters.server_csv_file` (via its
  ``_atomic_write_csv`` wrapper),
  :mod:`vtsearch.settings_io.exporters.server_json_file`, plus the two
  ``_do_save`` paths on the sync sources.  The historical
  ``vtscore.exporters.server_json_file._atomic_write_text`` private
  name is re-exported as a shim so any third-party exporter that
  imported it directly keeps working.
- **#8: :func:`vtsearch.threading.spawn`.** New helper that
  snapshots the caller's ``(user, dataset_ctx, detector_ctx)`` at spawn
  time and re-installs them inside the new daemon thread.  Snapshot
  rules: an explicit ``set_thread_user`` wins, else Flask ``g.user``,
  else ``None`` (so the spawn does not clobber the spawned thread's
  thread-local with the ``"default"`` fallback when nothing was ever
  set explicitly).  Migrated in-tree call sites:
  :func:`vtsearch.routes.datasets.registry.load_registered_dataset` and
  the two thread-launch sites in
  :mod:`vtsearch.routes.detectors.registry`
  (``_maybe_reembed_for_active_dataset`` /
  ``load_detector_route``).  Each call site lost ~5 lines of boilerplate
  per task.  :class:`~vtscore.concurrency.async_jobs.JobManager` keeps
  its own context replay path; it already handles a richer job/cancel
  surface and isn't a 1:1 fit for ``spawn``.

Coverage in :file:`tests_lib/io/test_io_helpers.py` (concurrent-write
race tests included) and :file:`tests/integration/test_spawn.py`.
4334 in-suite tests pass.

#### Migration impact: external plugins

| Family | Cost |
|---|---|
| `DatasetImporter` / `LabelImporter` / `LabelsetExporter` / `LabelsetSource` / `SettingsImporter` / `SettingsExporter` / `SettingsSource` | None required. The helpers are additive; existing inline JSON reads and atomic-write loops keep working unchanged. To clean up: switch the body to :func:`vtscore.io.read_server_json` / :func:`vtscore.io.atomic_write_json`. |
| Third-party background-thread call sites | None required. ``threading.Thread(target=..., daemon=True).start()`` still works; the recommendation for new code is :func:`vtsearch.threading.spawn` instead, so the user/dataset/detector context plumbing disappears from the body. |

#### Open follow-ups

- ``vtscore.exporters.server_json_file._atomic_write_text`` is a
  shim re-export.  After a soak period confirms no third-party
  exporter imports it directly, delete the alias.

### Phase E: Candidates #10 + #11 (P3 boilerplate trim)

**Status:** **shipped (#11 only; #10 was already done).**

- **#10: converter metadata naming.** No work needed.  A pre-existing
  rename had already collapsed ``converter_description`` →
  ``description`` on :class:`~vtscore.converters.base.MediaConverter`;
  every in-tree converter and the docs reference the unified name.
  This entry stays in the plan as a record that the candidate was
  closed, not because anything shipped in Phase E.
- **#11: plugin metadata defaults.** :meth:`PluginBase.__init_subclass__`
  now auto-derives :attr:`name` / :attr:`display_name` /
  :attr:`description` for any concrete subclass that doesn't declare
  them.  Derivation rules:
  - ``name`` strips a family suffix (``DatasetImporter`` /
    ``LabelsetExporter`` / ``LabelImporter`` / ``LabelsetSource`` /
    ``SettingsImporter`` / ``SettingsExporter`` / ``SettingsSource`` /
    ``MediaConverter`` / ``MediaSource`` / or the bare nouns
    ``Importer`` / ``Exporter`` / ``Source`` / ``Converter``) from the
    class name and snake-cases the remainder.  ``MyShinyDatasetImporter``
    → ``"my_shiny"``.
  - ``display_name`` title-cases the resulting ``name``
    (``"my_shiny"`` → ``"My Shiny"``).
  - ``description`` is the first line of the class docstring (or
    empty when the docstring is missing).
  Explicit declarations always win.  The framework-level abstract
  bases listed in
  :data:`vtscore.plugins._PLUGIN_FAMILY_BASE_NAMES`
  (``LabelImporter`` / ``DatasetImporter`` / etc.) skip
  auto-derivation entirely so a derived ``name`` doesn't pollute every
  concrete subclass via MRO inheritance; third-party intermediates
  that should behave the same way opt out via
  ``_is_plugin_family_base = True`` in their own ``__dict__``.
  :class:`~vtscore.converters.base.MediaConverter.name` (a
  :func:`property`) is also untouched; the MRO-descriptor check in
  ``_autoderive_plugin_metadata`` skips any attr already provided by
  an ancestor as a descriptor or non-empty string.

Coverage in :file:`tests_lib/core/test_plugin_metadata_defaults.py`.
No in-tree plugins were migrated to rely on the defaults; every
in-tree plugin already declares ``name``, ``display_name``, and
``description`` explicitly, and rewriting that to use derivation would
make the source less self-documenting.  The helper is for
third-party plugins that don't want to type the boilerplate.

#### Migration impact: external plugins

| Family | Cost |
|---|---|
| All plugin families | None required. Explicit declarations still win.  Plugins that didn't declare these attrs but were getting away with it through some other mechanism (typically via an :class:`AttributeError`-tolerant inventory route) now get a sensible auto-default instead. |

## What shipped

- **Phase E: Candidate #11 (plugin metadata defaults).**
  :meth:`PluginBase.__init_subclass__` auto-derives :attr:`name`,
  :attr:`display_name`, and :attr:`description` from the class name +
  docstring for any concrete subclass that doesn't declare them.
  Explicit declarations win; framework-level abstract bases and
  third-party intermediates that set ``_is_plugin_family_base = True``
  skip derivation so a derived ``name`` doesn't pollute downstream via
  MRO inheritance.  Candidate #10
  (``converter_description`` → ``description``) was already done by
  an earlier rename; no Phase E work was needed there.
- **Phase D: Candidates #5 + #6 + #8 (quality-of-life helpers).**
  :func:`vtscore.io.read_server_json` collapses the
  exists/is_file/read/parse/dict-check ritual into one call;
  :func:`vtscore.io.atomic_write_text` /
  :func:`vtscore.io.atomic_write_json` provide a single shared
  tmp+``fsync``+``os.replace`` writer (with per-writer unique tmp
  suffixes so concurrent writers don't fight over the same in-flight
  file).  :func:`vtsearch.threading.spawn` snapshots the caller's
  ``(user, dataset_ctx, detector_ctx)`` and replays them in a new
  daemon thread; closing the recurring "background thread forgot to
  call ``set_thread_user``" footgun for new code.  Migrated in-tree:
  the two ``server_json_file`` exporters, the ``server_csv_file``
  exporter, the two ``server_json_file`` importers, the labelset and
  settings JSON sources, and three ad-hoc thread spawn sites under
  :mod:`vtsearch.routes`.  External plugins keep working unchanged
  (helpers are additive; raw ``threading.Thread`` and inline JSON
  reads still work).
- **Phase C: Candidates #2 + #9 + #13 (shape unification).**
  :meth:`MediaConverter.convert_normalized` is the new framework
  entry-point: validation + default-fill runs once before
  :meth:`convert` is reached, so subclass bodies (and clipper-chain /
  importer call sites) can trust ``params`` is non-``None`` and
  fully-populated.  :mod:`vtscore.plugins.uploads` ships the
  :class:`UploadedFile` protocol plus :class:`CliUploadedFile` /
  :class:`BytesIOUploadedFile` adapters; the default ``run_cli`` on
  every file-accepting plugin family wraps path strings before
  dispatching so library-tier plugin bases stop mentioning Werkzeug.
  :meth:`DatasetImporter.yield_precomputed` collapses
  ``content_vectors`` / ``content_md5s`` / ``custom_metadata_map``
  writes into one call.  External plugins keep working unchanged
  (Werkzeug `FileStorage` satisfies the new protocol; ``get_param``
  remains as a shim; the three precomputed dicts still accept direct
  writes).  Generator-based `RawMedia` (#2's ambitious form) and
  implicit cancellation (#12) are explicitly deferred.
- **Phase A: Candidate #1 (declarative origin construction).** Six
  in-tree `build_origin` overrides deleted; framework default driven by
  `PluginField.include_in_origin` / `origin_serializer` and
  `DatasetImporter.extra_origin_keys` / `origin_suppressed`. Password
  and file fields now default-excluded from origin (closes the latent
  leak). External `DatasetImporter` plugins keep working unchanged
  (override-wins shim); other plugin families (`MediaSource`,
  `LabelImporter`, `LabelsetExporter`) untouched.
- **Phase B: Candidates #3 + #4 + #7 (declarative validation &
  templates).** New `vtscore.plugins.normalize.normalize_field_values`
  pass; `PluginField.template_vars` opt-in; `field_type="url"` /
  `"server_path"` fields auto-validated; in-tree plugins shed their
  manual strip / `raise ValueError` / `validate_*` / template calls;
  `validate_filepath_field` and its hardcoded `"filepath"` key deleted.
  `SyncSource` (settings + labelset) now wraps its public methods
  around new underscored template hooks (`_do_load` / `_do_save` /
  `_do_load_full` / `_do_peek_version`); breaking change for
  third-party sync source subclasses, which must rename their
  overrides to the `_do_*` form. Other external plugin families
  (`DatasetImporter`, `LabelImporter`, `LabelsetExporter`,
  `MediaSource`, settings importers/exporters) keep working unchanged
  (re-validation is idempotent on already-validated values;
  `sanitize_template_value` is idempotent on already-sanitised
  strings).

## Open follow-ups

- The full generator-based `RawMedia` shape (candidate #2's ambitious
  form, paired with #12 implicit between-yield cancellation) remains
  unscheduled. The minimal `yield_precomputed` helper shipped in
  Phase C handles the parallel-dict footgun without restructuring
  `run()`; revisit the generator form when a concrete importer wants
  the cancellation win as well.
- `MediaConverter.get_param` is retained as a back-compat shim. After
  a soak period confirms no third-party converter call sites rely on
  it, delete the helper.
- `PickleDatasetImporter.run_chunked_cli` still takes a bare path
  string (its own override, untouched by Phase C). Consolidate with
  the wrapping default in any future chunked-load refactor.
- `vtscore.exporters.server_json_file._atomic_write_text` is now a
  shim re-export of :func:`vtscore.io.atomic_write_text`. After a soak
  period confirms no third-party importer imports it directly, delete
  the alias.
- Library-tier background-thread spawn sites
  (:mod:`vtscore.datasets.load_pipeline`,
  :mod:`vtscore.embedding.loader`,
  :mod:`vtscore.media.embedder`) still call
  ``threading.Thread(target=..., daemon=True).start()`` directly.
  They aren't candidates for :func:`vtsearch.threading.spawn` as-is
  (the helper imports from ``vtsearch.*`` and would create a layering
  inversion); a library-tier ``vtscore.threading.spawn`` that handles
  only the dataset/detector context would fit the remaining sites.
  Park until a concrete bug surfaces.
