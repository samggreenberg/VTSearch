# Plugin author interface streamlines

Status: **Discovery / brainstorm.** No code changes yet. This file
enumerates simplifications to the plugin-author interfaces across every
plugin family — the same shape of change as the recent
"importers no longer own embedding or converting" refactor.

## Background

We just split **embedding** and **converting** out of `DatasetImporter`.
Before: every importer had to call the embedder, run converters, and
populate a single combined media dict. After: importers yield raw
source-type media records; the framework owns the rest. The win wasn't
"less typing" — it was that *importers stopped mixing what they do with
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

## The recurring pattern — what counts as a candidate

A simplification belongs here when it satisfies all three of:

1. The behavior is **identical across implementations** (or at least
   uniformly defaulted).
2. The behavior is a **framework concern, not a plugin concern** — the
   plugin author has no business knowing about it.
3. Forgetting it is a **silent footgun** (security gap, wrong path
   produced, missing user context, etc.) rather than a loud error.

The embedding/converting extraction hit all three. The candidates below
do too.

## Candidates

Each candidate names the **leak**, where it shows up, the **fix**, and a
short evaluation. Numbering is for cross-reference only — see "Priority"
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
expressed as plugin code. Low risk — the default already does the right
thing for new fields; the overrides exist only because the data model
didn't have an "omit me" knob.

**Shim.** Trivial — `build_origin()` stays a hook on the base class. If a
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
or — cleaner — let importers yield a `RawMedia(filename, bytes_or_path,
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
(`validate_filepath_field` — which is hard-coded to the literal key
`"filepath"` rather than driven by the field schema). The webhook
exporter calls `validate_url(url)` in its `export()` body — the
labelset/settings sources sprinkle `sanitize_template_value` over
`{detector_name}` / `{username}` substitutions one by one.

**Fix.** Make the field schema carry the validator. We already have
`field_type="server_path"` and `field_type="url"` as declared types but
they're cosmetic — the framework doesn't enforce anything based on
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
special-casing the literal key `"filepath"` — every field carries its
own validation rule.

**Shim.** Purely additive — the existing `validate_url`,
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
flag but the framework doesn't enforce it consistently — plugins still
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
and runs — it just never fires, because the framework rejected the empty
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

**Shim.** Option (b) is intrinsically backwards-compatible — it's a new
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

**Shim.** Purely additive — `vtscore.io.atomic_write_text` /
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
Same shape as #3 — declarative replaces imperative.

**Shim.** Default `template_vars=None` on `PluginField` means "framework
does no substitution" — i.e. existing fields keep receiving the raw
template string and the plugin keeps doing its own substitution as
today. Only fields that explicitly declare `template_vars=[...]` opt
into framework substitution. Migration is per-field and per-plugin, on
the plugin's own schedule. If a third-party plugin later opts in and
*also* leaves a `sanitize_template_value` call in its body, that's a
no-op on an already-resolved string (sanitize is idempotent).

### 8. Background-thread context propagation

**Leak.** CLAUDE.md flags this explicitly: "Background threads spawned
from a request handler must call `vtsearch.auth.set_thread_user()` so
per-user writes resolve correctly." Same applies to
`set_thread_dataset_context()` / `set_thread_detector_context()` for
the `medias` / `good_votes` / etc. proxies. `JobManager` and the
dataset-load thread spawn sites do this; ad-hoc background threads in
plugins are on their own.

**Fix.** Provide `vtsearch.threading.spawn(target, ...)` that snapshots
the current `(user, dataset_ctx, detector_ctx)` and replays them inside
the new thread. Plugin / route code uses `spawn()` instead of
`threading.Thread(target=...).start()`; the context plumbing disappears
from sight.

**Eval.** Medium win. Lower volume than #1–#3 but high consequence —
when this leak fires it silently writes per-user settings to the wrong
user or crashes on a missing context. The "logical-bug-audit.md" plan
already calls out context-propagation gaps as a recurring root-cause
pattern; this would close most of them.

**Shim.** Purely additive — `vtsearch.threading.spawn(...)` is a new
helper. `set_thread_user()` / `set_thread_dataset_context()` /
`set_thread_detector_context()` stay exported with the same signatures.
A third-party plugin that builds its own `threading.Thread` and calls
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

**Eval.** Solid win. Same as #4 — converts an opt-in helper plugin
authors might forget into a framework-guaranteed input shape.

**Shim.** Keep `get_param()` as a method on `MediaConverter` — once the
framework has pre-populated `params`, `get_param(params, key)` collapses
to a plain `params[key]` lookup, so old converters that route through it
keep working. Converters that did `params["key"]` directly *start*
working (they previously crashed on `None`), which is a strict
improvement. Keep `validate_params()` as a hook the framework calls
before normalizing — if a third-party converter overrode it for custom
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
`description` — check). One-line change per converter.

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
— just typing. Lowest priority of the candidates.

**Shim.** Purely additive — the defaults only fire when the class attr
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

**Shim.** Tracks #2 — if the importer uses the old non-generator `run()`
shape, the framework can't insert cancellation points between yields, so
those importers keep relying on their hand-placed
`check_dataset_cancelled()` calls (which stay exported and functional).
Only importers that opt into the generator/`RawMedia` shape get free
between-yield cancellation. Third-party importers that mix both
(generator yields + explicit `check_dataset_cancelled()`) are fine —
the explicit call is a no-op redundant check.

### 13. `field_type="file"` returns Werkzeug `FileStorage`

**Leak.** Library-tier plugin bases (`vtscore/labels/importers/base.py`,
`vtscore/datasets/importers/base.py`) document that
`field_type="file"` *receives a Werkzeug FileStorage* — a Flask/app-tier
type — but the module is library-tier and supposedly Flask-clean.
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
the FileStorage attrs old plugins already use — `.filename` (str) and
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

**P0 — uniform safety + the field-schema collapse**
- #3 Declarative path/URL/template validation (security; highest
  consequence; unblocks #7)
- #4 Framework-enforced `required` (consistency; low risk)
- #7 Declarative template variables (couples to #3)

**P1 — uniform shape**
- #1 Origin construction default (deletes overrides)
- #2 Content optimization dicts → `RawMedia` (eliminates parallel-dict
  bug class; enables #12)
- #9 Converter param normalization (closes the `get_param` hole)
- #13 Drop Werkzeug from library-tier plugin bases

**P2 — quality of life**
- #5 Server-JSON read helper
- #6 Promote `atomic_write_text` / `atomic_write_json`
- #8 `spawn()` for thread context propagation (touches a different
  axis; can land independently)
- #12 Implicit cancellation (depends on #2)

**P3 — boilerplate trim**
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
  *not* under this protection — those can change shape as needed.
- Not unifying base classes across families into a single
  super-base — they have legitimately different return shapes
  (`run` vs `export` vs `convert`). The unification proposed here is
  *behind* the bases, in the framework that calls them.

## Open follow-ups

- None yet. This is a discovery doc; pick which candidates to schedule
  and crack them open in separate plan files (or fold them inline into
  EXTENDING.md once they ship).
