# OpenAPI schema + generated TS client

Status: in progress (pilot landing alongside this doc). Tracking issue: feature-brainstorm.md §12.9.

## The problem

The Flask backend and the Angular frontend agree on JSON shapes by
**convention only**. Every endpoint in `vtsearch/routes/` returns hand-
assembled dicts (`return jsonify({"good_applied": ..., "results": ...})`)
and every consumer in `frontend/src/app/` reads them through a hand-
written TypeScript interface in `frontend/src/app/models/api.models.ts`.

Nothing ties the two together. When a route adds, renames, or removes a
field, the matching DTO has to be updated by hand — and when it isn't,
the bug is silent. TypeScript's structural typing happily accepts a
response that "looks close enough" and the mismatch surfaces as a
runtime undefined-access or a quietly-missing UI element. The settings
PUT endpoint is the worst offender: ~200 lines of nearly-identical
type-coerce-then-validate-then-set blocks, none of which is reflected
anywhere the frontend can see.

## Goals

1. **Single source of truth.** The shape of every request body, query
   string, and response is declared once, in a schema, and that
   declaration is *load-bearing* — it parses and validates incoming
   requests and serialises outgoing responses. Drift between docs and
   code becomes impossible by construction.
2. **Self-documenting API.** A Swagger UI at `/api/docs` lets a
   developer (or third-party integrator) browse every endpoint, see the
   request/response shapes, and try requests live.
3. **Generated frontend client.** A TypeScript client generated from
   the OpenAPI spec replaces `frontend/src/app/models/api.models.ts`'s
   hand-maintained interfaces. Backend changes that aren't reflected in
   the frontend become **compile errors**, not runtime surprises.
4. **Pythonic validation.** Routes stop hand-parsing `request.json`
   with try/except towers. Bad input gets a standardised 422 response
   instead of a custom 400 message.

## Non-goals

- **No backwards-compatibility shim for old error envelopes.** The
  frontend reads `response.error.error` (a single string) today.
  flask-smorest emits a richer envelope (see below). The frontend's
  error handlers will be updated in lockstep — no compatibility layer
  in either direction. This is acceptable per the project's BC policy.
- **No incremental "documentation-only" pass.** The whole point is to
  make schemas load-bearing; declaring schemas without using them for
  parsing/serialisation just moves the drift problem down one level
  (now the schema disagrees with the route body).
- **Not migrating CLI entry points.** `python app.py --autodetect`
  produces JSON via the exporter system, which is already typed at its
  own boundary. The OpenAPI schema covers the HTTP API only.

## Approach

### Library: flask-smorest (+ marshmallow)

`flask-smorest` extends Flask's `Blueprint` with two decorators:

- `@blp.arguments(SomeSchema, location="json"|"query"|"form")` — parses
  the request, validates it against the schema, raises 422 on failure,
  and injects the result as a kwarg.
- `@blp.response(status, SomeSchema)` — runs the route's return value
  through the schema's `dump()`, guaranteeing the response matches the
  declared shape (extra keys are dropped, missing required keys raise).

Schemas are written in **marshmallow** (flask-smorest's native dialect).
We considered Pydantic v2 — flask-smorest has experimental support via
`apispec-pydantic-plugin` — but marshmallow's integration is older and
more battle-tested, and the schemas are small enough that porting later
(if §12.17 pulls us toward Pydantic for `_SETTING_SPECS`) is mechanical.

### Where schemas live

`vtsearch/schemas/` — a new sibling of `vtsearch/routes/`, mirroring the
route sub-package layout:

```
vtsearch/schemas/
    __init__.py
    common.py        # ErrorResponse, OK, pagination helpers
    settings.py      # AppSettingsSchema, SettingsUpdateSchema, ...
    labels.py        # LabelExportSchema, LabelImportSchema, ...
    datasets/        # mirrors vtsearch/routes/datasets/
    ...
```

Schemas are imported into the route module that uses them. There is no
global registry — flask-smorest discovers them through the decorator
chain.

### Swagger UI and spec endpoint

flask-smorest's `Api` object serves both:

- `/api/openapi.json` — the OpenAPI 3.x spec (machine-readable).
- `/api/docs` — Swagger UI (human-browsable).

Config goes in `app.py` alongside blueprint registration.

### Error envelope

flask-smorest's default `abort()` emits:

```json
{
  "code": 422,
  "status": "Unprocessable Entity",
  "errors": {"json": {"volume": ["Not a valid number."]}},
  "message": "Validation error"
}
```

vs. today's:

```json
{"error": "volume must be a number"}
```

The frontend's error pipeline (`HttpErrorResponse → error.error`) will
read `errors` (per-field) when present and fall back to `message`. The
hand-rolled `jsonify({"error": "..."})` patterns in route bodies are
replaced by `abort(400, message=...)` calls that produce the same
envelope as validation failures, so the frontend has one shape to
handle, not two.

### TS client generation

```
npm run generate-api-client
```

A new script under `frontend/scripts/` calls
`openapi-typescript-codegen` (or `openapi-fetch`) against
`/api/openapi.json` (or a checked-in snapshot for CI determinism) and
writes typed clients + DTOs into
`frontend/src/app/generated/api-client/`. The existing
`SettingsApiService` and friends are rewritten to thin wrappers around
the generated client; the hand-maintained interfaces in
`frontend/src/app/models/api.models.ts` are deleted blueprint-by-
blueprint as the corresponding backend route migrates.

CI guards drift two ways:
1. A `regenerate-and-diff` job rebuilds the spec from a running Flask
   instance and diffs against the checked-in snapshot. Mismatch fails
   the build (you forgot to regenerate after changing a schema).
2. `npm run build:prod` already typechecks every consumer; renaming a
   field that the frontend reads now breaks compilation immediately.

## Migration strategy

Blueprint by blueprint. Each blueprint migration is one PR:

1. Write schemas under `vtsearch/schemas/<area>.py`.
2. Replace `flask.Blueprint` with `flask_smorest.Blueprint` in the
   route module. Add `@blp.arguments` / `@blp.response` decorators.
3. Delete the hand-rolled validation and `jsonify` boilerplate.
4. Regenerate the TS client. Rewrite the matching Angular service to
   use it.
5. Delete the corresponding hand-maintained DTOs from
   `frontend/src/app/models/api.models.ts`.
6. Run `./run-tests.sh core api` and verify Swagger UI renders the
   migrated endpoints.

### Order (smallest → biggest)

1. **`settings/api.py`** (pilot, this PR) — 3 routes, ~230 LOC, the
   most repetitive validation in the codebase. High-leverage proof of
   concept.
2. **`auth.py`, `achievements.py`, `main.py`** — tiny, mostly GETs.
3. **`labels/`** — vote.py, importers.py, exporters.py.
4. **`detectors/`** — store.py, registry.py, scoring.py, find.py.
5. **`processors/`** — crud.py, scoring.py.
6. **`media/`** — list.py, server.py, embed.py.
7. **`datasets/`** — crud.py, registry.py, ui.py.
8. **`sorting.py`, `eval.py`, `file_browser.py`** — last because they
   have the loosest schemas (free-form sort results, eval JSON).

After each blueprint migration, the corresponding section of
`api.models.ts` is deleted.

### Concrete deletion targets

When the migration completes, these files should not exist:

- `frontend/src/app/models/api.models.ts` — every interface moves to
  `generated/api-client/`.
- Most of `vtsearch/routes/_shared.py`'s `get_json_or_400`,
  `get_json_safe`, `extract_plugin_fields`, `validate_required_fields`
  — replaced by `@blp.arguments` decorators. Helpers that operate on
  *plugin* fields (`extract_plugin_fields`, `validate_required_fields`)
  may survive in some form, since plugin field schemas are dynamic per-
  plugin and don't lend themselves to a static marshmallow schema.

## Relationship to the pre-existing permissive spec

The previous OpenAPI work (feature-brainstorm §12.9, shipped in commit
`44e9657`) added a separate, lighter-weight implementation:
`vtsearch.openapi.generate_openapi_spec` walks `app.url_map` and emits
a permissive spec (every route/method/path-param/docstring, but
`{type: object}` for every body and response). It's served at
`/openapi.json` and dumpable via `python app.py --openapi-schema`.

This plan's flask-smorest implementation serves a richer spec at
`/api/openapi.json` (plus Swagger UI at `/api/docs`) with real
request/response schemas. The two coexist today — same routes, two
specs. Once enough blueprints have migrated that the flask-smorest
spec covers the surface the permissive one does, **delete**:

- `vtsearch/openapi/` — the url-map walker.
- The `/openapi.json` Flask route registration in `app.py`.
- The `--openapi-schema` CLI flag in `app.py`'s argparse setup.
- The `--openapi-schema` references in `docs/CLI.md` and
  `docs/API.md § Machine-readable schema`.

The point of consolidation is "enough blueprints" — concretely, every
blueprint listed in the migration order below. Until then, both
endpoints are useful: integrators who only need the route inventory
can read `/openapi.json` without flask-smorest's typed
request/response gates getting in the way during their migration.

## Resolved questions

- **Plugin field endpoints** (`/api/exporters/<name>`,
  `/api/importers/<name>/run`, `/api/label-importers/import/<name>`,
  `/api/dataset/stage-import/<name>`, `/api/dataset/import/<name>`,
  `/api/detectors/registry/from-labelset/<name>`, etc.) — request
  shapes depend on each plugin's declared `fields`. **Decision:
  option (c) — generate a per-plugin marshmallow schema at startup
  from each plugin's `fields` declaration**, and pass it to
  `@blp.arguments(...)` per call site. The `FieldType` literal is
  `{file, folder, url, text, password, email, select, server_path,
  checkbox}`, which maps mechanically to marshmallow:
  - `text` / `url` / `email` / `password` / `server_path` / `folder` →
    `fields.String` (with `validate.Length(min=1)` when `required`).
  - `select` with static `options` → `fields.String` +
    `validate.OneOf(options)`. `dynamic_options=True` falls back to
    `fields.String` (frontend re-fetches via the existing
    `<name>/options` route).
  - `checkbox` → `fields.Boolean`.
  - `file` → not representable in JSON schema; routes that take a
    `file` field stay on multipart with `@arguments` omitted, declare
    error responses via `alt_response`, and declare the success body
    via `response` — same pattern as `add-to-pile` and
    `server-media-files/upload`. The handler still uses
    `extract_plugin_fields` / `validate_required_fields` internally
    for these.

  Schemas are built once at startup (after plugin discovery) and
  cached on the plugin instance; the route helper looks up
  `plugin._arg_schema` and `plugin._response_schema` (when defined).
  Spec consumers see real per-field types instead of
  `additionalProperties: true`, which is the whole point of this
  plan. (a) leaves the largest chunk of the API permanently
  un-typed; (b) gets the route into the spec but loses per-field
  typing.

- **Pagination.** flask-smorest has a built-in helper, but no current
  list endpoint actually needs it: `/api/medias/ids` is already the
  lightweight half of an ids+batch split (the metadata comes back
  from `POST /api/medias/batch` for the visible IDs), and nothing
  else returns lists big enough to matter. **Decision: closed — keep
  current shapes. Revisit only if a future endpoint actually needs
  paging.**

- **Spec snapshot.** **Decision: yes, check both
  `frontend/src/app/generated/api-client/` and the generated
  `openapi.json` into git.** Pros: deterministic builds, no need to
  spin up Flask in frontend CI, no install hop for a new dev. The
  "noisy diffs" downside is the feature here, not a bug — the diff
  *is* the API-change review, and the `regenerate-and-diff` CI job
  enforces it.

## Status

- [x] Plan written
- [x] `flask-smorest` + `marshmallow` added to `requirements/base.txt`
- [x] `Api` instance configured in `app.py`; `/api/openapi.json` and
      `/api/docs` serve the spec / Swagger UI
- [x] `vtsearch/schemas/` package created
- [x] Pilot: `settings/api.py` migrated to flask-smorest
- [x] `auth.py`, `achievements.py`, `main.py` (`/api/version`) migrated
      to flask-smorest. Raw-markdown + SPA-serving routes stay
      undecorated on the same `flask_smorest.Blueprint` (regular Flask
      routing, simply absent from the spec since they don't return
      JSON).
- [x] `labels/vote.py` migrated to flask-smorest (export / import /
      fill-from-sort). Validation errors now surface as 422 with the
      standard error envelope (matching the settings migration); the
      tests that previously asserted 400 + `{"error": ...}` for
      schema-level failures were updated to 422 + `errors` to match.
- [ ] `labels/exporters.py`, `labels/importers.py` — plugin-field
      routes, blocked on the *Open questions* decision below ("Plugin
      field endpoints"). Tracked here so we don't re-spend the
      research cost when we pick the labels migration back up.
- [x] `detectors/crud.py` migrated to flask-smorest (list / create /
      get / delete / rename / set-examples / combine). Schema-level
      validation failures (missing required fields, length / OneOf
      checks) surface as 422 with the standard ``errors`` envelope;
      handler-level rejects (``media_type='any'``, name collisions,
      empty merge) keep their HTTP codes (400 / 409 / 422) but now
      carry the standard ``message`` envelope instead of legacy
      ``{"error": str}``. Detector tests updated to match.
- [x] `detectors/registry.py` migrated to flask-smorest (list / register
      / load / unload / delete / rename / autorun / cancel). Schema-level
      validation failures (missing required fields, length checks)
      surface as 422 with the standard ``errors`` envelope; handler-level
      rejects (``media_type='any'``, whitespace-only name, not loaded)
      keep their HTTP codes (400 / 404 / 409 / 500) with the standard
      ``message`` envelope. The ``POST /api/detectors/registry/from-labelset/<importer>``
      route stays on plain Flask — its body is a plugin-field shape that
      doesn't fit a static marshmallow schema (see *Open questions /
      Plugin field endpoints* above). Registry tests in
      ``tests/api/test_error_recovery.py`` updated to match.
- [x] `detectors/scoring.py` migrated to flask-smorest (find-label /
      auto-detect). Schema-level validation failures (missing required
      ``detector_id``) surface as 422 with the standard ``errors``
      envelope; handler-level rejects (no medias loaded, detector not
      found, untrainable detector) keep their HTTP codes (400 / 404)
      with the standard ``message`` envelope. The find-label diagnostic
      response (detailed resolution stats + ``warning`` text) is passed
      as extra ``abort()`` kwargs so it flows through alongside
      ``message``. Find-label tests in ``tests/detectors/test_find_label.py``
      updated to match.
- [x] `detectors/find.py` migrated to flask-smorest (find /
      find/check-labels). The ``dataset_ids`` / ``detector_ids`` arrays
      stay un-validated at the schema layer so the handler can reject
      empty lists with 400 while resetting ``find_progress`` to idle on
      the way out (a schema-level 422 would bypass the handler and leave
      the tracker stale). Find tests in ``tests/api/test_dashboard.py``
      updated to read the ``message`` field instead of ``error``.
- [x] `detectors/labels.py` migrated to flask-smorest. JSON-shaped
      routes (save labels / labels-detail / vote) use the standard
      ``arguments`` + ``response`` decorators; schema-level validation
      failures (invalid ``vote`` value) surface as 422. The two binary
      GET routes — ``preview`` and ``thumbnail`` — declare their
      non-default JSON error responses via ``alt_response`` but do not
      model the success body in the spec (binary stream or text-content
      JSON depending on media type). The plugin-field
      ``import-labels/<importer_name>`` route stays on the legacy plain-
      Flask path (same reason as
      ``detectors/registry/from-labelset/<imp>``). Labelset-element
      tests in ``tests/detectors/test_labelset_elements_api.py``
      updated to match.
- [x] `processors/crud.py` and `processors/scoring.py` migrated to
      flask-smorest (autorun-extractor / autorun-localizer CRUD,
      pregen-processor list/add, single-shot extract/localize, and
      auto-extract/auto-localize). Schema-level validation failures
      (missing required ``name`` / ``media_type`` / ``extractor_type`` /
      ``localizer_type`` / ``config``) surface as 422 with the standard
      ``errors`` envelope; handler-level rejects (no medias loaded,
      media-type mismatch, unbuildable config, rename collisions, no
      autorun processors for active media type) keep their HTTP codes
      (400 / 404) with the standard ``message`` envelope. The shared
      sub-package ``processors_bp`` aggregator is gone — both
      blueprints are now registered directly with the flask-smorest
      ``Api`` in ``app.py`` (mirroring the ``detectors/`` layout).
      Extractor tests in ``tests/detectors/test_extractors.py`` and the
      cross-cutting checks in ``tests/integration/test_multi_media_coverage.py``
      updated to match.
- [x] `media/list.py` and `media/server.py` migrated to flask-smorest
      (medias/ids, medias/batch, vote, paragraph/text, add-to-pile,
      server-media-files CRUD + thumbnail, example-sort-server,
      example-sort-origin). Schema-level validation failures (missing
      required ``vote`` / ``filename`` / ``origin`` / ``key`` / ``ids``;
      invalid ``vote`` value) surface as 422 with the standard
      ``errors`` envelope; handler-level rejects (region_box length /
      range, bad-vote with region_box, no medias loaded, unknown origin
      type, path traversal) keep their HTTP codes (400 / 404 / 500) with
      the standard ``message`` envelope. The binary-streaming routes
      (audio / video / image / generic media / thumbnail) and the
      multipart-upload routes (add-to-pile, server-media-files/upload)
      declare error responses via ``alt_response`` and omit
      ``arguments``; the success body for the binary routes is left
      undescribed (mirroring the detector preview / thumbnail pattern).
      `media/embed.py` stays undecorated on a ``flask_smorest.Blueprint``
      because the dual-mode multipart-or-JSON dispatcher doesn't fit a
      single marshmallow schema (same SPA-pattern as the plugin-field
      routes). Tests in ``tests/core/test_medias.py``,
      ``tests/core/test_votes.py``, ``tests/api/test_error_recovery.py``,
      ``tests/api/test_api_contracts.py``,
      ``tests/detectors/test_patch_embedder.py``,
      ``tests/integration/test_multi_media_coverage.py``,
      ``tests/datasets/test_media_sources.py``, and
      ``tests/cli/test_load_sort_window.py`` updated to match.
- [x] `datasets/listings.py`, `datasets/status.py`, `datasets/ui.py`
      migrated to flask-smorest (media-types / embedders / clippers /
      converters / importers listings; dataset status + cancel; demo-list
      + demo-categories + browse-media-files (+select) + dashboard
      info/rename/disk-usage). The plugin ``to_dict()`` payloads are
      declared as ``fields.Dict()`` rather than nested schemas — the
      inner shapes are plugin-dependent and already round-trip cleanly
      via ``to_dict()``; redeclaring every field would only duplicate
      the source of truth. Schema-level validation failures (missing
      required ``source`` / ``path`` on the select endpoint, missing
      ``name`` on the rename endpoint) surface as 422 with the standard
      ``errors`` envelope; handler-level 400s (path traversal,
      whitespace-only rename) use the standard ``message`` envelope.
      404s are intercepted by the app-level ``NotFound`` errorhandler
      in ``app.py``, which matches a more specific exception subclass
      than flask-smorest's ``HTTPException`` handler and so wins on
      every 404 — those keep the legacy ``{"error": "Not Found",
      "request_id": "..."}`` shape regardless of the ``message=`` kwarg
      passed to ``abort()``. Tests in ``tests/api/test_dashboard.py``
      and ``tests/datasets/test_datasets.py`` updated to match
      (``message`` for 400, ``error`` for 404). The heavier dataset
      modules (``staging.py``, ``registry.py``) stay on plain Flask
      blueprints for now — they involve plugin-field shapes that need
      the *Resolved questions / Plugin field endpoints* decision in
      hand (importer staging / import).
- [x] `eval.py` migrated to flask-smorest (labeling-progress,
      labeling-status, indicator-score-history, eval/train-and-score +
      /result). Schema-level validation failures (missing required
      ``metric`` / ``job_id``; invalid metric value) surface as 422 with
      the standard ``errors`` envelope; handler-level rejects (no
      good/bad votes, no label history, job not found, wait-mode error)
      keep their HTTP codes (400 / 404 / 500) with the standard
      ``message`` envelope. The train-and-score response schema uses
      ``unknown = "include"`` so the metric-specific data key
      (``error_cost`` / ``stability`` / ``diversity``) and the historical
      cancelled-status response flow through unchanged. The
      ``test_invalid_metric_rejected`` test was updated from 400 to 422
      to match.
- [x] `file_browser.py` migrated to flask-smorest (single ``GET
      /api/browse`` endpoint). Schema-level validation failures surface
      as 422 with the standard ``errors`` envelope; handler-level
      rejects (path traversal, permission denied) keep their HTTP codes
      (400 / 403) with the standard ``message`` envelope. 404s
      (directory not found) are intercepted by the app-level
      ``NotFound`` errorhandler in ``app.py`` and keep the legacy
      ``{"error": "Not Found", "request_id": ...}`` shape. File-browser
      tests in ``tests/api/test_file_browser.py`` updated to read
      ``message`` instead of ``error`` for the 400 cases.
- [x] `datasets/load.py` migrated to flask-smorest (import-local-folder,
      load-demo, load-file, load-folder, load-source, export, clear).
      JSON-shaped routes (load-demo, load-folder, load-source, clear)
      use the standard ``arguments`` + ``response`` decorators;
      schema-level validation failures (missing required ``name`` /
      ``path`` / ``source``) surface as 422 with the standard
      ``errors`` envelope. Multipart routes (import-local-folder,
      load-file) and the binary-stream route (export) omit
      ``arguments`` and declare error responses via ``alt_response``,
      mirroring ``add-to-pile`` and ``server-media-files/upload``.
      Handler-level rejects (unknown demo, invalid path, importer
      unavailable, no dataset loaded, no files uploaded, bad
      clipper_params / vectors_file) keep their HTTP codes (400 / 500)
      with the standard ``message`` envelope. Tests in
      ``tests/api/test_path_validation.py``,
      ``tests/io/test_local_folder_upload.py``,
      ``tests/io/test_npz_dataset_import.py``,
      ``tests/converters/test_document_and_converters.py``, and
      ``tests/converters/test_converter_selection.py`` updated to
      read ``message`` instead of ``error``.
- [ ] Frontend `SettingsApiService` rewired to generated client
- [ ] `frontend/src/app/models/api.models.ts` settings section deleted
- [ ] Remaining dataset blueprints: ``datasets/staging.py`` and
      ``datasets/registry.py`` — both involve plugin-field shapes
      (``stage-import/<name>``, ``import/<name>``,
      ``registry/<id>/load`` field overrides) for which we now have
      the *Resolved questions / Plugin field endpoints* decision in
      hand, so they're unblocked.
- [ ] Remaining blueprints (see Order above)
- [ ] Delete the pre-existing permissive `/openapi.json` +
      `--openapi-schema` once flask-smorest covers every blueprint
      (see "Relationship to the pre-existing permissive spec" above)
