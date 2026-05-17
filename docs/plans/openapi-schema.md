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
npm run generate-api-client                # regenerate TS client from frontend/openapi.json
npm run regenerate-openapi-snapshot        # re-dump the spec from a fresh Flask import
```

The generator is `ng-openapi-gen` (Angular-specific; emits HttpClient-
based functions so the existing `activeContextInterceptor` keeps
attaching `X-Dataset-Id` / `X-Detector-Id` headers without a separate
fetch middleware). It reads `frontend/openapi.json` and writes typed
clients + DTOs into `frontend/src/app/generated/api-client/`.

Hybrid checked-in policy (deviation from the original plan's "check in
both"):

- `frontend/openapi.json` IS tracked — small, semantically meaningful,
  the git diff really does show API changes at PR time.
- `frontend/src/app/generated/` is **gitignored** — generated TS is a
  deterministic function of the spec + generator version, so checking
  it in adds noisy diffs and merge conflicts for no benefit.
- The Angular build's `prebuild` / `prebuild:prod` npm hooks run
  `ng-openapi-gen` automatically before every `ng build`, so a fresh
  checkout's first `npm run build:prod` populates the directory.

CI guards drift three ways:
1. The `Pyright` workflow gains a final "OpenAPI snapshot drift check"
   step: it regenerates the spec via `python scripts/dump_openapi.py`
   and diffs against `frontend/openapi.json`. Mismatch fails CI (you
   changed a schema without re-running `npm run regenerate-openapi-snapshot`).
2. `npm run build:prod` always re-runs the generator first (via the
   `prebuild:prod` hook) and then typechecks every consumer, so
   renaming a field that the frontend reads now breaks compilation.
3. The `ApiConfiguration`, function modules, and model types are
   reachable via direct paths under `generated/api-client/` only — the
   barrel index files are intentionally disabled in
   `ng-openapi-gen.json` (`indexFile: false`, `functionIndex: false`,
   `modelIndex: false`) so that an unrelated import doesn't accidentally
   drag the whole 186-model surface into the initial bundle.

### Bundle-size discipline

Consumers MUST `import type { Foo }` (not `import { Foo }`) for any
generated DTO, and import functions / `ApiConfiguration` from their
specific module paths under `generated/api-client/`. Pulling a model in
as a runtime import (or going through a barrel) adds the entire
`models/` graph to the initial chunk — measured at ~36 kB versus the
~6 kB cost of importing just the functions a service actually calls.

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
`vtsearch.openapi.generate_openapi_spec` walked `app.url_map` and
emitted a permissive spec (every route/method/path-param/docstring, but
`{type: object}` for every body and response). It was served at
`/openapi.json` and dumpable via `python app.py --openapi-schema`.

That permissive spec has been **deleted** now that flask-smorest covers
every blueprint and serves a richer spec at `/api/openapi.json` (plus
Swagger UI at `/api/docs`) with real request/response schemas. The
removals — `vtsearch/openapi.py`, the `/openapi.json` route on
`main_bp`, the `--openapi-schema` CLI flag, the
`tests/api/test_openapi_schema.py` tests, and the docs references in
`docs/CLI.md` / `docs/API.md` — all landed together in the same PR that
checks off this follow-up.

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
- [x] `labels/exporters.py` migrated to flask-smorest (``GET
      /api/exporters``, ``POST /api/exporters/export``). Schema-level
      validation failures (missing required ``exporter_name``) surface
      as 422 with the standard ``errors`` envelope; handler-level
      rejects (unknown exporter, missing plugin field, invalid
      ``filepath``, exporter raised) keep their HTTP codes (404 / 400 /
      500) with the standard ``message`` envelope. ``field_values`` is
      declared as ``fields.Dict`` because its inner keys depend on the
      named exporter; the handler validates it against the selected
      plugin's :attr:`fields`. Tests in
      ``tests/api/test_error_recovery.py``,
      ``tests/api/test_api_contracts.py``, and
      ``tests/api/test_path_validation.py`` updated to expect 422 for
      schema rejections and ``message`` instead of ``error`` for
      handler-level 400s.
- [x] `labels/importers.py` migrated to flask-smorest (``GET
      /api/label-importers``, ``POST
      /api/label-importers/ingest-missing``). Schema-level validation
      failures (missing required ``entries``, empty list) surface as
      422. The plugin-field route ``POST
      /api/label-importers/import/<importer_name>`` stays on the
      legacy plain-Flask path on the same ``flask_smorest.Blueprint``
      (same pattern as ``detectors/labels.py``'s
      ``import-labels/<importer_name>`` and
      ``detectors/registry/from-labelset/<importer>``) — its body is a
      plugin-field shape that doesn't fit a static marshmallow schema.
      See *Resolved questions / Plugin field endpoints*.
- [x] `datasets/staging.py` migrated to flask-smorest (available-files,
      combine, stage-file, stage-demo, clear-staging, import field
      options). Schema-level validation failures (missing required
      ``datasets`` / ``field_key``; ``datasets`` shorter than 2)
      surface as 422 with the standard ``errors`` envelope;
      handler-level rejects (path validation, unknown demo, missing
      importer) keep their HTTP codes (400 / 500) with the standard
      ``message`` envelope. Multipart upload (``stage-file``) omits
      ``arguments`` and declares error responses via ``alt_response``.
      The two plugin-field routes (``stage-import/<importer>``,
      ``import/<importer>``) stay on the legacy plain-Flask path. The
      combine-datasets path-validation test was updated to read
      ``message`` instead of ``error``.
- [x] `datasets/registry.py` migrated to flask-smorest (list, load,
      unload, delete, rename, readers, stats). Schema-level validation
      failures (missing required ``name`` on rename, missing or
      wrong-typed ``readers`` on the readers endpoint) surface as 422
      with the standard ``errors`` envelope; handler-level rejects
      (not loaded, not the creator) keep their HTTP codes (400 / 403)
      with the standard ``message`` envelope. 404s are intercepted by
      the app-level ``NotFound`` errorhandler in ``app.py`` and keep
      the legacy ``{"error": "Not Found", "request_id": ...}`` shape.
      The ``readers`` field is declared as ``fields.Raw`` with a
      custom validator (rather than ``fields.List(fields.String())``)
      so that numeric items are rejected as 422 instead of silently
      coerced to strings. Tests in
      ``tests/api/test_multi_user_dataset_access.py`` updated from 400
      to 422 for the two invalid-body cases.
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
- [x] `sorting.py` migrated to flask-smorest (sort / learned-sort start +
      result, votes get/clear/seed-from-examples, textsort-suggestions
      get/post, inclusion get/post, safe-thresholds get/post,
      diversity-tree/next, plus the two multipart routes
      example-sort and label-file-sort). Schema-level validation
      failures (missing required ``text`` / ``job_id`` / ``examples`` /
      ``inclusion`` / ``safe_thresholds``; type-mismatched
      ``inclusion`` / ``safe_thresholds``) surface as 422 with the
      standard ``errors`` envelope. The ``inclusion`` field is
      declared as ``fields.Raw`` with a custom validator (rather than
      ``fields.Integer(strict=True)``) so that ``3.7`` continues to
      round to ``3`` while booleans are still rejected explicitly —
      preserving the pre-migration coercion behavior. Handler-level
      rejects (empty / whitespace ``text``, no good/bad votes, no
      medias loaded, embedder doesn't support text, multipart no-file
      cases, invalid label file, etc.) keep their HTTP codes (400 /
      404 / 500) with the standard ``message`` envelope. The
      ``supports_text=false`` flag and the missing-job ``status=missing``
      flag flow through as extra ``abort()`` kwargs. The
      ``diversity-tree/next`` route serves both GET and POST on the
      same function and reads its optional body with
      ``request.get_json(silent=True)`` rather than ``@arguments`` so
      the two methods behave identically (a missing body must not
      422). Tests in ``tests/sorting/test_sorting.py``,
      ``tests/sorting/test_safe_thresholds.py``,
      ``tests/sorting/test_label_sorting.py``,
      ``tests/core/test_inclusion.py``,
      ``tests/api/test_api_contracts.py``,
      ``tests/api/test_error_recovery.py``,
      ``tests/api/test_error_envelope.py``, and
      ``tests/detectors/test_image_embedders.py`` updated to match
      (``message`` for 400, ``errors`` for 422; the error-envelope
      test that pinned the legacy ``{"error": ..., "request_id": ...}``
      shape was repointed at ``/api/embed``, which is still on the
      legacy ``get_json_or_400`` helper because its dual-mode
      multipart-or-JSON dispatcher doesn't fit a single marshmallow
      schema).
- [x] `settings/io.py` and `settings/sources.py` migrated to flask-smorest
      (settings-importers / settings-exporters / settings-sources /
      labelset-sources). The two plugin-field routes
      (``POST /api/settings-importers/import/<importer_name>``) stay on
      the legacy plain-Flask path on the same smorest blueprint — same
      pattern as ``labels/importers.py``. Schema-level validation
      failures (missing required ``exporter_name`` / ``source_name``)
      surface as 422 with the standard ``errors`` envelope; handler-
      level rejects (unknown exporter / source, detector not loaded,
      missing plugin field, invalid ``filepath``, exporter raised) keep
      their HTTP codes (404 / 400 / 500) with the standard ``message``
      envelope. The nullable-GET endpoints
      (``/api/settings-sources/active`` and
      ``/api/detectors/<n>/labelset-source``) return ``jsonify(None)``
      to short-circuit flask-smorest's ``schema.dump`` (which would
      otherwise turn ``None`` into ``{}``); the sync endpoint always
      includes a ``keys`` list in its response (even when empty)
      because marshmallow's ``getattr`` fallback resolves ``keys`` on a
      dict to the built-in method. The
      ``test_missing_exporter_name_400`` test in
      ``tests/io/test_settings_io.py`` was updated to expect 422 with
      the ``errors`` envelope.
- [x] TS client generation pilot landed (settings only):
      ``ng-openapi-gen`` wired up via ``frontend/ng-openapi-gen.json``;
      ``frontend/openapi.json`` snapshot checked in;
      ``frontend/src/app/generated/api-client/`` gitignored and
      regenerated by the ``prebuild`` / ``prebuild:prod`` npm hooks
      (and on demand via ``npm run generate-api-client``);
      ``scripts/dump_openapi.py`` dumps the live flask-smorest spec for
      both the snapshot regen (``npm run regenerate-openapi-snapshot``)
      and the CI drift guard (new step on the ``Pyright`` workflow).
      ``SettingsApiService`` and ``SettingsStateService`` now call the
      generated ``apiSettingsGet`` / ``apiSettingsPut`` /
      ``apiSettingsDefaultsGet`` / ``apiVersionGet`` functions; the
      ``AppSettings`` interface was deleted from
      ``frontend/src/app/models/api.models.ts`` and consumers
      (``settings-modal``, ``view-controls``, ``settings-state``)
      ``import type``-only it from the generated module.
- [x] Pre-existing permissive `/openapi.json` + `--openapi-schema`
      deleted. Removed ``vtsearch/openapi.py`` (the url-map walker),
      the ``/openapi.json`` route on ``main_bp``, the
      ``--openapi-schema`` CLI flag in ``app.py``, and
      ``tests/api/test_openapi_schema.py``. The ``--format`` flag's
      help text no longer mentions ``--openapi-schema``. Docs updated:
      ``docs/API.md § Machine-readable schema`` and ``docs/CLI.md``
      both point at ``GET /api/openapi.json`` (and Swagger UI at
      ``/api/docs``) as the single source.
- [x] ``AuthService`` and ``AchievementsService`` rewired to the
      generated TS client. ``AuthService`` now calls
      ``apiAuthStatusGet`` / ``apiAuthLoginPost`` / ``apiAuthLogoutPost``;
      ``AchievementsService`` now calls ``apiAchievementsGet`` /
      ``apiAchievementsCategoryIdAcknowledgePost`` /
      ``apiAchievementsCheckPhrasePost``. The local
      ``AuthStatus`` / ``AchievementInfo`` / ``AchievementsState`` /
      ``DocInfo`` / ``PendingAnnouncement`` / ``PhraseCheckResult``
      interfaces were deleted from the service files; consumers
      (``achievements-tab``, ``achievement-unlock-host``)
      ``import type``-only the generated ``AchievementEntry`` /
      ``AchievementState`` / ``DocEntry`` / ``PendingAnnouncement`` from
      their direct module paths under ``generated/api-client/models/``.
      ``HttpContext``-based ``SKIP_ERROR_TOAST`` flagging on
      ``/api/auth/status`` and ``/api/auth/login`` is preserved by
      passing the ``HttpContext`` through the generated function's
      4th-positional ``context`` argument.

## Open follow-ups

- **Migrate the remaining Angular services to the generated client.**
  Settings, auth, and achievements have all moved over. Each follow-up
  PR picks one blueprint area (medias, sorting, detectors, datasets,
  eval, labels, exporters, …), rewires the matching Angular service(s)
  to call the generated function modules under
  ``frontend/src/app/generated/api-client/fn/``, and deletes the
  corresponding hand-maintained interfaces from
  ``frontend/src/app/models/api.models.ts``. The hybrid imports
  (``import type`` for DTOs, direct function-module paths for
  runtime symbols, no barrel) are required to keep the initial bundle
  under the 525 kB budget — see *Bundle-size discipline* above.
- **Per-plugin schemas for plugin-field routes.** The four routes
  whose request body is a plugin-field shape stay on the legacy
  plain-Flask path on their (now smorest-typed) blueprints:
  ``POST /api/label-importers/import/<importer_name>``,
  ``POST /api/dataset/stage-import/<importer_name>``,
  ``POST /api/dataset/import/<importer_name>``, and
  ``POST /api/detectors/<name>/import-labels/<importer_name>``
  (plus ``POST /api/detectors/registry/from-labelset/<importer_name>``).
  The *Resolved questions / Plugin field endpoints* section above
  describes the eventual design — generate a per-plugin marshmallow
  schema at startup from each plugin's ``fields`` declaration and
  attach it to the route call site. That requires either registering
  one Flask URL rule per plugin variant at startup or invoking
  ``plugin._arg_schema.load(...)`` manually inside the handler.
  Neither is wired yet; the routes are documented in their module
  docstrings as "Plugin-dependent body shape: not described in the
  OpenAPI spec" so spec consumers see them in the URL map but without
  a request body schema.
