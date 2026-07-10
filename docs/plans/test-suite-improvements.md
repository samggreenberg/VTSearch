# Test Suite Improvements (coverage)

Open work from a full evaluation of the test suite (2026-07). Baseline measurements that motivate the items below, taken on a 4-core cloud container, cold caches, `pytest tests/ tests_lib/ -q -n auto` with coverage enabled:

- Backend line coverage (vtsearch + vtscore combined): **83%**. Cold spots are concentrated in embedder wrappers (stubbed session-wide), `vtsearch/cli_main.py` (6% — only covered by the excluded `slow` subprocess tests), and a handful of network-facing/plugin modules.
- Frontend: 115 spec files; ~68% of components and ~76% of services have specs; **0 of 4 HTTP interceptors** are tested.

Each item below is independent; pick any and delete it when it ships.

## Coverage — backend

<!-- item-sep -->

- **In-process tests for `vtsearch/cli_main.py`** — 253 statements at **6% coverage** in the default suite: its only real exercise is the two `slow`-marked subprocess tests, which are excluded by default. Add in-process tests that call the CLI entry points with mocked heavy stages (argument parsing, flag validation and combinations, exporter/importer wiring, error exits). This closes the biggest single route-to-user coverage hole *and* could let the 16s subprocess tests shrink to one smoke test.

<!-- item-sep -->

- **PullWrest source behavior** — `vtscore/datasets/sources/pullwrest.py` (250 lines, 31% coverage) is a network-facing origin source with zero behavioral tests: HTTP download failure, contentID-keyed temp-cache reuse, and `cleanup()` of the owned temp dir (a past temp-dir bug is referenced in a comment in `tests/detectors/test_resolver.py` but has no regression test). Test with mocked HTTP.

<!-- item-sep -->

- **Cross-dataset label restoration (`_resolve_unmatched`)** — `vtscore/detectors/label_restoration.py` (121 lines): the second-pass fallback that re-matches a detector's labels on a *different* dataset by resolving origins and comparing MD5s — i.e. the exact "detectors are reusable across compatible datasets" product promise — has no test. Exercise: train on dataset A, load the detector against dataset B with overlapping files, assert labels re-attach; plus the no-match path.

<!-- item-sep -->

- **Embedder wrapper code paths** — Every embedder is stubbed session-wide, and the GPU suite (`tests_lib/gpu/test_gpu.py`) loads CLAP/CLIP/X-CLIP/E5 straight from transformers, bypassing VTSearch's own wrappers. Consequently `_paraspeechclap_model.py` is at 0%, `_clap_shared.py` 29%, `embedder_ast.py` 28%, `embedder_whisper.py` 29%, `_eupe_shared.py` 35%, `embedder_siglip2.py` 35%, `_dinov2/3_shared.py` 39%, `embedder_videomae.py`/`embedder_languagebind.py` 54%, `embedder_e5.py` 25%. Full model loads belong behind `gpu`/`slow` markers, but the pre/post-processing around the forward pass (input prep, pooling, normalization, batching, device selection) can be unit-tested on CPU with a fake tiny model. At minimum, route the GPU tests through the wrappers instead of raw transformers.

<!-- item-sep -->

- **Video media type parity** — `vtscore/media/video/media_type.py` (863 lines, 44% coverage, referenced by 1 test file) versus audio's media type (1,011 lines, 9 test files). Cover thumbnailing, clipping/segment handling, byte/format edge cases, and error paths at parity with the audio tests.

<!-- item-sep -->

- **`routes/media/list.py` filter/pagination edges** — the largest route module (558 statements, 66% coverage). The ~180 missed statements are concentrated in filter combinations, pagination edges, and error paths. Cover via API-level tests against the existing test medias.

<!-- item-sep -->

- **Eval runner + eval routes** — `vtscore/eval/runner.py` (56%) and `vtsearch/routes/eval.py` (67%): failure paths (bad dataset/detector refs, cancellation mid-run, empty categories) look untested relative to the happy-path eval tests.

<!-- item-sep -->

- **Embedder-switch re-embed flow** — `vtscore/detectors/embedder_sync.py` (137 lines): `maybe_start_label_reembed` and the switch-embedder-then-re-embed-labels flow are barely pinned down (two symbol mentions across the suite). Test the trigger conditions, progress reporting, and the no-op path.

<!-- item-sep -->

- **`ImporterBase` validation/error paths** — `vtscore/datasets/importers/base/core.py` (611 lines) is exercised only through subclasses; the base-class error handling (bad params, partial-batch failures, cancellation propagation) has no direct tests, so a base regression surfaces as a confusing subclass failure.

<!-- item-sep -->

- **Load profiler instrument** — `vtscore/datasets/stages/_load_profiler.py` (226 lines, 28% coverage, zero test references). It's the env-gated measurement tool behind `docs/plans/progress-weight-calibration.md`; a silent regression corrupts calibration data with nothing to catch it. A couple of tests with the env flag set (records stages, writes the expected schema) suffice.

<!-- item-sep -->

- **Assorted smaller backend gaps** — worth folding into nearby work rather than dedicated efforts: `vtscore/datasets/metadata.py` (49%), `vtsearch/settings_store.py` (883 lines, thin direct coverage vs the settings routes), the achievements engine (`vtsearch/achievements.py`, 719 lines, 2 test files; `vtsearch/schemas/achievements.py` has zero references), `vtscore/exporters/gui/` origin-formatting output, `vtscore/datasets/sources/local_archive_member.py` (the importer twin is tested; this source path isn't), the `vtscore/sync/` normalize-before-delegate base contract, and the SSE event bus (`vtscore/concurrency/events.py`, one test file).

## Coverage — frontend

<!-- item-sep -->

- **HTTP interceptors (0 of 4 tested)** — `frontend/src/app/interceptors/`: `error.interceptor.ts` (197 lines, the global error-to-toast funnel for every API call), `active-context.interceptor.ts` (stamps `X-Dataset-Id`/`X-Detector-Id` — the backbone of the multi-context state model; a regression silently mistargets every mutation), `achievements-refresh` and `timezone`. These are pure functions over `HttpRequest`/`HttpHandler`, cheap to test with `HttpTestingController`, and currently the highest-risk untested frontend code.

<!-- item-sep -->

- **Browse-canvas subsystem** — `components/browse-canvas/browse-canvas.component.ts` (2,640 lines — largest file in the frontend) plus `browse-minimap.component.ts` (538) have no specs; only the extracted `view-transform.ts` util does. Full canvas rendering isn't unit-testable, but the extracted pure logic is: start by testing `hex-render.util.ts` (219) and `bin-geometry.ts` (143), and extract+test hit-testing/pan-zoom math from the component rather than testing the component wholesale.

<!-- item-sep -->

- **Active-context state chain** — `services/active-context.service.ts` (189 lines, the source-of-truth singleton for the active dataset/detector; its *watcher* is tested but the service isn't) and `components/context-pulldown/context-pulldown.component.ts` (579 lines, the context switcher UI) plus `browse-context.guard.ts`. Together with the interceptor item this covers the whole context-selection pipeline the backend's H34 header checks depend on.

<!-- item-sep -->

- **Cross-cutting singleton services** — no specs for: `toast.service.ts` (215 — every error path funnels through it), `dashboard-modals.service.ts` (201 — orchestrates import/export/new-detector modal flows), `media-metadata-cache.service.ts` (175) and `tile-cache.service.ts` (143 — caching layers with eviction/staleness logic), `detectors-crud-api.service.ts` (138 — the only substantial `*-api.service` without a spec), `new-thing-flows.service.ts` (146), `running-jobs.service.ts` (139), `labelset-state.service.ts` (134), `embedder-capability.service.ts` (149). API services have an established `HttpTestingController` pattern to copy.

<!-- item-sep -->

- **Achievements feature (zero tests end-to-end)** — `services/achievements.service.ts` (196), `components/achievements-tab/` (179), and the `achievements-refresh` interceptor: an entire user-facing feature with no frontend tests (and thin backend tests — see the backend item).

<!-- item-sep -->

- **Untested large utils/components** — `utils/managed-columns.ts` (355 lines, pure logic, trivially testable), `components/folder-browser/folder-browser.component.ts` (420, server-filesystem navigation used by importer flows, plus untested `file-browser-api.service`), `panel-resize.directive.ts` (101), importer-modal internals (`import-advanced` 279, `source-specs-picker` 259, `source-picker` 235, `import-config` 205), media-crop overlays (~530 across three files), settings sub-panels (`import-defaults-settings` 320, `auto-find-settings` 219).

<!-- item-sep -->

- **Frontend test-infra cleanups** — (a) `pretest:ci` runs `ng-openapi-gen` on every test invocation even when `openapi.json` is unchanged; cache on a hash of the spec. (b) 58 spec files hand-copy `provideHttpClient() + provideHttpClientTesting()` and ~40 redeclare mock services; add a shared providers/mocks helper next to the existing `app/testing/zoneless-testbed.ts`. (c) Four components carry both a classic and a `.zoneless.spec.ts`; prune the classic twins as the zoneless migration (`docs/plans/zoneless-migration.md`) completes.
