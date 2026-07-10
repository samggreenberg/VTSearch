# Test Suite Improvements (speed + coverage)

Open work from a full evaluation of the test suite (2026-07). Baseline measurements that motivate the items below, taken on a 4-core cloud container, cold caches, `pytest tests/ tests_lib/ -q -n auto` with coverage enabled:

- 5,614 Python tests, 197s wall. The worst individual tests are numba-JIT-bound UMAP fits (~50s each) and real-time backoff/ticker sleeps, not training or I/O.
- Backend line coverage (vtsearch + vtscore combined): **83%**. Cold spots are concentrated in embedder wrappers (stubbed session-wide), `vtsearch/cli_main.py` (6% — only covered by the excluded `slow` subprocess tests), and a handful of network-facing/plugin modules.
- Frontend: 115 spec files; ~68% of components and ~76% of services have specs; **0 of 4 HTTP interceptors** are tested.

Each item below is independent; pick any and delete it when it ships.

## Speed

<!-- item-sep -->

- **Fix the static-bundle ordering race under bare `pytest -n auto`** — With no pre-built `static/` bundle, 9 tests in `tests/api/test_dashboard.py` fail: the session-autouse fixture in `tests/core/test_frontend.py:39` runs `npm run build:prod` (~28s) only on whichever xdist worker collects that file, while dashboard tests on other workers read `static/` before it exists. (Under `./run-tests.sh` the bundle is built beforehand, so the gate passes — but `pytest tests/ tests_lib/ -m 'not gpu'` is a documented invocation in CLAUDE.md and currently red on a fresh clone.) Fix: hoist ensure-frontend-built into a shared session fixture guarded by a cross-worker file lock (build once, others wait), and make `test_dashboard.py` depend on it; that also prevents N concurrent 16s npm builds when several workers hit `test_frontend.py`.

<!-- item-sep -->

- **Stop paying numba JIT compilation for UMAP/librosa tests** — The two slowest tests, `tests_lib/projection/test_gpu_backends.py::test_umap_fit_transform_falls_back_when_cuml_fit_raises` (53s) and `tests_lib/projection/test_umap_projection.py::test_umap_fit_shape_and_metadata` (49s), run real `umap-learn` fits on tiny matrices (60×16); nearly all the time is numba JIT compiling UMAP's kernels, paid once per xdist worker per run. Options, in order of preference: (a) set `NUMBA_CACHE_DIR` to a persistent path in the test conftests so compiled kernels cache across runs and workers; (b) `NUMBA_DISABLE_JIT=1` for the test env (verify the librosa mel-spectrogram test, `tests/converters/test_audio2image_and_image2text.py::test_convert_mel_spectrogram` at 9.8s, stays tolerable interpreted); (c) mark the real-fit tests `slow` and keep a mocked-UMAP fast path. Expected saving: ~100s of worker CPU per cold run.

<!-- item-sep -->

- **Patch `time.sleep` in the HF backoff-failure test** — `tests_lib/cli/test_preload_progress.py::TestLoadPretrainedLocalFirst::test_network_fallback_error_propagates` (14s) exercises the retry loop in `vtscore/media/embedder.py` (`_HF_RETRY_BACKOFF_BASE` sleeps of 2+4+8s) with real sleeps. The sibling test at `tests_lib/cli/test_preload_progress.py:804` already does `@patch("vtscore.media.embedder.time.sleep")`; apply the same to this one. Saves 14s.

<!-- item-sep -->

- **Parameterize the `timed_progress` tick interval** — `timed_progress` in `vtscore/media/embedder.py` hardcodes `stop.wait(timeout=1.0)`, forcing 9 tests in `tests_lib/cli/test_preload_progress.py` (lines ~1037–1180) to sleep 1.5–2.5s each (~14.5s of pure sleeping). Add a `tick_interval: float = 1.0` parameter and pass ~0.05 in tests; replace the unconditional "verify ticker stopped" post-block sleeps with a wait of a couple of tick intervals.

<!-- item-sep -->

- **Stub or cache waveform thumbnails in the media fixtures** — `tests/fixtures/medias.py:89` (and the `tests_lib` twin) call `generate_waveform_thumbnail()` for each of the 20 generated WAVs at session start, which imports librosa/numba and decodes every file — in every xdist worker, every run. Either cache `thumbnail_bytes` in the existing per-worker `media_embedding_cache*.npz`, or stub `generate_waveform_thumbnail` the way `embed_audio_file` is stubbed. Also: the NPZ embedding cache stores conftest-faked vectors (saves nothing) and writes into the repo's `data/` dir — consider dropping it or pointing it at a temp dir while in there.

<!-- item-sep -->

- **Replace fixed race-window sleeps with "waiter parked" event hooks** — Negative-assertion sleeps that always pay full price and are inherently flaky: `tests/datasets/test_parallel_loading.py:669,729,733` (0.3–0.5s around the download gate), `tests/io/test_sync_sources.py:1819` (0.3s cancelled-timer probe), `tests_lib/detectors/test_new_embedders.py:1322` (0.1s lock race). Add a test-visible `threading.Event` set when a waiter blocks on the gate/timer so the tests become deterministic and near-instant.

<!-- item-sep -->

- **Trim the voting-iteration eval sweeps** — `tests_lib/detectors/test_eval_voting_iterations.py` makes 27 calls to `simulate_voting_iterations` / `run_voting_iterations_eval`, each training an MLP per voting step and calibrating per split; several tests take 1.3–5.5s. Where the assertion doesn't need the full sweep (shape/plumbing tests like `test_full_cross_product_shape`, `test_multiple_seeds`), pass minimal step counts / `calibrate_count` and smaller media pools. Same review applies to `tests/sorting/test_safe_thresholds.py` (4 tests at 1.8–3.6s) and the 500-epoch run at `tests/sorting/test_sorting.py:272` (could assert on a smaller epoch bound).

<!-- item-sep -->

- **Slim the combine-datasets pickle round-trips** — `tests/datasets/test_combine_datasets.py` has ~10 tests at ~2s each; `make_dataset_file` (`tests/helpers.py:55`) pickles the full 20-media snapshot including `media_bytes` (~3–4MB per write) even for tests that only assert on metadata/dedup/registry behavior. Use 2–3-media subsets (several tests already do) or strip `media_bytes` where the test doesn't read it back.

<!-- item-sep -->

- **Correct the stale `slow`-marker note in CLAUDE.md** — Only 2 subprocess CLI tests remain (`tests/cli/test_cli_main_subprocess.py`, ~16s each); CLAUDE.md still says "total ~290s". Update when touching CLAUDE.md for other reasons (or as part of the in-process CLI item below, which may remove them entirely).

## Coverage — backend

<!-- item-sep -->

- **In-process tests for `vtsearch/cli_main.py`** — 253 statements at **6% coverage** in the default suite: its only real exercise is the two `slow`-marked subprocess tests, which are excluded by default. Add in-process tests that call the CLI entry points with mocked heavy stages (argument parsing, flag validation and combinations, exporter/importer wiring, error exits). This closes the biggest single route-to-user coverage hole *and* could let the 16s subprocess tests shrink to one smoke test.

<!-- item-sep -->

- **PullWrest source behavior** — `vtscore/datasets/sources/pullwrest.py` (250 lines, 31% coverage) is a network-facing origin source with zero behavioral tests: HTTP download failure, contentID-keyed temp-cache reuse, and `cleanup()` of the owned temp dir (a past temp-dir bug is referenced in a comment in `tests/detectors/test_resolver.py` but has no regression test). Test with mocked HTTP.

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
