# Testing

How the test suite is organised and gated, and how to write tests that stay isolated and deterministic. [`CLAUDE.md`](../CLAUDE.md) carries the rules every session needs (the commit-before-testing workflow, how to read a verdict, and a summary of the test-writing rules below); this file is the reference behind them. [`docs/SETUP.md`](SETUP.md) covers installing the dependencies the suite needs.

## Contents

- [What `run-tests.sh` gates](#what-run-testssh-gates)
- [Test Groups](#test-groups)
- [Test Markers](#test-markers)
- [Test Isolation](#test-isolation)
- [Avoiding Flaky Tests](#avoiding-flaky-tests)

## What `run-tests.sh` gates

No CI runs tests — the repo's two workflows only publish (slide decks, and the State of the App links) — so a **full** `./run-tests.sh` is the only gate, and it still runs every check. **This list is derived from `run-tests.sh`; when you add or remove a gate there, update it here in the same commit.**

The run is staged so the box is never idle and a failure surfaces as early as its cost allows. The stale-tree check runs alone; the cheap gates run **concurrently**, every one to completion, and their failures are reported together; pyright and the frontend build run as **gating lanes** that must be green before pytest starts, and the first to fail stops the run with a `TESTS BLOCKED: <gate>` banner and its log; pip-audit, the vulture whitelist check, `npm audit` and the Vitest suite are **free lanes** that start beside the gating ones and are collected after pytest; pytest then runs with the box to itself. A final `RUN PASSED` / `RUN FAILED: <gates>` banner closes the run.

Measured on a warm 4-vCPU container with 17k tests (2026-10). The serial chain this replaced took 791s: 37s of cheap gates one after another, 27s of frontend build, 239s of pyright, and 488s of pytest with the four lanes started beside its workers (411s with the box to itself; the lanes cost it 77s, mostly by stretching its startup from 25s to 73s). Staged, the first full run of a session costs about pyright plus pytest alone (the cheap gates and the lanes hide inside pyright's window); a re-run after a small fix costs about pytest alone, because pyright re-checks only the changed files and their importers (~3s for one test file against 239s); a full run that follows a group run on the same tree skips that group's tests; and a full run on an unchanged tree finishes in the time it takes pytest to collect (every test is already recorded). Absolute times vary more with the machine than with the tree — the same cloud container read cold files at ~20 MB/s late in a session, which made pyright's first check several times slower than the 239s above — so treat the breakdown, not the total, as the number to remember.

Wrapping everything: a wall-clock cap (`VTSEARCH_TEST_TIMEOUT`, default **1800s = 30 min**, `0` opts out for a deliberately long run) and `.claude/hooks/ensure-test-deps.sh` (minutes on a cold container, near-instant after). The hook installs with `python -m pip`, so the stack lands in whichever interpreter `python` resolves to, never in a bare `pip`'s, which a cloud image has put on a different version. It skips on its marker only while `flask` and `vulture` still import (#4662).

### Two records a run leaves for the next one

Both let a later run skip work an earlier run on the *same inputs* already did, and both are exact rather than heuristic: neither guesses which tests or files a change can reach; each only declines to repeat a check whose inputs are byte-identical. `VTSEARCH_FRESH=1` ignores both (and still writes them), which is what the release gate uses ([`docs/RELEASE.md`](RELEASE.md) step 8).

- **Incremental pyright** ([`scripts/pyright-incremental.py`](../scripts/pyright-incremental.py)). After a green check it records, in the gitignored `.gates-cache/pyright-baseline.json`, the blob hash of every `.py`/`.pyi` the repo tracks or carries untracked, the two config files, an environment key (the pyright pin, the interpreter, every installed distribution) and each file's import statements. The next run re-checks only the files that changed plus everything that transitively imports one. A file's diagnostics are a function of its text, its transitive imports, the installed packages and the config, and of nothing else, so an unchanged file whose imports did not change is still green — the same invalidation pyright's language server performs on every edit. The whole tree is checked when there is no baseline (a fresh clone), when the environment key, `pyrightconfig.json` or `pyproject.toml` changed, when the scoped set is over 60% of the checkable files, or under `VTSEARCH_FRESH=1`; a red run never writes the baseline, so it always describes a tree the whole of which passed. Resolution is conservative (every candidate file for an import becomes an edge, and the closure is taken over the old and the new import graph), and any failure to read git, the config or the baseline falls back to a full check, never to a smaller one. Measured: 239s full, ~3s for one changed test file. [`tests_lib/meta/test_pyright_incremental.py`](../tests_lib/meta/test_pyright_incremental.py) pins the resolver and the closure rules against fixture repos and the real tree.
- **The pass cache** ([`tests_shared/pass_cache.py`](../tests_shared/pass_cache.py), wired through both conftests). Every test whose setup, call and teardown passed is recorded in `.pytest_cache/v/vtsearch/pass-cache` under a key over the committed tree, every uncommitted and untracked change, the interpreter, the installed packages and the `VTSEARCH_*` environment; a later run under the same key deselects those tests and says so in pytest's summary (`Pass cache: N tests not re-run (passed on this identical tree at …); K newly recorded.`). When nothing is left to run the headline reads `ALL N TESTS ALREADY PASSED ON THIS TREE (nothing re-run)` and the run is green. One changed byte in one tracked or untracked file changes the key and reuses nothing; the four most recent keys are kept. This is what makes the recommended loop — a group run, then the full run — stop repeating the group: `detectors` alone is 55% of all test time and `meta` another 20%. Skips, xfails and xpasses are never recorded, and a red run still records the tests that did pass. [`tests_lib/meta/test_pass_cache.py`](../tests_lib/meta/test_pass_cache.py) pins the key, the recording rule and the end-to-end behaviour on fixture projects.

### The stages

**Stage 1 — the stale-tree check, alone (~0.2s; every invocation except `vtscore-clean`):**

| Gate | Command it runs | Notes |
|------|-----------------|-------|
| Stale tree | `scripts/check-phantom-base.py` | **Runs first, by itself.** Refuses a branch whose tree matches an *ancestor* of `origin/dev` rather than `origin/dev` — the signature of a stale checkout committed onto a fresh HEAD. Two signals: paths deleted that the branch never created, and a run of two or more consecutive `dev` commits the branch keeps *nothing* of (which is what catches a clobber that deletes no file at all, because the reverted hunks sit inside files that survive). Compares against the working tree, so it fires before the clobber is committed. A deliberate deletion (a shipped plan file) passes with `VTSEARCH_ALLOW_DELETIONS=1`; a deliberate revert of consecutive merges with `VTSEARCH_ALLOW_REVERTS=1`. |

**Stage 2 — cheap gates, concurrently, all to completion (~15s warm, the length of codespell; every invocation except `vtscore-clean`, and a `slides` run keeps only the rows that can see a deck).** Each is a function in `run-tests.sh` so its fix-it hint prints with its failure, and the per-gate report shows each one's last line of output.

| Gate | Command it runs | Notes |
|------|-----------------|-------|
| Lint | `ruff check .` | |
| Format | `ruff format --check .` | Fix with `ruff format .`. |
| Spelling | `codespell --toml pyproject.toml` | The slowest of the stage (~9s warm, ~18s on a cold page cache). |
| Documentation | `scripts/check-docs.py` | Pure invariants over every tracked markdown file: relative links, `#anchors` (GitHub slug rules), backticked repo paths, absolute-path leaks, `docs/plans/*.md` citations **anywhere in the tree**, broken code fences, and `docs/experiments/` study directories named `YYYY-MM-DD-<slug>`. Nothing to re-pin; fix the doc, or add an allowlist entry with a reason. |
| Dependencies | `python -m deptry .` | |
| OpenAPI snapshot drift | `scripts/dump_openapi.py` diffed against `frontend/openapi.json` | The generated TS client is built from this snapshot. Regenerate with `npm run regenerate-openapi-snapshot` and commit the result. |
| Doc inventories | `scripts/gen-docs-inventories.py --check` | Regenerate with `python scripts/gen-docs-inventories.py` and commit the result. |
| Dockerfiles | `scripts/check-dockerfiles.py` | A RUN that executes Python (any `python`, `python3`, `python3.12` command; a package-manager install and `-m venv` are exempt), or pip-installs the local project (directly or through a requirements file that forwards to `-e .`), must come after both `vtsearch/` and `vtscore/` are copied in. An editable install maps only the packages present when it runs (#4894). |
| User-docs screenshot wiring | `scripts/screenshots/wiring-check.py` | Browser-free; the pixel-diff (`check.sh`) stays a manual chore. Also what makes the reshoot queue un-rottable. |
| vtscore package docs | `scripts/check-vtscore-docs.py` | |
| Extension docs | `scripts/check-extension-docs.py` | Holds `docs/EXTENDING-*.md` and `vtscore/docs/extending/` to the plugin ABCs they both document: every member named in a contract table must exist, and neither set may present a public wrapper as the override point when the class defines an `_impl` hook behind it. AST sweep, imports nothing. Register a new contract section in `SECTIONS` — an unregistered one fails the gate rather than going unchecked. |
| Calibration script index | `scripts/check-calibration-index.py` | Every `.py`/`.sh` in `scripts/experiments/calibration/` is filed under exactly one study (or the shared layer) in that directory's `README.md`, and every file the index names exists. That directory is flat by decision (#3409), so the table *is* the navigation; unchecked, it decays back into 120 unclassified files. |
| Slide decks | `slides/build.py --check` | Preflights every deck manifest: fragments exist, figures resolve, no headline breaks more than once (the two-line ceiling in `slides/STYLE.md`; *where* a two-line headline breaks needs a browser and lives in `slides/balance-titles.mjs`). Marp only warns on a missing figure and exits 0, so a rotted deck is otherwise silent. |
| Eval/app sync | `scripts/check-eval-app-sync.py` | Digests both sides of every mirror, so `harness-changed` is as loud as `app-changed`. Re-pin with `--update` **after** reconciling the two. |

**Stage 3 — heavy gates, concurrently. Gating lanes must be green before pytest starts, and the first to fail stops the run; free lanes are collected after pytest:**

| Gate | Command it runs | Lane | When | Notes |
|------|-----------------|------|------|-------|
| Types | `scripts/pyright-incremental.py --threads N` (pyright pinned via `PYRIGHT_PYTHON_FORCE_VERSION`; scope is `pyrightconfig.json`) | gating | Full run, or a group run with `VTSEARCH_FULL_GATES=1`; never a markdown-only or frontend-only run | Incremental (see above). Two threads when at least 12 GB is free and one otherwise: on a 4-vCPU box a full check took ~283s on one thread (peak ~5.4 GB RSS), ~239s on two (~9.6 GB), and ~340s on four — each thread holds its own copy of the program, so more threads re-parse more. Those numbers predate the local `transformers` stub, [`typings/transformers/__init__.pyi`](../typings/transformers/__init__.pyi) (#5077), which pyright reads in place of the installed package: binding the real one's wildcard cascade through ~530 model packages cost every thread ~2,600 files. Measured on one container, the stub took a two-thread check from 191s to 104s (461 to 223 CPU-seconds) and a one-thread check from 207s to 171s. Every name in it is `Any`, so a new `from transformers import X` fails pyright until `X` is declared there; [`tests_lib/meta/test_transformers_stub.py`](../tests_lib/meta/test_transformers_stub.py) checks each declared name exists in the installed package. |
| Frontend build | `cd frontend && npm run build:prod` | gating | When the branch touches `frontend/` (full run and `core`); always for `frontend`; also, as a convenience rather than a gate, when `static/` holds no bundle | Any `▲ [WARNING]` line is a hard failure. Gating because some tests serve the bundle it writes to `static/`, so building while pytest reads would race. |
| Known CVEs | `pip-audit` | free | As Types | Audits the resolved venv, not the requirements files. `PIP_AUDIT_IGNORE` in the script lists advisories with no upstream fix, and ones in the Ubuntu base image's apt packages that the requirements cannot upgrade; re-audit and remove an entry once a patched release exists. |
| Vulture whitelist | `scripts/vulture-audit.py --check-whitelist` | free | As Types | Whitelist hygiene, **not** the dead-code audit: fails when an entry in `.vulture-whitelist.py` suppresses no finding. The findings themselves stay a manual pre-release chore, because a hit on a public `vtscore` name is not evidence of anything. Whole-repo scan, ~14s. |
| Frontend audit | `scripts/npm-audit-gate.py` (wraps `npm audit --json` in `frontend/`) | free | As the build, minus the bundle clause | Whole tree, dev deps included. Fails on any advisory not in the script's `WAIVERS`, which holds only advisories with no patched release anywhere, each pinned to its vulnerable range so it lapses (and fails the gate) once upstream ships a fix; a waiver that suppresses nothing fails as stale. Prefer a lockfile refresh or an `overrides` pin whenever a fixed version exists. |
| Frontend unit tests | `cd frontend && npm run test:ci` | free, started the moment the build lane is green (both regenerate the API client, and must not do so at once) | Full run when the branch touches `frontend/`, and `frontend` — deliberately **not** `core` | Headless Vitest; ~67s wall and ~145 CPU-seconds alone, which is why it no longer starts beside pytest's workers. |

**Stage 4 — pytest, alone:** `pytest tests/ tests_lib/ -n auto --dist loadgroup`, every run except a `frontend`-only or `slides` run. A markdown-only run narrows it to the paths in `tests_shared/markdown_surface.py`, a frontend-only run to the paths in `tests_shared/frontend_surface.py`, and the pass cache deselects whatever already passed on this tree. The free lanes are then collected and every failure — pytest's and theirs — is reported together.

**Frontend gates and the diff.** `scripts/check-frontend-gate.py` decides, before stage 1, whether the build, `npm audit` and Vitest run on a full or `core` run: **run** when the branch changes a `frontend/` path and `frontend/node_modules` is present; **skip** when the branch changes no `frontend/` path — whether or not `node_modules` is present, because the gates can only judge the frontend and a backend-only branch's frontend is `origin/dev`'s; **block** when `node_modules` is absent and either the branch changes a frontend path or there is no `origin/dev` to diff against (#4019). A skip is named in the verdict (`RUN PASSED (all gates green; frontend build, npm audit and Vitest skipped: …)`) and is still the full gate: those three had nothing to judge. The explicit `frontend` group asks for the gates by name, so there only `node_modules` decides. `tests_lib/meta/test_frontend_gate.py` pins every case.

**Group runs skip the whole-repo gates in stage 3** (pyright, pip-audit, the vulture whitelist check) so the edit/test loop stays in the seconds — the skip is announced in the output and named in the verdict, and `VTSEARCH_FULL_GATES=1` forces those three back on. The frontend gates follow the group and the diff, not that variable. Stages 1 and 2 run on every invocation. This is a deliberate trade: the fast inner loop may miss a type error or CVE, which is why **a full `./run-tests.sh` remains mandatory before pushing** — with two exceptions, `slides` and `docs`, whose narrower scope is a proof rather than a gamble (see the Test Groups table). Both block themselves the moment the diff stops matching.

**The `docs` and frontend-only narrowings do not wait to be asked:** a bare `./run-tests.sh` on a markdown-only branch narrows itself to the markdown surface, and on a branch that changes nothing outside `frontend/` to the frontend surface; each prints a banner listing the changed files and every lane it skipped, and `VTSEARCH_FULL_GATES=1` turns both off (see the `docs` note under Test Groups).

## Test Groups

Tests are grouped by folder under `tests/` and `tests_lib/`. Each folder is a pytest marker; `./run-tests.sh <group>` runs all tests in `tests[_lib]/<group>/`. New tests inherit their group from the folder they're added to. Not every group lives in both trees — the "Tier" column below says which one has the folder; missing an app-tier folder for `projection` / `downloads` / `meta` / `gpu` (or a lib-tier folder for `api` / `converters`) is by design, not a gap.

| Group | Tier | Description |
|-------|------|-------------|
| `core` | both | Basic app functionality (audio, medias, votes, settings, frontend, torch config) |
| `api` | app only | API contracts, error handling, security, dashboard, embed |
| `sorting` | both | Sort algorithms, diversity, safe thresholds, enriched text sort |
| `datasets` | both | Dataset loading, splitting, dedup, parallel/chunked/thin loading, multi-dataset context |
| `io` | both | Importers, exporters, label I/O, settings I/O, sync sources, PDF/NPZ import |
| `detectors` | both | Detectors, embedders, clippers, eval, processors, training |
| `downloads` | lib only | Demo dataset downloads, one file per source, plus the shared download-and-extract helper and the demo-cache script |
| `integration` | both | End-to-end workflows, thread safety, async jobs |
| `cli` | both | CLI autodetect, load sort window, progress bars |
| `converters` | app only | Media converters (document, video, image) |
| `projection` | lib only | VTSBrowse UMAP projection + hex-tile pyramid |
| `meta` | lib only | Repo/tooling meta-tests: packaging and requirements files, Dockerfiles, docs, the frontend's SCSS text, `scripts/`, `.claude/hooks/`, the `run-tests.sh` gates themselves, and the test harness in `tests_shared/`. Nothing here tests shipped `vtsearch`/`vtscore` behaviour — see the note below the table. |
| `frontend` | n/a | Frontend-only gate: Angular `build:prod` + `npm audit` + the headless Vitest unit suite. No Python tests; `./run-tests.sh frontend` skips pytest and closes with `RUN PASSED (frontend-only; no Python tests in the 'frontend' group)`, naming the whole-repo gates it skipped beneath it: not a pre-push gate. The same three gates run inside a full `./run-tests.sh` when the branch touches `frontend/` (see "Frontend gates and the diff" above), and a full run on a branch that touches *nothing else* narrows itself to the frontend-only fast path below, whose `RUN PASSED (frontend-only; this is the full gate …)` verdict *is* a pre-push gate. |
| `slides` | n/a | Slide-deck gate — see the note below the table. |
| `docs` | n/a | Markdown-only gate — see the note below the table. Auto-engages on a bare run. The frontend-only fast path (no group name; see the note below the `docs` one) is its mirror. |
| `gpu` | lib only | CUDA-only tests (excluded by default) |

The `meta` group is the one whose subject is *this repository* rather than the product: a reader asking "what does VTSearch do?" would never open a file in it, and a reader asking "how is this repo built and checked?" would. That is the whole membership test — packaging and requirements files, Dockerfiles, docs and their anchors, SCSS text, `scripts/` (including the experiment tooling under `scripts/experiments/`), `.claude/hooks/`, the `run-tests.sh` gates' own self-tests, and the shared test harness. It exists because those tests had silted into `core` (#3421). It lives under `tests_lib/` because every member satisfies the library tier's contract trivially (it imports no product code at all), which keeps `./run-tests.sh vtscore-clean` proving it. **Do not put a test here just because it is slow, awkward, or hard to file** — that is how `core` became a junk drawer in the first place.

The `slides` group runs only the stage-1 and stage-2 gates that can see a deck (stale tree, `ruff check`/`ruff format --check`, `codespell`, `check-docs.py`, `slides/build.py --check`); no Python tests, no whole-repo gates. **The one group that also gates a push**, because a change confined to `slides/` cannot reach the rest of the repo: nothing imports `slides/build.py`, `pyrightconfig.json` excludes it, and no test in either tree reads a deck. Self-policing — the group refuses to run when the branch changes anything outside `slides/`, so the exemption can't be taken by mistake. Also runs as part of the full `./run-tests.sh`.

The `docs` group is the same exemption one step wider, and it is the only one that **engages by itself**: a bare `./run-tests.sh` narrows to it when the branch changes nothing but tracked markdown. It keeps the *whole* of stages 1 and 2 — those gates are precisely the ones that read markdown (`check-docs.py`, `codespell`, the doc-inventory and screenshot-wiring snapshots, the deck preflight) — and skips pyright, pip-audit, the vulture whitelist check, the frontend build/audit/unit suite, and all of pytest except the tests that can open a doc. About 40s against a full run (measured warm: ~15s of gates, 23s of pytest).

That last clause is the load-bearing one, and it is checked rather than asserted. The tests that read a repo doc are named in [`tests_shared/markdown_surface.py`](../tests_shared/markdown_surface.py) — `tests_lib/meta/` plus `tests/core/test_achievements.py`, whose Readme Reader achievement pins a phrase inside `README.md`, `docs/user/USER_GUIDE.md`, `docs/CLI.md` and `docs/API.md` and reaches them through `vtsearch/achievements_catalog.py` rather than naming a doc itself. `tests_lib/meta/test_markdown_surface.py` fails when a test outside that list learns to read a doc, in either shape: naming a tracked `.md` path directly, or importing a registered module that carries one. **If you add a test that reads a repo doc, put it in `tests_lib/meta/` or add it to that list** — otherwise markdown changes silently stop running it.

The **frontend-only fast path** is the `docs` narrowing's mirror, and like it engages by itself: a bare `./run-tests.sh` on a branch that changes nothing outside `frontend/` keeps every cheap gate, runs the production build, `npm audit` and the Vitest suite in full, skips pyright (`pyrightconfig.json` excludes the frontend), pip-audit and the vulture whitelist check, and narrows pytest to the tests that can observe the frontend. Those are named in [`tests_shared/frontend_surface.py`](../tests_shared/frontend_surface.py) — `tests_lib/meta/` plus the handful of files that read a file under `frontend/`, serve the built bundle out of `static/`, or ask the Flask client for a route the SPA catch-all in `vtsearch/routes/main.py` answers with that bundle (`/healthz` and `/readyz` are their own handlers and do not count). `tests_lib/meta/test_frontend_surface.py` fails when a test outside that list learns to reach the frontend in any of those three shapes: a code string naming a `frontend/` path or a bundle artefact, the `angular_bundle` fixture, or a client call on a non-API route. **If you add such a test, put it in `tests_lib/meta/` or add its path to that list.** There is no group name for this path: the explicit `frontend` group stays the inner-loop gate for a mixed branch (build, audit and Vitest, no pytest). `VTSEARCH_FULL_GATES=1` turns the narrowing off.

**Recommended workflow**: Run `./run-tests.sh <group>` for the area you changed, then `./run-tests.sh` for the full suite. The full run does not repeat what the group run just passed: the pass cache (see "What `run-tests.sh` gates") deselects every test that passed on this identical working tree, so the two together cost about what the full run alone used to.

`tests/` is the app-tier suite (uses `client`, `vtsearch.routes`, `vtsearch.settings`, `vtsearch.auth`, etc.). `tests_lib/` mirrors the same layout but every file must be import-clean of Flask **and of `vtsearch` entirely** — not just `vtsearch.routes` / `settings` / `auth` / `shim` / `autorun_processors` / `settings_io`, but `vtsearch.state` and every other app-tier module too. Two gates enforce that, and they see different things: `./run-tests.sh vtscore-clean` bans Flask at import time (it cannot ban `vtsearch`, which ships in the same distribution and must stay importable), while `tests_lib/meta/test_library_layering.py` statically scans `tests_lib/` for `vtsearch` imports **and for `mock.patch("vtsearch…")` targets**, which are imports the AST cannot otherwise see. Add a new test to `tests_lib/` if it doesn't touch any app-tier module; otherwise add it to `tests/`.

**When the library has no seam, add one — don't reach across the tier.** The library-tier equivalent of the `vtsearch.state` proxies is `get_active_context().medias` and friends; the library-tier equivalent of patching `vtsearch.logging_config.install_transformers_logging_bridge` is patching `vtscore.embedding.loader._install_transformers_logging_bridge`, which exists precisely so `tests_lib/` need not. Note that the proxies' laziness is load-bearing where it is used: `reset_shared_state()` re-registers the default context *before* refilling medias, so a mapping resolved before the call would refill the previous test's context. That is why it resolves the mapping itself rather than accepting one.

## Test Markers

The default filter lives in `pyproject.toml`'s `addopts`: `-m 'not gpu and not slow' --timeout=300 --timeout-method=thread`.

- **Default** (`./run-tests.sh` with no group, or a bare `pytest`): fast CPU tests only. Excludes `gpu` and `slow`.
- **`slow`**: 5 tests, in **two** trees — three CLI subprocess tests that spawn `python app.py` (`--autodetect --dry-run` and two `--pipeline --progress-format json` runs, in `tests/cli/test_cli_main_subprocess.py`) and two real-`toponymy` fit tests (`tests_lib/projection/test_toponymy_smoke.py`, module-level `pytestmark`, ~1 min each; `importorskip`ped when toponymy isn't installed). Run with `python -m pytest tests/ tests_lib/ -m slow` — passing only `tests/` silently misses half of them.
- **`gpu`**: CUDA-only tests (`tests_lib/gpu/test_gpu.py`). Run with `-m gpu`.
- **All tests**: `-m ''`.
- **Per-test timeout**: 300s, thread-based (signal-based interruption doesn't work on xdist workers). One hung test fails by name instead of stalling the run; the wall-clock cap in `run-tests.sh` is the backstop for a worker that dies outright and can no longer fire its own timeout. Under xdist the timeout ends its worker, so the test fails as `worker 'gwN' crashed while running '<test>'`, and the log carries a `+++ Timeout +++` block above it with every thread's stack. Both come from [`tests_shared/xdist_plumbing.py`](../tests_shared/xdist_plumbing.py) (#4925). Without it the stacks went to the worker's stdout, which is `/dev/null`, and pytest-xdist 3.8.0's `--dist loadgroup` hung on the dead worker until the 30-minute cap. A test that needs more than a minute or two on a GRID node is one to split: the Xeon Silver 4114 nodes run the suite ~1.5x slower than the EPYCs.

**Gotcha: naming a group re-opens `slow` and `gpu`.** `./run-tests.sh <group>` passes `-m "<group>"` on the command line, and a command-line `-m` *replaces* the one in `addopts` rather than combining with it. So `./run-tests.sh cli` does run the slow subprocess tests, and `./run-tests.sh projection` does run the toponymy smoke tests — a group run is slower than the same tests in a default run. That is also why `./run-tests.sh gpu` works at all. To get the default exclusions back inside a group, spell the filter out after `--` (a later `-m` wins): `./run-tests.sh cli -- -m 'cli and not slow'`.

## Test Isolation

**The session's data dir comes first, before any fixture.** Both conftests call `isolate_data_dir()` from [`tests_shared/data_dir.py`](../tests_shared/data_dir.py) before anything imports `vtscore.config`. So `config.DATA_DIR` is a fresh temp dir for the whole run, shared by every xdist worker and every subprocess a test starts, and removed when the run ends. Whatever `VTSEARCH_DATA_DIR` the shell exported is ignored. On the HLTCOE GRID that is the live app's data (#4706); on a laptop it is the checkout's own `data/`. The model cache is the one exception. If the dir the app would read models from exists, `VTSEARCH_MODELS_DIR` keeps naming it, so the `gpu` group does not download every model again; nothing is ever created in the inherited dir. The dir is shared across the run, so a test that needs an empty one still takes its own `tmp_path`.

All mutable global state is reset automatically before each test via these autouse fixtures:

1. **`reset_state`** — Clears all dataset contexts and creates a fresh `_test_default` context with the pre-generated test medias replayed into it. Also clears:
   - `autorun_extractors`, `autorun_localizers` (global state)
   - Progress cache and progress trackers
   - Login provider and dataset/model registries

2. **`isolated_settings`** — Redirects both settings tiers (server `settings.json` and per-user `user_settings.json`) to a per-test temp dir so writes never touch `data/`. Yields a path-like whose `read_text()` returns the merged JSON, for tests that need to inspect it.

3. **`_isolated_example_media_dir`** (from `tests_shared`, so `tests_lib/` gets it too) — Points single-user `example_media_dir()` at `tmp_path / "example_media"`, so exemplars that tests upload, copy or write never land in the checkout's `data/example_media/`, where the app would list them as server example media (#4271). Multi-user mode still runs the real resolver. Read the directory back through `example_media_dir()`; a test needs no redirect of its own.

**When writing new tests:**
- Do NOT add per-file or per-class autouse fixtures to clear autorun state, reset settings, or reset votes — `conftest.py` handles all of this automatically.
- Do NOT add inline `.pop()` or `.clear()` cleanup at the end of tests — the conftest fixtures run before each test regardless of whether the previous test passed or failed.
- If a test needs to temporarily empty `medias`, use the save/restore pattern with try/finally (since `medias` is intentionally NOT reset between tests to avoid expensive re-generation):
  ```python
  saved = dict(medias)
  medias.clear()
  try:
      # ... test logic ...
  finally:
      medias.update(saved)
  ```
- If a test needs to read the settings file path (e.g. to verify persistence), use `isolated_settings` as a parameter: `def test_foo(self, isolated_settings): ...`

`tests_lib/conftest.py` provides app-free, settings-free shared fixtures: `reset_contexts` (autouse, resets dataset/detector contexts, progress trackers, async jobs, label-sync, registries), `_allow_test_tmp_paths` (autouse, widens path validation for tmp dirs), `_isolated_example_media_dir` (autouse, points `example_media_dir()` at `tmp_path`), `_stub_embedding_models` (session, stubs every embedder). It also installs a library-only `CoreConfig.from_settings()` builder so library code that calls it works without the app shim.

**The machinery behind those fixtures is single-sourced in `tests_shared/`, not duplicated.** Both conftests import the fake embedders, the tmp-path widener, the example-media redirect, the embedder-stub fixture factory, `reset_shared_state()` (the whole library-tier half of the per-test reset), the group-marker hook and the end-of-run summary printer from that package; each conftest keeps only its tier-specific extras (the app tier's `client`, `isolated_settings` and autorun-processor reset; the library tier's Flask blocker, native-thread caps and `CoreConfig` builder). Add a new library-tier global's reset to `tests_shared/state_reset.py` so *both* suites get it — copying it into one conftest is how the two drifted before (#3424). `tests_shared/` must never import `flask`, `werkzeug`, `flask_smorest`, or any `vtsearch.*` module, because `tests_lib/` imports it under the Flask blocker; pass app-tier objects (the `medias` map) in as arguments instead. `tests_lib/meta/test_test_tier_helpers.py` gates both halves: neither conftest may re-define a name `tests_shared` owns, and nothing under `tests_shared/` may import the app tier. `tests/helpers.py` and `tests_lib/helpers.py` (and likewise `tests/fixtures/medias.py` / `tests_lib/fixtures/medias.py`) are intentional duplicates so each tier is self-contained. **Import them tier-qualified** — `from tests.helpers import ...` inside `tests/`, `from tests_lib.helpers import ...` inside `tests_lib/` — never as a bare `from helpers import ...`. There is deliberately no `pythonpath` entry in `pyproject.toml`: with both directories on `sys.path`, the bare name resolved to `tests/helpers.py` for *both* trees. `tests_lib/meta/test_test_tier_helpers.py` gates both halves of this: the duplicated files must stay byte-identical, and no bare `helpers` import may reappear.

## Avoiding Flaky Tests

When writing new tests, avoid these three common sources of flakiness.

### 1. Always seed random number generators

Never call `np.random.randn()`, `np.random.rand()`, `torch.randn()`, or similar without a fixed seed. Random embeddings feed into neural net training and sorting, where different values cause non-deterministic convergence — making assertions pass or fail depending on the random draw.

**Do this:**
```python
rng = np.random.default_rng(42)
fake_embeddings = rng.standard_normal((n, dim)).astype(np.float32)
```

**Not this:**
```python
fake_embeddings = np.random.randn(n, dim).astype(np.float32)  # FLAKY; unseeded
```

### 2. Never use `time.sleep()` for thread synchronization

`time.sleep(0.2)` to "wait for a thread to start" is unreliable on loaded machines. Use `threading.Event` for deterministic synchronization, and set generous polling timeouts.

**Do this:**
```python
started = threading.Event()
def target():
    started.set()
    # ... work ...
thread = threading.Thread(target=target)
thread.start()
started.wait(timeout=5)
```

**Not this:**
```python
thread.start()
time.sleep(0.2)  # FLAKY; may not be enough on a loaded machine
```

### 3. Never use bounded loops to simulate "cancellable" or "interruptible" work

A `for i in range(100): sleep(0.05)` loop finishes in 5 seconds — but on a loaded machine the code that's supposed to interrupt it (e.g. setting a cancel flag) can take longer than 5 seconds to run. If the loop completes before the interrupt arrives, the test follows the wrong code path and fails.

**Do this:**
```python
def slow_load():
    started.set()
    while True:                            # exits ONLY via CancelledError
        tracker.check_cancelled()
        time.sleep(0.05)
```

**Not this:**
```python
def slow_load():
    started.set()
    for i in range(100):                   # FLAKY; can finish before cancel arrives
        tracker.check_cancelled()
        time.sleep(0.05)
```
