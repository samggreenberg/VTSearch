# VTSearch

Trainable media search tool. Searches collections of audio, images, text, video, and documents using a **detector**: a small trained ranker that scores each item by how well it matches. Two ways to search: **train a new detector** (vote good/bad on a handful of items; a small MLP learns to rank the rest) or **use an existing detector** (saved or imported). Trained detectors are reusable across compatible datasets. Text queries (LAION-CLAP, SigLIP, X-CLIP, E5 embeddings) seed either flow or work as a quick stand-alone search. Flask + Angular + PyTorch.

Architecture, state model, plugin systems, auth, and the directory map all live in **`docs/ARCHITECTURE.md`**. This file holds the testing rules and the policy/gotchas that must be in context every turn.

## Ask Questions via `AskUserQuestion`: NOT prose (CRITICAL, READ FIRST)

This is the **#1 rule** in this repo. Read it on every turn. If you only read one section of CLAUDE.md, read this one.

When you have a question for the user (to disambiguate requirements, choose between approaches, confirm scope, or surface a non-obvious tradeoff), **ask it via the `AskUserQuestion` tool**. Do not guess silently. Do not bury the question in prose at the end of a response. A 10-second clarification beats a 10-minute wrong-direction implementation, and a one-click answer beats a typed answer every time.

**Always ask via the `AskUserQuestion` tool when the question fits its shape** (a discrete choice with a small number of options). Do not leave dangling questions like "Want me to go with approach A or approach B?" at the end of a prose response; those are easy to miss and force the user to type out an answer that could have been a single click. The tool also captures the choice cleanly in the transcript.

This applies *especially* to end-of-investigation "what scope should I take next?" prompts: when a research/investigation turn ends by offering Phase 1 / Phase 2 / smaller scope, the scope choice goes through `AskUserQuestion`, **not** into the prose summary. The investigation findings stay in prose; the "what next?" question is a tool call.

Use plain prose questions only when the answer is genuinely open-ended (e.g. "What should this field be named?") and a multiple-choice list would be artificial.

### Trip-wire: scan your turn before sending

Before sending a turn, scan its last paragraph for any of these phrases:

- "Want me to …?"
- "Should I …?"
- "Do you want … or …?"
- "Let me know if …"
- "(a) … and/or (b) …?"
- "Recommend I …?"

If you see one, **stop**: that sentence is an `AskUserQuestion` call you almost emitted as prose. Convert it into the tool call before sending; even if you're confident the user will say yes, even if the options feel obvious, even if you've already invested effort in the prose summary. The cost of the extra tool call is zero; the cost of a missed or typed-out answer is a wasted round-trip.

This rule has **no exceptions for "quick" yes/no follow-ups.** Yes/no offers belong in the tool too (with `["Yes", "No"]` options); they are exactly the case where a one-click reply beats a typed reply. A pure progress update with no question at the end is fine; an update that ends in an offer is not.

## Branch Policy (CRITICAL)

- **Always base work on `dev`.** The `.claude/hooks/session-start.sh` SessionStart hook runs `git fetch origin --prune && git rebase origin/dev` automatically in remote sessions, so a fresh container lands rebased onto `dev`. If the hook reports "skipping" (dirty tree, detached HEAD) or "rebase failed", run the fetch+rebase yourself before making any changes. The harness cuts the working branch off `main` (the GitHub default), so this rebase is required to pick up work already merged to `dev`. The GitHub default stays `main` so new users land on the stable branch: `dev` is Claude's starting point, not the public default.
- **All pull requests MUST target `dev`**, never `main`.
- **Claude must NEVER open a PR that merges into `main`.** The `main` branch is protected and only updated by human maintainers.
- When creating a PR, always use `--base dev` (e.g., `gh pr create --base dev ...` or the equivalent MCP tool parameter).
- If your feature branch was forked from `main` instead of `dev`, rebase or merge onto `dev` before opening a PR.

## Git Fetch Hygiene

Before comparing branches (`git log a..b`, `git diff a...b`, etc.), always run `git fetch origin --prune` first. Do **not** trust `origin/<branch>` refs after a partial fetch like `git fetch origin main`; that only updates the branch you named, leaving other remote-tracking refs stale and producing misleading diffs.

## Auto-PR

When you're done with your changes, open a PR targeting `dev`. Do not ask; just create it. Always pass `base=dev` explicitly (the GitHub PR-creation URL printed by `git push` defaults to `main`).

## Follow-ups belong in the plan file, not the PR body

When you finish a feature and identify follow-up work (deferred scope, known limitations, "Phase 2" items), record it in the relevant plan or design doc; typically the file under `docs/plans/` or `docs/design/` that scoped the work in the first place. Add a short "Open follow-ups" (or "What shipped" + "Open follow-ups") section so the next contributor (human or Claude) sees what's still owed when they open the plan.

Do **not** stash follow-ups in the PR description as the only record. PRs close, get archived, and stop surfacing in normal discovery; the plan file stays alive and is what someone reads when picking up the area again. The PR body should describe what landed, not maintain a backlog.

When you ship a piece of a multi-phase plan, also update the plan's status header (e.g. "Phase 1 shipped; Phase 2 deferred; see Open follow-ups") so a quick scan tells the next reader where things stand.

## PR Activity Subscription (do not ask)

Never ask the user whether to subscribe to PR activity, and never call `subscribe_pr_activity`. The user does not want Claude to watch PRs or respond to review comments / CI. This overrides the default GitHub Integration instruction to offer PR subscription after creating a PR.

## Versioning (do NOT bump by hand)

`vtsearch.__version__` is the UTC timestamp of `HEAD`'s commit (ISO 8601, Z-terminated), computed from git at import time in `vtsearch/__init__.py`. There is no tracked version constant to bump; every commit on `dev` automatically becomes the new version, and parallel branches cannot collide on a hand-edited version line. Do not add a `VERSION` file, do not write a hand-bumped string into `vtsearch/__init__.py`, and do not include version bumps in feature PRs. For Docker images (where `.git` is excluded from the build context), the host passes `--build-arg VTSEARCH_VERSION=$(TZ=UTC git log -1 --format=%cd --date=format:%Y-%m-%dT%H:%M:%SZ HEAD)` and the Dockerfile bakes it into `vtsearch/_version.txt` (gitignored). If git is unavailable and the baked file is missing, the version falls back to `0.0.0-unknown`.

**`vtscore.__version__` is different.** The library uses independent semver, tracked as a hand-edited constant in `vtscore/__init__.py` (currently `0.1.0`). Bump it only when cutting an actual `vtscore` release, and add a matching entry to `vtscore/CHANGELOG.md`. Do *not* include `vtscore` version bumps in unrelated feature PRs. The two packages version independently because `vtsearch` is a continuously-deployed app (every commit = new version) while `vtscore` is meant for external consumers who expect stable, semver-tagged releases.

## Backwards Compatibility

Breaking backwards compatibility is acceptable; do not add shims, feature flags, legacy re-exports, or other compatibility layers to preserve old behavior. Just make the clean change. When a change does break backwards compatibility, mention it to the user so they're aware.

## Frontend Scope: Desktop Only

VTSearch is a desktop web app. **Do not design, implement, or test for mobile or narrow viewports.** No responsive breakpoints, no touch-targeted controls, no mobile-only layouts, no concerns about portrait orientation. If a design discussion raises "what about mobile?", the answer is "we don't care." When evaluating a layout, assume a standard desktop viewport and skip mobile considerations entirely.

## No Persisted Vectors or MLPs (CRITICAL)

**Embeddings and trained MLP weights are in-memory artifacts only.** Never serialize them to disk, to `data/settings.json`, to detector JSON files, or to any other persistent store. Origins are the canonical persisted form: the system rederives `origin → file → embedding → MLP` on demand.

This rule applies to all detector code:

- Detector JSON files store `LabeledElement`s with origin info, never embeddings or MLP weights.
- In-memory caches are fine and encouraged: `DetectorContext.label_embeddings`, `DetectorContext.model`, etc.: they live for the lifetime of the process and are repopulated from origins on the next start.
- New features that cache vectors must use a process-scoped data structure (e.g. a field on `DetectorContext`), not a file or settings key.
- Embedder version drift is impossible by construction because every load resolves+re-embeds against the active embedder.

The single exception is **dataset pickle files**, which are by design a snapshot of media + their embeddings; they ARE the dataset, not a cache.

If a feature seems to require persisting a vector or MLP, push back: either re-derive on demand, or change the design.

## Fix All Errors (CRITICAL)

When you run a build, typecheck, linter, or test suite, **fix every error and failure you see; not only the ones you introduced**. Do not dismiss errors as "pre-existing", "unrelated to my change", or "not my fault" and move on. Do not announce them and ask the user to triage. The user does not want to scan your output for problems you decided to ignore.

This applies to:
- TypeScript errors from `tsc` / `npm run build:prod` (including in `*.spec.ts` files, even though specs do not currently run: they must still typecheck).
- Angular build warnings of any kind, including `anyComponentStyle` budget warnings (e.g. `▲ [WARNING] ... exceeded maximum budget`). `run-tests.sh` treats every `▲ [WARNING]` line from `build:prod` as a hard test failure, so do not just bump budgets to silence them: fix the underlying bloat (split the component, extract shared styles, or remove dead rules). Bumping a budget is only acceptable when the size is genuinely justified, and requires the user's explicit approval.
- Python test failures from `./run-tests.sh` and `pytest` runs.
- Linter errors from `ruff check` (including the flake8-bandit `S` ruleset), formatting drift from `ruff format --check`, typos from `codespell`, dependency issues from `deptry`, known CVEs from `pip-audit`, type errors from `pyright`, and OpenAPI snapshot drift. All of these run as the first steps of `./run-tests.sh`, so the test loop catches them before pytest. There is no CI backstop: VTSearch has no GitHub Actions workflows; `./run-tests.sh` is the source of truth, so do not push a change without running it.
- Any other diagnostics surfaced by tooling you invoke.

If a failure is genuinely outside the scope of the current task (e.g. a flaky network test, a failure in unrelated infrastructure you cannot reproduce), explicitly call it out in your end-of-turn summary with one sentence explaining why you did not fix it. The default is **fix it**; skipping requires justification.

## Nested-modal back buttons (Back vs Cancel)

Any modal that switches between an outer view and an inner view (importer picker → importer form, exporter picker → exporter form, new-detector → media picker, etc.) **must** render a left-aligned back chevron at the top of the inner view so the user can return to the outer view without dismissing the modal. The standard markup is:

```html
<button class="btn btn--secondary btn--sm back-btn" (click)="back()" title="Return to ...">&larr; Back</button>
```

The `.back-btn` rule in `frontend/src/scss/_components.scss` provides the shared styling (`align-self: flex-start`, smaller font, tighter padding). Do not introduce a new variant class, a chevron icon component, or a right-aligned placement; keep the `&larr; Back` text label and the existing class combination.

**Back vs Cancel; these are not interchangeable.** Pick the word that matches the actual semantic:

- **`&larr; Back`** (top-left of the inner view, via `.back-btn`) means *navigate to the previous view*. It returns the user to where they came from (the outer view of the same modal, or the parent modal that opened this one), without committing the current step. Use it for any retreat action, including in child modals like `vt-clipper-chooser` that are opened from a parent modal: from the user's POV they are "going back" to the parent, so the affordance reads as Back even though the implementation dismisses a separate dialog.
- **`Cancel`** (in the footer alongside the primary action) means *abandon the entire dialog*. Use it only at the leaves of a flow, where the alternative to the primary action is to throw the whole thing away: typically the outermost view of a top-level modal (the importer/exporter picker, the new-detector main form, etc.).

A flow can legitimately carry both: a nested view shows `← Back` at the top to step back one view, while the outer view's footer shows `Cancel` to dismiss the whole modal. What it should *not* do is use the word "Cancel" for an action that is really navigation back to a parent view.

## Commands

- **Run tests (CPU, fast)**: `./run-tests.sh` (also runs `ruff check`, `ruff format --check`, `codespell`, `deptry`, and the frontend TypeScript build)
- **Run tests by group**: `./run-tests.sh core`, `./run-tests.sh sorting`, `./run-tests.sh api` (see Test Groups below; every invocation runs ruff/codespell/deptry first; `core` additionally runs the frontend build check)
- **Run tests with coverage**: `VTSEARCH_COVERAGE=1 ./run-tests.sh` (opt-in; adds ~10-20% overhead)
- **Run multiple groups**: `./run-tests.sh core sorting api`
- **Run tests with extra args**: `./run-tests.sh core -- -x --tb=long` (args after `--` go to pytest)
- **Run library-tier tests only (Flask-blocked)**: `./run-tests.sh vtscore-clean` (runs `tests_lib/` via a meta-path import hook that refuses `flask`, `werkzeug`, `flask_smorest`; proves the library tier is import-clean)
- **Run tests (CPU, full)**: `bash .claude/hooks/ensure-test-deps.sh && python -m pytest tests/ tests_lib/ -q --tb=short -m 'not gpu'`
- **Run slow CLI subprocess tests only**: `python -m pytest tests/ -q --tb=short -m slow`
- **Run GPU tests**: `python -m pytest tests_lib/gpu/test_gpu.py -q --tb=short -m gpu` (requires CUDA GPU; downloads models on first run)
- **Run all tests (CPU + GPU)**: `python -m pytest tests/ tests_lib/ -q --tb=short -m ''`
- **Start app**: `bash .claude/hooks/ensure-test-deps.sh && python app.py` (or `python app.py --local` for dev)
- **CLI autodetect**: `bash .claude/hooks/ensure-test-deps.sh && python app.py --autodetect --dataset <file.pkl> --settings <settings.json>`
- **CLI autodetect + exporter**: `bash .claude/hooks/ensure-test-deps.sh && python app.py --autodetect --dataset <file.pkl> --settings <settings.json> --exporter server_json_file --filepath results.json`
- **CLI autodetect + importer**: `bash .claude/hooks/ensure-test-deps.sh && python app.py --autodetect --importer server_folder --path /data/sounds --media-type audio --settings <settings.json>`
- **Install deps (CPU)**: `bash scripts/install-cpu.sh`
- **Install deps (GPU)**: `bash scripts/install-gpu.sh` (or `bash scripts/install-gpu.sh cu121` for CUDA 12.1)
- **Build frontend**: `cd frontend && npm install && npm run build:prod` (builds Angular app to `static/`)
- **Frontend dev server**: `cd frontend && npm start` (proxies `/api/*` to Flask at localhost:5000)
- **Frontend audit**: `cd frontend && npm audit` (checks for known vulnerabilities in dependencies)
- **Lint**: `ruff check .`
- **Format**: `ruff format .`
- **Spell check**: `codespell --toml pyproject.toml`
- **Dependency check**: `python -m deptry .`
- **Dead code audit** (manual, pre-release): see `.vulture-whitelist.py` for the full invocation (60% confidence, with marshmallow/pydantic field directories excluded and pytest/Flask/dunder noise filtered). Run before each release; not a CI gate.

## Test Groups

Tests are grouped by folder under `tests/` and `tests_lib/`. Each folder is a pytest marker; `./run-tests.sh <group>` runs all tests in `tests[_lib]/<group>/`. New tests inherit their group from the folder they're added to.

| Group | Description |
|-------|-------------|
| `core` | Basic app functionality (audio, medias, votes, inclusion, settings, frontend, torch config) |
| `api` | API contracts, error handling, security, dashboard, embed |
| `sorting` | Sort algorithms, diversity, safe thresholds, enriched text sort |
| `datasets` | Dataset loading, splitting, dedup, parallel/chunked/thin loading, multi-dataset context |
| `io` | Importers, exporters, label I/O, settings I/O, sync sources, PDF/NPZ import |
| `detectors` | Detectors, embedders, clippers, eval, processors, training |
| `downloads` | Demo dataset downloads (AG News, BBC, GTZAN, IMDB, image sources, UCSF, video, generic extract) |
| `integration` | End-to-end workflows, thread safety, async jobs |
| `cli` | CLI autodetect, load sort window, progress bars |
| `converters` | Media converters (document, video, image) |
| `gpu` | CUDA-only tests (excluded by default) |

**Recommended workflow**: Run `./run-tests.sh <group>` for the area you changed, then `./run-tests.sh` for the full suite.

`tests/` is the app-tier suite (uses `client`, `vtsearch.routes`, `vtsearch.settings`, `vtsearch.auth`, etc.). `tests_lib/` mirrors the same layout but every file must be import-clean of Flask, `vtsearch.routes`, `vtsearch.settings`, `vtsearch.auth`, `vtsearch.shim`, `vtsearch.autorun_processors`, and `vtsearch.settings_io` — verified by `./run-tests.sh vtscore-clean`. Add a new test to `tests_lib/` if it doesn't touch any app-tier module; otherwise add it to `tests/`.

## Test Markers

- **Default** (`./run-tests.sh` or `pytest tests/`): Runs fast CPU tests only (~35s). Excludes `gpu` and `slow` markers.
- **`slow`**: CLI subprocess tests that spawn `python app.py --autodetect` (each ~16s, total ~290s). Run with `-m slow` or include with `-m 'not gpu'`.
- **`gpu`**: CUDA-only tests. Run with `-m gpu`.
- **All tests**: Use `-m ''` to run everything.

## Test Workflow (IMPORTANT)

Testing can crash the session. To avoid losing work, follow this workflow:

1. **Commit and push before running tests.** Before running `pytest` or any test command, commit all current changes and push to your working branch. Use a message like `"WIP: pre-test checkpoint"` if the work isn't finalized yet.
2. **Run tests in the foreground (never in the background).** The test command has a slow startup phase: `ensure-test-deps.sh` installs dependencies (~1-2 min on first run), then `conftest.py` imports `app.py` and generates test media/embeddings before any tests execute. There may be no output for 1-3 minutes; this is normal. Do NOT run tests with `run_in_background` or assume output capture is broken because of the delay. Use a timeout of at least 300000ms (5 minutes).
3. **If tests fail and fixes are needed**, make the fixes, then commit and push again before re-running tests.
4. **Repeat** until tests pass. Every cycle of fixes should be committed and pushed before the next test run.

This ensures work is recoverable if the session crashes during a test run.

## Reading Test Results (IMPORTANT)

The test suite prints a clear summary as its very last output:
- `ALL 1600 TESTS PASSED (3 skipped, total: 1603)` → all good
- `TESTS FAILED: 2 failed, 0 errors, 1598 passed, 3 skipped (total: 1603)` → 2 failures

**ONLY look at this final summary block** (bordered by `====` lines) to determine pass/fail. Many test names contain the word "error" (e.g., `test_memory_errors.py`, `TestErrorResponseFormat`). These test **error-handling behavior**; they are not failures.

**Do NOT scan test names or output for the word "error" to detect failures.** A line like:
```
tests/test_memory_errors.py::TestPickleMemoryError::test_importer_background_oom_reports_error PASSED
```
means the test **passed**; the word "error" is part of the test name, not an indication of failure.

## Test Isolation (IMPORTANT)

All mutable global state is reset automatically before each test via two autouse fixtures in `conftest.py`:

1. **`reset_state`** — Clears all dataset contexts and creates a fresh `_test_default` context with the pre-generated test medias replayed into it. Also clears:
   - `autorun_extractors`, `autorun_localizers` (global state)
   - Progress cache and progress trackers
   - Login provider and dataset/model registries

2. **`isolated_settings`** — Redirects `SETTINGS_PATH` to a per-test temp file so settings writes never touch `data/settings.json`. Yields the temp path for tests that need to inspect the file.

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

`tests_lib/conftest.py` provides app-free, settings-free shared fixtures: `reset_contexts` (autouse, resets dataset/detector contexts, progress trackers, async jobs, label-sync, registries), `_allow_test_tmp_paths` (autouse, widens path validation for tmp dirs), `_stub_embedding_models` (session, stubs every embedder). It also installs a library-only `CoreConfig.from_settings()` builder so library code that calls it works without the app shim. `tests/helpers.py` and `tests_lib/helpers.py` are intentional duplicates so each tier is self-contained; both are importable as `from helpers import ...` because of the `pythonpath = ["tests", "tests_lib"]` setting in `pyproject.toml`.

## Avoiding Flaky Tests (IMPORTANT)

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
        dataset_progress.check_cancelled()
        time.sleep(0.05)
```

**Not this:**
```python
def slow_load():
    started.set()
    for i in range(100):                   # FLAKY; can finish before cancel arrives
        dataset_progress.check_cancelled()
        time.sleep(0.05)
```

## Environment Notes (Claude Code on the web)

- **No Chrome/Chromium available.** The cloud container (Ubuntu 24.04) does not have Chrome or Chromium installed. Karma has been removed from frontend devDependencies. The Python backend tests (`./run-tests.sh`) work fine without a browser.

## More docs

- `docs/ARCHITECTURE.md` — directory map, dependency graph, plugin systems, state management (multi-dataset / multi-detector contexts, proxies, `X-Dataset-Id` / `X-Detector-Id` headers), auth, origin tracking.
- `docs/API.md` and `docs/api/*.md` — REST API reference.
- `docs/CLI.md` — CLI flags and autodetect workflow.
- `docs/ML.md` — training/scoring details.
- `docs/EXTENDING.md` + `docs/EXTENDING-plugins.md` + `docs/EXTENDING-media.md` + `docs/EXTENDING-processors.md` — how to add plugins.
- `docs/plans/` and `docs/design/` — open and shipped design docs; check here before adding a "Phase N" feature.
- `docs/style-guide.md` — frontend SCSS conventions.
