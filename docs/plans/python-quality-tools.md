# Python quality tools

*Status: all planned phases shipped. The current stack: pre-commit +
pip-audit (Phase 1); deptry + codespell (Phase 2); ruff `S` security
ruleset + opt-in coverage + vulture whitelist (Phase 3); pyproject
consolidation; CI coverage publication; full vulture audit pass; ruff
`C901` McCabe complexity gate with `max-complexity = 25`; and a
`diff-cover` patch-coverage gate on PRs. `pre-commit autoupdate` is the
only remaining recurring maintenance item — see "Open follow-ups" at
the bottom.*

The ruff + pyright + pytest stack already covers the modern core of
Python quality tooling. This plan tracks what else is worth adding,
grouped by friction. Each tool gets a clear "what it catches", "how to
wire it", and a success criterion so it's a one-sitting task.

## What shipped (Phase 1)

- **pre-commit** (`.pre-commit-config.yaml`) — runs `ruff check`,
  `ruff format`, trailing-whitespace, end-of-file-fixer, yaml/toml/json
  syntax checks, merge-conflict markers, a 500 KB large-file guard,
  plus codespell and deptry at commit time. Pinned to
  `ruff-pre-commit v0.15.8`, `pre-commit-hooks v5.0.0`, and
  `codespell v2.4.2`; bump with `pre-commit autoupdate`.
- **`pip-audit`** (`.github/workflows/audit.yml`) — scans installed
  packages against the PyPI advisory database. Runs on PRs that touch
  `requirements/**` or `pyproject.toml`, on pushes to `main`/`dev`, and
  on a weekly cron so newly-disclosed CVEs surface even when nothing
  has changed in the repo.
- **`pre-commit` package** added to `requirements/base.txt`;
  `scripts/install-cpu.sh` runs `pre-commit install --install-hooks`
  automatically when `.git` is present so contributors get the git hook
  on first install.

## What shipped (Phase 2)

- **deptry** (`[tool.deptry]` in `pyproject.toml`, `.pre-commit-config.yaml`
  local hook, `.github/workflows/lint.yml` job) — verifies that every
  imported package is declared as a runtime dependency. Configured with
  a `package_module_name_map` for the usual module/distribution
  mismatches (`PIL`/`Pillow`, `cv2`/`opencv-python-headless`,
  `sklearn`/`scikit-learn`, `fitz`/`PyMuPDF`, `dns`/`dnspython`, etc.)
  and a `per_rule_ignores` list for optional/dev-only deps
  (`paddleocr`, `whisper`, `mediapipe`, `rarfile`, `torchvision`,
  `safetensors`, `huggingface_hub` for DEP001; `pytest`, `ruff`,
  `gunicorn`, `sentencepiece`, `protobuf`, etc. for DEP002). The runtime
  dep list lives in `[project.dependencies]` so deptry sees it; the
  install spec (with `--extra-index-url` and the plugin `.txt`
  fan-out) remains `requirements/base.txt`.
- **codespell** (`[tool.codespell]` in `pyproject.toml`,
  `.pre-commit-config.yaml`, `.github/workflows/lint.yml` job) — catches
  typos in identifiers, comments, docstrings, and Markdown. Configured
  with a `skip` list for generated/binary files and an
  `ignore-words-list` of project-specific terms (embedder names like
  `clap`/`siglip`/`xclip`, code identifiers like `numer`/`denom`/`medias`/`fpr`,
  and stylistic spellings the project uses consistently like `re-use`
  and `pre-select`).

## What shipped (pyproject consolidation)

- **`pyproject.toml` is now the single source of truth for runtime + dev
  dependencies.** `[project.dependencies]` and
  `[project.optional-dependencies].dev` hold the full list; the
  top-level `requirements/base.txt` (CPU) and `requirements/gpu.txt`
  (GPU) just forward to it via `-e .[dev]`. The per-plugin
  `vtsearch/.../requirements*.txt` files, the auto-generated
  `requirements/plugins.txt`, and `scripts/install-plugin-deps.sh` are
  gone — deptry now catches drift from either side because there is no
  "other side" any more. The labbench /
  image-embedders requirements files stay standalone (curated minimal
  subsets for size-constrained Docker images). Updated:
  `scripts/install-cpu.sh`, `scripts/install-gpu.sh`,
  `docker/Dockerfile`, `docker/Dockerfile.gpu`,
  `.claude/hooks/ensure-test-deps.sh`, the four plugin-family
  `base.py` docstrings, and the docs (`SETUP.md`, `DEPLOYMENT.md`,
  `HANDOFF.md`, `EXTENDING.md`, `EXTENDING-plugins.md`,
  `plans/RCDatasetImporter.md`).

## What shipped (CI coverage publication)

- **Coverage in CI** (`.github/workflows/coverage.yml`) — runs the fast
  CPU test suite (`-m 'not gpu and not slow'`) with `pytest-cov`, then
  publishes the per-file `coverage report --skip-covered --sort=cover`
  table to `$GITHUB_STEP_SUMMARY` and uploads the HTML report (and the
  raw `coverage.xml`) as artifacts with a 14-day retention. Runs on
  every PR plus pushes to `main`/`dev`. This is the **first** workflow
  that runs pytest in CI — prior to this, tests were only enforced via
  local `./run-tests.sh` and pre-commit, so the new job also gates merges
  on test pass/fail (a real failure surfaces in the same job that
  publishes coverage). There is no coverage-delta gate yet; that
  decision waits until we have a baseline.

## What shipped (Phase 3)

- **Ruff `S` ruleset** (`select = ["E4", "E7", "E9", "F", "S"]` in
  `[tool.ruff.lint]`) — flake8-bandit security checks now run as part
  of `ruff check`. Several rules are globally disabled because they
  don't match VTSearch's threat model (single-user, on-prem) and
  produce noise more than signal: S101 (assert used for type
  narrowing), S104 (bind 0.0.0.0 is intentional for the dev server),
  S105 (false-positive-prone on env-var names), S110/S112 (silent
  except is intentional in best-effort code paths), S311 (random
  used for dataset splitting, not crypto), S324 (md5/sha1 used as
  content fingerprints). Real security-relevant rules — S301
  (pickle.load), S314 (xml parsing), S603/S607 (subprocess) — stay
  enabled with targeted per-file ignores for the few legitimate
  uses (`safe_pickle_load`, the arXiv RSS-feed parser, git/ffmpeg/unrar
  invocations).
- **Coverage via `pytest-cov`** (`[tool.coverage]` in `pyproject.toml`,
  opt-in flag in `run-tests.sh`) — coverage is gathered when
  `VTSEARCH_COVERAGE=1 ./run-tests.sh` is run; off by default to keep
  the fast-loop test time unchanged. No PR gate yet — the next step is
  to publish a step summary in CI and then decide on a delta gate.
- **Vulture whitelist** (`.vulture-whitelist.py`) — a curated list of
  plugin sentinels (`IMPORTER`, `EXPORTER`, `CONVERTER`, `SOURCE`,
  etc.) and other reflectively-referenced symbols, so
  `vulture vtsearch .vulture-whitelist.py --min-confidence 80` runs
  cleanly. Vulture is intentionally NOT a CI gate — it's a manual
  pre-release audit since lower confidence settings produce too many
  false positives against the plugin-discovery pattern.
- **Vulture audit pass** at `--min-confidence 60` — completed in
  May 2026. The invocation now lives in `.vulture-whitelist.py`'s
  module docstring and is referenced from `CLAUDE.md`'s commands list.
  Key tunings: `--exclude` for `vtsearch/schemas/*` and
  `vtsearch/settings_models.py` (every marshmallow field assignment and
  pydantic field declaration looks "unused" to vulture because both
  frameworks collect fields via metaclass — there is no static reference
  to attach), `--ignore-decorators` covering the full Flask + pytest
  decorator surface, and `--ignore-names` for Meta inner classes,
  `model_config`, the HuggingFace `_keys_to_ignore_on_load_unexpected`
  attribute, pytest `test_*`/`Test*`/`pytest_*`/`pytestmark` patterns,
  and Python protocol dunders (`__enter__`, `__exit__`, `__package__`).
  The triage:
  * **Deleted as genuinely dead:** `serialize_job` and the unused
    `finished_at` attribute in `vtsearch/concurrency/async_jobs.py`,
    `_search_dir_for_file` in the http-archive importer,
    `_activate_new_context` in `vtsearch/datasets/load_pipeline.py`,
    `_pop_embedding_key` in `vtsearch/datasets/loader.py`,
    `embed_image_file_from_pil` in `vtsearch/datasets/loader_pickle.py`,
    `_ProtoLeaf` in `vtsearch/media/patch_embed.py`, `_save_for_key` and
    `_get_active_cache_for_key` in `vtsearch/settings.py`, plus a
    handful of dead test helpers (`_read_sse_event`, `_make_minimal_wav`,
    `_make_text_file`, `make_document_media`, `make_minimal_pdf_bytes`,
    `build_results_dict`, `make_trainable_model_file`, `_populate`,
    `_StubMedias`) and several unused local variables. Renamed an
    unused loop variable and a few unused function parameters to the
    `_<name>` convention.
  * **Whitelisted as public API / reflective use:** Flask's
    `secret_key`, the API-symmetry progress wrappers
    (`check_dataset_cancelled`, `get_sort_progress`, `get_find_progress`),
    documented public constants (`SAVED_DATASETS_DIR`, `DETECTORS_DIR`,
    `SAMPLE_VIDEOS_DOWNLOAD_SIZE_MB`), public training/labelset APIs
    (`find_by_pkl_path`, `recreate_model_at_time`, `update_cache_for_cid`,
    `collect_media_origins`, `train_detector_from_origins`), the public
    state context managers (`with_dataset_context`, `with_detector_context`),
    `default_concurrent_downloads`/`default_concurrent_embeddings`
    (called from the excluded `settings_models.py`), and the
    TYPE_CHECKING-only settings accessor stubs that pyright needs but
    vulture sees as orphans (`get_audio_playing`, `get_swipe_animation`,
    `get_hide_autopilot`, `get_autopilot_resort_interval`).

  The tuned invocation now exits 0 on a clean tree, so introducing a
  new piece of dead code reliably surfaces in the audit.

## What shipped (C901 complexity gate)

- **Ruff `C901` McCabe complexity** added to `select` in
  `[tool.ruff.lint]`, with `max-complexity = 25` configured under
  `[tool.ruff.lint.mccabe]`. The bar is intentionally a *soft ceiling*
  rather than the conventional 10: VTSearch's data-pipeline functions
  (folder loaders, demo-source loaders, multi-find) legitimately weave
  many branches, and forcing them down to 10 would mean a multi-day
  refactor sprint without changing behaviour. 25 catches genuinely
  runaway complexity — the rule fires on anything new that creeps into
  the 25+ range — while letting the existing pipeline code stay as-is.
  The ten functions currently above 25 are grandfathered with a
  per-function `# noqa: C901` so the rule passes on a clean tree:
  * `run_converters_on_folder` (28) — `vtsearch/converters/runner.py`
  * `_run_origin_load_in_background` (31) and nested `task` (27) —
    `vtsearch/datasets/load_pipeline.py`
  * `load_dataset_from_folder` (34) and
    `load_dataset_from_folder_chunked` (35) —
    `vtsearch/datasets/loader_folder.py`
  * `load_dataset_from_pickle` (26) —
    `vtsearch/datasets/loader_pickle.py`
  * `load_demo_source` (52) — `vtsearch/media/image/_demo_sources.py`
  * `load_demo_source` (31) — `vtsearch/media/text/media_type.py`
  * `multi_find` (41) — `vtsearch/routes/detectors/find.py`
  * `learned_sort` (29) — `vtsearch/routes/sorting.py`

  The intent is to ratchet `max-complexity` down over time as those
  functions get split. Future contributors should prefer fixing the
  underlying function over adding another `# noqa: C901` — adding a
  noqa is allowed but counts as a known regression of the ratchet.

## What shipped (patch-coverage gate via diff-cover)

- **`diff-cover` gate** in `.github/workflows/coverage.yml` — installs
  `diff-cover` on PR runs, fetches `origin/${{ github.base_ref }}` to a
  depth that lets diff-cover see the merge base, and runs
  `diff-cover coverage.xml --compare-branch=origin/<base>
  --fail-under=80`. The job uploads a markdown report into
  `$GITHUB_STEP_SUMMARY` so reviewers see exactly which patch lines are
  uncovered. **Patch-level only — not total-coverage gating.** Total
  coverage moves up and down with unrelated test reshuffles and would
  produce noisy red on PRs that didn't touch tested code; the patch
  number measures "did this PR cover the lines it added?", which is the
  signal we actually care about. The 80% threshold is a starting point
  and is easy to revisit once we have a few PRs of real data — change
  `--fail-under=80` in `coverage.yml` to retune. The gate runs only on
  `pull_request` events; push runs to `main`/`dev` still publish the
  total-coverage summary but don't compute a patch number (there's no
  meaningful "base" to diff against).

## What we considered and skipped

- **mypy** — second type checker, redundant with pyright. Adding it
  would mean reconciling two type-checker opinions on every diff. Not
  worth it unless we hit a concrete pyright limitation.
- **interrogate** (docstring coverage) — we don't enforce docstrings
  anywhere, and the codebase doesn't have a "every public function
  needs a docstring" norm. Adding it would create a large backlog with
  little payoff.
- **radon / xenon** (cyclomatic complexity) — ruff already has McCabe
  via the `C901` rule, which is now enabled (see "What shipped (C901
  complexity gate)"). No second tool needed.
- **semgrep** — powerful pattern-based static analysis but heavy for a
  single-app repo. Revisit if we ever start shipping VTSearch as a
  library or hosting multi-tenant.
- **safety** (the package) — overlaps `pip-audit` and has a more
  restrictive license. `pip-audit` is the right choice here.

## Open follow-ups

- **Re-run the vulture audit before each release.** The tuned
  invocation lives in `.vulture-whitelist.py`'s module docstring; the
  audit is not a CI gate, so introducing new dead code only surfaces
  when someone runs it. A release-checklist reminder is the lightest-
  weight way to keep it from rotting.
- **Ratchet `max-complexity` downward over time.** Current value is
  25, with 10 grandfathered `# noqa: C901` sites listed under "What
  shipped (C901 complexity gate)". As those functions get split (or
  when somebody is in the area for unrelated reasons), drop the
  threshold a notch and remove the corresponding noqa. The end state
  is something close to ruff's 10 default — but there is no rush.
- **Revisit the diff-cover threshold.** Currently `--fail-under=80` in
  `coverage.yml` — a guess. After a handful of PRs, look at what
  patch-coverage numbers feel like noise vs. signal and retune. If
  legitimate refactor-only PRs keep tripping the gate (because
  refactored lines have to be re-touched and so look "new" but
  weren't tested explicitly), consider relaxing it.
- **Periodic `pre-commit autoupdate`.** Pinned hook versions drift over
  time vs. CI's `pip install ruff` (which always pulls latest). Worth a
  quarterly reminder to bump `.pre-commit-config.yaml`.
