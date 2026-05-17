# Python quality tools

*Status: Phases 1 + 2 + 3 shipped, plus the pyproject-consolidation
follow-up. deptry, codespell, ruff's `S` ruleset, opt-in coverage, and a
vulture whitelist are all in place alongside the original pre-commit +
pip-audit, and `pyproject.toml` is now the single source of truth for
runtime + dev dependencies. See "Open follow-ups" at the bottom for the
remaining maintenance items.*

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

## Phase 2 — Low-friction additions

These are each a one-config-file change with an obvious failure mode.
Worth doing soon; can be done independently in any order.

### deptry — unused / missing / transitive deps

Catches three kinds of dependency bugs:

- A package is imported but not listed in `requirements/base.txt` (works
  on your laptop because something else dragged it in, breaks in a
  fresh container).
- A package is listed but never imported (dead weight in installs).
- A package is imported but only available transitively (will break the
  day the direct dep drops it).

Important for VTSearch because plugin auto-discovery means
`requirements/plugins.txt` can drift from the actual `import` graph
quickly.

**Wire-up:**
- Add `deptry` to `requirements/base.txt` dev tools.
- `[tool.deptry]` block in `pyproject.toml`:
  - `extend_exclude = ["frontend", "scripts", "data"]`
  - Map heavy-but-aliased packages (e.g. `torch` is `pytorch`?
    `scikit-learn` is `sklearn`) via
    `[tool.deptry.package_module_name_map]`.
  - Mark `pytest`, `pytest-xdist`, `pre-commit`, `ruff`, `pip-audit` as
    dev-only via `[tool.deptry.per_rule_ignores]` (DEP002) — they're
    intentionally not imported.
- Add to `.pre-commit-config.yaml` (local hook, since deptry needs the
  project venv to resolve imports):
  ```yaml
  - repo: local
    hooks:
      - id: deptry
        name: deptry
        entry: deptry .
        language: system
        pass_filenames: false
        files: ^(vtsearch/|pyproject\.toml|requirements/)
  ```
- Add a `deptry` job to `lint.yml` so PRs are gated.

**Success criterion:** `deptry .` returns 0 against the current tree;
introducing an unused import or a missing dep fails CI.

### codespell — misspellings in code and docs

Catches typos in identifiers, comments, docstrings, and Markdown. Very
low signal-to-noise once configured. The cost is one config block to
exclude domain-specific terms that aren't in the dictionary.

**Wire-up:**
- Add codespell hook to `.pre-commit-config.yaml`:
  ```yaml
  - repo: https://github.com/codespell-project/codespell
    rev: v2.3.0
    hooks:
      - id: codespell
        additional_dependencies: [tomli]
  ```
- `[tool.codespell]` block in `pyproject.toml` with:
  - `skip = "*.lock,*.svg,*.json,frontend/node_modules,data,static,*.pkl"`
  - `ignore-words-list = "..."` for ML terms it doesn't know
    (`clap`, `siglip`, `xclip`, `wrest`, `caller`, etc. — populate as
    false-positives surface).
- Add codespell to `lint.yml` for the CI gate.

**Success criterion:** `codespell` passes against the current tree;
introducing a typo in a comment or docstring fails CI.

## Phase 3 — Medium-friction additions

These each need a real config pass — not because the tool is hard but
because the first run will surface a backlog of findings that someone
has to triage. Plan for an afternoon per tool, not five minutes.

### coverage.py / pytest-cov — line + branch coverage

Tells us which lines and branches the test suite exercises. Useful for
two things: spotting modules without test coverage, and (eventually)
gating PRs on "don't lower coverage by more than N%".

**Open questions to settle before wiring:**
- Coverage target — pick a starting number from the first measured run,
  don't pluck one out of the air. (Realistic: VTSearch probably lands
  in the 60–80% range given the size of the test suite.)
- Whether to gate PRs on coverage delta (e.g. via `diff-cover`) or just
  publish a report to the GitHub step summary. Start with publish-only
  for the first cycle.

**Wire-up:**
- Add `pytest-cov` to `requirements/base.txt`.
- `pyproject.toml`:
  ```toml
  [tool.coverage.run]
  source = ["vtsearch"]
  branch = true
  omit = ["vtsearch/_version.txt"]

  [tool.coverage.report]
  exclude_lines = [
      "pragma: no cover",
      "if TYPE_CHECKING:",
      "raise NotImplementedError",
  ]
  ```
- `./run-tests.sh` invokes `pytest --cov=vtsearch --cov-report=term-missing`.
- `lint.yml` (or a dedicated `coverage.yml`) uploads HTML report as an
  artifact + writes summary to `$GITHUB_STEP_SUMMARY`.

**Success criterion:** `./run-tests.sh` prints a coverage summary at
the end; CI artifact shows the per-file breakdown.

### bandit (or ruff's `S` ruleset) — security linter

Catches `pickle.load` on untrusted input, `subprocess` with `shell=True`,
weak crypto, hard-coded passwords, SQL injection patterns, etc. VTSearch
already does some of this manually (`safe_pickle_load`,
`validate_server_filepath`); a linter is a backstop.

**Important call:** ruff has the `S` (flake8-bandit) ruleset built in.
Enabling that is strictly simpler than adding bandit as a second tool —
no new dep, no new CI job, and config lives in the existing ruff block.
The catch is ruff's S rules are a subset of full bandit. For VTSearch's
threat model (single-user, on-prem, no untrusted callers in the
critical path), the ruff subset is enough.

**Wire-up:**
- Add to `[tool.ruff.lint]` in `pyproject.toml`:
  ```toml
  select = ["E", "F", "S"]   # default rules + flake8-bandit
  ```
- `[tool.ruff.lint.per-file-ignores]`:
  - `"tests/**" = ["S101"]` (allow `assert` in tests)
  - `"vtsearch/security/pickle.py" = ["S301"]` (the whole point of the
    file is to call `pickle.load` carefully)
  - Other case-by-case waivers as findings surface.
- First-pass triage commit fixes the easy ones and adds targeted
  `# noqa: S###` for the legitimate exceptions.

**Success criterion:** `ruff check .` with `S` rules enabled returns 0
against the current tree; introducing a new `subprocess(..., shell=True)`
or `pickle.load` of untrusted bytes fails lint.

### vulture — dead code finder

Finds functions, classes, imports, and variables that are defined but
never referenced. Best run as a periodic audit (it has false positives
from dynamic dispatch and plugin discovery) rather than a CI gate.

**Why not in Phase 2:** VTSearch's plugin-discovery pattern (sentinel
constants like `IMPORTER`, `EXPORTER`, `SETTINGS_SOURCE`,
`LABEL_IMPORTER`, etc.) means every plugin class looks unused to
vulture. The whitelist will need real curation, and the value is a
one-shot cleanup rather than ongoing enforcement.

**Wire-up (when we get to it):**
- Add `vulture` to dev deps.
- Create `.vulture-whitelist.py` exporting all plugin classes + any
  reflection-only symbols.
- Run as a manual command: `vulture vtsearch .vulture-whitelist.py
  --min-confidence 80`.
- Don't gate CI on it — just run it before each release and act on the
  high-confidence findings.

**Success criterion:** the audit produces a short, hand-reviewable
list of dead code; the obvious findings are deleted, the rest are
added to the whitelist with a comment explaining why.

## What we considered and skipped

- **mypy** — second type checker, redundant with pyright. Adding it
  would mean reconciling two type-checker opinions on every diff. Not
  worth it unless we hit a concrete pyright limitation.
- **interrogate** (docstring coverage) — we don't enforce docstrings
  anywhere, and the codebase doesn't have a "every public function
  needs a docstring" norm. Adding it would create a large backlog with
  little payoff.
- **radon / xenon** (cyclomatic complexity) — ruff already has McCabe
  via the `C901` rule. Enable that in the same pass as Phase 3's
  bandit/S work if we want complexity gating.
- **semgrep** — powerful pattern-based static analysis but heavy for a
  single-app repo. Revisit if we ever start shipping VTSearch as a
  library or hosting multi-tenant.
- **safety** (the package) — overlaps `pip-audit` and has a more
  restrictive license. `pip-audit` is the right choice here.

## Open follow-ups

- **Coverage publish in CI.** `VTSEARCH_COVERAGE=1` runs locally but
  there's no CI publication yet. Next step: add a `coverage` job (or
  extend the existing test job) to upload an HTML report as an
  artifact and write a per-file summary to `$GITHUB_STEP_SUMMARY`.
  Decide on a delta gate (e.g. `diff-cover`) once we have a baseline.
- **Vulture audit pass.** The whitelist makes `--min-confidence 80`
  clean; doing a real pass at 60 % and deleting confirmed dead code
  (after extending the whitelist for plugin/reflective references) is
  still outstanding. Run before each release rather than as a CI gate.
- **Pyright `S`-equivalent and `C901` complexity.** ruff has McCabe
  complexity via `C901`; consider enabling alongside future
  security-rule tightening if we want a complexity gate.
- Periodic: `pre-commit autoupdate` on a cadence so pinned hook
  versions don't drift too far from CI's `pip install ruff` (which
  always pulls latest). Worth a quarterly reminder.
