# Python quality tools

*Status: Phase 1 shipped — pre-commit (ruff + safety hooks) is wired up
and `pip-audit` runs in CI. Phases 2 and 3 are proposed; see "Open
follow-ups" at the bottom for what's still owed.*

The ruff + pyright + pytest stack already covers the modern core of
Python quality tooling. This plan tracks what else is worth adding,
grouped by friction. Each tool gets a clear "what it catches", "how to
wire it", and a success criterion so it's a one-sitting task.

## What shipped (Phase 1)

- **pre-commit** (`.pre-commit-config.yaml`) — runs `ruff check`,
  `ruff format`, trailing-whitespace, end-of-file-fixer, yaml/toml/json
  syntax checks, merge-conflict markers, and a 500 KB large-file guard
  at commit time. Pinned to `ruff-pre-commit v0.15.8` and
  `pre-commit-hooks v5.0.0`; bump with `pre-commit autoupdate`.
- **`pip-audit`** (`.github/workflows/audit.yml`) — scans installed
  packages against the PyPI advisory database. Runs on PRs that touch
  `requirements/**` or `pyproject.toml`, on pushes to `main`/`dev`, and
  on a weekly cron so newly-disclosed CVEs surface even when nothing
  has changed in the repo.
- **`pre-commit` package** added to `requirements/base.txt`;
  `scripts/install-cpu.sh` runs `pre-commit install --install-hooks`
  automatically when `.git` is present so contributors get the git hook
  on first install.

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

- Phase 2: wire up **deptry** and **codespell** (both ~1-hour tasks).
- Phase 3a: turn on **ruff `S` ruleset** (covers most of what bandit
  would). Triage the first-run findings, add targeted ignores.
- Phase 3b: wire up **coverage** via `pytest-cov`. Publish-only first;
  decide on a delta gate after we see real numbers.
- Phase 3c: one-shot **vulture** audit. Build a whitelist for the
  plugin sentinels, delete what's actually dead, then leave vulture as
  a manual pre-release command rather than a CI gate.
- Periodic: `pre-commit autoupdate` on a cadence so pinned hook
  versions don't drift too far from CI's `pip install ruff` (which
  always pulls latest). Worth a quarterly reminder.
