# Pyright Type-Checking

Adopt **pyright in basic mode** as a hard CI gate across the `vtsearch/`
package. Rolled out in stages so each PR stays reviewable.

## Status

- **Stage 0 + Stage 1:** ✅ shipped in PR #1349. The gated CI job covers
  `auth/`, `cli.py`, `concurrency/`, `config.py`, `exporters/`, `labels/`,
  `plugins/`, `settings_io/`, `sync/`, `utils/`. The advisory job runs over
  the full `vtsearch/` package on every PR and prints the residual error
  count to the GitHub step summary.
- **Stage 2:** ⏳ next — `settings.py`, `settings_factory.py`, `state/`,
  `security/`.
- **Stages 3–6:** 📋 not started.
- **Stage 7 (`tests/`):** 📋 optional, deferred.

## Goal

- `pyrightconfig.json` at repo root with `typeCheckingMode: "basic"`.
- A CI job (`.github/workflows/pyright.yml`) that fails the build on any
  pyright error in the **gated** scope.
- An **advisory** CI step that runs pyright over the entire `vtsearch/`
  package on every PR (does not fail the build) so we always know the
  residual error count for the next stage.
- Each stage expands the gated scope by editing `include` in
  `pyrightconfig.json`. Once stage 6 lands, the gated scope == the
  advisory scope and the advisory job goes away.

## Why pyright, why basic mode

- **Pyright over mypy**: faster, better inference, fewer plugins
  required to handle Flask/numpy/torch idioms.
- **Basic mode over strict**: basic flags real bugs (None-attribute
  access, wrong arg counts, undefined names, override mismatches)
  without forcing exhaustive annotations on a 49k-line codebase.
  Strict can come later if we want it.

## Baseline (2026-05-15)

`pyright vtsearch/` from a clean working tree reports **482 errors** and
1 warning, broken down by directory:

| Dir | Total | Non-import |
|---|---:|---:|
| `media/` | 211 | 68 |
| `routes/` | 96 | 57 |
| `security/` | 20 | 20 |
| `models/` | 33 | 9 |
| `eval/` | 29 | 1 |
| `datasets/` | 26 | 13 |
| `detectors/` | 18 | 7 |
| `converters/` | 15 | 1 |
| `state/` | 13 | 9 |
| `utils/` | 6 | 0 |
| `settings.py` | 5 | 5 |
| `auth/` | 3 | 0 |
| `labels/` | 2 | 2 |
| other | ~5 | ~2 |

The 288 `reportMissingImports` errors come from heavy third-party
packages (`torch`, `transformers`, ...) that aren't installed in a bare
shell. CI installs them via `requirements/base.txt`, so import errors
will resolve there. **Local dev needs `bash scripts/install-cpu.sh`
for clean reports.**

## Stages

Each stage is a separate PR. The "errors to fix" column counts real
(non-import) errors at the start of the stage.

| # | Scope added to `include` | Errors to fix | Status |
|---|---|---:|---|
| 0 | (no scope; config + advisory CI only) | 0 | ✅ shipped (PR #1349) |
| 1 | `utils/`, `auth/`, `plugins/`, `sync/`, `concurrency/`, `exporters/`, `labels/`, `settings_io/`, `cli.py`, `config.py` | 4 | ✅ shipped (PR #1349) |
| **2** | `settings.py`, `settings_factory.py`, `state/`, `security/` | ~27 | ⏳ next |
| 3 | `datasets/`, `detectors/`, `eval/`, `models/` | ~30 | 📋 |
| 4 | `routes/`, `converters/` | ~58 | 📋 |
| 5 | `media/` (heaviest — may need `.pyi` stubs or per-file `# pyright: ignore`) | ~68 | 📋 |
| 6 | Whole `vtsearch/` — advisory job removed | 0 | 📋 |
| 7 *(optional)* | `tests/` | TBD | 📋 |

**Stage 0 + Stage 1 landed together in PR #1349.** Subsequent stages are
separate PRs; each one bumps `include` in `pyrightconfig.json` and fixes
the real errors that the gate then surfaces.

### Stage 1 fixes (shipped)

Documenting these because the patterns are likely to recur in later
stages:

1. **Forward references that pyright can't resolve at runtime.**
   `vtsearch/labels/sources/base.py` referenced `LabelSet` only in type
   annotations; the runtime import had been removed. Fix: add a
   `TYPE_CHECKING`-guarded import so the symbol resolves for the
   type-checker without re-introducing the runtime dependency.
2. **Override signature mismatches.** `SettingsSource.save()` and
   `LabelsetSource.save()` rename the data argument to `settings` /
   `labelset` for readability. Pyright treats that as a parameter-name
   override violation against `SyncSource.save(self, data)`. Fix: make
   the base-class parameter positional-only (`def save(self, data, /)`)
   so the override contract is on type and position, not name.
3. **`__file__` is `Optional[str]`.** Namespace packages have
   `__file__ is None`, but `Path(...)` requires `str`. The
   `PluginRegistry._discover` walker hit this. Fix: explicit early-return
   when `__file__ is None` instead of letting `Path` raise on `None`.
4. **`importlib.metadata.entry_points(group=...)` requires `str`, not
   `str | None`.** Same shape as #3 — narrow the optional via an early
   return at the call site (e27bd8e). Worth checking other `entry_points`
   callers as later stages widen scope.

### Operational notes from Stage 1

- The CI workflow tees the install-step output and the gated pyright
  output into `$GITHUB_STEP_SUMMARY`. When a stage's PR fails, the
  summary tab on the run is the fastest place to see what broke — no
  need to expand the raw step log.
- Pyright pins to `1.1.408` in the workflow. Bump deliberately when
  taking a new stage so the version change is reviewable on its own,
  not bundled with error fixes.

## Configuration choices

`pyrightconfig.json`:

```json
{
  "include": ["..."],
  "exclude": ["**/__pycache__", "**/node_modules", "tests", "frontend", "scripts", "data"],
  "pythonVersion": "3.10",
  "typeCheckingMode": "basic",
  "useLibraryCodeForTypes": true,
  "reportMissingImports": "error",
  "reportMissingTypeStubs": "none"
}
```

- `useLibraryCodeForTypes: true` lets pyright infer from installed
  package code when stubs are missing — important for `torch`,
  `transformers`, `laion_clap`.
- `reportMissingTypeStubs` is silenced; we don't ship stubs.
- `tests/` is excluded for now (Stage 7).

## CI shape

Single workflow `.github/workflows/pyright.yml`:

1. **`gated` job** — installs deps, runs `pyright` (uses
   `pyrightconfig.json` `include`). Hard gate.
2. **`advisory` job** — installs deps, runs
   `pyright vtsearch/ || true`. Posts the error count in the job
   summary. Does not fail the build.

The advisory job is deleted in Stage 6.

## Non-goals (now)

- Strict mode.
- Annotating every public function.
- Type-checking `tests/`, `frontend/`, or `scripts/`.
- Wiring pyright into `./run-tests.sh` (CI-only — keeps the local
  test loop fast since pyright on a fresh repo takes ~30 s).

## Local usage

```bash
pip install pyright              # or use pre-built CI binary
bash scripts/install-cpu.sh      # so reportMissingImports passes
pyright                          # gated scope (matches CI hard gate)
pyright vtsearch/                # full-package check (matches advisory job)
```

## Backwards compatibility

This is additive — no runtime behavior changes. Type-only fixes
applied during each stage may change function signatures (e.g.
parameter renames to match base classes); those are noted in each
stage's PR description.
