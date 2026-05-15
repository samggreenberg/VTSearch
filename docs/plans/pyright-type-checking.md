# Pyright Type-Checking

Adopt **pyright in basic mode** as a hard CI gate across the `vtsearch/`
package. Rolled out in stages so each PR stays reviewable.

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

| # | Scope added to `include` | Errors to fix |
|---|---|---:|
| 0 | (no scope; config + advisory CI only) | 0 |
| **1** | `utils/`, `auth/`, `plugins/`, `sync/`, `concurrency/`, `exporters/`, `labels/`, `settings_io/`, `cli.py`, `config.py` | 4 |
| 2 | `settings.py`, `settings_factory.py`, `state/`, `security/` | ~27 |
| 3 | `datasets/`, `detectors/`, `eval/`, `models/` | ~30 |
| 4 | `routes/`, `converters/` | ~58 |
| 5 | `media/` (heaviest — may need `.pyi` stubs or per-file `# pyright: ignore`) | ~68 |
| 6 | Whole `vtsearch/` — advisory job removed | 0 |
| 7 *(optional)* | `tests/` | TBD |

**Stage 0 + Stage 1 land in the same PR** (the one that introduces this
plan). Subsequent stages are separate PRs.

## Configuration choices

`pyrightconfig.json`:

```json
{
  "include": ["..."],
  "exclude": ["**/__pycache__", "**/node_modules", "tests", "frontend"],
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
