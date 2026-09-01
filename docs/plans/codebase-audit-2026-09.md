# Codebase audit — September 2026 (structure & organization)

**Background.** A structure-and-organization audit was run at `0de6fb63` (dev):
six specialist reviewers over disjoint areas (vtscore core, vtscore ML,
eval/experiments, vtsearch app tier, Angular frontend, tests/tooling/docs), each
instructed to verify every claim by reading the code and to skip anything already
tracked in [`codebase-audit-2026-08.md`](codebase-audit-2026-08.md). Unlike the
August audit (defects), this one targeted **tech debt**: god modules, duplicated
logic, dead code, layering violations, and unnecessary complexity.

**Everything concrete became an issue.** 79 items are tracked as GitHub issues
(#3375–#3453) with their bodies deleted from this file, per the one-item-one-home
rule. What remains here is the umbrella: a pointer list per area so the slices
stay legible as a group. One design fork is still an open question rather than a
task, and keeps its body at the bottom, because there is nothing to file until
someone decides.

The August audit's improvement proposals remain open and complementary; several
issues below note when they should ship together with one of them.

---

## Extension safety (read before closing any "dead code" issue)

**An in-repo grep does not prove a surface is dead.** Third-party extensions
import `vtscore` symbols this repository cannot see, so "no callers here" means
unused *by us*, not unused. This rule bit the first draft of this audit: four
documented `vtscore` promises were written up as deletions on grep evidence
alone, and #3395/#3397/#3398/#3404/#3386 were rewritten to preserve every public
name once that was caught.

Before removing anything, classify it:

- **Safe to delete outright** — private (`_`-prefixed) symbols, `vtsearch/` app-tier
  internals, Angular frontend code, tests, and one-off scripts. CLAUDE.md's
  Backwards Compatibility section already licenses these.
- **Keep the name; retire the body** — anything exported from a `vtscore` package
  `__init__`, documented under `vtscore/docs/`, a plugin ABC method, a registry or
  `register_*` function, an entry-point-facing name, or a public module-level
  constant. Collapse it to a thin delegation, or mark it deprecated with an
  `[Unreleased]` entry in `vtscore/CHANGELOG.md` — never a silent removal.
- **Public-but-undocumented is still public.** A name without a leading underscore
  is importable from its module even when it is absent from `__all__` and from the
  docs (`detector_score_embedder` is the worked example, in #3386).

A genuine removal is a deliberate library break: raise it with the user first.

## Ground rules for implementer sessions

- Base on `dev`; one issue per PR; run a **full** `./run-tests.sh` before pushing.
  Regenerate the OpenAPI snapshot (`cd frontend && npm run regenerate-openapi-snapshot`)
  whenever a route or schema changes.
- Each issue carries its own difficulty, recommended model, evidence with
  file:line pointers, and constraints. Check the box here when the issue closes.
- Issues flagged in their Constraints as moving logic that
  `scripts/check-eval-app-sync.py` pins must update the `Mirror` paths and run
  `--update` **after** reconciling the harness — re-pinning without looking
  defeats the gate.
- Module splits are non-breaking at the import surface: every public name stays
  importable from its old path via a package `__init__` re-export or a shim.

**Suggested first wave** (high value, low risk): #3441 and #3434 (verified-dead
frontend code and repo hygiene), #3389/#3392/#3399 (mechanical vtscore dedup and
converter logging), #3400 (eval defaults that no longer match the shipped
algorithm), #3382/#3402/#3404. The god-module splits (#3381, #3377, #3405, #3417)
and the settings rework (#3412) are the highest-payoff items but need Opus-tier
care.

---

## Library tier — god modules & misplaced code

- [ ] #3375 — Split `vtscore/config.py` into a package along its five seams (Sonnet 5)
- [ ] #3377 — Split `vtscore/state/core.py` and centralize `DatasetContext` cache invalidation (Opus 4.8)
- [x] #3381 — Split the thresholds module into the `vtscore/training/thresholds/` package along its five seams (Opus 4.8)
- [ ] #3384 — Extract the load-progress and torch-ops subsystems out of `vtscore/media/embedder.py` (Sonnet 5)
- [ ] #3387 — Mirror the image demo-source layout for audio, video and text (Haiku 4.5)
- [ ] #3390 — Replace `labeling_progress.py`'s 15 module globals with a keyed cache dataclass (Opus 4.8)
- [ ] #3391 — Move the three pure-algorithm modules out of `vtscore/state/` (Haiku 4.5)
- [ ] #3393 — Dissolve the `vtscore/datasets/loader.py` re-export façade and its circular bottom imports (Sonnet 5)
- [ ] #3396 — Move `evt_mixture.py` out of the shipped `vtscore/training/` surface (Haiku 4.5)

## Library tier — duplication

- [ ] #3378 — Unify the two independent implementations of clip replay from `origin.params` (Opus 4.8)
- [ ] #3379 — Collapse the five copies of the clip-dict builder in `image/_demo_sources.py` (Sonnet 5)
- [ ] #3383 — Deduplicate the clipper family: tiling math, segment emission, six no-op clippers (Sonnet 5)
- [ ] #3386 — Collapse the near-synonymous embedder-resolution wrappers (Sonnet 5)
- [ ] #3389 — Deduplicate the media registries, the atomic-write ritual, and JSON label extraction (Haiku 4.5)
- [ ] #3392 — Centralize `_default_progress()` and the `ProgressCallback` alias (Haiku 4.5)
- [ ] #3394 — Extract one background-import harness shared by both import pipelines (Sonnet 5)

## Library tier — dead code & unkept promises

- [ ] #3395 — Reconcile four documented `vtscore` promises that don't match the code (Haiku 4.5)
- [ ] #3397 — Keep the resolver extension point but delete its auto-wire dance and import-error mask (Sonnet 5)
- [ ] #3398 — Stop route modules reaching past the `labelset_ops` façade for private symbols (Sonnet 5)
- [ ] #3399 — Replace `print()` error reporting in every shipped converter with logging (Haiku 4.5)
- [ ] #3401 — Declare `image_response` on the `MediaType` ABC and document both undeclared hooks (Sonnet 5)
- [ ] #3402 — Apply the sub-output disambiguators in the converted-demo emitter (Sonnet 5)
- [ ] #3404 — Small vtscore batch: `JOB_MANAGERS` coverage, registry construction, `SAVED_DATASETS_DIR` (Haiku 4.5)

## Concurrency & progress

- [ ] #3376 — Delete the legacy global dataset-progress system in favour of the per-task registry (Opus 4.8)
- [ ] #3380 — Back `AsyncJob` with a `ProgressTracker` instead of re-implementing it (Opus 4.8)
- [ ] #3382 — Route the raw staging thread through `vtsearch.threading.spawn` (Haiku 4.5)

## Layering & host seams

- [ ] #3385 — Replace the nine hand-rolled app-to-library hook seams with one typed registry (Sonnet 5)
- [ ] #3388 — Drive `PluginBase` auto-derivation from family-base opt-in instead of three hardcoded tables (Opus 4.8)

## App tier — settings

- [ ] #3412 — Generate the settings schemas from the pydantic models (Opus 4.8)
- [ ] #3413 — Delete the settings migration shims for old persisted formats (Sonnet 5)
- [ ] #3415 — Collapse the six CLI-override knobs into one declarative `AdminOverride` descriptor (Sonnet 5)
- [ ] #3416 — Give `inclusion` one owner and one clamp (Sonnet 5)

## App tier — routes, schemas, facades

- [ ] #3418 — Move `routes/projection.py`'s orchestration into a vtscore projection service (Opus 4.8)
- [ ] #3419 — Move `routes/sorting.py`'s ML pipeline logic into `vtscore/training/` (Sonnet 5)
- [ ] #3420 — Split `routes/_shared.py`: nine unrelated modules in one 866-line file (Haiku 4.5)
- [ ] #3422 — Standardize on one error envelope (Sonnet 5)
- [ ] #3425 — `schemas/`: copy-pasted validators, passthrough hooks, hand-mirrored plugin descriptor (Haiku 4.5)
- [ ] #3427 — Register one dynamic plugin route and generate its bodies at spec-build time (Opus 4.8)
- [ ] #3430 — `achievements.py`: build the response shape once, stop reaching into settings privates (Sonnet 5)
- [ ] #3432 — Collapse the CLI autodetect 2×2 matrix, keeping all four public names as shims (Sonnet 5)
- [ ] #3435 — Delete `state_proxies.py`: 375 lines of facade for one production call site (Sonnet 5)
- [ ] #3438 — Small app-tier batch: exempt prefixes as a route attribute, plus the orphan-endpoint decision (Sonnet 5)

## Eval harness & experiments

- [ ] #3400 — Make the calibration experiment defaults match the shipped algorithm (Sonnet 5)
- [ ] #3403 — `simulate_voting_iterations`: 45 positional parameters, and the two mirrors that pin it (Opus 4.8)
- [ ] #3405 — Split `voting_iterations.py`: schema tuples, trainers, per-study arm emitters (Haiku 4.5 → Opus 4.8, staged)
- [ ] #3406 — `check-eval-app-sync` is one-directional: harness-side edits never trip the gate (Sonnet 5)
- [ ] #3407 — Eight hand-rolled `load_cells` copies, and the live `bench_cells._SIDECARS` regression (Sonnet 5)
- [ ] #3408 — `run_autopilot_sweep.py` re-implements the harness vote loop against a retired configuration (Sonnet 5)
- [ ] #3409 — `scripts/experiments/calibration/` is 124 files in one flat directory (Sonnet 5)
- [ ] #3410 — `build_pile.py`: a seven-subcommand multi-tool with a 303-line dataset loader (Opus 4.8)
- [ ] #3411 — `common.py` forked seven ways; `_cells_io.py` forked twice (Haiku 4.5)
- [ ] #3414 — The Smart-indicator FP/FN cost loop is a mirror that doesn't need to be one (Opus 4.8)

## Frontend — god components & extraction seams

- [ ] #3417 — Extract browse-canvas's thumbnail store and animation controller (Opus 4.8)
- [ ] #3423 — browse-bin-popup: split the member grid, then signalize (Sonnet 5 → Opus 4.8, staged)
- [ ] #3428 — Promote `SortStateService` from anemic store to orchestrator; extract `autoSelectNext` (Opus 4.8)
- [ ] #3433 — Seven hand-rolled divider drags against one shared `PanelResizeDirective` (Opus 4.8)

## Frontend — duplication & dead code

- [ ] #3436 — The audio-audition state machine is triplicated across three Browse components (Opus 4.8)
- [ ] #3441 — Verified dead frontend code: two services, five orphan inputs/outputs, three `.sr-only` copies (Haiku 4.5)
- [ ] #3443 — Frontend utilities exist but are bypassed; helpers reimplemented per component (Sonnet 5)

## Frontend — state & idiom consistency

- [ ] #3445 — Dashboard selection: mirrored ladders, state duplicated between component and service (Opus 4.8)
- [ ] #3446 — Two projection pollers in two idioms, and three verbatim find-progress blocks (Sonnet 5 → Opus 4.8, staged)
- [ ] #3447 — Per-media-type settings preferences hand-rolled in 14 components (Opus 4.8)
- [ ] #3448 — find-view and label-view duplicate the pair-change reset and inclusion seeding (Opus 4.8)
- [ ] #3449 — `ActiveDetectorService` abandoned at 15 call sites; no dataset counterpart (Sonnet 5)
- [ ] #3450 — Four coexisting subscription-teardown idioms (Sonnet 5 → Opus 4.8, staged)

## Tests & tooling

- [ ] #3421 — `tests_lib/` is not the tier it claims; its conftest imports the app tier (Sonnet 5)
- [ ] #3424 — The two conftests are a 90% copy that has drifted in the embedding stub (Sonnet 5)
- [ ] #3426 — ~300 test assertions read global state through a conftest-installed alias layer (Haiku 4.5)
- [ ] #3429 — The vulture audit scans 23% of the Python; 13% of its whitelist is unfalsifiable (Sonnet 5)
- [ ] #3431 — `Dockerfile.image-embedders` and its GPU twin are a 90% copy (Sonnet 5)
- [ ] #3434 — Repo-hygiene batch: stale allowlists, three unreferenced scripts, `slides/Makefile` hardening (Haiku 4.5)
- [ ] #3437 — `gridenv.sh` contradicts itself: the "untracked" shim is tracked (Sonnet 5)
- [ ] #3439 — `@angular-devkit/build-angular` is an unused devDependency narrowing the audit gate (Sonnet 5)
- [ ] #3440 — The ensure-test-deps `PreToolUse` hook reads `$TOOL_INPUT` only (Sonnet 5)

## Documentation

- [ ] #3442 — Two independently written extension-authoring doc sets cover the same plugin families (Opus 4.8)
- [ ] #3444 — `vtscore/docs/packages/exporters.md` teaches the deprecated exporter contract (Sonnet 5)

---

# Open questions (not yet tasks)

Four of the five questions this audit raised have been answered by the repo
owner and became issues (or, for the punch-card, a decision to change nothing):

- [ ] #3451 — Delete the four never-implemented integration plugins (Sonnet 5). Bigger than its 865 lines: `docs/EXTENDING-plugins.md` teaches the bulk-fetch hook *from* ReCaller/PullWrest, so the guide needs replacement examples rather than deletions.
- [ ] #3452 — Find out who uses the autorun extractor/localizer surface before touching it (Sonnet 5). Kept as-is pending an answer from the external developers; #3441 is scoped so the rest of the frontend dead-code sweep lands without waiting.
- [ ] #3453 — Document the settings-source sync engine and retire its dead cross-worker layer (Opus 4.8). The capability stays — more sources are coming — so this is documentation plus the `.syncmark` layer that `workers = 1` already made unreachable.

The release punch-card stays exactly as it is; that question is closed.

What remains is one genuine design fork:

<!-- item-sep -->

- **What is `CoreConfig` for?** — `vtscore/config.py:793`

  All 14 call sites call `CoreConfig.from_settings()` ad hoc, each invoking ~18 settings getters through the app shim, so the frozen-value-object abstraction buys nothing while costing a full settings snapshot per lookup. The design comment at `config.py:793-816` still says "Until those land this class is unused at runtime" — stale for a while now.

  *The fork:* either restore the original design (build one snapshot per operation and pass it down, which is a real plumbing change) or accept that the getters won and replace `CoreConfig` with direct calls. Both are defensible; picking one is a design call, not a cleanup. The stale comment should go either way.
