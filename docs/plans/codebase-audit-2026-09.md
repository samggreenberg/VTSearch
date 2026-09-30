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

---

## Library tier — duplication

- [ ] #3379 — Collapse the five copies of the clip-dict builder in `image/_demo_sources.py` (Sonnet 5)
- [ ] #3383 — Deduplicate the clipper family: tiling math, segment emission, six no-op clippers (Sonnet 5)
- [ ] #3386 — Collapse the near-synonymous embedder-resolution wrappers (Sonnet 5)

## Library tier — dead code & unkept promises

- [ ] #3401 — Declare `image_response` on the `MediaType` ABC and document both undeclared hooks (Sonnet 5)
- [ ] #3402 — Apply the sub-output disambiguators in the converted-demo emitter (Sonnet 5)

## Layering & host seams

- [ ] #3388 — Drive `PluginBase` auto-derivation from family-base opt-in instead of three hardcoded tables (Opus 4.8)

## App tier — settings

- [ ] #3413 — Delete the settings migration shims for old persisted formats (Sonnet 5)
- [ ] #3416 — Give `inclusion` one owner and one clamp (Sonnet 5)

## Tests & tooling

- [ ] #3421 — `tests_lib/` is not the tier it claims; its conftest imports the app tier (Sonnet 5)
- [ ] #3431 — `Dockerfile.image-embedders` and its GPU twin are a 90% copy (Sonnet 5)

---

# Open questions (not yet tasks)

Four of the five questions this audit raised have been answered by the repo
owner and became issues (or, for the punch-card, a decision to change nothing):

- [ ] #3452 — Find out who uses the autorun extractor/localizer surface before touching it (Sonnet 5). Kept as-is pending an answer from the external developers; #3441 is scoped so the rest of the frontend dead-code sweep lands without waiting.

The release punch-card stays exactly as it is; that question is closed.

What remains is one genuine design fork:

<!-- item-sep -->

- **What is `CoreConfig` for?** — `vtscore/config/core_config.py`

  All 19 call sites call `CoreConfig.from_settings()` ad hoc, each invoking ~18 settings getters through the app shim, so the frozen-value-object abstraction buys nothing while costing a full settings snapshot per lookup. The design comment at the top of `vtscore/config/core_config.py` still says "Until those land this class is unused at runtime" — stale for a while now.

  *The fork:* either restore the original design (build one snapshot per operation and pass it down, which is a real plumbing change) or accept that the getters won and replace `CoreConfig` with direct calls. Both are defensible; picking one is a design call, not a cleanup. The stale comment should go either way.
