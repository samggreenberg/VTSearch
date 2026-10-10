# Documentation accuracy and structure

## Background

A full audit of every documentation surface in the repo (134 markdown files across
`docs/`, `vtscore/docs/`, `README`/`CLAUDE`, the plan and experiment archives, plus
mechanical sweeps for links, anchors, code-path references and orphans) produced 324
evidence-backed findings. This file tracks what is still owed. The issues listed below
own the concrete fixes; everything under "Open findings not yet promoted" is recorded
here because it has no issue yet.

**The systemic cause is measurable, and it is not carelessness.** Most of the doc set was
written in one batch on 2026-07-20 (the vtscore-extraction docs drop). Since then 557
commits landed, 168 of them touching `vtscore/` code — against 20 touching `vtscore/docs/`.
Docs are written in bursts and never revisited, because nothing ties a code change to the
prose that describes it.

Three things follow from that, and they shape how this work should be done:

- **Inventory drift dominates.** The single most repeated defect across all 15 audit areas
  was a hand-maintained list of registry contents (embedders, plugin families, exporters,
  demo datasets, env vars, settings keys) disagreeing with the code and with the other nine
  copies of the same list. This is a generation problem, not ten editing mistakes.
- **The mechanical defects are mechanically detectable.** Dead links, dead anchors, dead
  file paths and leaked absolute paths accounted for roughly 30 findings. `scripts/check-docs.py`
  now gates that whole class in `run-tests.sh`; what remains below is the part no invariant can
  check, because it needs a human to know what the prose *should* say.
- **Prefer invariants over generation over pinning.** The repo already has all three shapes
  — `wiring-check.py` (invariant), the OpenAPI snapshot (generation), `check-eval-app-sync.py`
  (digest pinning). A noisy gate gets `--update`'d blindly, a failure mode CLAUDE.md already
  names for the eval pins, so reach for the cheapest shape that catches the class.

Fixing individual docs without the first two bullets buys about six weeks.

## Open findings not yet promoted

Grouped by document. Each bullet is a defect with no issue of its own yet; promote one to
an issue (and replace the body here with a pointer line) when it becomes worth shipping
separately.

<!-- item-sep -->

- **The `run-tests.sh` gate list is still hand-maintained in two places.** The "What
  `run-tests.sh` gates" table (now in `docs/TESTING.md`) and the script's own usage header each
  restate the chain by hand; the table's stage-2 row had drifted again (it predated
  `scripts/check-frontend-gate.py`) when the 2026-09-28 audit caught it. An invariant check —
  the set of stage banners in `run-tests.sh` must match the table's rows — is a few lines inside
  `scripts/check-docs.py` and cheaper than generating the table.

<!-- item-sep -->

- **CHANGELOG.md has readers but no release step.** It is maintained and linked from
  `CLAUDE.md`, but `docs/RELEASE.md` never mentions it, so nothing closes an `Unreleased` section
  at a release: everything since the library extraction sits under one `## Unreleased`. Decide
  whether the release runbook cuts a dated section; if so, add the step.

<!-- item-sep -->

- **Line-number anchors are gated only in `vtscore/docs/`, and only in one spelling.** The
  2026-09-28 audit converted every remaining `file.py:NNN` and `(line NNN)` reference in the
  tracked docs (experiment reports aside) to module-and-symbol form. `scripts/check-vtscore-docs.py`
  refuses new ones, but only under `vtscore/docs/` and only the `.py:NNN` spelling (`(line NNN)`
  slipped past it). Moving the invariant into `scripts/check-docs.py`, with both spellings, would
  hold `docs/`, the plans and the READMEs to the same rule.

<!-- item-sep -->

- **`patch-embedder.md` is a shipped plan kept alive as a spec.** Everything it owes has
  shipped or is explicitly out of scope, but its "living spec" (the V3 trio, score precedence,
  per-detector embedder type) is cited by ~20 source docstrings and exists nowhere in the permanent
  docs. Fold it into `docs/ARCHITECTURE.md` (or `vtscore/docs/concepts.md`), repoint the citations,
  then delete the plan. `structural-embedder.md`'s design spec (two-stage VLAD + RANSAC, the
  match-statistics classifier, why 4-DoF) wants the same treatment once its open work ships, and
  `user-docs-screenshots.md` is the de-facto reference for the screenshot system.

<!-- item-sep -->

- **Small code-side drift the doc audit found but did not fix.** `slides/render.sh`: its usage line
  omits `png` / `--no-pageno`, and the comment above the `PIPESTATUS` check says the script has no
  `pipefail` though it sets `set -euo pipefail` (so the check may be dead). `docker/Dockerfile*`:
  the "OMP/MKL thread limits are already set in app.py" comment is misleading — `app.py` overwrites
  them with the CPU allocation. `slides/figs/ui-autopilot.png` is referenced by nothing.
  `scripts/experiments/pile/launch_pile.sh`'s header still shows a `--force` rebuild of `vg_scale`,
  which can no longer be rebuilt; `scripts/experiments/max_patch/queue_all.sh` defaults
  `NCELLS=240` against a 288-cell grid. `docs/user/screenshots.manifest.ts` pins
  `settings-appearance` to the Solo-media-type section though the shot shows the Appearance pane.
  `frontend/src/app/models/projection.models.ts` hand-writes three types the generated client
  already has (FRONTEND.md records it as a known exception). `vtscore/eval/label_curve_main.py`'s
  docstring uses `flowers102_s`, which is not a registered demo; `vtscore/exporters/__init__.py`
  still labels its registry "labelset exporter"; `DetectorRegistryCreateRequestSchema.trainable` is
  accepted and never read; the labelset-source routes name a detector-id path parameter
  `detector_name`.
