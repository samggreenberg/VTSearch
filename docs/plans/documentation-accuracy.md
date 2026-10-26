# Documentation accuracy and structure

## Background

A full audit of every documentation surface in the repo (134 markdown files across
`docs/`, `vtscore/docs/`, `README`/`CLAUDE`, the plan and experiment archives, plus
mechanical sweeps for links, anchors, code-path references and orphans) produced 324
evidence-backed findings. This file tracks what is still owed; the issues listed below
own the concrete fixes.

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

## Open findings

Each finding below has its own issue. A new finding with no issue yet goes here as a bullet
until it is worth shipping separately; promote it then, and replace its body with a pointer
line.

<!-- item-sep -->

- [ ] #5289 — check-docs: `run-tests.sh`'s stage banners must match `docs/TESTING.md`'s gate table

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

- [ ] #5252 — CHANGELOG.md has readers but no release step (one ever-growing `Unreleased`)

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

- [ ] #5290 — check-docs: ban line-number anchors repo-wide, in both spellings

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->
