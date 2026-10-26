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

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

- **Line-number anchors are gated only in `vtscore/docs/`, and only in one spelling.** The
  2026-09-28 audit converted every remaining `file.py:NNN` and `(line NNN)` reference in the
  tracked docs (experiment reports aside) to module-and-symbol form. `scripts/check-vtscore-docs.py`
  refuses new ones, but only under `vtscore/docs/` and only the `.py:NNN` spelling (`(line NNN)`
  slipped past it). Moving the invariant into `scripts/check-docs.py`, with both spellings, would
  hold `docs/`, the plans and the READMEs to the same rule.

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

- **Two plans double as specs.** `structural-embedder.md`'s design spec (two-stage VLAD +
  RANSAC, the match-statistics classifier, why 4-DoF) belongs in the permanent docs once its open
  work ships, the way `patch-embedder.md`'s went into `docs/ARCHITECTURE.md`; and
  `user-docs-screenshots.md` is the de-facto reference for the screenshot system.

<!-- item-sep -->

<!-- item-sep -->
