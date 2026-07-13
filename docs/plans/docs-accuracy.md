# Docs Accuracy

**What this is:** The residue of a seven-reviewer documentation sweep
(2026-07-07) that cross-checked ~17k lines of project docs against the code.
Nearly all of its ~23 correctness findings and coverage gaps were already
implemented by follow-up PRs (the async learned-sort docs, the auth rewrite,
the EXTENDING `_do_load`/`_do_save` corrections, the `ML.md`/`DEPLOYMENT.md`/
`ARCHITECTURE.md` fixes, the API context-headers section, etc.), and the one
concrete code-example drift it found — the `load_media_data` signature in
`EXTENDING-media.md` — is fixed. What remains is a single soft onboarding gap.

Items are named (stable labels, never renumbered) and separated by
`<!-- item-sep -->` sentinels; when you ship a slice, delete only your item's
own lines and leave the sentinels intact (see the plan-file policy in
`CLAUDE.md`). When this item ships and nothing is left, delete this file.

---

<!-- item-sep -->

- **README quick-start (judgment call)** — `README.md` has no short
  "install → build frontend → run" quick-start near the top; a new reader has
  to assemble the three commands from scattered sections. **Fix:** add a
  3–4 line quick-start (`bash scripts/install.sh` → `cd frontend && npm install
  && npm run build:prod` → `python app.py --local`) above the deeper content.
  Verify against the README's current top matter first — if a good-enough
  getting-started block already exists, close this instead of duplicating it.
  **Files:** `README.md`.
