# #3160 — resume note

Written 2026-08-19, after §5.1 landed. **The study is complete**; this file
tracks only what is owed outside the report. Delete it when both PRs merge.

## State

- **PR #3178** (`claude/gpu-provenance-3160`) — study + pile-side code. Suite
  green post-merge with the 2026-08-18 release (job `521271`, 8728 passed).
  `REPORT.md` is complete through §5.1.
- **PR #3182** (`claude/embedding-stack-pin`) — `embedding_stack` in dataset
  `meta.json`. Suite green (job `521373`, 8736 passed).
- Issue #3160 commented and labelled `solved`.

## Housekeeping owed

- Five worktrees under `/exp/$USER/projects/` from this work: `vts-gpu-3160`,
  `vts-3160-tests`, `vts-3160-merge`, `vts-stackpin`, `vts-stackpin-tests`.
  Remove the `*-tests`/`*-merge` ones once both PRs land — `/exp` is a 50 G quota.
  Note `vts-gpu-3160` is several commits behind its branch: it was deliberately
  not pulled while arrays were reading code from it.
- Artifacts on scratch: `/expscratch/$USER/gpu-node-3160/{census,mechanism,
  cpuinfo,backend,bench,bench-rep,figures}`. Purgeable; every number that
  matters is in `REPORT.md`, but the census and mechanism JSONs are the only
  copies of measurements that cost GPU time.

## One open question, not owed by this issue

Whether transformers 4.x `use_fast=True` produces bit-identical pixels to 5.x
`backend="torchvision"`. If yes, naming the backend (#3173/#3176) closes the
version axis completely; if no, `>=4.49` still admits two answers even with the
backend named. Needs a second venv on 4.x, which was never built. Recorded on
#3173 by the #3146 session, as a question rather than a proposal.
