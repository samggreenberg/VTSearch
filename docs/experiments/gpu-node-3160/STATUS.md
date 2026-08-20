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

## Done: the two AVX-512 bands were rebuilt (2026-08-20)

`vg_box_medium` and `vg_box_large` `siglip2_l` cells rebuilt under the pin on the
originals' card and swapped into the shared pile; `vg_box_small` was rebuilt as a
control, came back **bit-identical**, and had only its sidecar upgraded. All three
bands now read `Tesla V100-SXM2-32GB-LS / rack8n06 / AVX2` in
`build_pile.py --provenance`; `--verify` passes on all 21 cells; the manifest is
refreshed. Superseded cells and their original sidecars are in
`/expscratch/$USER/gpu-node-3160/pre-3160-backup/`.

Full account in `REPORT.md` §6.1, including the two near-misses (a rebuild on the
wrong card, and a scan regeneration that would have re-selected categories).

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
