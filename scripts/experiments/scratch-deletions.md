# `/expscratch/$USER` deletions log

`/expscratch` keeps no snapshots and has no trash, so **a delete there is final**.
This file records what was deleted, when, why it was safe, and how to rebuild
it, so a later session does not go looking for an input that is gone. Add a
dated section per cleanup, newest first.

Before deleting a directory, the checks (`GRID-PLAYBOOK.md`, and the
2026-09-18 lesson that a tool's *default argument* counts as reading a path):

1. the study's issue is closed or `solved`, and its report is merged;
2. `git grep` over `scripts/` and `vtscore/` finds no tool whose default reads the
   directory. A launcher's default *output* location is fine; an input is not;
3. no running job writes or reads it;
4. the owner approves the list.

## 2026-10-11: 254 GB (`/expscratch/sgreenberg` was at 100%, 2.8 GB free)

The volume filled while three sessions' runs were writing: #4783's arrays held their pending tasks, and
#4788's session run could not pass preflight. No cell was truncated. Two owner-approved deletes took it from
1,022 G used to 794 G (78%, 231 G free).

**#4776's raw cells, 62 GB** (deleted by the #4776 session, owner-approved): the `results/cells` of
`fhibe/<release>-derived/runs/2026-10-10-4776*` and `2026-10-10-4776prior*`, and of
`state-of-the-app/2026-10-10-{handover4776,prior4776}-b*`. The report's aggregates are in PR #4787.

**Closed studies, 192 GB, 11:17** (`/expscratch/sgreenberg/logs/scratch-delete-4788.sh`, job 970903). All four
checks above passed; the owner approved all three groups proposed:

| path | GB | study | why it is safe |
|---|---:|---|---|
| `acq-sweep-4428` | 68 | #4428 | closed; report `docs/experiments/2026-10-02-acquisition-sweep-4428/` |
| `p-aware-acq-4409` | 27 | #4409 | closed; report `docs/experiments/2026-10-01-acquisition-fbeta-4409/` |
| `labels-4452` | 50 | #4452 | closed; shipped, and later reports cite it (no report of its own) |
| `presets-4448` | 21 | #4448 | closed; shipped, and later reports cite it (no report of its own) |
| `fhibe/<release>-derived/runs/2026-10-09-k{1,4}/results/cells` | 13 | #4699 | report merged (`2026-10-09-fhibe-baseline-4699`) |
| `.../runs/2026-10-09-4731-{ctl,dry16,dry16q16}-k1/results/cells` | 5.9 | #4731 | report merged |
| `.../runs/2026-10-10-4732-k{1,4}/results/cells` | 3.3 | #4732 | report merged |
| `.../runs/2026-10-10-4743-{app,ctl}-k1/results/cells` | 4.3 | #4743 | report merged |
| `.../runs/2026-10-10-repro-sota-*` | 0.1 | #4777 | its check is recorded in that report |

**Kept on purpose:** in `2026-10-09-k{1,4}`, `identities.txt`, `results/prepare_info.json` and
`results/crops`, which `fhibe/sota_face.sh` and `sota_face_analyze.sh` read by default (the face review's
bench and prepared grid). Every FHIBE run dir's `logs/` and `grid_shape.json`, and all analysis dirs.

Rebuild: a FHIBE run's cells come back from `fhibe/launch.sh` (or `sota_face.sh`) at the report's commit, with
the same identity file. The sweeps' drivers are committed in their report dirs.

## 2026-10-01: 91 GB (`/expscratch/sgreenberg` was at 100%)

The volume filled at about 09:00 and killed a running job (#4415's arm C). The
deletes below took it from 500 G used to 403 G (81%).

**Superseded experiment cells (#3928), 09:10.** These went before the audit,
to unblock running jobs; no tool's default reads any of them.

| path | about | why |
|---|---:|---|
| `stage1-v50-3928/cell-{m1-compact,m1-float,m2-fit-spods-staver,m2-fit-t800-ucsf,v1asset}` | 1.4 GB | M1/M2 and asset-pricing cells; their eval CSVs are committed in `docs/experiments/2026-09-30-fullmarks-tiled-stage1-app-3928/measurements/` |
| `fullmarks/stage1-cell/tier-s/{raw,d256,d128}`, `tier-m/{d256,d128}` | ~5.6 GB | widths the app does not use; measured in `docs/experiments/2026-09-18-fullmarks-stage1-3928/` |

Rebuild any of them with `scripts/experiments/fullmarks/stage1_cell.py build`
(the commands are in those reports).

**Closed studies (owner-approved audit groups B and C), 10:15.** All 19 issues
are closed and their reports are merged. The only code references are
launchers' default **output** directories:

| directory | GB | issue |
|---|---:|---|
| `size-vs-size-4160` | 18.5 | #4160 |
| `atlas-3595` | 13.4 | #3595 |
| `overtrain-3945` | 9.0 | #3945 |
| `logreg-4219` | 8.3 | #4219 |
| `logreg-4114` | 7.2 | #4114 |
| `gp-grid-3959` | 6.5 | #3959 |
| `drystop-4222` | 5.5 | #4222 |
| `textgood-4222` | 4.5 | #4222 |
| `div-4197` | 3.4 | #4197 |
| `provenance-4256` | 1.8 | #4256 |
| `progression-4184-h0.05` | 1.6 | #4184 |
| `stall-3853` | 1.3 | #3853 |
| `maxiter-3839` | 1.2 | #3839 |
| `pframes-4220` | 1.2 | #4220 |
| `anchem-3825` | 1.0 | #3825 |
| `logreg-4213` | 0.9 | #4213 |
| `prefetch-3896` | 0.9 | #3896 |
| `gmm-3585` | 0.8 | #3585 |
| `guard-4303` | 0.8 | #4303 |

**Kept on purpose:**
- **Read by another tool's default:**
  - `vlm-3720`: the pile annotation scripts read its slates and queues;
  - `abres-3840`: `gmm_init/launch_3839.sh` reads its `AB_OFF` default;
  - `progression-4184`: `launch_textgood_4222.sh` reads its `PREPARE_SRC`.
- **The dataset, the Pile, the venv:** the FullMarks corpus images, `vts-cache`,
  `vtsearch-venv`, and `vtsearch-data` (which holds the detectors).
- **`keep/`.**
- **Active work in other sessions:** `born-digital`, and today's
  `state-of-the-app` runs.
- **The owner deferred:** `fullmarks/features` (35 GB, the tier-`m` SIFT cache),
  the rest of `fullmarks/stage1-cell`, `state-of-the-app/2026-09-26`, and
  `fullmarks/raw`.
