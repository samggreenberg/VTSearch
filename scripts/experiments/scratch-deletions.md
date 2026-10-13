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

## 2026-10-13: 93 GB (#4786's raw cells, owner-approved)

#4786's issue is `solved` and its report merged with PR #4810. #4736 priced its hold on three of these COCO sets
(`line4786all-b025/-b1`, `line4786thin-b4`); its report merged with PR #4819, and its follow-up #4816 plans a run of its
own. No tool reads them by default (`git grep` over `scripts/` and `vtscore/`: only a comment in `hold_4736.sh`), and no
job used them. Only each run's `results/cells` went; `logs/`, `grid_shape.json` and the `prepare_info.json` / `crops`
symlinks stay, and the analyses (`fhibe/<release>-derived/runs/2026-10-11-4786-analysis`) were kept. 874 G used after
(86%, from 95%).

| directory (each `results/cells`) | GB | what |
|---|---:|---|
| `fhibe/<release>-derived/runs/2026-10-11-4786-roll-k{1,4}-b{025,1,4}` | 24.2 | today's face sessions with the line counts (`cb26dcfae`) |
| `…/2026-10-11-4786thin-roll-k{1,4}-b4` | 7.8 | the beta-4 arm (`4131eef90`) |
| `…/2026-10-11-4786all-roll-k{1,4}-b{025,1}` | 16.4 | the beta-1/4 and beta-1 arm (`8338a115c`) |
| `…/2026-10-12-4786{mctl,marm}-roll-k{1,4}-b{025,1,4}` | 29.0 | today and the rule re-run on dev after #4799 (`386a4e211`) |
| `…/2026-10-11-4786{hc,floor}-roll-*` | 0.01 | dead arms (a superseded gate's 2 smoke cells; a cancelled floor arm) |
| `state-of-the-app/2026-10-11-line4786-b{025,1,4}` | 7.5 | today's COCO Binary sessions with the line counts |
| `…/2026-10-11-line4786thin-b4`, `…/2026-10-11-line4786all-b{025,1}` | 7.5 | the COCO arm |
| `…/2026-10-11-line4786{hc,floor}-b*` | 0.04 | dead arms (15 cells of the cancelled floor arm) |

Rebuild: the report's *Reproduce* block (`docs/experiments/2026-10-12-beta4-line-4786/REPORT.md`): `sota_face.sh` with
`SOTA_FACE_TAG=4786 SOTA_FACE_ARM=roll CALIB_LINE_COUNTS=1` (today) or the arm's tag, and `line_4786.sh`
(`LINE4786_ARM`); today's line is `CALIB_HEAD_COUNT_BADS=0` on any later code.

## 2026-10-13: 1 GB (dead runs from #4736, owner-approved)

Two run sets #4736 superseded on the night it ran. Its issue is `solved`; its report merges with PR #4819.
Neither is read by any tool (`git grep` over `scripts/` and `vtscore/`), and no job used them. The report's
evidence stays: the 60-vote arm (`2026-10-12-hold4736v60-b{025,1,4}`) and the Region guard
(`2026-10-12-hold4736region-b1`). The empty `…region-b025/-b4` and `…v60region-*` scaffolds, made by
`hold_4736.sh dirs`, stay too.

| directory (whole run dir) | GB | what |
|---|---:|---|
| `state-of-the-app/2026-10-12-hold4736-b{025,1,4}` | 0.92 | #4736's first arm, cut at 30 votes: it picked differently from vote ~16 (the prompted check's budget gate), so it was cancelled at ~60% and re-run at 60 votes |
| `state-of-the-app/2026-10-12-line4786smoke4736-b{025,1,4}` | 0.05 | 12 cells re-run on dev c4ad866e6 to check #4786's sessions were today's (they were, click for click) |

Rebuild: the first is `hold_4736.sh` with `HOLD4736_STEPS=30` (its dirs are then `…hold4736v30-b*`); the
smoke is `line_4786.sh` with `LINE4786_ARM=smoke4736 LINE4786_SEEDS=1` and `launch.sh pack 5,10,100,120`.

## 2026-10-11: 62 GB (`/expscratch/sgreenberg` was at 100%)

At 10:49 the volume had 2.8 G free and a #4777/#4788 run needed 25 G. The owner
approved deleting #4776's raw cells (its issue is `solved`; its report, with every
aggregate the analysis wrote, is in PR #4787, not yet merged). Only each run's
`results/cells` went; `logs/`, `grid_shape.json` and the `prepare_info.json` /
`crops` symlinks stay, and the analyses (`…-4776-analysis`, `…-4776prior-analysis`,
`/expscratch/sgreenberg/handover-4776/coco-*`) were kept. The only code references
are `handover_4776.sh`'s default **output** names. 980 G used after (96%).

| directory (each `results/cells`) | GB | what |
|---|---:|---|
| `fhibe/<release>-derived/runs/2026-10-10-4776-k{1,4}-b{025,1,4}` | 23.3 | #4776 shadow: today's face sessions with the centroid's Test past the quota |
| `fhibe/<release>-derived/runs/2026-10-10-4776prior-k{1,4}-b{025,1,4}` | 23.8 | #4776 arm: the head started from the Goods' centroid |
| `state-of-the-app/2026-10-10-handover4776-b{025,1,4}` | 7.5 | #4776 shadow, COCO Better Binary |
| `state-of-the-app/2026-10-10-prior4776-b{025,1,4}` | 7.5 | #4776 arm, COCO Better Binary |

Rebuild with the report's *Reproduce* block
(`docs/experiments/2026-10-11-face-handover-4776/REPORT.md`): the face runs are
`sota_face.sh` with `SOTA_FACE_TAG=4776` (`CALIB_HANDOVER_SHADOW=1`) or `4776prior`
(`CALIB_HEAD=linear_svm_prior`), ~4 h on four V100 packs; the COCO runs are
`state_of_app/handover_4776.sh`, ~6 h at 66 cells. Before #4785 the shadow ran on
the plain SVM's head; re-running today on single-vector data needs
`CALIB_HEAD=linear_svm` to reproduce it.

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
