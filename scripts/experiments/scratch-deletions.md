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

## 2026-10-16: 55 GB after the volume filled overnight (owner-approved)

`/expscratch/sgreenberg` (1 TB) hit 100% at ~04:40, during #4951's State of the App run. Every job that wrote
after that failed, and the live app's data dir shares the volume. The owner approved four groups at ~06:30;
by `df`, the volume went from 256 KB free to 55 GB free.

- **#4951's partial seeds 45–49** (18,856 cell files, a few GB): the cells of the seeds that were still running
  when the disk filled, in `state-of-the-app/2026-10-16-{b025,b1,b4}/results/cells` and `-b1/ceiling/results/cells`.
  16 of them were empty, cut short or NUL-filled. The review analyses seeds 0–44, which were complete and checked.
  **Rebuild:** not needed. More seeds would come from re-running those indices with the review's launcher at its
  commit (`7b3a3d1e3`).
- **Three floor-era State of the App run dirs** (13 GB): `state-of-the-app/2026-09-30-walk2` (5.8 GB, the
  2026-10-01 Binary Photo review's run), `2026-10-04-b1-pframes` (4.2 GB, #4492's precision-frame capture) and
  `2026-10-04-floor-b4` (3.0 GB). Their reports are committed. No tool on dev names them.
  **Rebuild:** re-run the review at the commit its report names (`state_of_app/launch.sh`, hours).
- **The #4778 and #4780 FHIBE study runs** (28 dirs, ~38 GB by `du`):
  `fhibe/.../runs/2026-10-11-4778{weak,end}-roll-*` and `2026-10-12-4780{arc,fn}-roll-*`, each study's run,
  `-analysis`, `-baseline` and `-logs` dirs. Both studies
  are merged (#4796 and #4802; #4806), and their reports and figures are in `docs/experiments/`. No tool on dev
  names them. **Rebuild:** their launchers in `scripts/experiments/fhibe/`, from the reports' provenance.
- **23 merged worktrees** (10.7 GB), by `scripts/grid/prune-worktrees.sh --apply`. Each was clean, unused for
  24 h, on a branch merged into dev (or a detached `-tests` checkout with no job). The branches themselves are
  kept.

**Why it filled:** a review run's cells are written uncompressed, ~0.6 GB a seed for each preset's directory.
#4951 took as many seeds as fit by morning, 85 GB by 04:40, on a volume that held ~915 GB before it (by
subtraction). Check `df` before a run sized by its deadline rather than its seed count.

## 2026-10-15: 6.1 GB, the recorded ratio-test calls from #4515 (owner-approved)

- `/expscratch/sgreenberg/ratio-4515/rec-s/calls/` (8 `call_<k>.npz`): real 1,000-page
  `ratio_test_matches` calls (template, uint8 candidates, output pairs), recorded on 2026-10-15 to
  prove #4933's rewrite returns dev's pairs. #4933 is merged and #4515 is `solved`; only the
  bench scripts beside them read these files.
- **Rebuild:** `sbatch --export=ALL,W=<a worktree of dev> /expscratch/sgreenberg/ratio-4515/record.sbatch`
  re-records them from `profile_retrain.py`'s tier-s CPU loop in ~6 min. The rest of `ratio-4515/` is
  kept: the scripts, `calls.json` (per-call shapes and timings) and the `prof/` step logs.

## 2026-10-14: 24 GB, the torch-2.6 venv backups from #4838 (owner-approved)

#4838 moved both GRID venvs from torch 2.6.0+cu124 to 2.14.1+cu126 by a rename swap, so the old venvs were left
beside them. Both are now deleted:

- `/expscratch/sgreenberg/vtsearch-venv.t26` (11 GB), the app venv until 2026-10-13 22:52.
- `/exp/sgreenberg/projects/VTSearch/.venv.t26` (13 GB), the experiment venv until 20:20. Deleting it
  took `/exp/sgreenberg` from 82% to 58% of its 50 GB.

Why it was safe:
- #4838 is `solved` and PR #4847 is merged.
- After the swap, the new venvs passed the GRID suite twice and the post-swap checks. A 48-cell SotA A/B gave identical
  picks and F-beta under both torches.
- No tool names a `.t26` path; `VTS_VENV` and gridenv.sh's default name the live venvs.
- No running job used them. The running #4384 tasks were checked through `/proc/<pid>/environ`: they had the live venv.

How to rebuild one (about 20 min): make a venv with the `python/3.12.3` module and
`pip install --no-deps -r` the saved freeze list. Torch comes from the cu124 index and cuML from pypi.nvidia.com.
The lists, taken just before the swap copies, are `/expscratch/sgreenberg/venv-4838/torch26-freeze-{app,exp}.txt`.
The editable `vtsearch` line points at the deploy clone (app) or the `/exp` checkout (exp).

## 2026-10-13: 2.9 MB of test fixtures in the live `vtsearch-data` (#4706, owner-approved)

Until #4706's fix (PR #4823), GRID suite runs inherited `VTSEARCH_DATA_DIR` from `~/.bashrc`, and that names the live app's
data. Every test that wrote under `config.DATA_DIR` wrote there, so these were deleted:

- **336 `autodetect_results_*.json`** at the top level. They are the `server_json_file` exporter's default output, and
  every one held fixture results (`cat_meow`, `dog_bark`).
- **192 `local_uploads/local_folder_*`** dirs, each holding exactly `f/a.wav`.
- **2 `staging/stage_*.pkl`**, from a suite run at 03:53 that day.

Why it was safe:
- Nothing reads them. The exporters only write `autodetect_results_*`. No JSON or pickle in `vtsearch-data` names
  `local_uploads` or either stage id. Staging pickles are transient files for the combine flow.
- No app job was running.
- Each file was re-checked as a fixture just before the `rm`.

`local_uploads/`, `staging/` and everything else in `vtsearch-data` stayed. That includes the registries, `detectors/`,
`models/`, the settings files, and the empty `ESC-50-master/` that suite runs create. Nothing to rebuild.

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

## 2026-10-11: 254 GB (`/expscratch/sgreenberg` was at 100%, 2.8 GB free)

The volume filled while three sessions' runs were writing: #4783's arrays held their pending tasks, and
#4788's session run could not pass preflight. No cell was truncated. Two owner-approved deletes took it from
1,022 G used to 794 G (78%, 231 G free).

**#4776's raw cells, 62 GB** (deleted by the #4776 session, owner-approved): the `results/cells` of
`fhibe/<release>-derived/runs/2026-10-10-4776*` and `2026-10-10-4776prior*`, and of
`state-of-the-app/2026-10-10-{handover4776,prior4776}-b*`. The report's aggregates are in PR #4787. Its own
section, below, has the table and the rebuild.

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
