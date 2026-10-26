# 2026-10-11: 254 GB (`/expscratch/sgreenberg` was at 100%, 2.8 GB free)

The volume filled while three sessions' runs were writing: #4783's arrays held their pending tasks, and
#4788's session run could not pass preflight. No cell was truncated. Two owner-approved deletes took it from
1,022 G used to 794 G (78%, 231 G free).

**#4776's raw cells, 62 GB** (deleted by the #4776 session, owner-approved): the `results/cells` of
`fhibe/<release>-derived/runs/2026-10-10-4776*` and `2026-10-10-4776prior*`, and of
`state-of-the-app/2026-10-10-{handover4776,prior4776}-b*`. The report's aggregates are in PR #4787. Its own
entry, [`2026-10-11-raw-cells-4776.md`](2026-10-11-raw-cells-4776.md), has the table and the rebuild.

**Closed studies, 192 GB, 11:17** (`/expscratch/sgreenberg/logs/scratch-delete-4788.sh`, job 970903). All four
[checks](../scratch-deletions.md) passed; the owner approved all three groups proposed:

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
