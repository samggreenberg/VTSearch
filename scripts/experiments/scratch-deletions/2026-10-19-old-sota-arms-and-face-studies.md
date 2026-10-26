# 2026-10-19: closed face studies, old SotA arms and leftovers (139 GB by `du`, owner-approved)

`/expscratch/sgreenberg` was at 99% (20 GB free by `df`); after these, 87% and 140 GB free. An audit
(`/expscratch/sgreenberg/logs/scratch-audit-1019*.out`, inventory `scratch-inventory-1019.tsv`) put 990 of the
1,005 GB in four directories: `state-of-the-app` 340, `fullmarks` 294, `fhibe` 267, `vts-cache` 91. Every path below
passed [the four checks](../scratch-deletions.md): its issue is closed and its report merged; `git grep` over `scripts/`, `vtscore/` and
`.claude/skills/` finds no other study's tool that reads it by default (a study's own launcher or analyzer naming its
own output does not count); no queued or running job names it (the deleting job re-checked each path); and the owner
approved the groups. The full path list is `/expscratch/sgreenberg/logs/delete-1019.tsv`, and the job's log
`delete-1019-1099753.out` has one `DELETED` line per path.

- **Closed FHIBE face studies** under `<FHIBE release>-derived/runs/` (152 paths, 90 GB):
  - #4788's arms `2026-10-11-sota-side-*` and `2026-10-11-sota-roll-*` (+ `-analysis`, `-baseline`, `-logs`,
    `-tasks.txt`, `2026-10-11-prep-{side,roll}-k{1,4}`, `2026-10-11-orient-4788`; 51 GB). **Rebuild:** the
    Reproduce block of `docs/experiments/2026-10-11-face-sideways-4788/` (`sota_face.sh` with `SOTA_FACE_ARM=side`
    and `roll`) at a86f0a498.
  - #4762's plain-arm review `2026-10-10-sota-k{1,4}-b*` (+ `-analysis`, `-baseline`, `-logs`, `-tasks.txt`, and
    the `2026-10-10-smoke-sota-*` smoke; 25 GB): the app before #4777 aligned its crops.
    **Rebuild:** `docs/experiments/2026-10-10-state-of-the-app-face/`.
  - The small closed studies' runs (~15 GB): `2026-10-10-4759*` (#4759), `2026-10-11-4779*` (#4779),
    `2026-10-11-4783*` (#4783), `2026-10-1{1,2}-4786*` (#4786), `2026-10-12-4798*` (#4798) and `2026-10-10-4776*`
    (#4776). **Rebuild:** each study's report under `docs/experiments/`.
  - **Kept:** `2026-10-10-sota-roll-*`, which the state-of-the-app skill names as the last face review's numbers for
    the shipped crops; `2026-10-10-prep-roll-k{1,4}`; and `2026-10-09-k4/identities.txt`, which `sota_face.sh`
    reads by default.
- **Old SotA arms** under `state-of-the-app/` (16 paths, 38 GB): #4496's weak-check arms `2026-10-04-weak-b*`,
  `2026-10-04-weakr-b*` and `2026-10-05-weakl-b*` (25 GB); #4492's `2026-10-04-floor-b{025,1}` (6 GB; its `-b4` went
  on 2026-10-16); #3827's GMM-cap arms `2026-10-04-b1-c{100,125}` (4.5 GB); #4936's `2026-10-15-fscale4936-binary-b1`
  (1.9 GB); `2026-10-11-centroid{4779,4783}-b1` (1.2 GB). **Rebuild:** each study's report under
  `docs/experiments/`.
- **Leftovers** (3 paths, 10 GB): `ceiling-4490` (#4490, `docs/experiments/2026-10-07-ceiling-line-4490/`),
  `structural-rerank-4169` (#4169, `docs/experiments/2026-09-30-structural-rerank-mlp-4169/`) and `cache/pip`
  (pip's download cache, which refills itself).
- **Worktrees:** `prune-worktrees.sh --apply` removed one merged, clean worktree (812 MB) and kept 47. Its name was
  in the part of the output the job did not keep.

**Not deleted, the owner's to decide:** `fullmarks/features` (185 GB; `strength_order.py`, `sota_documents.py` and
`profile_retrain.py` read it by default), the `coco_better_full_s{0..7}__dinov3_patch` embeddings in
`vts-cache/datadir/embeddings` (37 GB; re-embedding on a GPU), and `fullmarks/stage1-cell` (12 GB, deferred
2026-10-01). **Kept:** the SotA runs other tools read (`2026-10-0{5,6,8}-b*`, `band4482`, `walk4637`,
`gooddry4731`, `eg2-4837`, `s2l-4874`, `region4776`, `hold4736v60`), and the 2026-10-16 review with #4959's
scenarios. #5092 gzips their cells in place, which should return most of their ~230 GB.
