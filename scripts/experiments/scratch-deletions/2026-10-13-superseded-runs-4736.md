# 2026-10-13: 1 GB (dead runs from #4736, owner-approved)

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
