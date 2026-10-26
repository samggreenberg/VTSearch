# 2026-10-19: #5092 and #5095 gzip the kept runs' cells in place (238 GB, owner-approved; a rewrite, nothing deleted)

After [that day's deletions](2026-10-19-old-sota-arms-and-face-studies.md), `/expscratch/sgreenberg` was at 87% (886 GB used, 139 GB free); after this, 64% (650 GB
used, 375 GB free). `scripts/experiments/calibration/gzip_cells.py --apply` converted every plain cell frame
(`task_NNNN.csv` and its `__` side frames) of 103 kept run directories to `task_NNNN.csv.gz`: the SotA runs under
`state-of-the-app/` (the 2026-10-16 review and #4959's scenarios, 2026-10-03, -05, -06, the band4482, walk4637,
gooddry4731, eg2-4837, s2l-4874 arms and the rest) and FHIBE's `2026-10-10-sota-roll-*`. Each frame was read back and
compared with the original before the plain copy was removed; empty (failed) frames and the npz archives were left as
they were. Every reader on dev takes either spelling (#5092), so nothing that read these runs changes. The list is
`/expscratch/sgreenberg/gz5092-verify/convert-dirs.txt`; the logs are `convert-1100175.out` and `convert-1100237.out`.

- **`state-of-the-app/2026-10-08-b{025,1,4}`** were held back from the first passes on a mistaken reading that
  #4800 pairs its cells with `2026-10-08-b1`'s by file name. #4800 only symlinks that run's `prepare_info.json`
  and `crops/` (its prepared grid); its analyzer's optional `--reference` would pair by name, and #4800 has a note
  about it. Converted the same day under #5095 (34.1 GB -> 2.6 GB; 650 GB used, 375 GB free after; log
  `gz5092-verify/convert-5095-1101351.out`), while #4800's tasks ran on unaffected.
- **The `-common` directories** (`2026-10-06-b*-common`, 966 symlinks each into their preset's cells) were re-pointed
  to the gzipped targets, not copied. The first pass stopped on `2026-10-06-b025-common` after converting its target,
  which left those links dangling for about 35 minutes; the second pass re-pointed all 2,898.
- **Undo:** `zcat` a frame back to `.csv` and remove the `.gz` (a cell must not exist in both spellings).
