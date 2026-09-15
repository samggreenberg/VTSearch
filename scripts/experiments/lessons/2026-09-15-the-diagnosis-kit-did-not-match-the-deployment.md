# 2026-09-15 — the diagnosis kit did not match the deployment (#3853)

**What broke.** #3860 shipped three stall instruments, a launcher change
that writes the app log to a file, and a driver that replays the SPA's
per-vote chain. On the morning the capture was supposed to happen, none of
the three could have produced a usable capture on the reviewer's instance:

1. The deployed `~/.local/bin/vtsearch` was a *copy* of the repo launcher
   from before #3860. It set no `VTSEARCH_LOG_FILE`, so the instruments were
   armed and writing to a tmux pane that nobody persists. The runbook said
   "pull dev and launch as usual"; pulling dev does not update a copied
   launcher.
2. The driver registered `patch_semantic` detectors and sent region boxes.
   The reviewer labels ~500-image SigLIP slates through a plain semantic
   text-query detector. The offline reproduction in #3860 therefore
   exercised the region-vote retrain, a path the felt stall never touched.
3. The driver built `GET /api/detectors/<name>` URLs without percent-encoding.
   Every real detector name has spaces, so all three per-vote detector GETs
   failed client-side with `status 0` — and the summary still printed
   healthy percentiles for them, because a 0 ms failure is a fast request.

**What it cost.** About an hour of a half-day window, and one launch of the
driver whose numbers had to be thrown away.

**What now prevents it.**

- *Advice, not a control:* after changing `scripts/slurm/vtsearch-slurm.sh`,
  re-copy it to `~/.local/bin/vtsearch` on the GRID (there is a dated backup
  convention, `vtsearch.bak.YYYYMMDD`). The launcher prints `>>> App log:
  <path>` on start; if you do not see that line, the copy is stale.
- *Prevented:* `drive_labeling.py` now percent-encodes paths and takes
  `--embedder-type` / `--text-query`; the kit README says to match the
  reviewer's shape and shows the exact invocation.
- *Advice:* a driver's per-endpoint summary should count `status 0` rows as
  failures in the headline, not only in the "failed requests" footer. A
  fast failure is indistinguishable from a fast success in a percentile.
