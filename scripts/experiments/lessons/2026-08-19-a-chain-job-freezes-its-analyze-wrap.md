# A chain job freezes its analyze step at the moment the CHAIN fires, not when the arrays end

**Study:** #2808 linear-head convergence. **Cost:** near-miss — caught by
inspection minutes after launch; would have produced a report naming the wrong
arm as production.

The #2808 launcher submits `prepare`, then a `chain` job (`--dependency=afterok`)
that computes the cell count and submits the arm arrays *plus* the analysis step.
That shape is right: it survives a disconnected laptop, which was the point.

What is easy to miss is **when the analysis step's environment is fixed**. The
chain job runs `sbatch --wrap="… export SPIKE_ARMS=… && python analyze_spikes.py"`.
The wrap is a **string, captured at submission time**. Prepare finished in ~2
minutes because it read a pre-embedded pile instead of embedding anything, so the
chain fired almost immediately — and a fix pushed to the branch a minute later
reached the *worktree* (which the arrays read at run time) but **not** the
analysis step's already-frozen wrap.

The fix was to `scancel` the analysis job and resubmit it with the same
`--dependency` and a corrected environment. That works, but only because someone
looked. Nothing errors: the analysis runs, produces a report, and names the wrong
arm as production.

**The generalisation:** a pushed fix reaches jobs that read the tree *at run
time* and never reaches jobs whose behaviour was baked into a `--wrap` string at
*submit* time. In a chained pipeline those two categories exist side by side and
look identical in `squeue`.

**Status: advice, not prevented.** The mechanically checkable part — that the
analysis step names its arm roles explicitly — is now a code guard in
`analyze_spikes.py` (see the sibling lesson). The general "your fix did not reach
the frozen wrap" hazard is not checkable; the practice is to treat any
`--dependency` job as *already launched* for the purpose of deciding whether a
late edit lands, and to re-submit rather than assume.
