# 2026-08-28 — a committed figure is *output*, and git merged it clean (#3280)

**Study:** #3280, backfilling #3267's quality-over-clicks figures.
**Cost:** a rebase, a 12-minute regeneration of every figure and the viewer, and
a second full `run-tests.sh`. Caught before merge, by reading the upstream
commit rather than by any tool.

**What happened.** #3280's branch was cut from #3279's branch, because the
curve/viewer machinery it needed lived only there and not yet on `dev`. Mid-task
#3279 merged — and it had gained a commit making the *same* change to the figures
that this branch had made independently, by a different and better route (drop
the dotted baseline rule *and* the separate marker; make click 0 each series' own
leftmost point, with the dashed segment bridging to its first trained click).

Dropping the superseded commit was easy. The trap was what came with it: this
branch had **committed PNGs and a `viewer.html`**, and those are *output of the
code that had just changed underneath them*. The two cherry-picks applied with no
conflict at all, so at that moment the branch held figures drawn by a renderer
that no longer existed in its own tree, plus a report caption describing a black
dot the new renderer no longer drew.

**Why nothing caught it.** Git compares *files*. A PNG that only one side edited
merges clean by definition, and nothing in the repo records that
`figures/cost_vs_clicks.png` is a function of `curves.py`. Every gate agreed:
`ruff` and `pyright` do not read PNGs, `check-docs.py` verified the image link
resolves — it did, to a stale image — and `run-tests.sh` passed against the
stale figures without complaint. A conflict-free merge is evidence about text,
not about whether an artifact still describes the code that made it.

**The tell, when there is one.** The prose was the only thing that broke
visibly: the report said "marked by the black dot at the far left" while the new
renderer drew no such marker, and the analyzer's own `FIGURE_CAPTIONS` had been
rewritten upstream to say something else. Two copies of the same caption, one
generated and one hand-written into `REPORT.md`, disagreed — which is the same
argument for generating the reading copy rather than hand-writing it.

Regenerating was cheap insurance and also the check: every number came back
**bit-identical** across the two renderings, so the rendering change was
confirmed cosmetic rather than assumed to be.

**Prevented?** No — still advice, and worth a control:

- **If a branch is stacked on an open PR, expect that PR to move under it**, and
  before resolving anything, read what upstream actually did. The right
  resolution here was to *drop* a commit, not to merge two versions of one idea.
  Rebasing onto a moved base is not finished when the conflicts are gone.
- **Any commit that touches a generator must regenerate that generator's
  committed output, or say why it need not.** `curves.py`, `viewer.py` and
  `analyze_startup.py` all have artifacts in `docs/experiments/`.
- A real control would record the generating commit beside the artifact — a line
  in the study's `REPORT_generated.md` naming the SHA the figures were built
  from — so a check can compare it against the last commit touching the
  generator. That is mechanically checkable and does not exist yet.

**See also:** [a fresh worktree ran the *other* worktree's
code](2026-08-07-a-fresh-worktree-ran-the-other-worktrees-code.md) and [a stale
base changed the default head](2026-08-26-a-stale-base-changed-the-default-head.md)
— the same family: the code that ran was not the code you were reading.
