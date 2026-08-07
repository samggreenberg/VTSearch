# Experiment ops: incident log

Things that went wrong while running eval experiments, what they cost, and what
now prevents them. **Append when something breaks — do not rewrite history.**

This file exists because the same class of mistake kept recurring: each study
diagnosed its own failures well, explained them in a summary, and then the next
study made a variant of the same one. An explanation that lives only in a
conversation is not a control.

Two companions:

- **`GRID-PLAYBOOK.md`** — SLURM *resource* practice (memory sizing, QOS caps,
  chunking allocations). Read before sizing a sweep.
- **`preflight.sh`** — the subset of these lessons that is mechanically
  checkable, enforced rather than advised. Run before submitting arms.

## How to add an entry

Keep it short and keep the cost in it — the cost is what makes the next person
take it seriously. State plainly whether it is now *prevented* or still only
*advice*, because "we learned X" without a control means it will happen again.

```markdown
## YYYY-MM-DD — #ISSUE short name
**Cost:** ~Nh
**What broke:** one or two sentences, mechanism not blame.
**Now prevented by:** preflight check N / code change / nothing (still advice).
```

<!-- entry-sep -->

## 2026-08-05 — #2841 mix-in schedule

**Cost:** ~7h across a day, most of it overnight.

**Two grids shared one experiment dir.** A binary run (300 votes, 26 categories)
and a region run (200 votes, 14 categories) were both pointed at
`CALIB_EXP=/exp/$USER/mixin-2841-long`, so both wanted `results-ab/prod/cells`.
The resume logic saw 304 cells where its grid expected 84, concluded the arm was
complete, and aborted the whole batch at 00:53. Nobody was reading its log, so
the region run simply did not happen overnight. The "fail loudly" check added
hours earlier is the only reason this was a clean no-op instead of two grids
silently mixed in one directory and analysed as one.
→ **Prevented by:** preflight check 1.

**`df` on the wrong mount.** `/exp` showed 394G free; `/exp/$USER` was its own
50G mount at 100%. ~950 cells died mid-write over ~7 minutes and I reported the
volume as roomy in the meantime.
→ **Prevented by:** preflight check 2 (stats the path, not its parent).

**Zero-byte cells are invisible to resume.** The dead cells left 0-byte CSVs,
which count as "present", so the resume pass skipped exactly the cells that
needed re-running, and the analyzer then crashed on the first one.
→ **Prevented by:** preflight check 3; the analyzer now counts unreadable cells
out loud instead of dying or dropping them.

**sbatch writes to stderr on success.** `--parsable` job id capture with `2>&1`
folded in an informational `Set partition to cpu` line, so a *successful*
submission looked refused and two arms were silently skipped. The first version
of the "fail loudly" fix introduced this while fixing a different silent
failure.
→ **Prevented by:** capture stderr separately; validate the id is numeric.

**A pre-commit hook that rewrites files fails the commit.** `ruff format`
reformats, exits non-zero, and the commit does not happen. Piping the commit
through `| tail` hides it, the following `git push` succeeds having pushed
nothing, and the GRID then runs the *previous* commit — twice this happened, and
once a full 7600-test suite ran against the wrong code.
→ **Still advice:** check the commit's own exit code, never the pipeline's.

**Fire-and-forget waiters need a completion notification.** Two long waits were
armed GRID-side (good — they survive a VPN drop) but with nothing watching their
output, so a failure at 00:53 was not seen until 06:39. Surviving a dropped
connection is not the same as being observed.
→ **Still advice:** arm a notification on the launch, not only on the part you
are awake for; and never quote an ETA for a launch you have not confirmed
started.

<!-- entry-sep -->

## 2026-08-07 — a five-seed check nearly produced a wrong negative (#2847)

**What happened.** To bridge #2847's figure to a dev-side study, I reran the
issue's exact `scripts/sod/sweep.py` command on `evaluation-framework` HEAD. It
produced **zero** deep spikes. I reran it with the threshold blend off; **zero**
again. Two independent-looking clean runs is persuasive, and the draft report
said the branch no longer reproduced its own figure.

**It was a sampling artefact.** The command's `--iterations 5` runs seeds 0–4,
and at 20 seeds the same command spikes in **7 of 20 runs (35%)** with a
worst-step cost of 1.00 — the same rate and character as the figure. Seeds 0–4
are simply the quiet ones. The finding flipped from "the old path is fixed" to
"the old path is confirmed live, and independently corroborates the study's
control arm."

**Cost.** ~25 minutes and a rewritten report section. It would have been much
worse if it had shipped: a wrong negative about someone else's branch, in a
report whose whole purpose is attributing a fix.

**Two things made it dangerous rather than merely wrong.**
1. A 5-seed check has only **76% power** against a 25%-per-run phenomenon, so
   P(zero | unchanged) = 0.24. That is not a small number.
2. **The two "independent" runs were not independent** — same five seeds, so the
   second run could only confirm the first's sampling draw. Repeating an
   underpowered check with the same seeds buys nothing.

**Still only advice (no control).** Before reporting that something *stopped*
happening, state the per-run rate it happened at and the power of the check
against it. If the check has less than ~90% power, it cannot support a negative;
raise the replicate count instead of reporting the null. This applies to every
"the fix worked" claim on these curves, not just sweeps — and it is a sibling of
the #2825 lesson that a magnitude rule without a power number is meaningless.

**Prevented, separately:** `.pre-commit-config.yaml`'s
`check-added-large-files --maxkb=500` was rejecting 200 dpi report figures and
pushing studies to downsample their own evidence. Raised to 2 MB; the hook is
there to keep datasets and model weights out of git, not to ration figure
quality.

<!-- entry-sep -->

## 2026-08-07 — a fresh worktree ran the *other* worktree's code (#2846)

**What happened.** The #2846 Grid re-measure got a fresh worktree off `dev`, as
every study should. `source gridenv.sh` succeeded, `PYTHONPATH` pointed at it,
and the first command — `selftest_analyze_cut.py` — died on
`cannot import name '_CUT_DIAGNOSTIC_COLUMNS' from
'/exp/sgreenberg/projects/vts-calib/vtscore/eval/voting_iterations.py'`. A
worktree created ten minutes earlier was importing a *different* checkout.

**Two independent hijacks, and each one alone is silent.**

1. `gridenv.sh` prepends `$WT/.shadow` to `PYTHONPATH` — a directory holding a
   no-op `__editable___vtsearch_0_1_0_finder.py` that shadows the venv's
   editable-install finder. That directory is **untracked**, so it does not
   exist in a new worktree, and nothing created it. The finder then wins.
2. `common.setup_env()` does `sys.path.insert(0, VTS_REPO)` with VTS_REPO
   **defaulting to `/exp/$USER/projects/vts-calib`** — a shared worktree from an
   older study. `sys.path[0]` beats `PYTHONPATH`, so even a correct `.shadow`
   would not have saved it.

The traceback only appeared because the branch had *added* a symbol. Had this
re-measure only *changed* behaviour — which is the usual case, and is exactly
what a cut-rule study does — every job would have run the wrong `cut_rules.py`
and produced a clean, plausible, wrong table.

**Cost.** ~15 minutes, caught before launch. The counterfactual is a full study.

**Now prevented (code, not advice):** `gridenv.sh` creates the `.shadow` shim if
it is missing and exports `VTS_REPO="${VTS_REPO:-$_VTS_WT}"`, so sourcing it from
a worktree pins *that* worktree at `sys.path[0]`. `preflight.sh` already checked
that VTS_REPO was set and clean — it just could not check the one thing that was
wrong, which was that nobody had set it at all.

**Also prevented (code):** `preflight.sh` now resolves `import vtscore` the way a
job does — through `common.setup_env()` — and fails if the file it lands on is
not inside `VTS_REPO`. That is the direct evidence the run measures your branch,
and it is checked rather than remembered. Setting `VTS_REPO` correctly is not the
same thing: this run had it right and still imported the other checkout.

<!-- entry-sep -->

## 2026-08-07 — the fidelity check failed because the incumbent had shipped (#2846)

**What happened.** The #2846 Grid re-measure came back with the check that
licenses the whole study **red**: `pooled_mid`, the variant labelled "this is
what production does", disagreed with the run's own threshold on **84 % of
13 653 steps** (max abs diff 0.24, against 0.0 in #2836 four days earlier). The
obvious readings were "the harness broke" or "the branch under test broke it",
and both would have led to hours of bisecting a diff that was innocent.

**Neither. Production moved.** Splitting the mismatches by the base row's
`threshold_provenance` gave a perfect 1:1 partition:

| production took | steps | `pooled_mid` reproduces it? |
|---|---|---|
| `gmm_blend` | 2 158 | yes — max abs diff **0.0** |
| `fold_anchored[*]` | 11 495 | no |

Between the two runs, `d195b004` shipped the fold-anchored threshold, `196085b5`
moved it to κ=0.3 + midpoint cut, and `b03d54e5` made the fused path
unconditional. `pooled_mid` was still bit-for-bit correct on every step that took
the old path. The study's *baseline definition* was two ship decisions stale.

**Why it is worth an entry rather than a shrug.** The consequence was not a
broken run — every within-rule contrast stayed valid — it was that one *class* of
claim silently expired: "rule X beats the shipped midpoint" no longer meant "rule
X beats what we ship". A re-measure that had not looked at the provenance column
would have republished that claim in good faith. On this cluster a study's
baseline is a moving target, because studies here ship things.

**Now prevented (code), twice over:**

1. `analyze_cut.py`'s `production_blend_sanity` no longer just reports
   `ok: false` — on failure it breaks the mismatch down by
   `threshold_provenance`, so "the harness is broken" and "the incumbent moved"
   are distinguishable at a glance instead of by investigation.
2. The ship decision no longer depends on a reconstruction at all.
   `base_row_contrasts` pairs every rule against **the run's own base row** —
   the threshold production actually used on that step, whatever it was — and
   `decisions.beats_production` / `ship_candidate` are computed from that.
   `beats_midpoint` survives as the historical #2836 contrast and gates nothing.
   A baseline that is read rather than reconstructed cannot go stale, so the
   next incumbent can ship without expiring anyone's conclusions.

**Still advice:** when a study defines *any* arm as "what production does",
re-read that definition against `git log` on the path it names. The base row
covers the threshold; the next study will name something else.

<!-- entry-sep -->

## 2026-08-07 — an 8-seed grid could not answer the question it was run to answer (#2877)

**What happened.** The VG region-voting generalisation check reused #2876's
sizing verbatim — 8 seeds, which on COCO had been comfortable — and drained
clean: 1288/1288 cells, no failures. It reproduced the mechanism perfectly. It
also put a 95% CI of **[−0.014, +0.019]** on the decision endpoint against a
pre-registered tolerance of **+0.01**. That interval contains "the offset is
free, keep it global" *and* "the offset costs something, gate it" — **opposite
shipping decisions**. The run had measured nothing decision-relevant.

**Why the transplanted sizing failed.** Sizing does not travel with an arm
table; it travels with the *endpoint's variance in that environment*. VG
region-voting costs sit near 0.43 where COCO's sit near 0.137, and the paired
per-cell SD is correspondingly larger (0.111). At that SD, a ±0.010 half-width
needs **n≈473**; 8 seeds × 23 categories delivered 180. The *positives* endpoint
was hugely over-powered at the same n in both environments, which is exactly how
this hides — the run looks healthy because the endpoint you can see moving is
the one that was never binding.

**Cost.** ~55 minutes of cpu-partition time to rerun at 24 seeds (n=540), which
put the CI at [+0.003, +0.022] and made the answer unambiguous. Cheap only
because these cells are single-threaded and GPU-free.

**Still only advice (no control).** When porting a study to a new environment,
**re-derive n from a pilot's observed SD on the decision endpoint** before
running the full grid — do not inherit the seed count along with the arm table.
One arm's worth of pilot is enough to compute it: `n = (1.96·SD/half_width)²`.
And report the CI on the decision endpoint even when the ship rule passes, so a
wide null is never mistaken for a tight one. (`analyze_acq.py` already refuses to
read a p-value as a null for this reason; the gap was that nothing checked
whether the *design* could produce a usable interval.)

**Prevented, separately — smoke on a representative cell, not on cell 0.** The
first smoke ran array index 0 and wrote **zero rows**, which looks exactly like
a broken harness. It was not: rows are only emitted from the first positive
onward, and index 0 was `bag`/seed 0, whose first positive arrives at vote 106 —
the worst of 92 cells, where the median is vote 3. Fifteen minutes went to
confirming the harness was fine. Cell 0 is the alphabetically-first category at
seed 0, which is a biased draw, not a neutral one; and "0 rows" is a legitimate
outcome in a starved environment, so it cannot be treated as a failure signal on
its own. Smoke a mid-grid index, and check the row count against a known-good
run of the same environment before concluding anything.
