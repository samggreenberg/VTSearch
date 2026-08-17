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

<!-- entry-sep -->

## 2026-08-07 — a study reported an environment it never ran (#2877)

**What happened.** #2877 pre-registered `visual_genome_m × siglip` as the
**region-voting** generalisation check for the acquisition cut, and justified
the whole exercise on region voting's scoring geometry: *"a media's score is a
max over ~24 region-node scores, so the Bad mode is an extreme-value
statistic."* The harness takes `region_voting=True` for that arm, the run
drained clean, the analysis was careful, and a report went out describing a
region-voting result. **That arm does not region-vote.**

`region_voting` is a *request*, not a guarantee. `_good_training_vec` pools the
dragged ground-truth box only when the media carries a stored `patch_grid`, and
silently falls back to the whole-image embedding otherwise. `siglip` is a
single-vector embedder: no `patch_grid`, no `patch_regions`. So the run trained
on whole-image vectors (verified: the region-voting vector is **byte-identical**
to the whole-image vector on 200/200 medias carrying a box), scored whole-image
(`region_aware=False`), and blended under **`cap50`** — the *binary* schedule.
It was a second **binary**-voting environment throughout.

**Why it survived every check.** Nothing was broken, so nothing complained. The
dataset really is boxed; the flag really was set; `REGION_VOTING_BY_DATASET` is
keyed by **dataset** while whether region voting happens is a property of the
**embedder**, and the two only coincide for patch embedders. The harness's own
config docstring called the whole VG block "region voting", so the mislabel was
inherited rather than invented, and it had been sitting there across several
studies.

**Cost.** A published report, a PR and an issue comment all had to be corrected,
including a headline recommendation ("gate the offset by voting mode") that the
run could not support — both environments were binary. The measurement itself
survived intact; only what it was evidence *about* changed. It would have been
far worse merged: a voting-mode conditional shipped on evidence from two
binary-voting runs.

**Now prevented (code, not advice):** `simulate_voting_iterations` warns when
`region_voting=True` is requested and no media carries a `patch_grid`, naming
the consequence ("this run is BINARY voting"). `experiment_config.py`'s
docstring now says which VG arms actually region-vote and which silently do not.

**Still advice — a flag you passed is not a property you got.** When an
experiment's *rationale* rests on a mechanism ("max over region nodes",
"grouped calibration", "patch geometry"), verify the mechanism is present in the
data before running, not the flag that requests it. One line is enough:
`any(m.get("patch_grid") is not None for m in medias.values())`. The general
form: **assert the premise, not the parameter.** A silent fallback is designed
to keep a run working, which is exactly why it will not tell you the run changed
meaning. This is a sibling of the #2846 lesson — there a worktree silently ran
another checkout's code; here a config silently ran another environment's
geometry.

<!-- entry-sep -->

## 2026-08-07 — a signal that something else can satisfy (#2897)

**Cost:** ~40 min of run time, and one wrong "it's finished" told to the owner.

**What broke.** Three times in one study I waited on a signal that something
*other than the thing I was waiting for* could satisfy.

1. **An empty job id read as a launch.** `launch_all.sh`'s prepare stage
   hardcoded `--partition=gpu` and passed `--gres="$GRES"` through literally.
   Every recent study sets `CALIB_GRES=none` for CPU-only cells — and
   `--gres=none` is not ignored here: the submit filter rewrites it into a
   `--gpus-per-task` form and rejects the job for having no `-n`. `--parsable`
   returns an **empty string** on refusal, so `--dependency=afterok:` got a blank
   id and the dependent launcher failed too, burying the real error. **Both live
   A/B arms silently failed to launch** into a log nobody was reading. Latent for
   every study using the full chain with `CALIB_GRES=none`; the screen escaped
   only because its prepare had been run by hand.
2. **A stale file read as completion.** Completion was armed as "does
   `STATUS.md` exist" — but an earlier stage had already written one. The watcher
   fired 18 seconds later and I reported the study complete about an hour early,
   with 126 jobs still queued.
3. **A degenerate arm read as a broken one.** The A/B builds
   `CALIB_FOLD_COUNTS="2,$K"`, which at K=2 collapses to a single value, so the
   *control* arm's analyze died with `KeyError: 'voting'` — a traceback where the
   honest answer was "nothing to contrast".

**Now prevented (code):**

1. `launch_all.sh` drops `--gres` when it is `none`/empty (the `GRES_ARG` shape
   `launch_cells.sh` already used) and honours `CALIB_PREP_PARTITION` /
   `CALIB_PARTITION`; `require_jobid` after **every** `sbatch` aborts on a
   non-numeric id, naming the stage.
2. `analyze_folds_2897.py` returns the empty shape for a baseline-only arm.

**Still advice — the shape of the mistake, which no single check covers.**
Prefer a signal that *cannot* pre-exist: a terminal job state, or a marker
written only by the stage you are waiting for. Before arming any wait, ask what
else could satisfy it. And note the launchers already printed "confirm each arm
came back with a numeric job id": that instruction was correct and was ignored,
because the thing reading the output was a batch job. **A check whose only
enforcement is a human reading a log does not survive being chained** — if a
stage can be chained, its checks have to be code.

*(This study also re-hit the #2877 premise trap above: `visual_genome_m ×
siglip` carries no `patch_grid`, so its `region_voting=True` is a silent no-op.
The one-line assertion recommended there is what caught it.)*

<!-- entry-sep -->

## 2026-08-08 — a gate that reported "ok" without having looked (#2905)

**What happened.** `preflight.sh`'s first and third checks — "this arm's results
dir does not already hold another grid's cells" and "no zero-byte cells that
resume would skip" — only ever looked under `$EXP/results-ab/`. Every
acquisition and anchor study puts its arms under `$EXP/results/`. So for those
studies both checks ran, found no such directory, and printed **`ok`**.

That is worse than not having the check. A missing check is a known gap; a check
that passes vacuously is a *positive* signal that the thing was verified. #2877
launched behind a green preflight whose arm-collision check had not examined a
single file.

**Cost.** None yet, by luck — no acquisition study has collided two grids in one
dir. The exposure was the whole point of the check, silently absent for four
studies.

**Now prevented (code).** Both checks iterate `results-ab` *and* `results`, and
the arm check reports which root it looked in, so "ok" now names its evidence.

**Still advice — a check's silence is not the same as its success.** When a
gate's finding is "nothing wrong here", make sure it can distinguish *nothing
wrong* from *nothing examined*. The cheap form is to print what was inspected
(paths, counts) alongside the verdict, so a vacuous pass reads as vacuous. This
is the same shape as #2897's two failures — an empty job id read as a
submission, a stale file read as a completion — a signal satisfiable by
something other than the thing being waited on.

<!-- entry-sep -->

## 2026-08-08 — the tempting story was regression to the mean (#2905)

**What happened.** Three environments disagreed about
`ACQUISITION_INCLUSION_OFFSET`, and a clean unifying explanation presented
itself: the offset is a *starvation remedy*, paying where the detector has few
positives and charging its price everywhere. I tested it by binning each cell on
how many positives the **`prod` arm** found in that cell, then reading the
treatment response per bin. The curve appeared in both voting modes, monotone
and beautiful, with a sharp crossover from benefit to harm.

The response was measured **against that same `prod` run**. Cells where `prod`
happened to do unusually well are, by construction, more likely to show a
negative delta. Mean reversion manufactures exactly that curve with no mechanism
at all.

**Re-cut on axes independent of the arm being scored** — the category's
`realized_prevalence`, and a leave-one-out baseline (the mean `prod` positives
of the category's *other* seeds) — the binary-voting curve **survived** at full
strength (AP slope −0.0207 on log prevalence, CI [−0.0259, −0.0159]; −0.0402 on
LOO). The region-voting curve **vanished**: significant on the contaminated axis
(−0.0074, CI excludes 0) and null on both clean ones. Half the finding was real
and half was an artefact, and they looked identical.

**Cost.** ~20 minutes, because the check was run before the report was written
rather than after. Had it not been, the report would have recommended a
supply-based rule on evidence that was half self-fulfilling, and the
voting-mode conclusion would have been backwards — at *matched* prevalence the
modes still differ, which is only visible once the contaminated axis is gone.

**Still advice — never bin on a quantity that also appears in the contrast.**
If cells are grouped by a baseline arm's own outcome and then scored on
`treatment − baseline`, the grouping variable is inside the response and the
slope is partly arithmetic. Two cheap fixes, both used here: bin on something
fixed by the data (prevalence, category, pool size), or leave the cell's own
observation out of the statistic it is binned on. The diagnostic signature is
worth memorising: **an effect that is significant on the contaminated axis and
absent on the clean ones is mean reversion, not mechanism.**

<!-- entry-sep -->

## 2026-08-12 — a study invalidated by a fix that landed after it (#2905 / #2943)

**What happened.** The #2905 region-voting run completed clean on 2026-08-07/08 —
3864/3864 cells, zero failures, premise verified, preflight green — and shipped a
report and a constant. Two days later #2943 (`b7d528d8`) fixed `_score_pool`,
which on a patch dataset scored each pool item by its **whole-image** vector while
every threshold was cut on the style's **region max-pooled** scores. Autopilot's
`hard` pick compares `ranking[cid] <= threshold` absolutely, so the two have to
share a space, and on `visual_genome_m × dinov3_patch` they did not. The run was
therefore measuring a harness artefact. PR #2909 merged *after* the fix, so `dev`
briefly carried a fixed harness alongside a report written on the broken one.

**Cost.** A published report voided, a merged PR corrected, and ~7 h of cluster
time to redo. The shipped constant survived only because it never depended on the
region numbers.

**Why the existing gates all passed.** The premise gate this same study *added*
(`preflight.sh --require-region-voting`) confirmed `patch_grid` on 4193/4193 —
correctly. The bug was **downstream of the premise**: the environment really did
region-vote, and the pool was then scored in the wrong space. A gate that proves
the input geometry says nothing about whether the pipeline honours it.

**How it hid.** The bug's own signature lived in the columns added to catch it.
`acq_pool_percentile` / `report_pool_percentile` were computed in the same
mismatched space as the pool scores, so they read as healthy (`prod` median
0.9859) while the cut was in fact detached from the ranking. #2943 says this
plainly: the columns "were computed in the same mismatched space and so could not
reveal it."

**The damage was asymmetric, which is what made it fatal rather than noisy.**
Steps with the cut pinned at exactly 1.0 — above the *entire* pool, where raising
`k` changes nothing: `prod` 5.8%, **`acq_m3` 39.2%**, `acq_p2` (the k=+2
falsifier) **1.5%**. The treatment arms were clamped; the falsifier moves the cut
*down*, away from the ceiling, and was spared. So the one contrast that appeared
to validate the mechanism is the one contrast the bug did not touch. **When a bug
clamps a lever at one end, the falsification arm at the other end will still
behave — and will certify the run.**

**Prevented (partly), by others:** #2943 fixed the scoring and made `_score_pool`
take the threshold's space explicitly.

**Still advice — a clean run is not a valid run, and freshness is a property of
the harness, not the data.** Before publishing, re-check what has landed on `dev`
*since the run's base commit* in the code paths the study exercises. This run's
base was `84789040`; the invalidating fix was five days later, and its author had
even written "**Wait for #2943 before running it**" into `REPORT.md` — a warning
that post-dated the run and so was never seen. A cheap habit that would have
caught it: at analysis time, diff `git log <base>..origin/dev -- <the modules the
harness calls>` and read the subject lines. Ten seconds, and it is the only step
that looks *forward* from a run instead of inward at it.

<!-- entry-sep -->

## 2026-08-12 — a column the fix redefines cannot be the fix's acceptance test (#2905)

**What happened.** Verifying the #2943 fix before committing ~7 h of cluster time,
I pre-registered three pass criteria on a single re-run cell, two of which were
"`acq_pool_percentile` pinning should **fall**" and "its median should move off
the ceiling". Both failed — pinning went **0% → 60.6%**, the median **0.9905 →
1.0000** — and for a moment that read as the fix not working.

It was my criteria that were wrong. #2943 changed *what those columns measure*:
pre-fix they were computed in the pool's whole-image space (self-consistent, and
therefore healthy-looking), post-fix in the threshold's max-pooled space. They are
**not the same statistic on either side of the fix**, so no comparison across it
means anything — including a comparison designed to validate the fix.

**Cost.** Minutes, because the criteria were written down in advance and so failed
loudly instead of being quietly reinterpreted. The real risk was the opposite
outcome: had they *passed*, I would have launched on a false green.

**What actually showed the fix working** was the outcome, not the instrument — the
same cell, same trajectory, went from 6 positives / cost 0.452 / AP 0.209 to 3 /
0.625 / 0.130, i.e. the optimistic bias #2943 describes, removed.

**Still advice — validate a fix on quantities the fix does not redefine.**
Prefer end-state outcomes (positives found, cost, AP) or a within-harness contrast
(does arm A differ from arm B *on the fixed code*) over any before/after on a
column whose semantics the change touches. And write the criteria down first: the
value of a pre-registered check is not that it is right, but that when it is wrong
you find out instead of rationalising.

<!-- entry-sep -->

## 2026-08-12 — a study default is not a shipped default (#3129)

**What happened.** The overview benchmark exists to measure *what a current user
gets*, so every behavioural knob was deliberately left unset. `CALIB_PATCH_STYLES`
was one of them — and its default is `max_patch,max_patch_pca_hac`, because the
**calibration study** wanted that contrast. `max_patch_pca_hac` lost the
Max-Patch study at the operating point (PR #2749) and production no longer
carries the tree it delegates to. So "leave the defaults alone" silently added a
retired arm to a baseline, and doubled the cost of every patch cell: all three
sizing cells finished `max_patch` and were still grinding through the HAC style
when they were cancelled.

Caught by the owner reading a status line, not by any check. Cost: one cancelled
array ~3 minutes in, plus ~50 minutes of sizing.

**Prevented?** *Advice only.* The general form — "is this default a product
default or this study's default?" — is not mechanically checkable without a
per-knob provenance table that does not exist. What *is* now pinned: the
benchmark launcher sets `CALIB_PATCH_STYLES=max_patch` explicitly and passes it
in `ENVX` rather than relying on `--export=ALL`, so the value cannot drift with
the submitting shell.

**Rule of thumb:** when a study's premise is "defaults", enumerate the defaults
and say where each one comes from. A knob whose default was chosen by another
study is a knob you are setting, not a knob you are leaving alone.

## 2026-08-12 — a header-only CSV passes `-size +0` (#3129)

**What happened.** The array's watcher reported `non-empty cells: 189/189, zero-byte: 0`
and the run looked complete. It was not: **seven cells held a header row and no
data**. `find -size +0` counts bytes, and a header is bytes. The analyzer caught
it only because it counted loaded rows separately from files found and printed
both numbers.

Those seven turned out to be the most interesting result in the study — runs that
never surfaced a single positive in 150 votes — so the cost of missing them would
have been a wrong headline, not just a small denominator.

**Prevented?** *Partly.* The wave-2 watcher counts cells with `wc -l > 1` rather
than bytes. The general control is the one already in the playbook: **count what
you dropped, and print both numbers**. A completion count that cannot distinguish
"wrote results" from "wrote a header" is the same blind gate as a preflight that
reports ok without looking.

**Related, worth knowing:** `simulate_voting_iterations` never emits a row with
`n_good == 0`, so a starved cell writes a header and exits 0 with no warning
anywhere. Verifying `min(n_good) == 1` across every row is what proved the seven
were starvation rather than an I/O incident. A `starved` column and a one-line
warning would make this visible without an analyst noticing a row-count mismatch.

## 2026-08-12 — `echo "exit=$?"` after a pipeline reads the pipeline (#3129)

**What happened.** The playbook already says to check a commit's own exit code.
It was checked as `git commit -q -m "..." 2>&1 | tail -3; echo "commit exit=$?"`,
which reports **`tail`'s** status. It printed `exit=0` while the pre-commit hooks
had rewritten the files and failed the commit; `git push` then pushed nothing and
`git log` still showed the previous head.

Also on the same run: `git commit -F /tmp/commitmsg2.txt` picked up a **stale
file from an earlier session** and committed the report under an unrelated #2877
message, which had to be amended.

**Prevented?** *Advice only, but sharpened.* Never pipe the command whose status
you are about to read — run it bare, then `echo $?`, then inspect `git log -1`
to confirm the head actually moved. Use a run-unique path for `-F` message files
(`/tmp/msg-<jobid>.txt`); `/tmp` on a shared login node is not yours alone.

<!-- entry-sep -->

## 2026-08-12 — memory, not CPU, is what you run out of (#3129)

**What happened.** Every stage of this study asked for 64–96G without measuring
anything. Actual peak RSS was ~1.1 GB for whole-image cells and ~14 GB for patch
cells. The per-user cap is 1.07 TB, so `16 x 64G` claimed 95 % of it and three
separate later submissions queued behind this study's own array.

**Prevented?** *Yes* — `preflight.sh --mem --conc` computes the footprint against
the tightest applicable QOS cap and fails at ≥90 %. Verified against tonight's
exact configuration: `64G x 16` fails at 95 %, `24G x 11` passes at 25 %. Sizing
guidance and measured RSS figures are in GRID-PLAYBOOK.md.

## 2026-08-12 — two arrays, one job name (#3129)

**What happened.** Both the wave-2 rerun and the binary-voting arm were submitted
as `--job-name=bench-cells`. Every per-name query then spanned both runs —
including the completion waiter this repo's own skill recommends
(`squeue -u $USER -h -n JOBNAME`), whose count would have declared one run
finished while the other was still going. It also made `squeue` output
unreadable at a glance while triaging the memory jam.

**Prevented?** *Yes* — `preflight.sh --job-name` fails when that name already has
tasks queued or running.

## 2026-08-12 — I quoted a maximum as if it were a mean (#3129)

**What happened.** Asked for an ETA on a 270-cell array, I took the *slowest*
patch cell from a previous grid (1913 s) and multiplied. That produced "~00:30"
for something that finished nearer 23:00 — a 90-minute error, in the pessimistic
direction, offered while proposing to cancel work to save time. The mean was
~456 s and four cells of this grid had already finished at 12–14 minutes; I had
the data and did not look at it.

**Prevented?** *Advice only.* Two habits: quote ETAs from **this** grid's own
completed cells, not a previous grid's; and when the distribution is wide, say so
rather than collapsing it to one number. The existing rule — *never quote an ETA
for a launch you have not confirmed started* — needs a sibling: **never quote one
from a cell you have not timed here.**

## 2026-08-12 — an assertion that fires after the write is not a guard (#3129)

**What happened.** A patch script checked `assert "import os" in s` *after*
`p.write_text(s)`. The assertion was correct and caught a real bug — the helper
it inserted used `os.environ` in a module that never imports `os` — but it fired
against an already-modified file, so it reported the problem instead of
preventing it.

**Prevented?** *Advice only.* Validate every precondition before the first
mutation. Patch scripts that edit code in place should be idempotent *and*
fail-closed: check anchors, check imports, then write.

## 2026-08-12 — heredocs through ssh (#3129)

**What happened.** Three separate attempts to send a Python or bash snippet
through `ssh grid '... <<EOF ...'` were mangled by shell quoting — one silently
produced a syntactically invalid script that only failed at run time.

**Prevented?** *Advice only.* Write the script to a file locally and `scp` it.
The round trip is cheaper than one debugging cycle, and the file is then a
reviewable artifact rather than a string inside a command.

<!-- entry-sep -->

## 2026-08-12 — an idempotency guard matched the wrong occurrence (#3129)

**What happened.** Twice in one session, a patch script guarded "have I already
applied this?" on a substring that appears elsewhere in the target file, so it
skipped the edit **and reported success**:

* `if "vg_box_small" in s` matched the `BOXED_BY_DATASET` entry added by an
  earlier patch, so the `DATASET_EMBEDDERS` registration never happened. The
  verifier then reported PASS over an empty grid.
* `if "--job-name" in s` matched the `sbatch --job-name=bench-prep` lines already
  in the launcher, so the preflight flags were never wired.

Both were caught only by inspecting the file afterwards. A patch that skips
silently is worse than one that fails: it leaves the tree looking done.

**Prevented?** *Advice only.* Guard on a string **unique to the insertion** — a
new symbol name, or a literal marker from the inserted block — never on a token
that could plausibly appear in the surrounding file. And verify the postcondition
(grep for what you inserted) rather than trusting the script's own report.

<!-- entry-sep -->

## 2026-08-13 — the report died on the laptop's RAM, not on the GRID's (#3131)

**What happened.** Every expensive thing in this study ran on the cluster, and
then the write-up ran `./run-tests.sh` on the laptop — a 3 GB box. `pyright`
runs under node with a ~1.8 GB default heap, so it aborted with *"Ineffective
mark-compacts near heap limit"*, which reads like a type-check failure and is
not one. Bumping `NODE_OPTIONS=--max-old-space-size=8192` on a machine with 3 GB
total made it worse rather than better, and the retry was killed outright: the
session ended on `Exit code 137`, the OOM killer, with the report committed but
the gate never green.

Two smaller traps rode along. The wall-clock wrapper in `run-tests.sh` treats
137 as *its* kill signal, so an OOM death is announced as **`TESTS TIMED OUT`** —
the banner names the wrong cause. And a standalone `pyright` had already
returned `0 errors` minutes earlier; the same command inside the suite died,
because the suite had the frontend build's memory alongside it.

**Cost.** The session, and the three hours of work in its context.

**Prevented?** *Partly.* `run-tests.sh` now reads `MemAvailable` up front and
prints a LOW MEMORY banner with the `sbatch` line to run it on the GRID instead.
It warns rather than blocks — 6 GB is the observed want, not a measured cliff.

**The general form, and the one worth keeping:** *the machine you analyse on is
part of the experiment's resource plan.* The GRID was sitting there with 502 GB
on the login node and idle `cpu` nodes. A gate that needs 6 GB belongs on a
node that has it, and moving it there costs one `sbatch`. Nothing about "the run
is finished" means the laptop can now take the load — the write-up's gates are
the heaviest thing a study asks of a *local* machine, and they arrive exactly
when there is the most uncommitted context to lose.
