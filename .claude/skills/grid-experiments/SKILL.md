---
name: grid-experiments
description: Practices for running eval experiments and sweeps on the GRID (SLURM). Use when launching, monitoring, resuming, or analysing a study under scripts/experiments/ — anything involving sbatch arms, cells, CALIB_EXP dirs, or a long run whose results feed a REPORT.md. Also use when an experiment run fails, to record the lesson.
---

# Running experiments on the GRID

Long runs on a shared cluster fail in ways that are cheap to prevent and
expensive to discover late — usually hours late, usually overnight. The
recurring shape is not "the science was wrong" but "the run silently did not
happen, or did not run the code you thought."

Three companions, each with a different job:

| File | Job | When |
|---|---|---|
| `scripts/experiments/preflight.sh` | **Blocks** the checkable mistakes | Before submitting arms |
| `scripts/experiments/GRID-PLAYBOOK.md` | SLURM resource practice (memory, QOS, chunking) | When sizing a sweep |
| `scripts/experiments/LESSONS.md` | Incident log index; entries are `lessons/*.md`, one per incident | Read once; **add a file when something breaks** |

## Before launching

Run the preflight. It is a gate, not a reminder:

```bash
bash scripts/experiments/preflight.sh --exp "$CALIB_EXP" --arms a,b,c \
  --job-name "$JOB_NAME" --mem "$MEM" --conc "$CONC"
```

Pass `--job-name`, `--mem` and `--conc` whenever you know them. They gate the two
mistakes that cost the most queue time in #3129: an array that claims your whole
**per-user memory** allowance (memory is the binding quota here, not CPU — see
GRID-PLAYBOOK.md for measured RSS per cell type), and a **job name you are
already using**, which silently breaks the per-name completion waiter below.

Add `--reuse-prepare "$PREPARE_DIR"` when you are skipping a GPU prepare stage by
reusing a finished study's output. It checks that every `crops/` entry still
resolves — the links point into the study that generated them, and `readlink -f`
recreates them happily after that study is archived (#2881).

It refuses to pass when the results dir already holds another grid's cells,
when the *actual* mount is low on space, when zero-byte cells from a previous
incident would be skipped by resume, or when `VTS_REPO` is unset or points at a
worktree that isn't what you committed.

Then check the two things a script cannot:

- **One study, one `CALIB_EXP`.** If the grid differs in *any* way — categories,
  seeds, steps, voting mode — it is a different study and needs its own dir.
  Sharing one is how a batch aborts overnight or, worse, how two grids get
  analysed as one.
- **Size it from a real cell, not a guess.** Run one cell, read its actual
  seconds, multiply. Per-step cost is often flat in label count, so a long
  horizon can be far cheaper than it looks — and a region/patch cell can be 10×
  a whole-image one, which changes the arm budget entirely.

## After launching — confirm it started

**A submission is not a launch.** Verify every arm came back with a numeric job
id and that cells begin appearing. An arm that was refused, or a waiter that
aborted, looks exactly like an arm that is merely queued.

**Never quote an ETA for a launch you have not confirmed started** — and
never quote one from a cell you have not timed *on this grid*. Read the
distribution of completed cells (`sacct -j <id> --format=Elapsed,State`), not
its maximum: quoting a previous grid's slowest cell produced a 90-minute
overestimate in #3129, offered alongside a proposal to cancel work.

Arm a completion notification on the run itself — a background command that
exits when the queue drains:

```bash
ssh grid 'until [ "$(squeue -u $USER -h -n JOBNAME -o %i | wc -l)" -eq 0 ]; do sleep 120; done; echo DONE; <status summary>'
```

Chaining work GRID-side (`--dependency=afterany`, a nohup'd waiter) is right —
it survives a dropped VPN — but **surviving a disconnect is not the same as
being observed.** Something must report the outcome, including the failures:
if the process crashed right now, would anything tell you?

## While it runs

- **Poll the real signal**: count cell files per arm against the expected total,
  not just `squeue`. A drained queue with missing cells means failures.
- **Watch the disk on the right mount.** `df` the experiment path itself; a
  parent mount's free space can be wildly different.
- **A transient cluster failure leaves debris that resume cannot see.** Before
  resuming, delete zero-byte outputs — they count as "done".

## Committing between steps

The pre-commit hooks **rewrite files and fail the commit** when they do. Always
check the commit's own exit code, never a pipeline's:

```bash
git add -A && git commit -q -m "..." ; echo "exit=$?"
```

A masked commit failure means `git push` pushes nothing and the cluster then
runs the *previous* commit — including through a full test suite, which will
pass against code you did not write.

## Analysing

- **Count what you dropped.** Unreadable or missing cells must be reported, not
  silently excluded. Analysing 1295 of 1344 cells while reporting neither number
  is how a disk incident becomes a wrong verdict.
- **Verify the harness reproduces production** before trusting any comparison —
  a counterfactual arm that reproduces the live path bit-for-bit is what
  licenses the rest of the table.
- **Band the axis the mechanism runs on.** An average across a crossover is
  precisely the number that hides it, and the axis a user spends (clicks) is
  often not the axis the method converges on (positives).

## Writing the report

A `REPORT.md` is read by someone deciding what to do next, and it has to survive
their disbelief. Three things earn that, and all three were missing from the
overview-bench report until its owner asked for them:

**Show only the digits your sample can support.** Default to **two significant
digits** (`cost 0.22`, `AP 0.70`, `fpr 0.071`); print a third only where a
decision turns on it. Four digits are not more rigorous, they are harder to read
and they *invent* findings — an unpaired `0.0462` against `0.0508` reads as a
trend that a ±0.03 standard error cannot support, and someone will build a
follow-up on it. So quote every arm-vs-arm difference **paired** (same category,
same seed, same split) with its standard error, and say plainly when a
difference is smaller than twice that: "not resolvable here" is a finding, and a
much more useful one than a false decimal.

**Every report gets figures, generated by a committed script from the same CSVs
as the tables.** Prose plus a deep-regime table cannot show a curve's shape, a
crossover, or how unlike the mean a single run is. At minimum:

- **the quality-over-clicks pair** (see below) — mandatory for every simulated-user
  study, no exceptions;
- **the interactive viewer** (see below) — likewise mandatory, and linked from the
  report;
- whatever axis the study's mechanism runs on (scale band, prevalence, κ …);
- the binding constraint, if the study found one.

Put them in the study's own `figures/` directory, PNG at ~130 dpi, embedded with
a relative-path image link and a caption that says how to read the figure and what it does *not* license
(averaging across prevalences, log axes, unpaired panels).

### The quality-over-clicks pair (mandatory, and there is one implementation)

**Any study that simulates a VTSearch user clicking owes two figures showing how
good that user's detector is as they click more.** Not positives mined — that is
what the *acquisition* did; an arm can mine well and rank badly. The metric is
the one the ship decision reads (`cost`, and `average_precision` beside it),
plotted against the axis the user actually spends:

- **the averages** — one panel **per dataset**, one line **per arm**, averaged
  over every seed and category on that dataset, with an inter-quartile band.
  This is the figure someone reads to pick an arm.
- **the individuals** — one file per dataset, one panel **per arm**, and inside
  it **every seed of that arm on that dataset as its own line**. A mean hides
  that some runs never leave the floor, and on this axis the spread is routinely
  the finding: two arms with the same mean can be "every run is mediocre" and
  "half the runs are excellent and half never start".

Do **not** write this plotting code again. `scripts/experiments/calibration/curves.py`
is the single implementation — `curves.quality_vs_clicks(main, figdir, arms=…,
denominator=…, baseline=…)` — so that the figure a reader learns to read in one
report is literally the same figure in the next one. It also has a CLI, for
regenerating a finished study's curves without redoing its analysis.
`selftest_curves.py` is its planted-answer test.

Three things it enforces, each of which is how one of these figures lies:

- **Click 0 is the zero-click text sort, and it is not optional.** There is no
  measurable detector at the far left, so the tempting thing is to start the
  axis at the first trainable click — which throws away the comparison that
  decides whether the loop is worth anything at all. Typing a query and reading
  the ranked haystack is **free**; that is the number the clicked detector has
  to beat. Anchor every curve at `t=0` on the cell's own text-sort quality
  (`text_baseline.py` computes it, keyed by cell), carry it across the panel as
  a reference line, and report the **crossover** — the first click at which the
  arm is worth more than the query. An arm that never crosses must say `never`.
  The left end is then "what typing got me", the right end is "what clicking got
  me", and the distance between them is the study's whole subject.
- **The denominator, drawn.** The metric frame starts at the first *trainable*
  step, so a cell that never found both classes contributes no rows and silently
  leaves the average. An arm starving on a third of its grid then gets its mean
  computed over the two thirds that worked — and looks *better* for it. Pass the
  cell list as `denominator`, and the figure carries a coverage strip plus a
  dashed-where-partial rule so no level is quoted off a subset by accident.
- **No silent subsampling.** If the per-run panel is capped, the cap goes in the
  panel title. A hairball with a third of its lines removed reads as a tighter
  arm, not a truncated figure.

### The interactive viewer (mandatory, and there is one implementation)

Two PNGs answer the questions the *analyzer* asked. They cannot answer the ones
a reader has after reading it — *"does that hold on the other dataset?"*, *"is it
the scarce categories doing all the work?"*, *"does it survive on recall, or only
on cost?"* — because each is a different slice, and a PNG is one slice chosen in
advance. Every simulated-user study therefore also ships
`scripts/experiments/calibration/viewer.py`'s output, `viewer.html`, in its own
directory, linked from the first section of the report. It carries every slice:

| Control | Choices |
|---|---|
| dataset | one, **all** (averaged), or **each** (a line/panel per dataset) |
| category | one, **all** (averaged), or **each** |
| embedder | any **non-empty** subset — **one panel each, never averaged** |
| arms | any **non-empty** subset |
| seeds | averaged, or every seed its own line |
| metric | every metric the run emitted |

Four rules it enforces, none of which is optional:

- **Hue means whatever the reader is comparing, and it is stated.** Three
  dimensions can vary — arm, category, dataset. The first varying one (in that
  order) with at most 8 values takes hue; every other varying one becomes a
  panel. A ninth hue is never invented: a dimension with more values than the
  palette has validated slots folds into small multiples instead. The legend
  says *"Colour = arm · one panel per embedder × category"* in as many words,
  because a legend that only lists values leaves the reader guessing which
  dimension they are looking at. Colour follows the value's position in the
  **study**, never in the current selection, so deselecting a series does not
  repaint the survivors.
- **Embedders are never averaged.** Two embedders are two representations of the
  haystack and their mean describes no system anyone could run. Faceting makes
  that structural rather than a rule someone has to remember.
- **Pooling is weighted by the cells that contributed**, not a mean of means —
  the payload stores `n` beside the moments so "all categories" is exact. The
  two differ exactly when one category trained on fewer cells, which is the
  survivorship the coverage strip exists to expose.
- **Nothing is thinned in silence.** The per-seed payload is packed to a byte
  budget by coarsening the *click* axis — never by dropping runs or metrics —
  and the page says which grid it landed on. If it cannot fit, per-seed mode is
  disabled with the reason on screen.
- **Every subset control is non-empty, and says so.** An empty selection has no
  honest rendering: the page either goes blank or quietly falls back to "all",
  and a reader who did not notice takes a chart of everything for a chart of
  nothing. The last remaining chip is **locked** rather than silently snapped
  back on — both stop the same thing, but only one of them tells the reader why
  the click did nothing, and a control that ignores a click reads as broken.

`selftest_viewer.py` is its planted-answer test: it checks the codec round-trip,
the weighted pooling against a hand-computed answer, the click-0 anchor, and the
budget note.

**The metrics come from the harness, not from the viewer.** `cost`, `precision`,
`recall`, `f1`, `fpr`, `fnr`, `average_precision` and `auroc` are emitted by
every run through `vtscore.eval.calibration_metrics.detection_metrics` and
`DETECTION_METRICS`, which also carries each metric's label and *direction*. A
report or viewer that decides for itself which way is "better" is how "lower is
better" gets attached to recall. If you add a metric, add it there.

**Show the errors themselves, not just the error rate.** Any claim about a
*kind* of mistake ("it over-includes on scene categories") owes literal
examples: individual items, with their score, the threshold, the source file id,
and every label the dataset carries on that item. Set `VTS_DUMP_TEST_SCORES`
and use `launch_errdump.sh` / `error_report.py` / `label_noise.py` (see
[`docs/EVAL.md`](../../../docs/EVAL.md)). This is not decoration — it is the
only way a reader can tell **a model error from an annotation error**, and it
changes what happens next: a wrong model means more work on the model, while a
wrong label means cleaning the dataset and re-running. On this benchmark the
examples showed VG's `sky` labels are simply missing on 6.6-9.5% of the flagged
images, and that `sunglasses` and `glasses` are the same object under two
labels — neither of which any aggregate could have said.

**On image data, show the images.** A listing of file ids is enough to count
with and not enough to adjudicate with: "is this a missing label?" is a question
about the picture. `make_error_sheets.py` renders the dumped rows as contact
sheets — score, the dataset's own annotations under each thumbnail, and the
target's ground-truth box drawn where the dataset has one — and it runs on the
cluster, because that is where the source images are. A boxed sheet is often the
whole argument: the `tip` "positives" are a plane's nose, a church spire and a
bollard, which no amount of prose establishes as convincingly.

**Give a long report a reading copy.** `REPORT.md` is the archive and renders
inline on GitHub, but bitmap plots cannot be zoomed and a 900-line document wants
a table of contents. `make_bench_html.py` builds a self-contained page under
[`docs/reports/`](../../../docs/reports/) from the markdown — vector figures
inlined, photographs embedded, one file. **Generate it, never hand-write it:** a
second copy of the narrative drifts from the first, and the point is that both
renderings say the same thing.

Reports cite only analysis code that is **in the tree**: `scripts/check-docs.py`
now enforces that for `docs/experiments/`. A report whose script never got
committed cannot be reproduced or extended, however good its numbers were.

## When something breaks

**Add a file to `scripts/experiments/lessons/`**, named `YYYY-MM-DD-short-slug.md` — same day, while
the mechanism is still clear — then re-run `python scripts/gen-docs-inventories.py`
to refresh the index in `LESSONS.md`. One incident per file, so that two studies
recording lessons in parallel do not conflict. Keep the cost in it, and say
plainly whether it is now *prevented* (a preflight check, a code change) or still
only *advice*. A lesson without a control will recur; saying so is more useful
than implying it is handled.

If the failure is mechanically checkable, add a check to `preflight.sh` rather
than a paragraph anywhere.
