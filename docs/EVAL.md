# Evaluation Guide

VTSearch includes a built-in evaluation framework that measures how well its sorting methods work on demo datasets. This guide covers how to run evaluations, write custom eval scripts, and interpret the results.

> **The default arm is the shipped algorithm.** Every experiment here is a measured *deviation* from what the app does, which only means something if the un-deviated arm matches the app. Where the harness can't call app code directly it copies it, and those copies are pinned by `scripts/check-eval-app-sync.py` — a `./run-tests.sh` gate that fails when an app-side surface moves. See [The Eval Default Arm IS the App](#the-eval-default-arm-is-the-app) below.

## Quick start

Run the full evaluation across all demo datasets:

```bash
python -m vtscore.eval --plot-dir eval_output
```

This will:

1. Download each demo dataset (cached after first run).
2. Run **text sort** (embedding-based ranking) and **learned sort** (the app's detector head trained on simulated votes) evaluations.
3. Print a summary table to the terminal.
4. Save visualisation charts as PNGs in `eval_output/`.

### Finding past studies

Every completed study is written up under `docs/experiments/<date>-<slug>/`, and its launchers live under `scripts/experiments/`. There is deliberately no index file: the date prefix sorts `ls docs/experiments/` chronologically, and each `REPORT.md` opens with its question as the heading and its verdict right under it, so

```bash
head -n 8 docs/experiments/*/REPORT.md
```

is an always-current index. Check it before starting a study — the question may already be answered. A study that is planned but not yet run holds only a `PLAN.md` or `PREREG.md`.

A study keeps in the tree only what its report shows or links; its bulk data (cell tables, measurement dumps, unembedded figures) is listed in its `archived-data.json` and lives on the `experiments-data` release. `python scripts/experiments/study_data.py fetch <study>` puts it back in place (#5404).

## Prerequisites

Install dependencies (if you haven't already):

```bash
bash scripts/install.sh
```

Matplotlib and pandas are required for plot generation and are included in the dev dependencies.

## CLI reference

```
python -m vtscore.eval [OPTIONS]
```

| Flag | Description | Default |
|------|-------------|---------|
| `--datasets ID [ID ...]` | Only evaluate these dataset IDs | all |
| `--mode {text,learned,both}` | Which evaluation to run | `both` |
| `--k K [K ...]` | k values for P@k / R@k | `5 10 20` |
| `--train-fraction F` | Fraction of learned-sort's simulation half that is voted (the other half is held out for test) | `0.5` |
| `--seed N` | Random seed for reproducible splits | `42` |
| `--output FILE` | Write JSON results to FILE | none |
| `--plot-dir DIR` | Save visualisation PNGs to DIR | none |
| `--no-plot` | Disable plot generation | off |
| `--enrich-descriptions` | Use enriched (wrapper-averaged) text embeddings for text-sort. A no-op on embedders that declare no wrappers, which since #3341 is most of them — `siglip`, `clap`, `e5` and `bge` all measured worse enriched than plain | off |
| `--calibrate-count K` | Number of random Train/Calibrate splits for threshold calibration | `2` |
| `--calibration-fraction F` | Fraction of training data reserved for calibration | unset = the app's per-space default (0.3 single-vector / 0.5 patch) |
| `--embedder NAME` | Build each demo dataset with this embedder | empty = the media type's default embedder |
| `--region-voting` | Region-pool learned-sort Good votes from each media's ground-truth box (needs `--embedder <patch>` + a dataset with stored regions, e.g. Visual Genome) | off |
| `--beta B` | Balance (F-beta's beta) learned-sort draws its line at, in `[0.25, 4]`, or `off` for the Inclusion 0 cut | the app's default balance (`DEFAULT_BETA`, 1) |
| `--list` | List available eval datasets and exit | - |

### Examples

```bash
# Text sort only, on image datasets, save JSON
python -m vtscore.eval --mode text --datasets caltech101_s caltech256_a --output results.json --plot-dir eval_output

# Learned sort voting more of the simulation half
python -m vtscore.eval --mode learned --train-fraction 0.7 --seed 123 --plot-dir eval_output

# Learned sort at the Inclusion 0 cut, the line every learned-sort run measured
# before #4971, instead of the app's labels line
python -m vtscore.eval --mode learned --beta off --plot-dir eval_output

# Learned sort with more calibration folds (the "safe thresholds" toggle is gone;
# the trainer's fold-anchored path is now unconditional)
python -m vtscore.eval --mode learned --calibrate-count 4 --plot-dir eval_output

# Region voting on Visual Genome: re-embed with a patch embedder, then have
# each Good vote use the object's ground-truth box instead of the whole image.
# Run once without and once with --region-voting and diff the F1s.
python -m vtscore.eval --mode learned --datasets visual_genome_s --embedder dinov3_patch
python -m vtscore.eval --mode learned --datasets visual_genome_s --embedder dinov3_patch --region-voting

# List available eval datasets
python -m vtscore.eval --list
```

## Available eval datasets

Each eval dataset wraps a demo dataset and defines text queries targeting specific categories (`EVAL_DATASETS` in [`vtscore/eval/config.py`](../vtscore/eval/config.py) is the source of truth; `python -m vtscore.eval --list` prints it). The `_s`, `_m`, `_l` suffixes denote **small**, **medium**, and **large** size variants of the same dataset (more clips = slower evaluation but more statistically robust results); `_a` denotes the **all** (full) variant.

| Eval dataset ID | Media type | Demo dataset | Categories |
|----------------|-----------|--------------|------------|
| `esc50_s` | Audio | esc50_s | All 50 ESC-50 categories (animals, nature, urban, domestic, human) |
| `esc50_m` | Audio | esc50_m | All 50 ESC-50 categories |
| `esc50_l` | Audio | esc50_l | All 50 ESC-50 categories |
| `caltech101_s` | Image | caltech101_s | 25 Caltech-101 categories (airplanes, bonsai, dolphin, helicopter, watch, etc.) |
| `caltech101_m` | Image | caltech101_m | 25 Caltech-101 categories |
| `caltech256_a` | Image | caltech256_a | 25 Caltech-256 categories (backpack, butterfly, camel, giraffe, lighthouse, etc.) |
| `enrico_m` | Image (UI screenshots) | enrico_m | 18 Enrico screen functions (camera, chat, dialer, login, etc.) |
| `enrico_a` | Image (UI screenshots) | enrico_a | 18 Enrico screen functions |
| `rico_screen2words_m` | Image (UI screenshots) | rico_screen2words_m | 16 app genres (Google Play category of the screen's app) |
| `rico_screen2words_a` | Image (UI screenshots) | rico_screen2words_a | 16 app genres |
| `rvl_cdip_m` | Image (document scans) | rvl_cdip_m | All 16 RVL-CDIP document types (letter, form, email, invoice, etc.) |
| `rvl_cdip_a` | Image (document scans) | rvl_cdip_a | All 16 RVL-CDIP document types |
| `visual_genome_s` | Image (multi-label) | visual_genome_s | 40 Visual Genome object categories (person, car, dog, tree, building, etc.); an image can be a positive for several at once |
| `visual_genome_m` | Image (multi-label) | visual_genome_m | 40 Visual Genome object categories |
| `vggface2_faces_s` | Image (faces) | vggface2_faces_s | 40 celebrity identities; `category` = person, so learned sort measures same-person identity matching (train on a few of a person's photos → recover their held-out photos) |
| `vggface2_faces_m` | Image (faces) | vggface2_faces_m | 40 celebrity identities (40 photos/person) |
| `20newsgroups_s` | Text | 20newsgroups_s | 15 topics (sports, science, cars, religion, politics, medicine, etc.) |
| `20newsgroups_m` | Text | 20newsgroups_m | 15 topics |
| `20newsgroups_l` | Text | 20newsgroups_l | 15 topics |
| `ucf101_s` | Video | ucf101_s | 10 UCF-101 actions (ApplyEyeMakeup, ApplyLipstick, Archery, BabyCrawling, BalanceBeam, etc.) |
| `ucf101_m` | Video | ucf101_m | 10 UCF-101 actions |
| `ucf101_l` | Video | ucf101_l | 10 UCF-101 actions |

## Understanding the metrics

### Text sort metrics

Text sort evaluation measures how well embedding-based search ranks clips. For each query:

- **Average Precision (AP):** How well all relevant items are ranked near the top. 1.0 means every relevant item appeared before every irrelevant one.
- **Precision@k (P@k):** Of the top-k results, what fraction is relevant.
- **Recall@k (R@k):** Of all relevant items, what fraction appears in the top-k.
- **Mean Average Precision (mAP):** AP averaged across all queries for a dataset.

### Learned sort metrics

Learned sort evaluation splits each category's media into a simulation half and a held-out test half, votes `--train-fraction` of the simulation half, trains a binary classifier on the simulation half alone, then scores the test half with it. The threshold is fitted on the simulation half's unvoted rest, never on the test half, as in `simulate_voting_iterations` (#5003). It cuts at the line the app draws, the labels line under the default balance, and records that `threshold` and its `beta` with each category's metrics:

- **Accuracy:** Fraction of correct predictions.
- **Precision:** Of items predicted positive, fraction that is actually positive.
- **Recall:** Of actual positives, fraction predicted positive.
- **F1:** Harmonic mean of precision and recall.

### Seeing the errors behind the error rate

An fpr says how often a configuration is wrong. It cannot say whether the
**model** is wrong or the **label** is — and those have opposite remedies (fix
the model vs. clean the dataset and re-run). Set `VTS_DUMP_TEST_SCORES` to a
directory and any voting-iterations run writes one row per scored media —
score, dataset label, threshold in force, source filename, and every category
the dataset annotates on that media — with `VTS_DUMP_TAG` naming the file
(`vtscore/eval/score_dumps.py`). It is off unless asked for: a dump is evidence
for a handful of hand-picked cells, not something a 270-cell array should do.

Two scripts read a dump directory:

```bash
# Ranked false positives / false negatives, with annotations, per cell.
python scripts/experiments/calibration/error_report.py --dumps "$DUMPS"
# The entailment test: are the flagged images enriched for categories that
# cannot occur without the target (clouds without sky)?  If so the "false"
# positives are missing labels.
python scripts/experiments/calibration/label_noise.py --dumps "$DUMPS"
```

`scripts/experiments/calibration/text_baseline.py --dump-dir` writes the same
schema for a *typed* query, so the zero-click path's errors are inspectable the
same way. `scripts/experiments/calibration/launch_errdump.sh` re-runs chosen
cells of a finished study with dumping on, reusing that study's category
selection and exemplar crops so the dumped cell is the same cell.

## Visualisations

When `--plot-dir` is set, the following charts are generated:

### Text sort plots

| File | Description |
|------|-------------|
| `text_sort_map_by_dataset.png` | Horizontal bar chart of mAP per dataset |
| `text_sort_ap_by_query.png` | Bar chart of AP for each individual query |
| `text_sort_precision_at_k.png` | Line chart of Precision@k curves |
| `text_sort_recall_at_k.png` | Line chart of Recall@k curves |

### Learned sort plots

| File | Description |
|------|-------------|
| `learned_sort_f1_by_category.png` | Bar chart of F1 score per category |
| `learned_sort_metrics_breakdown.png` | Grouped bar chart comparing accuracy, precision, recall, and F1 |

### Voting iterations plots (from custom scripts)

| File | Description |
|------|-------------|
| `voting_iterations_cost.png` | Cost curve over voting iterations (mean ± std across seeds) |
| `voting_iterations_fpr_fnr.png` | FPR and FNR curves over voting iterations |

## Writing a custom evaluation script

For more control (looping over parameter values, running voting-iteration simulations, or combining results across experiments), write a Python script that uses the eval API directly.

### Example: sweep over train fractions

This script evaluates learned-sort quality at different vote fractions of the simulation half:

```python
#!/usr/bin/env python
"""Sweep train_fraction and plot learned-sort F1."""

from vtscore.embedding import initialize_models
initialize_models()

from vtscore.eval.runner import run_eval
from vtscore.eval.visualize import plot_eval_results

train_fractions = [0.3, 0.5, 0.7, 0.9]
datasets = ["caltech101_s"]

for frac in train_fractions:
    print(f"\n=== train_fraction={frac} ===")
    results = run_eval(
        dataset_ids=datasets,
        mode="learned",
        train_fraction=frac,
        seed=42,
    )

    # Generate plots for each setting
    plot_eval_results(results, output_dir=f"eval_sweep/frac_{frac}")

    for r in results:
        print(f"  {r.dataset_id}: mean_F1={r.mean_learned_f1:.4f}")
```

Run it with:

```bash
python my_eval_sweep.py
```

### Example: sweep over seeds

Evaluate multiple random seeds to measure variance:

```python
#!/usr/bin/env python
"""Run evaluation across multiple seeds to measure stability."""

from vtscore.embedding import initialize_models
initialize_models()

from vtscore.eval.runner import run_eval
from vtscore.eval.visualize import plot_eval_results

seeds = [1, 2, 3, 42, 100]
datasets = ["caltech101_s", "esc50_s"]
all_results = []

for seed in seeds:
    results = run_eval(
        dataset_ids=datasets,
        mode="both",
        seed=seed,
    )
    all_results.extend(results)

    for r in results:
        line = f"  seed={seed}  {r.dataset_id}"
        if r.text_sort:
            line += f"  mAP={r.mean_average_precision:.4f}"
        if r.learned_sort:
            line += f"  F1={r.mean_learned_f1:.4f}"
        print(line)

# Plot the last seed's results as a representative sample
plot_eval_results(results, output_dir="eval_seeds")
```

### Example: voting iterations evaluation

The voting-iterations evaluation measures how classification quality improves as more votes are cast. This is useful for understanding how many labels a user needs to provide before the model converges.

**Two entry points, two sets of knobs.** `simulate_voting_iterations` (in `vtscore/eval/voting_iterations.py`) runs **one** simulated session — one dataset, one target category, one seed — and takes every knob described below. `run_voting_iterations_eval` is the grid driver the examples use: it loops over datasets × categories × seeds (and over the list-valued arms `strategies`, `trainers`, `styles`, `prevalence_arms`) and forwards only the knobs in its own signature — `inclusion`, `beta`, `sim_fraction`, `calibrate_count`, `calibration_fraction`, `region_voting`, `max_steps`, `atlas_min_node_size`, `seed_scores`, `autopilot_fidelity`, `startup_schedule`, `calibration_seed` and `standalone_cut`, among others. The rest — `acq_inclusion_offset`, `acq_rank_percentile`, `head`, `blend_schedule`, `emit_calibration_metrics`, `skyline_arms`, `pick_sink` and the diagnostic sinks — exist **only** on `simulate_voting_iterations`; passing them to the driver raises `TypeError`. The calibration study runner (`scripts/experiments/calibration/`) calls `simulate_voting_iterations` directly and exposes most of them as `CALIB_*` environment variables.

Votes are cast in the order the app's **Autopilot** would present them — the eval reproduces the real user flow rather than an academic active-learning heuristic. Autopilot seeds the first few positives from text sort when available (pass `seed_scores`, a per-media cosine-to-query ranking), else from a handful of random known-good examples, then gathers the initial negatives and works through the standard Good / Bad / Hard / New phases. The session walks that sort over its **voting half** alone, as the app sorts the dataset it has loaded: the Bad phase's cut, the rank distances its pick measures, a startup schedule's cuts and the pick log's seed ranks are all read over the sim half, so the withheld half's cosines cannot move a vote, while a typed query's Test still reads them from the same map (#5004; a default-arm vote order from a run before it can differ). A dataset with nothing to type, such as a face set whose classes are people, takes the app's **example sort** instead: `seed_examples=K` starts the session from K of the target's photos in the voting half, casts them as its first votes (phase `example`), and ranks by cosine to their centroid, and `stratify_target=True` splits the target's positives on their own so a class with only a few keeps one withheld (`vtscore/eval/example_opening.py`, #4699). `autopilot` is the vote-order strategy every study's headline numbers are read off, and every default-arm result row carries `strategy="autopilot"`. Two experiment arms sit beside it (issue #3954): `autopilot_uncertainty` and `autopilot_maxvar` keep every Autopilot phase and replace only the Hard pick with a rule that reads the detector's posterior spread — the straddle `argmax 1.96·std − |p − cut|` and plain `argmax std` respectively. They run only against a trainer that reports a spread (the `gp_*` trainers below) and raise otherwise, so a run can never attribute the app's own picks to a GP rule.

#### Autopilot fidelity (`autopilot_fidelity`, default `True`)

The simulated user follows the app's **own** phase machine, ported in [`vtscore/eval/autopilot_flow.py`](../vtscore/eval/autopilot_flow.py). That matters because the harness's earlier approximation diverged from the app in ways that changed what studies measured:

| | app (and the harness now) | old approximation |
|---|---|---|
| First trained detector | at quorum — 3 good **and** 4 bad (or, once the Good walk runs dry, a Good and 16 bad, #4731) | at the first `(≥1 good, ≥1 bad)` pair |
| Bad-phase pick | the **text sort's acquisition cut** (Select `hard` on a text sort: the mixture midpoint the route sends as `acq_threshold`, not the guarded line it paints green, #4136) | the bottom of the sort |
| Hard-phase pick | nearest the cutoff **by rank** | nearest **by score** |
| Hard → New | when the *smart* and *stable* indicators go green | alternating on step parity |

The first row is the one that bites. Because the app stays on the text sort through the Bad phase, its first learned sort has ≥2 votes of each class unless the Good walk ran dry on one Good (#4731), so it hits the calibrator's `too_few_default` path only there — while the old harness trained at 3 good + 1 bad on every trajectory and recorded a flat 0.5 threshold there. Issue #2788's cold-start degenerate thresholds were largely that artifact.

Two columns make this visible in the output:

- **`phase`** — the Autopilot phase after that vote (`good` / `bad` / `more` / `hard` / `new` / `spot` / `done` / `exhausted`); `example` on an example-sort opening's own votes, `s0`, `s1`, … inside a startup schedule, and `prompt` / `check` on spot-check rows.
- **`app_trained`** — `1` exactly when the app would have had a trained detector on screen. **A threshold recorded where this is `0` is one no user would ever see**, so any analysis of threshold quality should filter on it.
- **`detector_tier`** — which detector a Test at that click gives, as the app's label quota decides it ([`ML.md`](ML.md#below-the-label-quota-the-goods-centroid-not-a-head)): `centroid`, the Goods' centroid, from the first Good until the votes hold 3 Goods and 4 Bads (or a Good and 16 Bads, #4731); `query` on a run that opens on a typed query, as the app's typed-query detector does (#4758), from the first vote until the quota, its rows the typed query's own set on the withheld half (the opening's text-sort cosines cut where the text sort draws its display line there at the run's balance, `_query_test`); else `trained`. The row's test metrics are that detector's, cut where Test cuts it on the withheld half. The Train side (acquisition, the lights, the spot check) runs on the trained head wherever there is a Good and a Bad, as the Train view's learned sort does, and the study extras that vary the trained head's cut (variant rows, fit-quality, rank and precision frames) ride only trained steps under the quota.

  The knobs that move the tier or price another one on the same session:

  | Knob | Default (the app) | What it does |
  |---|---|---|
  | `label_quota` | on | `False`: no row before the first Good and Bad, the head from there (before #4643). |
  | `query_tier` / `CALIB_QUERY_TIER` | on | `False` / `0`: a typed-query run gives the centroid from the first Good (before #4758). |
  | `query_hold` / `CALIB_QUERY_HOLD` | 30 | On a single-vector space (the run's media carry no patch grid) the query holds past the quota until this many votes, as the app's does (#4736, `label_quota.query_holds`); `0` hands over at the quota. A held click's untagged row is the typed query's set with `app_trained` 0, and the head's rows ride tagged `gmm_variant="hold:head"`, so a hand-over rule is priced on the same session. |
  | `query_miss` / `CALIB_QUERY_MISS` | on | On a patch space, once the typed query has missed (a Good, and a Bad before the Goods quota) the untagged row is the Goods' centroid's until the quota, as the app's Find gives it (#4800, `label_quota.query_missed`); `0` keeps the typed query's set. |
  | `CALIB_CENTROID_LINE` | the app's rule | The centroid's line: `count_gap_floor` is the line before #4783, `count` the one before #4759, `midpoint` the one before #4732. |
  | `centroid_line_variants` / `CALIB_CENTROID_LINE_VARIANTS="<rule>@<beta>,..."` | none | Adds rows tagged `gmm_variant="centroid_line:<rule>@<beta>"` pricing other centroid lines; `<rule>~pool@<beta>` / `<rule>~unvoted@<beta>` draws the line over every media of the cell, voted ones in or out (#4779), still scored on the withheld half. A typed-query run writes them from its first Good, and the variant `query@<beta>` adds the typed query's own set from the first vote (#4800). |
  | `handover_shadow` / `CALIB_HANDOVER_SHADOW=1` | off | Past the quota, adds the centroid's Test as rows tagged `gmm_variant="handover:centroid"` and a rule's evidence in `task_NNNN__handover.csv`, so one session prices every centroid-to-head hand-over rule (#4776). |
  | `set_overlap_sink` / `CALIB_SET_OVERLAP=1` | off | At every click a trained head scored the pool, records how the head's set and the typed query's split the unvoted media (`task_NNNN__overlap.csv`), and where each pick sat in them at the click before it (the pick log's `picked_in_head`, `picked_in_query`, ...), so a judge for the hold that reads the votes is priced on the same session (#4816). |

Metrics are still recorded at every trainable step in both modes — fidelity changes the *vote order* and the `app_trained` flag, not measurement coverage.

#### Stopping point and stopping cost (issue #3560)

A study's headline number is its **final cost** — the metric at the last click of `CALIB_MAX_STEPS` (usually 150). Nobody in the app is asking that question. The app has *stopping rules*: when the Smart, Stable and Span indicators are all green (or, since #4878, when the labels line expects `STOP_LEVEL` of F-beta at the balance; or, since #4903, at `STOP_CAP` votes) the phase becomes `done` and the panel says *"All quality indicators are green. You can continue labeling or export your results."* (or quotes the F-beta it expects). The number a user leaves with is the metric **there**, at a click count they did not pick in advance. So every simulated-user study owes two more numbers:

| | what it is | column it comes from |
|---|---|---|
| **stopping point** | the click at which the rules first fired — the *width* | first `t` where `phase == "done"` |
| **stopping cost** | the metric at that click — the *height* | that row's objective, `fbeta` on a run that drew its line at a balance and `cost` on one that did not (`average_precision` beside it) |

`phase` has been on every metric row since the harness adopted the app's phase machine, so **this needs no re-run**: a finished study's cells carry it already. Columns beside it (added by #3560) say *which* rule was doing the holding, which `phase` alone cannot:

- **`smart` / `stable` / `span`** — the three indicator lights that produced the phase, each `red` / `yellow` / `green`. `done` is all three green (or the line's level met, #4878) and `new` is Span alone short, so the phase already encodes those; what it cannot encode is whether Smart, Stable or both hold a run in `hard`.
- **`span_level` / `span_depth` / `span_target`** — the raw atlas counts the Span light thresholds, and the bar itself: the light is exactly `span_level >= span_target`, where the target is `min(autopilot_goal_diversity, span_depth)`. The bar is emitted rather than assumed to be 40 because it is a per-run knob *and* a per-step quantity (a small atlas caps it, and the atlas grows).

All of them are blank / `-1` where no phase machine ran (a non-`autopilot` strategy, `autopilot_fidelity=False`) and throughout a startup schedule's rounds, which own the phase without consulting the indicators — *not measured* is deliberately distinguishable from *not green*. They cost nothing: the phase machine already computed all three every step and threw them away.

##### The margins: how close each rule came

A light is a threshold test, and a study that records only the answer cannot say whether a run sat one noisy window short of green for a hundred clicks or was never within reach of it. Those are different findings — the first wants the rule de-flapped, the second wants a better embedding — and "Stable held it" is three of them, because Stable is a conjunction of three gates. So every step also carries the continuous quantities those gates are thresholds **on**:

| column | the gate it feeds | green when |
|---|---|---|
| `smart_slope` | F-beta flatness (#5134; the cost's before) | `<= SMART_RISE_THRESHOLD` (+0.015) |
| `smart_slope_t` | that slope against the window's own scatter (#3832) | `< SMART_SLOPE_T` (2.0) |
| `stable_confident_flip_rate` | confident-flip average over the pool | `< STABLE_RATE_THRESHOLD` (0.005) |
| `stable_max_confident_flip_rate` | its worst single step in the window | `< STABLE_MAX_THRESHOLD` (0.01) |
| `stable_flip_rate_early` / `stable_flip_rate_late` | the raw rate having stopped falling | not (`late >= 0.005` and `late < STABLE_FALLING_RATIO × early`) |
| `stable_flip_rate` | — (the window average the app's panel quotes) | |

Smart's two are a **disjunction** — either clears it — so neither alone says a run was held; Stable's three and Span's one are conjunctive. Every value is read off the dict the rule itself built (`autopilot_flow.smart_detail` / `stable_detail`), so a margin can never disagree with the light beside it, and `NaN` means the rule **declined to fit** one — below `MIN_PER_CLASS` votes of either class, under three model steps for Smart, under five entries for Stable — never "far from green".

**These cost nothing to record and cannot be back-filled.** The rules fit all of it on every step already, because the vote order depends on the lights they produce; before #3560 the numbers were discarded and only the three-way status kept. But unlike `phase` and the lights, a finished study's cells *cannot* be enriched with them — the slope and flip-rate windows are per-step state the run threw away — so this half, and only this half, needs a re-run to answer on an old grid. `stopping.has_margins()` is the guard, and every margin function returns an empty frame rather than a table of zeros on a frame that predates them.

##### Smart in the objective's units (#5134)

Since #5134 Smart reads each of the last 10 models' **best F-beta on the current votes**, at the run's balance (F1 with none), and goes green once it stops **rising**: the trend-and-scatter test (#3832) with its sign turned over (`cost_trend.rising_status_from_scores`, `SMART_RISE_THRESHOLD` +0.015). Until then it read each model's FPR + FNR at its own Inclusion 0 cut, green once it stopped falling. `AutopilotFlow(smart_signal=)` (`simulate_voting_iterations(smart_signal=)`, `CALIB_SMART_SIGNAL`) keeps the A/B's readings as arms; `"fbeta_best"`, the default, is the app ([the A/B](experiments/2026-10-20-smart-fbeta-5134/REPORT.md)).

| reading | what one point is | kept as |
|---|---|---|
| `fbeta_best` (the app) | the best F-beta at the run's beta any cut through one cached model's scores on the current votes reaches | the window, re-scored each step |
| `cost` (the app before #5134) | FPR + FNR of that model at its own Inclusion 0 cut, green once it stops *falling* | the window |
| `ap` | average precision of that model on the current votes | the window |
| `fbeta` | F-beta at the run's beta at that model's own Inclusion 0 cut | the window |
| `heldout_ap` | AP of the calibration folds' held-out scores of the votes | one value per retrain, the last `SMART_WINDOW` kept |
| `heldout_fbeta` | the best F-beta of those scores | per retrain |
| `line` | the labels line's expected F-beta (`stop_level_reading`) | per retrain, from the first one whose class model cuts |

None reads the served line. `smart_readings=True` (`CALIB_SMART_READINGS=1`) records every reading's trend on each row without moving the light: `smart_<reading>_slope` (relative slope), `_mean`, `_t` and `_last` (the newest point). Green is `slope <= 0.015` or `t < 2` (for `cost`: `slope >= -0.015` or `t > -2`), so any threshold can be re-read offline, and `slope * mean` is the slope in the reading's own units. `NaN` means not recorded, or too few points or votes to fit. `smart_slope` / `smart_slope_t` are the margins of the light the run reads. `line` needs a balance.

`stopping.margins()` / `summarise_margins()` / `margin_table()` report them per run over its **held** steps (everything before its first fire, or the whole trajectory when it never fired — a converged run sits at a comfortable margin for the rest of its budget and would otherwise swamp the steps that actually held it), and within those only over the stretch that waits on each gate: Smart's and Stable's over Hard, Span's over New. One sign convention throughout: **positive means the gate is satisfied with that much room, negative means the run is short by that much**, whichever way the underlying inequality points. Beside each margin is the share of those steps the gate was satisfied at, pooled over runs, which is what separates a flap from a wall, and `stable_block_warmup` / `stable_block_avg` / `stable_block_max` / `stable_block_falling` attribute Stable's not-green Hard steps to Stable not measuring yet or to one of its three gates in the rule's own short-circuit order.

**Read each light in the stretch that waits on it** (2026-10-14). Autopilot's walk to Done is a sequence of stretches, each waiting on something different: the opening on its targets, **Hard** on Smart and Stable, **New** on Span alone, and two band walks whose rounds are clicks — a mid-session check on weak separation and the check before Done. New *is* the atlas walk that fills Span, so Span is never green in Hard and is never a gate that fails: it is a walk of so many clicks. A count of not-green over every held click therefore names Span binding in every session, which is what the 2026-10-08 review reported. `stopping_points` now splits each run's clicks before the stop into those stretches (`clicks_opening` / `clicks_hard` / `clicks_new` / `clicks_check` / `clicks_done_check`, which add up to the stop), reads `blocked_smart` / `blocked_stable` over Hard's clicks only, names the light Hard last waited on (`hard_exit_light`), counts each light's lapses from green before the stop (`lapses_*`) and what broke Done after it (`done_lapses_*`), and reads the objective where Hard last ended (`{metric}_at_hard_exit`). `stopping.stage_table()` and `binding_note()` report them; there is no `blocked_span`.

**Do not truncate the run at `done`.** The simulated user keeps clicking to the budget exactly as before, because the stretch past the stopping point is what says whether stopping there was the right call — and because arms can only be compared at a fixed `t`.

[`scripts/experiments/calibration/stopping.py`](../scripts/experiments/calibration/stopping.py) is the one implementation of the derivation, and there are three reasons not to write it again by hand:

- **The rules flap.** The phase is derived from the current labelset every step, never latched, so a run can go `done` on one vote and back to `hard` on the next. The app announces on the **first** fire and never re-announces, so first-fire is the faithful stopping point (`t_stop`); `t_sustained` reports the stricter reading, and `n_done_episodes` says how far apart the two are.
- **They often never fire**, which makes every average a **censored** statistic. Averaging the runs that stopped excludes precisely the slow ones, so the mean flatters, and flatters harder the worse the arm is. `summarise()` leads with the fire *rate*, and its `km_t_stop` is a Kaplan–Meier median that carries the non-firing runs as censored at their own budget — returning `NaN`, honestly, when fewer than half of them ever fired.
- **Cost at the stop and cost at the budget are different numbers, and the difference has a sign.** Report both, paired within run.
- **A spot check prompted mid-session hides the phase.** Its rounds are clicks with `phase == "prompt"` (`spot_check="weak"` and the check before Done, `done_check`, below), so the phase machine's answer on those steps is not in the row. The check before Done also moves the first `done`: the flow reaches `spot` when every light goes green, and `done` only once the check has closed, as the app reports Done. `stopping.UNREAD_PHASES` carries the last read state across them: a prompted check can neither fire the rules nor split a `done` stretch into two episodes.

Each stop is also held against **the run's own best** (`{metric}_best`, `{metric}_t_best`, and the distance `{metric}_from_best` and `{metric}_clicks_past_best`, the last two columns of `stopping_table`): a rule that fires long after the run's best costs users clicks, one that fires long before it costs them quality. The best is the extreme of a noisy series, so read the two together.

The State of the App analyzer ([`scripts/experiments/state_of_app/analyze.py`](../scripts/experiments/state_of_app/analyze.py)) writes all of it on every review: `stops.csv`, `margins.csv`, and a "Where the app said stop" block in `summary.md`, per production path and per band, at the review's objective, with a run that never trained kept as one the rules never stopped.

Pass the result to `curves.quality_vs_clicks(..., stops=...)` and the mandatory averaged figure carries a `▽` at each arm's median stopping click, with the arms that mostly never stopped named in the caption rather than silently unmarked.

#### The reporting line (`beta`, default: the app's own balance)

The app draws a detector's line at its **balance** — F-beta's beta, 1 unless the user changes it (#4413; see [`docs/ML.md`](ML.md#threshold-calibration)). So does the default arm: `simulate_voting_iterations(beta=None)` resolves through `resolve_line_knobs` to the balance at `DEFAULT_BETA`, the value an unset user setting resolves to, and the line comes from the app's own `fit_labels_line` (#4452). A number pins a balance: a preset, `beta=0.25|1|4`, or any beta in [0.25, 4] (`CALIB_BETA` on the calibration runner).

**A row measures "train on X, how well would I Find on Y".** The **Train side** is the line the step's votes draw: the class model the step's fold orderings (the held-out scores of its votes) imply, with the corpus side fitted on the sim set this step's model scored. The **test side models Find**: each row's metrics cut the withheld half at the threshold Find would draw there, the same class model with the corpus side re-fitted on that half (`LabelsLine.on_corpus` over the test scores), not at the Train side's. Knobs that widen the world it is measured in (#4452):

- `sim_size` (`CALIB_SIM_SIZE`) trains on a seeded subsample of the sim set, the withheld half kept whole, so Train and Find corpora differ in size.
- `test_score_sink` (`CALIB_SAVE_TEST_SCORES=1`, files `task_*__testscores.npz`) saves the withheld half's scores and labels, with Train's threshold and the class model, at the last ordinary step and after the check, so Find corpora of other sizes and prevalences can be drawn from it after the run. Each snapshot also carries the step's calibration folds (`fold_scores` / `fold_labels` / `fold_index`) and the images' `ids`, so a line rule that changes the class model is replayed rather than re-run. `CALIB_SAVE_TEST_SCORES_AT=5,10,...` keeps a snapshot at those clicks too (`t5`, `t10`, ...), and the full-label ceiling leaves its own (`ceiling`) when the skyline arm runs (#4490).
- **The Test arm** (#4523; `line_test_sink`, `CALIB_LINE_TEST=1`, files `task_*__linetest.csv`, schema `LINE_TEST_COLUMNS`) runs Tune's autopilot (`vtscore.training.thresholds.line_test`, the app's own phase machine, delegated) on that withheld half after the last ordinary click, its uniform picks answered from the truth, to Done. It writes one row per session: the picks and stop per phase, the ranges at Done beside the truth, and the verdict's reading beside the oracle's (`vtscore.eval.line_test_arm`). Test votes never train, so the trajectory is unchanged; the arm runs only under a balance. It is a pure function of the saved snapshot, so `line_test_arm.row_from_snapshot` replays a `__testscores.npz` entry at any `LineBudgets` to the same row — a grid over targets and budgets is a replay, not a rerun — and the `line_test.budgets_default` mirror pins the app's defaults the arm resolves to.
- A forced `walk_shape` (`walk`, `advisory` or `trim`) runs the count line the app drew before #4452 — the mixture's F-beta argmax under the balance's cap, 32 at beta ≤ 1 and 128 above, through `balance_line` — as a measurement arm, with the test side at the Train side's threshold.

Neither the app nor the default arm builds the #4220 precision estimate any more (#4362), so the harness records no pick provenance for it.

**A second corpus to Find on** (#4960; `find_corpus`, `CALIB_FIND_CORPUS=<dataset>` on the calibration runner). The withheld half is the corpus the votes came from, so it cannot show what a detector does on a corpus unlike it.

- **The corpus.** `find_corpus` hands the simulator another dataset's medias in the run's embedder (the runner reads that dataset's pickle for the cell's embedder); its cell of the same name is the Find corpus (`bus@medium` in `o365_better` for a `coco_better` run, #4815), or the whole dataset where it has none, where the target has no positives. `find_mix` (`CALIB_FIND_MIX`) draws it: `only`, the default, is that cell alone; `equal`, `0` or any share adds that many times the withheld half's size of it to the withheld half as extra haystack, a draw seeded off the run's seed. The corpus is only ever scored: it never enters the pool, the votes, the atlas or the folds.
- **What every row gains.** At every click the detector the click's Test gives (its `detector_tier`) scores the Find corpus and its line's corpus side is refitted there, as a Find there draws it, and the row carries `FIND_CORPUS_COLUMNS` (`find_fbeta` and its presets, `find_precision`, `find_recall`, `find_n_flagged`, `find_average_precision`, `find_n_pos`, `find_n_neg`) beside the in-corpus columns, which do not move (`tests_lib/eval/test_find_corpus.py` pins them byte-identical, and the picks with them). A corpus with no positive scores NaN F-beta and still counts `find_n_flagged`.
- **The domain-shift flags.** At the snapshot clicks (`find_flag_steps`, `CALIB_FIND_FLAGS_AT`, by default `CALIB_SAVE_TEST_SCORES_AT`'s list: the first click at or past each, named in `flags_at`) the row also reads the app's two flags against that corpus (`FIND_FLAG_COLUMNS`), each a delegated call into the report the app's route calls: the Training-domain overlap (`domain_shift_report` against the sim pool's vectors, as the app reads the reference dataset's; NaN only when the pool is under 40 media) and the Evidence coverage (`evidence_coverage_report` from the votes' vectors and the Find cut, its verdict ranked against the training baseline the app records on the detector, #5089: here the sim pool's unvoted media found as a Find there draws it, `pool_baseline`, drawn fresh at each snapshot click). The Evidence row also carries the counts behind its shares, by predicted class, over the corpus and over the pool (`evidence_n*`, `evidence_pool_n*`, `evidence_n_{pos,neg}_beyond_pool`). How the routes call them is ported, and the `find.training_domain` and `find.evidence_coverage` mirrors pin it. `analyze_autopilot_4671.py` reads the `find_*` columns the way it reads the in-corpus ones and adds a flag table per arm.

**The spot check** (`spot_check`; default `None`, which resolves to the app's for the run's medias: `"weak"`, or `"end"` on face crops). The walk is the app's, delegated — the harness calls `balance_schedule` / `SpotCheck.start_balance` / `balance_state` — and pinned against the analysis that priced it by the schedule, likely-range and band-walk mirrors in `scripts/check-eval-app-sync.py` ([`ML.md`](ML.md#threshold-calibration) has the rule). Each band's 5 picks are answered from ground truth and **cast as votes** (provenance `check`), and the model retrains on them. Under the balance the walk is advisory at every preset, as the app's is (`check_shape`, #4452): it runs as an audit, `checked` on the F-beta peak, and the line stays the labels' line. Until a check ends every row reports `unchecked`, which is exactly what a headless run exports.

| `spot_check` | What runs |
|---|---|
| `"off"` | No check, ever. |
| `"end"` | Once the voting steps are spent (`max_steps` reached, or the pool exhausted), the final line is checked: the harness's measurement, never a click (no app user starts that check). One row per band, `phase == "check"`, `t` still counting every vote cast, so the rows sit past `max_steps`; `check_audited` says what was audited. They are not clicks, so the calibration scripts' per-arm loader, `_cells_io.load_arm`, drops them unless called with `keep_check=True`, and the standard curves and viewer end at the last click (#4364). The default on face crops, where the app never prompts (#4799: `weak_check_prompts`, `WEAK_CHECK_OFF_MEDIA_TYPES`; `production_spot_check` reads the medias' `media_type` the way the app reads its detector's), because #4778 priced the prompt a loss there; `CALIB_SPOT_CHECK=weak` is the arm before. |
| `"weak"` | The end check, plus the check the app's Autopilot runs mid-session (#4496): at the first ordinary step from `weak_min_t` votes (`WEAK_CHECK_MIN_VOTES`, 10) whose labels line separates weakly — `LabelsLine.separation`, d' = (Good mean - Bad mean) / the spread the line was cut with (the folds' own gap, 0 or below, when they rank the held-out Goods no higher than the Bads, #4899), below `weak_separation` (`WEAK_SEPARATION_D`, 1.5). Its picks are clicks: they count in `t` and the voting budget, its rows carry `phase == "prompt"` and read as ordinary clicks, and frames record on them. It starts only when its opening bands fit the budget left, and a deeper band that would overrun the budget ends it there, so a run's clicks still end at `max_steps`. The default elsewhere. |

The weak check's other knobs: `weak_repeat` (default 0, the app's `WEAK_CHECK_REPEAT` since #4909) prompts only while no check has ended, and a positive value prompts again that many votes after the last check ended while separation stays weak (25 is the app before #4909); `weak_phase="learned"` (the default) prompts only once Autopilot has left its text-sort opening (`good`/`bad`/`more`), where the app trains no detector, and `"any"` is the arm first priced. The calibration launcher sets them with `CALIB_SPOT_CHECK`, `CALIB_WEAK_D`, `CALIB_WEAK_MIN_T`, `CALIB_WEAK_REPEAT` and `CALIB_WEAK_PHASE`. `walk_shape` forces a count-line arm: `walk`, the full walk whose end moves the line; `advisory`; or `trim`, where the walk may only step shallower from the bands holding the line, and the line takes its end.

**The check before Done** (`done_check`, default `True`, the app's; `CALIB_DONE_CHECK=0` is the arm before it): the first time every quality light is green the Autopilot flow reaches `spot` (`AutopilotFlow`, the port of the app's Spot Check step), the simulated user runs the check then and there, as clicks with `phase == "prompt"` rows inside the budget, and the flow then reports `done`. Once a run, on every media type, face crops included. It is skipped, the flow going straight on to `done`, when a check ended in the last `WEAK_CHECK_COOLDOWN` votes (`done_check_due`, the app's rule) or when its opening bands do not fit the budget left; `spot_check="off"` and the Inclusion arm never run it. It is not priced: the app runs it by the owner's call. The check is seeded off the run's RNG after every trajectory draw, so the voting steps are byte-identical with or without it.

`beta="off"` (`NO_BALANCE`) is the **Inclusion arm** — the line at `inclusion`, which is what every study before #4245 measured. An arm that sweeps `inclusion` has to pass it: under a balance the line is the set the balance keeps, so a non-zero `inclusion` would only re-weight `cost`, and is refused (#4361). The app has no counterpart since #4269 retired Inclusion as a user preference; the arm is a measurement arm only, and it never checks. The calibration runner spells the knob `CALIB_BETA` (unset, `off`, or a number), and `python -m vtscore.eval.al_benchmark` spells it `--beta` (`off` or a number).

Each row records what the balance said:

| Column | Meaning |
|---|---|
| `beta` | The balance that drew the line; NaN on the Inclusion arm. |
| `floor_status` | `unchecked` or `checked`; empty with no balance line. The `floor_*` names predate the balance. |
| `floor_count` | How many unvoted items the line keeps: what clears the labels' threshold, possibly 0; -1 with no line. |
| `range_lo` / `range_hi` | The check's likely range for the kept set's precision; NaN before a check ends. |
| `check_labelled` / `check_right` | The labels behind that range; -1 before. |
| `check_stale` | 1 once the retrain after the check moved the set under it, 0 while it holds, -1 with no finished check. |
| `check_audited` | The set the last walk audited, its end as a band edge (#4427); under the advisory check the line keeps `floor_count`, not it; -1 before a check. |

The base calibration row also carries both sides of the line (#4452) and the labels line's own reading of how well it does at the row's beta (#4878); every other row type carries them as NaN. Every one is a reading the app has at each retrain, except `find_efbeta`:

| Column | Meaning |
|---|---|
| `train_threshold` | The Train side's threshold; the row's `threshold` is then Find's, on the withheld half. |
| `train_prevalence` / `find_prevalence` | The prevalence the class model estimated on the sim set / on the withheld half. |
| `line_efbeta` | The expected F-beta at the line's cut over the sim set's unvoted items, what `corpus_cut` maximises; it falls as the votes take the corpus's positives. |
| `line_efbeta_corpus` | The same class model's line over the whole sim set read as unlabelled: a Find on a corpus like it. |
| `line_efbeta_fresh` | The class model's own normals (`parametric_expected_fbeta`). |
| `line_separation` | The labels' d'. |
| `line_head_count` | 1 where the head's count line served, 0 where the class model's cut did, -1 with no line. |
| `find_efbeta` | A diagnostic the app cannot read: the expected F-beta of Find's line on the withheld half, from that half's scores alone. |

Read as a trend, none of these times a stop better than counting clicks; read as a level, `line_efbeta` puts the clicks where they pay better than Done does ([the stop-signal reading](experiments/2026-10-14-stop-signal-4878/REPORT.md)).

**Done's stop rule** (`stop_rule`, `CALIB_STOP_RULE`, #4878) is `"lights|level"` by default, the app's: Done once every light is green, or once the step's labels line expects `stop_level` (`CALIB_STOP_LEVEL`, `STOP_LEVEL` 0.5) of F-beta at the run's balance, from the first click its class model cuts the line. The harness reads that with the app's own `stop_level_met`, and `AutopilotFlow(stop_rule=)` hands `next_phase` the lights that make it decide so, as `smart_gate` does. The other arms are `"lights"` (the app before #4878), `"level"` (the level alone) and `"lights+level"` (both); a session with every light green that the rule does not stop reads Hard. Without a balance or the Autopilot flow there is no level to read and the default is the lights alone; the two arms that wait on the level refuse such a run. `preflight_knobs.py` lists a non-default rule. **The click cap** (`stop_cap`, `CALIB_STOP_CAP`, `STOP_CAP` 200, #4903) backs every rule: a session that holds that many votes goes to the check before Done and to Done whatever the lights and the level say, as the app's `STOP_CAP_VOTES` does; `CALIB_STOP_CAP=none` is no cap. It never binds inside the eval's 150 clicks ([the wider-stop reading](experiments/2026-10-15-wider-stop-4903/REPORT.md)).

#### The acquisition cut (`acq_target_p`, `acq_inclusion_offset`; default: whatever `vtscore.training.thresholds` ships)

The selector and the metrics read **different thresholds**. Reporting and every emitted metric stay on the reporting line; the Hard and New picks read a separate acquisition cut, mirroring production (see [`docs/ML.md`](ML.md#the-acquisition-cut) for the mechanism and the measured effect). Which cut, by default, is the app's:

- **Under a balance** (every default run): the **target pick precision** (#3546), the score where the labels line's corpus posterior falls below `ACQUISITION_TARGET_PRECISION` (0.5), through the app's own `target_precision_threshold`; `resolve_acquisition_target` is the harness's side of the `acquisition.target_precision` mirror. A step with no labels line falls back to the offset cut below. `acq_target_p=<number>` (`CALIB_ACQ_TARGET_P`) pins another target; `"off"` forces the offset cut.
- **The offset cut**: the same fold-anchored fit re-cut `acq_inclusion_offset` inclusion steps from the reporting line's inclusion — `inclusion` on the Inclusion arm, and under a balance the inclusion the line derives to (`inclusion_for_threshold`). The default is `ACQUISITION_INCLUSION_OFFSET`, the shipped value, **not** `0`, so an unconfigured run measures what users get (read the constant, not a number written here). `acq_inclusion_offset=0` (`CALIB_ACQ_INCLUSION_OFFSET=0`) is the pre-#2876 control where one threshold did both jobs, which is also what any pre-#2876 study implicitly ran at: reproducing one byte-for-byte means passing it explicitly, the way `autopilot_fidelity=False` reproduces the pre-fidelity harness. `acq_origin` (`CALIB_ACQ_ORIGIN`, `line` or `inclusion`) says where the offset counts from.
- **The F-beta argmax share** (#4409): `ACQUISITION_ARGMAX_FACTOR` is `None` since the #4427 revert, so it does not ship. `acq_p_crossing=<number>` (`CALIB_ACQ_P_CROSSING`) runs it as an arm, at that share of the depth of the mixture's F-beta argmax over the unvoted ranking (`acquisition_threshold`; `resolve_acquisition_factor` is the `acquisition.balance_cut` mirror; `docs/experiments/2026-10-01-acquisition-fbeta-4409/REPORT.md`).

Three columns make the lever verifiable rather than assumed, all measured in the **pool** distribution the selector ranks:

- **`acq_threshold`** — the cut the picks actually saw that step (equal to `threshold` on steps with no fold-anchored fit to re-cut, ~5% of steps, concentrated in the cold start — the schedule blend has no inclusion-aware form).
- **`acq_pool_percentile`** / **`report_pool_percentile`** — where the two cuts sat in the ranking.

The pool is scored in **the same geometry the cuts are fitted in** — the style's region max-pool on a patch dataset, the whole-image vector on a single-vector one — because the Hard pick locates its cutoff by comparing scores against the threshold *absolutely*, so a ranking and a cut in different spaces put the cutoff index in the wrong place. Patch-dataset studies published before #2943, including the #2876 acquisition-inclusion report, scored the pool whole-image against cuts fitted on pooled scores, and measured that mismatch.

The direction is counter-intuitive (negative offset → *higher* cut → *more* positives, because the pick reads the threshold as a rank position), so a sign error would otherwise look exactly like the lever not working. `acq_rank_percentile` is the alternative parameterisation — pin the cut at a fixed quantile of the simulation-set scores — and it requires `acq_inclusion_offset=0`, since the two name the same cut. It is measured and **worse**; it exists as an arm, not as an option.

#### The opening (`startup_schedule`, default: the app's own)

Autopilot's **opening** — the clicks spent before the first learned sort — is what decides how many positives a run has when its detector is first trained, and a Good-starved run is the one that fails (issue #3267). It is a parameter, so a study can ask whether a different opening mines better.

The parameterisation rests on one observation: both of today's opening phases are the *same operation* at two different cuts. The Good phase's `top` select is the rank-space `hard` select against a cut placed above every score; the Bad phase's is that select against the seed sort's own fitted GMM, split at the production midpoint. So an opening is a list of rounds, each naming how many clicks to spend and where on the sort — which is what a schedule spells:

| round | meaning |
|---|---|
| `g3` / `b4` | stay until 3 goods / 4 bads exist (a **global** count, as in the app) |
| `n8` | stay for 8 clicks, whatever they turn out to be |
| `+dry1/8` | …or until the round's last 8 picks held fewer than 1 good: the walk has run dry (`g` and `n` rounds only; `g20+dry1/8@top`) |
| `@top` | cut above every score — the top of the sort |
| `@mid` | the shipped GMM midpoint, i.e. every cosine sort's cutoff |
| `@k-3` | that same fitted GMM, split at **inclusion −3** |
| `@q0.05` | the sort's 5th rank percentile, named directly |

`PRODUCTION_STARTUP` (`"g3@top,b4@mid"`; no More walk since #4740) is today's opening, and [`vtscore/eval/startup_schedule.py`](../vtscore/eval/startup_schedule.py) is the full reference. `startup_schedule=None` — the default — leaves every trajectory byte-for-byte what it was; the explicit production spec is *required* to reproduce a default run click for click, which `tests_lib/eval/test_startup_schedule.py` asserts and `scripts/check-eval-app-sync.py`'s `autopilot.startup_default` mirror keeps true — except where the Good walk runs dry. Since #4731 the app's Good phase also ends after `moreDryRun` picks in a row without a Good, with a Good in hand, which the grammar cannot spell (a `+dry<m>/<w>` stop fires with no Good in hand). The mirror declares the divergence; `good_dry_run="off"` on the default arm is the pre-#4731 opening.

Rounds appear in the `phase` column as `s0`, `s1`, …, and `app_trained` is `0` throughout one: a round is on the seed sort by construction, so the app would have no detector on screen however many votes have been cast. Every row also carries **`startup_schedule`**, so a pooled frame says which arm it came from without depending on the directory it was read out of.

`@k` and `@q` are not redundant. `@k` is the arm that could *ship* — the app measures its acquisition cut in inclusion steps and has no rank-position knob — but how far a given inclusion moves the pick is a property of the fitted mixture, so on a steep sort the whole usable range can land inside a couple of rank percent. `@q` names the position directly, which is what establishes whether *position* is the mechanism before asking whether `k` is a usable handle on it.

#### The calibration draw (`calibration_seed`, default: the app's own pin)

The cell `seed` moves the **data** — which media are voted, in what order, and which half is held out — and nothing else. Everything downstream of the votes is pinned, and deliberately so: production splits Train/Calibrate off a fresh `RandomState(CALIBRATION_SPLIT_SEED)` (42), because issue #2934 fixed the unseeded global draw that let one detector's verdicts move between two runs on identical votes. The harness pins the same 42 so that the default arm is the app, per [The Eval Default Arm IS the App](#the-eval-default-arm-is-the-app) below.

That pin has a cost the harness could not previously report. Every number this eval quotes is one draw of that split, and nothing says how wide the distribution around it is. `calibration_seed` (issue #3794) is the measurement:

```python
# The usual grid sweeps data at a pinned split.  This sweeps the split at pinned
# data — one cell seed, many draws — which is the only arrangement that isolates it.
frames = [
    run_voting_iterations_eval({"vg": medias}, seeds=[7], calibration_seed=c)
    for c in range(20)
]
```

Read the spread across those runs as the noise floor: a contrast smaller than it is not a finding. Every row records the resolved value in the **`calibration_seed`** column (42 on a default run), so the frames concatenate and stay tellable apart.

Two things it is *not*:

- **Not extra realism.** "Simulate the app's randomness here" has nothing to simulate — the app is pinned at these sites too. Reseeding them on a default run would measure a detector nobody ships, which is exactly the failure the drift gate below exists to prevent.
- **Not a knob on the model fit.** The shipped head is a convex liblinear solve in the primal, which does not read its `seed=42` at all; reseeding *it* is a no-op. The Train/Calibrate split is the one place a seed still decides something. (The retired MLP head was the opposite — its weight-init seed moved the ranking, which is part of why it is retired; see [`docs/ML.md`](ML.md#the-heads-which-one-is-shipped-and-why).)

`tests_lib/eval/test_calibration_seed_arm.py` pins both halves: an unset knob reproduces a default run row for row, and an explicit one actually redraws the split.

#### What every step measures

Each metric row carries the operating point at that step's threshold — `cost` (the inclusion-weighted `FPR + FNR`), `fpr`, `fnr`, `precision`, `recall` and `f1` — plus the counts they come from (`n_test_pos`, `n_test_neg`, `n_flagged`) and the two threshold-independent ranking metrics, `auroc` and `average_precision`, which isolate "how good is the ranking" from "how good is the threshold".

`precision` / `recall` / `f1` come from one definition, [`vtscore.eval.calibration_metrics.detection_metrics`](../vtscore/eval/calibration_metrics.py), shared by both row builders so they cannot come to disagree. Three conventions in it are load-bearing:

- **`precision` is `NaN` when the detector flags nothing** — genuinely undefined, since there is no retrieved set to be right about. A 0 would drag an average down for a reason that is not "the model was wrong"; a 1 is simply false.
- **`f1` uses `2TP / (2TP + FP + FN)`**, so it needs no precision and is well-defined at 0 when nothing is flagged.
- **`recall` is exactly `1 - fnr`** and is emitted anyway: it is the word a reader picks off a menu, and asking them to invert an FNR in their head is where reading errors come from.

**The objective** (#4427, #4584) rides beside them: `fbeta`, F-beta of the withheld half above the row's threshold at the row's own `beta` — the balance that drew the line — and `fbeta_b025` / `fbeta_b1` / `fbeta_b4`, the same returned set scored at each preset (`FBETA_COLUMNS`). One definition, `calibration_metrics.fbeta_metrics`, computed from the counts through the app's own `fbeta_score`: a cut that returns nothing scores 0, a sample with no positives is `NaN`, and `fbeta` is `NaN` where no balance drew the line (`beta="off"`, a skyline row). `fbeta_from_rates` is the same number read back off `precision` and `recall`, which is how the analyzers fill the columns on cells written before them. `cost` stays on every row as a diagnostic: it is priced at the run's Inclusion (0 on the calibration runner) whatever beta drew the line, so on a balance run it measures a preference the app no longer holds. The calibration analyzers decide and draw on `fbeta` whenever a frame carries a beta, and on `cost` only when it does not (`scripts/experiments/calibration/objective.py`), and `preflight.sh --resolve-delta` sizes an A/B with that metric's σ.

`DETECTION_METRICS` in the same module carries each metric's label and its **direction**, and is what the report figures and the interactive viewer read — so nothing downstream decides for itself which way is "better".

#### The supervised skyline and training regret (`skyline_arms`)

The calibration frame carries two oracles, one per objective. `oracle_threshold` / `oracle_cost` / `oracle_fpr` / `oracle_fnr` is the **cost** objective's: the cut with the lowest weighted FPR + FNR on the test half. It prices the two error rates, so on a rare class it cuts deep (recall about 0.85 at precision about 0.1 on COCO Better) and is no ceiling on F-beta. `oracle_fbeta` (at the row's beta) and `oracle_fbeta_b025` / `_b1` / `_b4` are the **F-beta** objective's: the best F-beta any cut of the test ranking reaches, with `fbeta_oracle_threshold` / `_fpr` / `_fnr` the cut that reaches `oracle_fbeta` (`calibration_metrics.oracle_fbeta_metrics`, #4654). A row's `fbeta_*` over its `oracle_fbeta_*` is the State of the App's "share of the best cut".

`cost = oracle_cost + regret` splits a step's cost into "the ranking" and "the cut", but `oracle_cost` still conflates two causes that call for opposite fixes: **no linear head in this embedding can separate the class**, and **a head could, but 10–200 clicks did not find it**. Pass `skyline_arms=["skyline_train_full"]` (requires `emit_calibration_metrics`) to split them:

```
cost = skyline_oracle_cost   (learnability floor: embedding + head capacity)
     + training_regret       (headroom the interactive loop left on the table)
     + regret                (what the cut rule gave away on the ranking it got)
```

`skyline_train_full` is the standard supervised-skyline arm: the **same head, through the same trainer**, trained on the entire simulation split with **full ground-truth labels** and evaluated on the untouched test split — same hypothesis class, same features, full supervision, disjoint eval. It delegates to `_train_and_calibrate` rather than fitting an estimator of its own, so the gap it measures is "fewer labels" and not "a different trainer" (and so it needs no `Mirror(...)` entry to stay drift-proof).

`training_regret` is defined on **rankings** — `oracle_cost(mortal) − oracle_cost(skyline)` — which is what makes the telescope exact. Routing the skyline through a calibrated cut would re-mix in the term `regret` already isolates, so the skyline's own threshold is the test oracle's: `cost == oracle_cost` and `regret == 0` on a skyline row **by construction**. Read its `oracle_cost`, never its `regret`.

Four things to know before reading the numbers:

- **The row is per *run*, not per step.** A skyline is vote-independent, so it is emitted once per `(cell, seed)` at `t = 0` with `app_trained = 0`, tagged in `gmm_variant` like every other variant family. The four decomposition columns (`skyline_oracle_cost`, `skyline_oracle_cost_honest`, `training_regret`, `training_regret_honest`) are then filled on *every* row of the run, so the identity holds within a row. Cost: one extra fit per arm per cell, not per click.
- **Negative `training_regret` is legal.** Unlike the threshold oracle, the skyline is not a per-run optimum over the same object — region votes carry box information an image-labelled skyline lacks, and small samples get lucky. Nothing clamps it.
- **The terms are sum-pinned**, so the same caution `vtscore.eval.row_metrics.operating_metrics` documents for `rule_inefficiency` / `calibration_shift` applies verbatim: they share noise by construction, so don't read one half moving as an effect when the knob also moves the yardstick.
- **The gap bundles several causes** — how many labels, *which* labels (the acquisition policy), and the full split's imbalance. Read it as "headroom left by the interactive loop", not as one cause.

`skyline_test_xfit` is the optional bracket partner, and it is **cross-fitted**: the test split is partitioned into folds and each item is scored by a head that never saw it. A naive train-on-test fit is not an option — a ~769-parameter linear head on a test set of comparable size can shatter near-arbitrary labelings, so it would report near-zero cost on a genuinely unlearnable class and its "regret" would measure `d / n_test` rather than learnability. It is the SVM analogue of `honest_test_oracle`, which does the same thing one level down for the *cut*; its pooled scores share a ranking but not a calibrated scale, so read it for `oracle_cost` / `auroc` / `average_precision` only.

**v1 is the whole-image column.** Full supervision hands out *image* labels, but the mortal `max_patch` flow trains on GT-box-pooled vectors, so a patch column's skyline is a design decision — "oracle boxes + all images", or the multiple-instance problem of which patch of a positive image is the positive — and issue #3321 holds it open. The harness warns and skips on a patch style rather than improvising one, so a sweep over both columns still gets the decomposition on the column that has one.

Set `CALIB_SKYLINE_ARMS=skyline_train_full` to turn the arm on in the calibration experiment runner.

#### The pick log (`pick_sink`)

Pass a list as `pick_sink` to get one row per **click** (columns: `PICK_COLUMNS`) — what was picked, whether it was a positive, and where on the seed sort it came from. The main frame starts at the first Good (#4643: from there Test gives the Goods' centroid; before it, at the first Good and Bad), because before it there is no detector, no threshold and no metrics row; so the start of the opening is exactly the part it does not record. An opening that never finds a Good emits **no main row at all**, which is a result about that opening rather than a missing cell. A run that opens on a typed query has a row at every vote since #4758, its Test giving the query's own set until the quota.

Pass `autopilot_fidelity=False` to reproduce studies published before the flow was aligned (the Max-Patch, MLP-vs-SVM, and Inclusion-knob reports); that path is byte-for-byte the old behaviour. New studies should leave it on.

```python
#!/usr/bin/env python
"""Evaluate learned-sort cost over simulated voting iterations."""

from pathlib import Path

from vtscore.embedding import initialize_models
initialize_models()

from vtscore.datasets.loader import load_demo_dataset
from vtscore.eval.visualize import plot_voting_iterations
from vtscore.eval.voting_iterations import run_voting_iterations_eval

# Load datasets
datasets_to_eval = ["esc50_s", "caltech101_s"]
dataset_clips = {}
for name in datasets_to_eval:
    medias = {}
    load_demo_dataset(name, medias)
    dataset_clips[name] = medias

# Run the voting iterations eval
df = run_voting_iterations_eval(
    dataset_clips=dataset_clips,
    seeds=[1, 2, 3, 42, 100],           # multiple seeds for averaging
    categories={                          # specific categories to test
        "esc50_s": ["rain", "frog"],
        "caltech101_s": ["dolphin", "flamingo"],
    },
    inclusion=0,                          # inclusion bias setting
    sim_fraction=0.5,                     # fraction used for simulated voting
)

# Save raw data
df.to_csv("eval_output/voting_iterations.csv", index=False)
print(df.groupby(["dataset", "category"])["cost"].agg(["mean", "std"]))

# Generate plots
paths = plot_voting_iterations(df, output_dir="eval_output")
for p in paths:
    print(f"Saved: {p}")
```

#### Region voting

On a **patch-embedder** dataset that carries ground-truth boxes (Visual Genome,
loaded with `embedder_name="dinov3_patch"`), pass `region_voting=True` to make
each simulated Good vote train on the object's box instead of the whole image:

```python
medias = {}
load_demo_dataset("visual_genome_s", medias, embedder_name="dinov3_patch")

# Same dataset, two runs — the only difference is the Good-vote training vector.
baseline = run_voting_iterations_eval({"vg": medias}, seeds=[1, 2, 3], region_voting=False)
region   = run_voting_iterations_eval({"vg": medias}, seeds=[1, 2, 3], region_voting=True)
```

A Good vote uses the **minimal box covering every annotated instance** of the
target category (two apples → one box around both); images with no annotated box
fall back to the whole-image vector. Scoring is region-aware (max-pool over the
image's score rows) in both runs, so the comparison isolates region voting's
effect. Region voting is a no-op on single-vector datasets (no `patch_grid` to
sample a patch from).

**Visual Genome's ground truth.** It is the one multi-label demo. Its vocabulary
is a fixed list of the 100 most frequent VG object names
(`VISUAL_GENOME_CATEGORIES`), the same for every slice. Membership is
closed-world: an image is positive for a category when that name is in its
annotated object set, negative otherwise, so VG's incomplete annotations leave a
few accepted false negatives. Object names reach a category through a case- and
plural-folding heuristic; a spelling it does not fold is lost, and VG's own
`names` lists hold one name per object, so they offer no aliases (#3618). The
boxes ride on each media as `media["regions"]`, a list of
`{"box": [x0, y0, x1, y1], "label": category}` in normalised `[0, 1]`
coordinates.

### Example: voting iterations from pickle files

If you have pre-exported dataset pickle files, load them directly:

```python
#!/usr/bin/env python
"""Run voting iterations eval from pre-exported pickle files."""

from vtscore.embedding import initialize_models
initialize_models()

from vtscore.eval.visualize import plot_voting_iterations
from vtscore.eval.voting_iterations import run_voting_iterations_eval_from_pickles

df = run_voting_iterations_eval_from_pickles(
    dataset_paths={
        "my_audio": "data/embeddings/my_audio_dataset.pkl",
        "my_images": "data/embeddings/my_image_dataset.pkl",
    },
    seeds=[1, 2, 3],
)

plot_voting_iterations(df, output_dir="eval_output")
```

## Using the visualisation API directly

The `plot_eval_results` and `plot_voting_iterations` functions can be called from any Python code:

```python
from vtscore.eval.visualize import plot_eval_results, plot_voting_iterations

# Standard eval results -> list of PNG paths
paths = plot_eval_results(results, output_dir="my_plots")

# Voting iterations DataFrame -> list of PNG paths
paths = plot_voting_iterations(df, output_dir="my_plots")
```

Both functions:
- Create the output directory if it doesn't exist.
- Return a list of `Path` objects pointing to the generated PNGs.
- Skip chart types that don't apply (e.g., no learned-sort plots if only text-sort was run).

## `trainer` vs `head`: which knob picks what

The voting simulation takes two knobs that both sound like "which model?". They are not the same question, and until issue #3764 they also shared the string `"mlp"`.

**`trainer` picks the pipeline.** `trainer="app"` (the default) runs VTSearch's own: `train_model` plus production fold calibration, the arm every study's headline numbers are read off. Any `svm_*` value instead fits a standalone sklearn/cuML estimator that thresholds itself, from the registry in `vtscore/eval/sweep_trainers.py`; any `gp_*` value fits a standalone Gaussian-process classifier the same way (issue #3954) and additionally exposes its posterior spread as `StepModel.predict_std`, which is what the two uncertainty strategies read. Both standalone families take the plain cross-calibration cut: their fold models are not the app's head, so the fold-anchored fusion has nothing to anchor on and `safe_thresholds` lands them on the blend fallback — compare them against `trainer="app", safe_thresholds=False`, not against the shipped cut, when the question is the head. The `gp_*` arms additionally take `standalone_cut="rank"`, which carries that cut from the fold models to the final one by rank instead of as a raw score; [`docs/experiments/2026-09-17-gp-head-3954/REPORT.md`](experiments/2026-09-17-gp-head-3954/REPORT.md) measures both transfers (and the blend) on a GP and finds the rule, not the head, decides the outcome. The arm was spelled `"mlp"` before #3764, which named no MLP — its default head is the linear SVM — so the old spelling is now accepted as an input alias and normalised away; result rows always record `app`.

**`head` picks the model that pipeline fits**, and applies to `trainer="app"` alone:

| `head` | What is fitted |
|---|---|
| `linear_svm` | `Linear(d, 1)` fitted by liblinear, the plain SVM: the shipped head until #4785, now the arm before it. |
| `linear` | The same `Linear(d, 1)` fitted by balanced BCE — the logistic head the SVM replaced (#2790/#2809). |
| `mlp` | An auto-sized hidden layer, BCE — the head shipped before #2790 (#2781). |
| `linear_logreg` | The same `Linear(d, 1)` fitted by converged, balanced logistic regression (scikit-learn), the in-loop test of #3197 (#4114). |
| `linear_svm_prior` | The SVM's objective with the Goods' centroid made a cheap direction: one extra feature, 10 x the dot with the unit centroid of the rows being fitted, folded back into a `Linear(d, 1)`, so a head on a few labels starts at the centroid (#4776). **The shipped detector head** since #4785; `head=None` resolves here. Every fit takes it: Test, the picks, the folds and the line. On a typed-query run given its `query_vector` (single-vector media, no example opening) every fit also takes the typed query's direction at `svm_prior.QUERY_PRIOR_SCALE`, as the app's typed-query detector does since #4859; `CALIB_QUERY_PRIOR=0` (`arms_query_prior.query_prior_scale(0)`) is the arm before. |

Passing `head=` alongside an `svm_*` or `gp_*` trainer is an error: those arms fit their own estimator and have no head to choose, which is why their rows carry an empty `head` column.

Two names are worth reading slowly. `svm_linear` is a **trainer**: a standalone SVM scored through its own `predict_proba`. `linear_svm` is a **head**: the app's `Linear(d, 1)` whose weights come from liblinear, scored and thresholded exactly as production does. A run of `trainer="svm_linear"` and a run of `trainer="app", head="linear_svm"` fit a similar boundary by very different routes, and only the second measures the shipped detector.

The label-curve and timing sweeps (`run_label_curve_eval`, `run_timing_benchmark`) use the *other* registry, where a trainer is a bare estimator and `"mlp"` genuinely is one. See [`vtscore/docs/packages/eval.md`](../vtscore/docs/packages/eval.md) for the side-by-side table.

## The Eval Default Arm IS the App

Every experiment in this framework is a *deviation* from the shipped algorithm — a different Good-vote geometry, a different blend schedule, a different acquisition cut. A deviation is only interpretable against a baseline that is the real thing, so the framework's default arm has to be exactly what the app ships. When the app moves and the harness doesn't, the studies don't fail loudly; they keep producing plausible numbers about a detector nobody uses, and everything measured after the drift is quietly devalued.

Three ways the harness relates to the app, in descending order of safety:

| | How | Can it drift? |
|---|---|---|
| **Delegated** | The harness calls the app's function. `MaxPatchStyle.good_vec` / `.bad_vecs` / `._rows_for_media` are thin wrappers over `pool_box_from_media`, `bad_negative_vecs`, and `media_score_rows`. | No — by construction. |
| **Ported** | The app's logic is re-implemented, because the original is unreachable or unusable. `vtscore/eval/autopilot_flow.py` ports the phase machine from `AutopilotStateService.checkPhaseTransition` (TypeScript — nothing to import) and the three indicators from `vtscore.detectors.labeling_progress` (wrapped in an interactive, lock-guarded per-(dataset, detector) cache a simulation can't use). | Yes — a copy goes stale the moment the original moves. |
| **Default resolution** | The harness resolves "no explicit arm" to whatever the app currently defaults to: `style=None` → `max_patch` on a patch dataset, `blend_schedule=None` → `production_schedule_for(...)`, `calibration_fraction=None` → `production_split_for(...)` (0.3 single-vector / 0.5 patch, keyed on `patch_grid` presence — the harness's spelling of the app's `supports_patch_regions` predicate). | Yes — the app changes its default and the harness keeps serving the old one *under the name "default"*. |

Prefer delegation whenever it's possible; it's the only fix that can't rot.

### The drift gate

`scripts/check-eval-app-sync.py` pins a digest of each mirrored surface — Python symbols by parsing the module, TypeScript blocks by brace-matching an anchor — and `./run-tests.sh` fails when one changes. It parses rather than imports, so it's dependency-free and takes ~0.3s. A failure names the mirror, both sides of it, and what to re-check:

```
  * autopilot.phase_machine  [ported, app-changed]
      app:     frontend/src/app/services/autopilot-state.service.ts::checkPhaseTransition(
      harness: vtscore/eval/autopilot_flow.py::next_phase
      The phase ordering and every transition trigger of the simulated Autopilot user. ...
```

**Both sides are pinned**, because a copy stays faithful only while neither half moves without the other, and the reason says which half did:

| Reason | What happened | What to do |
|---|---|---|
| `app-changed` | The original moved. | Reconcile the harness copy to it. |
| `harness-changed` | The copy moved while the original stood still. | Re-read the two against each other. A harness edit can silently re-point the default arm with the app untouched — that is exactly what #2923 was, and the gate was green throughout. |
| `app-unpinned` / `harness-unpinned` | A new mirror, or a `--update` never run. | Run `--update`. |
| `unresolvable` | One side no longer exists under the name the manifest claims. | Fix the manifest, or restore what was deleted. |

Reconcile, then re-pin:

```bash
python scripts/check-eval-app-sync.py --update
```

Digests ignore comments, docstrings, and formatting (including the magic trailing comma `ruff format` adds when it wraps a line, though not the comma that makes `(x,)` a tuple), so only real logic changes trip the gate. Re-pinning without reading the other side defeats the whole thing — the digest is a prompt to check, not a checkbox.

A handful of mirrors name a harness anchor too coarse to digest — one function serving several mirrors and carrying arm knobs that no mirror is about (today `_safe_threshold_for_step`, and `simulate_voting_iterations` for the label-quota mirror). Those declare `no_harness_pin=<reason>` and keep the app-side pin alone; the reason prints with the mirror when it trips. A `ported` mirror may never opt out, since the harness side of a hand copy *is* the copy. When a coarse anchor's blind spot starts to matter, the fix is to extract the reproduction into a helper small enough to pin — what `_resolve_production_defaults` is — not to digest a thousand lines.

### Adding and diverging

A new mirror is a new `Mirror(...)` entry in `MIRRORS`, plus `--update`. Give it a `note` that says what to re-check when either side moves, not just what the code is. Name every top-level symbol the reproduction spans — `file.py::GOOD_TARGET,BAD_TARGET` — since watching one of a pair is half a mirror. The app side lists them the same way (`py:package.module.first,second`), and a reproduction that lives on a method is named where it stands, `file.py::Class.method`, rather than lifted out of its class.

When the harness *intentionally* differs from the app at a mirror, record why in `divergence=`. That doesn't exempt it from the digest — both sides are still pinned — but the text prints whenever the mirror trips, so whoever reconciles it next knows which differences are deliberate. The ported indicators use this: they take their histories as arguments rather than reading a `_ProgressCache`'s `steps`, which is plumbing, not a rule change.

Named experiment arms (`whole_image`, `max_patch_hac`, `max_patch_pca_hac`) are *supposed* to differ from the app — that's what makes them arms. This gate is about the default arm only.

## Re-measuring a decision taken on an older metric

The objective has changed under the studies more than once, and it will again. Until late September 2026 every detector decision was read on `cost = w_fp · FPR + w_fn · FNR` (`inclusion_cost_weights` in `vtscore/training/thresholds/knobs.py`), nearly always at Inclusion 0, so at `FPR + FNR`, as a paired Δcost against a 0.01 tolerance. The preference then became a precision floor (#4224, #4269, #4272) and, on 2026-10-01, an F-beta balance (#4413; presets 1/4, 1 and 4 since #4448). Since #4427 the objective is **the F-beta, at the preset's beta, of the withheld half above the threshold the app holds**, and since #4452 that threshold is the labels line ([`ML.md`](ML.md#threshold-calibration)).

#4581 asked which Cost-era decisions would have gone differently on F-beta. [#4582's re-scoring](experiments/2026-10-06-cost-fbeta-rescore-4582/REPORT.md) holds the answer one decision at a time; the ones it handed on were rerun as [#4583](experiments/2026-10-07-calsplit-4583/REPORT.md), [#3546](experiments/2026-10-07-acquisition-cut-3546/REPORT.md) and [#4359](experiments/2026-10-07-smart-bound-4359/REPORT.md). The rules below are what that pass settled, written for the next metric change.

### What decides whether a decision is owed a re-measurement

A decision taken on an older metric is owed a run on the current objective when both hold:

1. **It is still on the shipped path.** A rule that only ever placed something the app has since stopped using is retired, not re-measured. At the Cost → F-beta switch, every rule that only placed the *line* (the GMM midpoint, the blend, the fused rank transfer) was retired by #4452's labels line, and the [F1 ladder](experiments/2026-10-05-ladder-fbeta-4519/REPORT.md) had already read them. What survived was whatever the labels line still reads (the calibration folds), whatever the fused estimator still feeds (acquisition, the Smart light, the no-class-model fallback), and the opening.
2. **What it moved is where a cut lands or what gets asked**, not ranking quality alone. A verdict read on oracle cost, AUROC or AP agrees with any monotone metric and is settled. A verdict whose ship rule was a cost tolerance, or whose gain was an FNR drop against flat FPR (a lower cut), is the kind F-beta re-weighs, and at three betas it can re-weigh three ways.

Numerical choices (EM tolerances, solver settings) and data, infrastructure and timing studies move no cut, so they are settled by construction. So is a study that priced a control the app has retired: the Inclusion-knob studies priced a knob that no longer exists (#4269), and the floor-era rules priced a promise the balance gave up on purpose (#4413).

Two lessons from that switch bound every rerun. The *reading* of a metric decides as much as the metric: #4409 shipped on a rank-count re-draw of the line and was reverted on the threshold the app holds (#4427). And a bench's prevalence is a knob of its own (#4201): the 5% pool made the ladder's midpoint rungs look like progress whatever the metric.

### How a rerun is run

- **The default arm is today's app** ([above](#the-eval-default-arm-is-the-app)): the balance, the labels line, the weak check. Each preset runs its own sessions (`CALIB_BETA` ∈ {0.25, 1, 4}) and is scored at its own beta.
- **Score the objective**: F-beta of the withheld half above the threshold the app holds, unchecked and after the check, with AP beside it (a cut decision should leave AP alone) and the returned set's size and precision, which is what the user sees. Never a rank-count re-draw of the line.
- **Bench and pools**: `coco_better`, 144 cells × 5 seeds, 150 clicks, the user's pool at 1% (`CALIB_HAYSTACK_PREVALENCE=0.01`) with the withheld half at the bench's own 0.44%. Where the old decision was taken at another prevalence (several Cost-era ones were at 5 to 7%), add that pool once, so the metric and the prevalence are read apart.
- **Pair by cell and seed**, same commit for every arm, and size the grid with the objective's σ, not the old metric's. [#4584](experiments/2026-10-07-objective-sigma-4584/README.md) measured it on balance-era pairs and set the defaults `preflight.sh --resolve-delta` sizes with (0.13 / 0.08 / 0.10 at beta 1/4 / 1 / 4). A grid sized on the cost σ (about 0.04 per cell, #3840) resolves two to three times coarser on F-beta than it was meant to.
- **Re-score before re-running.** Every Cost-era row carries `precision` and `recall`, so F-beta at any beta is derivable from the cells that still exist on `/expscratch`. Do that first and hand only the decisions that flip to a grid; #4582 is the worked example.
