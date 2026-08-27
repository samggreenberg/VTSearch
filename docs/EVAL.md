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
2. Run **text sort** (embedding-based ranking) and **learned sort** (neural net trained on simulated votes) evaluations.
3. Print a summary table to the terminal.
4. Save visualisation charts as PNGs in `eval_output/`.

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
| `--train-fraction F` | Fraction of clips used for training in learned-sort | `0.5` |
| `--seed N` | Random seed for reproducible splits | `42` |
| `--output FILE` | Write JSON results to FILE | none |
| `--plot-dir DIR` | Save visualisation PNGs to DIR | none |
| `--no-plot` | Disable plot generation | off |
| `--enrich-descriptions` | Use enriched (wrapper-averaged) text embeddings for text-sort | off |
| `--calibrate-count K` | Number of random Train/Calibrate splits for threshold calibration | `2` |
| `--calibration-fraction F` | Fraction of training data reserved for calibration | `0.5` |
| `--embedder NAME` | Build each demo dataset with this embedder (empty = media-type default) | `` |
| `--region-voting` | Region-pool learned-sort Good votes from each media's ground-truth box (needs `--embedder <patch>` + a dataset with stored regions, e.g. Visual Genome) | off |
| `--list` | List available eval datasets and exit | - |

### Examples

```bash
# Text sort only, on image datasets, save JSON
python -m vtscore.eval --mode text --datasets caltech101_s caltech256_a --output results.json --plot-dir eval_output

# Learned sort with a different train/test split
python -m vtscore.eval --mode learned --train-fraction 0.7 --seed 123 --plot-dir eval_output

# Learned sort with safe thresholds and calibration tuning
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

Each eval dataset wraps a demo dataset and defines text queries targeting specific categories. The `_s`, `_m`, `_l` suffixes denote **small**, **medium**, and **large** size variants of the same dataset (more clips = slower evaluation but more statistically robust results); `_a` denotes the **all** (full) variant.

| Eval dataset ID | Media type | Demo dataset | Categories |
|----------------|-----------|--------------|------------|
| `esc50_s` | Audio | esc50_s | All 50 ESC-50 categories (animals, nature, urban, domestic, human) |
| `esc50_m` | Audio | esc50_m | All 50 ESC-50 categories |
| `esc50_l` | Audio | esc50_l | All 50 ESC-50 categories |
| `caltech101_s` | Image | caltech101_s | 25 Caltech-101 categories (airplanes, bonsai, dolphin, helicopter, watch, etc.) |
| `caltech101_m` | Image | caltech101_m | 25 Caltech-101 categories |
| `caltech256_a` | Image | caltech256_a | 25 Caltech-256 categories (backpack, butterfly, camel, giraffe, lighthouse, etc.) |
| `visual_genome_s` | Image (multi-label) | visual_genome_s | ~40 Visual Genome object categories (person, car, dog, tree, building, etc.); an image can be a positive for several at once |
| `visual_genome_m` | Image (multi-label) | visual_genome_m | ~40 Visual Genome object categories |
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

Learned sort evaluation simulates voting, trains a binary classifier, then measures on a held-out test set:

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

This script evaluates learned-sort quality at different train/test split ratios:

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

Votes are cast in the order the app's **Autopilot** would present them — the eval reproduces the real user flow rather than an academic active-learning heuristic. Autopilot seeds the first few positives from text sort when available (pass `seed_scores`, a per-media cosine-to-query ranking), else from a handful of random known-good examples, then gathers the initial negatives and works through the standard Good / Bad / Hard / New phases. `autopilot` is the only vote-order strategy; every result row carries `strategy="autopilot"`.

#### Autopilot fidelity (`autopilot_fidelity`, default `True`)

The simulated user follows the app's **own** phase machine, ported in [`vtscore/eval/autopilot_flow.py`](../vtscore/eval/autopilot_flow.py). That matters because the harness's earlier approximation diverged from the app in ways that changed what studies measured:

| | app (and the harness now) | old approximation |
|---|---|---|
| First trained detector | at quorum — 3 good **and** 4 bad | at the first `(≥1 good, ≥1 bad)` pair |
| Bad-phase pick | the **text sort's cutoff** (Select `hard` on a text sort) | the bottom of the sort |
| Hard-phase pick | nearest the cutoff **by rank** | nearest **by score** |
| Hard → New | when the *smart* and *stable* indicators go green | alternating on step parity |

The first row is the one that bites. Because the app stays on the text sort through the Bad phase, its first learned sort always has ≥2 votes of each class, so it can never hit the calibrator's `too_few_default` path — while the old harness trained at 3 good + 1 bad on every trajectory and recorded a flat 0.5 threshold there. Issue #2788's cold-start degenerate thresholds were largely that artifact.

Two columns make this visible in the output:

- **`phase`** — the Autopilot phase after that vote (`good` / `bad` / `hard` / `new` / `done` / `exhausted`).
- **`app_trained`** — `1` exactly when the app would have had a trained detector on screen. **A threshold recorded where this is `0` is one no user would ever see**, so any analysis of threshold quality should filter on it.

Metrics are still recorded at every trainable step in both modes — fidelity changes the *vote order* and the `app_trained` flag, not measurement coverage.

#### The acquisition cut (`acq_inclusion_offset`, default: whatever `vtscore.training.thresholds` ships)

The selector and the metrics read **different thresholds**. Reporting and every emitted metric stay at `inclusion`; the threshold handed to the picks is re-cut at `inclusion + acq_inclusion_offset` from the same fold-anchored fit. This mirrors production, which decoupled the two jobs in PR #2876 — see [`docs/ML.md`](ML.md#threshold-calibration) for the mechanism and the measured effect.

The default is `ACQUISITION_INCLUSION_OFFSET` — the shipped value, **not** `0` — so an unconfigured run measures what users actually get. (PR #2876 shipped `-3`; PR #2891 cut it to `-1` after a second environment rejected `-3`. Read the constant, not a number written here.) Pass `acq_inclusion_offset=0` for the pre-#2876 control where one threshold did both jobs; that is also the value the study's `prod` arm ran at. Note that this changes what a re-run of any *pre-#2876* study measures: those runs were all implicitly at offset 0, so reproducing one byte-for-byte means passing it explicitly, the same way `autopilot_fidelity=False` reproduces the pre-fidelity harness.

Three columns make the lever verifiable rather than assumed, all measured in the **pool** distribution the selector ranks:

- **`acq_threshold`** — the cut the picks actually saw that step (equal to `threshold` on steps with no fold-anchored fit to re-cut, ~5% of steps, concentrated in the cold start — the schedule blend has no inclusion-aware form).
- **`acq_pool_percentile`** / **`report_pool_percentile`** — where the two cuts sat in the ranking.

The pool is scored in **the same geometry the cuts are fitted in** — the style's region max-pool on a patch dataset, the whole-image vector on a single-vector one — because the Hard pick locates its cutoff by comparing scores against the threshold *absolutely*, so a ranking and a cut in different spaces put the cutoff index in the wrong place. Before #2943 the pool was scored whole-image while every cut was fitted on pooled scores; since a max over ~197 patch rows dominates the single whole-image row, the whole pool sat below the cut, both percentiles pinned at `1.0` on every patch step, and the simulated picks came from systematically higher-ranked items than the app's. Patch-dataset studies published before that fix — including the #2876 acquisition-inclusion report — measured that mismatch.

The direction is counter-intuitive (negative offset → *higher* cut → *more* positives, because the pick reads the threshold as a rank position), so a sign error would otherwise look exactly like the lever not working. `acq_rank_percentile` is the alternative parameterisation — pin the cut at a fixed quantile of the simulation-set scores — and it requires `acq_inclusion_offset=0`, since the two name the same cut. It is measured and **worse**; it exists as an arm, not as an option.

#### The opening (`startup_schedule`, default: the app's own)

Autopilot's **opening** — the clicks spent before the first learned sort — is what decides how many positives a run has when its detector is first trained, and a Good-starved run is the one that fails (issue #3267). It is a parameter, so a study can ask whether a different opening mines better.

The parameterisation rests on one observation: both of today's opening phases are the *same operation* at two different cuts. The Good phase's `top` select is the rank-space `hard` select against a cut placed above every score; the Bad phase's is that select against the seed sort's own fitted GMM, split at the production midpoint. So an opening is a list of rounds, each naming how many clicks to spend and where on the sort — which is what a schedule spells:

| round | meaning |
|---|---|
| `g3` / `b4` | stay until 3 goods / 4 bads exist (a **global** count, as in the app) |
| `n8` | stay for 8 clicks, whatever they turn out to be |
| `@top` | cut above every score — the top of the sort |
| `@mid` | the shipped GMM midpoint, i.e. every cosine sort's cutoff |
| `@k-3` | that same fitted GMM, split at **inclusion −3** |
| `@q0.05` | the sort's 5th rank percentile, named directly |

`PRODUCTION_STARTUP` (`"g3@top,b4@mid"`) is today's opening, and [`vtscore/eval/startup_schedule.py`](../vtscore/eval/startup_schedule.py) is the full reference. `startup_schedule=None` — the default — leaves every trajectory byte-for-byte what it was; the explicit production spec is *required* to reproduce a default run click for click, which `tests_lib/detectors/test_startup_schedule.py` asserts and `scripts/check-eval-app-sync.py`'s `autopilot.startup_default` mirror keeps true.

Rounds appear in the `phase` column as `s0`, `s1`, …, and `app_trained` is `0` throughout one: a round is on the seed sort by construction, so the app would have no detector on screen however many votes have been cast. Every row also carries **`startup_schedule`**, so a pooled frame says which arm it came from without depending on the directory it was read out of.

`@k` and `@q` are not redundant. `@k` is the arm that could *ship* — the app has an Inclusion knob and no rank-position knob — but how far a given inclusion moves the pick is a property of the fitted mixture, so on a steep sort the whole usable range can land inside a couple of rank percent. `@q` names the position directly, which is what establishes whether *position* is the mechanism before asking whether `k` is a usable handle on it.

#### The pick log (`pick_sink`)

Pass a list as `pick_sink` to get one row per **click** (columns: `_PICK_COLUMNS`) — what was picked, whether it was a positive, and where on the seed sort it came from. The main frame starts at the first *trainable* step, because before one Good and one Bad vote coexist there is no model, no threshold and no metrics row; so the opening is exactly the part it does not record. An opening that never finds both classes emits **no main row at all**, which is a result about that opening rather than a missing cell.

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

## The Eval Default Arm IS the App

Every experiment in this framework is a *deviation* from the shipped algorithm — a different Good-vote geometry, a different blend schedule, a different acquisition cut. A deviation is only interpretable against a baseline that is the real thing, so the framework's default arm has to be exactly what the app ships. When the app moves and the harness doesn't, the studies don't fail loudly; they keep producing plausible numbers about a detector nobody uses, and everything measured after the drift is quietly devalued.

Three ways the harness relates to the app, in descending order of safety:

| | How | Can it drift? |
|---|---|---|
| **Delegated** | The harness calls the app's function. `MaxPatchStyle.good_vec` / `.bad_vecs` / `._rows_for_media` are thin wrappers over `pool_box_from_media`, `bad_negative_vecs`, and `media_score_rows`. | No — by construction. |
| **Ported** | The app's logic is re-implemented, because the original is unreachable or unusable. `vtscore/eval/autopilot_flow.py` ports the phase machine from `AutopilotStateService.checkPhaseTransition` (TypeScript — nothing to import) and the three indicators from `vtscore.detectors.labeling_progress` (wrapped in an interactive, lock-guarded single-detector cache a simulation can't use). | Yes — a copy goes stale the moment the original moves. |
| **Default resolution** | The harness resolves "no explicit arm" to whatever the app currently defaults to: `style=None` → `max_patch` on a patch dataset, `blend_schedule=None` → `production_schedule_for(...)`. | Yes — the app changes its default and the harness keeps serving the old one *under the name "default"*. |

Prefer delegation whenever it's possible; it's the only fix that can't rot.

### The drift gate

`scripts/check-eval-app-sync.py` pins a digest of each mirrored app surface — Python symbols by parsing the module, TypeScript blocks by brace-matching an anchor — and `./run-tests.sh` fails when one changes. It parses rather than imports, so it's dependency-free and takes ~0.3s. A failure names the mirror, both sides of it, and what to re-check:

```
  * autopilot.phase_machine  [ported, changed]
      app:     frontend/src/app/services/autopilot-state.service.ts::checkPhaseTransition(
      harness: vtscore/eval/autopilot_flow.py::next_phase
      The phase ordering and every transition trigger of the simulated Autopilot user. ...
```

Reconcile the harness side, then re-pin:

```bash
python scripts/check-eval-app-sync.py --update
```

Digests ignore comments, docstrings, and formatting (including the magic trailing comma `ruff format` adds when it wraps a line), so only real logic changes trip the gate. Re-pinning without reading the harness defeats the whole thing — the digest is a prompt to check, not a checkbox.

### Adding and diverging

A new mirror is a new `Mirror(...)` entry in `MIRRORS`, plus `--update`. Give it a `note` that says what to re-check when the app moves, not just what the code is.

When the harness *intentionally* differs from the app at a mirror, record why in `divergence=`. That doesn't exempt it from the digest — you still re-pin — but the text prints whenever the mirror trips, so whoever reconciles it next knows which differences are deliberate. The ported indicators use this: they take their histories as arguments rather than reading the app's `_cached_steps`, which is plumbing, not a rule change.

Named experiment arms (`whole_image`, `max_patch_hac`, `max_patch_pca_hac`) are *supposed* to differ from the app — that's what makes them arms. This gate is about the default arm only.
