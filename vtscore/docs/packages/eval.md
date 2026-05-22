# `vtscore.eval` — Offline evaluation

Reproducible evaluation of text-sort and learned-sort quality on demo
datasets, plus a voting-iteration simulator that tracks classification
cost as a function of how many labels have been cast. Everything in
this package is computation only — the numbers come out as dataclasses
and DataFrames. Rendering them to PNGs lives in `vtscore.eval.visualize`,
which is the one module here that imports matplotlib.

The package wraps the demo datasets registered under
`vtscore.datasets` with text descriptions (the queries a user would
type in the Text Sort box) so a single CLI invocation can sweep across
audio, image, video, and paragraph datasets and report comparable
metrics. The full user-facing guide is at
[`docs/EVAL.md`](../../../docs/EVAL.md); this doc covers the library
surface a programmatic consumer calls into.

## Contents

| Module                                  | Concern                                                  |
|-----------------------------------------|----------------------------------------------------------|
| `vtscore/eval/config.py`                | `EvalQuery` dataclass and `EVAL_DATASETS` registry       |
| `vtscore/eval/metrics.py`               | `QueryMetrics`, `LearnedSortMetrics`, `DatasetResult`, metric functions |
| `vtscore/eval/runner.py`                | `eval_text_sort`, `eval_learned_sort`, `run_eval`, `format_results_json` |
| `vtscore/eval/voting_iterations.py`     | Per-step cost simulator and multi-dataset sweep          |
| `vtscore/eval/label_curve.py`           | MLP-vs-SVM label-curve sweep (`run_label_curve_eval`)    |
| `vtscore/eval/label_curve_main.py`      | CLI for the label-curve sweep                            |
| `vtscore/eval/visualize.py`             | `plot_eval_results`, `plot_voting_iterations` (matplotlib) |
| `vtscore/eval/__main__.py`              | CLI for `python -m vtscore.eval`                         |

The package `__init__.py` re-exports the main entry points:

```python
from vtscore.eval import (
    EVAL_DATASETS, EvalQuery,
    compute_metrics, run_eval,
    simulate_voting_iterations, run_voting_iterations_eval,
    run_voting_iterations_eval_from_pickles,
    plot_eval_results, plot_voting_iterations,
)
```

---

## Data classes

### `EvalQuery`

`vtscore/eval/config.py:18`. One natural-language query targeting one
ground-truth category.

```python
@dataclass
class EvalQuery:
    text: str               # what a user would type, e.g. "a dog barking"
    target_category: str    # the ground-truth category, e.g. "dog"
```

`EVAL_DATASETS` is a `dict[str, dict]` keyed by demo dataset id
(`esc50_s`, `caltech101_m`, `20newsgroups_l`, `ucf101_s`, ...). Each
value is `{"demo_dataset": "...", "queries": list[EvalQuery]}`. The
registry covers all 50 ESC-50 categories, 25 Caltech-101 / Caltech-256
categories, 15 of the 20-Newsgroups categories, and 10 UCF-101
categories — see `vtscore/eval/config.py` for the full lists.

### `QueryMetrics`

`vtscore/eval/metrics.py:11`. One text-sort query's results.

```python
@dataclass
class QueryMetrics:
    query_text: str
    target_category: str
    average_precision: float
    precision_at_k: dict[int, float]   # default factory: {}
    recall_at_k:    dict[int, float]   # default factory: {}
    num_relevant: int = 0
    num_total: int = 0
    elapsed_seconds: float = 0.0
```

### `LearnedSortMetrics`

`vtscore/eval/metrics.py:25`. One learned-sort fold's results.

```python
@dataclass
class LearnedSortMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    num_train: int
    num_test: int
    target_category: str = ""
    elapsed_seconds: float = 0.0
```

### `DatasetResult`

`vtscore/eval/metrics.py:39`. Aggregated results for one eval dataset.

```python
@dataclass
class DatasetResult:
    dataset_id: str
    media_type: str
    text_sort: list[QueryMetrics]            # default []
    learned_sort: list[LearnedSortMetrics]   # default []

    @property
    def mean_average_precision(self) -> float: ...   # mAP across text_sort
    @property
    def mean_learned_f1(self) -> float: ...          # mean F1 across folds

    def to_dict(self) -> dict[str, Any]: ...
```

`to_dict()` produces the JSON shape `format_results_json` emits.

---

## Metrics

`vtscore/eval/metrics.py` provides four pure functions on lists of ids
and labels. None of them depend on a dataset context, embedder, or
sort — they take what the runner produces and return numbers.

| Function                                                          | Behaviour                                                 |
|-------------------------------------------------------------------|-----------------------------------------------------------|
| `compute_average_precision(ranked_ids, relevant_ids)` (line 107)  | AP = Σ(precision@k) / num_relevant over relevant positions; 0 when `relevant_ids` is empty |
| `compute_precision_recall_at_k(ranked_ids, relevant_ids, k_values=None)` (line 132) | Tuple of `(precision_at_k, recall_at_k)` dicts keyed by k. Defaults to `[5, 10, 20]` |
| `compute_metrics(ranked_ids, relevant_ids, query_text, target_category, k_values=None)` (line 163) | Bundle: returns a populated `QueryMetrics` |
| `compute_binary_classification_metrics(predictions, labels)` (line 196) | Returns `(accuracy, precision, recall, f1)` from 0/1 lists |

```python
from vtscore.eval.metrics import compute_metrics

ranked = [3, 1, 2, 5, 4]            # cids sorted descending by score
relevant = {1, 2, 4}                # cids in the target category

qm = compute_metrics(ranked, relevant,
                     query_text="a dog barking",
                     target_category="dog",
                     k_values=[3, 5])
print(qm.average_precision, qm.precision_at_k, qm.recall_at_k)
```

---

## Runners

### `eval_text_sort(medias, queries, media_type, ...)`

`vtscore/eval/runner.py:65`. For each query: embed the query text via
`vtscore.embedding.helpers.embed_text_query`, score every media by
cosine similarity, sort descending, and compute metrics treating
medias whose `"category"` matches `query.target_category` as relevant.
Returns a list of `QueryMetrics`. Pass `enrich=True` to use wrapper-
averaged text embeddings; pass `start_time` (a `time.monotonic()`
baseline) to populate `elapsed_seconds` on each result.

### `eval_learned_sort(medias, queries, train_fraction=0.5, seed=42, ...)`

`vtscore/eval/runner.py:106`. For each query/category: split target-
category vs. other medias, take `train_fraction` of each as training
data, build the synthetic `good_votes` / `bad_votes` dicts, call
`train_and_score` from [`vtscore.detectors`](detectors.md), and
measure accuracy / precision / recall / F1 on the held-out test set
using the cross-calibrated threshold. Returns a list of
`LearnedSortMetrics`. Honours `safe_thresholds`, `calibrate_count`,
and `calibration_fraction` exactly the way the production training
path does.

```python
from vtscore.eval.runner import eval_text_sort, eval_learned_sort
from vtscore.eval.config import EvalQuery
from vtscore.datasets.loader import load_demo_dataset

medias = {}
load_demo_dataset("esc50_s", medias)

queries = [EvalQuery("a dog barking", "dog"),
           EvalQuery("rain falling", "rain")]

text_results = eval_text_sort(medias, queries, media_type="audio")
learned_results = eval_learned_sort(medias, queries,
                                    train_fraction=0.5, seed=42)
```

### `run_eval(dataset_ids=None, mode="both", ...)`

`vtscore/eval/runner.py:211`. The full pipeline, written for the CLI
but usable from Python directly. Iterates over `dataset_ids` (or every
key of `EVAL_DATASETS` when `None`), loads each demo dataset into a
fresh `medias` dict via `load_demo_dataset`, runs `eval_text_sort`
when `mode in ("text", "both")` and `eval_learned_sort` when
`mode in ("learned", "both")`, prints progress to stdout, and returns
a list of `DatasetResult`.

Args:

| Arg                       | Meaning                                                       |
|---------------------------|---------------------------------------------------------------|
| `dataset_ids`             | List of eval dataset ids, or `None` for all                   |
| `mode`                    | `"text"`, `"learned"`, or `"both"`                            |
| `k_values`                | k values for P@k / R@k (default `[5, 10, 20]`)                |
| `train_fraction`          | Train/test split for learned-sort (default 0.5)               |
| `seed`                    | Random seed (default 42)                                      |
| `enrich`                  | Use wrapper-averaged text embeddings (default False)          |
| `safe_thresholds`         | Blend cross-cal threshold with GMM (default False)            |
| `calibrate_count`         | Cross-cal folds (default 2)                                   |
| `calibration_fraction`    | Cross-cal calibrate split (default 0.5)                       |

### `format_results_json(results)`

`vtscore/eval/runner.py:336`. Serialise a list of `DatasetResult` to a
JSON string by calling `r.to_dict()` on each and round-tripping
through `json.dumps(indent=2)`.

---

## Voting iterations

`vtscore/eval/voting_iterations.py` simulates an interactive labelling
session — votes are cast one at a time in a shuffled order, and at
each step (once both polarities have at least one vote) a fresh MLP is
trained, a threshold is computed, the held-out test set is scored,
and the inclusion-weighted cost is recorded. This is how the team
answers "how does cost drop as the user labels?" without spinning up
a UI session.

### `simulate_voting_iterations(clips_dict, target_category, seed, ...)`

`vtscore/eval/voting_iterations.py:122`. Run one `(dataset, category,
seed)` simulation. Splits `clips_dict` into `D_sim` (used to draw
votes) and `D_test` (held out for cost evaluation) by `sim_fraction`,
shuffles the vote order, and iterates:

1. Apply the next vote.
2. When both polarities have at least one vote, train an MLP
   (`train_model`) and pick a threshold
   (`calculate_cross_calibration_threshold`, with optional
   `calculate_safe_threshold` blend).
3. Score `D_test`, compute FPR / FNR, weight by inclusion, record the
   cost.

Returns a list of row dicts:

```python
{
    "seed": 0, "dataset": "esc50_s", "category": "dog",
    "t": 7, "cost": 0.124, "fpr": 0.05, "fnr": 0.21,
    "elapsed_seconds": 12.4,
}
```

Honours the same threshold knobs as the runner. When
`safe_thresholds=True`, the GMM blend uses scores over the simulation
set only (not the test set) so test scores don't leak into
calibration.

### `run_voting_iterations_eval(dataset_clips, seeds, categories=None, ...)`

`vtscore/eval/voting_iterations.py:250`. Sweep
`simulate_voting_iterations` over `(seed × dataset × category)` and
return a `pandas.DataFrame` with columns `seed, dataset, category, t,
cost, fpr, fnr, elapsed_seconds`. When `categories` is `None` or a
dataset is missing from the dict, every unique category in that
dataset is used.

```python
from vtscore.eval.voting_iterations import run_voting_iterations_eval

df = run_voting_iterations_eval(
    dataset_clips={"esc50_s": medias},
    seeds=[0, 1, 2, 3, 4],
    inclusion=0,
    sim_fraction=0.5,
)
```

### `run_voting_iterations_eval_from_pickles(dataset_paths, seeds, ...)`

`vtscore/eval/voting_iterations.py:315`. Convenience wrapper that
loads each dataset from a pickle path (via
`vtscore.datasets.loader.load_dataset_from_pickle`) and then calls
`run_voting_iterations_eval`. The returned DataFrame has the same
columns.

---

## Label curve

`vtscore/eval/label_curve.py` is a separate sweep that compares the
MLP and SVM trainers head-to-head as a function of training-set size.
The sweep iterates over `dataset × target_category × trainer ×
label_count × seed` and writes one row per cell to a tidy DataFrame.

The headline metrics are rank-based on purpose: VTSearch never trusts
the raw score as a probability — it derives the operating threshold
via cross-calibration. So `AUROC`, `AP`, and the production-path
`f1_at_xcal` (which uses
`find_optimal_threshold` on a held-out calibration slice) are what
matter; Brier and F1@0.5 are kept as diagnostics. See
`TRAINERS` for the plug-in registry of trainer functions.

`label_curve_main.py` is the CLI:

```bash
python -m vtscore.eval.label_curve \
    --datasets esc50_s flowers102_s \
    --trainers mlp svm_linear svm_rbf \
    --label-counts 5 10 20 50 100 200 \
    --seeds 0 1 2 3 4 \
    --output label_curve.csv
```

---

## CLI

`python -m vtscore.eval` is the main entry point.
`vtscore/eval/__main__.py` calls `initialize_models()` to set up the
torch runtime, parses argparse args, runs `run_eval`, and optionally
writes a JSON dump and matplotlib plots.

```bash
# Default: text-sort + learned-sort on every registered eval dataset
python -m vtscore.eval

# Subset
python -m vtscore.eval --datasets esc50_s caltech101_s --mode both

# Custom split + JSON output
python -m vtscore.eval --mode learned --train-fraction 0.6 --output results.json

# Generate visualisations
python -m vtscore.eval --plot-dir eval_plots

# List available eval datasets
python -m vtscore.eval --list
```

Notable flags:

| Flag                     | Meaning                                                      |
|--------------------------|--------------------------------------------------------------|
| `--datasets ID [ID ...]` | Restrict to these eval dataset ids                           |
| `--mode {text,learned,both}` | Which evaluation to run                                  |
| `--k K [K ...]`          | k values for P@k / R@k                                       |
| `--train-fraction F`     | Learned-sort split ratio                                     |
| `--seed N`               | Random seed                                                  |
| `--enrich-descriptions`  | Wrapper-averaged text embeddings                             |
| `--safe-thresholds`      | Blend cross-cal threshold with GMM                           |
| `--calibrate-count K`    | Cross-cal folds                                              |
| `--calibration-fraction F` | Cross-cal calibration split                                |
| `--output FILE`          | Write JSON results to `FILE`                                 |
| `--plot-dir DIR`         | Generate visualisation PNGs in `DIR`                         |
| `--no-plot`              | Disable plots even when `--plot-dir` is set                  |
| `--list`                 | Print the eval-dataset registry and exit                     |

The full user guide — including how the demo datasets are sourced,
what each metric means in practice, and how to interpret the output —
is at [`docs/EVAL.md`](../../../docs/EVAL.md).

---

## Visualisation

`vtscore/eval/visualize.py` is the one module in `vtscore.eval` that
imports matplotlib. It is the presentation layer; library callers that
want raw numbers should consume the `DatasetResult` / DataFrame
return values directly and skip this module.

| Function                                                   | Output                                                          |
|------------------------------------------------------------|-----------------------------------------------------------------|
| `plot_eval_results(results, output_dir="eval_output")`     | PNGs for mAP-by-dataset, AP-by-query, P@k curves, R@k curves, learned-sort F1, learned-sort metrics breakdown |
| `plot_voting_iterations(df, output_dir="voting_output")`   | Cost-over-iterations and FPR/FNR-over-iterations line charts, one line per (dataset, category), with shaded ±1σ band over seeds |

Both functions create `output_dir` if missing and return a list of the
generated `Path`s. They apply a clean default matplotlib style
(`white` facecolor, grid on, top/right spines off) before plotting.
matplotlib is not in the library's core dependencies — installing
`matplotlib` separately is required to use these helpers.

---

## Invariants worth restating

- **No persisted weights.** `eval_learned_sort` and
  `simulate_voting_iterations` train MLPs in memory and discard them
  after scoring; no detector files are written.
- **Deterministic.** Every function that produces randomness takes a
  `seed` argument; `np.random.RandomState(seed)` controls splits and
  vote order, `train_model` uses its own thread-safe RNG (see
  [`training.md`](training.md)).
- **Pure computation.** The runner and voting-iterations modules take
  pre-loaded medias dicts as input. The CLI is the only entry point
  that calls `load_demo_dataset` — programmatic consumers are
  expected to load their own data.
- **No Flask, no settings.** Every threshold knob (`inclusion`,
  `safe_thresholds`, `calibrate_count`, `calibration_fraction`,
  `sim_fraction`) is a function argument, not a global lookup.
