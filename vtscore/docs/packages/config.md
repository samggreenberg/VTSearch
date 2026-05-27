# `vtscore.config`

Runtime constants and the `CoreConfig` value object that every other
`vtscore` package reads when it needs a knob. Two responsibilities live
here: module-level constants resolved from environment variables at
import time (filesystem roots, thread caps, model IDs), and the
`CoreConfig` dataclass that bundles the per-call configuration that
would otherwise force the library to import `vtsearch.settings`. The
file is import-clean - it never reaches into the app - and is the only
place library code is allowed to read environment variables directly.

**Source:** `vtscore/config.py` (~263 lines).
**Related:** [`cli.md`](cli.md) for the CLI entry points that build a
`CoreConfig` before running, and the broader public-API sketch in
[`/home/user/VTSearch/docs/vtscore-api.md`](../../../docs/vtscore-api.md).

## Why `CoreConfig` exists

Library code under `vtscore/` is meant to run with or without the Flask
app. Before this seam existed, every loader, trainer, and embedder
reached into `vtsearch.settings` for tunables like `saved_datasets_dir`,
`calibrate_count`, or `safe_thresholds` - which made the library
impossible to vendor independently. `CoreConfig` is the seam:

- **Library-only consumers** construct a `CoreConfig(...)` directly and
  hand it to the API they call. No Flask, no `vtsearch.settings`, no
  shim. This is the supported public path.
- **The app** registers a builder via `register_core_config_builder()`
  at startup that snapshots the active `vtsearch.settings` into a fresh
  `CoreConfig` at every request boundary. Library code calls
  `CoreConfig.from_settings()` and gets a value object back without
  knowing the app exists.

The dataclass is `frozen=True`, so a config handed to a background
thread cannot be mutated underneath it.

## The `CoreConfig` dataclass

Defined at `vtscore/config.py:208-263`. Every field is required - the
class has no defaults so consumers can never accidentally inherit
stale state from a previous run.

| Field                             | Type           | Tier        | Meaning                                                                                          |
|-----------------------------------|----------------|-------------|--------------------------------------------------------------------------------------------------|
| `saved_datasets_dir`              | `Path`         | server      | Where dataset pickles are read from / written to.                                                |
| `detectors_dir`                   | `Path`         | server      | Where detector JSON files are read from / written to.                                            |
| `max_concurrent_dataset_downloads`| `int`          | server      | Cap on parallel dataset downloads (bandwidth/disk-bound stage).                                  |
| `max_concurrent_dataset_embeddings`| `int`         | server      | Cap on parallel dataset embedding (CPU/GPU-bound stage).                                         |
| `autorun_detectors`               | `tuple[str, ...]` | server   | Detector names to train + score automatically on every freshly-loaded dataset.                   |
| `safe_thresholds`                 | `bool`         | per-user    | When `True`, prefer cross-calibrated thresholds with explicit margin over GMM-fit thresholds.    |
| `calibrate_count`                 | `int`          | per-user    | Number of fold-training passes used to calibrate the operating threshold. Min 1.                 |
| `calibration_fraction`            | `float`        | per-user    | Fraction of labels held out per calibration fold. Typical 0.5.                                   |
| `enrich_descriptions`             | `bool`         | per-user    | When `True`, attach `custom_metadata` from origins to result rows on export.                     |
| `autopilot_goal_diversity`        | `int`          | per-user    | Diversity-tree depth target used by autopilot pacing.                                            |
| `inclusion`                       | `int`          | per-user    | Inclusion filter: `0` = all medias, `1` = only labeled, `2` = only unlabeled.                    |
| `data_dir`                        | `Path`         | bootstrap   | Filesystem root for caches, embeddings, and model downloads. Mirrors `DATA_DIR` at construction. |

"Server" and "per-user" refer to where the app stores the corresponding
setting - both tiers flow into the same `CoreConfig` so library code
never has to know the difference. Library-only callers just pass
whatever values they want.

### Constructing one directly

```python
from pathlib import Path
from vtscore.config import CoreConfig, DATA_DIR

config = CoreConfig(
    saved_datasets_dir=DATA_DIR / "datasets",
    detectors_dir=DATA_DIR / "detectors",
    max_concurrent_dataset_downloads=2,
    max_concurrent_dataset_embeddings=1,
    autorun_detectors=(),
    safe_thresholds=True,
    calibrate_count=1,
    calibration_fraction=0.5,
    enrich_descriptions=False,
    autopilot_goal_diversity=8,
    inclusion=0,
    data_dir=DATA_DIR,
)
```

### `from_settings()` and the app-builder hook

```python
@classmethod
def from_settings(cls, settings_path: str | Path | None = None) -> CoreConfig:
    """Snapshot the current user's vtsearch.settings into a CoreConfig."""
```

This classmethod has no library-side implementation. It calls a builder
that the app installs at startup via
`register_core_config_builder(fn)` (defined at `vtscore/config.py:196`).
The builder is a function `(settings_path: str | Path | None) -> CoreConfig`.

If `from_settings()` is called and no builder is registered, it raises
`RuntimeError` with a message pointing the caller at the library-only
path (construct `CoreConfig(...)` directly). Library-only consumers
without the app shim should never reach this method.

```python
from vtscore.config import register_core_config_builder, CoreConfig

def _build(settings_path):
    # Example: read your own JSON or YAML config here, return a CoreConfig.
    ...
    return CoreConfig(...)

register_core_config_builder(_build)

# Now this works:
config = CoreConfig.from_settings()
```

When `settings_path` is supplied, the builder is expected to redirect
its server-tier settings file lookup to that path first. The CLI uses
this so a `--settings my-run.json` invocation doesn't have to touch
the app's default settings file.

## Module-level filesystem constants

All paths are resolved once at import time. They are `Path` instances,
not strings, and they are anchored to the repo root - *not* the current
working directory - so launching the app from `systemd`, `cron`, or a
fresh dev shell will not silently create an empty `data/` next to the
service launcher.

| Constant            | Default                       | Override env var          | Meaning                                                              |
|---------------------|-------------------------------|---------------------------|----------------------------------------------------------------------|
| `DATA_DIR`          | `<repo>/data`                 | `VTSEARCH_DATA_DIR`       | Canonical data root. All runtime artefacts live underneath this.     |
| `EMBEDDINGS_DIR`    | `DATA_DIR / "embeddings"`     | -                         | Embedding cache root. Derived from `DATA_DIR`.                       |
| `MODELS_CACHE_DIR`  | `DATA_DIR / "models"`         | `VTSEARCH_MODELS_DIR`     | HuggingFace + torch hub cache root.                                  |

**Invariant.** `DATA_DIR` is the only path library code is allowed to
derive runtime locations from. No file under `vtscore/` should hardcode
`./data/...` - always start from `DATA_DIR` (or a `CoreConfig.data_dir`
when one is in scope). Tests rely on this: they override
`VTSEARCH_DATA_DIR` per-test to redirect every cache to a `tmp_path`.

## Runtime tunables

| Constant                 | Default | Override env var              | Meaning                                                                                                          |
|--------------------------|---------|-------------------------------|------------------------------------------------------------------------------------------------------------------|
| `TORCH_THREADS`          | `1`     | `VTSEARCH_TORCH_THREADS`      | Native thread count for OpenMP / MKL / `torch.set_num_threads`. Default 1 keeps RSS low in constrained envs.     |
| `DEVICE`                 | `"auto"`| `VTSEARCH_DEVICE`             | Preferred compute device. `"auto"` resolves at call time; explicit `"cuda"`, `"cuda:0"`, `"cpu"`, `"mps"` pass through. |
| `MAX_UPLOAD_MB`          | `0`     | `VTSEARCH_MAX_UPLOAD_MB`      | HTTP body cap in megabytes. `0` = unlimited (Flask's out-of-the-box behaviour).                                  |
| `TRAIN_EPOCHS`           | `200`   | `VTSEARCH_TRAIN_EPOCHS`       | Upper bound on MLP training epochs. `vtscore.training.mlp.train_model` may early-stop sooner.                    |
| `TRAIN_PATIENCE`         | `10`    | `VTSEARCH_TRAIN_PATIENCE`     | Epochs the training loss must fail to improve before early-stop fires. `0` disables early-stop.                  |
| `DEFAULT_CALIBRATE_COUNT`| `1`     | `VTSEARCH_CALIBRATE_COUNT`    | First-run default for `CoreConfig.calibrate_count`. Min 1.                                                       |
| `MLP_HIDDEN_MIN`         | `4`     | -                             | Auto-sizing floor for MLP hidden width.                                                                          |
| `MLP_HIDDEN_MAX`         | `32`    | -                             | Auto-sizing ceiling for MLP hidden width.                                                                        |
| `MLP_DROPOUT`            | `0.5`   | -                             | Dropout rate for trained MLPs.                                                                                   |

### `resolve_device()`

Defined at `vtscore/config.py:45-62`. Returns the concrete device
string this host will actually use:

```python
from vtscore.config import resolve_device

torch_device = resolve_device()
# "cuda" if CUDA is visible
# "mps" on Apple Silicon
# "cpu" otherwise
```

Imports `torch` lazily so simply importing `vtscore.config` does not
pull torch into the process. Returns `"cpu"` if torch isn't installed
at all. When `DEVICE` is anything other than `"auto"`, the env var's
value is returned unchanged - this is how you pin to `"cuda:1"` or
`"mps"` for a specific run.

## Server-path roots

```python
SERVER_ROOTS: tuple[Path, ...]
```

Allowed roots for server-side filesystem importers, exporters, and the
file browser. Parsed from `VTSEARCH_SERVER_ROOTS` (PATH-separated:
`:` on Unix, `;` on Windows). When unset, defaults to a single-element
tuple `(Path.cwd().resolve(),)`, which reproduces the historical
"anything under the launch directory" behaviour.

```bash
# Restrict to two specific roots:
export VTSEARCH_SERVER_ROOTS=/srv/data:/srv/imports
```

The first entry is what `/api/browse` shows when no `path` parameter
is provided. Multi-user mode is unaffected - each user is still
confined to their own `<get_user_data_dir(user)>/...` subtree
regardless of this setting.

## Embedder model identifiers

Every public embedder ID is a plain string constant. They are
*identifiers only* - none of these constants load anything at import
time. The actual download + load is lazy, driven by
`vtscore.embedding.loader` getters (`get_clap_model`, `get_xclip_model`,
`get_e5_model`, etc.).

| Constant                       | Value                                                                                  | Notes                                                                                  |
|--------------------------------|----------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| `CLAP_MODEL_ID`                | `"laion/clap-htsat-unfused"`                                                           | Default audio embedder.                                                                |
| `CLAP_SAMPLE_RATE`             | `48000`                                                                                | Required input sample rate for the CLAP family.                                        |
| `CLAP_MUSIC_MODEL_ID`          | `"laion/larger_clap_music_and_speech"`                                                 | CLAP variant for music + speech.                                                       |
| `CLAP_GENERAL_MODEL_ID`        | `"laion/larger_clap_general"`                                                          | Larger general-purpose CLAP.                                                           |
| `AST_MODEL_ID`                 | `"MIT/ast-finetuned-audioset-10-10-0.4593"`                                            | Audio Spectrogram Transformer. 16 kHz mono via `AST_SAMPLE_RATE`.                      |
| `AST_SAMPLE_RATE`              | `16000`                                                                                |                                                                                        |
| `WHISPER_MODEL_ID`             | `"openai/whisper-base"`                                                                | Used by the audio→text converter (ASR).                                                |
| `WHISPER_SAMPLE_RATE`          | `16000`                                                                                |                                                                                        |
| `XCLIP_MODEL_ID`               | `"microsoft/xclip-base-patch32"`                                                       | Default video embedder.                                                                |
| `VIDEOMAE_MODEL_ID`            | `"OpenGVLab/VideoMAEv2-Base"`                                                          | Vision-only video encoder (no paired text tower; `supports_text=False`).               |
| `LANGUAGEBIND_VIDEO_MODEL_ID`  | `"LanguageBind/LanguageBind_Video_V1.5_FT"`                                            | Alternative video embedder with text-aligned latent.                                   |
| `SIGLIP_MODEL_ID`              | `"google/siglip-base-patch16-224"`                                                     | Default image embedder.                                                                |
| `SIGLIP2_MODEL_ID`             | `"google/siglip2-base-patch16-224"`                                                    | SigLIP v2.                                                                             |
| `CLIP_MODEL_ID`                | `"openai/clip-vit-base-patch32"`                                                       | OpenAI CLIP.                                                                           |
| `DINOV2_MODEL_ID`              | `"facebook/dinov2-base"`                                                               | Self-supervised image encoder.                                                         |
| `DINOV3_MODEL_ID`              | `"facebook/dinov3-vitb16-pretrain-lvd1689m"`                                           | DINO v3.                                                                               |
| `EUPE_MODEL_ID`                | `"https://huggingface.co/facebook/EUPE-ViT-B/resolve/main/EUPE-ViT-B.pt"`              | Direct HF URL to EUPE ViT-B weights. Loaded via `torch.hub.load`. FAIR Non-commercial. |
| `E5_MODEL_ID`                  | `"intfloat/e5-base-v2"`                                                                | Default text embedder.                                                                 |
| `BGE_MODEL_ID`                 | `"BAAI/bge-base-en-v1.5"`                                                              | Alternative text embedder.                                                             |

The EUPE model is not the same as `facebook/PE-Core-B16-224`; the
constant points at a single `.pt` weight file (an `AutoModel.from_pretrained`
path will not work, because the HF repo has no `config.json`).

## Environment variables - summary

Every env var consulted by `vtscore.config`, in one place:

| Variable                    | Effect                                                                                          |
|-----------------------------|-------------------------------------------------------------------------------------------------|
| `VTSEARCH_DATA_DIR`         | Override `DATA_DIR`. Use to relocate state outside the repo.                                    |
| `VTSEARCH_MODELS_DIR`       | Override `MODELS_CACHE_DIR`. Independent of `VTSEARCH_DATA_DIR`.                                |
| `VTSEARCH_TORCH_THREADS`    | Set `TORCH_THREADS` (and therefore OMP / MKL caps). Floor of 1.                                 |
| `VTSEARCH_DEVICE`           | Set `DEVICE`. `"auto"` is the default; pin to `"cuda"`, `"cuda:0"`, `"cpu"`, or `"mps"`.        |
| `VTSEARCH_SERVER_ROOTS`     | PATH-separated list of allowed roots for server-path importers/exporters.                       |
| `VTSEARCH_MAX_UPLOAD_MB`    | Set `MAX_UPLOAD_MB`. `0` = unlimited.                                                           |
| `VTSEARCH_TRAIN_EPOCHS`     | Set `TRAIN_EPOCHS` (MLP training upper bound).                                                  |
| `VTSEARCH_TRAIN_PATIENCE`   | Set `TRAIN_PATIENCE` (early-stop patience). `0` disables.                                       |
| `VTSEARCH_CALIBRATE_COUNT`  | Set `DEFAULT_CALIBRATE_COUNT` (first-run default; later writes go to per-user settings).        |

All of these are read at import time. Setting them after `vtscore.config`
has loaded has no effect; do the export before `python -m vtscore` /
`python app.py` runs.

## Typical flow

The end-to-end picture for a library + app deployment:

1. Process starts; `vtscore.config` imports and reads env vars.
2. The Flask app's `vtsearch.shim` calls
   `register_core_config_builder(builder_fn)` where `builder_fn`
   snapshots `vtsearch.settings` into a `CoreConfig` for the current
   user.
3. A request arrives. The before-request handler resolves the active
   user, then library code calls `CoreConfig.from_settings()`, which
   delegates to `builder_fn` and returns a frozen `CoreConfig` valid
   for this request only.
4. The handler hands that `CoreConfig` to whatever library function it
   calls - datasets loader, detector trainer, exporter. Background
   threads spawned from the request keep using the same frozen object.

For library-only consumers, steps 2–3 collapse: the consumer builds
`CoreConfig(...)` directly and passes it down. `from_settings()` is
never called. The library is identical in both cases - it does not
care which path produced the config.

## Invariants

- `vtscore.config` does **not** import `vtsearch.settings`. Ever.
  Adding such an import is the bug `register_core_config_builder`
  exists to prevent.
- No code under `vtscore/` should hardcode `./data` or `data/` -
  always derive from `DATA_DIR` or `CoreConfig.data_dir`.
- Module-level constants are resolved at import time. They are
  effectively `Final`; don't reassign them.
- `CoreConfig` is `frozen=True`. Background threads can safely hold
  a reference; in-flight reconfiguration requires building a new
  `CoreConfig` and routing it explicitly.
- No persisted embeddings, no persisted MLP weights. This config
  module does not introduce a place to cache them.
