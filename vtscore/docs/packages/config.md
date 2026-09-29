# `vtscore.config`

Runtime constants and the `CoreConfig` value object that every other
`vtscore` package reads when it needs a knob. Two responsibilities live
here: module-level constants resolved from environment variables at
import time (filesystem roots, thread caps, model IDs), and the
`CoreConfig` dataclass that bundles the per-call configuration that
would otherwise force the library to import `vtsearch.settings`. The
package is import-clean - it never reaches into the app - and is the only
place library code is allowed to read environment variables directly.

**Source:** `vtscore/config/` - a package of six submodules, listed under
[Contents](#contents) below. Everything public is re-exported
from `vtscore/config/__init__.py`, so `vtscore.config.X` and
`from vtscore.config import X` are the import paths for all of it; the
submodules are an internal organisation, not a second public surface.

## Contents

The submodules are layered: each reads only from ones above it.

| Submodule | Holds | Reads |
|---|---|---|
| `paths` | `DATA_DIR`, `EMBEDDINGS_DIR`, `MODELS_CACHE_DIR` | - |
| `runtime` | thread and decode-worker sizing, the upload/decode caps, the training/MLP/SVM knobs, the UMAP projection defaults | - |
| `models` | every embedder's Hugging Face id and per-checkpoint constants | - |
| `device` | `DEVICE` / `resolve_device()` (with the CUDA smoke-test), `EMBED_PRECISION` / `embed_precision()` and the dtype resolvers | - |
| `processor_backend` | which `transformers` image-processor implementation runs, and on what device | `device` |
| `core_config` | `CoreConfig` and `register_core_config_builder()` | `runtime` |

Two consequences matter to test authors, and only to test authors:

- **Stubs go where the reader is.** A caller *outside* the package resolves
  `vtscore.config.X`, so patching the package reaches it. A function *inside*
  a submodule resolves its own module global, of which the package attribute
  is only a copy - so stubbing `resolve_device` for `embed_precision()`, or
  `allocated_cpus` for `resolve_decode_workers()`, means patching
  `vtscore.config.device` / `vtscore.config.runtime`. Private names
  (`_cuda_can_run`, `_core_config_builder`, ...) are deliberately not
  re-exported, so an attempt to stub one on the package fails loudly instead
  of being ignored.
- **Re-reading the environment needs `config._reload_all()`.**
  `importlib.reload(vtscore.config)` only re-runs the re-exports; the
  submodules are already in `sys.modules` and do not re-execute, so the env
  vars are not re-read. `_reload_all()` reloads them in dependency order and
  then the package.

## Two ways to get a `CoreConfig`

- **Library-only consumers** construct a `CoreConfig(...)` directly and
  hand it to the API they call. This is the supported public path.
- **A host app** registers a builder via `register_core_config_builder()`
  at startup; library code then calls `CoreConfig.from_settings()` and
  gets a value object back without knowing the app exists.

The rationale for the seam is in
[`architecture.md`](../architecture.md#the-coreconfig-bridge). The
dataclass is `frozen=True`, so a config handed to a background thread
cannot be mutated underneath it.

## The `CoreConfig` dataclass

Defined in `vtscore/config/core_config.py`. The twelve fields below are **required** -
they have no defaults, so a consumer can never accidentally inherit stale
state from a previous run. (The handful of newer optional fields listed
after the table do have defaults, precisely so a library-only
`CoreConfig(...)` construction written before they existed keeps working.)

| Field                             | Type           | Tier        | Meaning                                                                                          |
|-----------------------------------|----------------|-------------|--------------------------------------------------------------------------------------------------|
| `saved_datasets_dir`              | `Path`         | server      | Where saved-dataset containers are read from / written to.                                       |
| `detectors_dir`                   | `Path`         | server      | Where detector JSON files are read from / written to.                                            |
| `max_concurrent_dataset_downloads`| `int`          | server      | Cap on parallel dataset downloads (bandwidth/disk-bound stage).                                  |
| `max_concurrent_dataset_embeddings`| `int`         | server      | Cap on parallel dataset embedding (CPU/GPU-bound stage).                                         |
| `autofind_detectors`               | `tuple[str, ...]` | server   | Detector names to train + score automatically on every freshly-loaded dataset.                   |
| `dataset_max_age_days`            | `int \| None`  | server      | Age-off horizon stamped into saved datasets. `None` = never expire.                              |
| `calibrate_count`                 | `int`          | per-user    | Number of fold-training passes used to calibrate the operating threshold. Min 1.                 |
| `calibration_fraction`            | `float \| None` | per-user    | Explicit fraction of labels held out per calibration fold. `None` = per-embedder default (0.3 single-vector / 0.5 patch). |
| `enrich_descriptions`             | `bool`         | per-user    | When `True`, attach `custom_metadata` from origins to result rows on export.                     |
| `autopilot_goal_diversity`        | `int`          | per-user    | Diversity target used by autopilot pacing.                                                       |
| `inclusion`                       | `int`          | per-user    | The Inclusion knob, `[-10, +10]`. A pure threshold shift (it never enters training); `0` is the default operating point. See [`training.md`](training.md#decision-thresholds). |
| `data_dir`                        | `Path`         | bootstrap   | Filesystem root for caches, embeddings, and model downloads. Mirrors `DATA_DIR` at construction. |

Optional (defaulted) fields: `autofind_exporter` (`str`, `""`),
`autofind_exporter_field_values` (`dict[str, dict[str, str]]`, `{}` -
keyed by exporter name), `projection_n_neighbors` (`int`,
`PROJECTION_N_NEIGHBORS`), `projection_min_dist` (`float`,
`PROJECTION_MIN_DIST`), `signpost_captioner` (`dict[str, bool]`, `{}`),
`signpost_vocab` (`dict[str, list[str]]`, `{}`), and `hide_ingest_eta`
(`bool`, `False` - when `True`, ingest progress bars publish no ETA; see
[concurrency.md](concurrency.md#progresstracker)).

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
    autofind_detectors=(),
    dataset_max_age_days=None,
    calibrate_count=1,
    calibration_fraction=0.5,
    enrich_descriptions=False,
    autopilot_goal_diversity=8,
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
`register_core_config_builder(fn)`. The builder is a function
`(settings_path: str | Path | None) -> CoreConfig` - it is always called
with that one positional argument, so a zero-argument builder raises
`TypeError`.

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
| `DEFAULT_DECODE_WORKER_CAP` | `8`  | `VTSEARCH_DECODE_WORKERS`     | Ceiling on the image-decode prefetch pool (see `resolve_decode_workers()`). The pool itself is sized from the allocation, not this constant. |
| `DEVICE`                 | `"auto"`| `VTSEARCH_DEVICE`             | Preferred compute device. `"auto"` resolves at call time; explicit `"cuda"`, `"cuda:0"`, `"cpu"`, `"mps"` are honoured, but a CUDA device the installed torch wheel can't actually run on falls back to `"cpu"`. |
| `MAX_UPLOAD_MB`          | `2048`  | `VTSEARCH_MAX_UPLOAD_MB`      | HTTP body cap in megabytes (default 2 GiB), for a host app to enforce. `0` = unlimited. |
| `MAX_DECODE_PIXELS`      | `64_000_000` | `VTSEARCH_MAX_DECODE_PIXELS` | Bitmap budget for one image decode; larger sources are downsampled (aspect kept). Crop/clip paths bypass it. `0` disables. |
| `MAX_STRUCTURAL_DETECT_PIXELS` | `2_000_000` | `VTSEARCH_MAX_STRUCTURAL_DETECT_PIXELS` | Resolution budget for structural (SIFT) keypoint detection. `0` detects at native size. |
| `TRAIN_EPOCHS`           | `200`   | `VTSEARCH_TRAIN_EPOCHS`       | Upper bound on BCE-head training epochs. `vtscore.training.mlp.train_model` may early-stop sooner. The production SVM head is one liblinear solve and ignores this.  |
| `TRAIN_PATIENCE`         | `10`    | `VTSEARCH_TRAIN_PATIENCE`     | Epochs the training loss must fail to improve before early-stop fires. `0` disables early-stop. BCE heads only.   |
| `DEFAULT_CALIBRATE_COUNT`| `2`     | `VTSEARCH_CALIBRATE_COUNT`    | First-run default for `CoreConfig.calibrate_count`. Min 1.                                                       |
| `MLP_HIDDEN_MIN`         | `8`     | -                             | Auto-sizing floor for MLP hidden width. Legacy MLP head only - the production linear SVM head has none.              |
| `MLP_HIDDEN_MAX`         | `32`    | -                             | Auto-sizing ceiling for MLP hidden width. Legacy MLP head only.                                                  |
| `MLP_DROPOUT`            | `0.5`   | -                             | Dropout rate for trained MLPs. Ignored by the production linear SVM head.                                            |
| `MLP_LABEL_SMOOTHING`    | `0.05`  | -                             | Label-smoothing epsilon for MLP targets (keeps calibration scores distinct for the conformal threshold).          |
| `SVM_HEAD_C`             | `1.0`   | `VTSEARCH_SVM_HEAD_C`         | Inverse regularisation strength of the production linear SVM head (`fit_linear_svm_head`). Lower regularises harder on few votes. |
| `PROJECTION_N_NEIGHBORS` / `PROJECTION_MIN_DIST` | `15` / `0.1` | - | Global UMAP defaults for the browse projection (see [`projection.md`](projection.md)). |
| `PROJECTION_DEFAULTS_BY_EMBEDDER` | dict | - | Per-embedder `(n_neighbors, min_dist)` overrides of the globals (e.g. `"siglip": (10, 0.05)`). |
| `PROJECTION_COMPACT_DEFAULT` | `False` | - | Default for layout compaction.                                                                                |

### `allocated_cpus()` / `resolve_decode_workers()`

`allocated_cpus()` answers "how many CPUs may this process actually run
on", via `os.sched_getaffinity` (falling back to `os.cpu_count()` off
Linux). That is deliberately *not* `os.cpu_count()`: a SLURM job holding
`--cpus-per-task=8` on a 96-core node is entitled to 8, and the affinity
mask is what reflects that.

`resolve_decode_workers(cap=DEFAULT_DECODE_WORKER_CAP)` sizes the image-decode prefetch pool used by
`vtscore.media.image._image_bulk` — the pool that decodes the next batch
while the current batch's GPU forward runs. It returns
`min(cap, allocated_cpus() - 1)`, floored at 1: one
CPU is left for the calling thread, which runs the model processor and
tensor marshalling. Read at call time, so it tracks an allocation the
process only learns about after import.

`VTSEARCH_DECODE_WORKERS` overrides the count outright; `0` disables the
pool and decodes inline on the calling thread.

### `resolve_device()`

Returns the concrete device string this host will actually use:

```python
from vtscore.config import resolve_device

torch_device = resolve_device()
# "cuda" if CUDA is visible AND torch can run a kernel on it
# "mps" on Apple Silicon
# "cpu" otherwise
```

Imports `torch` lazily so simply importing `vtscore.config` does not
pull torch into the process. Returns `"cpu"` if torch isn't installed
at all.

**CUDA is smoke-tested, not just probed for availability.**
`torch.cuda.is_available()` returns `True` whenever a driver and device
are visible, even when the installed torch wheel was built without a
kernel image for that GPU's compute capability - in which case every real
op raises `cudaErrorNoKernelImageForDevice`. `resolve_device()` therefore
launches one tiny kernel (`_cuda_can_run()`, cached per device per process)
before committing to CUDA; if it can't run, the host falls back to `"cpu"`
with a one-time warning instead of crash-looping. This holds for an
explicit `VTSEARCH_DEVICE=cuda`/`cuda:N` pin too: the pin is honoured when
it works and falls back to CPU when it doesn't, so one install runs across
a heterogeneous GPU fleet. The one-time warning names the GPU's compute
capability and the arch list the installed wheel was compiled for, so the
mismatch is legible at a glance (e.g. *"Tesla V100S-PCIE-32GB, compute
capability 7.0; this torch build ships kernels for sm_75, sm_80, ..."*). To
actually use the GPU, reinstall a torch build whose CUDA tag covers its
arch (see `scripts/install.sh`). The right tag is not simply the newest:
the newest wheels drop the oldest architectures, so an old GPU needs an
*older* tag (`cu128` dropped Volta/`sm_70`, so a V100 needs `cu124`).

### Embedding precision and image-processor backend

| Name | Meaning |
|------|---------|
| `EMBED_PRECISION` / `embed_precision()` | Requested / effective embedding compute precision. Half modes are CUDA-only: off CUDA, or on a device without bf16 for a bf16 mode, the effective value is `"fp32"`. `auto` resolves to `bf16`, else `fp16`. |
| `embed_weight_dtype()` | dtype to cast weights to (`fp16` / `bf16` modes), else `None` |
| `embed_autocast_dtype()` | dtype for a `torch.autocast` block (`autocast_*` modes), else `None` |
| `IMAGE_PROCESSOR_BACKEND` / `IMAGE_PROCESSOR_DEVICE` | Requested `transformers` image-processor implementation and resize/normalise device |
| `image_processor_load_kwargs()` / `image_processor_call_kwargs()` | Kwargs for `from_pretrained` / the processor call that apply those two settings |
| `resolved_processor_backend(processor)` / `processor_backend_from_class_name(...)` / `verify_image_processor_backend(processor, *, embedder)` | Report (and warn on) the backend a loaded processor actually uses |

## Server-path access

Not a config concern: there is no configurable server-root allow-list.
Server-side filesystem access is governed by the active login provider;
see [`security.md`](security.md).

## Embedder model identifiers

Every public embedder ID is a plain string constant. They are
*identifiers only* - none of these constants load anything at import
time. The actual download + load is lazy, driven by each embedder's
`load_models()` (reached via `MediaEmbedder.loaded_backbone()`, or the
`vtscore.embedding.loader` getters `get_clap_model`, `get_xclip_model`,
`get_e5_model` that wrap it).

| Constant                       | Value                                                                                  | Notes                                                                                  |
|--------------------------------|----------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| `CLAP_MODEL_ID`                | `"laion/clap-htsat-unfused"`                                                           | Smaller/faster general CLAP; the `clap` embedder.                                      |
| `CLAP_SAMPLE_RATE`             | `48000`                                                                                | Required input sample rate for the CLAP family.                                        |
| `CLAP_MUSIC_MODEL_ID`          | `"laion/larger_clap_music_and_speech"`                                                 | CLAP variant for music + speech.                                                       |
| `CLAP_GENERAL_MODEL_ID`        | `"laion/larger_clap_general"`                                                          | Larger general-purpose CLAP. **Default audio embedder.**                               |
| `AST_MODEL_ID`                 | `"MIT/ast-finetuned-audioset-10-10-0.4593"`                                            | Audio Spectrogram Transformer. 16 kHz mono via `AST_SAMPLE_RATE`.                      |
| `AST_SAMPLE_RATE`              | `16000`                                                                                |                                                                                        |
| `BEATS_CHECKPOINT_REPO`        | `"lpepino/beats_ckpts"`                                                                | Hub mirror of the MIT-licensed BEATs release (no `transformers` implementation).       |
| `BEATS_CHECKPOINT_FILE`        | `"BEATs_iter3_plus_AS2M.pt"`                                                           | Self-supervised encoder, not an AudioSet-finetuned classifier variant.                 |
| `BEATS_SAMPLE_RATE`            | `16000`                                                                                | 16 kHz mono; features are Kaldi fbanks, not waveforms. Siblings: `BEATS_EMBED_DIM`, `BEATS_MAX_SAMPLES`, `BEATS_MIN_SAMPLES`, `BEATS_FBANK_MEAN`, `BEATS_FBANK_STD`. |
| `WHISPER_MODEL_ID`             | `"openai/whisper-base"`                                                                | Used by the audio→text converter (ASR).                                                |
| `WHISPER_SAMPLE_RATE`          | `16000`                                                                                |                                                                                        |
| `XCLIP_MODEL_ID`               | `"microsoft/xclip-base-patch32"`                                                       | Default video embedder.                                                                |
| `VIDEOMAE_MODEL_ID`            | `"OpenGVLab/VideoMAEv2-Base"`                                                          | Vision-only video encoder (no paired text tower; `supports_text=False`).               |
| `LANGUAGEBIND_VIDEO_MODEL_ID`  | `"LanguageBind/LanguageBind_Video_V1.5_FT"`                                            | Alternative video embedder with text-aligned latent.                                   |
| `SIGLIP_MODEL_ID`              | `"google/siglip-base-patch16-224"`                                                     | Default image embedder.                                                                |
| `SIGLIP2_MODEL_ID`             | `"google/siglip2-base-patch16-224"`                                                    | SigLIP v2.                                                                             |
| `SIGLIP2_L_MODEL_ID`           | `"google/siglip2-so400m-patch14-384"`                                                  | SigLIP v2, SO400M/384 (1152-d).                                                        |
| `CLIP_MODEL_ID`                | `"openai/clip-vit-base-patch32"`                                                       | OpenAI CLIP.                                                                           |
| `DINOV2_MODEL_ID`              | `"facebook/dinov2-base"`                                                               | Self-supervised image encoder.                                                         |
| `DINOV3_MODEL_ID`              | `"facebook/dinov3-vitb16-pretrain-lvd1689m"`                                           | DINO v3.                                                                               |
| `EUPE_MODEL_ID`                | `"https://huggingface.co/facebook/EUPE-ViT-B/resolve/main/EUPE-ViT-B.pt"`              | Direct HF URL to EUPE ViT-B weights. Loaded via `torch.hub.load`. FAIR Non-commercial. |
| `CLIP_L_MODEL_ID`              | `"openai/clip-vit-large-patch14"`                                                      | OpenAI CLIP ViT-L/14.                                                                  |
| `SIGLIP_L_MODEL_ID` / `SIGLIP_L_PRETRAINED` | `"ViT-SO400M-14-SigLIP-384"` / `"webli"`                                  | open_clip model name + pretrained tag (not a HF id).                                   |
| `PARASPEECHCLAP_*`             | `SPEECH_MODEL_ID="microsoft/wavlm-large"`, `TEXT_MODEL_ID="ibm-granite/granite-embedding-278m-multilingual"`, `CHECKPOINT_REPO="ajd12342/paraspeechclap-combined"`, `CHECKPOINT_FILE="slap-combined.pth.tar"` | Plus `EMBED_DIM=768`, `SAMPLE_RATE=16000`, `MAX_SAMPLES=16000*30`. |
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
| `VTSEARCH_DECODE_WORKERS`   | Override the image-decode prefetch pool size. `0` decodes inline. Read at call time, not import time. |
| `VTSEARCH_DEVICE`           | Set `DEVICE`. `"auto"` is the default; pin to `"cuda"`, `"cuda:0"`, `"cpu"`, or `"mps"`.        |
| `VTSEARCH_MAX_UPLOAD_MB`    | Set `MAX_UPLOAD_MB` (default 2048 MB). `0` = unlimited.                                         |
| `VTSEARCH_TRAIN_EPOCHS`     | Set `TRAIN_EPOCHS` (head training upper bound).                                                 |
| `VTSEARCH_TRAIN_PATIENCE`   | Set `TRAIN_PATIENCE` (early-stop patience). `0` disables.                                       |
| `VTSEARCH_CALIBRATE_COUNT`  | Set `DEFAULT_CALIBRATE_COUNT` (first-run default; later writes go to per-user settings).        |
| `VTSEARCH_MAX_DECODE_PIXELS`| Set `MAX_DECODE_PIXELS`, the bitmap budget for a single image decode. `0` disables bounding.    |
| `VTSEARCH_MAX_STRUCTURAL_DETECT_PIXELS` | Set `MAX_STRUCTURAL_DETECT_PIXELS` (default 2 MP), the resolution budget for structural local-feature detection. `0` detects at native size. |
| `VTSEARCH_SVM_HEAD_C`       | Set `SVM_HEAD_C`, the production linear SVM head's inverse regularisation strength.             |
| `VTSEARCH_EMBED_PRECISION`  | Set `EMBED_PRECISION`: `fp32` (default), `fp16`, `bf16`, `autocast_fp16`, `autocast_bf16`, `auto`. Compute only - stored vectors stay fp32. |
| `VTSEARCH_IMAGE_PROCESSOR_BACKEND` | Pin the `transformers` image-processor implementation: `torchvision` (default), `pil`, or `auto`. |
| `VTSEARCH_IMAGE_PROCESSOR_DEVICE`  | Where the torchvision processor resizes/normalises: `auto` (default), `cpu`, `cuda`.      |

All of these are read at import time except `VTSEARCH_DECODE_WORKERS`,
which `resolve_decode_workers()` reads per call. Setting any of the
others after `vtscore.config` has loaded has no effect; do the export
before `python -m vtscore` / `python app.py` runs. (Tests that need a
different value re-read the whole package with `config._reload_all()` -
see [Contents](#contents).)

## Typical flow

The end-to-end picture for a library + app deployment:

1. Process starts; `vtscore.config` imports and reads env vars.
2. The host app calls `register_core_config_builder(builder_fn)`, where
   `builder_fn` snapshots its settings store into a `CoreConfig` for
   the current user.
3. Library code calls `CoreConfig.from_settings()`, which delegates to
   `builder_fn` and returns a frozen `CoreConfig` for this call.
4. Background threads spawned from that call keep using the same
   frozen object.

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
- No persisted embeddings, no persisted model weights. This config
  module does not introduce a place to cache them.

---

## Cross-references

- [`architecture.md`](../architecture.md#the-coreconfig-bridge) - why the `CoreConfig` seam exists.
- [`cli.md`](cli.md) - the CLI entry points that build a `CoreConfig` before running.
- [`embedding.md`](embedding.md) - `get_torch_device()` and the concurrency defaults built on `resolve_device()`.
- [`security.md`](security.md) - server-path access rules.
