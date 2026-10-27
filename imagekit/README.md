# imagekit

Face localisation and embedding with a pluggable backend ("provider")
architecture, on ONNX Runtime. A third top-level package beside `vtsearch/`
and `vtscore/`, importable on its own (`import imagekit`), with two CLIs under
`scripts/`.

The point of the seam: an application codes against one `FacePipeline`
interface and one `FaceRecord` type, and a server swaps the backend with an
environment variable. Two providers ship:

| Provider | What runs | Selected by |
|---|---|---|
| `oss` | This package's own SCRFD (or RetinaFace) detector and MagFace/ArcFace embedder, open weights, ONNX Runtime | `IMAGEKIT_FACE_PROVIDER=oss` (the default) |
| `vista` | A proprietary `vistaimage` package behind the same interface, imported lazily; absent on most machines | `IMAGEKIT_FACE_PROVIDER=vista` |

A third party registers its own provider under a name of its choosing (see
[Registering an external provider](#registering-an-external-provider)).

Inference only: there is no training code here. Python 3.10+.

## Contents

- [Quick start](#quick-start)
- [Models: where to get them](#models-where-to-get-them)
- [ONNX export notes](#onnx-export-notes)
- [The image tensor](#the-image-tensor)
- [JSON config schema](#json-config-schema)
- [Core types](#core-types)
- [Provider architecture](#provider-architecture)
- [The index-compat guard](#the-index-compat-guard)
- [CSV and NPZ formats](#csv-and-npz-formats)
- [CLI usage](#cli-usage)
- [Inside VTSearch](#inside-vtsearch)
- [Package layout](#package-layout)
- [Testing](#testing)

## Quick start

```python
from imagekit import RuntimeOptions, read_image_tensor
from imagekit.face import get_pipeline, read_pipeline_config

config = read_pipeline_config("faces.json")        # model paths; see "JSON config schema"
pipeline = get_pipeline()                          # IMAGEKIT_FACE_PROVIDER, else "oss"
pipeline.initialize(RuntimeOptions(num_threads=4), config)

records = pipeline.process(read_image_tensor("photo.jpg"), media_id="photo.jpg")
for face in records:                               # best first; every detection is embedded
    print(face.bbox, face.confidence, face.quality, face.compat, face.unit_embedding()[:4])
```

Or the pieces on their own:

```python
from imagekit.face import FaceDetector, FaceEmbeddingExtractor

detector = FaceDetector()
detector.initialize(RuntimeOptions(), "models/det_10g.onnx")          # + models/det_10g.json if present
extractor = FaceEmbeddingExtractor()
extractor.initialize(RuntimeOptions(), "models/w600k_r50.onnx", "models/arcface_r50.json")

image = read_image_tensor("photo.jpg")
for det in detector.detect(image):                                     # FaceDetection: bbox, landmarks, confidence
    raw = extractor.extract(det, image)                                # (512,) float32, NOT L2-normalised
```

Calling `detect()` or `extract()` before `initialize()` raises
`Must initialize FaceDetector first!` / `Must initialize FaceEmbeddingExtractor first!`;
`initialize()` raises on any failure (a missing file, a bad model, an
uninstalled execution provider) and leaves nothing half-built.

## Models: where to get them

Model paths are **always arguments**: nothing here downloads weights or
hard-codes a path. Read the licence that ships with any weights you deploy;
the InsightFace model-zoo weights are for non-commercial research only.

### Detector

- **SCRFD** (primary). The [InsightFace model zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo)
  ships `buffalo_l.zip`, whose `det_10g.onnx` is SCRFD-10GF with keypoints, the
  model the packaged default config (`scrfd_10g_gnkps`) describes. The
  [SCRFD page](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)
  lists the other sizes (500M, 2.5G, 10G, 34G, with and without `_kps`) and the
  export script (`tools/scrfd2onnx.py`) for a checkpoint of your own. Any SCRFD
  export with 2 anchors at strides 8/16/32 works; set `feat_stride` /
  `num_anchors` in the config for another head layout.
- **RetinaFace-ResNet50**. Selected by a config whose `model.name` starts with
  `retinaface`; decoded with the usual prior boxes (`min_sizes`
  `[[16, 32], [64, 128], [256, 512]]`, `variances` `[0.1, 0.2]`) over `loc` /
  `conf` / `landms` heads, as the common PyTorch RetinaFace exports produce.
  `imagekit/face/detection/retinaface_r50.json` is the reference config:
  ImageNet normalisation and a 2048 px cap, as specified.
- **MTCNN** is the legacy alternative (VTSearch's face path before #5320,
  through `facenet-pytorch`). It is deliberately not implemented here.

### Embedder

- **MagFace** (preferred: the magnitude of its vector encodes face quality).
  The [official release](https://github.com/IrvingMeng/MagFace) publishes an
  iResNet-100 checkpoint (112 x 112 in, 512-d out). Export it to ONNX (below)
  and describe it with `imagekit/face/extraction/magface_ir100.json`
  (compat `oss-magface-ir100:1`).
- **ArcFace** (InsightFace). `buffalo_l.zip`'s `w600k_r50.onnx` is an
  ArcFace-loss ResNet-50 with the same 112 x 112 / 512-d contract; describe it
  with `imagekit/face/extraction/arcface_r50.json` (compat
  `oss-arcface-r50:1`). Other InsightFace recognition models (`glintr100`,
  `w600k_mbf`, ...) take the same config with their own `name` and
  `compatibility`.

## ONNX export notes

A PyTorch face embedder exports with a dynamic batch dimension so the
extractor can batch every face of a photo in one run:

```python
import torch
model.eval()                                                   # 112x112 RGB in, 512-d out
torch.onnx.export(
    model, torch.zeros(1, 3, 112, 112), "magface_ir100.onnx",
    input_names=["input.1"], output_names=["embedding"],
    dynamic_axes={"input.1": {0: "N"}, "embedding": {0: "N"}},
    opset_version=13,
)
```

A fixed batch dimension also works: the extractor reads the input shape and
feeds one crop per run when it is fixed. The detector takes a `(1, 3, H, W)`
input; SCRFD exports with a dynamic spatial size are letterboxed to the
config's `Resize`, fixed ones must match it. Verify an export with
`onnxruntime.InferenceSession(path).get_inputs()` / `get_outputs()`: an SCRFD
model with keypoints has 9 outputs (3 score maps, 3 box maps, 3 keypoint
maps, strides 8/16/32 in that order); without keypoints, 6.

**Channel order and normalisation are the config's business, not the
code's.** This package feeds RGB (see [The image tensor](#the-image-tensor)).
A model trained on OpenCV-ordered input gets a `{"name": "ChannelSwap"}` step
in its transforms; one trained on `(x - 127.5) / 128` gets that `Normalize`;
one trained on `(x/255 - 0.5) / 0.5` gets `mean` / `std` `0.5`. The reference
configs encode the MagFace / ArcFace contract the package was specified
against; a deployment whose weights differ edits its sidecar JSON, never the
code.

To run on a GPU, install `onnxruntime-gpu` in place of `onnxruntime` and pass
`RuntimeOptions(use_gpu=True, gpu_device_id=N)`; a request for CUDA on a build
without it raises at `initialize()` rather than silently running on the CPU.

## The image tensor

Every API takes and returns one image representation: a `numpy` array of
shape `(H, W, 3)`, `uint8`, **RGB**. `imagekit.read_image_tensor(path)` and
`imagekit.decode_image_buffer(bytes)` produce it (via OpenCV, converting from
BGR); `imagekit.util.ensure_image_tensor` accepts grayscale and RGBA
near-misses and refuses anything else. A PIL image becomes one with
`np.asarray(img.convert("RGB"))`.

## JSON config schema

Three JSON files, each read next to the thing it describes.

### Detector config: `<model>.json` beside the detector `.onnx`

Looked up as the sidecar `<model stem>.json`, else an explicit
`model_config_path`, else the packaged default
`imagekit/face/detection/scrfd_10g_gnkps.json`, which is exactly:

```json
{
  "model": {
    "name": "scrfd_10g_gnkps",
    "feat_stride": [8, 16, 32],
    "num_anchors": 2,
    "score_threshold": 0.5,
    "iou_threshold": 0.45,
    "nms_pre": 1000,
    "max_nms": 30000,
    "topk": 400,
    "min_image_size": 12,
    "max_image_size": 2048,
    "transforms": [
      {"name": "Normalize",
       "args": {"mean_vals": [127.5, 127.5, 127.5],
                "scale_vals": [0.0078125, 0.0078125, 0.0078125]}},
      {"name": "Resize",
       "args": {"height": 640, "width": 640,
                "preserve_aspect": true, "pad": 1}},
      {"name": "ToTensor"}
    ]
  }
}
```

| Field | Meaning |
|---|---|
| `name` | Which decoder: a name starting with `retinaface` selects the RetinaFace prior-box decoder; anything else is SCRFD. |
| `feat_stride`, `num_anchors` | The head layout: one score/box/keypoint map per stride, `num_anchors` anchors per cell. RetinaFace reads its anchor sizes from `min_sizes` and its variances from `variances` instead. |
| `score_threshold` | Minimum confidence; `detect(..., confidence_threshold=)` overrides it per call. |
| `iou_threshold` | NMS overlap. |
| `nms_pre` | Candidates kept per level before NMS (the best by score). |
| `max_nms` | Total candidates entering NMS. |
| `topk` | Detections returned at most, best first. |
| `min_image_size` | An image smaller than this on either side yields no faces. |
| `max_image_size` | A longer side above this is shrunk first (the scale is folded into the map-back). |
| `transforms` | The preprocessing chain, in order, ending in `ToTensor`. |

Transforms: `Normalize` (`mean_vals` / `scale_vals` on the 0-255 scale, or
torchvision-style `mean` / `std` on 0-1), `Resize` (`height` + `width` with
`preserve_aspect` and `pad` for a letterbox, `preserve_aspect` alone to fit
without padding, neither to stretch, or `max_size` alone to cap the longer
side), `ChannelSwap` (RGB to BGR), `ToTensor` (`(H, W, 3)` to
`(1, 3, H, W)` float32), and `Align` below. A third party adds one with
`imagekit.util.register_transform`.

**Inference flow** (SCRFD): letterbox to 640 x 640 keeping the aspect
(top-left, zero padding bottom/right), normalise `(x - 127.5) / 128`, NCHW
float32; per stride 8/16/32 and per anchor, decode the distance-coded box and
the keypoint offsets from the anchor centre; filter by score; keep `nms_pre`
per level; NMS at IoU 0.45; cap at `topk`; divide by the letterbox scale and
clamp to the image.

### Extractor config: `<model>.json` beside the embedder `.onnx` (required)

```json
{
  "model": {
    "name": "magface_ir100",
    "compatibility": "oss-magface-ir100:1",
    "embedding_dim": 512,
    "transforms": [
      {"name": "Align", "args": {"height": 112, "width": 112, "fallback": "crop"}},
      {"name": "Normalize", "args": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}},
      {"name": "ToTensor"}
    ]
  }
}
```

There is **no packaged default** for the extractor, on purpose: the config is
what names the embedding space (`compatibility`, or `name` + `version`, which
becomes `oss-<name>:<version>`), and a default would let two different weights
claim the same space. `FaceEmbeddingExtractor.get_compatibility()` returns
that string verbatim; different weights get a different string by
construction. The reference configs for the two open models sit beside the
extractor module (`imagekit/face/extraction/magface_ir100.json`,
`imagekit/face/extraction/arcface_r50.json`): copy one beside your weights as
`<model stem>.json`, or pass it as `model_config_path`.

`Align` is `imagekit.face.face_align.norm_crop`: the least-squares similarity
taking the detection's five landmarks onto the standard ArcFace 112 x 112
template, applied to the **full** image, so the crop is what the embedder was
trained on. `fallback` says what a detection without landmarks gets:
`"crop"` (its box, resized) or `"error"`.

### Pipeline config: the file `IMAGEKIT_FACE_CONFIG` and `process_faces.py` take

```json
{
  "provider": "oss",
  "runtime": {"use_gpu": false, "gpu_device_id": 0, "num_threads": 4},
  "detector": {"model_bundle": "models/det_10g.onnx",
               "config": null,
               "confidence_threshold": 0.5},
  "embedder": {"model": "models/w600k_r50.onnx",
               "config": "models/arcface_r50.json"}
}
```

`provider` and `runtime` are optional (the environment's provider and the
runtime defaults otherwise). Relative paths resolve against the config file's
own directory, so a models directory moves as one unit. `model_bundle` is the
`.onnx` file or a directory holding exactly one. The `vista` provider reads
the same keys and passes `model_bundle` / `model` / `config` to the
proprietary classes.

## Core types

`imagekit.RuntimeOptions(use_gpu=False, gpu_device_id=0, num_threads=0)` is
the plain options object every `initialize()` takes; `num_threads=0` leaves
ONNX Runtime's default.

`imagekit.face.FaceRecord` is **the only face type an application sees**:

| Field | Type | Meaning |
|---|---|---|
| `bbox` | `tuple[float, float, float, float]` | `(x1, y1, x2, y2)` in image pixels |
| `landmarks` | `np.ndarray (5, 2)` or `None` | left eye, right eye, nose, left mouth corner, right mouth corner |
| `confidence` | `float` | detector confidence |
| `embedding` | `np.ndarray (512,) float32` | **unnormalised**, as the model produced it |
| `quality` | `float` | L2 norm of `embedding` (magnitude-aware losses encode quality there) |
| `compat` | `str` | embedding-space tag, e.g. `oss-magface-ir100:1`, `vistaface:3` |
| `media_id` | `str` | the image it came from |

Helpers: `unit_embedding()` (an L2-normalised copy), `to_row()` /
`from_row()` (CSV, embedding omitted), `from_detection(detection, embedding,
compat, media_id)`. `FaceDetection` is the detector-only half (`bbox`,
`landmarks`, `confidence`), with `detection_csv_header()` /
`detection_to_row()` / `detection_from_row()` for the detection CLI.

## Provider architecture

```python
class FacePipeline(Protocol):
    def initialize(self, opts: RuntimeOptions, config: Mapping[str, Any]) -> None: ...
    def process(self, image, *, confidence_threshold=None, media_id="") -> list[FaceRecord]: ...
    def process_buffer(self, buffer: bytes, *, confidence_threshold=None, media_id="") -> list[FaceRecord]: ...
    def compatibility(self) -> str: ...
```

`process` is detect + align + embed on one decoded image: one `FaceRecord`
per detection (**every** detection is embedded, not just the first), highest
confidence first. `process_buffer` decodes an encoded image first.

- **`imagekit.face.providers.oss.OssFacePipeline`** composes this package's
  `FaceDetector` and `FaceEmbeddingExtractor`.
- **`imagekit.face.providers.vista.VistaFacePipeline`** wraps the proprietary
  `vistaimage` package (with `vistacommon`), which exposes nearly the same
  classes: `vistaimage.face.FaceDetector` with `initialize(runtime_opts,
  model_bundle)` / `detect(image_tensor, confidence_threshold)`,
  `vistaimage.face.FaceEmbeddingExtractor` with `initialize(runtime_opts,
  model_filepath, model_config_path=None)` / `extract(detection, image_tensor)`
  / `get_compatibility()`, `vistaimage.read_image_tensor(image_filepath=...)`,
  and `vistacommon.RuntimeOptions(use_gpu, gpu_device_id, num_threads)`. Two
  things differ and are mapped at the boundary: its embedding object carries a
  `.values` sequence, and its compatibility is an integer, which becomes
  `vistaface:<n>`. Both packages are imported **lazily inside `initialize()`**,
  so the module loads everywhere and the missing package is named
  (`VistaProviderUnavailable`, an `ImportError`) only when the provider is
  asked for. Nothing of `vistaimage`'s leaks out: every result is a
  `FaceRecord`.

### The registry

```python
from imagekit.face import get_pipeline, available_providers, default_provider_name

get_pipeline("oss")          # a new, uninitialised OssFacePipeline
get_pipeline("vista")        # a new VistaFacePipeline (vistaimage imported at initialize())
get_pipeline()               # the default: IMAGEKIT_FACE_PROVIDER, else "oss"
available_providers()        # ["oss", "vista", ...entry points...]
```

`imagekit.face.load_pipeline(config, opts=None, name=None)` is `get_pipeline`
plus `initialize`, taking the provider from `name`, else the config's
`provider`, else the environment.

### Registering an external provider

At runtime:

```python
from imagekit.face import register_provider
register_provider("acme", AcmeFacePipeline)       # a zero-argument factory
```

Or, for an installed package, an entry point in the `imagekit.face_providers`
group, loaded on the first lookup of a name the registry does not know:

```toml
[project.entry-points."imagekit.face_providers"]
acme = "acme_faces:AcmeFacePipeline"
```

A plugin cannot shadow a built-in or another plugin's name (a clash is logged
and ignored; `register_provider(..., replace=True)` is the explicit override).
The provider implements the protocol above and converts its own types to
`FaceRecord` at its boundary, as `vista` does.

## The index-compat guard

`imagekit.face.compat` turns the `compat` tag into a rule for any index or
dataset that stores face vectors:

```python
from imagekit.face import stamp_compat, check_compat, CompatibilityError

stamp_compat(index_metadata, pipeline.compatibility())   # at creation
check_compat(index_metadata, record.compat)               # before every add and every query
```

`check_compat` raises `CompatibilityError` when the tags differ, and also
when the index carries no stamp at all (a guard that passed on a missing
stamp would pass every pre-existing mixed index). `single_compat(records)`
asserts a batch shares one space.

## CSV and NPZ formats

**Detections CSV** (`detect_faces.py`): header
`media_id, face_index, x1, y1, x2, y2, confidence, left_eye_x, left_eye_y, right_eye_x, right_eye_y, nose_x, nose_y, mouth_left_x, mouth_left_y, mouth_right_x, mouth_right_y`.
Missing landmarks are empty cells. Faces are numbered per `media_id` in
detection order (0 is the best).

**Face records CSV** (`process_faces.py`, `<prefix>.csv`): header
`media_id, face_index, x1, y1, x2, y2, confidence, quality, compat, <the ten landmark columns>`.
The embedding is deliberately absent.

**Embeddings NPZ** (`<prefix>.npz`): one array per face under the key
`"<media_id>::<face_index>"` (the raw `(512,)` float32 vector), plus
`__compat__` (the run's compat tag; every vector in an archive shares one)
and `__format__` (`imagekit-face-embeddings:1`). `load_face_records(csv,
npz)` joins the two and refuses a row whose vector is missing or whose tag is
not the archive's.

## CLI usage

```
python scripts/detect_faces.py <detect_model> <output> (--dir DIR | --list FILE | --img PATH)
    [--use-gpu] [--gpu-device-id N] [--num-threads N] [--confidence-threshold T] [--model-config PATH]
```

Detection only. `<detect_model>` is the `.onnx` or a directory holding it and
its JSON config; `<output>` is the detections CSV. Prints `Total Detects: N`.

```
python scripts/process_faces.py <config> <output_prefix> (--dir DIR | --list FILE | --img PATH)
    [--provider NAME] [--use-gpu] [--gpu-device-id N] [--num-threads N] [--confidence-threshold T]
```

The full pipeline through the provider registry: `<config>` is a pipeline
config (above); writes `<output_prefix>.csv` and `<output_prefix>.npz`;
prints `Total Faces: N`. `--provider` overrides the config's and the
environment's; the runtime flags override the config's `runtime` block.

Both: `--dir` walks a directory recursively for images (ids are paths
relative to it), `--list` reads one path per line (`#` comments allowed),
`--img` is one file. A file that cannot be read or processed is reported on
stderr as a `WARNING:` line and skipped; the run continues. The logic lives in
`imagekit/cli.py`, so both entry points are importable and tested in-process.

## Inside VTSearch

Every face in the app goes through this package: the `image2face` converter
(which cuts face crops out of photos) and the `face` embedder both reach the
server's provider through `vtscore/media/face/imagekit_bridge.py`.

- **Face is on only where models are.** The app downloads none. It reads the
  pipeline config `IMAGEKIT_FACE_CONFIG` names, else `face.json` in the
  `imagekit` directory of its models folder (`VTSEARCH_MODELS_DIR`, default
  `data/models/`), and checks that the provider
  exists and every model file the config names is on disk, loading nothing
  (`imagekit_bridge.face_models()`). Without them the converter hides itself,
  the app's listings leave out the `face` media type and embedder, and a run
  that asks for faces anyway fails naming what is missing. Models put in place
  later turn Face on with no restart. The provider is `IMAGEKIT_FACE_PROVIDER`,
  else the config's `provider`, else `oss`. See "Face models" in
  `docs/DEPLOYMENT.md`.
- **The converter** hands each photo to the provider, which detects **and
  embeds** every face from the full photo in one pass; each crop leaves with
  its vector filed under the `face` embedder and its quality and compat tag in
  `custom_metadata` (`Face Quality`, `Face Compat`), so the embed stage has
  nothing to do and no face is embedded twice. The stored crop is cut from the
  provider's box and, with `align: roll` (the default), turned upright by its
  landmarks.
- **The `face` embedder** (`vtscore/media/face/embedder_imagekit.py`) fronts
  the seam for crops that arrive without a vector (a reload that lost its
  vectors): the provider runs on the crop itself and the best detection's
  embedding is the crop's. It refuses to add its vector to a crop stamped with
  another provider's `Face Compat`, so one dataset holds one embedding space.
- The runtime options follow the app's own device (`VTSEARCH_DEVICE`) and
  thread (`VTSEARCH_TORCH_THREADS`) settings; a config's `runtime` block is
  for the CLIs.

## Package layout

```
imagekit/
    __init__.py            # read_image_tensor, decode_image_buffer, RuntimeOptions
    util.py                # the image tensor, letterbox/NMS, the transform registry, config lookup
    runtime.py             # RuntimeOptions -> onnxruntime providers and session options
    cli.py                 # detect_faces_main / process_faces_main (behind scripts/)
    face/
        __init__.py        # the public face API
        records.py         # FaceRecord, FaceDetection, CSV rows, compat tag helpers
        pipeline.py        # FacePipeline protocol, provider registry, pipeline config
        compat.py          # the index-compat guard
        face_util.py       # CSV / NPZ save and load
        face_align.py      # norm_crop and the Align transform
        detection/
            face_detector.py           # FaceDetector: SCRFD and RetinaFace decoders
            scrfd_10g_gnkps.json       # the packaged default detector config
            retinaface_r50.json        # reference RetinaFace config
        extraction/
            face_embedding_extractor.py  # FaceEmbeddingExtractor (ONNX, open weights)
            magface_ir100.json           # reference extractor configs (copy beside your weights)
            arcface_r50.json
        providers/
            oss.py         # OssFacePipeline
            vista.py       # VistaFacePipeline (lazy import of vistaimage / vistacommon)
scripts/
    detect_faces.py        # detection-only CLI
    process_faces.py       # full pipeline CLI
```

## Testing

`./run-tests.sh imagekit` runs `tests_lib/imagekit/`: the detector and
embedder on **synthetic ONNX graphs** built with `onnx.helper`
(`tests_lib/imagekit/_synthetic_models.py`), whose heads encode known faces so
the decoders, NMS, the scale map-back and the whole pipeline run through ONNX
Runtime with no weights on disk; alignment; the records and their CSV / NPZ
forms; the registry and entry points; the `vista` wrapper against a fake
in-test `vistaimage` / `vistacommon` (lazy import, record conversion, the
`vistaface:<n>` mapping, the clear error when absent); and the two CLIs.

Tests against real weights skip unless `IMAGEKIT_TEST_MODEL_DIR` names a
directory holding `det_10g.onnx` and `w600k_r50.onnx`; set
`IMAGEKIT_TEST_FACE_IMAGE` to a photo with a face for the positive cases.
