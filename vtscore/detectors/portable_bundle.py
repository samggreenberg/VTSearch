"""Build a standalone, portable detector bundle for transfer to other parties.

CRITICAL - this is the one sanctioned exception to the "No Persisted Vectors or
MLPs" rule (see ``CLAUDE.md``).  The bundle persists the *trained MLP* (as an
ONNX graph) so a third party can score their own media without VTSearch.  It
deliberately does **not** include any embeddings or raw media: a scoring
detector only needs three things - the trained classifier, the name of the
embedder to run new media through, and the decision threshold.  Everything in
this module is derived from those; no media-derived vectors are ever written.

The bundle is a zip with three members:

* ``detector.onnx``  - the MLP with its sigmoid baked in (input ``embedding``
  ``[batch, dim]`` -> output ``score`` ``[batch, 1]`` in ``[0, 1]``).
* ``manifest.json``  - machine-readable embedder name/dim, threshold, scoring
  convention, and label counts.  Lets the bundle be re-imported and scripted.
* ``README.md``      - human-readable instructions: which embedder to run, the
  threshold, and a copy-paste ``onnxruntime`` inference snippet.

The MLP architecture is fixed (``Linear -> ReLU -> Dropout -> Linear -> 1``; see
:func:`vtscore.training.mlp.build_model`), so the ONNX graph is hand-assembled
via ``onnx.helper`` rather than going through ``torch.onnx``.  That keeps this
module torch-free (it operates on the ``serialize_weights`` nested-list dict),
avoids the dynamo/onnxscript export machinery, and is fully deterministic.
"""

from __future__ import annotations

import io
import json
import zipfile
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

#: ``manifest.json`` format tag and version - bump the version on any
#: backwards-incompatible change to the manifest shape or the ONNX I/O contract.
BUNDLE_FORMAT = "vtsearch-portable-detector"
BUNDLE_FORMAT_VERSION = 1

#: ONNX graph I/O names and the op/ir versions the graph is emitted at.  Gemm,
#: ReLU and Sigmoid are ancient ops, so a conservative opset/ir pair maximises
#: the range of ``onnxruntime`` versions that can load the file.
ONNX_INPUT_NAME = "embedding"
ONNX_OUTPUT_NAME = "score"
_ONNX_OPSET = 17
_ONNX_IR_VERSION = 8

#: Where the README points consumers for context on what produced the bundle.
_PROJECT_URL = "https://github.com/samggreenberg/vtsearch"


def _split_linear_weights(
    weights: dict[str, list],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pull the two ``Linear`` layers out of a ``serialize_weights`` dict.

    Returns ``(W1, b1, W2, b2)`` as ``float32`` arrays, ordered by the
    layer's integer prefix in the state-dict key (so the legacy ``"2."`` and
    current ``"3."`` final-layer keys both resolve correctly).  ``W1`` is the
    hidden layer ``[hidden, input_dim]``; ``W2`` is the output layer
    ``[1, hidden]``.
    """
    import numpy as np  # noqa: PLC0415

    weight_keys = sorted((k for k in weights if k.endswith(".weight")), key=lambda k: int(k.split(".")[0]))
    bias_keys = sorted((k for k in weights if k.endswith(".bias")), key=lambda k: int(k.split(".")[0]))
    if len(weight_keys) != 2 or len(bias_keys) != 2:
        raise ValueError(
            f"Portable export expects a 2-layer MLP (2 weight + 2 bias tensors); got keys {sorted(weights)}"
        )
    return (
        np.asarray(weights[weight_keys[0]], dtype=np.float32),
        np.asarray(weights[bias_keys[0]], dtype=np.float32),
        np.asarray(weights[weight_keys[1]], dtype=np.float32),
        np.asarray(weights[bias_keys[1]], dtype=np.float32),
    )


def embedding_dim_from_weights(weights: dict[str, list]) -> int:
    """Return the input embedding dimensionality of a serialized detector MLP."""
    w1, _b1, _w2, _b2 = _split_linear_weights(weights)
    return int(w1.shape[1])


def mlp_weights_to_onnx(weights: dict[str, list]) -> bytes:
    """Serialise a trained detector MLP to a standalone ONNX graph.

    The graph computes ``sigmoid(Gemm(relu(Gemm(x, W1, b1)), W2, b2))`` - i.e.
    the exact forward pass of the trained model with the inference-time sigmoid
    baked in (the trained ``nn.Sequential`` emits raw logits).  Dropout is a
    no-op at inference and is omitted.  The batch dimension is dynamic.
    """
    from onnx import TensorProto, checker, helper, numpy_helper  # noqa: PLC0415

    w1, b1, w2, b2 = _split_linear_weights(weights)
    input_dim = int(w1.shape[1])

    initializers = [
        numpy_helper.from_array(w1, "hidden.weight"),
        numpy_helper.from_array(b1, "hidden.bias"),
        numpy_helper.from_array(w2, "output.weight"),
        numpy_helper.from_array(b2, "output.bias"),
    ]
    nodes = [
        # Gemm with transB=1 computes Y = X @ W^T + b, matching nn.Linear.
        helper.make_node(
            "Gemm", [ONNX_INPUT_NAME, "hidden.weight", "hidden.bias"], ["hidden_pre"], name="hidden", transB=1
        ),
        helper.make_node("Relu", ["hidden_pre"], ["hidden"], name="relu"),
        helper.make_node("Gemm", ["hidden", "output.weight", "output.bias"], ["logit"], name="output", transB=1),
        helper.make_node("Sigmoid", ["logit"], [ONNX_OUTPUT_NAME], name="sigmoid"),
    ]
    graph = helper.make_graph(
        nodes,
        "vtsearch_detector",
        [helper.make_tensor_value_info(ONNX_INPUT_NAME, TensorProto.FLOAT, ["batch", input_dim])],
        [helper.make_tensor_value_info(ONNX_OUTPUT_NAME, TensorProto.FLOAT, ["batch", 1])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="vtsearch",
        opset_imports=[helper.make_opsetid("", _ONNX_OPSET)],
    )
    model.ir_version = _ONNX_IR_VERSION
    checker.check_model(model)
    return model.SerializeToString()


def build_manifest(
    *,
    detector_name: str,
    media_type: str,
    embedder: str,
    embedder_display_name: str,
    embedder_type: str,
    embedding_dim: int,
    threshold: float,
    good_count: int,
    bad_count: int,
    exported_by: str,
    exported_at: str,
) -> dict[str, Any]:
    """Assemble the machine-readable ``manifest.json`` payload.

    Captures everything a consumer needs to score with the bundle: the embedder
    to run new media through, the embedding dimensionality, the scoring
    convention (sigmoid + threshold), and provenance.  ``contains_media_data``
    is always ``False`` - the manifest documents that the bundle carries the
    classifier only, never embeddings or raw media.
    """
    return {
        "format": BUNDLE_FORMAT,
        "format_version": BUNDLE_FORMAT_VERSION,
        "detector_name": detector_name,
        "media_type": media_type,
        "model_file": "detector.onnx",
        "embedder": {
            "name": embedder,
            "display_name": embedder_display_name,
            "type": embedder_type,
            "embedding_dim": embedding_dim,
        },
        "scoring": {
            "input": ONNX_INPUT_NAME,
            "output": ONNX_OUTPUT_NAME,
            "activation": "sigmoid",
            "threshold": round(float(threshold), 6),
            "decision": "good if score >= threshold else bad",
        },
        "training_labels": {"good": int(good_count), "bad": int(bad_count)},
        "exported_by": exported_by,
        "exported_at": exported_at,
        "contains_media_data": False,
        "notes": (
            "Standalone scoring model derived from labeled media. Contains the trained "
            "classifier only - no raw media and no embeddings are included."
        ),
    }


def render_readme(manifest: dict[str, Any]) -> str:
    """Render the human-readable ``README.md`` for a bundle from its manifest."""
    emb = manifest["embedder"]
    scoring = manifest["scoring"]
    dim = emb["embedding_dim"]
    threshold = scoring["threshold"]
    emb_label = emb["display_name"] or emb["name"]
    return f"""# Portable detector: {manifest["detector_name"]}

This is a **standalone scoring model** exported from
[VTSearch]({_PROJECT_URL}). It ranks **{manifest["media_type"]}** items by how
well they match the trained detector, without needing VTSearch itself.

> **What's in here:** the trained classifier only. There is **no raw media and
> there are no embeddings** in this bundle - just a small neural network and the
> instructions to run it.

## Files

| File | Purpose |
| --- | --- |
| `detector.onnx` | The trained classifier. Input `{scoring["input"]}` `[batch, {dim}]`, output `{scoring["output"]}` `[batch, 1]` in `[0, 1]`. |
| `manifest.json` | Machine-readable embedder, threshold, and scoring details. |
| `README.md` | This file. |

## How to score your own items

1. **Embed** each item with the **{emb_label}** embedder
   (VTSearch embedder id: `{emb["name"]}`, type: `{emb["type"] or "n/a"}`).
   This must be the *same* embedder family the detector was trained on, and it
   must produce a **{dim}-dimensional** vector. This bundle does **not** include
   the embedder - obtain it separately.
2. **Run** the embedding through `detector.onnx`. The model applies its sigmoid
   internally, so the output is already a probability in `[0, 1]`.
3. **Decide**: an item is **good** when `score >= {threshold}` and **bad**
   otherwise. The threshold is tunable - raise it for precision, lower it for
   recall.

## Example (Python + onnxruntime)

```python
import json
import numpy as np
import onnxruntime as ort

manifest = json.load(open("manifest.json"))
threshold = manifest["scoring"]["threshold"]

session = ort.InferenceSession("detector.onnx")

# embeddings: float32 array of shape [N, {dim}] from the {emb["name"]} embedder.
embeddings = np.zeros((1, {dim}), dtype=np.float32)  # replace with real vectors

scores = session.run(["{scoring["output"]}"], {{"{scoring["input"]}": embeddings}})[0]
labels = ["good" if float(s) >= threshold else "bad" for s in scores.ravel()]
print(scores.ravel(), labels)
```

## Provenance

- Trained on {manifest["training_labels"]["good"]} good / {manifest["training_labels"]["bad"]} bad labeled examples.
- Exported by {manifest["exported_by"]} at {manifest["exported_at"]}.
"""


def build_bundle(*, weights: dict[str, list], manifest: dict[str, Any]) -> bytes:
    """Build the full portable-detector zip (``.onnx`` + manifest + README).

    *weights* is a :func:`vtscore.detectors.training.serialize_weights` dict;
    *manifest* is the output of :func:`build_manifest`.  Returns the zip bytes,
    ready to stream as a file download.
    """
    onnx_bytes = mlp_weights_to_onnx(weights)
    readme = render_readme(manifest)
    manifest_text = json.dumps(manifest, indent=2)

    buf = io.BytesIO()
    # Fixed entry timestamps so the same detector/dataset exports byte-stable.
    fixed_date = (1980, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in (
            ("detector.onnx", onnx_bytes),
            ("manifest.json", manifest_text),
            ("README.md", readme),
        ):
            info = zipfile.ZipInfo(name, date_time=fixed_date)
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, data)
    return buf.getvalue()
