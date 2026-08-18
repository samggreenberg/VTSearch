"""#3160: localise *where* the SO400M/384 forward diverges between two devices.

    python probe_mechanism.py --out <dir> [--images 8]

#3143 closed two hypotheses (TF32, cuDNN algorithm choice) and left two open: a
capability-selected attention backend, or GEMM tiling/accumulation order that
follows SM count.  Those are distinguishable, and cheaply, because they predict
different *shapes* of divergence:

* a **backend** switch is a step -- the first attention call differs, and the
  same layer's inputs before it are bit-identical;
* **tiling** is diffuse -- every matmul differs a little from the very first
  block, and the error grows with depth.

So this probe records, for a fixed 8-image batch:

1. the vision tower's hidden state after **every block**, hashed and summarised,
   so a cross-node diff shows the first layer that moves and the depth profile;
2. the same forward under each **SDPA backend forced** (math / efficient /
   flash / cudnn).  If forcing ``math`` makes two nodes agree, the cause is
   named *and* a determinism knob exists.  If they still disagree, it is not the
   attention backend;
3. bare-op fingerprints at several GEMM shapes, to see whether a plain matmul
   already differs on this device.

Run it on two nodes and diff the JSON with ``analyze_mechanism.py``.  It touches
no shared state and takes about a minute.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VG_SOURCE = Path("/exp/scale26/datasets/external/vtsearch-demos/visual_genome")


def log(msg: str) -> None:
    print(msg, flush=True)


def _stamp(tensor) -> dict:
    t = tensor.detach().float().cpu().contiguous()
    return {
        "sha256": hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16],
        "sum": float(t.double().sum()),
        "absmax": float(t.abs().max()),
        "shape": list(t.shape),
    }


def _images(n: int) -> list[Path]:
    files = sorted((VG_SOURCE / "VG_100K").glob("*.jpg"), key=lambda p: int(p.stem))
    if not files:
        raise SystemExit(f"no images under {VG_SOURCE}")
    return files[:n]


def _inputs(images: list[Path], emb):
    """The shipped preprocessing path, so what enters the tower is what a build feeds it."""
    from PIL import Image

    from vtscore.media.embedder import to_model_inputs

    pil = [Image.open(p).convert("RGB") for p in images]
    inputs = emb._processor(images=pil, return_tensors="pt")  # noqa: SLF001 -- a probe
    return to_model_inputs(inputs, emb._model)  # noqa: SLF001


def _encoder_layers(model):
    """The vision tower's block list, whatever this checkpoint calls it."""
    for path in ("vision_model.encoder.layers", "vision_tower.vision_model.encoder.layers"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return list(obj), path
    raise SystemExit("could not find the vision encoder's block list on this model")


def tower_layers(emb, inputs, backend: str | None) -> dict:
    """Per-block hidden states from the **shipped** forward, SDPA backend forced.

    Hooks rather than ``output_hidden_states=True``: the public call is
    ``get_image_features``, and re-entering the tower by hand risks measuring a
    forward the builder never runs (SigLIP2 towers differ in signature across
    the fixed-res and NaFlex variants).
    """
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from vtscore.media.embedder import embed_autocast

    model = emb._model  # noqa: SLF001
    layers, path = _encoder_layers(model)
    captured: dict[str, dict] = {}
    handles = []

    def hook(idx):
        def fn(_module, _args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured[str(idx)] = _stamp(tensor)

        return fn

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(hook(i)))

    backends = {
        "math": [SDPBackend.MATH],
        "efficient": [SDPBackend.EFFICIENT_ATTENTION],
        "flash": [SDPBackend.FLASH_ATTENTION],
        "cudnn": [SDPBackend.CUDNN_ATTENTION],
    }

    try:
        with torch.no_grad(), embed_autocast():
            if backend is None:
                out = model.get_image_features(**inputs)
            else:
                with sdpa_kernel(backends[backend]):
                    out = model.get_image_features(**inputs)
    finally:
        for h in handles:
            h.remove()

    tensor = out[0] if isinstance(out, tuple) else out
    return {"block_path": path, "n_layers": len(layers), "layers": captured, "image_features": _stamp(tensor)}


def gemm_shapes() -> dict:
    """A ladder of matmul shapes, all from CPU-seeded inputs."""
    import torch

    gen = torch.Generator(device="cpu").manual_seed(0)
    out = {}
    for m, k, n in ((64, 64, 64), (729, 1152, 1152), (729, 1152, 4304), (4096, 4096, 4096)):
        a = torch.randn(m, k, generator=gen).cuda()
        b = torch.randn(k, n, generator=gen).cuda()
        out[f"{m}x{k}x{n}"] = _stamp(a @ b)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--images", type=int, default=8)
    ap.add_argument("--embedder", default="siglip2_l")
    args = ap.parse_args(argv)

    import torch

    from vtscore.embedding import initialize_models
    from vtscore.media import get_embedder

    node = socket.gethostname()
    out_dir = Path(args.out) / node
    out_dir.mkdir(parents=True, exist_ok=True)

    initialize_models()
    emb = get_embedder(args.embedder)
    emb.load_models()

    props = torch.cuda.get_device_properties(0)
    record: dict = {
        "hostname": node,
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "embedder": args.embedder,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": "sm_%d%d" % torch.cuda.get_device_capability(0),
        "multi_processor_count": props.multi_processor_count,
        "torch": torch.__version__,
        "gemm": gemm_shapes(),
    }
    log(f"{node}: {record['gpu_name']} ({record['multi_processor_count']} SMs)")

    images = _images(args.images)
    inputs = _inputs(images, emb)
    record["input"] = {
        "images": [p.name for p in images],
        "tensors": {k: _stamp(v) for k, v in inputs.items() if hasattr(v, "detach")},
    }

    for backend in (None, "math", "efficient", "flash", "cudnn"):
        key = backend or "default"
        try:
            record.setdefault("backends", {})[key] = tower_layers(emb, inputs, backend)
            log(f"  backend {key}: ok")
        except Exception as exc:  # noqa: BLE001 -- an unsupported backend is a result, not a crash
            record.setdefault("backends", {})[key] = {"error": repr(exc)[:200]}
            log(f"  backend {key}: unavailable ({exc!r:.80})")

    (out_dir / "mechanism.json").write_text(json.dumps(record, indent=2) + "\n")
    log(f"wrote {out_dir / 'mechanism.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
