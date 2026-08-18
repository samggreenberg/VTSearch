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
    """Hash for "did it change at all", projection for "by how much".

    The hash alone cannot answer the question this probe exists for -- a step
    versus a ramp -- and keeping whole hidden states would be 700 MB. Summing
    over every axis but the last leaves a per-channel vector: small enough to
    write down, big enough that a relative L2 difference between two nodes is a
    real magnitude rather than a yes/no.
    """
    t = tensor.detach().float().cpu().contiguous()
    proj = t.double().reshape(-1, t.shape[-1]).sum(0)
    return {
        "sha256": hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16],
        "sum": float(t.double().sum()),
        "absmax": float(t.abs().max()),
        "shape": list(t.shape),
        "proj": [round(v, 10) for v in proj.tolist()],
    }


def host_record() -> dict:
    """The *host* half of the provenance, which v1 of this probe did not take.

    v1 assumed the pixels entering the tower were a constant and only varied the
    device. They are not: `rack5n03` preprocesses the same JPEGs into a different
    tensor than `rack7n03` and the L40S, which agree exactly. So the CPU, the
    thread counts and the preprocessing stack are part of the measurement.
    """
    import platform

    import torch

    cpu = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass

    def _version(mod: str) -> str | None:
        try:
            return __import__(mod).__version__
        except Exception:  # noqa: BLE001 -- a missing optional dep is a fact, not a crash
            return None

    return {
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        # The AVX-512 hypothesis is testable only downwards: an AVX-512 host can
        # be told to dispatch as AVX2, and if that reproduces the AVX2 host's
        # pixels the instruction path is named.
        "aten_cpu_capability": os.environ.get("ATEN_CPU_CAPABILITY"),
        "vtsearch_torch_threads": os.environ.get("VTSEARCH_TORCH_THREADS"),
        "transformers": _version("transformers"),
        "torchvision": _version("torchvision"),
        "PIL": _version("PIL"),
    }


def _images(n: int) -> list[Path]:
    files = sorted((VG_SOURCE / "VG_100K").glob("*.jpg"), key=lambda p: int(p.stem))
    if not files:
        raise SystemExit(f"no images under {VG_SOURCE}")
    return files[:n]


def _preprocess(images: list[Path], emb):
    """The shipped preprocessing path, **before** anything is moved to the GPU.

    Returned on the CPU on purpose: the question v2 exists to answer is whether
    two nodes already disagree here, which a tensor stamped after the device move
    cannot distinguish from a disagreement introduced by the move.
    """
    from PIL import Image

    pil = [Image.open(p).convert("RGB") for p in images]
    return emb._processor(images=pil, return_tensors="pt")  # noqa: SLF001 -- a probe


def _to_device(inputs, emb):
    from vtscore.media.embedder import to_model_inputs

    return to_model_inputs(inputs, emb._model)  # noqa: SLF001


def _processor_classes(emb) -> dict:
    proc = getattr(emb, "_processor", None)
    image_proc = getattr(proc, "image_processor", None)
    return {
        "processor_class": type(proc).__name__ if proc is not None else None,
        "image_processor_class": type(image_proc).__name__ if image_proc is not None else None,
    }


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

    # `get_image_features` returns a model-output object on this transformers
    # version, not a bare tensor; v1 stamped it directly, threw inside the
    # per-backend loop, and lost every layer it had just captured.
    from vtscore.media.embedder import extract_tensor

    return {
        "block_path": path,
        "n_layers": len(layers),
        "layers": captured,
        "image_features": _stamp(extract_tensor(out)),
    }


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
    ap.add_argument(
        "--tag",
        default="",
        help="suffix on the output dir, so several runs on ONE node (different CPU dispatch, different embedder) do not overwrite each other",
    )
    ap.add_argument(
        "--pixels",
        default="",
        help="a pixels.npy written by another node: run the forward on ITS tensor as well as this node's own",
    )
    args = ap.parse_args(argv)

    import numpy as np
    import torch

    from vtscore.embedding import initialize_models
    from vtscore.media import get_embedder

    node = socket.gethostname()
    out_dir = Path(args.out) / f"{node}{args.tag}"
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
        "host": host_record(),
        "preprocessing": _processor_classes(emb),
        "gemm": gemm_shapes(),
    }
    log(f"{node}: {record['gpu_name']} ({record['multi_processor_count']} SMs) on {record['host']['cpu']}")

    images = _images(args.images)
    own = _preprocess(images, emb)
    px = own["pixel_values"]
    np.save(out_dir / "pixels.npy", px.numpy())
    record["input"] = {
        "images": [p.name for p in images],
        "own_cpu": {k: _stamp(v) for k, v in own.items() if hasattr(v, "detach")},
    }

    runs = [("own", own)]
    if args.pixels:
        # The decisive arm: the *reference node's* pixels, run through this
        # node's GPU. If the features then match the reference, every bit of the
        # divergence entered before the GPU did and the card is innocent.
        ref_px = torch.from_numpy(np.load(args.pixels))
        supplied = dict(own)
        supplied["pixel_values"] = ref_px
        record["input"]["reference_cpu"] = _stamp(ref_px)
        record["input"]["reference_path"] = args.pixels
        record["input"]["reference_matches_own"] = bool(torch.equal(ref_px, px))
        log(
            f"  reference pixels {'MATCH' if record['input']['reference_matches_own'] else 'DIFFER FROM'} this node's own"
        )
        runs.append(("reference_pixels", supplied))

    for label, inputs in runs:
        moved = _to_device(inputs, emb)
        for backend in (None, "math", "efficient", "flash", "cudnn"):
            key = backend or "default"
            try:
                record.setdefault(label, {})[key] = tower_layers(emb, moved, backend)
                log(f"  {label}/{key}: ok")
            except Exception as exc:  # noqa: BLE001 -- an unsupported backend is a result, not a crash
                record.setdefault(label, {})[key] = {"error": repr(exc)[:200]}
                log(f"  {label}/{key}: unavailable ({exc!r:.80})")

    (out_dir / "mechanism.json").write_text(json.dumps(record, indent=2) + "\n")
    log(f"wrote {out_dir / 'mechanism.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
