"""#3160: fingerprint the *device* a GPU job actually landed on.

    python probe_device.py --out <dir> [--images 256] [--embedders siglip,siglip2_l]

``gres/gpu:v100`` is a **type**, and #3143 measured that a type is not a device:
two nodes both answering to it disagree by 1.5e-04 on ``siglip2_l`` while three
other devices agree to ~1e-12.  A type label is therefore not enough provenance
to say whether two pile cells are comparable.

This script is what a census needs: run it once per node and it writes, for that
node,

* the device as **torch** reports it -- name, capability, SM count, memory,
  driver and runtime versions (the fields SLURM cannot tell you apart);
* the shipped image-embedding forward over a **fixed, sorted** image list, saved
  as raw ``float32`` vectors so any two nodes can be differenced exactly;
* three **bare-op fingerprints** at the shapes the SO400M/384 tower actually
  uses -- a GEMM, the patch-embed conv, and one scaled-dot-product attention
  call.  These cost milliseconds and they separate the two hypotheses left
  standing in #3143: if a plain GEMM already differs, the cause is tiling and
  accumulation order; if only attention differs, it is a capability-selected
  SDPA backend.

Nothing here writes to the shared pile.  The fingerprints are the measurement,
not an artifact anyone else consumes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: The demo source the pile's ``visual_genome_m`` cells are built from.  Fixed
#: rather than discovered so every node in the census embeds the same bytes in
#: the same order -- a census whose input differs per node measures nothing.
VG_SOURCE = Path("/exp/scale26/datasets/external/vtsearch-demos/visual_genome")


def log(msg: str) -> None:
    print(msg, flush=True)


def image_list(n: int) -> list[Path]:
    """The first ``n`` VG images in sorted (id) order, across both shards."""
    files: list[Path] = []
    for shard in ("VG_100K", "VG_100K_2"):
        d = VG_SOURCE / shard
        if d.is_dir():
            files.extend(sorted(d.glob("*.jpg"), key=lambda p: int(p.stem)))
    if not files:
        raise SystemExit(f"no images under {VG_SOURCE}")
    return files[:n]


def device_info() -> dict:
    import torch

    info: dict = {
        "hostname": socket.gethostname(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "slurm_gres": os.environ.get("SLURM_JOB_GRES") or os.environ.get("SBATCH_GRES"),
        "torch": torch.__version__,
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if not torch.cuda.is_available():
        info["error"] = "no CUDA device visible"
        return info
    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    info.update(
        {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_capability": f"sm_{major}{minor}",
            "multi_processor_count": props.multi_processor_count,
            "total_memory_gb": round(props.total_memory / 1e9, 1),
            "driver": _driver_version(),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        }
    )
    return info


def _driver_version() -> str | None:
    try:
        import pynvml  # noqa: PLC0415

        pynvml.nvmlInit()
        raw = pynvml.nvmlSystemGetDriverVersion()
        return raw.decode() if isinstance(raw, bytes) else str(raw)
    except Exception:  # noqa: BLE001 -- a missing driver query is not a reason to lose the census
        return None


def op_fingerprints() -> dict:
    """Bare-op outputs at SO400M/384 shapes, from inputs generated on the CPU.

    Generating on the CPU with a fixed seed matters: a CUDA RNG is itself
    device-dependent, so seeding on the GPU would make every node's *input*
    differ and the comparison would be meaningless.
    """
    import torch

    gen = torch.Generator(device="cpu").manual_seed(0)
    out: dict = {}

    def stamp(key: str, tensor) -> None:
        t = tensor.detach().float().cpu().contiguous()
        out[key] = {
            "sha256": hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16],
            "sum": float(t.double().sum()),
            "absmax": float(t.abs().max()),
        }

    # GEMM at the tower's MLP shape: (tokens=729, 1152) x (1152, 4304).
    a = torch.randn(729, 1152, generator=gen)
    b = torch.randn(1152, 4304, generator=gen)
    stamp("gemm_729x1152x4304", (a.cuda() @ b.cuda()))

    # Patch-embed conv: 3x384x384 -> 1152 channels, 14x14 stride 14.
    x = torch.randn(2, 3, 384, 384, generator=gen)
    w = torch.randn(1152, 3, 14, 14, generator=gen) * 0.02
    stamp("conv_patch_embed", torch.nn.functional.conv2d(x.cuda(), w.cuda(), stride=14))

    # One attention call at (batch 2, heads 16, tokens 729, head_dim 72).
    q = torch.randn(2, 16, 729, 72, generator=gen)
    k = torch.randn(2, 16, 729, 72, generator=gen)
    v = torch.randn(2, 16, 729, 72, generator=gen)
    stamp("sdpa_default", torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda()))
    return out


def embed(embedders: list[str], images: list[Path], out_dir: Path) -> dict:
    import numpy as np

    from vtscore.embedding import initialize_models
    from vtscore.media import get_embedder

    initialize_models()
    medias = [{"id": i, "media_type": "image", "media_path": str(p)} for i, p in enumerate(images)]
    summary: dict = {}
    for name in embedders:
        emb = get_embedder(name)
        emb.load_models()
        t0 = time.time()
        vecs = emb.embed_media_bulk(medias)
        wall = time.time() - t0
        missing = [i for i, v in enumerate(vecs) if v is None]
        if missing:
            raise SystemExit(f"{name}: {len(missing)} images failed to embed (first {missing[:3]})")
        arr = np.stack([np.asarray(v, dtype=np.float32) for v in vecs])
        path = out_dir / f"vectors_{name}.npy"
        np.save(path, arr)
        summary[name] = {
            "n": int(arr.shape[0]),
            "dim": int(arr.shape[1]),
            "sha256": hashlib.sha256(arr.tobytes()).hexdigest()[:16],
            "seconds": round(wall, 1),
            "param_dtype": str(next(emb._model.parameters()).dtype) if getattr(emb, "_model", None) else None,
        }
        log(f"  {name}: {arr.shape} in {wall:.0f}s -> {path.name}")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="census root; results land in <out>/<hostname>/")
    ap.add_argument("--images", type=int, default=256)
    ap.add_argument("--embedders", default="siglip,siglip2_l")
    args = ap.parse_args(argv)

    node = socket.gethostname()
    out_dir = Path(args.out) / node
    out_dir.mkdir(parents=True, exist_ok=True)

    info = device_info()
    log(
        f"{node}: {info.get('gpu_name')} ({info.get('gpu_capability')}, "
        f"{info.get('multi_processor_count')} SMs, driver {info.get('driver')})"
    )
    if "error" in info:
        (out_dir / "device.json").write_text(json.dumps(info, indent=2) + "\n")
        raise SystemExit(info["error"])

    record = {
        "device": info,
        "images": {"n": args.images, "source": str(VG_SOURCE)},
        "ops": op_fingerprints(),
    }
    # Ops first, then the embedders: if a model download or an OOM sinks the
    # heavy half, the cheap half is already on disk and the node still counts.
    (out_dir / "device.json").write_text(json.dumps(record, indent=2) + "\n")

    images = image_list(args.images)
    record["images"]["first"] = images[0].name
    record["images"]["last"] = images[-1].name
    record["embedders"] = embed([e for e in args.embedders.split(",") if e], images, out_dir)
    (out_dir / "device.json").write_text(json.dumps(record, indent=2) + "\n")
    log(f"wrote {out_dir / 'device.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
