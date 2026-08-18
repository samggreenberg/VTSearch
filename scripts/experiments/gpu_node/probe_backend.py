"""Does the transformers v5 processor flip change the pixels the app feeds its models?

    python probe_backend.py --out <dir> [--images 64]

`requirements/image-embedders.txt` pins `transformers>=4.49.0` with no ceiling,
and v5 moved the plain `SiglipImageProcessor` name onto the **torchvision**
implementation while the PIL one became `SiglipImageProcessorPil` (#3146). So the
range admits two different resamplers, and nothing records which one an install
resolved.

#3160 measured the neighbouring axis -- CPU kernel dispatch -- and found it
spares 224px entirely, which is why the shipped default embedder is unexposed
there. That result does **not** transfer here: PIL vs torchvision is a different
code path, not an ISA variation inside one kernel. This probe settles whether the
default embedder is exposed to the *backend* axis, which is what decides whether
the missing version ceiling is urgent or tidy-up.

Preprocessing is pure CPU, so this needs no GPU and no pile.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

VG_SOURCE = Path("/exp/scale26/datasets/external/vtsearch-demos/visual_genome")

#: (embedder name, HF id, the pair of processor classes to compare).
#: The app resolves the *plain* name, so that is the "shipped" arm.
MODELS = {
    "siglip": ("google/siglip-base-patch16-224", "SiglipImageProcessor", "SiglipImageProcessorPil"),
    "siglip2_l": ("google/siglip2-so400m-patch14-384", "Siglip2ImageProcessor", "Siglip2ImageProcessorPil"),
}


def log(msg: str) -> None:
    print(msg, flush=True)


def images(n: int) -> list[Image.Image]:
    files = sorted((VG_SOURCE / "VG_100K").glob("*.jpg"), key=lambda p: int(p.stem))[:n]
    if not files:
        raise SystemExit(f"no images under {VG_SOURCE}")
    return [Image.open(p).convert("RGB") for p in files]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--images", type=int, default=64)
    args = ap.parse_args(argv)

    import torch
    import transformers

    pil = images(args.images)
    out: dict = {
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "cpu_capability": str(torch.backends.cpu.get_cpu_capability()),
        "n_images": len(pil),
        "models": {},
    }
    log(f"transformers {out['transformers']}, dispatch {out['cpu_capability']}, {len(pil)} images")

    for name, (model_id, shipped_cls, pil_cls) in MODELS.items():
        tensors = {}
        for label, cls_name in (("shipped", shipped_cls), ("pil", pil_cls)):
            cls = getattr(transformers, cls_name)
            proc = cls.from_pretrained(model_id)
            tensors[label] = proc(images=pil, return_tensors="pt")["pixel_values"].double().numpy()
        a, b = tensors["shipped"], tensors["pil"]
        d = np.abs(a - b)
        ne = d > 0
        rec = {
            "model_id": model_id,
            "shipped_class": shipped_cls,
            "pil_class": pil_cls,
            "shape": list(a.shape[1:]),
            "fraction_differing": float(ne.mean()),
            "max_abs": float(d.max()),
            "mean_abs": float(d.mean()),
            "levels_8bit": float(d.max() * 255 / 2),
        }
        out["models"][name] = rec
        log(
            f"  {name:<10} {tuple(a.shape[1:])}  differing {rec['fraction_differing']:.2%}  "
            f"max {rec['max_abs']:.3e} ({rec['levels_8bit']:.2f} 8-bit levels)"
        )

    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "backend.json").write_text(json.dumps(out, indent=2) + "\n")
    log(f"wrote {dest / 'backend.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
