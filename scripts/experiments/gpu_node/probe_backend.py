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

#: Embedders to probe. The processor class is **not** hardcoded: the shipped arm
#: is whatever the app's own loader resolves, and the comparison arm is that
#: class's `...Pil` counterpart. Guessing the class gets it wrong -- `siglip2_l`
#: loads through `AutoProcessor`, and for the so400m-patch14-384 checkpoint that
#: resolves to SigLIP *1*'s `SiglipImageProcessor` (a 3x384x384 tensor), not to
#: `Siglip2ImageProcessor` (a 256x768 patch sequence). Comparing the class the
#: app does not use measures a path no user takes.
MODELS = ("siglip", "siglip2_l")


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

    from vtscore.embedding import initialize_models
    from vtscore.media import get_embedder

    initialize_models()
    for name in MODELS:
        emb = get_embedder(name)
        emb.load_models()
        shipped_proc = emb._processor  # noqa: SLF001 -- a probe, deliberately reading what the app resolved
        image_proc = getattr(shipped_proc, "image_processor", shipped_proc)
        shipped_cls = type(image_proc).__name__
        pil_cls = f"{shipped_cls}Pil"
        model_id = emb.model_id
        if not hasattr(transformers, pil_cls):
            log(f"  {name:<10} no {pil_cls} in transformers {transformers.__version__}; skipped")
            continue

        tensors = {"shipped": image_proc(images=pil, return_tensors="pt")["pixel_values"].double().numpy()}
        pil_proc = getattr(transformers, pil_cls).from_pretrained(model_id)
        tensors["pil"] = pil_proc(images=pil, return_tensors="pt")["pixel_values"].double().numpy()
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
