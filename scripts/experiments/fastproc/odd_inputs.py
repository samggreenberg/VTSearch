"""Does the torchvision path handle everything the PIL path did? (issue #3146, req. 3)

    python odd_inputs.py --out <dir>

The issue asks for this as a checklist — "confirm the fast path handles CMYK,
palette, EXIF-rotated, grayscale, since ``decode_bounded_rgb`` hands it whatever
the corpus contains".  Reading the decoder changes the question:
``decode_bounded_rgb`` ends in an unconditional ``img.convert("RGB")``
(``vtscore/media/image/decode.py``), and both bulk forward paths call
``im.convert("RGB")`` again, so **on the corpus path the processor never sees a
CMYK or palette image at all**.

That does not make the check pointless, it makes it a check of two different
things, and only one of them was the worry:

* **The corpus path** — the mode conversion is upstream of the treatment, so
  every backend receives the identical RGB bitmap.  The check here is that this
  is *true*, not assumed: if any backend disagreed on an image that reached it
  as RGB, the conversion would not be doing the job the decoder claims.
* **The direct path** — ``embed_pil_image`` and the extractor hand the processor
  PIL images from PDF rendering and demo datasets.  Those convert too, but the
  processors also carry their own ``do_convert_rgb``, so a backend that handled
  an odd mode *differently* rather than *identically* would surface here.

EXIF is the one the issue named that is genuinely not handled: nothing in
``decode.py`` calls ``exif_transpose``, so a rotated JPEG reaches the model
un-rotated. That is a real (pre-existing) defect, but it is **constant across
backends**, so it is reported here and excluded from the arm comparison rather
than being allowed to look like a treatment effect.

Every mode is checked for three things: that it does not raise, that the output
geometry matches the RGB baseline, and how far the pixels land from it.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import fastproc_config as fcfg  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


def odd_images() -> dict[str, object]:
    """One image per awkward input class, all derived from the same RGB source.

    Same source means a mode difference is the *only* difference, so a drift
    number here is attributable to mode handling and nothing else.
    """
    from PIL import Image

    rng = np.random.default_rng(3146)
    base_arr = (rng.random((419, 637, 3)) * 255).astype(np.uint8)
    # A hard edge and a smooth ramp: resampling differences live on edges, and a
    # pure-noise image would hide them in noise.
    base_arr[100:300, 150:400] = 240
    base_arr[:, :, 0] = np.linspace(0, 255, 637, dtype=np.uint8)[None, :]
    base = Image.fromarray(base_arr, "RGB")

    out: dict[str, object] = {"RGB (baseline)": base}
    out["L (grayscale)"] = base.convert("L")
    out["LA (grayscale+alpha)"] = base.convert("LA")
    out["P (palette)"] = base.convert("P", palette=Image.Palette.ADAPTIVE, colors=64)
    out["PA (palette+alpha)"] = base.convert("P", palette=Image.Palette.ADAPTIVE, colors=64).convert("PA")
    out["CMYK"] = base.convert("CMYK")
    out["RGBA"] = base.convert("RGBA")
    out["I;16 (16-bit)"] = base.convert("I;16")
    out["1 (bilevel)"] = base.convert("1")

    # EXIF orientation 6 (rotate 90 CW).  Written and re-read as a real JPEG so
    # the tag is genuinely in the file rather than attached to an in-memory object.
    buf = io.BytesIO()
    exif = Image.Exif()
    exif[0x0112] = 6
    base.save(buf, format="JPEG", exif=exif, quality=95)
    buf.seek(0)
    out["EXIF orientation=6"] = Image.open(buf)

    # Degenerate sizes: a 1-pixel image and a very tall sliver both make the
    # aspect-preserving resize divide by something small.
    out["1x1"] = Image.new("RGB", (1, 1), (128, 64, 32))
    out["3x900 (sliver)"] = base.resize((3, 900))
    return out


def build_processor(model_id: str, backend: str, cache_dir: str):
    from transformers import AutoImageProcessor

    from vtscore.media.embedder import hf_token

    kw: dict[str, object] = {} if backend == "auto" else {"backend": backend}
    proc = AutoImageProcessor.from_pretrained(model_id, cache_dir=cache_dir, token=hf_token(), **kw)
    return getattr(proc, "image_processor", proc)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embedders", default=",".join(fcfg.EMBEDDERS))
    ap.add_argument("--out", default=str(fcfg.results_dir()))
    args = ap.parse_args(argv)

    import torch

    from vtscore.config import MODELS_CACHE_DIR
    from vtscore.media import get_embedder

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = str(MODELS_CACHE_DIR)
    have_cuda = torch.cuda.is_available()

    images = odd_images()
    rows: list[dict] = []

    for emb_name in [e for e in args.embedders.split(",") if e]:
        model_id = get_embedder(emb_name).model_id
        log(f"\n=== {emb_name} ({model_id}) ===")
        variants = [("torchvision", "cpu"), ("pil", "cpu")]
        if have_cuda:
            variants.append(("torchvision", "cuda"))

        # The reference for every mode is the *same backend on plain RGB*, so a
        # row answers "did this mode change the answer" rather than "is this
        # backend different", which is the other script's job.
        for backend, device in variants:
            proc = build_processor(model_id, backend, cache_dir)
            call_kw: dict[str, object] = {} if device == "cpu" else {"device": device}
            cls = type(proc).__name__

            baseline = None
            for mode, img in images.items():
                row = {
                    "embedder": emb_name,
                    "backend": backend,
                    "device": device,
                    "processor_class": cls,
                    "mode": mode,
                    "pil_mode": getattr(img, "mode", "?"),
                    "size": list(getattr(img, "size", ())),
                }
                # The corpus path converts before the processor sees it; the
                # direct path does not.  Both are run because they are different
                # code paths with different exposure.
                for path_name, prepared in (
                    ("as-decoded (corpus path)", img.convert("RGB")),
                    ("raw mode (direct path)", img),
                ):
                    key = "corpus" if path_name.startswith("as-decoded") else "direct"
                    try:
                        pv = proc(images=[prepared], return_tensors="pt", **call_kw)["pixel_values"]
                        arr = pv.float().cpu().numpy()
                        row[f"{key}_ok"] = True
                        row[f"{key}_shape"] = list(arr.shape)
                        if mode == "RGB (baseline)" and key == "corpus":
                            baseline = arr
                        if baseline is not None and arr.shape == baseline.shape:
                            row[f"{key}_max_abs_vs_rgb"] = float(np.abs(arr - baseline).max())
                    except Exception as e:  # noqa: BLE001
                        row[f"{key}_ok"] = False
                        row[f"{key}_error"] = f"{type(e).__name__}: {str(e)[:140]}"
                rows.append(row)

            fails = [r for r in rows if r["backend"] == backend and r["device"] == device and not r.get("direct_ok")]
            log(f"  {backend}/{device:5s} {cls:28s} {len(images)} modes, {len(fails)} direct-path failures")

    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "odd_inputs.csv", index=False)

    log("\n=== per-mode outcome (corpus path = converted to RGB first) ===")
    log(
        f"{'embedder':10s} {'mode':24s} "
        + " ".join(f"{b + '/' + d:>18s}" for b, d in sorted({(r["backend"], r["device"]) for r in rows}))
    )
    variants = sorted({(r["backend"], r["device"]) for r in rows})
    for emb_name in sorted(df["embedder"].unique()):
        for mode in images:
            cells = []
            for backend, device in variants:
                sel = df[
                    (df.embedder == emb_name) & (df.backend == backend) & (df.device == device) & (df["mode"] == mode)
                ]
                if sel.empty:
                    cells.append(f"{'-':>18s}")
                    continue
                r = sel.iloc[0]
                if not r.get("corpus_ok"):
                    cells.append(f"{'RAISED':>18s}")
                else:
                    d = r.get("corpus_max_abs_vs_rgb")
                    cells.append(
                        f"{('same' if (d is not None and d == 0) else (f'{d:.2e}' if d is not None else 'ok')):>18s}"
                    )
            log(f"{emb_name:10s} {mode:24s} " + " ".join(cells))

    log("\n=== direct path (raw PIL mode, no convert) — failures only ===")
    bad = df[~df["direct_ok"].astype(bool)]
    if bad.empty:
        log("  none: every backend accepted every mode without a pre-conversion")
    else:
        for _, r in bad.iterrows():
            log(f"  {r['embedder']:10s} {r['backend']}/{r['device']:5s} {r['mode']:24s} {r.get('direct_error')}")

    log(f"\nwrote {outdir}/odd_inputs.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
