"""What the *model sees*: pixel-level perturbation and cost, per processor arm.

    python pixel_drift.py --n 384 --out <dir>

This runs before any embedding drift, and it answers a question the embedding
drift cannot: **how big is the perturbation at the input**.  Every downstream
number — cosine drift, rank flips, benchmark cost — is a function of this one,
and unlike them it needs no model, so it is measurable exactly and cheaply.

It matters here because #3146 asserts a size ordering it never measured: "this
is a larger perturbation than fp16, not a smaller one."  fp16's was measured
(#3143): 2.9e-6 on the *vectors*.  The comparable quantity for a resize change
is the pixel tensor, and the two are only comparable if both are quoted
relative to their own scale — so the drift is reported both in absolute units
and as a fraction of the reference tensor's actual range.

Three things are reported per arm, and the third is the one that keeps the
first two honest:

1. **Perturbation** — per-image max/mean |Δpixel| against the reference arm,
   as a distribution rather than a single number.  A resize difference is
   concentrated at edges, so a mean over 150k pixels understates it by orders
   of magnitude and a max over the whole corpus overstates it by reporting one
   image.
2. **Cost** — wall time for the processor call alone, at the batch size the
   bulk path actually uses.  Reported per image so it composes with cell size.
3. **Shape agreement** — that every arm produced the same tensor shape and
   dtype.  A backend that quietly resized to a different geometry would show up
   as enormous drift and be read as a numeric finding rather than a bug.

The images are real corpus images decoded through ``decode_bounded_rgb``, not
synthetic ones: the whole treatment is a *resample*, and a resample's error
depends on the scale factor, which on synthetic fixed-size input would be
constant and on this corpus is different for every image.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import fastproc_config as fcfg  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


def sig2(x: float) -> str:
    """Two significant digits — the report standard (docs/experiments/overview-bench)."""
    if x == 0 or not np.isfinite(x):
        return f"{x:.0f}"
    if abs(x) >= 0.01:
        return f"{x:.2g}"
    return f"{x:.1e}"


def corpus_images(n: int) -> tuple[list[Path], list]:
    """*n* real corpus images, decoded exactly as the embedder decodes them."""
    from vtscore.media.image.decode import decode_bounded_rgb

    roots = [
        fcfg.SHARED_PILE / "datadir" / "visual_genome" / "VG_100K",
        fcfg.SHARED_PILE / "datadir" / "visual_genome" / "VG_100K_2",
    ]
    paths: list[Path] = []
    for root in roots:
        if root.is_dir():
            paths.extend(sorted(root.glob("*.jpg")))
        if len(paths) >= n:
            break
    paths = paths[:n]
    if not paths:
        raise SystemExit(f"no images found under {roots}")
    return paths, [decode_bounded_rgb(p)[0] for p in paths]


def build_processor(model_id: str, backend: str, cache_dir: str):
    from transformers import AutoImageProcessor

    from vtscore.media.embedder import hf_token

    kw: dict[str, object] = {}
    if backend != "auto":
        kw["backend"] = backend
    proc = AutoImageProcessor.from_pretrained(model_id, cache_dir=cache_dir, token=hf_token(), **kw)
    return getattr(proc, "image_processor", proc)


def backend_of(class_name: str) -> str:
    """The backend a loaded class actually is — see build_arm._backend_of."""
    if class_name.endswith("Pil"):
        return "pil"
    if class_name.endswith("Fast"):
        return "torchvision"
    return "torchvision"


def run_arm(proc, images, batch: int, device: str, reps: int):
    """``(pixel_values float32 cpu, seconds)`` for the whole corpus, batched."""
    import torch

    call_kw: dict[str, object] = {} if device in ("auto", "cpu") else {"device": device}
    batches = [images[i : i + batch] for i in range(0, len(images), batch)]

    # Warm up: the first torchvision call pays a one-off import/compile cost that
    # would otherwise be charged to whichever arm happened to run first.
    proc(images=batches[0][: min(8, len(batches[0]))], return_tensors="pt", **call_kw)

    best = float("inf")
    out_chunks = None
    for _ in range(reps):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        chunks = [proc(images=b, return_tensors="pt", **call_kw)["pixel_values"] for b in batches]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        if elapsed < best:
            best = elapsed
            out_chunks = chunks
    pv = torch.cat([c.float().cpu() for c in out_chunks], dim=0)
    return pv.numpy(), best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=384, help="corpus images (the issue measured on 384)")
    ap.add_argument("--batch", type=int, default=32, help="processor batch; the bulk path's default")
    ap.add_argument("--reps", type=int, default=3, help="timing repeats; the MINIMUM is reported")
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
    log(f"device: {'cuda ' + torch.cuda.get_device_name(0) if have_cuda else 'cpu only'}")
    log(f"node: {__import__('os').uname().nodename}")

    paths, images = corpus_images(args.n)
    sizes = [im.size for im in images]
    log(f"{len(images)} corpus images, sizes {min(sizes)}..{max(sizes)}, batch {args.batch}, reps {args.reps}")

    rows: list[dict] = []
    per_image: list[dict] = []

    for emb_name in [e for e in args.embedders.split(",") if e]:
        emb = get_embedder(emb_name)
        model_id = emb.model_id
        log(f"\n=== {emb_name} ({model_id}) ===")

        # (backend, device) exactly as the arm table names them, plus the
        # `auto` probe: what a host that passes nothing actually resolves to.
        variants = [("auto", "cpu"), ("torchvision", "cpu"), ("pil", "cpu")]
        if have_cuda:
            variants.append(("torchvision", "cuda"))

        ref_pv = None
        for backend, device in variants:
            label = f"{backend}/{device}"
            try:
                proc = build_processor(model_id, backend, cache_dir)
            except Exception as e:  # noqa: BLE001
                log(f"  {label:18s} LOAD FAILED {type(e).__name__}: {str(e)[:120]}")
                continue
            cls = type(proc).__name__
            got = backend_of(cls)
            # transformers WARNS and falls back when a backend is unavailable, so
            # an arm can silently be a different arm.  Say so in the row rather
            # than dropping it: "pil is not available for this model" is itself
            # a finding (it is why dinov3 has no pil arm).
            honoured = backend == "auto" or got == backend
            pv, secs = run_arm(proc, images, args.batch, device, args.reps)

            row = {
                "embedder": emb_name,
                "backend_requested": backend,
                "device": device,
                "processor_class": cls,
                "backend_resolved": got,
                "backend_honoured": honoured,
                "shape": list(pv.shape),
                "seconds": round(secs, 4),
                "ms_per_image": round(1000 * secs / len(images), 3),
                "pixel_min": float(pv.min()),
                "pixel_max": float(pv.max()),
            }
            if ref_pv is None and backend == "torchvision" and device == "cpu":
                ref_pv = pv
            rows.append(row)
            log(
                f"  {label:18s} {cls:28s} {row['ms_per_image']:7.2f} ms/img  "
                f"shape={tuple(pv.shape)} range=[{row['pixel_min']:.2f},{row['pixel_max']:.2f}]"
                + ("" if honoured else "  ** BACKEND NOT HONOURED **")
            )

        if ref_pv is None:
            log("  no reference (torchvision/cpu) arm; skipping drift for this embedder")
            continue

        # Drift against the shipped path, per image.
        pix_range = float(ref_pv.max() - ref_pv.min())
        for row in [r for r in rows if r["embedder"] == emb_name]:
            backend, device = row["backend_requested"], row["device"]
            if (backend, device) == ("torchvision", "cpu"):
                continue
            proc = build_processor(model_id, backend, cache_dir)
            pv, _ = run_arm(proc, images, args.batch, device, 1)
            if pv.shape != ref_pv.shape:
                row["shape_mismatch"] = True
                log(f"  {backend}/{device}: SHAPE MISMATCH {pv.shape} vs {ref_pv.shape} — drift not comparable")
                continue
            d = np.abs(pv - ref_pv)
            flat = d.reshape(d.shape[0], -1)
            per_img_max = flat.max(axis=1)
            per_img_mean = flat.mean(axis=1)
            row.update(
                {
                    "drift_max": float(per_img_max.max()),
                    "drift_max_median": float(np.median(per_img_max)),
                    "drift_mean": float(per_img_mean.mean()),
                    "drift_mean_p95": float(np.percentile(per_img_mean, 95)),
                    "pixel_range": pix_range,
                    "drift_max_frac_range": float(per_img_max.max() / pix_range),
                    "drift_mean_frac_range": float(per_img_mean.mean() / pix_range),
                    "frac_images_identical": float((per_img_max == 0).mean()),
                }
            )
            worst = np.argsort(-per_img_max)[:5]
            for i in worst:
                per_image.append(
                    {
                        "embedder": emb_name,
                        "arm": f"{backend}/{device}",
                        "file": paths[i].name,
                        "source_size": list(sizes[i]),
                        "max_abs": float(per_img_max[i]),
                        "mean_abs": float(per_img_mean[i]),
                        "frac_of_range": float(per_img_max[i] / pix_range),
                    }
                )

    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "pixel_drift.csv", index=False)
    (outdir / "pixel_drift_examples.json").write_text(json.dumps(per_image, indent=2) + "\n")

    log("\n=== cost: the processor call alone ===")
    log(f"{'embedder':12s} {'arm':20s} {'class':30s} {'ms/img':>8s} {'vs tv/cpu':>10s}")
    for emb_name, g in df.groupby("embedder"):
        base = g[(g.backend_requested == "torchvision") & (g.device == "cpu")]["ms_per_image"]
        b = float(base.iloc[0]) if len(base) else float("nan")
        for _, r in g.iterrows():
            log(
                f"{r['embedder']:12s} {r['backend_requested'] + '/' + r['device']:20s} "
                f"{r['processor_class']:30s} {r['ms_per_image']:8.2f} {b / r['ms_per_image']:9.2f}x"
            )

    log("\n=== perturbation vs the shipped path (torchvision/cpu) ===")
    log("Per-image |delta pixel|.  'max' is the worst pixel in the worst image; 'max median'")
    log("is the worst pixel in the median image — the honest headline, since a resize")
    log("difference concentrates at edges and a corpus max reports one picture.")
    log(f"{'embedder':12s} {'arm':20s} {'max':>10s} {'max median':>12s} {'mean':>10s} {'max/range':>10s}")
    for _, r in df.iterrows():
        if "drift_max" not in r or not np.isfinite(r.get("drift_max", np.nan)):
            continue
        log(
            f"{r['embedder']:12s} {r['backend_requested'] + '/' + r['device']:20s} "
            f"{sig2(r['drift_max']):>10s} {sig2(r['drift_max_median']):>12s} "
            f"{sig2(r['drift_mean']):>10s} {sig2(r['drift_max_frac_range']):>10s}"
        )

    log(f"\nwrote {outdir}/pixel_drift.csv and pixel_drift_examples.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
