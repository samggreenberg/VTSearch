"""Build one image-processor arm's cells, asserting what the process actually did.

Run inside the arm's SLURM job::

    python build_arm.py --arm tv_cuda

The probe here is not ceremony — it is the whole reason this study exists.
Issue #3146 was filed on a reading of the source ("no ``use_fast`` argument,
therefore the slow PIL path"), and the source says exactly that while the
installed transformers does the opposite.  A class name is not a measurement:
transformers v5 renamed the fast implementation to the slow one's name, so the
same identifier means different code on either side of a dependency range we
pin only a lower bound on.

So every arm records the **class it actually loaded**, the device the pixel
tensor actually came back on, and the transformers version that decided both —
and refuses to build a single cell when any of the three contradicts the arm
table.  A `pil` arm that silently got torchvision would produce cells identical
to the reference, which reads as "the backend does not matter" when in fact it
never changed.  That is the #2877 / #2897 / #2905 failure exactly, and it is
especially easy to hit here because transformers *warns and continues* when a
backend is unavailable rather than raising.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pile"))

import fastproc_config as fcfg  # noqa: E402


def log(msg: str) -> None:
    print(f"[fastproc] {msg}", flush=True)


def _image_processor(emb) -> object | None:
    """The image half of a processor, whether or not it is wrapped.

    ``SiglipProcessor`` wraps an image processor and a tokenizer; ``AutoImage-
    Processor`` returns the image processor directly.  Both shapes are in the
    tree, so the check has to handle both or it silently passes on one of them.
    """
    proc = getattr(emb, "_processor", None)
    if proc is None:
        return None
    return getattr(proc, "image_processor", proc)


def _probe(embedders: list[str]) -> dict:
    """Load each embedder and record the processor class it really resolved."""
    import torch
    import transformers
    from PIL import Image

    from vtscore.config import (
        IMAGE_PROCESSOR_BACKEND,
        IMAGE_PROCESSOR_DEVICE,
        image_processor_call_kwargs,
        image_processor_load_kwargs,
        resolve_device,
    )
    from vtscore.embedding import initialize_models
    from vtscore.media import get_embedder

    initialize_models()
    device = resolve_device()
    info: dict = {
        "requested_backend": IMAGE_PROCESSOR_BACKEND,
        "requested_device": IMAGE_PROCESSOR_DEVICE,
        "load_kwargs": {k: str(v) for k, v in image_processor_load_kwargs().items()},
        "call_kwargs": {k: str(v) for k, v in image_processor_call_kwargs().items()},
        "device": device,
        "torch": torch.__version__,
        "torchvision": None,
        "transformers": transformers.__version__,
        "pillow": getattr(Image, "__version__", None),
        "embedders": {},
    }
    try:
        import torchvision

        info["torchvision"] = torchvision.__version__
    except ImportError:
        pass
    if device.startswith("cuda"):
        major, minor = torch.cuda.get_device_capability(0)
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_capability"] = f"sm_{major}{minor}"
        # Pinning --nodelist pins the node, not the GPU within it, and this
        # node has 8 L40S.  `gres/gpu:v100` turning out to be two different
        # parts (#3160) is the same assumption one level up, so the part is
        # recorded rather than trusted: a straggler then shows up as a labelled
        # group instead of as inflated variance in the floor arm.
        props = torch.cuda.get_device_properties(0)
        info["gpu_multi_processor_count"] = int(props.multi_processor_count)
        info["gpu_total_memory"] = int(props.total_memory)
        info["gpu_uuid"] = str(getattr(props, "uuid", "") or "")

    # A 3-channel probe image at a size nothing divides evenly, so the resize is
    # a real resample rather than a crop or a no-op.
    probe_img = Image.new("RGB", (637, 419), (11, 200, 90))

    for name in embedders:
        emb = get_embedder(name)
        emb.load_models()
        proc = _image_processor(emb)
        if proc is None:
            raise SystemExit(f"{name}: processor did not load; cannot verify its backend")
        out = proc(images=[probe_img], return_tensors="pt", **image_processor_call_kwargs())
        pv = out["pixel_values"]
        info["embedders"][name] = {
            "processor_class": type(proc).__name__,
            "wrapper_class": type(getattr(emb, "_processor")).__name__,
            "pixel_device": str(pv.device),
            "pixel_dtype": str(pv.dtype),
            "pixel_shape": list(pv.shape),
        }
        log(f"  {name}: {type(proc).__name__} -> pixels {tuple(pv.shape)} on {pv.device}")
    return info


#: Class-name suffix that marks the PIL implementation in transformers v5.  In
#: v4 the polarity was the other way round (``…ImageProcessorFast`` was the
#: torchvision one and the bare name was PIL), which is the whole confusion.
_PIL_SUFFIX = "Pil"
_FAST_SUFFIX = "Fast"


def _backend_of(class_name: str, transformers_major: int) -> str:
    if class_name.endswith(_PIL_SUFFIX):
        return "pil"
    if class_name.endswith(_FAST_SUFFIX):
        return "torchvision"
    # A bare name means torchvision on v5 and PIL on v4 — the rename is exactly
    # the thing that has to be version-aware, so it is resolved here and nowhere
    # else.
    return "torchvision" if transformers_major >= 5 else "pil"


def _assert_probe_matches_arm(arm: str, info: dict) -> None:
    """Refuse to build cells under a backend/device the process did not reach.

    transformers *warns* and falls back when a backend is unavailable, so the
    default outcome of asking for something impossible is a silently mislabelled
    arm rather than an error.  This turns that warning into a refusal.
    """
    want_backend = fcfg.ARMS[arm]["backend"]
    want_device = fcfg.ARMS[arm]["device"]
    try:
        major = int(str(info["transformers"]).split(".")[0])
    except ValueError:
        major = 5

    problems = []
    for name, per in info["embedders"].items():
        got_backend = _backend_of(per["processor_class"], major)
        if got_backend != want_backend:
            problems.append(
                f"{name}: asked for backend {want_backend!r} but loaded "
                f"{per['processor_class']} (= {got_backend!r}). transformers warns and "
                f"falls back rather than raising, so this arm would be the reference "
                f"arm under a different name."
            )
        got_device = per["pixel_device"].split(":")[0]
        if got_device != want_device:
            problems.append(
                f"{name}: asked for device {want_device!r} but pixel_values came back on {per['pixel_device']!r}."
            )
    if problems:
        raise SystemExit(
            f"arm {arm}: probe contradicts the arm table. NOTHING WAS BUILT.\n  "
            + "\n  ".join(problems)
            + f"\n  transformers={info['transformers']} torchvision={info['torchvision']} "
            f"load_kwargs={info['load_kwargs']} call_kwargs={info['call_kwargs']}"
        )


def _embed_category_text(dataset: str, embedders: list[str]) -> dict:
    """Embed every category name as a text query, per arm.

    The image processor cannot touch the text tower, so these vectors *should*
    be identical across arms — which is the point.  Retrieval score is
    cos(query, gallery); writing the query per arm lets the analysis confirm the
    query half held still, so any ranking change is attributable to the gallery
    rather than assumed to be.
    """
    import numpy as np

    from vtscore.media import get_embedder

    sys.path.insert(0, str(HERE.parent / "calibration"))
    from _cells_io import load_medias  # noqa: PLC0415

    import pile_config as pc  # noqa: PLC0415

    out: dict[str, dict] = {}
    for name in embedders:
        emb = get_embedder(name)
        if not getattr(emb, "supports_text", False):
            log(f"  {name}: no text tower; skipping text queries")
            continue
        medias = load_medias(pc.cell_path(dataset, name))
        cats = sorted({c for m in medias.values() for c in (m.get("categories") or [m.get("category")]) if c})
        vecs, kept = [], []
        for cat in cats:
            vec = emb.embed_text(cat)
            if vec is None:
                continue
            kept.append(cat)
            vecs.append(np.asarray(vec, dtype=np.float32))
        if not vecs:
            log(f"  {name}: no text vectors produced")
            continue
        path = pc.EMBEDDINGS / f"{dataset}__{name}__textq.npz"
        np.savez_compressed(path, categories=np.array(kept), vectors=np.vstack(vecs))
        log(f"  {name}: wrote {len(kept)} text-query vectors -> {path.name}")
        out[name] = {"n_categories": len(kept), "path": path.name}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=sorted(fcfg.ARMS))
    ap.add_argument("--dataset", default=fcfg.DATASET)
    ap.add_argument("--embedders", default=",".join(fcfg.EMBEDDERS))
    ap.add_argument("--force", action="store_true", help="rebuild cells that already exist")
    args = ap.parse_args(argv)

    arm = args.arm
    embedders = [e for e in args.embedders.split(",") if e]
    backend = fcfg.ARMS[arm]["backend"]
    device = fcfg.ARMS[arm]["device"]
    pile = fcfg.arm_pile(arm)

    # Set every knob before importing anything under vtscore: the backend is read
    # at config import time and the data dir decides where cells land.
    os.environ["VTSEARCH_IMAGE_PROCESSOR_BACKEND"] = backend
    os.environ["VTSEARCH_IMAGE_PROCESSOR_DEVICE"] = device
    os.environ["VTS_PILE"] = str(pile)
    os.environ["VTSEARCH_DATA_DIR"] = str(pile / "datadir")
    os.environ["VTSEARCH_MODELS_DIR"] = str(fcfg.SHARED_MODELS)
    os.environ["HF_HOME"] = str(fcfg.SHARED_MODELS)

    log(f"arm {arm}: backend={backend} device={device} pile={pile}")
    log(f"  role: {fcfg.ARMS[arm]['role']}")

    # A *missing* extraction dir reads as "not downloaded yet" and silently
    # substitutes a truncated re-download (that cost the pile a 1662-of-4193
    # cell once), so the demo source is linked in before the loader looks.
    datadir = pile / "datadir"
    datadir.mkdir(parents=True, exist_ok=True)
    link = datadir / "visual_genome"
    source = fcfg.SHARED_PILE / "datadir" / "visual_genome"
    if not link.exists():
        if not source.exists():
            raise SystemExit(f"missing demo source {source}; cannot build without it")
        link.symlink_to(source.resolve())
        log(f"  linked demo source {link} -> {source.resolve()}")

    import pile_config as pc

    pc.PILE = pile
    pc.DATADIR = datadir
    pc.EMBEDDINGS = datadir / "embeddings"
    pc.MODELS = fcfg.SHARED_MODELS
    pc.EMBEDDINGS.mkdir(parents=True, exist_ok=True)
    pc.setup_env()

    import build_pile

    build_pile.assert_vtscore_is_this_checkout()

    t0 = time.time()
    probe = _probe(embedders)
    log(
        f"probe: transformers={probe['transformers']} torchvision={probe['torchvision']} "
        f"gpu={probe.get('gpu_name')} load_kwargs={probe['load_kwargs']} call_kwargs={probe['call_kwargs']}"
    )
    _assert_probe_matches_arm(arm, probe)

    # Wall time around the whole build, not the builder's ``embed_seconds`` —
    # that field reads 0 on the demo path because the loader embeds inside
    # ``load_demo_source``.  End-to-end is also the honest number for a *stage*
    # speedup: #3151 already overlapped decode with the forward, so a processor
    # that is 4x faster in isolation buys whatever is left after the overlap,
    # which is the thing a user actually waits for.
    summaries = []
    for emb in embedders:
        t_cell = time.time()
        summary = build_pile.build_cell(args.dataset, emb, force=args.force)
        summary["wall_seconds"] = round(time.time() - t_cell, 1)
        if summary.get("status") == "built" and summary.get("n_medias"):
            summary["medias_per_second"] = round(summary["n_medias"] / max(summary["wall_seconds"], 1e-6), 1)
        summaries.append(summary)
        log(f"  {emb}: cell wall {summary['wall_seconds']}s ({summary.get('medias_per_second', '?')} medias/s)")

    text_out = _embed_category_text(args.dataset, embedders)

    provenance = {
        "arm": arm,
        "backend": backend,
        "processor_device": device,
        "requested_node": fcfg.PIN_NODE,
        "dataset": args.dataset,
        "embedders": embedders,
        "probe": probe,
        "cells": summaries,
        "text_queries": text_out,
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "hostname": os.uname().nodename,
        "cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
        "torch_threads": os.environ.get("VTSEARCH_TORCH_THREADS"),
        "wall_seconds": round(time.time() - t0, 1),
    }
    fcfg.provenance_path(arm).write_text(json.dumps(provenance, indent=2) + "\n")
    log(f"wrote {fcfg.provenance_path(arm)}")
    built = [s for s in summaries if s["status"] == "built"]
    log(f"done: {len(built)} built, {len(summaries) - len(built)} skipped, {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
