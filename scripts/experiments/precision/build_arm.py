"""Build one precision arm's cells, and record what the process actually did.

Run inside the arm's SLURM job::

    python build_arm.py --arm fp16_l40s

Two things happen, in this order, and the order matters:

1. **Probe.**  Resolve the precision, load each embedder, and record the loaded
   model's *parameter dtype* along with the card's name and compute capability.
   This is the difference between measuring an effect and measuring nothing: a
   mode that silently degraded (bf16 on an sm_70 card, a typo'd env var, a CPU
   fallback) produces vectors identical to fp32, which reads as "half precision
   is harmless" when it in fact never ran.  #2877 shipped a conclusion off
   exactly that shape of unasserted premise.
2. **Build.**  Delegate to the pile builder so the cells are produced by the
   same code path that produced the published pile — a bespoke embed loop here
   would make the fp32 arm a reproduction of nothing.

GPU nodes are ``Exclusive_Process``, so the probe and the build are sequential
in one process rather than two jobs racing for the card.
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

import precision_config as pcfg  # noqa: E402


def log(msg: str) -> None:
    print(f"[precision] {msg}", flush=True)


def _probe(embedders: list[str]) -> dict:
    """Resolve the precision and load each embedder, recording real dtypes."""
    import torch

    from vtscore.config import (
        EMBED_PRECISION,
        embed_autocast_dtype,
        embed_precision,
        embed_weight_dtype,
        resolve_device,
    )
    from vtscore.embedding import initialize_models
    from vtscore.media import get_embedder

    initialize_models()
    device = resolve_device()
    info: dict = {
        "requested": EMBED_PRECISION,
        "resolved": embed_precision(),
        "weight_dtype": str(embed_weight_dtype()),
        "autocast_dtype": str(embed_autocast_dtype()),
        "device": device,
        "torch": torch.__version__,
        "cuda": getattr(torch.version, "cuda", None),
        "embedders": {},
    }
    if device.startswith("cuda"):
        major, minor = torch.cuda.get_device_capability(0)
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_capability"] = f"sm_{major}{minor}"
        info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())

    for name in embedders:
        emb = get_embedder(name)
        emb.load_models()
        model = getattr(emb, "_model", None)
        if model is None:
            raise SystemExit(f"{name}: model did not load; cannot verify its dtype")
        param = next(model.parameters())
        info["embedders"][name] = {
            "param_dtype": str(param.dtype),
            "param_device": str(param.device),
        }
        log(f"  {name}: params {param.dtype} on {param.device}")
    return info


def _assert_probe_matches_arm(arm: str, info: dict) -> None:
    """Refuse to build cells under a precision the process did not actually reach.

    A silent degradation is worse than a crash here: the cells would be fp32
    wearing an fp16 label, and every later comparison against them would be
    quietly measuring nothing.
    """
    want = pcfg.ARMS[arm]["precision"]
    got = info["resolved"]
    if got != want:
        raise SystemExit(
            f"arm {arm} asked for precision {want!r} but the process resolved {got!r}.\n"
            f"  device={info.get('device')} gpu={info.get('gpu_name')} "
            f"capability={info.get('gpu_capability')} bf16={info.get('bf16_supported')}\n"
            f"  Nothing was built. Fix the arm/GPU pairing rather than accepting "
            f"cells that carry the wrong label."
        )
    # And the *behavioural* half: a weight-cast arm must have actually cast.
    expect_half = want in ("fp16", "bf16")
    for name, per in info["embedders"].items():
        is_half = per["param_dtype"] in ("torch.float16", "torch.bfloat16")
        if expect_half != is_half:
            raise SystemExit(
                f"arm {arm}: {name} loaded params as {per['param_dtype']}, "
                f"which contradicts precision {want!r}. Nothing was built."
            )


def _embed_category_text(dataset: str, embedders: list[str]) -> dict:
    """Embed every category name as a text query, per embedder.

    Written beside the cells so the CPU analysis can compute the quantity that
    actually matters — the *similarity* between a query and the gallery — rather
    than only the gallery drift.  ~100 short text forwards, so it is free next
    to the image pass, and it has to happen here because the model is loaded and
    the arm's precision is in force.

    ``dinov3``-style embedders have no text tower; they are skipped rather than
    faked.
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
    ap.add_argument("--arm", required=True, choices=sorted(pcfg.ARMS))
    ap.add_argument("--dataset", default=pcfg.DATASET)
    ap.add_argument("--embedders", default=",".join(pcfg.EMBEDDERS))
    ap.add_argument("--force", action="store_true", help="rebuild cells that already exist")
    args = ap.parse_args(argv)

    arm = args.arm
    embedders = [e for e in args.embedders.split(",") if e]
    precision = pcfg.ARMS[arm]["precision"]
    pile = pcfg.arm_pile(arm)

    # Set BOTH before importing anything under vtscore: the precision is read at
    # config import time, and the data dir decides where cells land.
    os.environ["VTSEARCH_EMBED_PRECISION"] = precision
    os.environ["VTS_PILE"] = str(pile)
    os.environ["VTSEARCH_DATA_DIR"] = str(pile / "datadir")
    # Weights come from the shared pile so no arm re-downloads them (and so a
    # download cannot land on /exp's 50G quota).
    os.environ["VTSEARCH_MODELS_DIR"] = str(pcfg.SHARED_MODELS)
    os.environ["HF_HOME"] = str(pcfg.SHARED_MODELS)

    log(f"arm {arm}: precision={precision} pile={pile}")
    log(f"  role: {pcfg.ARMS[arm]['role']}")

    # The demo source must be linked in before the loader looks: a *missing*
    # extraction dir reads as "not downloaded yet" and silently substitutes a
    # truncated re-download (that cost the pile a 1662-of-4193 cell once).
    datadir = pile / "datadir"
    datadir.mkdir(parents=True, exist_ok=True)
    link = datadir / "visual_genome"
    source = pcfg.SHARED_PILE / "datadir" / "visual_genome"
    if not link.exists():
        if not source.exists():
            raise SystemExit(f"missing demo source {source}; cannot build without it")
        link.symlink_to(source.resolve())
        log(f"  linked demo source {link} -> {source.resolve()}")

    import pile_config as pc

    pc.PILE = pile
    pc.DATADIR = datadir
    pc.EMBEDDINGS = datadir / "embeddings"
    pc.MODELS = pcfg.SHARED_MODELS
    pc.EMBEDDINGS.mkdir(parents=True, exist_ok=True)
    pc.setup_env()

    import build_pile

    build_pile.assert_vtscore_is_this_checkout()

    t0 = time.time()
    probe = _probe(embedders)
    log(
        f"probe: requested={probe['requested']} resolved={probe['resolved']} "
        f"gpu={probe.get('gpu_name')} ({probe.get('gpu_capability')})"
    )
    _assert_probe_matches_arm(arm, probe)

    # Time each cell here rather than trusting the builder's ``embed_seconds``.
    # That field reads 0 on this path and it is not a bug: the demo loader embeds
    # *inside* ``load_demo_source``, so by the time ``embed_missing`` runs there
    # is nothing left to do.  The wall time around the whole build_cell call is
    # therefore the only honest per-cell number — and it is the one that answers
    # the question that matters, because #3151 already overlapped decode with the
    # forward. The forward-only 4.2x is an upper bound on a stage that is no
    # longer the whole cost; what a user gets is this difference, end to end.
    summaries = []
    for emb in embedders:
        t_cell = time.time()
        summary = build_pile.build_cell(args.dataset, emb, force=args.force)
        summary["wall_seconds"] = round(time.time() - t_cell, 1)
        if summary.get("status") == "built" and summary.get("n_medias"):
            summary["medias_per_second"] = round(summary["n_medias"] / max(summary["wall_seconds"], 1e-6), 1)
        summaries.append(summary)
        log(f"  {emb}: cell wall {summary['wall_seconds']}s ({summary.get('medias_per_second', '?')} medias/s)")

    # Text queries, embedded in the SAME precision as the gallery.  Retrieval is
    # cosine(query, gallery), so a study that perturbed only the gallery would be
    # measuring half the change users would actually see — and the text tower is
    # the half nobody thinks to check.
    text_out = _embed_category_text(args.dataset, embedders)

    provenance = {
        "arm": arm,
        "precision": precision,
        "requested_gpu": pcfg.ARMS[arm]["gpu"],
        "dataset": args.dataset,
        "embedders": embedders,
        "probe": probe,
        "cells": summaries,
        "text_queries": text_out,
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "hostname": os.uname().nodename,
        "wall_seconds": round(time.time() - t0, 1),
    }
    pcfg.provenance_path(arm).write_text(json.dumps(provenance, indent=2) + "\n")
    log(f"wrote {pcfg.provenance_path(arm)}")
    built = [s for s in summaries if s["status"] == "built"]
    log(f"done: {len(built)} built, {len(summaries) - len(built)} skipped, {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
