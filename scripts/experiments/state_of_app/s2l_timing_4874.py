#!/usr/bin/env python
"""#4874 runtime: SigLIP, SigLIP 2 L and EmbeddingGemma 2, end to end through the shipped embedders, interleaved.

#4837's ``embeddinggemma2/timing.py`` with SigLIP 2 L added. On N coco_better images (paths from the SigLIP
pile cell, the JPEG bytes read once into memory so no arm pays the NFS read), each embedder runs
``embed_media_bulk`` (threaded decode, the processor, the forward), arms interleaved over REPS so drift on
the node hits all alike. SigLIP runs at batch 128 as #4837 timed it; the others at their own default.
Also:

* each pile cell's vectors against a re-embed under this venv (median and max 1 - cos), since the
  SigLIP 2 L cell was built under an older transformers than the one the sessions run;
* a text-query timing for each.

    source gridenv.sh && source scripts/experiments/pile/pile_env.sh
    VTS_REPO=<worktree> python s2l_timing_4874.py --out timing.jsonl [--embedders siglip,siglip2_l,embeddinggemma2]

Writes one JSON line per measurement to --out.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import zipfile

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--out", required=True)
ap.add_argument("--embedders", default="siglip,siglip2_l,embeddinggemma2")
ap.add_argument("--siglip-batch", type=int, default=128)
args = ap.parse_args()

import torch  # noqa: E402
import transformers  # noqa: E402

import vtscore.media as vm  # noqa: E402
from vtscore.config import EMBEDDINGS_DIR, embed_precision, resolve_device  # noqa: E402
from vtscore.embedding.media_vectors import media_embedding  # noqa: E402

sys.path.insert(0, os.path.join(os.environ["VTS_REPO"], "scripts/experiments/calibration"))
from _cells_io import load_medias  # noqa: E402

dev = resolve_device()
gpu = torch.cuda.get_device_name(0) if dev.startswith("cuda") else ""
host = socket.gethostname()
print(
    f"host {host} device {dev} {gpu} transformers {transformers.__version__} torch {torch.__version__} "
    f"precision {embed_precision()} threads {torch.get_num_threads()} cpus {len(os.sched_getaffinity(0))}",
    flush=True,
)

names = args.embedders.split(",")
piles = {name: load_medias(EMBEDDINGS_DIR / f"coco_better__{name}.pkl", repair=True) for name in names}
ids = sorted(piles["siglip"] if "siglip" in piles else piles[names[0]])
rng = np.random.default_rng(0)
pick = [ids[i] for i in rng.choice(len(ids), size=min(args.n, len(ids)), replace=False)]
_zips: dict = {}


def _bytes(origin_name: str) -> bytes:
    zpath, member = origin_name.split("::", 1)
    if zpath not in _zips:
        _zips[zpath] = zipfile.ZipFile(zpath)
    return _zips[zpath].read(member)


ref_pile = piles[names[0]]
medias = [{"media_bytes": _bytes(ref_pile[i]["origin_name"])} for i in pick]
print(f"{len(medias)} images", flush=True)
out = open(args.out, "a")


def record(**kw):
    kw.update(
        host=host,
        device=dev,
        gpu=gpu,
        transformers=transformers.__version__,
        torch=torch.__version__,
        n=len(medias),
        precision=embed_precision(),
        threads=torch.get_num_threads(),
    )
    out.write(json.dumps(kw) + "\n")
    out.flush()
    print(kw, flush=True)


def sync():
    if dev.startswith("cuda"):
        torch.cuda.synchronize()


def bulk(emb, items):
    """``embed_media_bulk`` at the batch #4837 timed SigLIP at, every other embedder at its own default."""
    if emb.name == "siglip":
        os.environ["VTSEARCH_EMBED_BATCH_SIZE"] = str(args.siglip_batch)
    try:
        return emb.embed_media_bulk(items)
    finally:
        os.environ.pop("VTSEARCH_EMBED_BATCH_SIZE", None)


embs = [vm.get_embedder(name) for name in names]
for emb in embs:
    t = time.time()
    emb.load_models()
    record(what="load", embedder=emb.name, seconds=round(time.time() - t, 2))
    bulk(emb, medias[:8])  # warm
sync()

# --- each pile cell against a re-embed under this venv
for emb in embs:
    v = np.stack(bulk(emb, medias))
    ref = np.stack([np.asarray(media_embedding(piles[emb.name][i]), dtype=np.float32) for i in pick])
    cos = np.sum(v * ref, 1) / np.linalg.norm(v, axis=1) / np.linalg.norm(ref, axis=1)
    record(what="vs_pile", embedder=emb.name, median_1mcos=float(np.median(1 - cos)), max_1mcos=float(np.max(1 - cos)))

# --- interleaved end-to-end timing
for rep in range(args.reps):
    for emb in embs:
        sync()
        t = time.time()
        bulk(emb, medias)
        sync()
        s = time.time() - t
        batch = args.siglip_batch if emb.name == "siglip" else emb.embed_batch_size
        record(
            what="bulk", embedder=emb.name, batch=batch, rep=rep, seconds=round(s, 2), per_s=round(len(medias) / s, 2)
        )

# --- text queries
qs = ["airplane", "dining table", "book", "bag or luggage", "banana", "traffic light", "person", "dog"] * 4
for emb in embs:
    emb.embed_text("warm up")
    sync()
    t = time.time()
    for q in qs:
        emb.embed_text(q)
    sync()
    s = time.time() - t
    record(what="text", embedder=emb.name, ms_per_query=round(1000 * s / len(qs), 1))
if dev.startswith("cuda"):
    record(what="peak_gpu_gb", value=round(torch.cuda.max_memory_allocated() / 1e9, 2))
print("DONE", flush=True)
