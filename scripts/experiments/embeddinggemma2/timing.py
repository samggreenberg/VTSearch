"""#4837 runtime: SigLIP vs EmbeddingGemma 2, end to end through the shipped embedders, interleaved.

On N coco_better images (paths from the SigLIP pile cell), each arm runs ``embed_media_bulk`` (threaded decode,
the processor, the forward), arms interleaved over REPS so drift on the node hits both alike. Also:
  * EmbeddingGemma 2 at a smaller vision budget (140, 70 soft tokens) through the same bulk path;
  * SigLIP's re-embedded vectors against the pile cell's (transformers 5.19 here vs 5.12.1 at build);
  * a text-query timing for both.
Writes one JSON line per measurement to --out.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--out", required=True)
ap.add_argument("--budgets", default="280,140,70")
ap.add_argument("--siglip-batch", type=int, default=128)
args = ap.parse_args()

import torch  # noqa: E402
import transformers  # noqa: E402

import vtscore.media as vm  # noqa: E402
from vtscore.config import EMBEDDINGS_DIR, embed_precision, resolve_device  # noqa: E402

dev = resolve_device()
gpu = torch.cuda.get_device_name(0) if dev.startswith("cuda") else ""
host = socket.gethostname()
print(
    f"host {host} device {dev} {gpu} transformers {transformers.__version__} torch {torch.__version__} "
    f"precision {embed_precision()} threads {torch.get_num_threads()} cpus {len(os.sched_getaffinity(0))}",
    flush=True,
)

import sys  # noqa: E402

sys.path.insert(0, os.path.join(os.environ["VTS_REPO"], "scripts/experiments/calibration"))
from _cells_io import load_medias  # noqa: E402

pile = load_medias(EMBEDDINGS_DIR / "coco_better__siglip.pkl", repair=True)
ids = sorted(pile)
rng = np.random.default_rng(0)
pick = [ids[i] for i in rng.choice(len(ids), size=min(args.n, len(ids)), replace=False)]
# The pile keeps no pixels: each media names its image inside COCO's train zip ("<zip>::<member>"). Read the
# bytes once up front, so both arms decode the same in-memory JPEGs and neither pays the NFS read.
import zipfile  # noqa: E402

_zips: dict = {}


def _bytes(origin_name: str) -> bytes:
    zpath, member = origin_name.split("::", 1)
    if zpath not in _zips:
        _zips[zpath] = zipfile.ZipFile(zpath)
    return _zips[zpath].read(member)


medias = [{"media_bytes": _bytes(pile[i]["origin_name"])} for i in pick]
print(f"{len(medias)} images", flush=True)

sig = vm.get_embedder("siglip")
eg2 = vm.get_embedder("embeddinggemma2")
out = open(args.out, "a")


def record(**kw):
    kw.update(
        host=host,
        device=dev,
        gpu=gpu,
        transformers=transformers.__version__,
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


for emb in (sig, eg2):
    t = time.time()
    emb.load_models()
    record(what="load", embedder=emb.name, seconds=round(time.time() - t, 2))

os.environ["VTSEARCH_EMBED_BATCH_SIZE"] = str(args.siglip_batch)
sig.embed_media_bulk(medias[:16])  # warm
os.environ.pop("VTSEARCH_EMBED_BATCH_SIZE")
eg2.embed_media_bulk(medias[:8])
sync()

# --- SigLIP vs its pile cell
os.environ["VTSEARCH_EMBED_BATCH_SIZE"] = str(args.siglip_batch)
v = np.stack(sig.embed_media_bulk(medias))
os.environ.pop("VTSEARCH_EMBED_BATCH_SIZE")
from vtscore.embedding.media_vectors import media_embedding  # noqa: E402

ref = np.stack([np.asarray(media_embedding(pile[i]), dtype=np.float32) for i in pick])
cos = np.sum(v * ref, 1) / np.linalg.norm(v, axis=1) / np.linalg.norm(ref, axis=1)
record(what="siglip_vs_pile", median_1mcos=float(np.median(1 - cos)), max_1mcos=float(np.max(1 - cos)))

# --- interleaved end-to-end timing
from vtscore.media._embeddinggemma2 import media_messages, pooled_forward  # noqa: E402
from vtscore.media.image import _image_bulk  # noqa: E402

budgets = [int(b) for b in args.budgets.split(",") if b]


def eg2_budget(b):
    def fwd(images):
        rgb = [im.convert("RGB") for im in images]
        return pooled_forward(
            eg2._model, eg2._processor, media_messages("image", rgb), images_kwargs={"max_soft_tokens": b}
        )

    return fwd


for rep in range(args.reps):
    os.environ["VTSEARCH_EMBED_BATCH_SIZE"] = str(args.siglip_batch)
    sync()
    t = time.time()
    sig.embed_media_bulk(medias)
    sync()
    s = time.time() - t
    os.environ.pop("VTSEARCH_EMBED_BATCH_SIZE")
    record(
        what="bulk",
        embedder="siglip",
        batch=args.siglip_batch,
        rep=rep,
        seconds=round(s, 2),
        per_s=round(len(medias) / s, 2),
    )
    for b in budgets:
        sync()
        t = time.time()
        if b == 280:
            eg2.embed_media_bulk(medias)
        else:
            _image_bulk.bulk_embed_image_files(
                medias,
                forward_pil_batch=eg2_budget(b),
                batch_size=eg2.embed_batch_size,
                on_progress=lambda *a, **k: None,
                label="eg2",
            )
        sync()
        s = time.time() - t
        record(
            what="bulk",
            embedder="embeddinggemma2",
            budget=b,
            batch=eg2.embed_batch_size,
            rep=rep,
            seconds=round(s, 2),
            per_s=round(len(medias) / s, 2),
        )

# --- text queries
qs = ["airplane", "dining table", "book", "bag or luggage", "banana", "traffic light", "person", "dog"] * 4
for emb in (sig, eg2):
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
