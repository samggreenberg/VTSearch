"""#4390 (from #4838): check one venv end to end — imports, EmbeddingGemma 2, every coco_better pile cell, cuML.

Run from run_arm.sh (which sets PATH / PYTHONPATH / VTS_REPO for the venv under test). Writes one JSON line
per measurement to --out/<arm>.jsonl and the re-embedded vectors to --out/<arm>.npz, so two arms (old venv vs
new) can be compared vector by vector, not only each against the pile.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import traceback
import zipfile

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--embedders", default="siglip,clip,clip_l,siglip2_l,embeddinggemma2,dinov3_patch")
ap.add_argument("--cuml", action="store_true")
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
log = open(os.path.join(args.out, f"{args.arm}.jsonl"), "a")
vecs: dict[str, np.ndarray] = {}


def record(**kw):
    kw = {"arm": args.arm, "host": socket.gethostname(), **kw}
    log.write(json.dumps(kw) + "\n")
    log.flush()
    print(kw, flush=True)


def step(name):
    def deco(fn):
        t = time.time()
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - every failure is a result here
            record(what=name, ok=False, error=f"{type(exc).__name__}: {exc}", seconds=round(time.time() - t, 1))
            traceback.print_exc()
        return fn

    return deco


import torch  # noqa: E402
import transformers  # noqa: E402

record(
    what="versions",
    python=sys.version.split()[0],
    torch=torch.__version__,
    torch_cuda=torch.version.cuda,
    arch_list=torch.cuda.get_arch_list(),
    cuda_available=torch.cuda.is_available(),
    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    transformers=transformers.__version__,
    cudnn=torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    cvd=os.environ.get("CUDA_VISIBLE_DEVICES"),
    threads=torch.get_num_threads(),
)


@step("import_model_classes")
def _():
    acc = getattr(torch, "accelerator", None)
    cur = None
    if acc is not None:
        try:
            cur = str(acc.current_accelerator())
        except RuntimeError as exc:  # torch 2.6 with no GPU: the #4838 bug
            cur = f"RAISES {exc}"
    from transformers import AutoModel, SiglipModel  # noqa: F401  (imports masking_utils)

    from transformers import EmbeddingGemma2Model  # noqa: F401

    record(what="import_model_classes", ok=True, current_accelerator=cur)


if torch.cuda.is_available():

    @step("cuda_matmul")
    def _():
        a = torch.randn(2048, 2048, device="cuda")
        b = (a @ a.T).float().cpu()
        c = a.cpu() @ a.cpu().T
        record(what="cuda_matmul", ok=True, max_abs_vs_cpu=float((b - c).abs().max()), dtype=str(b.dtype))
        h = torch.randn(64, 768, device="cuda", dtype=torch.float16)
        _ = torch.nn.functional.scaled_dot_product_attention(h[None, None], h[None, None], h[None, None])
        conv = torch.nn.Conv2d(3, 16, 3).cuda()
        _ = conv(torch.randn(2, 3, 64, 64, device="cuda"))
        record(what="cuda_sdpa_conv", ok=True)


import vtscore.media as vm  # noqa: E402
from vtscore.config import EMBEDDINGS_DIR, embed_precision, resolve_device  # noqa: E402
from vtscore.embedding.media_vectors import media_embedding  # noqa: E402

sys.path.insert(0, os.path.join(os.environ["VTS_REPO"], "scripts/experiments/calibration"))
from _cells_io import load_medias  # noqa: E402

record(what="device", device=resolve_device(), precision=embed_precision())

QUERIES = ["dog", "a red car", "people playing frisbee in a park", "a bowl of fruit on a table", "stop sign"]


@step("eg2_text")
def _():
    t = time.time()
    v = vm.get_embedder("embeddinggemma2").embed_text("dog")
    record(
        what="eg2_text",
        ok=v is not None,
        dim=None if v is None else int(v.shape[-1]),
        norm=None if v is None else float(np.linalg.norm(v)),
        seconds=round(time.time() - t, 1),
    )


@step("eg2_audio_text")
def _():
    v = vm.get_embedder("embeddinggemma2_audio").embed_text("dog barking")
    record(what="eg2_audio_text", ok=v is not None, dim=None if v is None else int(v.shape[-1]))


picked_ids = None
media_bytes = None
_zips: dict = {}


def _bytes(origin_name: str) -> bytes:
    zpath, member = origin_name.split("::", 1)
    if zpath not in _zips:
        _zips[zpath] = zipfile.ZipFile(zpath)
    return _zips[zpath].read(member)


for name in [e for e in args.embedders.split(",") if e]:

    @step(f"pile_{name}")
    def _(name=name):
        global picked_ids, media_bytes
        t0 = time.time()
        pile = load_medias(EMBEDDINGS_DIR / f"coco_better__{name}.pkl", repair=True)
        load_s = time.time() - t0
        if picked_ids is None:
            ids = sorted(pile)
            rng = np.random.default_rng(0)
            picked_ids = [ids[i] for i in rng.choice(len(ids), size=min(args.n, len(ids)), replace=False)]
            media_bytes = [_bytes(pile[i]["origin_name"]) for i in picked_ids]
        assert all(pile[i]["origin_name"] for i in picked_ids)
        ref = np.stack([np.asarray(media_embedding(pile[i], name), dtype=np.float32) for i in picked_ids])
        del pile
        emb = vm.get_embedder(name)
        t = time.time()
        emb.load_models()
        load_model_s = time.time() - t
        t = time.time()
        out = emb.embed_media_bulk([{"media_bytes": b} for b in media_bytes])
        embed_s = time.time() - t
        v = np.stack([np.asarray(x, dtype=np.float32) for x in out])
        cos = np.sum(v * ref, 1) / np.linalg.norm(v, axis=1) / np.linalg.norm(ref, axis=1)
        vecs[f"{name}__media"] = v
        overlap = None
        try:  # image-only embedders (dinov3_patch) have no text tower in this space
            q = np.stack([np.asarray(emb.embed_text(s), dtype=np.float32) for s in QUERIES])
        except Exception:  # noqa: BLE001
            q = None
        if q is not None and q.ndim == 2 and q.shape[1] == v.shape[1]:
            vecs[f"{name}__text"] = q
            # Ranking agreement: top-10 of each query over the sample, new vectors vs pile vectors.
            top_new = np.argsort(-(v @ q.T), axis=0)[:10].T
            top_ref = np.argsort(-(ref @ q.T), axis=0)[:10].T
            overlap = float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(top_new, top_ref)]))
        record(
            what="pile",
            embedder=name,
            ok=True,
            n=len(v),
            dim=int(v.shape[1]),
            median_1mcos=float(np.median(1 - cos)),
            p99_1mcos=float(np.quantile(1 - cos, 0.99)),
            max_1mcos=float(np.max(1 - cos)),
            top10_overlap=overlap,
            load_pile_s=round(load_s, 1),
            load_model_s=round(load_model_s, 1),
            embed_s=round(embed_s, 1),
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if args.cuml:

    @step("cuml")
    def _():
        import cuml  # noqa: PLC0415

        from vtscore import gpu_backends  # noqa: PLC0415

        rng = np.random.default_rng(0)
        x = rng.standard_normal((3000, 64)).astype(np.float32)
        x[:1500] += 3.0
        t = time.time()
        y = gpu_backends.umap_fit_transform(
            x, n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=0
        )
        umap_s = time.time() - t
        t = time.time()
        labels, _inertia = gpu_backends.kmeans_fit_predict(x, n_clusters=8, random_state=0, n_init=1)
        km_s = time.time() - t
        from cuml.manifold import UMAP  # noqa: PLC0415

        y2 = UMAP(n_components=2, random_state=0).fit_transform(x)
        from cuml.cluster import KMeans  # noqa: PLC0415

        l2 = KMeans(n_clusters=8, random_state=0).fit_predict(x)
        record(
            what="cuml",
            ok=not gpu_backends._cuml_runtime_failed,
            cuml=cuml.__version__,
            fell_back_to_cpu=gpu_backends._cuml_runtime_failed,
            cuml_enabled=gpu_backends.cuml_enabled(),
            umap_shape=list(np.asarray(y).shape),
            kmeans_k=int(len(set(np.asarray(labels).tolist()))),
            raw_umap_shape=list(np.asarray(y2).shape),
            raw_kmeans_k=int(len(set(np.asarray(l2).tolist()))),
            umap_s=round(umap_s, 1),
            kmeans_s=round(km_s, 1),
        )


np.savez(os.path.join(args.out, f"{args.arm}.npz"), ids=np.asarray(picked_ids if picked_ids else []), **vecs)
record(what="done")
