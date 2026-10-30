"""GroundingDINO zero-shot text prompts over a pool, one score per image per prompt (#5395).

The #5213 question again with a second open-vocabulary detector: a user types "a bus" and GroundingDINO returns
the images it finds a bus in. The prompts, the shard format and ``prompts.json`` are ``sam3_score_5213.py``'s, so
``text_baseline.py --sam3 DIR --zs-name gdino_<tag>`` grades these scores on each cell's pool and split exactly as
it grades SAM3's.

An image's score for a prompt is its best detection: max over GroundingDINO's object queries of the sigmoid logit,
maxed over the prompt's own tokens (the phrase, not [CLS], [SEP] or the closing period; the original repo's
positive map). One prompt per forward pass, never a caption of every class: the user types one query, and in a
concatenated caption the classes compete for the same queries.

The Swin backbone runs once per image batch and is reused for every prompt (``model.model.backbone.forward`` is
replaced by a cache); the BERT text tower, the fusion encoder and the decoder run per prompt, which is the cost.

Run on a GPU node with HF_HOME on node scratch. Checkpoints (transformers ports):

- ``IDEA-Research/grounding-dino-tiny``: Swin-T on O365 + GoldG + Cap4M, no COCO (48.4 zero-shot COCO AP);
- ``openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det``: MM-GDINO-T, no COCO (50.4);
- ``openmmlab-community/mm_grounding_dino_large_o365v2_oiv6_goldg``: MM-GDINO-L, no COCO.

``IDEA-Research/grounding-dino-base`` and the ``*_all`` MM checkpoints trained on COCO, whose train2017 is this pool.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402

from sam3_score_5213 import _Images, prompt_table  # noqa: E402


def caption(prompt: str) -> str:
    """GroundingDINO's caption form: lower case, ending in a period."""
    p = prompt.strip().lower()
    return p if p.endswith(".") else p + "."


def phrase_tokens(tokenizer, prompt: str) -> tuple[dict, list[int]]:
    """The tokenized caption and the positions of the phrase's own tokens."""
    cap = caption(prompt)
    enc = tokenizer(cap, return_tensors="pt", return_offsets_mapping=True)
    n = len(cap) - 1  # the closing period
    pos = [k for k, (a, b) in enumerate(enc.pop("offset_mapping")[0].tolist()) if b > a and a < n]
    return enc, pos


def _collate(batch):
    """Pad a batch to its largest image, with GroundingDINO's pixel mask (1 = real pixel)."""
    import torch  # noqa: PLC0415

    ks = torch.tensor([k for k, _ in batch])
    h = max(px.shape[1] for _, px in batch)
    w = max(px.shape[2] for _, px in batch)
    out = torch.zeros(len(batch), 3, h, w)
    mask = torch.zeros(len(batch), h, w, dtype=torch.long)
    for i, (_, px) in enumerate(batch):
        out[i, :, : px.shape[1], : px.shape[2]] = px
        mask[i, : px.shape[1], : px.shape[2]] = 1
    return ks, out, mask


class _Pixels:
    """``processor(images=...)`` for a GroundingDINO image processor: one unpadded image."""

    def __init__(self, image_processor):
        self.ip = image_processor

    def __call__(self, images, return_tensors):
        return self.ip(images=images, return_tensors=return_tensors, do_pad=False)


def cache_backbone(model):
    """Run the vision backbone once per image batch: ``fill(pixels, mask)`` then every forward reuses it."""
    bb = model.model.backbone
    orig = bb.forward
    state = {}

    def fill(px, mask):
        state["out"] = orig(px, mask)

    def cached(pixel_values, pixel_mask):
        feats, pos = state["out"]
        # The caller appends the extra levels' position embeddings to the list it gets back: hand out a copy.
        return list(feats), list(pos)

    bb.forward = cached
    return fill


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="grid results dir (prepare_info.json names the cells)")
    ap.add_argument("--dataset", default="coco_better")
    ap.add_argument(
        "--pickle-embedder", default="siglip", help="any pickle of the dataset: only ids + origins are read"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="first N images of the shard only (a smoke / timing run)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dtype", default="fp32", choices=["bf16", "fp16", "fp32"])
    ap.add_argument(
        "--no-cache", action="store_true", help="run the backbone on every prompt (the check that the cache is exact)"
    )
    args = ap.parse_args()

    import torch  # noqa: PLC0415
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor  # noqa: PLC0415

    from _cells_io import load_medias  # noqa: PLC0415
    from vtscore.config import EMBEDDINGS_DIR  # noqa: PLC0415

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"shard_{args.shard}_of_{args.nshards}" + (f"_limit{args.limit}" if args.limit else "")
    dest = out / f"{tag}.npz"
    if dest.exists():
        common.log(f"{dest} exists, skipping")
        return 0

    prepare = json.loads((Path(args.results) / "prepare_info.json").read_text())
    info = next(iter(prepare["datasets"][args.dataset].values()))
    table = prompt_table(args.dataset, info["selected_categories"])
    prompts = sorted({p for v in table.values() for kind in ("typed", "noun") for p in v[kind]})
    (out / "prompts.json").write_text(json.dumps({"classes": table, "prompts": prompts}, indent=1))

    medias = load_medias(EMBEDDINGS_DIR / f"{args.dataset}__{args.pickle_embedder}.pkl")
    ids = sorted(medias.keys())[args.shard :: args.nshards]
    if args.limit:
        ids = ids[: args.limit]
    items = [(i, medias[i]["origin_name"]) for i in ids]
    del medias
    common.log(f"{tag}: {len(items)} images x {len(prompts)} prompts on {torch.cuda.get_device_name(0)}")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    dev = torch.device("cuda")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model, torch_dtype=dtype).to(dev).eval()
    fill = None if args.no_cache else cache_backbone(model)

    texts = {}
    for p in prompts:
        enc, pos = phrase_tokens(processor.tokenizer, p)
        texts[p] = ({k: v.to(dev) for k, v in enc.items()}, torch.tensor(pos, device=dev))

    top1 = np.zeros((len(items), len(prompts)), np.float16)
    top2 = np.zeros_like(top1)
    n35 = np.zeros((len(items), len(prompts)), np.uint8)

    loader = torch.utils.data.DataLoader(
        _Images(items, _Pixels(processor.image_processor)),
        batch_size=args.batch,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=_collate,
    )
    t0, t_vis, t_dec, done = time.time(), 0.0, 0.0, 0
    with torch.inference_mode():
        for ks, px, pmask in loader:
            b = px.shape[0]
            px, pmask = px.to(dev, dtype=dtype, non_blocking=True), pmask.to(dev, non_blocking=True)
            torch.cuda.synchronize()
            a = time.time()
            if fill is not None:
                fill(px, pmask)
            torch.cuda.synchronize()
            t_vis += time.time() - a
            a = time.time()
            k = ks.numpy()
            for j, p in enumerate(prompts):
                enc, pos = texts[p]
                o = model(pixel_values=px, pixel_mask=pmask, **{n: v.expand(b, -1) for n, v in enc.items()})
                s = o.logits.float()[:, :, pos].sigmoid().amax(-1)  # (b, queries)
                top = s.topk(2, dim=1).values.cpu().numpy()
                top1[k, j], top2[k, j] = top[:, 0], top[:, 1]
                n35[k, j] = np.minimum((s > 0.35).sum(1).cpu().numpy(), 255)
            torch.cuda.synchronize()
            t_dec += time.time() - a
            done += b
            if done % (args.batch * 100) == 0 or done == len(items):
                el = time.time() - t0
                common.log(
                    f"{done}/{len(items)}  {done / el:.2f} img/s  vision {t_vis / done * 1e3:.0f} ms/img  "
                    f"decode {t_dec / done / len(prompts) * 1e3:.1f} ms/img/prompt"
                )

    el = time.time() - t0
    np.savez(
        dest,
        ids=np.asarray(ids, dtype=object),
        prompts=np.asarray(prompts),
        top1=top1,
        top2=top2,
        n35=n35,
        timing=np.asarray([el, t_vis, t_dec, len(items), len(prompts)]),
        meta=np.asarray(
            [
                json.dumps(
                    {
                        "model": args.model,
                        "dtype": args.dtype,
                        "gpu": torch.cuda.get_device_name(0),
                        "batch": args.batch,
                        "cache": not args.no_cache,
                        "host": os.uname().nodename,
                    }
                )
            ]
        ),
    )
    common.log(f"wrote {dest}: {len(items)} imgs in {el:.0f} s (vision {t_vis:.0f} s, decode {t_dec:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
