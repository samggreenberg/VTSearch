"""SAM3 zero-shot text prompts over a pool, one score per image per prompt (#5213).

The question: a user types "a bus" and SAM3 (Meta's open-vocabulary detector/segmenter) returns the images it
finds a bus in. How does that ranking compare with SigLIP's typed sort, and with Region voting after N votes?
This script is the GPU half: it runs every prompt over every image of a dataset pickle and writes per-image
scores. ``sam3_baseline_5213.py`` is the CPU half that scores them on each cell's pool and split exactly as
``text_baseline.py`` scores SigLIP's sort.

An image's score for a prompt is its best detection: max over SAM3's object queries of
sigmoid(pred_logit) x sigmoid(presence_logit), the same product ``post_process_object_detection`` thresholds.
The vision backbone runs once per image and is reused for every prompt (``get_vision_features``), so the
prompt count costs only the DETR decoder + mask head.

Two prompt sets are scored:

- ``typed``: the text the cells opened on (``seed_query_text``), so SAM3 and SigLIP answer the same query;
- ``noun``: short SAM3-style noun phrases (SAM3 is trained on noun phrases, not captions). A merged class
  ("a car or truck") gets one phrase per member; the analysis takes the max over them.

Output: ``<out>/shard_<i>_of_<n>.npz`` with ``ids`` (pickle keys), ``prompts``, ``top1``/``top2`` (N x P float16
scores) and ``n50`` (detections over 0.5, uint8). Resumable: an existing shard is skipped.

Run on a GPU node with HF_HOME on node scratch (the gated weights need an approved HF_TOKEN).
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
import zipfile
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402

import experiment_config as cfg  # noqa: E402

#: SAM3-style noun phrases per scale class. A tuple is a merged class: one prompt per member, max over them.
NOUNS: dict[str, tuple[str, ...]] = {
    "bicycle": ("bicycle",),
    "bird": ("bird",),
    "boat": ("boat",),
    "book": ("book",),
    "bus": ("bus",),
    "clock": ("clock",),
    "dog": ("dog",),
    "kite": ("kite",),
    "knife": ("knife",),
    "stop sign": ("stop sign",),
    "umbrella": ("umbrella",),
    "fork": ("fork",),
    "spoon": ("spoon",),
    "bowl": ("bowl",),
    "bottle": ("bottle",),
    "bench": ("bench",),
    "chair": ("chair",),
    "sink": ("sink",),
    "cell phone": ("cell phone",),
    "fire hydrant": ("fire hydrant",),
    "enclosed road vehicle": ("car", "truck"),
    "single serving drinking vessel": ("cup", "glass"),
    "bag or luggage": ("bag", "suitcase"),
    "vase or potted plant": ("vase", "potted plant"),
    "airplane": ("airplane",),
    "apple": ("apple",),
    "banana": ("banana",),
    "baseball bat": ("baseball bat",),
    "dining table": ("dining table",),
    "frisbee": ("frisbee",),
    "keyboard": ("computer keyboard",),
    "laptop": ("laptop",),
    "microwave": ("microwave",),
    "motorcycle": ("motorcycle",),
    "mouse": ("computer mouse",),
    "orange": ("orange",),
    "parking meter": ("parking meter",),
    "person": ("person",),
    "remote": ("remote control",),
    "scissors": ("scissors",),
    "skateboard": ("skateboard",),
    "skis": ("skis",),
    "snowboard": ("snowboard",),
    "surfboard": ("surfboard",),
    "tennis racket": ("tennis racket",),
    "tie": ("necktie",),
    "toothbrush": ("toothbrush",),
    "traffic light": ("traffic light",),
    "tv": ("television",),
}


def prompt_table(dataset: str, cells: list[str]) -> dict[str, dict[str, list[str]]]:
    """class -> {"typed": [text], "noun": [phrases]} for every class the cells name."""
    from vtscore.eval import scale_bands  # noqa: PLC0415

    out: dict[str, dict[str, list[str]]] = {}
    for cell in cells:
        cls, _ = scale_bands.parse_cell(cell)
        if cls in out:
            continue
        typed = cfg.seed_query_text(dataset, cell) or cls
        out[cls] = {"typed": [typed], "noun": list(NOUNS.get(cls, (cls,)))}
    return out


def _open_member(origin: str, zips: dict) -> bytes:
    """Bytes of ``archive.zip::member`` (or a plain path), keeping one open handle per archive."""
    if "::" in origin:
        arc, member = origin.split("::", 1)
        if arc not in zips:
            zips[arc] = zipfile.ZipFile(arc)
        return zips[arc].read(member)
    return Path(origin).read_bytes()


class _Images:
    """A torch Dataset over (id, origin) pairs: decodes and preprocesses in the loader's workers."""

    def __init__(self, items, processor):
        self.items, self.processor, self.zips = items, processor, {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        from PIL import Image  # noqa: PLC0415

        mid, origin = self.items[k]
        img = Image.open(io.BytesIO(_open_member(origin, self.zips))).convert("RGB")
        px = self.processor(images=img, return_tensors="pt").pixel_values[0]
        return k, px


def _no_masks():
    """A module to stand in for ``Sam3Model.mask_decoder`` (a child module must be a Module): computes nothing."""
    from types import SimpleNamespace  # noqa: PLC0415

    import torch  # noqa: PLC0415

    class NoMasks(torch.nn.Module):
        def forward(self, **_kw):
            return SimpleNamespace(pred_masks=None, semantic_seg=None, attentions=None)

    return NoMasks()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="grid results dir (prepare_info.json names the cells)")
    ap.add_argument("--dataset", default="coco_better")
    ap.add_argument(
        "--pickle-embedder", default="siglip", help="any pickle of the dataset: only ids + origins are read"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="facebook/sam3")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="first N images of the shard only (a smoke / timing run)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument(
        "--masks",
        action="store_true",
        help="run the mask head too (the scores never read it; it is most of the decode)",
    )
    args = ap.parse_args()

    import json  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from transformers import Sam3Model, Sam3Processor  # noqa: PLC0415
    from transformers.modeling_outputs import BaseModelOutputWithPooling  # noqa: PLC0415

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
    processor = Sam3Processor.from_pretrained(args.model)
    model = Sam3Model.from_pretrained(args.model, torch_dtype=dtype).to(dev).eval()
    if not args.masks:
        # Logits, boxes and presence come out of the DETR encoder/decoder before the mask head runs, so an image's
        # score is unchanged without it; the stub returns what forward() reads off the mask output.
        model.mask_decoder = _no_masks()

    # Text features once per prompt; expanded to the batch inside the loop.
    text_cache = {}
    with torch.inference_mode():
        for p in prompts:
            t = processor(text=p, return_tensors="pt").to(dev)
            emb = model.get_text_features(input_ids=t.input_ids, attention_mask=t.attention_mask, return_dict=True)
            text_cache[p] = (emb, t.attention_mask)

    top1 = np.zeros((len(items), len(prompts)), np.float16)
    top2 = np.zeros_like(top1)
    n50 = np.zeros((len(items), len(prompts)), np.uint8)

    loader = torch.utils.data.DataLoader(
        _Images(items, processor), batch_size=args.batch, num_workers=args.workers, pin_memory=True
    )
    t0, t_vis, t_dec, done = time.time(), 0.0, 0.0, 0
    with torch.inference_mode():
        for ks, px in loader:
            b = px.shape[0]
            torch.cuda.synchronize()
            a = time.time()
            vis = model.get_vision_features(pixel_values=px.to(dev, dtype=dtype, non_blocking=True))
            torch.cuda.synchronize()
            t_vis += time.time() - a
            a = time.time()
            for j, p in enumerate(prompts):
                emb, mask = text_cache[p]
                emb_b = BaseModelOutputWithPooling(
                    last_hidden_state=emb.last_hidden_state.expand(b, -1, -1),
                    pooler_output=emb.pooler_output.expand(b, -1, -1),
                )
                o = model(vision_embeds=vis, text_embeds=emb_b, attention_mask=mask.expand(b, -1))
                s = o.pred_logits.float().sigmoid()
                if o.presence_logits is not None:
                    s = s * o.presence_logits.float().sigmoid()
                top = s.topk(2, dim=1).values.cpu().numpy()
                k = ks.numpy()
                top1[k, j], top2[k, j] = top[:, 0], top[:, 1]
                n50[k, j] = np.minimum((s > 0.5).sum(1).cpu().numpy(), 255)
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
        n50=n50,
        timing=np.asarray([el, t_vis, t_dec, len(items), len(prompts)]),
        meta=np.asarray(
            [
                json.dumps(
                    {
                        "model": args.model,
                        "dtype": args.dtype,
                        "gpu": torch.cuda.get_device_name(0),
                        "batch": args.batch,
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
