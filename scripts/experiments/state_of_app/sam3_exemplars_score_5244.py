"""SAM3 cross-image exemplars from a session's votes, on #4711's Region grid (#5244).

SAM3 takes box exemplars (positive or negative) only inside the image it is searching. The workaround
(FSS-SAM3, arXiv 2604.05433) pastes the reference image, box and all, onto one canvas with the target, so the
backbone's attention can relate them. Here the reference boxes come from a Region session's own votes: the box
a simulated Good vote drags (``region_box_for_category``, the union of the class's instances) and, for a Bad
vote, the box SAM3's text prompt likes best on that image (the thing the user turned down).

Two stages:

- ``plan`` (CPU): per cell, seed 0, the withheld half ``text_baseline.py`` grades (same pool, same split) and the
  exemplars: the first three Good votes and the first Bad vote of #4711's shipped ``max_patch`` run, with the
  vote each arrived at. Writes ``<out>/plan/<cell>.json``.
- ``score`` (GPU): every withheld image on two canvases, scored under three prompts each (six arms):

  ====== ==================== ===========================================
  canvas  references            arms
  ====== ==================== ===========================================
  A       Good 1, Bad 1         ``c1`` text only, ``g1`` + Good box, ``g1b1`` + Good and Bad boxes
  B       Good 1-3, Bad 1       ``c3`` text only, ``g3`` + 3 Good boxes, ``g3b1`` + 3 Good and Bad boxes
  ====== ==================== ===========================================

  The ``c`` arms are the canvas controls: the same pixels, no box. Against them an exemplar arm measures the
  boxes; against #5213's native text-only score it measures what a user would get.

Canvas: 1008 x 1008 (SAM3's input, so the processor's resize is the identity). The target takes the half that
suits its shape (top half if landscape, left half if portrait); the references share the other half on the grid
that keeps them largest. A reference is cropped around its box (the box doubled, at least 40% of each side) so
a small object keeps some pixels. A score is the best detection whose box centre lies on the TARGET's part of
the canvas: the references hold the object themselves, and every canvas would score high otherwise.

Output ``<out>/scores/<cell>.npz``: ``ids`` (withheld half), ``arms``, ``score`` (N x arms, float32: max over
target-side queries of sigmoid(logit) x sigmoid(presence)), ``logit`` (the same query's sigmoid(logit)),
``presence``, ``on_ref`` (the canvas's best query overall sat on a reference), plus timing. Resumable per cell.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import zipfile
import zlib
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402

CANVAS = 1008
FILL = (124, 116, 104)
#: arm -> (canvas, which reference boxes go in: "g" = Good, "b" = Bad)
ARMS = {
    "c1": ("A", ""),
    "g1": ("A", "g"),
    "g1b1": ("A", "gb"),
    "c3": ("B", ""),
    "g3": ("B", "g"),
    "g3b1": ("B", "gb"),
}
CANVAS_REFS = {"A": (1, 1), "B": (3, 1)}  # (Goods, Bads) on the canvas
MAX_GOODS = 3
REGION_RUN = "/expscratch/sgreenberg/state-of-the-app/2026-10-23-regstyles4711-mp-region-b1/results/cells"


def cell_slug(cat: str) -> str:
    return cat.replace(" ", "_").replace("@", "__")


# --------------------------------------------------------------------------------------------------- plan (CPU)


def first_votes(region_cells: Path, seed: int) -> dict[str, dict]:
    """cell -> {"goods": [(t, id)], "bads": [(t, id)]}: the Region run's ordinary votes in click order."""
    import pandas as pd  # noqa: PLC0415

    out: dict[str, dict] = {}
    for f in sorted(region_cells.glob("task_*__picks.csv*")):
        d = pd.read_csv(f, usecols=["seed", "category", "t", "phase", "picked_id", "picked_label", "embedder"])
        d = d[(d["seed"] == seed) & (d["embedder"] == "siglip+dinov3_patch") & (d["phase"].astype(str) != "check")]
        for cat, g in d.groupby("category"):
            g = g.sort_values("t")
            lab = g["picked_label"].astype(int).to_numpy()
            tt, ids = g["t"].astype(int).to_numpy(), g["picked_id"].astype(int).to_numpy()
            out[cat] = {
                "goods": [(int(t), int(i)) for t, i, y in zip(tt, ids, lab) if y == 1][:MAX_GOODS],
                "bads": [(int(t), int(i)) for t, i, y in zip(tt, ids, lab) if y == 0][:1],
            }
    return out


def plan(args) -> int:
    import experiment_config as cfg  # noqa: PLC0415
    from text_baseline import _split  # noqa: PLC0415
    from vtscore.config import EMBEDDINGS_DIR  # noqa: PLC0415
    from vtscore.eval.labels import evaluable_pool, media_is_positive, region_box_for_category  # noqa: PLC0415

    from _cells_io import load_medias  # noqa: PLC0415

    out = Path(args.out) / "plan"
    out.mkdir(parents=True, exist_ok=True)
    prepare = json.loads((Path(args.results) / "prepare_info.json").read_text())
    cats = next(iter(prepare["datasets"]["coco_better"].values()))["selected_categories"]
    votes = first_votes(Path(args.region), args.seed)
    medias = load_medias(EMBEDDINGS_DIR / "coco_better__siglip.pkl")
    common.log(f"{len(medias)} medias, {len(cats)} cells, {len(votes)} cells with Region picks")
    for cat in cats:
        pool = evaluable_pool(medias, cat)
        sim_ids, test_ids = _split(pool, cfg.SIM_FRACTION, args.seed)
        sim, test = set(sim_ids), sorted(test_ids)
        v = votes.get(cat)
        if v is None:
            common.log(f"SKIP {cat}: no Region picks")
            continue
        goods = []
        for t, i in v["goods"]:
            assert i in sim and media_is_positive(pool[i], cat), (cat, i)
            goods.append(
                {"t": t, "id": i, "origin": pool[i]["origin_name"], "box": region_box_for_category(pool[i], cat)}
            )
        bads = []
        for t, i in v["bads"]:
            assert i in sim and not media_is_positive(pool[i], cat), (cat, i)
            bads.append({"t": t, "id": i, "origin": pool[i]["origin_name"]})
        rec = {
            "category": cat,
            "seed": args.seed,
            "goods": goods,
            "bads": bads,
            "test": [[i, pool[i]["origin_name"], int(media_is_positive(pool[i], cat))] for i in test],
        }
        (out / f"{cell_slug(cat)}.json").write_text(json.dumps(rec))
        npos = sum(r[2] for r in rec["test"])
        common.log(
            f"{cat:30s} test {len(test)} ({npos} pos)  goods at t={[g['t'] for g in goods]}  bad at {[b['t'] for b in bads]}"
        )
    return 0


def sample(args) -> int:
    """Importance-sample each plan's withheld half: every positive, the *top* negatives by SAM3's native text score
    (the ones AP turns on), and a seeded *frac* of the rest at weight 1/frac. ``test_all`` keeps the whole half's
    ids and labels, so the analyzer can check the weighted estimate against the exact native AP."""
    from sam3_scores import load_sam3  # noqa: PLC0415

    native = load_sam3(Path(args.native), "noun")
    src, dst = Path(args.out) / "plan", Path(args.out) / "plan_sampled"
    dst.mkdir(parents=True, exist_ok=True)
    for pf in sorted(src.glob("*.json")):
        rec = json.loads(pf.read_text())
        test = rec["test"]
        nat, _ = native(rec["category"], {i: None for i, _, _ in test})
        neg = sorted((r for r in test if not r[2]), key=lambda r: -nat[r[0]])
        top, rest = neg[: args.top], neg[args.top :]
        rng = np.random.default_rng(zlib.crc32(rec["category"].encode()))
        pick = rng.random(len(rest)) < args.frac
        kept = [[*r, 1.0] for r in test if r[2]] + [[*r, 1.0] for r in top]
        kept += [[*r, 1.0 / args.frac] for r, k in zip(rest, pick) if k]
        rec["test_all"] = [[i, y] for i, _, y in test]
        rec["test"] = kept
        rec["sample"] = {"top": args.top, "frac": args.frac}
        (dst / pf.name).write_text(json.dumps(rec))
        common.log(f"{rec['category']:30s} {len(test)} -> {len(kept)}")
    return 0


# -------------------------------------------------------------------------------------------------- canvas


def _open(origin: str, zips: dict):
    from PIL import Image  # noqa: PLC0415

    if "::" in origin:
        arc, member = origin.split("::", 1)
        if arc not in zips:
            zips[arc] = zipfile.ZipFile(arc)
        data = zips[arc].read(member)
    else:
        data = Path(origin).read_bytes()
    return Image.open(io.BytesIO(data)).convert("RGB")


def crop_ref(img, box):
    """Crop *img* around normalised *box*: the box doubled, at least 40% of each side. Returns (crop, box in crop)."""
    W, H = img.size
    x0, y0, x1, y1 = box[0] * W, box[1] * H, box[2] * W, box[3] * H
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w = min(W, max(2 * (x1 - x0), 0.4 * W))
    h = min(H, max(2 * (y1 - y0), 0.4 * H))
    a = min(max(cx - w / 2, 0), W - w)
    b = min(max(cy - h / 2, 0), H - h)
    c = img.crop((int(a), int(b), int(round(a + w)), int(round(b + h))))
    cw, ch = c.size
    nb = ((x0 - int(a)) / cw, (y0 - int(b)) / ch, (x1 - int(a)) / cw, (y1 - int(b)) / ch)
    return c, tuple(min(max(z, 0.0), 1.0) for z in nb)


def _fit(img, rect):
    """Letterbox *img* into rect (x, y, w, h); returns (resized image, its placed rect)."""
    x, y, w, h = rect
    s = min(w / img.width, h / img.height)
    nw, nh = max(1, int(img.width * s)), max(1, int(img.height * s))
    return img.resize((nw, nh)), (x + (w - nw) // 2, y + (h - nh) // 2, nw, nh)


def _grid(n: int, w: int, h: int, refs) -> tuple[int, int]:
    """The (rows, cols) grid for *n* references in a w x h half that keeps the smallest reference largest."""
    best, arg = -1.0, (1, n)
    for r in range(1, n + 1):
        c = -(-n // r)
        cw, ch = w / c, h / r
        s = min(min(cw / im.width, ch / im.height) for im in refs)
        if s > best:
            best, arg = s, (r, c)
    return arg


def compose(target, refs):
    """Canvas of *target* + reference crops [(img, box)]. Returns (canvas, target rect, [ref box on canvas])."""
    from PIL import Image  # noqa: PLC0415

    canvas = Image.new("RGB", (CANVAS, CANVAS), FILL)
    half = CANVAS // 2
    if target.width >= target.height:
        trect, rrect = (0, 0, CANVAS, half), (0, half, CANVAS, half)
    else:
        trect, rrect = (0, 0, half, CANVAS), (half, 0, half, CANVAS)
    t, tplaced = _fit(target, trect)
    canvas.paste(t, tplaced[:2])
    rows, cols = _grid(len(refs), rrect[2], rrect[3], [r[0] for r in refs])
    cw, ch = rrect[2] // cols, rrect[3] // rows
    boxes = []
    for k, (im, bx) in enumerate(refs):
        r, c = divmod(k, cols)
        im2, (px, py, pw, ph) = _fit(im, (rrect[0] + c * cw, rrect[1] + r * ch, cw, ch))
        canvas.paste(im2, (px, py))
        boxes.append(
            (
                (px + bx[0] * pw) / CANVAS,
                (py + bx[1] * ph) / CANVAS,
                (px + bx[2] * pw) / CANVAS,
                (py + bx[3] * ph) / CANVAS,
            )
        )
    tx, ty, tw, th = tplaced
    return canvas, (tx / CANVAS, ty / CANVAS, (tx + tw) / CANVAS, (ty + th) / CANVAS), boxes


class _Canvases:
    """torch Dataset: each withheld image on canvas A and canvas B (pixel values + target rect + ref boxes)."""

    def __init__(self, items, refs, processor):
        self.items, self.refs, self.processor, self.zips = items, refs, processor, {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        import torch  # noqa: PLC0415

        target = _open(self.items[k][1], self.zips)
        out = {"k": k}
        for name, refs in self.refs.items():
            canvas, trect, boxes = compose(target, refs)
            out[name] = self.processor(images=canvas, return_tensors="pt").pixel_values[0]
            out[f"{name}_trect"] = torch.tensor(trect)
            out[f"{name}_boxes"] = torch.tensor(boxes)
        return out


# --------------------------------------------------------------------------------------------------- score (GPU)


def _xyxy_to_cxcywh(b):
    import torch  # noqa: PLC0415

    x0, y0, x1, y1 = b.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], -1)


def score(args) -> int:
    import torch  # noqa: PLC0415
    from transformers import Sam3Model, Sam3Processor  # noqa: PLC0415
    from transformers.modeling_outputs import BaseModelOutputWithPooling  # noqa: PLC0415

    from sam3_score_5213 import NOUNS, _no_masks  # noqa: PLC0415
    from vtscore.eval import scale_bands  # noqa: PLC0415

    root = Path(args.out)
    plans = sorted((root / args.plan).glob("*.json"))
    plans = plans[args.shard :: args.nshards]
    (root / "scores").mkdir(parents=True, exist_ok=True)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    dev = torch.device("cuda")
    processor = Sam3Processor.from_pretrained(args.model)
    model = Sam3Model.from_pretrained(args.model, torch_dtype=dtype).to(dev).eval()
    model.mask_decoder = _no_masks()  # scores come out before the mask head (#5213)
    gpu = torch.cuda.get_device_name(0)

    def text(p):
        t = processor(text=p, return_tensors="pt").to(dev)
        emb = model.get_text_features(input_ids=t.input_ids, attention_mask=t.attention_mask, return_dict=True)
        return emb, t.attention_mask

    def decode(vis, emb, mask, b, boxes=None, labels=None):
        e = BaseModelOutputWithPooling(
            last_hidden_state=emb.last_hidden_state.expand(b, -1, -1),
            pooler_output=emb.pooler_output.expand(b, -1, -1),
        )
        kw = {}
        if boxes is not None:
            kw = {"input_boxes": boxes.to(dev, dtype=dtype), "input_boxes_labels": labels.to(dev)}
        return model(vision_embeds=vis, text_embeds=e, attention_mask=mask.expand(b, -1), **kw)

    for pf in plans:
        rec = json.loads(pf.read_text())
        cat = rec["category"]
        dest = root / "scores" / f"{pf.stem}{f'_limit{args.limit}' if args.limit else ''}.npz"
        if dest.exists():
            common.log(f"{dest.name} exists, skipping")
            continue
        cls, _ = scale_bands.parse_cell(cat)
        prompts = list(NOUNS.get(cls, (cls,)))
        if len(rec["goods"]) < 1 or len(rec["bads"]) < 1:
            common.log(f"SKIP {cat}: {len(rec['goods'])} Goods, {len(rec['bads'])} Bads")
            continue
        items = [tuple(r) for r in rec["test"]]
        if args.limit:
            # A smoke keeps every positive, so its AP means something.
            pos = [r for r in items if r[2]]
            items = pos + [r for r in items if not r[2]][: max(0, args.limit - len(pos))]
        zips: dict = {}
        with torch.inference_mode():
            embs = {p: text(p) for p in prompts}
            # The Bad's box: SAM3's best text detection on that image, alone.
            bad = rec["bads"][0]
            bimg = _open(bad["origin"], zips)
            px = processor(images=bimg, return_tensors="pt").pixel_values.to(dev, dtype=dtype)
            vis = model.get_vision_features(pixel_values=px)
            best = (-1.0, None)
            for p in prompts:
                o = decode(vis, *embs[p], 1)
                s = (o.pred_logits.float().sigmoid() * o.presence_logits.float().sigmoid())[0]
                j = int(s.argmax())
                if float(s[j]) > best[0]:
                    best = (float(s[j]), o.pred_boxes[0, j].float().clamp(0, 1).tolist())
            bad["box"], bad["sam3_score"] = best[1], best[0]
        goods = [crop_ref(_open(g["origin"], zips), g["box"]) for g in rec["goods"]]
        bads = [crop_ref(bimg, bad["box"])]
        refs = {"A": goods[:1] + bads}
        have_b = len(goods) >= MAX_GOODS
        if have_b:
            refs["B"] = goods[:MAX_GOODS] + bads
        arms = [a for a, (c, _) in ARMS.items() if c in refs]
        if args.dump_canvases:
            dd = root / "canvases"
            dd.mkdir(exist_ok=True)
            for k in range(min(4, len(items))):
                tgt = _open(items[k][1], zips)
                for name, r in refs.items():
                    from PIL import ImageDraw  # noqa: PLC0415

                    cv, tr, bx = compose(tgt, r)
                    d = ImageDraw.Draw(cv)
                    d.rectangle([z * CANVAS for z in tr], outline="#1d4ed8", width=4)
                    for q, b in enumerate(bx):
                        d.rectangle(
                            [z * CANVAS for z in b], outline="#15803d" if q < len(r) - 1 else "#b91c1c", width=4
                        )
                    cv.save(dd / f"{pf.stem}_{k}_{name}.jpg", quality=85)

        n = len(items)
        sc = np.zeros((n, len(arms)), np.float32)
        lg = np.zeros_like(sc)
        pr = np.zeros_like(sc)
        on_ref = np.zeros((n, len(arms)), np.uint8)
        loader = torch.utils.data.DataLoader(
            _Canvases(items, refs, processor), batch_size=args.batch, num_workers=args.workers, pin_memory=True
        )
        t0, t_vis, t_dec, done = time.time(), 0.0, 0.0, 0
        common.log(f"{cat}: {n} withheld x arms {arms} x prompts {prompts} on {gpu}")
        with torch.inference_mode():
            for batch in loader:
                ks = batch["k"].numpy()
                b = len(ks)
                for cname in refs:
                    torch.cuda.synchronize()
                    a = time.time()
                    vis = model.get_vision_features(pixel_values=batch[cname].to(dev, dtype=dtype, non_blocking=True))
                    torch.cuda.synchronize()
                    t_vis += time.time() - a
                    a = time.time()
                    tr = batch[f"{cname}_trect"].to(dev)  # b x 4 (xyxy)
                    rb = _xyxy_to_cxcywh(batch[f"{cname}_boxes"].float())  # b x nref x 4
                    ng = CANVAS_REFS[cname][0]
                    for arm, (c, which) in ARMS.items():
                        if c != cname:
                            continue
                        j = arms.index(arm)
                        if which == "":
                            boxes = labels = None
                        elif which == "g":
                            boxes, labels = rb[:, :ng], torch.ones(b, ng, dtype=torch.long)
                        else:
                            boxes = rb
                            labels = torch.cat([torch.ones(b, ng), torch.zeros(b, rb.shape[1] - ng)], 1).long()
                        best = torch.full((b,), -1.0, device=dev)
                        bl, bp = torch.zeros(b, device=dev), torch.zeros(b, device=dev)
                        ref_top = torch.zeros(b, dtype=torch.bool, device=dev)
                        for p in prompts:
                            emb, mask = embs[p]
                            o = decode(vis, emb, mask, b, boxes, labels)
                            lo = o.pred_logits.float().sigmoid()  # b x Q
                            presence_p = o.presence_logits.float().sigmoid().reshape(b, 1)
                            s = lo * presence_p
                            bx = o.pred_boxes.float()  # b x Q x 4 xyxy
                            cx, cy = (bx[..., 0] + bx[..., 2]) / 2, (bx[..., 1] + bx[..., 3]) / 2
                            on_t = (
                                (cx >= tr[:, None, 0])
                                & (cx <= tr[:, None, 2])
                                & (cy >= tr[:, None, 1])
                                & (cy <= tr[:, None, 3])
                            )
                            st = torch.where(on_t, s, torch.full_like(s, -1.0))
                            v, q = st.max(1)
                            better = v > best
                            best = torch.where(better, v, best)
                            bl = torch.where(better, lo.gather(1, q[:, None])[:, 0], bl)
                            bp = torch.where(better, presence_p[:, 0], bp)
                            top_any = s.argmax(1)
                            ref_top |= ~on_t.gather(1, top_any[:, None])[:, 0]
                        sc[ks, j] = best.clamp(min=0).cpu().numpy()
                        lg[ks, j] = bl.cpu().numpy()
                        pr[ks, j] = bp.cpu().numpy()
                        on_ref[ks, j] = ref_top.cpu().numpy()
                    torch.cuda.synchronize()
                    t_dec += time.time() - a
                done += b
                if done % (args.batch * 100) < args.batch or done == n:
                    el = time.time() - t0
                    common.log(
                        f"  {done}/{n}  {done / el:.2f} img/s  vision {t_vis / done / len(refs) * 1e3:.0f} ms/canvas  "
                        f"decode {t_dec / done / len(arms) / len(prompts) * 1e3:.1f} ms/img/arm/prompt"
                    )
        el = time.time() - t0
        meta = {
            "category": cat,
            "model": args.model,
            "dtype": args.dtype,
            "gpu": gpu,
            "host": os.uname().nodename,
            "batch": args.batch,
            "prompts": prompts,
            "goods": [{k: g[k] for k in ("t", "id", "box")} for g in rec["goods"]],
            "bad": {k: bad[k] for k in ("t", "id", "box", "sam3_score")},
            "canvases": list(refs),
        }
        np.savez(
            dest,
            ids=np.asarray([r[0] for r in items]),
            labels=np.asarray([r[2] for r in items], dtype=np.uint8),
            weights=np.asarray([r[3] if len(r) > 3 else 1.0 for r in items], dtype=np.float32),
            arms=np.asarray(arms),
            score=sc,
            logit=lg,
            presence=pr,
            on_ref=on_ref,
            timing=np.asarray([el, t_vis, t_dec, n, len(refs), len(arms), len(prompts)]),
            meta=np.asarray([json.dumps(meta)]),
        )
        common.log(f"wrote {dest.name}: {n} imgs in {el:.0f} s (vision {t_vis:.0f} s, decode {t_dec:.0f} s)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--results", required=True, help="the grid's results dir (prepare_info.json)")
    p.add_argument("--region", default=REGION_RUN, help="#4711's max_patch run cells (the pick logs)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    s.add_argument("--model", default="facebook/sam3")
    s.add_argument("--shard", type=int, default=0)
    s.add_argument("--nshards", type=int, default=1)
    s.add_argument("--limit", type=int, default=0, help="smoke: every positive + negatives up to N images")
    s.add_argument("--batch", type=int, default=4)
    s.add_argument("--workers", type=int, default=6)
    s.add_argument("--dtype", default="fp16", choices=["bf16", "fp16", "fp32"])
    s.add_argument("--plan", default="plan", help="plan dir under --out (plan_sampled for the importance sample)")
    s.add_argument("--dump-canvases", action="store_true", help="write the first 4 canvases per cell as JPEGs")
    m = sub.add_parser("sample")
    m.add_argument("--out", required=True)
    m.add_argument("--native", required=True, help="#5213's score shards")
    m.add_argument("--top", type=int, default=800, help="negatives kept by native rank")
    m.add_argument("--frac", type=float, default=0.08, help="share of the other negatives kept (weight 1/frac)")
    args = ap.parse_args()
    return {"plan": plan, "sample": sample, "score": score}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
