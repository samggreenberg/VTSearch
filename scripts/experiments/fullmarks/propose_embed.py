"""Completeness proposals from learned embeddings, for ``completeness_multi.py`` (#3951).

Two proposers, neither of which shares anything with SIFT:

``siglip_tiles``
    Every anchor page is cut into overlapping tiles at two sizes (the ``t4``
    layout #3928 found best, 0.25 x 0.18 of the page, and one half that size for
    small marks).  Each tile and each class query crop is embedded with SigLIP,
    and a page scores the best tile's cosine; the proposal box is that tile.

``dinov3_patches``
    Each anchor page runs through DINOv3 ViT-B/16 at ``LONG_SIDE`` pixels, giving
    a dense grid of patch features.  The query is the patch grid under the class's
    query box *on its own page*, so the query sees the same resolution and
    context as the pages it is matched against, at three grid scales.  A window
    scores the mean, over query patches, of that patch's best cosine within one
    patch of its expected position -- template matching on features, with a
    little slack for deformation.  The proposal box is the best window.

Only anchor pages (SPODS, StaVer, Tobacco800) are scored: only they can hold a
missing member.  Output goes to ``--out``/proposals-<method>.json; nothing under
the corpus is written.

    python propose_embed.py siglip_tiles   --out /expscratch/.../completeness2
    python propose_embed.py dinov3_patches --out /expscratch/.../completeness2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fullmarks_config as cfg  # noqa: E402
from completeness import ANCHOR_SOURCES  # noqa: E402
from completeness_multi import proposals_path, save_proposals  # noqa: E402
from sources._common import Page, read_manifest  # noqa: E402

TOP_K = 100
TILE_LAYOUTS = {"t4": (0.25, 0.18), "t8": (0.125, 0.09)}
LONG_SIDE = 2048
PATCH = 16
GRID_SCALES = (0.75, 1.0, 1.33)
MIN_GRID, MAX_GRID = 2, 40


def roster(corpus: Path) -> dict[str, Any]:
    classes = json.loads((corpus / "classes.json").read_text(encoding="utf-8"))
    return {cid: meta for cid, meta in sorted(classes.items()) if meta.get("on_roster") and meta.get("query_crop")}


def query_regions(classes: dict[str, Any], pages: dict[str, Page], corpus: Path) -> dict[str, list[tuple[str, tuple]]]:
    """Per class, (page_id, box) of the primary query plus any stored extra query crops."""
    store_path = corpus / "query_crops.json"
    store = json.loads(store_path.read_text(encoding="utf-8")).get("classes", {}) if store_path.exists() else {}
    out: dict[str, list[tuple[str, tuple]]] = {}
    for cid, meta in classes.items():
        regions = []
        page = pages.get(meta.get("query_page_id", ""))
        if page is not None:
            marks = [m for m in page.marks if m.class_id == cid]
            if marks:
                regions.append((page.page_id, tuple(max(marks, key=lambda m: m.box[2] * m.box[3]).box)))
        regions += [(e["page_id"], tuple(e["box"])) for e in store.get(cid, []) if e["page_id"] in pages]
        out[cid] = regions
    return out


def top_k(scores: np.ndarray, boxes: list[list[Optional[tuple]]], page_ids: list[str], order: list[str], k: int):
    """scores: (pages, classes); boxes[page][class]."""
    by_class = {}
    for ci, cid in enumerate(order):
        idx = np.argsort(-scores[:, ci], kind="stable")[:k]
        by_class[cid] = [(page_ids[i], float(scores[i, ci]), boxes[i][ci]) for i in idx]
    return by_class


def _load_rgb(path: str):
    from vtscore.media.image._image_bulk import _load_pil  # noqa: PLC0415

    return _load_pil(Path(path)).convert("RGB")


# ---------------------------------------------------------------------------
# SigLIP tiles
# ---------------------------------------------------------------------------


def tile_boxes(width: int, height: int) -> list[tuple[int, int, int, int]]:
    boxes = []
    for tw, th in TILE_LAYOUTS.values():
        xs = np.arange(0.0, max(1e-9, 1.0 - tw) + 1e-9, tw / 2)
        ys = np.arange(0.0, max(1e-9, 1.0 - th) + 1e-9, th / 2)
        for y in ys:
            for x in xs:
                boxes.append((int(x * width), int(y * height), max(1, int(tw * width)), max(1, int(th * height))))
    return boxes


class _TileSet:
    def __init__(self, pages: list[Page], size: int, mean: Sequence[float], std: Sequence[float]):
        self.pages, self.size = pages, size
        self.mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)

    def __len__(self) -> int:
        return len(self.pages)

    def prep(self, im) -> np.ndarray:
        from PIL import Image  # noqa: PLC0415

        arr = np.asarray(im.resize((self.size, self.size), Image.BICUBIC), dtype=np.float32).transpose(2, 0, 1) / 255.0
        return (arr - self.mean) / self.std

    def __getitem__(self, i: int):
        page = self.pages[i]
        im = _load_rgb(page.path)
        # manifest boxes are in manifest pixels; tiles are cut in the same frame
        if im.size != (page.width, page.height):
            im = im.resize((page.width, page.height))
        boxes = tile_boxes(page.width, page.height)
        tiles = np.stack([self.prep(im.crop((x, y, x + w, y + h))) for x, y, w, h in boxes])
        return i, tiles, boxes


def siglip_tiles(corpus: Path, out: Path, workers: int, limit: int = 0) -> dict[str, list]:
    import torch  # noqa: PLC0415

    from vtscore.media.image.embedder_siglip import EMBEDDER  # noqa: PLC0415

    all_pages = {p.page_id: p for p in read_manifest(corpus / "corpus.jsonl")}
    anchors = sorted((p for p in all_pages.values() if p.source in ANCHOR_SOURCES), key=lambda p: p.page_id)
    if limit:
        anchors = anchors[:: max(1, len(anchors) // limit)][:limit]  # a smoke run across all three sources
    classes = roster(corpus)
    order = list(classes)

    EMBEDDER.load_models()
    model, proc = EMBEDDER._model, EMBEDDER._processor.image_processor
    size = int(proc.size.get("height", 224)) if isinstance(proc.size, dict) else 224
    ds = _TileSet(anchors, size, proc.image_mean, proc.image_std)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    def embed(batch: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            feats = []
            for j in range(0, len(batch), 512):
                x = torch.from_numpy(batch[j : j + 512]).to(device=device, dtype=dtype)
                f = model.get_image_features(pixel_values=x)
                f = f.pooler_output if hasattr(f, "pooler_output") else f
                feats.append(torch.nn.functional.normalize(f.float(), dim=-1))
            return torch.cat(feats)

    regions = query_regions(classes, all_pages, corpus)
    q_rows, q_class = [], []
    for ci, cid in enumerate(order):
        crops = []
        for page_id, (x, y, w, h) in regions[cid]:
            page = all_pages[page_id]
            im = _load_rgb(page.path)
            if im.size != (page.width, page.height):
                im = im.resize((page.width, page.height))
            crops.append(ds.prep(im.crop((x, y, x + w, y + h))))
        if not crops:
            from PIL import Image  # noqa: PLC0415

            with Image.open(classes[cid]["query_crop"]) as im:
                crops.append(ds.prep(im.convert("RGB")))
        q_rows.append(np.stack(crops))
        q_class += [ci] * len(crops)
    queries = embed(np.concatenate(q_rows))  # (Q, D)
    q_class_t = torch.tensor(q_class, device=queries.device)
    print(
        f"siglip_tiles: {len(anchors)} anchor pages, {len(order)} classes, {queries.shape[0]} query crops", flush=True
    )

    scores = np.full((len(anchors), len(order)), -1.0, dtype=np.float32)
    boxes: list[list[Optional[tuple]]] = [[None] * len(order) for _ in anchors]
    loader = torch.utils.data.DataLoader(ds, batch_size=None, shuffle=False, num_workers=workers, prefetch_factor=2)
    t0 = time.time()
    for n, (i, tiles, tboxes) in enumerate(loader):
        tiles = tiles.numpy() if hasattr(tiles, "numpy") else tiles
        sims = embed(tiles) @ queries.T  # (T, Q)
        # best query crop per class, then best tile per class
        per_class = torch.full((sims.shape[0], len(order)), -1.0, device=sims.device)
        per_class.scatter_reduce_(1, q_class_t.expand(sims.shape[0], -1), sims, reduce="amax")
        best, arg = per_class.max(dim=0)
        scores[i] = best.cpu().numpy()
        arg = arg.cpu().numpy()
        boxes[i] = [tuple(int(v) for v in tboxes[a]) for a in arg]
        if (n + 1) % 250 == 0:
            print(f"  {n + 1}/{len(anchors)} pages, {time.time() - t0:.0f}s", flush=True)
    by_class = top_k(scores, boxes, [p.page_id for p in anchors], order, TOP_K)
    save_proposals("siglip_tiles", by_class, proposals_path(out, "siglip_tiles"))
    return by_class


# ---------------------------------------------------------------------------
# DINOv3 dense patches
# ---------------------------------------------------------------------------


def _dino_input(im, page: Page, mean, std):
    """Resize a page so its long side is LONG_SIDE (a multiple of PATCH); return (tensor array, scale)."""
    from PIL import Image  # noqa: PLC0415

    if im.size != (page.width, page.height):
        im = im.resize((page.width, page.height))
    scale = LONG_SIDE / max(page.width, page.height)
    w = max(PATCH, int(round(page.width * scale / PATCH)) * PATCH)
    h = max(PATCH, int(round(page.height * scale / PATCH)) * PATCH)
    arr = np.asarray(im.resize((w, h), Image.BICUBIC), dtype=np.float32).transpose(2, 0, 1) / 255.0
    arr = (arr - np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)) / np.asarray(std, dtype=np.float32).reshape(
        3, 1, 1
    )
    return arr, (w / page.width, h / page.height)


class _PageSet:
    def __init__(self, pages: list[Page], mean, std):
        self.pages, self.mean, self.std = pages, mean, std

    def __len__(self) -> int:
        return len(self.pages)

    def __getitem__(self, i: int):
        page = self.pages[i]
        arr, scale = _dino_input(_load_rgb(page.path), page, self.mean, self.std)
        return i, arr, scale


def match_grid(page_feats, query_feats):
    """Best window of *query_feats* (h, w, C) in *page_feats* (H, W, C), both L2-normalised.

    Score per window: mean over query patches of the patch's best cosine within
    one patch of its place.  Returns (score, row, col) of the best window.
    """
    import torch  # noqa: PLC0415

    H, W, C = page_feats.shape
    h, w, _ = query_feats.shape
    if h > H or w > W:
        return -1.0, 0, 0
    sims = torch.einsum("hwc,kc->khw", page_feats, query_feats.reshape(-1, C))  # (K, H, W)
    sims = torch.nn.functional.max_pool2d(sims.unsqueeze(0), 3, stride=1, padding=1).squeeze(0)
    Ho, Wo = H - h + 1, W - w + 1
    ki, kj = torch.meshgrid(torch.arange(h, device=sims.device), torch.arange(w, device=sims.device), indexing="ij")
    ys, xs = torch.meshgrid(torch.arange(Ho, device=sims.device), torch.arange(Wo, device=sims.device), indexing="ij")
    flat = sims.reshape(h * w, H * W)
    idx = (ys.reshape(1, -1) + ki.reshape(-1, 1)) * W + (xs.reshape(1, -1) + kj.reshape(-1, 1))  # (K, Ho*Wo)
    score = flat.gather(1, idx).mean(dim=0)
    best = int(torch.argmax(score))
    return float(score[best]), best // Wo, best % Wo


def dinov3_patches(corpus: Path, out: Path, workers: int, limit: int = 0) -> dict[str, list]:
    import torch  # noqa: PLC0415

    from vtscore.media.image.embedder_dinov3_single import ImageDinov3SingleEmbedder  # noqa: PLC0415

    all_pages = {p.page_id: p for p in read_manifest(corpus / "corpus.jsonl")}
    anchors = sorted((p for p in all_pages.values() if p.source in ANCHOR_SOURCES), key=lambda p: p.page_id)
    if limit:
        anchors = anchors[:: max(1, len(anchors) // limit)][:limit]  # a smoke run across all three sources
    classes = roster(corpus)
    order = list(classes)

    emb = ImageDinov3SingleEmbedder()
    emb.load_models()
    model, proc = emb._model, emb._processor
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    n_skip = 1 + int(getattr(model.config, "num_register_tokens", 4))

    def page_grid(arr: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(arr).unsqueeze(0).to(device=device, dtype=dtype)
        with torch.no_grad():
            tokens = model(pixel_values=x).last_hidden_state[0, n_skip:].float()
        gh, gw = arr.shape[1] // PATCH, arr.shape[2] // PATCH
        return torch.nn.functional.normalize(tokens.reshape(gh, gw, -1), dim=-1)

    # Query grids: slice the query region out of its own page's grid, then resample.
    regions = query_regions(classes, all_pages, corpus)
    queries: list[tuple[int, torch.Tensor]] = []
    for ci, cid in enumerate(order):
        for page_id, (x, y, w, h) in regions[cid]:
            page = all_pages[page_id]
            arr, (sx, sy) = _dino_input(_load_rgb(page.path), page, proc.image_mean, proc.image_std)
            grid = page_grid(arr)
            r0, c0 = int(y * sy // PATCH), int(x * sx // PATCH)
            r1 = max(r0 + 1, int(round((y + h) * sy / PATCH)))
            c1 = max(c0 + 1, int(round((x + w) * sx / PATCH)))
            base = grid[r0:r1, c0:c1]
            for s in GRID_SCALES:
                gh = min(MAX_GRID, max(MIN_GRID, int(round(base.shape[0] * s))))
                gw = min(MAX_GRID, max(MIN_GRID, int(round(base.shape[1] * s))))
                q = torch.nn.functional.interpolate(
                    base.permute(2, 0, 1).unsqueeze(0), size=(gh, gw), mode="bilinear", align_corners=False
                )
                queries.append((ci, torch.nn.functional.normalize(q[0].permute(1, 2, 0), dim=-1)))
    print(f"dinov3_patches: {len(anchors)} anchor pages, {len(order)} classes, {len(queries)} query grids", flush=True)

    scores = np.full((len(anchors), len(order)), -1.0, dtype=np.float32)
    boxes: list[list[Optional[tuple]]] = [[None] * len(order) for _ in anchors]
    loader = torch.utils.data.DataLoader(
        _PageSet(anchors, proc.image_mean, proc.image_std),
        batch_size=None,
        shuffle=False,
        num_workers=workers,
        prefetch_factor=2,
    )
    t0 = time.time()
    for n, (i, arr, (sx, sy)) in enumerate(loader):
        arr = arr.numpy() if hasattr(arr, "numpy") else arr
        sx, sy = float(sx), float(sy)
        grid = page_grid(arr)
        for ci, q in queries:
            s, r, c = match_grid(grid, q)
            if s > scores[i, ci]:
                scores[i, ci] = s
                boxes[i][ci] = (
                    int(c * PATCH / sx),
                    int(r * PATCH / sy),
                    max(1, int(q.shape[1] * PATCH / sx)),
                    max(1, int(q.shape[0] * PATCH / sy)),
                )
        if (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(anchors)} pages, {time.time() - t0:.0f}s", flush=True)
    by_class = top_k(scores, boxes, [p.page_id for p in anchors], order, TOP_K)
    save_proposals("dinov3_patches", by_class, proposals_path(out, "dinov3_patches"))
    return by_class


def recall_check(by_class: dict[str, list], corpus: Path) -> None:
    """Sanity: how many members each method ranks in its top K (members are dropped later)."""
    classes = roster(corpus)
    for cid, rows in by_class.items():
        members = set(classes[cid].get("page_ids", []))
        hit = sum(1 for p, _, _ in rows if p in members)
        print(f"  {cid}: {hit}/{len(members)} members in top {len(rows)}", flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("method", choices=("siglip_tiles", "dinov3_patches"))
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="score only this many anchor pages (smoke test)")
    args = ap.parse_args(argv)
    if args.out.resolve().is_relative_to(args.corpus.resolve()):
        ap.error("--out is inside the corpus; proposals never write there")
    by_class = (siglip_tiles if args.method == "siglip_tiles" else dinov3_patches)(
        args.corpus, args.out, args.workers, args.limit
    )
    recall_check(by_class, args.corpus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
