"""Does a Tobacco800 mark reach UCSF pages the contamination rule still scores? (#3914)

The per-page rule (#3904) keeps a Tobacco800 class away from UCSF's *Tobacco*
pages, the same IIT-CDIP archive.  It still scores UCSF's Food, Opioids,
Chemical, Drug and Fossil Fuel pages as negatives -- and the tobacco companies
owned food companies (Philip Morris / Kraft General Foods, RJR / Nabisco), so a
company mark can plausibly sit on a Food page unlabelled.

For each Tobacco800 roster class this ranks the tier's headline pool with
SigLIP, exactly as ``eval_retrieval.py`` does, and draws one sheet: the query
crop, then the top non-positive UCSF pages from outside the Tobacco industry,
each as the top 35% of the page with its industry and rank.  Whether those pages
carry the mark is decided by looking at the sheets.

    python food_sheet.py --tier m --out /expscratch/$USER/fullmarks/food-3914
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fullmarks_config as cfg  # noqa: E402
import embed_corpus  # noqa: E402
import eval_retrieval as ev  # noqa: E402

PER_CLASS = 8
COLS = 4
CELL_W, CELL_H = 520, 240
#: Cell size when whole pages are shown, tall enough to read a mark at page scale.
FULL_W, FULL_H = 520, 700
BAND_FRAC = 0.35


def top_off_industry_pages(
    scores: dict[str, float],
    pool: set[str],
    positives: set[str],
    source_of: dict[str, str],
    industry_of: dict[str, Optional[str]],
    *,
    n: int = PER_CLASS,
) -> list[dict]:
    """The *n* highest-ranked UCSF non-Tobacco non-positives in *pool*, with their pool rank."""
    out = []
    for rank, page_id in enumerate(ev.rank_pool(scores, pool), start=1):
        if page_id in positives or source_of.get(page_id) != "ucsf":
            continue
        if industry_of.get(page_id) == "Tobacco":
            continue
        out.append(
            {"page_id": page_id, "industry": industry_of.get(page_id), "rank": rank, "score": round(scores[page_id], 4)}
        )
        if len(out) == n:
            break
    return out


def draw_sheet(
    cid: str, crop: Path, hits: list[dict], path_of: dict[str, str], out: Path, *, band: float = BAND_FRAC
) -> None:
    cell_w, cell_h = (FULL_W, FULL_H) if band >= 1 else (CELL_W, CELL_H)
    rows = (len(hits) + COLS - 1) // COLS
    sheet = Image.new("L", ((COLS + 1) * (cell_w + 8), max(rows, 1) * (cell_h + 22)), 255)
    draw = ImageDraw.Draw(sheet)
    with Image.open(crop) as img:
        img = img.convert("L")
        img.thumbnail((cell_w, cell_h))
        sheet.paste(img, (0, 20))
    draw.text((2, 3), f"{cid} (query)", fill=0)
    for i, hit in enumerate(hits):
        x = (i % COLS + 1) * (cell_w + 8)
        y = (i // COLS) * (cell_h + 22)
        with Image.open(path_of[hit["page_id"]]) as img:
            img = img.convert("L")
            img = img.crop((0, 0, img.width, int(img.height * band)))
            img.thumbnail((cell_w, cell_h))
            sheet.paste(img, (x, y + 20))
        draw.text((x + 2, y + 3), f"[{i + 1}] {hit['page_id']} {hit['industry']} rank {hit['rank']}", fill=0)
    sheet.save(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from vtscore.media import get_embedder  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--tier", default="m")
    ap.add_argument("--per-class", type=int, default=PER_CLASS)
    ap.add_argument(
        "--band", type=float, default=BAND_FRAC, help="fraction of the page shown from the top; 1 shows it whole"
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.out.resolve().is_relative_to(args.corpus.resolve()):
        ap.error("--out is inside the corpus")
    args.out.mkdir(parents=True, exist_ok=True)

    classes = {
        cid: meta
        for cid, meta in json.loads((args.corpus / "classes.json").read_text(encoding="utf-8")).items()
        if meta.get("on_roster") and meta.get("query_crop") and meta.get("source") == "tobacco800"
    }
    pages = embed_corpus.pages_for_tier(args.corpus, args.tier)
    pages_by_source: dict[str, list[str]] = {}
    industry_of: dict[str, Optional[str]] = {}
    source_of: dict[str, str] = {}
    path_of: dict[str, str] = {}
    for page in pages:
        pages_by_source.setdefault(page.source, []).append(page.page_id)
        industry_of[page.page_id] = page.meta.get("industry")
        source_of[page.page_id] = page.source
        path_of[page.page_id] = page.path

    ids, matrix = ev.read_vectors(embed_corpus.cell_path(args.tier, "siglip"), "siglip")
    emb = get_embedder("siglip")
    emb.load_models()
    record = {}
    for cid, meta in sorted(classes.items()):
        pools = ev.class_pools(meta, pages_by_source, industry_of)
        pool = pools[ev.HEADLINE_POOL]
        vec, _ = ev.embed_query(emb, Path(meta["query_crop"]))
        scores = dict(zip(ids, (matrix @ np.asarray(vec, dtype=np.float32)).tolist()))
        hits = top_off_industry_pages(scores, pool, pools["positives"], source_of, industry_of, n=args.per_class)
        name = cid.replace("/", "__") + ".png"
        draw_sheet(cid, Path(meta["query_crop"]), hits, path_of, args.out / name, band=args.band)
        in_pool = {}
        for p in pool:
            if source_of[p] == "ucsf":
                key = industry_of.get(p) or "unknown"
                in_pool[key] = in_pool.get(key, 0) + 1
        record[cid] = {"sheet": name, "n_positive": len(pools["positives"]), "ucsf_in_pool": in_pool, "hits": hits}
        print(f"{cid}: {name} top ranks {[h['rank'] for h in hits]}", flush=True)
    (args.out / "hits.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
