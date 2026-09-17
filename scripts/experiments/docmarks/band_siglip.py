"""Cluster UCSF letterhead bands with SigLIP instead of phash (#3902).

v2 found no ``phash`` threshold that turns letterhead bands into classes; #3901
then found the marks are there (157 of 320 sampled bands carry a printed mark,
repeating within an author), so the question is whether a semantic descriptor
separates what a perceptual hash of band *layout* could not.  Three steps, each
reading only the manifest and the page rasters and writing only under ``--out``:

    python band_siglip.py embed  --out <dir>          # GPU: one vector per band
    python band_siglip.py sweep  --out <dir>          # CPU: threshold sweep
    python band_siglip.py sheets --out <dir> --threshold T
    python band_siglip.py purity --out <dir> --threshold T    # large classes, 24 members each

``embed`` caches ``bands.npz`` (page ids, authors, L2-normalised vectors).
``sweep`` writes ``band_sweep.json`` shaped like ``cluster_sweep.json``: per
author and pooled, class count, largest-component share, singleton share and
usable classes (``>= MIN_INSTANCES``) at each threshold.  ``sheets`` renders one
contact sheet per author of its largest proposed classes, for looking at.

Clustering is single linkage at a cosine-distance threshold -- the same rule
``cluster_marks.single_linkage`` applies, computed as connected components of
the thresholded graph because no adjudications exist for bands to constrain it.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import docmarks_config as cfg  # noqa: E402
from sources._common import read_manifest  # noqa: E402

GRID = (0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.26, 0.30)
#: Authors #3901 found using printed artwork; the other two are typeset-only.
MARK_AUTHORS = ("LOR, LORILLARD", "PHILIP MORRIS", "BROWN & WILLIAMSON", "RJR", "AMERICAN TOBACCO", "BATCO")


def band_of(image: Any) -> Any:
    return image.crop((0, 0, image.width, int(image.height * cfg.LETTERHEAD_BAND_FRAC)))


def letterhead_pages(corpus: Path) -> list[Any]:
    return sorted(
        (p for p in read_manifest(corpus / "corpus.jsonl") if p.meta.get("letterhead_author")),
        key=lambda p: p.page_id,
    )


def embed(corpus: Path, out: Path) -> int:
    from PIL import Image

    from vtscore.media.image.embedder_siglip import ImageSiglipEmbedder  # noqa: PLC0415

    pages = letterhead_pages(corpus)
    embedder = ImageSiglipEmbedder()
    dim = embedder.embedding_dim
    vecs = np.zeros((len(pages), dim), dtype=np.float32)
    failed = 0
    t0 = time.time()
    for i, page in enumerate(pages):
        with Image.open(page.path) as img:
            vec = embedder.embed_pil_image(band_of(img.convert("RGB")))
        if vec is None:
            failed += 1  # a zero row sits at distance 1.0 from everything: a singleton
        else:
            vecs[i] = vec
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(pages)} bands, {time.time() - t0:.0f}s", flush=True)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.clip(norms, 1e-8, None)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "bands.npz",
        page_ids=np.array([p.page_id for p in pages]),
        authors=np.array([p.meta["letterhead_author"] for p in pages]),
        paths=np.array([p.path for p in pages]),
        vectors=vecs,
    )
    print(f"embedded {len(pages)} bands ({failed} declined) in {time.time() - t0:.0f}s")
    return 0


def components(vecs: np.ndarray, threshold: float) -> np.ndarray:
    """Single-linkage labels at cosine distance *threshold* (connected components)."""
    from scipy.sparse import csr_matrix  # noqa: PLC0415
    from scipy.sparse.csgraph import connected_components  # noqa: PLC0415

    sim = vecs @ vecs.T
    adj = csr_matrix(sim >= (1.0 - threshold))
    _, labels = connected_components(adj, directed=False)
    return labels


def summarise(labels: np.ndarray, threshold: float) -> dict[str, Any]:
    n = len(labels)
    sizes = Counter(labels.tolist())
    return {
        "threshold": threshold,
        "bands": n,
        "classes": len(sizes),
        "largest": max(sizes.values()),
        "largest_share": max(sizes.values()) / n,
        "singleton_share": sum(1 for v in sizes.values() if v == 1) / n,
        "usable": sum(1 for v in sizes.values() if v >= cfg.MIN_INSTANCES),
        "usable_bands": sum(v for v in sizes.values() if v >= cfg.MIN_INSTANCES),
    }


def sweep(out: Path, grid: Sequence[float]) -> int:
    data = np.load(out / "bands.npz")
    vecs, authors = data["vectors"], data["authors"]
    result: dict[str, Any] = {"backend": "siglip", "band_frac": cfg.LETTERHEAD_BAND_FRAC, "grid": list(grid)}
    groups = {a: np.flatnonzero(authors == a) for a in sorted(set(authors.tolist()))}
    groups["ALL"] = np.arange(len(authors))
    for name, idx in groups.items():
        rows = [summarise(components(vecs[idx], t), t) for t in grid]
        result[name] = rows
        print(f"== {name} ({len(idx)} bands)")
        for r in rows:
            print(
                f"  t={r['threshold']:.2f} classes={r['classes']} largest={r['largest_share']:.2f} "
                f"singletons={r['singleton_share']:.2f} usable={r['usable']} ({r['usable_bands']} bands)"
            )
    (out / "band_sweep.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


def sheets(out: Path, threshold: float, *, per_author: int = 12, per_class: int = 6) -> int:
    from PIL import Image, ImageDraw

    data = np.load(out / "bands.npz")
    vecs, authors, paths, page_ids = data["vectors"], data["authors"], data["paths"], data["page_ids"]
    rng = random.Random(3902)
    thumb_w, thumb_h = 300, 84
    manifest: dict[str, Any] = {"threshold": threshold}
    sheet_dir = out / f"sheets_t{threshold:.2f}"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    for author in MARK_AUTHORS:
        idx = np.flatnonzero(authors == author)
        labels = components(vecs[idx], threshold)
        sizes = Counter(labels.tolist())
        classes = [c for c, n in sizes.most_common() if n >= cfg.MIN_INSTANCES][:per_author]
        sheet = Image.new("L", (160 + per_class * (thumb_w + 4), len(classes) * (thumb_h + 6) + 4), 255)
        draw = ImageDraw.Draw(sheet)
        rows = []
        for r, c in enumerate(classes):
            members = [int(idx[j]) for j in np.flatnonzero(labels == c)]
            shown = rng.sample(members, min(per_class, len(members)))
            y = r * (thumb_h + 6) + 2
            draw.text((4, y + 30), f"[{r + 1}] n={len(members)}", fill=0)
            for k, m in enumerate(shown):
                with Image.open(str(paths[m])) as img:
                    band = band_of(img.convert("L"))
                    band.thumbnail((thumb_w, thumb_h))
                    sheet.paste(band, (160 + k * (thumb_w + 4), y))
            rows.append({"row": r + 1, "size": len(members), "shown": [str(page_ids[m]) for m in shown]})
        name = author.replace(",", "").replace("&", "and").replace(" ", "_").lower() + ".png"
        sheet.save(sheet_dir / name)
        usable = [n for n in sizes.values() if n >= cfg.MIN_INSTANCES]
        manifest[author] = {"sheet": name, "bands": len(idx), "usable_classes": len(usable), "rows": rows}
        print(f"{name}: {len(idx)} bands, {len(usable)} usable classes, top sizes {[sizes[c] for c in classes]}")
    (sheet_dir / "sheets.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


def purity(out: Path, threshold: float, *, min_size: int = 20, sample: int = 24, cols: int = 6) -> int:
    """One sheet per large proposed class: a random *sample* of its members, to judge purity.

    ``sheets`` shows six members a class, which says what a class *is* but not
    whether an 800-band class is one mark; this shows enough of each to count.
    """
    from PIL import Image, ImageDraw

    data = np.load(out / "bands.npz")
    vecs, authors, paths, page_ids = data["vectors"], data["authors"], data["paths"], data["page_ids"]
    rng = random.Random(39021)
    tw, th = 320, 90
    sheet_dir = out / f"purity_t{threshold:.2f}"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"threshold": threshold, "sample": sample}
    for author in MARK_AUTHORS:
        idx = np.flatnonzero(authors == author)
        labels = components(vecs[idx], threshold)
        sizes = Counter(labels.tolist())
        tag = author.replace(",", "").replace("&", "and").replace(" ", "_").lower()
        for rank, (c, n) in enumerate(sizes.most_common()):
            if n < min_size:
                break
            members = [int(idx[j]) for j in np.flatnonzero(labels == c)]
            shown = rng.sample(members, min(sample, len(members)))
            rows = (len(shown) + cols - 1) // cols
            sheet = Image.new("L", (cols * (tw + 4), rows * (th + 16)), 255)
            draw = ImageDraw.Draw(sheet)
            for k, m in enumerate(shown):
                x, y = (k % cols) * (tw + 4), (k // cols) * (th + 16)
                with Image.open(str(paths[m])) as img:
                    band = band_of(img.convert("L"))
                    band.thumbnail((tw, th))
                    sheet.paste(band, (x, y + 14))
                draw.text((x + 2, y + 1), f"[{k + 1}] {page_ids[m]}", fill=0)
            name = f"{tag}_c{rank + 1}_n{n}.png"
            sheet.save(sheet_dir / name)
            manifest[name] = [str(page_ids[m]) for m in shown]
            print(f"{name}", flush=True)
    (sheet_dir / "purity.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("embed", "sweep", "sheets", "purity"))
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--grid", default=",".join(str(g) for g in GRID))
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args(argv)
    if args.out.resolve().is_relative_to(args.corpus.resolve()):
        ap.error("--out is inside the corpus; this study never writes there")
    if args.step == "embed":
        return embed(args.corpus, args.out)
    if args.step == "sweep":
        return sweep(args.out, [float(g) for g in args.grid.split(",")])
    if args.threshold is None:
        ap.error(f"{args.step} needs --threshold")
    if args.step == "purity":
        return purity(args.out, args.threshold)
    return sheets(args.out, args.threshold)


if __name__ == "__main__":
    raise SystemExit(main())
