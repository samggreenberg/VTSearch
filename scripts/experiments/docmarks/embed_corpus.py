#!/usr/bin/env python
"""Embed a built DocMarks corpus into pile-format cells, on the GRID.

    source scripts/experiments/pile/pile_env.sh
    python embed_corpus.py --list
    python embed_corpus.py --tier s --embedders sift_vlad,siglip
    python embed_corpus.py --verify

One cell = one ``docmarks_<tier>__<embedder>.pkl`` of media dicts carrying
vectors (plus ``local_features`` for structural embedders) and **no pixels**,
written into the shared pile's ``embeddings/`` directory so studies read it the
same way they read every other cell.

Why this is not just a new entry in ``pile_config.DATASETS``: the pile builds
the full ``dataset x embedder`` cross-product, so adding DocMarks *and*
``sift_vlad`` there would silently schedule ``sift_vlad`` cells for all six
existing datasets and three deep-embedder cells for each DocMarks tier — a
dozen-odd cells nobody asked for, on a mount the playbook already describes as
chronically full.  This script reuses the pile's *format*, *location* and
*pickle IO* while keeping the cell list explicit.

Tiers are nested, so build them small-first: ``--tier s`` produces a 5k cell you
can iterate against in minutes, and ``l`` is the same corpus with more
distractors.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import docmarks_config as cfg  # noqa: E402
from sources._common import Page, read_manifest  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_PILE_DIR = _REPO / "scripts" / "experiments" / "pile"
_CALIB_DIR = _REPO / "scripts" / "experiments" / "calibration"

#: Embedders worth caching for this corpus.  ``sift_vlad`` is the shipped
#: structural embedder and the reason the corpus exists; the deep embedders are
#: the baseline every structural result is quoted against, and the hybrid arm
#: needs both in the same media dicts.
EMBEDDERS: dict[str, dict[str, Any]] = {
    "sift_vlad": {"batch": None, "structural": True},
    "siglip": {"batch": 128, "structural": False},
    "siglip2_l": {"batch": 32, "structural": False},
}


def _load_by_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cells_io() -> Any:
    """The calibration harness's pickle IO — drops bytes, keeps vectors."""
    return _load_by_path("_docmarks_cells_io", _CALIB_DIR / "_cells_io.py")


def _pile_config() -> Any:
    return _load_by_path("_docmarks_pile_config", _PILE_DIR / "pile_config.py")


def cell_name(tier: str, embedder: str) -> str:
    return f"docmarks_{tier}__{embedder}.pkl"


def cell_path(tier: str, embedder: str) -> Path:
    return _pile_config().EMBEDDINGS / cell_name(tier, embedder)


@contextmanager
def _batch_size(embedder: str) -> Iterator[None]:
    """Apply this embedder's batch size for the embed pass.

    An explicitly-set environment variable always wins: whoever set it is
    tuning for the card in front of them, and a table in this file cannot know
    what that card is.
    """
    want = EMBEDDERS.get(embedder, {}).get("batch")
    if want is None or os.environ.get("VTSEARCH_EMBED_BATCH_SIZE", "").strip():
        yield
        return
    previous = os.environ.get("VTSEARCH_EMBED_BATCH_SIZE")
    os.environ["VTSEARCH_EMBED_BATCH_SIZE"] = str(want)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("VTSEARCH_EMBED_BATCH_SIZE", None)
        else:
            os.environ["VTSEARCH_EMBED_BATCH_SIZE"] = previous


def tiers_up_to(tier: str) -> set[str]:
    """Tier *tier* and every smaller one — tiers nest, so a cell is cumulative."""
    order = list(cfg.TIER_ORDER)
    return set(order[: order.index(tier) + 1])


def pages_for_tier(corpus: Path, tier: str) -> list[Page]:
    wanted = tiers_up_to(tier)
    return [p for p in read_manifest(corpus / "corpus.jsonl") if p.meta.get("tier") in wanted]


def load_medias(pages: Sequence[Page], classes: dict[str, Any], embedder: str) -> dict[int, dict]:
    """Turn manifest pages into the media dicts the embedding stage expects.

    ``regions`` carries every ground-truth mark, so a region-voting arm can drag
    the real box rather than the whole page.  Weak (boxless) marks are recorded
    in ``categories`` but contribute no region — a zero-area box would be
    indistinguishable from a real one downstream, which is exactly the
    distinction the corpus is built to preserve.
    """
    medias: dict[int, dict] = {}
    for index, page in enumerate(sorted(pages, key=lambda p: p.page_id)):
        regions = []
        categories = []
        for mark in page.marks:
            if not mark.class_id:
                continue
            categories.append(mark.class_id)
            if mark.area() > 0:
                x, y, w, h = mark.box
                regions.append(
                    {
                        "label": mark.class_id,
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        "provenance": mark.provenance,
                    }
                )
        ordered = sorted(dict.fromkeys(categories))
        medias[index] = {
            "id": index,
            "media_type": "image",
            "embedder": embedder,
            "duration": 0,
            "file_size": 0,
            "md5": "",
            "embeddings": {},
            "media_bytes": Path(page.path).read_bytes(),
            "media_string": None,
            "filename": Path(page.path).name,
            "category": ordered[0] if ordered else "",
            "categories": ordered,
            "regions": regions,
            "origin": {"importer": "docmarks", "params": {"embedder": embedder, "page_id": page.page_id}},
            "origin_name": page.page_id,
            # Carried through so a study can filter without re-reading the
            # manifest: which stratum, which tier, and how trustworthy the
            # labels on this page are.
            "docmarks": {
                "source": page.source,
                "tier": page.meta.get("tier"),
                "provenances": sorted({m.provenance for m in page.marks}),
                "industry": page.meta.get("industry"),
                "decade": page.meta.get("decade"),
            },
        }
    return medias


def build_cell(corpus: Path, tier: str, embedder: str, *, force: bool = False) -> dict[str, Any]:
    from vtscore.datasets.stages.embedding import embed_missing  # noqa: PLC0415

    out = cell_path(tier, embedder)
    if out.exists() and not force:
        print(f"skip docmarks_{tier} x {embedder} (exists: {out.name})")
        return {"tier": tier, "embedder": embedder, "status": "exists"}

    classes_path = corpus / "classes.json"
    classes = json.loads(classes_path.read_text(encoding="utf-8")) if classes_path.exists() else {}

    t0 = time.time()
    pages = pages_for_tier(corpus, tier)
    if not pages:
        raise SystemExit(f"no pages at tier {tier!r} in {corpus} — was the corpus built with this tier?")
    medias = load_medias(pages, classes, embedder)
    print(f"=== docmarks_{tier} x {embedder}: {len(medias)} page(s) loaded in {time.time() - t0:.0f}s")

    t1 = time.time()
    with _batch_size(embedder):
        embed_missing(medias, embedder)
    embed_s = time.time() - t1

    out.parent.mkdir(parents=True, exist_ok=True)
    nbytes = _cells_io().dump_medias(medias, out)
    n_local = sum(1 for m in medias.values() if m.get("local_features") is not None)
    print(
        f"  wrote {out.name}: {nbytes / 1e6:.0f} MB, {len(medias)} medias, "
        f"local_features {n_local}/{len(medias)}, embed {embed_s:.0f}s"
    )
    return {
        "tier": tier,
        "embedder": embedder,
        "status": "built",
        "n_medias": len(medias),
        "n_local_features": n_local,
        "megabytes": round(nbytes / 1e6, 1),
        "embed_seconds": round(embed_s, 1),
    }


def verify(corpus: Path) -> int:
    """Load every present cell and check it is usable.  Returns an exit code."""
    io = _cells_io()
    bad = 0
    for tier in cfg.TIER_ORDER:
        for embedder in EMBEDDERS:
            path = cell_path(tier, embedder)
            if not path.exists():
                continue
            try:
                medias = io.load_medias(path)
            except Exception as exc:  # noqa: BLE001 - verify reports, never raises
                print(f"  BROKEN {path.name}: {type(exc).__name__}: {exc}")
                bad += 1
                continue
            vectors = sum(1 for m in medias.values() if m.get("embeddings"))
            labelled = sum(1 for m in medias.values() if m.get("categories"))
            status = "ok" if vectors == len(medias) else f"MISSING {len(medias) - vectors} vector(s)"
            print(f"  {path.name}: {len(medias)} medias, {labelled} labelled, {status}")
            bad += status != "ok"
    return 1 if bad else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--tier", default="s", choices=cfg.TIER_ORDER)
    ap.add_argument("--embedders", default="sift_vlad,siglip")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true", help="show which cells exist, then exit")
    ap.add_argument("--verify", action="store_true", help="load every present cell and check it, then exit")
    args = ap.parse_args(argv)

    if args.list:
        for tier in cfg.TIER_ORDER:
            for embedder in EMBEDDERS:
                path = cell_path(tier, embedder)
                mark = f"{path.stat().st_size / 1e6:>8.0f} MB" if path.exists() else "        --"
                print(f"  {mark}  {path.name}")
        return 0

    if args.verify:
        return verify(args.corpus)

    requested = [e.strip() for e in args.embedders.split(",") if e.strip()]
    unknown = set(requested) - set(EMBEDDERS)
    if unknown:
        ap.error(f"unknown embedder(s): {sorted(unknown)}; known: {sorted(EMBEDDERS)}")

    summaries = [build_cell(args.corpus, args.tier, e, force=args.force) for e in requested]
    built = [s for s in summaries if s["status"] == "built"]
    print(f"\n{len(built)} cell(s) built, {len(summaries) - len(built)} already present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
