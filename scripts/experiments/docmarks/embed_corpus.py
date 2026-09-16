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

Cells are built **in chunks** (``--chunk``, default
:data:`DEFAULT_CHUNK`): a chunk's pages are read, embedded, written to the cell
and dropped before the next chunk is read, so peak memory is set by the chunk
size rather than by the tier.  Straight-through, tier ``s`` peaked at 34.5 GB of
RSS for 5,000 pages and nothing here could have built tier ``l`` at all (#3842).
It remains slow -- ``sift_vlad`` is ~1.3 pages/s, so tier ``l`` is ~43 h -- but
slow is a schedule and out-of-memory is not.
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

#: Pages held in memory at once by a streamed build.  Tier ``s`` peaked at
#: **34.5 GB of RSS for 5,000 pages** (#3842) -- ~7 MB per page resident across
#: the raster bytes, the decoded image and ``sift_vlad``'s per-image features --
#: so a 1,000-page chunk is roughly a 7 GB working set whatever the tier, which
#: is what makes tier ``l`` a scheduling problem rather than an impossible one.
#: Smaller trades a little write overhead for headroom on a shared node; ``0``
#: disables chunking and restores the one-shot cell.
DEFAULT_CHUNK = 1000


def _load_by_path(name: str, path: Path) -> Any:
    """Import a module from a file, with its own directory importable.

    The directory matters: these modules are written to be run from inside
    their own experiment directory and import their siblings by bare name.
    ``_cells_io`` does ``from _cells_paths import ...`` — the discovery half of
    itself, split out so the pandas-free figure scripts can share it — and
    loading it by path alone leaves that import with nowhere to resolve from.
    """
    directory = str(path.parent)
    if directory not in sys.path:
        sys.path.append(directory)
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


def labels_for(page: Page) -> tuple[list[str], list[dict[str, Any]]]:
    """``(categories, regions)`` for one page — everything a cell says about labels.

    Factored out so ``load_medias`` and ``relabel`` cannot drift: one derives
    the labels while embedding, the other rewrites them afterwards, and a cell
    whose two paths disagreed would be wrong in a way nothing checks.
    """
    regions: list[dict[str, Any]] = []
    categories: list[str] = []
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
    return sorted(dict.fromkeys(categories)), regions


def relabel(corpus: Path, *, apply: bool = False) -> int:
    """Rewrite the labels in every existing cell from the current manifest.

    "Embedding comes last, because the cells carry the labels" is the rule
    #3343 states, and it makes any later audit verdict cost a full re-embed —
    ~12 h here for a change to fifteen pages. It costs that only because the
    labels and the vectors are written in one pass; they do not *depend* on
    each other. ``embed_missing`` embeds pixels, so a merge, a split or a
    membership rejection changes what a page is called and nothing about its
    vector.

    So this rewrites `category`, `categories` and each region's `label` in
    place, from the manifest, and leaves every vector untouched. It is the
    repair for a verdict that lands after a cell is built — not a licence to
    embed first, because a membership *rejection* also changes which pages are
    positives, and only the manifest knows that.

    The real case: the corpus owner countersigned the v3 roster and overturned
    one pair, merging `logo_bad45f00_1` into `logo_afm90c00-first_1_0`. Fifteen
    pages in tiers already built were left naming a class that no longer exists.
    """
    io = _cells_io()
    classes_path = corpus / "classes.json"
    known = set(json.loads(classes_path.read_text(encoding="utf-8"))) if classes_path.exists() else set()
    touched = 0
    for tier in cfg.TIER_ORDER:
        pages = None
        for embedder in EMBEDDERS:
            path = cell_path(tier, embedder)
            if not path.exists():
                continue
            if pages is None:
                pages = {p.page_id: p for p in pages_for_tier(corpus, tier)}
            medias = io.load_medias(path)
            changed = 0
            orphans = 0
            for media in medias.values():
                page = pages.get(media.get("origin_name"))
                if page is None:
                    orphans += 1
                    continue
                categories, regions = labels_for(page)
                before = (media.get("category"), media.get("categories"), media.get("regions"))
                after = (categories[0] if categories else "", categories, regions)
                if before != after:
                    media["category"], media["categories"], media["regions"] = after
                    changed += 1
            # Labels naming a class that is not in `classes.json` are EXPECTED
            # and are not damage: under `--roster` only the chosen classes are
            # admitted, while the manifest keeps every candidate id it derived,
            # so a distractor page carrying an unrostered mark says so. Counted
            # rather than hidden, because an eval that grouped a cell by
            # `categories` alone would silently treat those as eval classes --
            # which is the whole distinction the roster exists to draw.
            offroster = sum(1 for m in medias.values() for c in (m.get("categories") or []) if c not in known)
            print(
                f"  {path.name}: {len(medias)} medias, {changed} relabelled"
                + (f", {orphans} not in the manifest" if orphans else "")
                + (f", {offroster} label(s) on candidate classes not on the roster" if offroster else "")
            )
            touched += changed
            if apply and changed:
                io.dump_medias(medias, path)
    print(f"\n{touched} media(s) relabelled" + ("" if apply else " — dry run, pass --apply to write"))
    return 0


def repair(corpus: Path, *, apply: bool = False) -> int:
    """Re-embed only the medias an existing cell holds no vector for.

    A page whose pixels will not decode costs the cell a *vector* but not a
    *row*: ``build_cell`` writes every media it loaded, so one undecodable PNG
    leaves a media with an empty ``embeddings``, and ``verify`` reports
    ``MISSING 1 vector(s)`` for as long as that cell exists.

    The repair for the *page* is to re-render it.  The repair for the *cell*
    would otherwise be a full rebuild -- 11 h of SIFT over 50,000 pages to
    obtain one vector -- because ``build_cell`` is all-or-nothing.
    ``embed_missing`` is not: it embeds exactly the medias with no vector under
    this embedder's key, so re-reading those few pages and calling it costs one
    image rather than the tier.

    Bytes are re-read for the missing medias *only*.  ``dump_medias`` drops
    ``media_bytes``, so a cell knows its pages by ``origin_name`` and nothing
    else -- and re-reading all 50,000 to repair one would cost exactly the
    memory the thin pickle exists to avoid.

    The real case: ``ucsf/qkmg0227#0`` was truncated on write during the UCSF
    pull and decoded to half a page of black.  Re-rendered from the cached PDF,
    it needed a vector in two tier-``m`` cells that cost 11 h to build.
    """
    from vtscore.datasets.stages.embedding import embed_missing  # noqa: PLC0415

    io = _cells_io()
    unrepaired = 0
    for tier in cfg.TIER_ORDER:
        pages = None
        for embedder in EMBEDDERS:
            path = cell_path(tier, embedder)
            if not path.exists():
                continue
            medias = io.load_medias(path)
            missing = {cid: m for cid, m in medias.items() if not m.get("embeddings")}
            if not missing:
                print(f"  {path.name}: {len(medias)} medias, no vector missing")
                continue
            if pages is None:
                pages = {p.page_id: p for p in pages_for_tier(corpus, tier)}
            blocked: list[str] = []
            for media in missing.values():
                page = pages.get(media.get("origin_name"))
                if page is None:
                    blocked.append(f"{media.get('origin_name')}: not in the manifest")
                    continue
                try:
                    media["media_bytes"] = Path(page.path).read_bytes()
                except OSError as exc:  # noqa: PERF203 - one page, reported not raised
                    blocked.append(f"{page.page_id}: {type(exc).__name__}")
            loadable = {cid: m for cid, m in missing.items() if m.get("media_bytes")}
            if apply and loadable:
                with _batch_size(embedder):
                    embed_missing(loadable, embedder)
            repaired = sum(1 for m in loadable.values() if m.get("embeddings"))
            # Drop the bytes again whatever happened: `dump_medias` would strip
            # them, but this dict stays live for the rest of the loop.
            for media in missing.values():
                media.pop("media_bytes", None)
            left = len(missing) - repaired
            detail = f"{repaired} repaired" if apply else f"{len(loadable)} re-readable"
            print(
                f"  {path.name}: {len(medias)} medias, {len(missing)} without a vector, {detail}"
                + (f", {left} still missing" if apply and left else "")
                + (f" [{'; '.join(blocked)}]" if blocked else "")
            )
            if apply:
                unrepaired += left
                if repaired:
                    io.dump_medias(medias, path)
    print("" if apply else "\ndry run -- pass --force to write")
    return 1 if unrepaired else 0


def load_medias(
    pages: Sequence[Page], classes: dict[str, Any], embedder: str, *, start_index: int = 0
) -> dict[int, dict]:
    """Turn manifest pages into the media dicts the embedding stage expects.

    ``regions`` carries every located mark, so a region-voting arm can drag the
    real box rather than the whole page.  A region records the ``provenance`` it
    came with, so a downstream arm can tell an adjudicated box from a coarse
    letterhead band; unlocated marks contribute no region at all, because a
    zero-area box would be indistinguishable from a real one and that is exactly
    the distinction the corpus exists to preserve.

    *start_index* is the id the first media takes, so :func:`build_cell` can
    call this once per chunk and still number the cell continuously.  Ids must
    not repeat across chunks -- ``CellWriter`` refuses a cell where they do,
    because merging the chunks would silently lose the repeat rather than fail.

    The sort is **per call**, so a chunked build must slice an already-sorted
    page list: sorting each chunk alone would order the chunk but not the cell.
    :func:`build_cell` sorts once and slices; the sort here is then a no-op and
    is kept because a single-chunk caller still needs it.
    """
    medias: dict[int, dict] = {}
    for offset, page in enumerate(sorted(pages, key=lambda p: p.page_id)):
        index = start_index + offset
        ordered, regions = labels_for(page)
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
                "year": page.meta.get("year"),
            },
        }
    return medias


def build_cell(
    corpus: Path, tier: str, embedder: str, *, force: bool = False, chunk: int = DEFAULT_CHUNK
) -> dict[str, Any]:
    """Embed one tier under one embedder and write its cell.

    **Streamed, in chunks of *chunk* pages** (``0`` disables it and writes the
    old one-shot cell).  The straight-through version -- load every page, embed
    them all, pickle the dict -- is what made tier ``l`` unbuildable rather than
    merely slow (#3842): tier ``s`` peaked at **34.5 GB of RSS for 5,000
    pages**, because ``load_medias`` holds the raster bytes of every page in the
    tier while ``embed_missing`` accumulates ``local_features`` beside them, and
    neither is released until ``dump_medias`` writes at the very end.  Neither
    term extrapolates: at 200,000 pages the bytes alone are ~50 GB and the
    ``sift_vlad`` features another ~34 GB (169 KB/page, measured).

    Chunking retires both.  Each chunk's bytes are read, embedded, written and
    dropped before the next is read, so peak memory is set by *chunk* rather
    than by the tier -- and the cell is appended to rather than assembled, so
    the finished 34 GB of it is never resident either.  What it does **not**
    change is the time: ``sift_vlad`` is ~1.3 pages/s whatever the chunk size,
    so tier ``l`` is still ~43 h of CPU.  That is a wall to schedule around,
    not one that stops the job starting.

    A chunked run is resumable only at the granularity of the whole cell: the
    writer renames its ``.part`` into place on clean exit, so a job that dies at
    hour 40 leaves nothing for ``--verify`` to mistake for a finished cell.
    """
    from vtscore.datasets.stages.embedding import embed_missing  # noqa: PLC0415

    out = cell_path(tier, embedder)
    if out.exists() and not force:
        print(f"skip docmarks_{tier} x {embedder} (exists: {out.name})")
        return {"tier": tier, "embedder": embedder, "status": "exists"}

    classes_path = corpus / "classes.json"
    classes = json.loads(classes_path.read_text(encoding="utf-8")) if classes_path.exists() else {}

    pages = pages_for_tier(corpus, tier)
    if not pages:
        raise SystemExit(f"no pages at tier {tier!r} in {corpus} — was the corpus built with this tier?")
    # Sorted ONCE, here: `load_medias` sorts what it is given, which orders a
    # chunk but not the cell.  Slicing an already-sorted list is what makes the
    # chunked cell identical to the one-shot one page for page.
    pages = sorted(pages, key=lambda p: p.page_id)
    size = chunk if chunk and chunk > 0 else len(pages)
    n_chunks = (len(pages) + size - 1) // size
    print(f"=== docmarks_{tier} x {embedder}: {len(pages)} page(s) in {n_chunks} chunk(s) of {size}")

    io = _cells_io()
    load_s = embed_s = 0.0
    n_local = 0
    t0 = time.time()
    with io.CellWriter(out) as writer:
        for start in range(0, len(pages), size):
            batch = pages[start : start + size]
            t = time.time()
            medias = load_medias(batch, classes, embedder, start_index=start)
            load_s += time.time() - t

            t = time.time()
            with _batch_size(embedder):
                embed_missing(medias, embedder)
            embed_s += time.time() - t

            n_local += sum(1 for m in medias.values() if m.get("local_features") is not None)
            writer.write(medias)
            # The chunk dies here, bytes and features together.  Holding it
            # while the next one loads is the whole of what this costs.
            medias.clear()
            del medias
            if n_chunks > 1:
                done = min(start + size, len(pages))
                print(f"  {done}/{len(pages)} pages, load {load_s:.0f}s, embed {embed_s:.0f}s", flush=True)

    print(
        f"  wrote {out.name}: {writer.nbytes / 1e6:.0f} MB, {writer.n_medias} medias in "
        f"{writer.n_chunks} chunk(s), local_features {n_local}/{writer.n_medias}, "
        f"load {load_s:.0f}s, embed {embed_s:.0f}s, total {time.time() - t0:.0f}s"
    )
    return {
        "tier": tier,
        "embedder": embedder,
        "status": "built",
        "n_medias": writer.n_medias,
        "n_chunks": writer.n_chunks,
        "n_local_features": n_local,
        "megabytes": round(writer.nbytes / 1e6, 1),
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
            # Streamed rather than loaded: verifying is a pure visit, and a
            # tier-`l` `sift_vlad` cell is ~34 GB, so the check that a cell is
            # usable must not be the thing that cannot hold it.
            try:
                n = vectors = labelled = 0
                for _cid, media in io.iter_medias(path):
                    n += 1
                    vectors += bool(media.get("embeddings"))
                    labelled += bool(media.get("categories"))
            except Exception as exc:  # noqa: BLE001 - verify reports, never raises
                print(f"  BROKEN {path.name}: {type(exc).__name__}: {exc}")
                bad += 1
                continue
            status = "ok" if vectors == n else f"MISSING {n - vectors} vector(s)"
            print(f"  {path.name}: {n} medias, {labelled} labelled, {status}")
            bad += status != "ok"
    return 1 if bad else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--tier", default="s", choices=cfg.TIER_ORDER)
    ap.add_argument("--embedders", default="sift_vlad,siglip")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--chunk",
        type=int,
        default=DEFAULT_CHUNK,
        metavar="N",
        help=f"pages held in memory at once (default {DEFAULT_CHUNK}); 0 builds the cell in one "
        "piece, which is what tier `l` cannot do",
    )
    ap.add_argument("--list", action="store_true", help="show which cells exist, then exit")
    ap.add_argument("--verify", action="store_true", help="load every present cell and check it, then exit")
    ap.add_argument(
        "--relabel",
        action="store_true",
        help="rewrite the labels in every existing cell from the current manifest, leaving the "
        "vectors alone; the repair for an audit verdict that lands after a cell was built",
    )
    ap.add_argument(
        "--repair",
        action="store_true",
        help="re-embed only the medias an existing cell has no vector for; the repair for a "
        "page that would not decode when the cell was built",
    )
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

    if args.relabel:
        return relabel(args.corpus, apply=args.force)

    if args.repair:
        return repair(args.corpus, apply=args.force)

    requested = [e.strip() for e in args.embedders.split(",") if e.strip()]
    unknown = set(requested) - set(EMBEDDERS)
    if unknown:
        ap.error(f"unknown embedder(s): {sorted(unknown)}; known: {sorted(EMBEDDERS)}")

    # Resolve the write path BEFORE embedding anything.  This is a preflight
    # rather than a tidiness: the cost of a cell is entirely in the embedding,
    # and `dump_medias` is the last line of it, so anything wrong with the
    # serializer surfaces only after the whole bill has been paid.  It did:
    # `_cells_io` imports a sibling by bare name, loading it by path left that
    # import unresolvable, and docmarks_s x sift_vlad died on
    # ModuleNotFoundError after **2h16m** of SIFT on 5,000 pages (#3343).
    # Nothing had run this path before, because stage 5 had never been reached.
    try:
        io = _cells_io()
        missing = [name for name in ("CellWriter", "iter_medias") if not hasattr(io, name)]
        if missing:
            ap.error(f"{_CALIB_DIR / '_cells_io.py'} has no {', '.join(missing)}; cells cannot be written")
        _pile_config().EMBEDDINGS.mkdir(parents=True, exist_ok=True)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - the point is to report, early, whatever broke
        ap.error(f"cannot write cells: {type(exc).__name__}: {exc}")

    summaries = [build_cell(args.corpus, args.tier, e, force=args.force, chunk=args.chunk) for e in requested]
    built = [s for s in summaries if s["status"] == "built"]
    print(f"\n{len(built)} cell(s) built, {len(summaries) - len(built)} already present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
