#!/usr/bin/env python
"""Assemble the DocMarks corpus: stamps and logos in scanned documents.

    python build_corpus.py --probe                      # what can I reach?
    python build_corpus.py --sources spods --limit 40   # small real build
    python build_corpus.py --survival                   # class counts vs threshold
    python build_corpus.py                              # the whole thing

Outputs, under ``docmarks_config.OUT``:

    corpus.jsonl      one record per page: path, size, marks, provenance, tier
    classes.json      the class inventory, with eligibility and audit slots
    queries/          one query crop per admitted class
    build_report.json counts, warnings, the survival curve, what was dropped

The three strata (anchor / haystack / synth) live in one manifest with **nested
tiers**, so ``docmarks_s`` and ``docmarks_l`` share class ids and a result on
one is comparable to a result on the other.

Read ``README.md`` before changing the contamination rules; they are the part of
this script that is easy to "simplify" and expensive to get wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import docmarks_config as cfg  # noqa: E402
from sources import _common  # noqa: E402
from sources._common import Mark, Page  # noqa: E402

ALL_SOURCES = ("spods", "staver", "tobacco800", "ucsf", "synth")

#: Mark kinds that may become query classes.  Signatures are excluded on
#: purpose: a handwritten signature is a different mark every time it is made,
#: so it is not an instance in the sense structural search means.  They stay in
#: the manifest as a documented negative control.
QUERYABLE_KINDS = ("logo", "stamp")


# --------------------------------------------------------------------------
# Class inventory
# --------------------------------------------------------------------------


def class_inventory(pages: Sequence[Page]) -> dict[str, list[tuple[int, int]]]:
    """``{class_id: [(page index, mark index), ...]}`` over every labelled mark."""
    inv: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for pi, page in enumerate(pages):
        for mi, mark in enumerate(page.marks):
            if mark.class_id:
                inv[mark.class_id].append((pi, mi))
    return dict(inv)


def survival_curve(inventory: dict[str, list[tuple[int, int]]], thresholds: Iterable[int]) -> dict[int, int]:
    """How many classes survive each ``min-instances`` bar.

    Printed on every build because the bar is the single most consequential
    knob in the corpus and the right value is a property of the data, not a
    preference.  Tobacco800's published protocol uses >=2, which cannot support
    a train-and-search eval at all.
    """
    sizes = [len(v) for v in inventory.values()]
    return {t: sum(1 for s in sizes if s >= t) for t in thresholds}


def admit_classes(
    pages: Sequence[Page],
    inventory: dict[str, list[tuple[int, int]]],
    *,
    min_instances: int,
    min_mark_px: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Split the inventory into admitted query classes and rejects-with-reasons.

    A class is admitted when it has enough instances *and* its typical instance
    is big enough to be findable.  The size bar matters as much as the count
    bar: the 2026-07-13 study measured a hard floor near 32 px below which no
    structural pipeline recovers anything, so a class built from sub-floor marks
    measures the floor rather than the method, and averaging it in with the rest
    drags every aggregate toward zero for a reason that has nothing to do with
    what is being compared.
    """
    admitted: dict[str, dict[str, Any]] = {}
    rejected: dict[str, str] = {}

    for class_id, refs in sorted(inventory.items()):
        marks = [pages[pi].marks[mi] for pi, mi in refs]
        kinds = {m.kind for m in marks}
        if not kinds & set(QUERYABLE_KINDS):
            rejected[class_id] = f"kind {sorted(kinds)} is not queryable"
            continue
        if len(refs) < min_instances:
            rejected[class_id] = f"{len(refs)} instance(s) < min_instances={min_instances}"
            continue

        boxed = [m for m in marks if m.area() > 0]
        provenances = {m.provenance for m in marks}
        median_px: Optional[int] = None
        if boxed:
            sides = sorted(m.longest_side() for m in boxed)
            median_px = sides[len(sides) // 2]
            if median_px < min_mark_px:
                rejected[class_id] = f"median mark {median_px}px < min_mark_px={min_mark_px}"
                continue
        elif provenances != {"weak"}:
            rejected[class_id] = "no boxed instances and not a weak-label class"
            continue

        source = class_id.split("/", 1)[0]
        admitted[class_id] = {
            "class_id": class_id,
            "source": source,
            "kind": sorted(kinds)[0],
            "n_instances": len(refs),
            "median_mark_px": median_px,
            "provenance": sorted(provenances),
            "page_ids": sorted(pages[pi].page_id for pi, _ in refs),
            "eligible_distractor_sources": sorted(
                s for s in ALL_SOURCES if cfg.eligible_distractor(source, s)
            ),
            # Filled by the human passes; see make_audit_slate.py.
            "audit": {"distinctive": None, "cluster_ok": None, "letterhead_precision": None, "notes": ""},
        }
    return admitted, rejected


# --------------------------------------------------------------------------
# Tiers
# --------------------------------------------------------------------------


def assign_tiers(
    pages: Sequence[Page],
    admitted: dict[str, dict[str, Any]],
    *,
    tiers: dict[str, int],
    tier_order: Sequence[str],
    salt: str,
    pinned_cutoffs: Optional[dict[str, float]] = None,
) -> tuple[dict[str, str], dict[str, float]]:
    """``({page_id: smallest tier containing it}, {tier: rank cutoff})``.

    Pages carrying an admitted class are in every tier: a tier that keeps 3 of a
    class's 30 instances does not measure that class more cheaply, it measures a
    different and much harder problem.  Distractors get a stable hash rank in
    ``[0, 1)`` and tiers are prefixes of that order, so ``s`` is always a subset
    of ``m`` is always a subset of ``l``.

    Two different stability promises are on offer here, and they genuinely
    conflict — you cannot both hit an exact page budget and keep membership
    fixed when the source pool changes size:

    * **Within a build** (the default): tiers hit their budgets exactly and are
      nested.  This is what makes "run it on ``s`` first, then on ``l``" work
      without a rebuild.
    * **Across builds** (``pinned_cutoffs``): tier membership is defined by an
      absolute rank threshold carried over from an earlier build, so adding
      pages to the source pool cannot evict a page from a tier it was already
      in.  Budgets then drift with the pool, which is the price.

    Every build records the cutoffs it used in ``build_report.json``; pass them
    back with ``--pin-tiers`` when a later build must stay comparable to an
    earlier one.  Without that, a build over a different page set is a new
    corpus version and should be named as one.
    """
    positive_pages: set[str] = set()
    for meta in admitted.values():
        positive_pages.update(meta["page_ids"])

    ranked = sorted(
        ((_common.stable_rank(p.page_id, salt), p.page_id) for p in pages if p.page_id not in positive_pages),
    )

    out: dict[str, str] = {pid: tier_order[0] for pid in positive_pages}
    cutoffs: dict[str, float] = {}
    n_positive = len(positive_pages)

    for tier in tier_order:
        if pinned_cutoffs and tier in pinned_cutoffs:
            cutoff = pinned_cutoffs[tier]
            selected = [pid for rank, pid in ranked if rank < cutoff]
        else:
            budget = max(0, tiers[tier] - n_positive)
            selected = [pid for _rank, pid in ranked[:budget]]
            # The cutoff sits just past the last selected rank, so replaying it
            # on this same pool reproduces this same selection exactly.
            cutoff = ranked[budget - 1][0] + 1e-12 if 0 < budget <= len(ranked) else 1.0
        cutoffs[tier] = cutoff
        for pid in selected:
            out.setdefault(pid, tier)

    # Anything past the largest tier is excluded from the corpus entirely.
    return out, cutoffs


# --------------------------------------------------------------------------
# Query crops
# --------------------------------------------------------------------------


def write_query_crops(
    pages: Sequence[Page],
    inventory: dict[str, list[tuple[int, int]]],
    admitted: dict[str, dict[str, Any]],
    out_dir: Path,
) -> list[str]:
    """One query crop per admitted class: its largest boxed instance.

    Largest, because the prior study measured a 2.2x AP advantage for a clean
    canonical query over a crop of a small in-scene instance — the query is the
    one place where more pixels are free.

    Weak-label classes have no box and therefore get no crop; they are returned
    as the list of classes still owing a hand-drawn query.
    """
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    needs_hand_crop: list[str] = []

    for class_id, meta in sorted(admitted.items()):
        refs = inventory[class_id]
        boxed = [(pi, mi) for pi, mi in refs if pages[pi].marks[mi].area() > 0]
        if not boxed:
            needs_hand_crop.append(class_id)
            continue
        pi, mi = max(boxed, key=lambda r: pages[r[0]].marks[r[1]].area())
        mark = pages[pi].marks[mi]
        x, y, w, h = mark.box
        dest = out_dir / f"{class_id.replace('/', '__')}.png"
        if not dest.exists():
            with Image.open(pages[pi].path) as im:
                im.convert("RGB").crop((x, y, x + w, y + h)).save(dest)
        meta["query_crop"] = str(dest)
        meta["query_page_id"] = pages[pi].page_id
    return needs_hand_crop


# --------------------------------------------------------------------------
# Source loading
# --------------------------------------------------------------------------


def load_anchor_sources(
    selected: Sequence[str],
    raw: Path,
    *,
    limit: Optional[int],
    warnings: list[str],
) -> list[Page]:
    """Fetch and parse the real-ground-truth sources."""
    pages: list[Page] = []

    if "spods" in selected:
        from sources import spods

        unpacked = spods.fetch(raw)
        pages.extend(spods.build_pages(unpacked, min_area_frac=cfg.MIN_MARK_AREA_FRAC, limit=limit))

    if "staver" in selected:
        from sources import staver

        unpacked = staver.fetch(raw)
        got, warns = staver.build_pages(unpacked, min_area_frac=cfg.MIN_MARK_AREA_FRAC, limit=limit)
        pages.extend(got)
        warnings.extend(warns)

    if "tobacco800" in selected:
        from sources import tobacco800

        unpacked = tobacco800.fetch(raw)
        got, warns = tobacco800.build_pages(unpacked, limit=limit)
        pages.extend(got)
        warnings.extend(warns)

    return pages


def load_ucsf(
    raw: Path,
    out_images: Path,
    *,
    distractor_budget: int,
    letterhead_per_author: int,
    split_by_decade: bool,
    warnings: list[str],
) -> list[Page]:
    """Pull the haystack and the weakly-labelled letterhead classes."""
    from sources import ucsf

    failures: list[str] = []

    def note(doc_id: str, exc: Exception) -> None:
        failures.append(f"{doc_id}: {type(exc).__name__}")

    pages: list[Page] = []

    for author in cfg.UCSF_LETTERHEAD_AUTHORS:
        if letterhead_per_author <= 0:
            break
        query = ucsf.build_query(author=author, doc_type="letter", max_pages=1)
        docs = list(ucsf.solr_docs(query, limit=letterhead_per_author))
        if not docs:
            warnings.append(f"ucsf: author {author!r} returned no documents")
            continue
        pages.extend(
            ucsf.fetch_and_render(
                docs,
                raw,
                out_images / "ucsf",
                letterhead_author=author,
                split_by_decade=split_by_decade,
                on_error=note,
            )
        )

    per_industry = max(1, distractor_budget // max(1, len(cfg.UCSF_INDUSTRIES)))
    for industry in cfg.UCSF_INDUSTRIES:
        query = ucsf.build_query(industry=industry, doc_type=None, max_pages=1)
        docs = list(ucsf.solr_docs(query, limit=per_industry))
        pages.extend(ucsf.fetch_and_render(docs, raw, out_images / "ucsf", on_error=note))

    if failures:
        warnings.append(f"ucsf: skipped {len(failures)} document(s) that failed to download or render")
    return pages


# --------------------------------------------------------------------------
# Probe
# --------------------------------------------------------------------------


def probe(raw: Path) -> int:
    """Report what each source can currently be reached and unpacked from.

    Run this before a grid job.  Every source here has a different failure mode
    — a decommissioned hostname, a missing Kaggle token, an absent RAR extractor
    — and finding out which one applies costs seconds now and an overnight queue
    slot later.
    """
    import requests

    from sources import spods, staver, tobacco800, ucsf

    ok = True
    print("DocMarks source probe\n")

    try:
        head = requests.head(spods.SPODS_URL, timeout=30, allow_redirects=True)
        size = int(head.headers.get("content-length", 0))
        status = "OK" if head.status_code == 200 else f"HTTP {head.status_code}"
        print(f"  spods       {status:<12} {size / 1e9:.2f} GB  {spods.SPODS_URL}")
        ok &= head.status_code == 200
    except Exception as exc:  # noqa: BLE001 - a probe reports, it does not raise
        print(f"  spods       UNREACHABLE  {type(exc).__name__}: {exc}")
        ok = False

    for name, slug in (("staver", staver.KAGGLE_SLUG), ("tobacco800", tobacco800.KAGGLE_SLUG)):
        try:
            _common.kaggle_download(slug, raw / f"_probe_{name}")
            print(f"  {name:<11} OK           kaggle:{slug}")
        except _common.FetchError as exc:
            print(f"  {name:<11} BLOCKED      {exc}")
            ok = False

    try:
        n = ucsf.count(ucsf.build_query(industry="Tobacco", doc_type="letter", max_pages=1))
        print(f"  ucsf        OK           {n:,} single-page tobacco letters")
    except Exception as exc:  # noqa: BLE001
        print(f"  ucsf        UNREACHABLE  {type(exc).__name__}: {exc}")
        ok = False

    print("\n  rar extractor:", end=" ")
    import shutil

    found = [t for t in ("bsdtar", "7z", "unar", "unrar") if shutil.which(t)]
    print(", ".join(found) if found else "NONE FOUND (needed for SPODS)")
    ok &= bool(found)

    print("\n" + ("probe passed" if ok else "probe FAILED — see above"))
    return 0 if ok else 1


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:  # noqa: C901
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default=",".join(ALL_SOURCES), help="comma-separated subset of " + ",".join(ALL_SOURCES))
    ap.add_argument("--raw", type=Path, default=cfg.RAW)
    ap.add_argument("--out", type=Path, default=cfg.OUT)
    ap.add_argument("--limit", type=int, default=None, help="cap pages per anchor source (smoke builds)")
    ap.add_argument("--min-instances", type=int, default=cfg.MIN_INSTANCES)
    ap.add_argument("--min-mark-px", type=int, default=cfg.MIN_MARK_PX)
    ap.add_argument("--cluster-backend", default=cfg.CLUSTER_BACKEND, choices=("phash", "siglip"))
    ap.add_argument("--cluster-threshold", type=float, default=cfg.CLUSTER_THRESHOLD)
    ap.add_argument("--ucsf-distractors", type=int, default=0, help="total UCSF distractor pages to pull")
    ap.add_argument("--ucsf-letterhead-per-author", type=int, default=0)
    ap.add_argument("--split-letterhead-by-decade", action="store_true")
    ap.add_argument("--synth-per-class", type=int, default=cfg.SYNTH_INSTANCES_PER_CLASS)
    ap.add_argument("--synth-pool-dir", type=Path, default=None, help="local artwork dir (else LogoDet-3K via Kaggle)")
    ap.add_argument("--synth-max-classes", type=int, default=200)
    ap.add_argument(
        "--pin-tiers",
        type=Path,
        default=None,
        help="an earlier build_report.json; reuse its tier cutoffs so this build stays comparable to it",
    )
    ap.add_argument("--probe", action="store_true", help="check every source is reachable, then exit")
    ap.add_argument("--survival", action="store_true", help="print the class survival curve and exit")
    args = ap.parse_args(argv)

    if args.probe:
        return probe(args.raw)

    selected = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = set(selected) - set(ALL_SOURCES)
    if unknown:
        ap.error(f"unknown source(s): {sorted(unknown)}")

    args.out.mkdir(parents=True, exist_ok=True)
    images_dir = args.out / "images"
    warnings: list[str] = []

    pages = load_anchor_sources(selected, args.raw, limit=args.limit, warnings=warnings)
    print(f"anchor sources: {len(pages)} page(s)")

    # Identity clustering, for the sources that ship location without identity.
    from cluster_marks import cluster_source, write_cluster_report

    summaries = []
    for source in ("spods", "staver"):
        if source in selected:
            summary = cluster_source(
                pages,
                source,
                backend=args.cluster_backend,
                threshold=args.cluster_threshold,
            )
            summaries.append(summary)
            print(
                f"  {source}: {summary['marks']} mark(s) -> {summary['classes']} candidate class(es) "
                f"({summary.get('singletons', 0)} singleton) via {summary['backend']}"
            )
    if summaries:
        write_cluster_report(summaries, args.out / "cluster_report.json")

    if "ucsf" in selected and (args.ucsf_distractors or args.ucsf_letterhead_per_author):
        ucsf_pages = load_ucsf(
            args.raw,
            images_dir,
            distractor_budget=args.ucsf_distractors,
            letterhead_per_author=args.ucsf_letterhead_per_author,
            split_by_decade=args.split_letterhead_by_decade,
            warnings=warnings,
        )
        pages.extend(ucsf_pages)
        print(f"ucsf: {len(ucsf_pages)} page(s)")

    inventory = class_inventory(pages)
    curve = survival_curve(inventory, (2, 5, 10, 15, 20, 30, 50))
    print("\nclass survival curve (min instances -> classes):")
    for t, n in sorted(curve.items()):
        marker = "  <- selected" if t == args.min_instances else ""
        print(f"  >={t:<3} {n:>5}{marker}")

    if args.survival:
        return 0

    # Synthesis last: it needs held-out backgrounds, which means it needs to know
    # which pages already carry a real mark.
    if "synth" in selected:
        from synth_compose import build_synthetic_pages
        from sources import artwork

        marked = {p.page_id for p in pages if p.marks}
        backgrounds = [Path(p.path) for p in pages if p.page_id not in marked]
        if not backgrounds:
            warnings.append("synth: no unmarked pages available as backgrounds — skipped")
        else:
            if args.synth_pool_dir:
                pool = artwork.load_pool_dir(args.synth_pool_dir, limit=args.synth_max_classes)
            else:
                root = artwork.fetch_logodet3k(args.raw)
                pool = artwork.build_pool_from_logodet(
                    root, args.raw / "artwork_pool", max_classes=args.synth_max_classes
                )
            synth_pages = build_synthetic_pages(
                backgrounds,
                pool,
                images_dir / "synth",
                instances_per_class=args.synth_per_class,
                size_px=cfg.SYNTH_SIZE_PX,
                rotation_deg=cfg.SYNTH_ROTATION_DEG,
                seed=cfg.SYNTH_SEED,
            )
            used = {str(p) for page in synth_pages for p in [Path(page.meta["background"])]}
            # Hold the backgrounds out: a page must not be both a synthetic
            # canvas and a distractor scored against its own pasted mark.
            pages = [p for p in pages if p.path not in used]
            pages.extend(synth_pages)
            print(f"synth: {len(synth_pages)} page(s) over {len(pool)} class(es); {len(used)} background(s) held out")
            inventory = class_inventory(pages)

    admitted, rejected = admit_classes(
        pages, inventory, min_instances=args.min_instances, min_mark_px=args.min_mark_px
    )
    print(f"\nadmitted {len(admitted)} class(es); rejected {len(rejected)}")

    needs_hand_crop = write_query_crops(pages, inventory, admitted, args.out / "queries")
    if needs_hand_crop:
        print(f"  {len(needs_hand_crop)} weak-label class(es) need a hand-drawn query crop")

    pinned: Optional[dict[str, float]] = None
    if args.pin_tiers:
        pinned = json.loads(args.pin_tiers.read_text(encoding="utf-8")).get("tier_cutoffs")
        print(f"\npinning tier cutoffs from {args.pin_tiers}: {pinned}")

    tier_of, tier_cutoffs = assign_tiers(
        pages,
        admitted,
        tiers=cfg.TIERS,
        tier_order=cfg.TIER_ORDER,
        salt=cfg.TIER_SALT,
        pinned_cutoffs=pinned,
    )
    kept = []
    for page in pages:
        tier = tier_of.get(page.page_id)
        if tier is None:
            continue
        page.meta["tier"] = tier
        kept.append(page)
    dropped = len(pages) - len(kept)

    n = _common.write_manifest(kept, args.out / "corpus.jsonl")
    (args.out / "classes.json").write_text(json.dumps(admitted, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    tier_counts = {t: sum(1 for p in kept if p.meta.get("tier") == t) for t in cfg.TIER_ORDER}
    cumulative = {}
    running = 0
    for t in cfg.TIER_ORDER:
        running += tier_counts[t]
        cumulative[t] = running

    report = {
        "pages_written": n,
        "pages_dropped_over_budget": dropped,
        "tier_counts": tier_counts,
        "tier_cumulative": cumulative,
        # Feed these back with --pin-tiers to keep a later, larger build's tier
        # membership comparable to this one.
        "tier_cutoffs": tier_cutoffs,
        "classes_admitted": len(admitted),
        "classes_rejected": len(rejected),
        "rejection_reasons": rejected,
        "survival_curve": {str(k): v for k, v in curve.items()},
        "needs_hand_crop": needs_hand_crop,
        "warnings": warnings,
        "settings": {
            "sources": selected,
            "min_instances": args.min_instances,
            "min_mark_px": args.min_mark_px,
            "cluster_backend": args.cluster_backend,
            "cluster_threshold": args.cluster_threshold,
            "tier_salt": cfg.TIER_SALT,
        },
    }
    (args.out / "build_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\nwrote {n} page(s) to {args.out / 'corpus.jsonl'}")
    print(f"  tiers (cumulative): " + ", ".join(f"{t}={cumulative[t]}" for t in cfg.TIER_ORDER))
    if dropped:
        print(f"  {dropped} page(s) dropped: past the largest tier's budget")
    for w in warnings:
        print(f"  warning: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
