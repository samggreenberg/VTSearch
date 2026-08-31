#!/usr/bin/env python
"""Sweep the clustering threshold against a built corpus, and show the shape.

    python tune_clustering.py --corpus <dir>                 # sweep + histogram
    python tune_clustering.py --corpus <dir> --apply 0.06     # re-cluster in place

Clustering is the single most consequential unverified step in the corpus: it
invents the class identities that SPODS and StaVer do not ship.  Its threshold
therefore cannot be a constant somebody picked — it has to be read off the data,
and re-read whenever the source or the descriptor changes.

The failure this exists to catch is **single-linkage chaining**.  It does not
degrade gracefully: at a slightly-too-loose threshold every mark in the corpus
joins one giant component, because a chain of near-neighbours connects
everything to everything.  A first real run of SPODS collapsed 2,176 marks into
one class at the default 0.18, having behaved perfectly on a small fixture.
A sweep makes the cliff visible; a single threshold hides it.

Descriptors are computed once and reused across every threshold, so the sweep
costs one pass over the pages rather than one per threshold — which matters,
because that pass is minutes of decoding full-page scans.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cluster_marks as _cluster  # noqa: E402
import docmarks_config as cfg  # noqa: E402
from sources._common import read_manifest, write_manifest  # noqa: E402

DEFAULT_GRID = (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22)


def describe_once(corpus: Path, source: str, kinds: Sequence[str], backend: str) -> tuple[list, Any, Any]:
    """``(pages, refs, distance matrix)`` — the expensive pass, done once."""
    pages = list(read_manifest(corpus / "corpus.jsonl"))
    refs = _cluster.collect_refs(pages, kinds=kinds, source=source)
    if not refs:
        # A rebuilt corpus already carries class ids; clustering is idempotent
        # over them, so fall back to every mark of the right kind.
        refs = [
            _cluster.MarkRef(pi, mi, p.page_id, m.kind, m.box)
            for pi, p in enumerate(pages)
            if p.source == source
            for mi, m in enumerate(p.marks)
            if m.kind in set(kinds)
        ]
    if not refs:
        raise SystemExit(f"no {kinds} marks from source {source!r} in {corpus}")
    desc = _cluster.describe_marks(pages, refs, backend=backend)
    dist = _cluster.distance_matrix(desc, refs, backend=backend)
    return pages, refs, dist


def sweep(dist: Any, grid: Sequence[float], cannot_link: Sequence[tuple[int, int]] = ()) -> list[dict[str, Any]]:
    """Class counts and the largest-cluster share at each threshold.

    ``largest_share`` is the number to read, not ``classes``.  A run can report
    a healthy-looking class count while one giant component holds 95% of the
    marks and the rest are singletons around it — the count alone cannot tell
    that apart from a genuinely well-separated inventory.
    """
    n = dist.shape[0]
    rows = []
    for t in grid:
        labels = _cluster.single_linkage(dist, t, cannot_link=list(cannot_link))
        sizes = Counter(labels)
        biggest = max(sizes.values())
        rows.append(
            {
                "threshold": t,
                "classes": len(sizes),
                "largest": biggest,
                "largest_share": biggest / n,
                "singletons": sum(1 for v in sizes.values() if v == 1),
                "usable": sum(1 for v in sizes.values() if v >= cfg.MIN_INSTANCES),
            }
        )
    return rows


def distance_histogram(dist: Any, bins: int = 20) -> list[tuple[float, float, int]]:
    """Off-diagonal pairwise distances, binned — the shape behind the sweep."""
    iu = np.triu_indices(dist.shape[0], k=1)
    values = dist[iu]
    values = values[values < 1.0]  # 1.0 is the aspect-gate sentinel, not a distance
    if values.size == 0:
        return []
    counts, edges = np.histogram(values, bins=bins, range=(0.0, float(values.max())))
    return [(float(edges[i]), float(edges[i + 1]), int(counts[i])) for i in range(len(counts))]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--source", default="spods")
    ap.add_argument("--kinds", default="logo,stamp")
    ap.add_argument("--backend", default=cfg.CLUSTER_BACKEND, choices=("phash", "siglip"))
    ap.add_argument("--grid", default=",".join(str(g) for g in DEFAULT_GRID))
    ap.add_argument("--apply", type=float, default=None, help="re-cluster the manifest at this threshold and save")
    args = ap.parse_args(argv)

    kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())
    pages, refs, dist = describe_once(args.corpus, args.source, kinds, args.backend)
    print(f"{len(refs)} mark(s) of kind {kinds} from {args.source!r}, backend={args.backend}\n")

    print("pairwise distance histogram (aspect-gated pairs excluded):")
    for lo, hi, count in distance_histogram(dist):
        bar = "#" * min(60, int(60 * count / max(1, max(c for _, _, c in distance_histogram(dist)))))
        print(f"  {lo:.3f}-{hi:.3f}  {count:>9,}  {bar}")

    grid = [float(g) for g in args.grid.split(",")]
    rows = sweep(dist, grid)
    print("\nthreshold sweep:")
    print(
        f"  {'thresh':>7}  {'classes':>8}  {'largest':>8}  {'share':>6}  {'singles':>8}  {'>=' + str(cfg.MIN_INSTANCES):>8}"
    )
    for r in rows:
        flag = "   <- chained" if r["largest_share"] > 0.5 else ""
        print(
            f"  {r['threshold']:>7.3f}  {r['classes']:>8,}  {r['largest']:>8,}  "
            f"{r['largest_share']:>6.1%}  {r['singletons']:>8,}  {r['usable']:>8,}{flag}"
        )

    (args.corpus / "cluster_sweep.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.corpus / 'cluster_sweep.json'}")

    if args.apply is not None:
        labels = _cluster.single_linkage(dist, args.apply)
        # Clear any previous identities on these marks first, so re-clustering
        # replaces the inventory rather than layering a second one over it.
        from sources._common import Mark

        for ref in refs:
            mark = pages[ref.page_index].marks[ref.mark_index]
            pages[ref.page_index].marks[ref.mark_index] = Mark(mark.kind, mark.box, None, mark.provenance)
        classes = _cluster.assign_class_ids(pages, refs, labels, source=args.source)
        write_manifest(pages, args.corpus / "corpus.jsonl")
        print(f"applied threshold {args.apply}: {len(classes)} class(es); rewrote corpus.jsonl")
        print("  now re-run build_corpus.py to refresh classes.json, or shortlist.py to rank them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
