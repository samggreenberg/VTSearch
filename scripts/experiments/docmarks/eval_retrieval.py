"""Score the DocMarks roster: can each embedder find a class from its query crop?

The first evaluation the corpus was built for (#3904).  Every roster class is
searched once per tier with its own query crop, three ways (plus a
``source_prior`` control that ignores the mark, and ``--rerank-all`` to verify
every page rather than a shortlist):

* ``siglip``       -- cosine between the crop's SigLIP vector and every page's;
* ``vlad``         -- cosine between the crop's VLAD vector and every page's
                      (structural search, Stage 1 alone);
* ``vlad_rerank``  -- Stage 1 then the app's own Stage-2 geometric re-rank of the
                      top ``--rerank-k`` (``structural_rerank`` with the
                      cold-start inlier gate, the crop as the one template).

**What gets scored is the point of the exercise.**  A class is ranked against
its positives plus the pages ``roster.eligible_pages`` says are safe to call
negatives -- decided *per page*, because the contamination rule is finer than a
source: Tobacco800 classes may be scored against UCSF's Food or Opioids pages but
never its Tobacco pages, which come from the same archive and can carry the same
letterhead unlabelled.  Every class is also scored over the ``naive`` pool (every
page in the tier) so the size of that exclusion is on record, and the
highest-ranked pages the exclusion removed are written out to be looked at.

The query crop's own page is dropped from both pools: finding the page a crop
was cut from measures nothing.

Reads cells, writes only under ``--out``:

    source scripts/experiments/pile/pile_env.sh
    python eval_retrieval.py --tiers s,m --out /expscratch/$USER/docmarks/eval-3904

``rows.csv`` holds one row per (tier, class, pool, method); ``surprise_hits.json``
each method's top-ranked presumed negatives, for review (#4089); ``excluded_hits.json``
the top excluded pages per class; ``summary.md`` the per-tier table.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import docmarks_config as cfg  # noqa: E402
import embed_corpus  # noqa: E402
import roster  # noqa: E402

METHODS = ("siglip", "vlad", "vlad_rerank", "source_prior")
#: ``eligible`` is the contamination rule as written; ``own_verified`` adds back
#: the class's own source as known negatives; ``naive`` is every page.  The
#: middle pool exists because the rule as written removes every same-source page
#: that is not a positive -- which leaves the positives the only pages in their
#: source's style, and a ranker that recognises SPODS paper wins without looking
#: at the mark.  ``source_prior`` is that ranker, as a control.
POOLS = ("eligible", "own_verified", "naive")
#: The pool numbers are quoted from.  The owner chose it on 2026-09-16 (#3913):
#: every mark on an anchor-source page is boxed and clustered, so a same-source
#: non-member is a verified negative, and excluding those pages instead lets
#: ``source_prior`` score AP 1.00 without looking at the mark.
HEADLINE_POOL = "own_verified"
#: Cut-offs reported beside AP.  50 is also the app's re-rank shortlist, so
#: ``r@50`` under ``vlad`` is exactly what Stage 2 gets to see.
RECALL_AT = (10, 50, 100)
#: Excluded pages kept per class for inspection.
EXCLUDED_TOP = 20
#: Presumed negatives kept per class and method for review (#4089): the pages a
#: method ranked highest that the headline pool scores as negatives unchecked.
#: Ten per method, over three methods, bounds a class at 30 questions before
#: the methods' overlap is removed.
SURPRISE_K = 10


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def average_precision(ranked: Sequence[str], positives: set[str]) -> float:
    """Non-interpolated AP of a full ranking; 0.0 when there is nothing to find."""
    if not positives:
        return 0.0
    hits = 0
    total = 0.0
    for rank, page_id in enumerate(ranked, start=1):
        if page_id in positives:
            hits += 1
            total += hits / rank
    return total / len(positives)


def recall_at(ranked: Sequence[str], positives: set[str], k: int) -> float:
    if not positives:
        return 0.0
    return sum(1 for p in ranked[:k] if p in positives) / len(positives)


def first_hit(ranked: Sequence[str], positives: set[str]) -> Optional[int]:
    for rank, page_id in enumerate(ranked, start=1):
        if page_id in positives:
            return rank
    return None


def metrics(ranked: Sequence[str], positives: set[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"ap": average_precision(ranked, positives), "first_hit": first_hit(ranked, positives)}
    for k in RECALL_AT:
        out[f"r@{k}"] = recall_at(ranked, positives, k)
    return out


# --------------------------------------------------------------------------
# Pools
# --------------------------------------------------------------------------


def class_pools(
    class_meta: dict[str, Any],
    pages_by_source: dict[str, list[str]],
    industry_of: dict[str, Optional[str]],
) -> dict[str, Any]:
    """Positives and the three scored pools for one class, query page removed."""
    query_page = class_meta.get("query_page_id")
    split = roster.eligible_pages(class_meta, pages_by_source, industry_of=industry_of)
    own = roster.eligible_pages(
        class_meta, pages_by_source, verified_negative_sources=[class_meta.get("source")], industry_of=industry_of
    )
    in_tier = {p for ids in pages_by_source.values() for p in ids}
    positives = {p for p in split["positive"] if p in in_tier and p != query_page}
    eligible = positives | set(split["known_negative"]) | set(split["presumed_negative"])
    own_verified = positives | set(own["known_negative"]) | set(own["presumed_negative"])
    naive = in_tier - {query_page}
    return {
        "positives": positives,
        "eligible": eligible - {query_page},
        "own_verified": own_verified - {query_page},
        "naive": naive,
        # The headline pool's negatives that nobody checked: where a correct
        # retrieval would be scored as a false positive (#4089).
        "presumed": set(own["presumed_negative"]) - {query_page},
    }


def rank_pool(scores: dict[str, float], pool: Iterable[str]) -> list[str]:
    """Pool ordered by descending score, page id breaking ties so runs repeat."""
    return sorted(pool, key=lambda p: (-scores.get(p, float("-inf")), p))


# --------------------------------------------------------------------------
# Cells and queries
# --------------------------------------------------------------------------


def read_vectors(path: Path, embedder: str) -> tuple[list[str], np.ndarray]:
    """``(page_ids, matrix)`` for a cell, streamed so local features are not kept."""
    io = embed_corpus._cells_io()
    ids: list[str] = []
    rows: list[np.ndarray] = []
    for _cid, media in io.iter_medias(path):
        vec = (media.get("embeddings") or {}).get(embedder)
        if vec is None:
            continue
        ids.append(media["origin_name"])
        rows.append(np.asarray(vec, dtype=np.float32))
    return ids, np.vstack(rows) if rows else np.zeros((0, 0), dtype=np.float32)


def read_local_features(path: Path, wanted: set[str]) -> dict[str, Any]:
    """``local_features`` for just the *wanted* pages -- the Stage-2 shortlists."""
    io = embed_corpus._cells_io()
    return {
        media["origin_name"]: media.get("local_features")
        for _cid, media in io.iter_medias(path)
        if media.get("origin_name") in wanted
    }


def embed_query(embedder: Any, crop: Path) -> tuple[Optional[np.ndarray], Any]:
    """Vector (and structural features, where the embedder has them) for a crop."""
    media = {"media_type": "image", "media_bytes": crop.read_bytes(), "origin_name": crop.name}
    vec = embedder.embed_media(media)
    feats = embedder.local_features_forward(media) if embedder.supports_geometric_verification else None
    return vec, feats


def rerank(
    ranked: list[str],
    scores: dict[str, float],
    features: dict[str, Any],
    template: Any,
    matcher: Any,
    k: int,
) -> list[str]:
    """The app's Stage 2 over a Stage-1 ranking -- ``structural_rerank`` unchanged."""
    from vtscore.training.structural_similarity import VerificationScorer, structural_rerank  # noqa: PLC0415

    if template is None or not getattr(template, "count", 0):
        return list(ranked)
    results = [{"id": p, "score": scores[p]} for p in ranked]
    snap = {p: {"local_features": features.get(p)} for p in ranked[:k]}
    out = structural_rerank(results, snap, [template], VerificationScorer(), matcher, top_k=k)
    return [entry["id"] for entry in out]


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


def source_prior_scores(class_source: str, source_of: dict[str, str]) -> dict[str, float]:
    """1 for a page from the class's own source, 0 otherwise, with a fixed-hash tiebreak.

    Knows nothing about the mark.  Whatever AP this reaches on a pool is what
    recognising the source's page style is worth there.
    """
    import zlib  # noqa: PLC0415

    return {p: (1.0 if s == class_source else 0.0) + zlib.crc32(p.encode()) / 2**33 for p, s in source_of.items()}


def run_tier(
    corpus: Path,
    tier: str,
    classes: dict[str, Any],
    *,
    rerank_k: int,
    rerank_all: bool = False,
    surprise_k: int = SURPRISE_K,
    surprise: Optional[dict[str, Any]] = None,
    log=print,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score every class at *tier*.  *surprise*, if given, is filled with each class's :func:`top_presumed` hits."""
    surprise = {} if surprise is None else surprise
    from vtscore.media import get_embedder  # noqa: PLC0415

    pages = embed_corpus.pages_for_tier(corpus, tier)
    pages_by_source: dict[str, list[str]] = {}
    industry_of: dict[str, Optional[str]] = {}
    source_of: dict[str, str] = {}
    for page in pages:
        pages_by_source.setdefault(page.source, []).append(page.page_id)
        industry_of[page.page_id] = page.meta.get("industry")
        source_of[page.page_id] = page.source
    pools = {cid: class_pools(meta, pages_by_source, industry_of) for cid, meta in classes.items()}
    log(f"=== tier {tier}: {len(pages)} pages, {len(classes)} classes")

    scores: dict[str, dict[str, dict[str, float]]] = {}  # method -> class -> page -> score
    templates: dict[str, Any] = {}
    matcher = None
    for method, name in (("siglip", "siglip"), ("vlad", "sift_vlad")):
        t0 = time.time()
        ids, matrix = read_vectors(embed_corpus.cell_path(tier, name), name)
        missing = len(pages) - len(ids)
        log(f"  {name}: {len(ids)} vectors ({missing} pages without one) in {time.time() - t0:.0f}s")
        emb = get_embedder(name)
        emb.load_models()
        scores[method] = {}
        for cid, meta in classes.items():
            vec, feats = embed_query(emb, Path(meta["query_crop"]))
            if vec is None:
                log(f"  {name}: no vector for {cid}'s query crop; class scores nothing")
                scores[method][cid] = {}
                continue
            sims = matrix @ np.asarray(vec, dtype=np.float32)
            scores[method][cid] = dict(zip(ids, sims.tolist()))
            if feats is not None:
                templates[cid] = feats
        if name == "sift_vlad":
            matcher = emb.structural_matcher
    scores["source_prior"] = {cid: source_prior_scores(meta.get("source"), source_of) for cid, meta in classes.items()}

    # Stage 2 needs local features only for each pool's Stage-1 shortlist.
    shortlists: dict[tuple[str, str], list[str]] = {}
    for cid in classes:
        for pool in POOLS:
            shortlists[(cid, pool)] = rank_pool(scores["vlad"][cid], pools[cid][pool])
    # --rerank-all verifies every page in the pool: not what the app does, but it
    # separates Stage 2's own discrimination from Stage 1's recall.
    wanted = set(source_of) if rerank_all else {p for ranked in shortlists.values() for p in ranked[:rerank_k]}
    t0 = time.time()
    features = read_local_features(embed_corpus.cell_path(tier, "sift_vlad"), wanted)
    log(f"  local features for {len(features)} shortlisted pages in {time.time() - t0:.0f}s")

    rows: list[dict[str, Any]] = []
    excluded: dict[str, Any] = {}
    for cid, meta in sorted(classes.items()):
        positives = pools[cid]["positives"]
        for pool in POOLS:
            members = pools[cid][pool]
            for method in METHODS:
                if method == "vlad_rerank":
                    ranked = rerank(
                        shortlists[(cid, pool)], scores["vlad"][cid], features, templates.get(cid), matcher, rerank_k
                    )
                else:
                    ranked = rank_pool(scores[method][cid], members)
                _emit(rows, tier, cid, meta, positives, pool, members, method, ranked)
                if pool == HEADLINE_POOL and method != "source_prior":
                    surprise.setdefault(cid, {})[method] = top_presumed(ranked, pools[cid]["presumed"], surprise_k)
                if method == "vlad_rerank" and rerank_all and pool != "naive":
                    full = rerank(
                        shortlists[(cid, pool)],
                        scores["vlad"][cid],
                        features,
                        templates.get(cid),
                        matcher,
                        len(members),
                    )
                    _emit(rows, tier, cid, meta, positives, pool, members, "rerank_all", full)
        # The pipeline control: where does the page the crop was cut from rank?
        query_page = meta.get("query_page_id")
        if query_page in source_of:
            diag = {m: rank_pool(scores[m][cid], source_of).index(query_page) + 1 for m in ("siglip", "vlad")}
            log(f"  {cid}: own page ranks siglip {diag['siglip']}, vlad {diag['vlad']} of {len(source_of)}")
        removed = pools[cid]["naive"] - pools[cid]["eligible"]
        excluded[cid] = {
            "n_excluded": len(removed),
            "by_stratum": _strata(removed, source_of, industry_of),
            "top": {
                method: [
                    {"page_id": p, "score": round(scores[method][cid].get(p, float("nan")), 4), "rank_naive": rank}
                    for rank, p in enumerate(rank_pool(scores[method][cid], pools[cid]["naive"]), start=1)
                    if p in removed
                ][:EXCLUDED_TOP]
                for method in ("siglip", "vlad")
            },
        }
    return rows, excluded


def top_presumed(ranked: Sequence[str], presumed: set[str], k: int) -> list[dict[str, Any]]:
    """The first *k* pages of *ranked* that are presumed negatives, with their ranks.

    These are the pages a method put highest that the benchmark scores as
    negatives without anyone having looked.  If one carries the mark, the method
    was right and the metric says otherwise -- so they are review items, not
    false positives (#3922 rung 3, #4089).
    """
    out: list[dict[str, Any]] = []
    for rank, p in enumerate(ranked, start=1):
        if p in presumed:
            out.append({"page_id": p, "rank": rank})
            if len(out) >= k:
                break
    return out


def _emit(rows, tier, cid, meta, positives, pool, members, method, ranked) -> None:
    rows.append(
        {
            "tier": tier,
            "class_id": cid,
            "source": meta.get("source"),
            "kind": meta.get("kind"),
            "median_mark_px": meta.get("median_mark_px"),
            "n_positive": len(positives),
            "pool": pool,
            "n_pool": len(members),
            "method": method,
            **metrics(ranked, positives),
        }
    )


def _strata(
    page_ids: Iterable[str], source_of: dict[str, str], industry_of: dict[str, Optional[str]]
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for p in page_ids:
        key = source_of[p] + (f":{industry_of[p]}" if industry_of.get(p) else "")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def summarise(rows: list[dict[str, Any]]) -> str:
    """Mean over classes per (tier, pool, method), and per source under :data:`HEADLINE_POOL`."""
    lines = [
        "| tier | pool | method | classes | mean AP | mean r@10 | mean r@50 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    keys = sorted(
        {(r["tier"], r["pool"], r["method"]) for r in rows},
        key=lambda k: (k[0], POOLS.index(k[1]), (METHODS + ("rerank_all",)).index(k[2])),
    )
    for tier, pool, method in keys:
        sel = [r for r in rows if (r["tier"], r["pool"], r["method"]) == (tier, pool, method)]
        lines.append(
            f"| {tier} | {pool} | {method} | {len(sel)} | {np.mean([r['ap'] for r in sel]):.2f} "
            f"| {np.mean([r['r@10'] for r in sel]):.2f} | {np.mean([r['r@50'] for r in sel]):.2f} |"
        )
    lines += ["", f"| tier | source | method | classes | mean AP ({HEADLINE_POOL}) |", "|---|---|---|---:|---:|"]
    for tier in sorted({r["tier"] for r in rows}):
        for source in sorted({r["source"] for r in rows}):
            for method in METHODS + ("rerank_all",):
                sel = [
                    r
                    for r in rows
                    if r["tier"] == tier
                    and r["source"] == source
                    and r["method"] == method
                    and r["pool"] == HEADLINE_POOL
                ]
                if sel:
                    lines.append(
                        f"| {tier} | {source} | {method} | {len(sel)} | {np.mean([r['ap'] for r in sel]):.2f} |"
                    )
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--tiers", default="s,m")
    ap.add_argument("--rerank-k", type=int, default=50)
    ap.add_argument(
        "--rerank-all", action="store_true", help="also verify every pool page (diagnostic; slow past tier s)"
    )
    ap.add_argument(
        "--surprise-k",
        type=int,
        default=SURPRISE_K,
        help="presumed negatives per class and method to write to surprise_hits.json for review (#4089)",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.out.resolve().is_relative_to(args.corpus.resolve()):
        ap.error("--out is inside the corpus; an eval never writes there")
    classes = {
        cid: meta
        for cid, meta in json.loads((args.corpus / "classes.json").read_text(encoding="utf-8")).items()
        if meta.get("on_roster") and meta.get("query_crop")
    }
    args.out.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    all_excluded: dict[str, Any] = {}
    all_surprise: dict[str, Any] = {}
    for tier in args.tiers.split(","):
        all_surprise[tier] = {}
        rows, excluded = run_tier(
            args.corpus,
            tier,
            classes,
            rerank_k=args.rerank_k,
            rerank_all=args.rerank_all,
            surprise_k=args.surprise_k,
            surprise=all_surprise[tier],
        )
        all_rows += rows
        all_excluded[tier] = excluded

    with (args.out / "rows.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    (args.out / "excluded_hits.json").write_text(json.dumps(all_excluded, indent=2) + "\n", encoding="utf-8")
    (args.out / "surprise_hits.json").write_text(json.dumps(all_surprise, indent=2) + "\n", encoding="utf-8")
    (args.out / "summary.md").write_text(summarise(all_rows), encoding="utf-8")
    print("\n" + summarise(all_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
