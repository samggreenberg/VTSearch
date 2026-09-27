"""Score any ranker on FullMarks: bring a scores file, get the benchmark's numbers (#4108).

FullMarks exists so that ideas about finding a stamp or logo from one query crop
can be tested against checked labels.  A ranker is anything that gives each
(class, page) pair a score -- higher means "more likely to carry this class's
mark".  Write those scores to a file and this scores them exactly as the
benchmark's own tools do:

* on the headline ``own_verified`` pool per class and tier (AP, recall@10/50/100,
  first hit), ranked by descending score with the page id breaking ties, so a
  run repeats bit for bit (``eval_retrieval.rank_pool``);
* beside two **mark-blind controls** that show what the pool gives away for
  free: ``source_prior`` (the class's own source first) and
  ``provenance_prior`` (the source *and industry* its positives come from
  first -- the letterhead-style shortcut the UCSF classes carried until v4.2);
* with ``surprise_hits.json``: the ranker's top presumed negatives per class,
  for ``surprise_review.py`` to queue.  A hit there is a review item, not a
  false positive (#4089): if the mark is really there, the ranker was right;

and stamps every result with the corpus version and whether the corpus on disk
still matches that version's frozen manifest (``versions/<version>.json``),
because a number is only comparable to one measured on the same version.

The scores file is CSV, optionally gzipped, with a header::

    class_id,page_id,score
    tobacco800/logo_ajj10e00_1,ucsf/fhjl0188#0,0.4113

A ranker's scores do not depend on the tier, so one file covering the largest
tier scores every tier below it.  The built-in cells are the exception: each
tier's cell was embedded on its own, so a page's vector differs in the last
float digits between cells.  To reproduce a tier's reference number exactly,
export from that tier's cell (``--tier s`` for tier ``s``).

A pool page with no score ranks last (``coverage`` records how many there
were), so a ranker that only scores a shortlist is scored as the ranking it
actually produced.

    # start from a built-in method's scores
    python score_ranker.py export --method siglip --tier m --out siglip_m.csv.gz
    python score_ranker.py export --sift-inliers <eval_sift_rank out>/inliers.json --out sift_m.csv.gz
    # score anything
    python score_ranker.py score --scores my_ranker.csv.gz --tiers s,m --name my_ranker --out <dir>
    # freeze the corpus on disk as CORPUS_VERSION (maintainers)
    python score_ranker.py freeze
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import sys
import zlib
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fullmarks_config as cfg  # noqa: E402

VERSIONS = Path(__file__).resolve().parent / "versions"
#: Files whose bytes define a corpus version.  Query crops are added per class.
VERSION_FILES = (
    "corpus.jsonl",
    "classes.json",
    "reviewed_negatives.json",
    "added_marks.json",
    "adjudications.json",
    "box_overrides.json",
)
#: Presumed negatives kept per class for the review queue.
SURPRISE_K = 10


# --------------------------------------------------------------------------
# Scores files
# --------------------------------------------------------------------------


def _open_text(path: Path, mode: str = "r"):
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, mode + "b"), encoding="utf-8", newline="")
    return open(path, mode, encoding="utf-8", newline="")


def read_scores(path: Path, classes: Optional[Iterable[str]] = None) -> dict[str, dict[str, float]]:
    """``{class_id: {page_id: score}}`` from a scores CSV (optionally ``.gz``).

    A duplicated (class, page) pair or a non-finite score is an error: both
    would make a ranking depend on file order, which the tie-break rule exists
    to prevent.
    """
    import math  # noqa: PLC0415

    want = set(classes) if classes is not None else None
    out: dict[str, dict[str, float]] = {}
    with _open_text(path) as fh:
        reader = csv.DictReader(fh)
        missing = {"class_id", "page_id", "score"} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: header lacks {sorted(missing)}")
        for n, row in enumerate(reader, start=2):
            cid = row["class_id"]
            if want is not None and cid not in want:
                continue
            score = float(row["score"])
            if not math.isfinite(score):
                raise ValueError(f"{path}:{n}: score {row['score']!r} is not finite")
            by_page = out.setdefault(cid, {})
            if row["page_id"] in by_page:
                raise ValueError(f"{path}:{n}: {cid} {row['page_id']} scored twice")
            by_page[row["page_id"]] = score
    return out


def write_scores(scores: dict[str, dict[str, float]], path: Path) -> int:
    n = 0
    with _open_text(path, "w") as fh:
        w = csv.writer(fh)
        w.writerow(["class_id", "page_id", "score"])
        for cid in sorted(scores):
            for pid, s in sorted(scores[cid].items()):
                w.writerow([cid, pid, repr(float(s))])
                n += 1
    return n


def sift_scores(inliers: dict[str, dict[str, Sequence[int]]]) -> dict[str, dict[str, float]]:
    """``eval_sift_rank.py``'s ``inliers.json`` as scores that rank exactly as it does.

    It ranks by inliers, then tentative matches, then page id; ``inliers +
    tentative / 1e6`` orders the same way while tentative counts stay below a
    million.  Pages it never matched are absent and rank last by page id, as
    they do there.
    """
    out: dict[str, dict[str, float]] = {}
    for cid, by_page in inliers.items():
        for pid, (n, t) in by_page.items():
            if t >= 1_000_000:
                raise ValueError(f"{cid} {pid}: {t} tentative matches breaks the tie-break encoding")
            out.setdefault(cid, {})[pid] = float(n) + float(t) / 1e6
    return out


# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------


def _tiebreak(page_id: str) -> float:
    return zlib.crc32(page_id.encode()) / 2**33


def provenance_prior_scores(
    positives: set[str], source_of: dict[str, str], industry_of: dict[str, Optional[str]]
) -> dict[str, float]:
    """Pages sharing the (source, industry) of the class's positives first; knows nothing of the mark.

    ``source_prior`` asks "same source?"; this asks "same kind of page?".  For a
    class whose positives all sit in one industry of one source -- the four UCSF
    classes, all on tobacco-company letters -- it is the style shortcut a
    method can learn instead of the mark.  For the other sources it reduces to
    ``source_prior`` (their pages carry no industry).
    """
    kinds = {(source_of[p], industry_of.get(p)) for p in positives if p in source_of}
    sources = {s for s, _i in kinds}
    return {
        p: (2.0 if (s, industry_of.get(p)) in kinds else 1.0 if s in sources else 0.0) + _tiebreak(p)
        for p, s in source_of.items()
    }


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def score_tier(
    corpus: Path,
    tier: str,
    classes: dict[str, Any],
    scores: dict[str, dict[str, float]],
    *,
    surprise_k: int = SURPRISE_K,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rows (one per class) and ``{class: top presumed negatives}`` for one tier."""
    import embed_corpus  # noqa: PLC0415
    import eval_retrieval as ev  # noqa: PLC0415

    pages = embed_corpus.pages_for_tier(corpus, tier)
    by_source: dict[str, list[str]] = {}
    industry_of: dict[str, Optional[str]] = {}
    source_of: dict[str, str] = {}
    for page in pages:
        by_source.setdefault(page.source, []).append(page.page_id)
        industry_of[page.page_id] = page.meta.get("industry")
        source_of[page.page_id] = page.source
    rows, surprise = [], {}
    for cid, meta in sorted(classes.items()):
        pools = ev.class_pools(meta, by_source, industry_of)
        pool, positives = pools[ev.HEADLINE_POOL], pools["positives"]
        mine = scores.get(cid, {})
        ranked = ev.rank_pool(mine, pool)
        src = ev.rank_pool(ev.source_prior_scores(meta.get("source"), source_of), pool)
        prov = ev.rank_pool(provenance_prior_scores(positives, source_of, industry_of), pool)
        m = ev.metrics(ranked, positives)
        rows.append(
            {
                "tier": tier,
                "class_id": cid,
                "source": meta.get("source"),
                "kind": meta.get("kind"),
                "n_positive": len(positives),
                "n_pool": len(pool),
                "coverage": round(sum(1 for p in pool if p in mine) / max(1, len(pool)), 4),
                **m,
                "source_prior_ap": ev.metrics(src, positives)["ap"],
                "provenance_prior_ap": ev.metrics(prov, positives)["ap"],
            }
        )
        surprise[cid] = ev.top_presumed(ranked, pools["presumed"], surprise_k)
    return rows, surprise


def summarise(rows: Sequence[dict[str, Any]], name: str, stamp: dict[str, Any]) -> str:
    import numpy as np  # noqa: PLC0415

    lines = [
        f"# {name} on FullMarks {stamp['corpus_version']}",
        "",
        f"Corpus on disk {'matches' if stamp['matches_frozen'] else '**does not match**'} "
        f"the frozen {stamp['corpus_version']} manifest"
        + (f" (differs: {', '.join(stamp['differs'])})" if stamp.get("differs") else "")
        + ". Pool: `own_verified`. Controls ignore the mark.",
        "",
        "| tier | source | classes | mean AP | r@10 | r@50 | source-only control | provenance control | coverage |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tier in sorted({r["tier"] for r in rows}, key=cfg.TIER_ORDER.index):
        groups = [("all", [r for r in rows if r["tier"] == tier])]
        groups += [
            (s, [r for r in rows if r["tier"] == tier and r["source"] == s])
            for s in sorted({r["source"] for r in rows if r["tier"] == tier})
        ]
        for label, sel in groups:
            sel = [r for r in sel if r["n_positive"]]
            if not sel:
                continue
            mean = lambda k: float(np.mean([r[k] for r in sel]))  # noqa: E731
            lines.append(
                f"| {tier} | {label} | {len(sel)} | {mean('ap'):.3f} | {mean('r@10'):.2f} | {mean('r@50'):.2f} "
                f"| {mean('source_prior_ap'):.3f} | {mean('provenance_prior_ap'):.3f} | {mean('coverage'):.2f} |"
            )
    lines += [
        "",
        "Classes with no positive in a tier are left out of its means (they score 0 for every ranker).",
        "A number that does not clear the provenance control is not evidence the ranker sees the mark.",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Versions
# --------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(corpus: Path) -> dict[str, str]:
    """``{name: sha256}`` of the files that define the corpus, and of every roster class's query crop."""
    out = {name: _sha256(corpus / name) for name in VERSION_FILES if (corpus / name).exists()}
    classes = json.loads((corpus / "classes.json").read_text(encoding="utf-8"))
    for cid, meta in sorted(classes.items()):
        if meta.get("on_roster") and meta.get("query_crop") and Path(meta["query_crop"]).exists():
            out[f"query_crop:{cid}"] = _sha256(Path(meta["query_crop"]))
    return out


def stamp(corpus: Path, version: str = cfg.CORPUS_VERSION, versions: Path = VERSIONS) -> dict[str, Any]:
    """The version a result was measured on, and whether the corpus still matches its frozen manifest."""
    frozen_path = versions / f"{version}.json"
    if not frozen_path.exists():
        return {"corpus_version": version, "matches_frozen": False, "differs": ["(no frozen manifest)"]}
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))["files"]
    now = fingerprint(corpus)
    differs = sorted(k for k in set(frozen) | set(now) if frozen.get(k) != now.get(k))
    return {"corpus_version": version, "matches_frozen": not differs, "differs": differs}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _roster(corpus: Path) -> dict[str, Any]:
    return {
        cid: meta
        for cid, meta in json.loads((corpus / "classes.json").read_text(encoding="utf-8")).items()
        if meta.get("on_roster") and meta.get("query_crop")
    }


def cmd_export(args) -> int:
    if args.sift_inliers:
        scores = sift_scores(json.loads(Path(args.sift_inliers).read_text(encoding="utf-8")))
    else:
        import numpy as np  # noqa: PLC0415

        import embed_corpus  # noqa: PLC0415
        import eval_retrieval as ev  # noqa: PLC0415
        from vtscore.media import get_embedder  # noqa: PLC0415

        cell = {"siglip": "siglip", "vlad": "sift_vlad"}[args.method]
        ids, matrix = ev.read_vectors(embed_corpus.cell_path(args.tier, cell), cell)
        emb = get_embedder(cell)
        emb.load_models()
        scores = {}
        for cid, meta in sorted(_roster(args.corpus).items()):
            vec, _ = ev.embed_query(emb, Path(meta["query_crop"]))
            if vec is None:
                print(f"  {cid}: no vector for its query crop; not scored", flush=True)
                continue
            scores[cid] = dict(zip(ids, (matrix @ np.asarray(vec, dtype=np.float32)).tolist()))
    n = write_scores(scores, Path(args.out))
    print(f"wrote {n} scores for {len(scores)} classes -> {args.out}")
    return 0


def cmd_score(args) -> int:
    if args.out.resolve().is_relative_to(args.corpus.resolve()):
        raise SystemExit("--out is inside the corpus; scoring never writes there")
    classes = _roster(args.corpus)
    scores = read_scores(args.scores, classes)
    unknown = sorted(set(scores) - set(classes))
    if unknown:
        print(f"  ignoring {len(unknown)} class(es) not on the roster: {unknown[:5]}")
    st = stamp(args.corpus)
    args.out.mkdir(parents=True, exist_ok=True)
    rows, surprise = [], {}
    for tier in args.tiers.split(","):
        r, s = score_tier(args.corpus, tier, classes, scores, surprise_k=args.surprise_k)
        rows += r
        surprise[tier] = {cid: {args.name: hits} for cid, hits in s.items()}
    with open(args.out / "rows.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (args.out / "surprise_hits.json").write_text(json.dumps(surprise, indent=1) + "\n", encoding="utf-8")
    (args.out / "stamp.json").write_text(json.dumps(dict(st, ranker=args.name), indent=1) + "\n", encoding="utf-8")
    text = summarise(rows, args.name, st)
    (args.out / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


def cmd_freeze(args) -> int:
    VERSIONS.mkdir(exist_ok=True)
    dest = VERSIONS / f"{cfg.CORPUS_VERSION}.json"
    if dest.exists() and not args.force:
        raise SystemExit(f"{dest} exists; a frozen version is never rewritten (--force if it was frozen wrongly)")
    files = fingerprint(args.corpus)
    dest.write_text(
        json.dumps({"version": cfg.CORPUS_VERSION, "files": files}, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"froze {cfg.CORPUS_VERSION}: {len(files)} files -> {dest}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="scores of a built-in method, as a scores file")
    e.add_argument("--corpus", type=Path, default=cfg.OUT)
    e.add_argument("--method", choices=("siglip", "vlad"), default="siglip")
    e.add_argument("--tier", default="m", help="the cell to read; scores cover every tier up to it")
    e.add_argument("--sift-inliers", default=None, help="convert eval_sift_rank.py's inliers.json instead")
    e.add_argument("--out", required=True)
    s = sub.add_parser("score", help="score a ranker's scores file")
    s.add_argument("--corpus", type=Path, default=cfg.OUT)
    s.add_argument("--scores", type=Path, required=True)
    s.add_argument("--tiers", default="s,m")
    s.add_argument("--name", default="ranker")
    s.add_argument("--surprise-k", type=int, default=SURPRISE_K)
    s.add_argument("--out", type=Path, required=True)
    f = sub.add_parser("freeze", help="record the corpus on disk as CORPUS_VERSION (maintainers)")
    f.add_argument("--corpus", type=Path, default=cfg.OUT)
    f.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    return {"export": cmd_export, "score": cmd_score, "freeze": cmd_freeze}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
