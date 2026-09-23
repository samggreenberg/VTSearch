"""Review each run's top-ranked presumed negatives: adjudicate on surprise (#3922 rung 3, #4089).

97.6% of what DocMarks scores is a ``presumed_negative``: a page from a
contamination-safe source that nobody looked at.  The contamination checks
bound the *rate* of unlabelled positives among them (0 of 239 anchor pages,
0 of 200 UCSF pages), but a rate bound is not a count bound -- 1% of a
46,700-page pool is ~470 pages -- and an unlabelled positive only distorts AP
where a method ranks it high.  So the pages worth a human look are the ones a
real run put at the top: ``eval_retrieval.py`` writes each method's first
``--surprise-k`` presumed negatives per class to ``surprise_hits.json``.

A hit is a **review item, not a false positive**.  This turns a run's hits into
Good/Bad queues, one per class, with planted known positives, and applies the
answers:

* **Bad** -> ``reviewed_negative_page_ids``: the page is now a known negative.
* **Good** -> ``excluded_page_ids``: seen to carry the mark, so it is scored as
  neither.  It is listed for boxing; promoting it to an instance is the
  completeness pass's job, because an instance needs a box and this question
  has none.

Nothing automated decides either way; the run only chooses what to show.

    python surprise_review.py emit  --hits <eval out>/surprise_hits.json --out <queue root>
    python binary_review.py bank --root <queue root>          # fills audit/surprise
    python audit_to_corrections.py --task surprise --audit-dir surprise --reviewer <name> --apply
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import date
from pathlib import Path
from typing import Any, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import docmarks_config as cfg  # noqa: E402

TASK = "surprise"
AUDIT_DIR = "surprise"
QUESTIONS_PER_CONTROL = 15
#: Methods whose hits are worth a person's time.  A hit is only a surprise from
#: a method that ranks above chance: on v4.1 tier m, VLAD and its re-rank score
#: mean AP 0.001 (SigLIP 0.084), so their "top" presumed negatives are random
#: pages -- an expensive uniform sample, which the contamination checks already
#: are.  Restricting to SigLIP took the first pass from 1,246 pages to 516.
DEFAULT_METHODS = ("siglip",)
SEED = 20260923


def pages_to_review(
    hits: dict[str, Any], classes: dict[str, Any], methods: Optional[Sequence[str]] = None
) -> dict[str, list[dict[str, Any]]]:
    """Per class, every hit page across tiers and methods, once, best rank first.

    A page someone already ruled on for the class -- a positive, a reviewed
    negative, an exclusion -- is dropped: the run can only have ranked it because
    it is in the pool, but the pool may have moved since.
    """
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for tier, by_class in sorted(hits.items()):
        for cid, by_method in sorted(by_class.items()):
            meta = classes.get(cid)
            if meta is None:
                continue
            ruled = (
                set(meta.get("page_ids", []))
                | set(meta.get("reviewed_negative_page_ids", []))
                | set(meta.get("excluded_page_ids", []))
            )
            for method, rows in sorted(by_method.items()):
                if methods is not None and method not in methods:
                    continue
                for r in rows:
                    pid = r["page_id"]
                    if pid in ruled:
                        continue
                    seen = out.setdefault(cid, {}).setdefault(
                        pid, {"page_id": pid, "best_rank": r["rank"], "found_by": []}
                    )
                    seen["best_rank"] = min(seen["best_rank"], r["rank"])
                    seen["found_by"].append(f"{tier}:{method}@{r['rank']}")
    return {cid: sorted(v.values(), key=lambda x: (x["best_rank"], x["page_id"])) for cid, v in out.items()}


def translate_surprise(
    rows: list[dict[str, Any]], questions: dict[str, dict[str, Any]], votes: dict[str, str]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Fill ``verdict`` (good/bad) on each slate row from the vote on its question."""
    out, unanswered = [], []
    for r in rows:
        r = dict(r)
        fn = r.get("filename")
        if r.get("task") == TASK and fn in questions:
            if fn in votes:
                r["verdict"] = votes[fn]
                r["verdict_source"] = "vtsearch"
            else:
                unanswered.append(f"{r['class_id']}: {r['page_id']} unanswered")
        out.append(r)
    return out, unanswered


def apply_surprise(
    classes: dict[str, Any], rows: Sequence[dict[str, Any]], reviewer: Optional[str] = None
) -> tuple[list[str], list[str], dict[str, list[str]], dict[str, list[str]]]:
    """Answered rows onto class metadata; returns ``(changes, problems, negatives, exclusions)``.

    Controls are scored, never applied: a missed control is an attention
    failure to report, not a ruling that a known positive lacks its mark.
    """
    changes: list[str] = []
    problems: list[str] = []
    negatives: dict[str, list[str]] = {}
    exclusions: dict[str, list[str]] = {}
    controls: dict[str, list[bool]] = {}
    for r in rows:
        if r.get("task") != TASK or r.get("verdict") not in ("good", "bad"):
            continue
        cid, pid = r["class_id"], r["page_id"]
        if cid not in classes:
            problems.append(f"{cid}: not in classes.json")
            continue
        if r.get("arm") == "control":
            controls.setdefault(cid, []).append(r["verdict"] == "good")
            continue
        (exclusions if r["verdict"] == "good" else negatives).setdefault(cid, []).append(pid)
    for cid in sorted(set(negatives) | set(exclusions)):
        meta = classes[cid]
        positives = set(meta.get("page_ids", []))
        out = sorted((set(meta.get("excluded_page_ids", [])) | set(exclusions.get(cid, []))) - positives)
        if out:
            meta["excluded_page_ids"] = out
        meta["reviewed_negative_page_ids"] = sorted(
            (set(meta.get("reviewed_negative_page_ids", [])) | set(negatives.get(cid, []))) - positives - set(out)
        )
        meta.setdefault("audit", {})[f"{TASK}_checked"] = {
            "reviewed_by": reviewer,
            "reviewed_on": date.today().isoformat(),
            "negatives": len(negatives.get(cid, [])),
            "carry_the_mark": len(exclusions.get(cid, [])),
        }
        changes.append(
            f"{cid}: {len(negatives.get(cid, []))} known negative(s), "
            f"{len(exclusions.get(cid, []))} page(s) carry the mark -> excluded"
            + (f" (box them: {sorted(exclusions[cid])})" if exclusions.get(cid) else "")
        )
    for cid, got in sorted(controls.items()):
        if not all(got):
            problems.append(f"{cid}: {got.count(False)} of {len(got)} planted control(s) missed")
    return changes, problems, negatives, exclusions


def main(argv: Optional[Sequence[str]] = None) -> int:
    from binary_review import PREFIX, class_refs, emit  # noqa: PLC0415
    from banded_review import question  # noqa: PLC0415
    from contamination_sample import controls_for  # noqa: PLC0415
    from sources._common import read_manifest  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("emit")
    e.add_argument(
        "--hits",
        type=Path,
        action="append",
        required=True,
        help="surprise_hits.json from eval_retrieval.py; repeatable",
    )
    e.add_argument("--methods", default=",".join(DEFAULT_METHODS), help="comma-separated; see DEFAULT_METHODS")
    e.add_argument("--corpus", type=Path, default=cfg.OUT)
    e.add_argument("--out", type=Path, required=True, help="queue root; one directory per class")
    args = ap.parse_args(argv)

    classes = json.loads((args.corpus / "classes.json").read_text(encoding="utf-8"))
    pages = {p.page_id: p for p in read_manifest(args.corpus / "corpus.jsonl")}
    hits: dict[str, Any] = {}
    for path in args.hits:
        hits.update(json.loads(path.read_text(encoding="utf-8")))
    todo = pages_to_review(hits, classes, methods=args.methods.split(","))
    slate = args.corpus / "audit" / AUDIT_DIR / "verdicts.jsonl"
    if slate.exists():
        raise SystemExit(f"{slate} exists; it may hold answers, so it is never overwritten")
    rng = random.Random(SEED)
    rows: list[dict[str, Any]] = []
    for cid, hits in sorted(todo.items()):
        refs = class_refs(cid, classes, pages)
        by_page = {h["page_id"]: h for h in hits}
        controls = controls_for(
            cid,
            classes,
            pages,
            refs,
            rng,
            max(1, round(len(hits) / QUESTIONS_PER_CONTROL)),
            prefer_source=classes[cid].get("source", "ucsf"),
        )
        items = [(h["page_id"], "hit") for h in hits] + [(p, "control") for p in controls]
        rng.shuffle(items)
        qs = []
        for i, (pid, arm) in enumerate(items):
            stem = "surprise__" + cid.split("/")[-1].replace("-", "_")
            q = question(cid, i, pid, arm, pages[pid], refs, stem=stem)
            q.task = TASK
            q.key = {"class_id": cid, "page_id": pid, "arm": arm}
            qs.append(q)
            rows.append(
                {
                    "task": TASK,
                    "class_id": cid,
                    "page_id": pid,
                    "arm": arm,
                    "filename": q.filename,
                    "best_rank": by_page.get(pid, {}).get("best_rank"),
                    "found_by": by_page.get(pid, {}).get("found_by", []),
                    "verdict": "",
                }
            )
        name = f"{PREFIX} {cid.split('/')[-1]} -- is this mark on this page? (top hits)"
        qdir = args.out / (cid.replace("/", "_").replace("-", "_") + "__surprise")
        emit(qs, qdir, name, pages=pages, corpus=args.corpus)
        print(f"  {cid}: {len(hits)} hit(s) + {len(controls)} planted -> {qdir}", flush=True)
    slate.parent.mkdir(parents=True, exist_ok=True)
    slate.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"slate: {slate} ({len(rows)} rows); seed {SEED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
