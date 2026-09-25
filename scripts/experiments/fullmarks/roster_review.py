"""Identity and membership for newly admitted roster classes, as Good/Bad VTSearch queues (#4143).

A class admitted by ``roster_grow.py admit`` is a clustering proposal until two
questions are settled by a person, the same two every roster class answered:

* **identity** -- is it a mark of its own?  Each new class is shown against its
  nearest existing classes (SigLIP cosine between query crops) and against the
  other new classes of its source.  "Same mark as the left?"  Good merges the
  two classes; Bad separates them for good.  Compiled to ``confusable`` rows,
  applied with ``audit_to_corrections.py --task confusable``.
* **membership** -- is every instance this mark?  One question per instance.
  Compiled to ``membership`` rows (``ok`` or the rejected indices), applied
  with ``audit_to_corrections.py --task membership --reviewer <name>``.

* **completeness** -- copies clustering missed.  ``completeness.py --classes``
  turns an ``eval_sift_rank.py`` run into candidates; ``completeness`` here
  queues them ("Is the left mark in the red box?"), and the answers apply with
  ``audit_to_corrections.py --task completeness --reviewer <name>``.

Slates are written under ``<corpus>/audit/roster_identity`` and
``roster_membership``; ``binary_review.py bank --corpus <corpus>`` fills them.

    python roster_review.py emit --corpus <staging> --classes a,b,c --root <queue root>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

IDENTITY = "roster_identity"
MEMBERSHIP = "roster_membership"
COMPLETENESS = "roster_completeness"
#: Existing classes each new class is compared with, nearest first.
NEAREST = 3


def _mark_box(page: Any, class_id: str) -> Optional[list[int]]:
    for m in page.marks:
        if m.class_id == class_id:
            return [int(v) for v in m.box]
    return None


def identity_pairs(
    new: Sequence[str], classes: dict[str, Any], similarity: dict[tuple[str, str], float], k: int = NEAREST
) -> list[tuple[str, str]]:
    """``(new class, other class)`` pairs to ask about, each unordered pair once.

    Every new class against its *k* most similar existing classes of the same
    source, and against every other new class of the same source.
    """
    new_set = set(new)
    pairs: set[tuple[str, str]] = set()
    for a in new:
        src = classes[a]["source"]
        old = [c for c in classes if c not in new_set and classes[c]["source"] == src and classes[c].get("on_roster")]
        old.sort(key=lambda c: -similarity.get((a, c), similarity.get((c, a), 0.0)))
        pairs.update(tuple(sorted((a, c))) for c in old[:k])
        pairs.update(tuple(sorted((a, b))) for b in new if b != a and classes[b]["source"] == src)
    return sorted(pairs)


def translate_identity(rows, questions, votes, drawn=None):
    """Good = the same mark (``same``); Bad = a different one (``different``)."""
    out, unanswered = [], []
    by_sheet = {q["key"]["sheet"]: fn for fn, q in questions.items() if q["task"] == IDENTITY}
    for r in rows:
        r = dict(r)
        fn = by_sheet.get(r["sheet"])
        if fn is not None:
            if fn in votes:
                r["verdict"] = "same" if votes[fn] == "good" else "different"
                r["verdict_source"] = "vtsearch"
            else:
                unanswered.append(f"{r['left_class_id']} vs {r['right_class_id']}: unanswered")
        out.append(r)
    return out, unanswered


def translate_membership(rows, questions, votes, drawn=None):
    """``ok`` when every instance is Good, else the Bad instances' indices; blank until all are answered."""
    got: dict[str, dict[int, Optional[str]]] = {}
    for fn, q in questions.items():
        if q["task"] == MEMBERSHIP:
            got.setdefault(q["key"]["class_id"], {})[q["key"]["index"]] = votes.get(fn)
    out, unanswered = [], []
    for r in rows:
        r = dict(r)
        asked = got.get(r["class_id"], {})
        if len(asked) < len(r["page_ids"]) or any(v is None for v in asked.values()):
            missing = len(r["page_ids"]) - sum(1 for v in asked.values() if v is not None)
            unanswered.append(f"{r['class_id']}: {missing} of {len(r['page_ids'])} unanswered")
        else:
            bad = sorted(i for i, v in asked.items() if v == "bad")
            r["verdict"] = ",".join(map(str, bad)) if bad else "ok"
            r["verdict_source"] = "vtsearch"
        out.append(r)
    return out, unanswered


def translate_completeness(rows, questions, votes, drawn=None):
    """Good candidates are accepted (``"0,3"``), ``"none"`` when all are Bad; blank until all are answered.

    A box drawn with a Good vote is kept on the candidate as ``drawn_box``.  With
    no existing mark it also becomes the candidate's ``box``, the box the new
    mark is added with; on an existing mark it is applied afterwards as a
    reviewed box override (:func:`drawn_overrides`).
    """
    drawn = drawn or {}
    by_cand = {
        (q["key"]["class_id"], q["key"]["index"]): fn for fn, q in questions.items() if q["task"] == COMPLETENESS
    }
    got: dict[str, dict[int, Optional[str]]] = {}
    for fn, q in questions.items():
        if q["task"] == COMPLETENESS:
            got.setdefault(q["key"]["class_id"], {})[q["key"]["index"]] = votes.get(fn)
    out, unanswered = [], []
    for r in rows:
        r = dict(r)
        asked = got.get(r["class_id"])
        if asked is None:
            out.append(r)
            continue
        if any(v is None for v in asked.values()):
            unanswered.append(f"{r['class_id']}: {sum(v is None for v in asked.values())} of {len(asked)} unanswered")
        else:
            good = sorted(i for i, v in asked.items() if v == "good")
            r["verdict"] = ",".join(map(str, good)) if good else "none"
            r["verdict_source"] = "vtsearch"
        cands = []
        for c in r.get("candidates", []):
            box = drawn.get(by_cand.get((r["class_id"], c["index"])))
            if box is not None:
                c = dict(c, drawn_box=list(box))
                if c.get("mark_index") is None:
                    c["box"] = list(box)
            cands.append(c)
        r["candidates"] = cands
        out.append(r)
    return out, unanswered


def drawn_overrides(rows: Sequence[dict[str, Any]], pages: dict[str, Any]) -> list[dict[str, Any]]:
    """``box_tighten`` rows for accepted candidates on existing marks whose box the reviewer redrew.

    Applied with ``audit_to_corrections.py --task box_tighten`` after the
    completeness apply, so the mark carries the reviewer's box, stored in
    ``box_overrides.json`` and replayed by a rebuild.
    """
    out = []
    for r in rows:
        verdict = str(r.get("verdict", ""))
        accepted = {int(i) for i in verdict.split(",")} if verdict not in ("", "none") else set()
        members = []
        for c in r["candidates"]:
            idx, box = c.get("mark_index"), c.get("drawn_box")
            if c["index"] not in accepted or idx is None or box is None or c["page_id"] not in pages:
                continue
            old = [int(v) for v in pages[c["page_id"]].marks[idx].box]
            if old != list(box):
                members.append(
                    {
                        "index": len(members),
                        "page_id": c["page_id"],
                        "mark_index": idx,
                        "old_box": old,
                        "new_box": list(box),
                    }
                )
        if members:
            out.append(
                {
                    "task": "box_tighten",
                    "class_id": r["class_id"],
                    "members": members,
                    "verdict": ",".join(str(m["index"]) for m in members),
                    "verdict_source": "boxes the reviewer drew on completeness candidates",
                }
            )
    return out


def completeness_questions(rows, classes, pages, stem: str = "complete") -> list[tuple[str, list[Any]]]:
    """One queue per class from ``completeness.py``'s slate: is the candidate this mark?

    *stem* prefixes every filename.  A later round must use a new one: the bank
    matches votes to questions by filename, reading old rounds' detector
    backups too, so a reused name would hand a new candidate an old answer.
    """
    from binary_review import PREFIX, Question, class_refs, slug  # noqa: PLC0415

    queues = []
    for r in rows:
        cid = r["class_id"]
        refs = class_refs(cid, classes, pages)
        qs = []
        for c in r["candidates"]:
            page = pages.get(c["page_id"])
            if page is None:
                continue
            # On an existing mark, a Good keeps THAT mark's box (apply_completeness
            # reassigns it), so show it.  The candidate's own box is the extent of
            # SIFT's matched keypoints, a subset of the mark (Sam, 2026-09-23).
            # With no mark to reassign, ask for the whole page and a drawn box.
            idx = c.get("mark_index")
            on_mark = idx is not None and idx < len(page.marks)
            qs.append(
                Question(
                    filename=f"{stem}__{slug(cid)}__{c['index']:03d}.jpg",
                    task=COMPLETENESS,
                    question="Is the left mark in the red box?"
                    if on_mark
                    else "Left mark on this page? Draw it, then Good.",
                    refs=refs,
                    page_id=c["page_id"],
                    box=list(page.marks[idx].box) if on_mark else [0, 0, page.width, page.height],
                    outline=on_mark,
                    key={"class_id": cid, "index": c["index"]},
                    item=f"{cid} candidate {c['index']} ({c['inliers']} inliers)",
                    ref_labels=False,
                    anonymous=True,
                    margin=0.4,
                )
            )
        if qs:
            queues.append((f"{PREFIX} {cid.split('/')[-1]} -- missing copy?", qs))
    return queues


def siglip_similarity(classes: dict[str, Any], ids: Sequence[str]) -> dict[tuple[str, str], float]:
    """Cosine between every pair of the named classes' query crops, with SigLIP."""
    import numpy as np  # noqa: PLC0415

    from eval_retrieval import embed_query  # noqa: PLC0415
    from vtscore.media import get_embedder  # noqa: PLC0415

    emb = get_embedder("siglip")
    emb.load_models()
    vecs = {}
    for cid in ids:
        v, _ = embed_query(emb, Path(classes[cid]["query_crop"]))
        if v is not None:
            v = np.asarray(v, dtype=np.float32)
            vecs[cid] = v / (np.linalg.norm(v) or 1.0)
    return {(a, b): float(vecs[a] @ vecs[b]) for a in vecs for b in vecs if a != b}


def main(argv: Optional[Sequence[str]] = None) -> int:
    from binary_review import PREFIX, Question, class_refs, emit, slug  # noqa: PLC0415
    from sources._common import read_manifest  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("completeness", help="queue completeness.py's candidates for the new classes")
    c.add_argument("--corpus", type=Path, required=True)
    c.add_argument("--root", type=Path, required=True)
    c.add_argument("--classes", default="", help="comma-separated; default every class in the slate")
    c.add_argument("--stem", default="complete", help="filename prefix; a new one per round")
    e = sub.add_parser("emit")
    e.add_argument("--corpus", type=Path, required=True)
    e.add_argument("--classes", required=True, help="the newly admitted class ids")
    e.add_argument("--root", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.cmd == "completeness":
        classes = json.loads((args.corpus / "classes.json").read_text(encoding="utf-8"))
        pages = {p.page_id: p for p in read_manifest(args.corpus / "corpus.jsonl")}
        slate = args.corpus / "audit" / "completeness" / "verdicts.jsonl"
        rows = [json.loads(line) for line in slate.read_text(encoding="utf-8").splitlines() if line]
        only = {c.strip() for c in args.classes.split(",") if c.strip()}
        rows = [r for r in rows if not only or r["class_id"] in only]
        for name, qs in completeness_questions(rows, classes, pages, stem=args.stem):
            d = args.root / (slug(name.split(" -- ")[0]) + "__complete")
            emit(qs, d, name, pages=pages, corpus=args.corpus)
            print(f"  {name}: {len(qs)} candidate(s) -> {d}")
        return 0

    classes = json.loads((args.corpus / "classes.json").read_text(encoding="utf-8"))
    new = [c.strip() for c in args.classes.split(",") if c.strip()]
    pages = {p.page_id: p for p in read_manifest(args.corpus / "corpus.jsonl")}
    roster_ids = [c for c, m in classes.items() if m.get("on_roster") and m.get("query_crop")]
    sim = siglip_similarity(classes, roster_ids)

    # identity
    rows, qs = [], []
    for i, (a, b) in enumerate(identity_pairs(new, classes, sim)):
        # the new class on the right, so the reviewer compares a candidate against a known class
        left, right = (a, b) if a not in new else (b, a)
        rpage = classes[right]["query_page_id"]
        sheet = f"identity__{i:03d}"
        rows.append(
            {
                "task": "confusable",
                "left_class_id": left,
                "right_class_id": right,
                "similarity": round(sim.get((left, right), 0.0), 4),
                "sheet": sheet,
                "verdict": "",
            }
        )
        qs.append(
            Question(
                filename=f"{sheet}.jpg",
                task=IDENTITY,
                question="Same mark as the left?",
                refs=class_refs(left, classes, pages),
                page_id=rpage,
                box=_mark_box(pages[rpage], right),
                key={"sheet": sheet},
                item=f"{left} vs {right}",
                ref_labels=False,
                anonymous=True,
                margin=0.15,
            )
        )
    _write(args.corpus / "audit" / IDENTITY, rows)
    emit(
        qs,
        args.root / "fullmarks_roster_identity",
        f"{PREFIX} new classes -- same mark as the left?",
        pages=pages,
        corpus=args.corpus,
    )
    print(f"identity: {len(qs)} pair(s)")

    # membership
    rows = []
    for cid in new:
        meta = classes[cid]
        refs = class_refs(cid, classes, pages)
        ids = list(meta["page_ids"])
        rows.append({"task": "membership", "class_id": cid, "n_instances": len(ids), "page_ids": ids, "verdict": ""})
        qs = [
            Question(
                filename=f"member__{slug(cid)}__{j:03d}.jpg",
                task=MEMBERSHIP,
                question="Same mark as the left?",
                refs=refs,
                page_id=pid,
                box=_mark_box(pages[pid], cid),
                key={"class_id": cid, "index": j},
                item=f"{cid} instance {j}",
                ref_labels=False,
                anonymous=True,
                margin=0.15,
            )
            for j, pid in enumerate(ids)
        ]
        emit(
            qs,
            args.root / f"{slug(cid)}__members",
            f"{PREFIX} {cid.split('/')[-1]} -- same mark as the left?",
            pages=pages,
            corpus=args.corpus,
        )
        print(f"membership {cid}: {len(qs)} instance(s)")
    _write(args.corpus / "audit" / MEMBERSHIP, rows)
    return 0


def _write(d: Path, rows: Sequence[dict[str, Any]]) -> None:
    if (d / "verdicts.jsonl").exists():
        raise SystemExit(f"{d / 'verdicts.jsonl'} exists; it may hold answers, so it is never overwritten")
    d.mkdir(parents=True, exist_ok=True)
    (d / "verdicts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
