#!/usr/bin/env python
"""Fold filled-in audit verdicts back into the corpus.

    python audit_to_corrections.py --task membership --apply
    python audit_to_corrections.py --task cluster --apply
    python audit_to_corrections.py --task confusable --apply
    python audit_to_corrections.py --task letterhead          # dry run (default)

Without ``--apply`` it prints what it would change and touches nothing.

Verdicts are additive and idempotent: they are recorded in ``classes.json``
under each class's ``audit`` block, and re-running with the same verdict file is
a no-op.  Nothing is ever deleted — a class judged ``generic`` keeps all its
instances and simply stops being part of the headline stratum, so both numbers
stay available and the decision stays visible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import docmarks_config as cfg  # noqa: E402
from sources._common import Mark, Page, read_manifest, write_manifest  # noqa: E402


def load_verdicts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"no verdict file at {path} — run make_audit_slate.py first")
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                if str(row.get("verdict", "")).strip():
                    out.append(row)
    return out


def apply_cluster(
    pages: list[Page], classes: dict[str, Any], verdicts: list[dict[str, Any]]
) -> tuple[list[str], list[str], list[str]]:
    """``ok`` / ``split`` / ``merge_into:<id>`` / ``drop`` on derived classes.

    Returns ``(changes, problems, resplit)``; *resplit* names classes the
    reviewer judged over-merged, which the caller re-clusters at a tighter
    threshold.
    """
    changes: list[str] = []
    problems: list[str] = []
    resplit: list[str] = []

    for row in verdicts:
        class_id = row["class_id"]
        verdict = str(row["verdict"]).strip()
        meta = classes.get(class_id)
        if meta is None:
            problems.append(f"{class_id}: not in classes.json")
            continue

        if verdict == "ok":
            meta["audit"]["cluster_ok"] = True
            changes.append(f"{class_id}: confirmed")
        elif verdict == "split":
            # A contact sheet says "this holds more than one mark"; it cannot
            # say which crop belongs to which. Rather than leave the class dead,
            # re-cluster *only its own instances* at a tighter threshold and
            # re-sheet the pieces. That converges: each round either resolves
            # into confirmable classes or gets split again, and no other class
            # is disturbed.
            meta["audit"]["cluster_ok"] = False
            meta["audit"]["notes"] = (row.get("notes") or "over-merged; re-cluster this class alone").strip()
            resplit.append(class_id)
            changes.append(f"{class_id}: queued for re-clustering at a tighter threshold")
        elif verdict.startswith("merge_into:"):
            target = verdict.split(":", 1)[1].strip()
            if target not in classes:
                problems.append(f"{class_id}: merge target {target!r} does not exist")
                continue
            for page in pages:
                for i, mark in enumerate(page.marks):
                    if mark.class_id == class_id:
                        page.marks[i] = Mark(mark.kind, mark.box, target, mark.provenance)
            classes[target]["n_instances"] += meta["n_instances"]
            classes[target]["page_ids"] = sorted(set(classes[target]["page_ids"]) | set(meta["page_ids"]))
            classes[target]["audit"]["cluster_ok"] = True
            classes.pop(class_id)
            changes.append(f"{class_id}: merged into {target}")
        elif verdict == "drop":
            for page in pages:
                page.marks = [m for m in page.marks if m.class_id != class_id]
            classes.pop(class_id)
            changes.append(f"{class_id}: dropped")
        else:
            problems.append(f"{class_id}: unrecognised verdict {verdict!r}")

    return changes, problems, resplit


def apply_membership(
    pages: list[Page], classes: dict[str, Any], verdicts: list[dict[str, Any]]
) -> tuple[list[str], list[str]]:
    """Remove hand-rejected instances and mark the class fully verified.

    A rejected crop loses its ``class_id`` but keeps its box and stays on its
    page.  It is not deleted, for two reasons: the page remains a *known*
    negative for this class — same scanner, same paper, verified clean, which is
    the hardest and most useful kind of negative — and the mark itself is still
    a real mark that a later roster may want.

    Setting ``membership_verified`` is the point of the pass.  Before it, a
    class is a clustering proposal; after it, every positive in the eval has
    been looked at, so a miss is the detector's fault and not possibly the
    label's.
    """
    changes: list[str] = []
    problems: list[str] = []

    for row in verdicts:
        class_id = row["class_id"]
        meta = classes.get(class_id)
        if meta is None:
            problems.append(f"{class_id}: not in classes.json")
            continue

        raw = str(row["verdict"]).strip().lower()
        page_ids: list[str] = row.get("page_ids", [])
        if raw == "ok":
            rejected_idx: list[int] = []
        else:
            try:
                rejected_idx = sorted({int(tok) for tok in raw.replace(" ", "").split(",") if tok})
            except ValueError:
                problems.append(f"{class_id}: verdict must be 'ok' or comma-separated indices, got {raw!r}")
                continue
        out_of_range = [i for i in rejected_idx if not 0 <= i < len(page_ids)]
        if out_of_range:
            problems.append(f"{class_id}: index/indices {out_of_range} are outside 0..{len(page_ids) - 1}")
            continue

        dropped = {page_ids[i] for i in rejected_idx}
        if dropped:
            for page in pages:
                if page.page_id in dropped:
                    page.marks = [
                        Mark(m.kind, m.box, None, m.provenance) if m.class_id == class_id else m for m in page.marks
                    ]
            meta["page_ids"] = [p for p in meta["page_ids"] if p not in dropped]
            meta["n_instances"] = len(meta["page_ids"])

        meta["audit"]["membership_verified"] = True
        meta["audit"]["rejected_page_ids"] = sorted(dropped)
        changes.append(f"{class_id}: verified, {len(dropped)} rejected, {meta['n_instances']} instance(s) remain")
    return changes, problems


def apply_distinctive(classes: dict[str, Any], verdicts: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    changes: list[str] = []
    problems: list[str] = []
    for row in verdicts:
        class_id = row["class_id"]
        verdict = str(row["verdict"]).strip().lower()
        meta = classes.get(class_id)
        if meta is None:
            problems.append(f"{class_id}: not in classes.json")
            continue
        if verdict not in ("distinctive", "generic"):
            problems.append(f"{class_id}: unrecognised verdict {verdict!r}")
            continue
        meta["audit"]["distinctive"] = verdict == "distinctive"
        changes.append(f"{class_id}: {verdict}")
    return changes, problems


def apply_confusable(
    pages: list[Page],
    classes: dict[str, Any],
    verdicts: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """``same`` merges the two classes; ``different`` separates them for good.

    Both verdicts are recorded as page-id pairs and replayed on every future
    re-cluster, so an afternoon of merging is not undone the next time a
    threshold moves.

    The two are deliberately not symmetric in cost, which is the whole reason
    the threshold runs strict: a split shows up here as one obvious pair to
    merge, while a bad merge never shows up at all.

    A ``different`` verdict is the only way the corpus can state "these must be
    told apart", and it is stored against **page ids** rather than class ids so
    it survives every future re-cluster: class ids move when a threshold moves,
    page ids do not. Without that, each rebuild would silently discard the
    adjudication it was supposed to be built on.
    """
    changes: list[str] = []
    problems: list[str] = []
    separations: list[dict[str, Any]] = []
    merges: list[dict[str, Any]] = []
    #: A class merged away is gone from `classes`, but a later verdict may
    #: still name it; follow the chain rather than reporting a missing class.
    moved: dict[str, str] = {}

    def resolve(cid: str) -> str:
        seen = set()
        while cid in moved and cid not in seen:
            seen.add(cid)
            cid = moved[cid]
        return cid

    for row in verdicts:
        left, right = resolve(row["left_class_id"]), resolve(row["right_class_id"])
        verdict = str(row["verdict"]).strip().lower()
        if left == right:
            changes.append(f"{row['left_class_id']} / {row['right_class_id']}: already one class")
            continue
        lmeta, rmeta = classes.get(left), classes.get(right)
        if lmeta is None or rmeta is None:
            problems.append(f"{left} / {right}: one of the pair is not in classes.json")
            continue

        if verdict == "same":
            # Merge into the larger class, so the surviving id is the one whose
            # instances dominate it and the name keeps meaning what it meant.
            keep, gone = (left, right) if lmeta["n_instances"] >= rmeta["n_instances"] else (right, left)
            kmeta, gmeta = classes[keep], classes[gone]
            merges.append(
                {
                    "left_page_id": kmeta["page_ids"][0],
                    "right_page_id": gmeta["page_ids"][0],
                    "kept_class_id": keep,
                    "merged_class_id": gone,
                    "note": row.get("notes", ""),
                }
            )
            for page in pages:
                for i, mark in enumerate(page.marks):
                    if mark.class_id == gone:
                        page.marks[i] = Mark(mark.kind, mark.box, keep, mark.provenance)
            kmeta["page_ids"] = sorted(set(kmeta["page_ids"]) | set(gmeta["page_ids"]))
            kmeta["n_instances"] = len(kmeta["page_ids"])
            classes.pop(gone)
            moved[gone] = keep
            changes.append(f"{gone} merged into {keep} ({kmeta['n_instances']} instances)")
        elif verdict == "different":
            lmeta.setdefault("distinct_from", [])
            rmeta.setdefault("distinct_from", [])
            if right not in lmeta["distinct_from"]:
                lmeta["distinct_from"].append(right)
            if left not in rmeta["distinct_from"]:
                rmeta["distinct_from"].append(left)
            # One representative page per side is enough to pin the constraint,
            # and keeps the store small; the cannot-link propagates to the whole
            # group through union-find.
            separations.append(
                {
                    "left_page_id": lmeta["page_ids"][0],
                    "right_page_id": rmeta["page_ids"][0],
                    "left_class_id": left,
                    "right_class_id": right,
                    "note": row.get("notes", ""),
                }
            )
            changes.append(f"{left} != {right}: separation recorded")
        else:
            problems.append(f"{left} / {right}: unrecognised verdict {verdict!r} (expected same|different)")

    return changes, problems, separations, merges


def apply_letterhead(classes: dict[str, Any], verdicts: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    changes: list[str] = []
    problems: list[str] = []
    for row in verdicts:
        author = row.get("author", "?")
        try:
            hits = int(str(row["verdict"]).strip())
        except ValueError:
            problems.append(f"{author}: verdict must be the count of bands carrying a printed mark")
            continue
        sampled = int(row.get("sampled", 0)) or 1
        yield_frac = hits / sampled
        flag = "" if yield_frac >= 0.5 else "  <- under half; this pool may not be worth clustering"
        changes.append(f"{author}: candidate yield {yield_frac:.2f} ({hits}/{sampled}){flag}")
    return changes, problems


def resplit_classes(
    pages: list[Page],
    classes: dict[str, Any],
    class_ids: Sequence[str],
    *,
    backend: str,
    threshold: float,
    factor: float = 0.5,
) -> list[str]:
    """Re-cluster each over-merged class alone, at a tighter threshold.

    Only that class's own instances are touched, so re-splitting one class can
    never disturb another's already-confirmed membership.  The resulting pieces
    come back as fresh candidate classes for the next ``cluster`` sheet.
    """
    from cluster_marks import assign_class_ids, describe_marks, distance_matrix, single_linkage

    notes: list[str] = []
    tighter = threshold * factor
    for class_id in class_ids:
        meta = classes.get(class_id)
        if meta is None:
            continue
        refs = _refs_for_class(pages, class_id)
        if len(refs) < 2:
            continue
        desc = describe_marks(pages, refs, backend=backend)
        dist = distance_matrix(desc, refs, backend=backend)
        labels = single_linkage(dist, tighter)
        source = class_id.split("/", 1)[0]
        provenance = "clustered_band" if meta.get("located_by") == "band" else "clustered"
        pieces = assign_class_ids(pages, refs, labels, source=source, provenance=provenance)
        classes.pop(class_id, None)
        notes.append(f"{class_id}: re-clustered at {tighter:.3f} into {len(pieces)} piece(s)")
    return notes


def _refs_for_class(pages: list[Page], class_id: str) -> list[Any]:
    from cluster_marks import MarkRef

    return [
        MarkRef(pi, mi, page.page_id, mark.kind, mark.box)
        for pi, page in enumerate(pages)
        for mi, mark in enumerate(page.marks)
        if mark.class_id == class_id
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--task", required=True, choices=("membership", "cluster", "confusable", "distinctive", "letterhead")
    )
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--apply", action="store_true", help="write the changes (default is a dry run)")
    ap.add_argument("--cluster-backend", default=cfg.CLUSTER_BACKEND, choices=("phash", "siglip"))
    ap.add_argument("--cluster-threshold", type=float, default=cfg.CLUSTER_THRESHOLD)
    args = ap.parse_args(argv)

    classes_path = args.corpus / "classes.json"
    manifest_path = args.corpus / "corpus.jsonl"
    adjudications_path = args.corpus / "adjudications.json"
    classes = json.loads(classes_path.read_text(encoding="utf-8"))
    verdicts = load_verdicts(args.corpus / "audit" / args.task / "verdicts.jsonl")

    mutates_pages = args.task in ("cluster", "membership", "confusable")
    pages = list(read_manifest(manifest_path)) if mutates_pages else []
    new_separations: list[dict[str, Any]] = []
    new_merges: list[dict[str, Any]] = []
    resplit: list[str] = []

    if args.task == "membership":
        changes, problems = apply_membership(pages, classes, verdicts)
    elif args.task == "cluster":
        changes, problems, resplit = apply_cluster(pages, classes, verdicts)
    elif args.task == "confusable":
        changes, problems, new_separations, new_merges = apply_confusable(pages, classes, verdicts)
    elif args.task == "distinctive":
        changes, problems = apply_distinctive(classes, verdicts)
    else:
        changes, problems = apply_letterhead(classes, verdicts)

    for c in changes:
        print(f"  {c}")
    for p in problems:
        print(f"  PROBLEM: {p}")
    print(f"\n{len(changes)} change(s), {len(problems)} problem(s) from {len(verdicts)} filled verdict(s)")

    if not args.apply:
        print("dry run — pass --apply to write")
        return 1 if problems else 0

    if resplit:
        for note in resplit_classes(
            pages, classes, resplit, backend=args.cluster_backend, threshold=args.cluster_threshold
        ):
            print(f"  {note}")
        print("  re-run make_audit_slate.py --task cluster to review the new pieces")

    if new_separations or new_merges:
        from cluster_marks import load_adjudications, save_adjudications

        old_same, old_diff = load_adjudications(adjudications_path)
        save_adjudications(
            [{"left_page_id": a, "right_page_id": b} for a, b in old_same] + new_merges,
            [{"left_page_id": a, "right_page_id": b} for a, b in old_diff] + new_separations,
            adjudications_path,
        )
        print(f"  wrote {len(new_merges)} merge(s) and {len(new_separations)} separation(s) to {adjudications_path}")

    classes_path.write_text(json.dumps(classes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if mutates_pages:
        write_manifest(pages, manifest_path)
    print(f"wrote {classes_path}" + (f" and {manifest_path}" if mutates_pages else ""))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
