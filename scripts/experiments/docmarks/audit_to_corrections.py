#!/usr/bin/env python
"""Fold filled-in audit verdicts back into the corpus.

    python audit_to_corrections.py --task merge --apply        # the merge slate
    python audit_to_corrections.py --task membership --apply
    python audit_to_corrections.py --task cluster --apply
    python audit_to_corrections.py --task confusable --apply
    python audit_to_corrections.py --task letterhead          # dry run (default)
    python audit_to_corrections.py --task completeness --reviewer <name> --apply
    python audit_to_corrections.py --task ucsf_classes --reviewer <name> --apply
    python audit_to_corrections.py --task query_crops --reviewer <name> --apply
    python audit_to_corrections.py --task box_tighten --reviewer <name> --apply
    python audit_to_corrections.py --task surprise --reviewer <name> --apply   # top-ranked presumed negatives (#4089)
    python audit_to_corrections.py --migrate-adjudications --apply

Without ``--apply`` it prints what it would change and touches nothing.

Verdicts are additive and idempotent: they are recorded in ``classes.json``
under each class's ``audit`` block, and re-running with the same verdict file is
a no-op.  Nothing is ever deleted — a class judged ``generic`` keeps all its
instances and simply stops being part of the headline stratum, so both numbers
stay available and the decision stays visible.

**Two files, two promises.**  ``classes.json`` is this corpus; whether a verdict
*survives the next build* is a question about ``adjudications.json``, which is
the only thing ``build_corpus.py`` replays.  So every verdict that changes the
partition — a merge, a separation, a split, and each instance a membership pass
accepts or rejects — is written to both.  Until #3343 the split and membership
passes wrote only the first, which meant the audit held exactly until somebody
rebuilt, and rebuilding is a documented step of the pipeline rather than an
accident.  ``--migrate-adjudications`` brings a pre-#3343 store up to the
current keying; see ``cluster_marks.resolve_pairs`` for why a bare page id is
no longer enough.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import docmarks_config as cfg  # noqa: E402
from sources._common import Mark, Page, read_manifest, write_manifest  # noqa: E402


def mark_indices(pages: Sequence[Page], class_id: str) -> dict[str, int]:
    """``page_id -> mark_index`` for every instance of *class_id*.

    Adjudications are keyed on ``(page_id, mark_index)``, and the index is the
    half a verdict file never carries: the reviewer ruled on a crop, the crop
    came from a class, and which of the page's marks that was is only knowable
    here, while the class ids are still on the manifest.  Look it up before
    mutating anything — ``apply_membership`` clears the ``class_id`` of a
    rejected mark, which is exactly the row an adjudication has to name.
    """
    out: dict[str, int] = {}
    for page in pages:
        for index, mark in enumerate(page.marks):
            if mark.class_id == class_id:
                out.setdefault(page.page_id, index)
    return out


def star(endpoints: Sequence[tuple[str, int]], note: str) -> list[dict[str, Any]]:
    """``n-1`` rows binding *endpoints* into one group, not ``n(n-1)/2``.

    Sameness is transitive and ``single_linkage`` applies every ``must_link``
    before any distance does, so a star is the whole constraint at a fraction
    of the rows.
    """
    rows: list[dict[str, Any]] = []
    first, *rest = endpoints
    for other in rest:
        rows.append(
            {
                "left_page_id": first[0],
                "left_mark_index": first[1],
                "right_page_id": other[0],
                "right_mark_index": other[1],
                "note": note,
            }
        )
    return rows


def migrate_adjudications(
    pages: Sequence[Page],
    classes: dict[str, Any],
    same_rows: Sequence[dict[str, Any]],
    diff_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Give every legacy page-id endpoint the mark index it always meant.

    A row written before #3343 names two pages and leaves which mark on each
    was ruled on to be inferred — and on these sources a page carries several,
    so ``resolve_pairs`` now refuses it rather than expanding it to every
    crossing.  The rows do carry enough to recover the answer: a ``different``
    row names the two classes, a ``same`` row names the class that survived the
    merge, and the marks still carry those ids on the manifest.  Where they do
    not, a page with exactly one classed mark settles it, and anything still
    ambiguous is **reported, not guessed**.

    Returns ``(same, different, problems)``.  Rows naming a page that is not in
    this corpus are passed through untouched: they belong to a tier or a build
    this one cannot speak for.
    """
    by_page = {page.page_id: page for page in pages}
    problems: list[str] = []

    def index_for(page_id: str, hints: Sequence[Optional[str]]) -> Optional[int]:
        page = by_page.get(page_id)
        if page is None:
            return None
        wanted = [h for h in hints if h]
        hit = [i for i, mark in enumerate(page.marks) if mark.class_id and mark.class_id in wanted]
        if len(hit) == 1:
            return hit[0]
        classed = [i for i, mark in enumerate(page.marks) if mark.class_id]
        if not wanted and len(classed) == 1:
            return classed[0]
        return None

    def fixed(row: dict[str, Any], direction: str) -> dict[str, Any]:
        out = dict(row)
        sides = {
            "left": [row.get("left_class_id"), row.get("kept_class_id")],
            "right": [row.get("right_class_id"), row.get("merged_class_id")],
        }
        if direction == "same":
            # Both endpoints are one class after the merge, and the manifest
            # relabelled the absorbed side — so each end answers to either id.
            shared = [h for hs in sides.values() for h in hs if h]
            if not shared:
                shared = [
                    cid
                    for cid, meta in classes.items()
                    if row["left_page_id"] in meta.get("page_ids", [])
                    and row["right_page_id"] in meta.get("page_ids", [])
                ]
            sides = {"left": shared, "right": shared}
        for side in ("left", "right"):
            page_id = row[f"{side}_page_id"]
            if row.get(f"{side}_mark_index") is not None or page_id not in by_page:
                continue
            index = index_for(page_id, sides[side])
            if index is None:
                problems.append(
                    f"{direction}: {row['left_page_id']} / {row['right_page_id']} — cannot tell which mark on "
                    f"{page_id} was ruled on; add {side}_mark_index by hand"
                )
                continue
            out[f"{side}_mark_index"] = index
        return out

    return (
        [fixed(r, "same") for r in same_rows],
        [fixed(r, "different") for r in diff_rows],
        problems,
    )


def pair_key(row: dict[str, Any]) -> tuple[tuple[str, int], tuple[str, int]]:
    """The two marks a row rules on, order-independent — the identity of a verdict."""

    def side(name: str) -> tuple[str, int]:
        index = row.get(f"{name}_mark_index")
        return (row[f"{name}_page_id"], -1 if index is None else int(index))

    left, right = sorted((side("left"), side("right")))
    return (left, right)


def describe_pair(row: dict[str, Any]) -> str:
    left, right = pair_key(row)
    names = [row.get("left_class_id"), row.get("right_class_id"), row.get("kept_class_id")]
    named = " / ".join(n for n in names if n)
    where = f"{left[0]}#{left[1]} vs {right[0]}#{right[1]}"
    return f"{named} ({where})" if named else where


def supersede(
    old_same: Sequence[dict[str, Any]],
    old_diff: Sequence[dict[str, Any]],
    new_merges: Sequence[dict[str, Any]],
    new_separations: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop stored rulings the new verdicts overturn.  Returns the survivors.

    A reviewer changing their mind is a normal event on a corpus adjudicated in
    rounds, and it is **not** the contradiction ``save_adjudications`` refuses.
    That refusal exists for a store holding a pair ruled both ways at once, where
    whichever is applied last would silently win.  Here the later verdict is the
    decision and the earlier one is history — but only a person can say which is
    which, so this reports the overlap and the caller requires ``--supersede``
    before any of it is dropped.

    The real case (#3343): the confusable pass ruled
    ``logo_afm90c00-first_1_0`` and ``logo_bad45f00_1`` **different** — one is a
    chief engraving and the other a "100 Years of Achievement" panel carrying
    that same engraving.  The corpus owner ruled them the **same** mark on the
    rule that a mark plus additional elements is still that mark.  Without a way
    to supersede, applying that ruling means hand-editing the store, which is
    the one thing every other path here exists to avoid.
    """
    same_keys = {pair_key(r) for r in new_merges}
    diff_keys = {pair_key(r) for r in new_separations}
    overturned = [r for r in old_diff if pair_key(r) in same_keys]
    overturned += [r for r in old_same if pair_key(r) in diff_keys]
    kept_same = [r for r in old_same if pair_key(r) not in diff_keys]
    kept_diff = [r for r in old_diff if pair_key(r) not in same_keys]
    return kept_same, kept_diff, overturned


#: What ``resplit_classes`` writes into a piece's ``audit.notes``.  Parsed, not
#: just displayed: on a corpus split before #3343 it is the only surviving
#: record of which parent a piece came out of.
RESPLIT_NOTE = re.compile(r"^re-clustered out of (?P<parent>\S+) at (?P<threshold>[0-9.]+)$")


def split_rows_from_notes(
    pages: Sequence[Page], classes: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Recover an already-applied split as the adjudications it should have written.

    A split applied before #3343 exists only as pieces in ``classes.json``, and
    the next build would fuse them back.  The pieces do each carry the parent's
    id in their audit note, and the manifest still says which marks are theirs,
    so the partition is recoverable exactly — no reviewer is being asked again
    and nothing is inferred beyond what they already ruled.
    """
    families: dict[str, list[str]] = {}
    for class_id, meta in sorted(classes.items()):
        match = RESPLIT_NOTE.match(str(meta.get("audit", {}).get("notes", "")))
        if match:
            families.setdefault(match.group("parent"), []).append(class_id)

    merges: list[dict[str, Any]] = []
    separations: list[dict[str, Any]] = []
    for parent, pieces in sorted(families.items()):
        anchors: list[tuple[str, int]] = []
        for piece in pieces:
            members = sorted(mark_indices(pages, piece).items())
            if not members:
                continue
            if len(members) > 1:
                merges.extend(star(members, f"split of {parent}: {piece} is one mark"))
            anchors.append(members[0])
        for i, left in enumerate(anchors):
            for right in anchors[i + 1 :]:
                separations.append(
                    {
                        "left_page_id": left[0],
                        "left_mark_index": left[1],
                        "right_page_id": right[0],
                        "right_mark_index": right[1],
                        "pins": "classes",
                        "note": f"split of {parent}",
                    }
                )
    return merges, separations


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
    pages: list[Page],
    classes: dict[str, Any],
    verdicts: list[dict[str, Any]],
    *,
    reviewer: Optional[str] = None,
) -> tuple[list[str], list[str], list[dict[str, Any]], list[dict[str, Any]]]:
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

    Returns ``(changes, problems, merges, separations)``.  The verdict is
    recorded in ``adjudications.json`` as well as in ``classes.json``, because
    those two files are not the same promise: ``classes.json`` is *this*
    corpus, and the adjudications are what the decision is replayed from when
    the corpus is rebuilt.  Until this wrote them, a membership pass was
    durable only for as long as nobody re-ran the builder — which stage 3 of
    #3343 does, on purpose, to stamp the roster.

    A verified class is a hand-confirmed group of same-mark instances, so it is
    recorded as a must-link star, and each rejected crop as one cannot-link
    against it.  One row per rejection is enough precisely *because* the star
    exists: ``single_linkage`` applies must-links first, so by the time the
    separation is registered the class is already one group and the constraint
    lands on all of it.
    """
    changes: list[str] = []
    problems: list[str] = []
    merges: list[dict[str, Any]] = []
    separations: list[dict[str, Any]] = []

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
        # Before the mutation, while the manifest still says which mark is which.
        located = mark_indices(pages, class_id)
        kept = [(p, located[p]) for p in meta["page_ids"] if p not in dropped and p in located]
        if dropped:
            for page in pages:
                if page.page_id in dropped:
                    page.marks = [
                        Mark(m.kind, m.box, None, m.provenance) if m.class_id == class_id else m for m in page.marks
                    ]
            meta["page_ids"] = [p for p in meta["page_ids"] if p not in dropped]
            meta["n_instances"] = len(meta["page_ids"])

        if len(kept) > 1:
            merges.extend(star(kept, f"membership: every instance of {class_id} confirmed by hand"))
        for page_id in sorted(dropped):
            if page_id not in located or not kept:
                continue
            separations.append(
                {
                    "left_page_id": page_id,
                    "left_mark_index": located[page_id],
                    "right_page_id": kept[0][0],
                    "right_mark_index": kept[0][1],
                    "right_class_id": class_id,
                    "note": f"membership: rejected from {class_id}",
                }
            )

        meta["audit"]["membership_verified"] = True
        meta["audit"]["rejected_page_ids"] = sorted(dropped)
        meta["audit"]["reviewed_by"] = reviewer
        meta["audit"]["reviewed_on"] = date.today().isoformat()
        changes.append(f"{class_id}: verified, {len(dropped)} rejected, {meta['n_instances']} instance(s) remain")
    return changes, problems, merges, separations


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
            kept_at, gone_at = mark_indices(pages, keep), mark_indices(pages, gone)
            merges.append(
                {
                    "left_page_id": kmeta["page_ids"][0],
                    "left_mark_index": kept_at.get(kmeta["page_ids"][0]),
                    "right_page_id": gmeta["page_ids"][0],
                    "right_mark_index": gone_at.get(gmeta["page_ids"][0]),
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
            # The two writers of class ids do not agree, and the disagreement is
            # invisible until a rebuild.  Merging keeps the *larger* class's id,
            # so the name goes on meaning what it meant; `assign_class_ids`
            # derives an id from the group's *smallest page id*, so the next
            # build renames this class.  Anything naming it by then -- a roster,
            # a report, a verdict file -- is pointing at a class that no longer
            # exists.  Measured on the real corpus (#3343): the `DY.Secretary`
            # merge of #3561 was `spods/stamp_00551_1` and came back
            # `spods/stamp_00546_1`.  Say so here rather than leaving it to the
            # roster-drift warning three steps downstream.
            located = {**gone_at, **kept_at}
            if located:
                anchor_page, anchor_mark = min(located.items())
                predicted = (
                    f"{keep.split('/')[0]}/{kmeta.get('kind', 'mark')}_{anchor_page.split('/')[-1]}_{anchor_mark}"
                )
                if predicted != keep:
                    changes.append(
                        f"  RENAME AHEAD: the next rebuild will call {keep} {predicted} — a class id comes "
                        "from its group's smallest page id, not from the side a merge kept. Use that name in "
                        "any roster or report written before the rebuild."
                    )
        elif verdict == "different":
            lmeta.setdefault("distinct_from", [])
            rmeta.setdefault("distinct_from", [])
            if right not in lmeta["distinct_from"]:
                lmeta["distinct_from"].append(right)
            if left not in rmeta["distinct_from"]:
                rmeta["distinct_from"].append(left)
            # One representative mark per side is enough to pin the constraint,
            # and keeps the store small — but only because the membership pass
            # ran first and must-linked each class into one group, which
            # `single_linkage` forms before any separation is registered.  On a
            # class that is still a clustering proposal it is NOT enough: the
            # forbidden pair is between the two named marks, so a cheaper
            # crossing edge elsewhere can fuse the classes and leave the two
            # representatives as the only members kept apart.  Hence the
            # warning rather than a quiet row.
            lat, rat = mark_indices(pages, left), mark_indices(pages, right)
            unverified = [
                cid
                for cid, meta in ((left, lmeta), (right, rmeta))
                if not meta.get("audit", {}).get("membership_verified")
            ]
            separations.append(
                {
                    "left_page_id": lmeta["page_ids"][0],
                    "left_mark_index": lat.get(lmeta["page_ids"][0]),
                    "right_page_id": rmeta["page_ids"][0],
                    "right_mark_index": rat.get(rmeta["page_ids"][0]),
                    "left_class_id": left,
                    "right_class_id": right,
                    # What the constraint will actually hold apart on a rebuild.
                    # Not a detail: a reader of this file should not have to
                    # re-derive which of their separations are load-bearing.
                    "pins": "marks" if unverified else "classes",
                    "note": row.get("notes", ""),
                }
            )
            changes.append(
                f"{left} != {right}: separation recorded"
                + (
                    f" (pins the two marks only — {', '.join(unverified)} not membership-verified)"
                    if unverified
                    else ""
                )
            )
        else:
            problems.append(f"{left} / {right}: unrecognised verdict {verdict!r} (expected same|different)")

    # Reconcile `distinct_from` against the merges just applied.  A merge pops
    # the absorbed class, and every separation already recorded against it is
    # left naming a class that no longer exists -- including, on the survivor,
    # a reference to the id it just absorbed, which reads as "distinct from
    # itself".  Measured on v3: one merge left all 23 classes holding a
    # dangling id.  The adjudications are keyed on marks and were never
    # affected; this is the human-readable half of the same fact, and letting
    # the two disagree is how a reader stops trusting either.
    for class_id, meta in classes.items():
        if not meta.get("distinct_from"):
            continue
        resolved: list[str] = []
        for other in meta["distinct_from"]:
            target = resolve(other)
            if target != class_id and target in classes and target not in resolved:
                resolved.append(target)
        if resolved != meta["distinct_from"]:
            meta["distinct_from"] = resolved

    return changes, problems, separations, merges


# --------------------------------------------------------------------------
# The merge slate
# --------------------------------------------------------------------------
#
# `merges.txt` is a partition, not a stream of verdicts, and that is the whole
# reason the slate is usable: a reviewer states the few groups that are one mark
# and says nothing about the thousand pairs that obviously are not.  Everything
# below turns that statement back into the same/different pairs `apply_confusable`
# already knows how to record, so the merge slate adds an input format and not a
# second code path through the ground truth.

REVIEWED_ALL = "REVIEWED-ALL"


def parse_merge_groups(text: str, n_classes: int) -> tuple[list[dict[str, Any]], bool, list[str]]:
    """Parse a filled-in ``merges.txt`` into ``(groups, reviewed_all, problems)``.

    A group is a set of slate indices the reviewer says are one mark.  Groups
    that share an index are **unioned** rather than refused: "3 8" and "8 12" are
    two observations of one equivalence class, and treating that as a
    contradiction would punish a reviewer for writing down the same truth twice.
    Sameness is transitive; the file is allowed to be redundant about it.

    What is refused is anything that cannot be resolved into indices at all --
    an out-of-range number, a one-element group, a token that is not a number --
    because each of those is a typo whose silent interpretation would write a
    wrong permanent merge.
    """
    groups: list[dict[str, Any]] = []
    problems: list[str] = []
    reviewed_all = False

    for lineno, raw in enumerate(text.splitlines(), start=1):
        body, _, note = raw.partition("#")
        body = body.strip()
        if not body:
            continue
        if body.upper() == REVIEWED_ALL:
            reviewed_all = True
            continue

        indices: list[int] = []
        bad = False
        for token in body.replace(",", " ").split():
            try:
                idx = int(token)
            except ValueError:
                problems.append(f"line {lineno}: {token!r} is not a slate index")
                bad = True
                continue
            if not 0 <= idx < n_classes:
                problems.append(f"line {lineno}: index {idx} is outside the slate (0..{n_classes - 1})")
                bad = True
                continue
            indices.append(idx)
        if bad:
            continue
        if len(set(indices)) < 2:
            problems.append(f"line {lineno}: {body!r} names fewer than two distinct classes — a group needs a pair")
            continue
        groups.append({"indices": sorted(set(indices)), "note": note.strip(), "line": lineno})

    return _union_groups(groups), reviewed_all, problems


def _union_groups(groups: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold groups that share an index into one, keeping every note."""
    parent: dict[int, int] = {}

    def find(i: int) -> int:
        parent.setdefault(i, i)
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[min(ra, rb)] = parent[max(ra, rb)] = min(ra, rb)

    for group in groups:
        head = group["indices"][0]
        for idx in group["indices"][1:]:
            union(head, idx)

    merged: dict[int, dict[str, Any]] = {}
    for group in groups:
        root = find(group["indices"][0])
        slot = merged.setdefault(root, {"indices": set(), "notes": [], "lines": []})
        slot["indices"].update(group["indices"])
        slot["lines"].append(group["line"])
        if group["note"]:
            slot["notes"].append(group["note"])

    return [
        {"indices": sorted(v["indices"]), "note": "; ".join(v["notes"]), "lines": sorted(v["lines"])}
        for _root, v in sorted(merged.items())
    ]


def merge_verdicts(
    index: dict[str, Any],
    groups: Sequence[dict[str, Any]],
    reviewed_all: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Compile the partition into ``confusable``-shaped verdict rows.

    Order matters and is not cosmetic: every ``same`` row is emitted before every
    ``different`` row, because :func:`apply_confusable` merges as it goes and
    follows the resulting chain when a later row names a class that has since
    been absorbed.  Emitting a separation first would pin it against a class id
    that is about to stop existing.

    ``reviewed_all`` is what licenses the separations, and it licenses exactly
    the appendix pairs -- the ones that got their own side-by-side sheet.  Pairs
    that appear nowhere but the far end of the distance ranking stay
    unadjudicated, because nobody looked at them; recording them would put a
    decision no human made into the file every future re-cluster is bound by,
    which is precisely the failure the separations exist to prevent.
    """
    by_index = {int(row["index"]): row["class_id"] for row in index.get("classes", [])}
    problems: list[str] = []
    rows: list[dict[str, Any]] = []

    #: Every index mapped to the class it will *be* once the merges are applied
    #: -- its group's lowest member, or itself. Two uses, and the first is a
    #: correctness gate rather than tidiness: a near pair inside a group must
    #: not also be separated, because `save_adjudications` refuses a pair ruled
    #: both ways and would abort the whole apply. The second is deduplication:
    #: once 3 and 8 are one class, the appendix pairs (3, 12) and (8, 12) are
    #: one statement about one pair of classes, and emitting both prints a
    #: contradiction-shaped log for a decision that was made once.
    root: dict[int, int] = {}
    for group in groups:
        for idx in group["indices"]:
            root[idx] = group["indices"][0]

    for group in groups:
        indices = group["indices"]
        missing = [i for i in indices if i not in by_index]
        if missing:
            problems.append(f"group {indices}: index {missing} is not on the slate")
            continue
        # A star from the first member, not the full clique: sameness is
        # transitive and `apply_confusable` merges the classes outright, so n-1
        # rows state the whole group and n(n-1)/2 would restate it.
        head = indices[0]
        for other in indices[1:]:
            rows.append(
                {
                    "left_class_id": by_index[head],
                    "right_class_id": by_index[other],
                    "verdict": "same",
                    "notes": group.get("note", ""),
                }
            )

    if reviewed_all:
        seen: set[frozenset[int]] = set()
        for pair in index.get("near_pairs", []):
            li, ri = int(pair["left_index"]), int(pair["right_index"])
            if li not in by_index or ri not in by_index:
                problems.append(f"near pair [{li}]/[{ri}]: not on the slate")
                continue
            key = frozenset((root.get(li, li), root.get(ri, ri)))
            if len(key) == 1:
                continue  # merged by the reviewer; not a separation
            if key in seen:
                continue  # a nearer appendix pair already separated these two classes
            seen.add(key)
            rows.append(
                {
                    "left_class_id": by_index[li],
                    "right_class_id": by_index[ri],
                    "verdict": "different",
                    "notes": f"slate REVIEWED-ALL; appendix rank {pair.get('rank')} at d={pair.get('distance')}",
                }
            )

    return rows, problems


def load_merge_answer(audit_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read ``index.json`` + ``merges.txt`` and compile them into verdict rows."""
    index_path, answer_path = audit_dir / "index.json", audit_dir / "merges.txt"
    if not index_path.exists():
        raise SystemExit(f"no slate at {index_path} — run make_audit_slate.py --task merge first")
    if not answer_path.exists():
        raise SystemExit(f"no answer file at {answer_path} — the slate should have written a template")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    groups, reviewed_all, problems = parse_merge_groups(
        answer_path.read_text(encoding="utf-8"), len(index.get("classes", []))
    )
    rows, more = merge_verdicts(index, groups, reviewed_all)
    n_same = sum(1 for r in rows if r["verdict"] == "same")
    print(f"  slate: {len(groups)} merge group(s) -> {n_same} same, {len(rows) - n_same} different")
    if not reviewed_all:
        print(f"  slate: no {REVIEWED_ALL} line — recording merges only, no separations")
    return rows, problems + more


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
    corpus: Path,
    min_mark_px: int = cfg.MIN_MARK_PX,
    factor: float = 0.5,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Re-cluster each over-merged class alone, at a tighter threshold.

    Only that class's own instances are touched, so re-splitting one class can
    never disturb another's already-confirmed membership.  The resulting pieces
    come back as fresh candidate classes for the next ``cluster`` sheet.

    Returns ``(notes, merges, separations)``.

    The pieces are **registered in** ``classes``, not merely written onto the
    marks.  ``assign_class_ids`` relabels the manifest, but the slate, the
    roster, the embedder and the report all read ``classes.json``; the only
    other thing that writes it is ``build_corpus.py``, which rebuilds from the
    sources.  Popping the parent without adding its pieces therefore does not
    defer the decision to the next sheet -- it deletes the class from everything
    downstream while leaving its marks pointing at ids nothing knows.

    And registering them is still not enough, which is the half this used to
    miss.  A rebuild re-clusters from the sources and replays
    ``adjudications.json`` -- so a split that lives only in ``classes.json`` is
    discarded by the next build, silently and completely, re-proposing the very
    over-merge a person had just taken apart.  #3343 hit exactly that: stage 3
    rebuilds the corpus to stamp the roster, and the two splits the #3561 audit
    had applied (a five-mark StaVer class, a four-mark SPODS one) would have
    come back fused.  So the partition is recorded the way every other hand
    verdict is: a must-link star inside each piece, one cannot-link between
    each pair of pieces.  The asymmetry is sound rather than lazy -- the stars
    are applied first, so by the time a separation is registered each piece is
    already one group and a single row pins all of it.
    """
    from build_corpus import admit_classes, write_query_crops
    from cluster_marks import assign_class_ids, describe_marks, distance_matrix, single_linkage

    notes: list[str] = []
    merges: list[dict[str, Any]] = []
    separations: list[dict[str, Any]] = []
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

        # `min_instances=1`: the reviewer said this class holds more than one
        # mark, so every piece is a finding.  Sizing them out here would drop
        # the small ones silently; the roster is where a piece too small to
        # search gets excluded, and it can only exclude what it can see.
        inventory = {cid: [(r.page_index, r.mark_index) for r in group] for cid, group in pieces.items()}
        fresh, rejected = admit_classes(pages, inventory, min_instances=1, min_mark_px=min_mark_px, roster=None)
        for cid, fresh_meta in fresh.items():
            fresh_meta["audit"]["notes"] = f"re-clustered out of {class_id} at {tighter:.3f}"
            classes[cid] = fresh_meta
        # The pieces are fresh classes, so each needs its own exemplar -- and a
        # piece that still holds more than one mark says so here.
        _hand, crop_warnings = write_query_crops(
            pages, inventory, fresh, corpus / "queries", backend=backend, threshold=tighter
        )
        # Record the partition so the next rebuild reproduces it.  Keyed on
        # `(page_id, mark_index)` off the refs themselves, which is the one
        # identifier that survives both a re-cluster and a renumbering.
        anchors: list[tuple[str, int]] = []
        for cid, group in sorted(pieces.items()):
            members = sorted((ref.page_id, ref.mark_index) for ref in group)
            if len(members) > 1:
                merges.extend(star(members, f"split of {class_id}: {cid} is one mark"))
            anchors.append(members[0])
        for i, left in enumerate(anchors):
            for right in anchors[i + 1 :]:
                separations.append(
                    {
                        "left_page_id": left[0],
                        "left_mark_index": left[1],
                        "right_page_id": right[0],
                        "right_mark_index": right[1],
                        "pins": "classes",
                        "note": f"split of {class_id} at {tighter:.3f}",
                    }
                )

        note = f"{class_id}: re-clustered at {tighter:.3f} into {len(pieces)} piece(s)"
        if rejected:
            note += f", {len(rejected)} not admitted ({'; '.join(sorted(rejected.values()))})"
        notes.append(note)
        notes.extend(crop_warnings)
    return notes, merges, separations


def _refs_for_class(pages: list[Page], class_id: str) -> list[Any]:
    from cluster_marks import MarkRef

    return [
        MarkRef(pi, mi, page.page_id, mark.kind, mark.box)
        for pi, page in enumerate(pages)
        for mi, mark in enumerate(page.marks)
        if mark.class_id == class_id
    ]


def dedupe_manifest(path: Path, *, apply: bool = False) -> int:
    """Repair a corpus.jsonl that already holds duplicate page records (#4054).

    ``write_manifest`` now dedupes on the way out, so this is only for a file
    written before that.  Names every id it would collapse, and every id whose
    copies disagreed about the marks, before writing anything.
    """
    from sources._common import dedupe_pages, read_manifest, write_manifest  # noqa: PLC0415

    pages = list(read_manifest(path))
    seen: dict[str, int] = {}
    marks: dict[str, set[int]] = {}
    for page in pages:
        seen[page.page_id] = seen.get(page.page_id, 0) + 1
        marks.setdefault(page.page_id, set()).add(len(page.marks))
    dups = {k: v for k, v in seen.items() if v > 1}
    if not dups:
        print(f"no duplicate page_id in {path} ({len(pages)} record(s))")
        return 0
    divergent = sorted(k for k in dups if len(marks[k]) > 1)
    print(f"{len(dups)} duplicated page_id(s) over {sum(v - 1 for v in dups.values())} extra record(s)")
    print(f"  {len(dups) - len(divergent)} identical, {len(divergent)} disagreeing about the marks:")
    for page_id in divergent:
        print(f"    {page_id}: mark counts {sorted(marks[page_id])}")
    kept, counts = dedupe_pages(pages)
    if not apply:
        print(f"would write {len(kept)} record(s) — dry run, pass --apply to write")
        return 0
    n = write_manifest(kept, path)
    print(f"wrote {n} record(s) to {path} (was {counts['records']})")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--task",
        choices=(
            "merge",
            "membership",
            "cluster",
            "confusable",
            "distinctive",
            "letterhead",
            "completeness",
            "ucsf_classes",
            "query_crops",
            "box_tighten",
            "surprise",
        ),
    )
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument(
        "--audit-dir",
        default=None,
        help="read verdicts from <corpus>/audit/<this> instead of audit/<task>, e.g. completeness2 for the "
        "multi-method completeness slate",
    )
    ap.add_argument("--apply", action="store_true", help="write the changes (default is a dry run)")
    ap.add_argument(
        "--supersede",
        action="store_true",
        help="let these verdicts replace stored rulings on the same pairs (a reviewer changing "
        "their mind); without it an overlap is reported and nothing is written",
    )
    ap.add_argument(
        "--reviewer",
        default=None,
        help="who worked these sheets; stamped onto every class the membership pass verifies, "
        "because 'a human checked it' is a claim about a person",
    )
    ap.add_argument(
        "--migrate-adjudications",
        action="store_true",
        help="stamp the mark index onto legacy page-id-only adjudications, then exit",
    )
    ap.add_argument(
        "--tidy-added-marks",
        action="store_true",
        help="rewrite added_marks.json without rows repeating a (page_id, box, class_id) already in it, then exit",
    )
    ap.add_argument(
        "--dedupe-manifest",
        action="store_true",
        help="rewrite corpus.jsonl with one record per page_id, keeping the longest mark list, then exit",
    )
    ap.add_argument("--cluster-backend", default=cfg.CLUSTER_BACKEND, choices=("phash", "siglip"))
    ap.add_argument("--cluster-threshold", type=float, default=cfg.CLUSTER_THRESHOLD)
    ap.add_argument("--min-mark-px", type=int, default=cfg.MIN_MARK_PX)
    args = ap.parse_args(argv)

    classes_path = args.corpus / "classes.json"
    manifest_path = args.corpus / "corpus.jsonl"
    adjudications_path = args.corpus / "adjudications.json"
    classes = json.loads(classes_path.read_text(encoding="utf-8"))

    if args.dedupe_manifest:
        return dedupe_manifest(args.corpus / "corpus.jsonl", apply=args.apply)

    if args.tidy_added_marks:
        # The repair for a store written before saving deduped (#4042): a
        # re-applied class stored every box it had already contributed again.
        from completeness import ADDED_MARKS, added_mark_key, load_added_marks, save_added_marks  # noqa: PLC0415

        store = args.corpus / ADDED_MARKS
        rows = load_added_marks(store)
        seen: set[Any] = set()
        kept, duplicates = [], []
        for row in rows:
            key = added_mark_key(row)
            (duplicates if key in seen else kept).append(row)
            seen.add(key)
        for row in duplicates:
            print(f"  duplicate: {row['page_id']} {row['box']} {row.get('class_id')}")
        print(f"{len(duplicates)} duplicate row(s) of {len(rows)} in {store}")
        if not duplicates:
            return 0
        if args.apply:
            save_added_marks(kept, store)
            print(f"wrote {store} with {len(kept)} row(s)")
        else:
            print("dry run — pass --apply to write")
        return 0

    if args.migrate_adjudications:
        from cluster_marks import load_adjudication_rows, save_adjudications

        pages = list(read_manifest(manifest_path))
        old_same, old_diff = load_adjudication_rows(adjudications_path)
        same, diff, problems = migrate_adjudications(pages, classes, old_same, old_diff)
        stamped = sum(
            1
            for before, after in zip(old_same + old_diff, same + diff)
            for side in ("left", "right")
            if before.get(f"{side}_mark_index") is None and after.get(f"{side}_mark_index") is not None
        )
        split_merges, split_separations = split_rows_from_notes(pages, classes)
        same, diff = same + split_merges, diff + split_separations
        for problem in problems:
            print(f"  PROBLEM: {problem}")
        print(f"{stamped} endpoint(s) resolved to a mark, {len(problems)} left ambiguous")
        print(
            f"{len(split_merges)} merge(s) and {len(split_separations)} separation(s) recovered "
            "from splits that were applied before they were recorded"
        )
        if problems:
            return 1
        if args.apply:
            save_adjudications(same, diff, adjudications_path)
            print(f"wrote {adjudications_path}")
        else:
            print("dry run — pass --apply to write")
        return 0

    if not args.task:
        ap.error("--task is required unless --migrate-adjudications is given")
    audit_dir = args.corpus / "audit" / (args.audit_dir or args.task)
    slate_problems: list[str] = []
    if args.task == "merge":
        # The slate is an input *format*, not a second way of recording ground
        # truth: it compiles to the same same/different rows the pairwise pass
        # produces and goes through the same applier.
        verdicts, slate_problems = load_merge_answer(audit_dir)
    elif args.task in ("ucsf_classes", "surprise"):
        # Every row, answered or not: a relation row carries no `verdict`, and an
        # unanswered sheet is still counted as unreviewed.
        path = audit_dir / "verdicts.jsonl"
        if not path.exists():
            raise SystemExit(f"no verdict file at {path} — run ucsf_classes.py slates first")
        verdicts = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        verdicts = load_verdicts(audit_dir / "verdicts.jsonl")

    mutates_pages = args.task in (
        "cluster",
        "membership",
        "confusable",
        "merge",
        "completeness",
        "ucsf_classes",
        "box_tighten",
    )
    pages = list(read_manifest(manifest_path)) if mutates_pages else []
    new_separations: list[dict[str, Any]] = []
    new_merges: list[dict[str, Any]] = []
    new_added_marks: list[dict[str, Any]] = []
    new_reviewed_negatives: dict[str, list[str]] = {}
    new_exclusions: dict[str, list[str]] = {}
    new_class_ids: list[str] = []
    resplit: list[str] = []

    if args.task == "membership":
        changes, problems, new_merges, new_separations = apply_membership(
            pages, classes, verdicts, reviewer=args.reviewer
        )
    elif args.task == "cluster":
        changes, problems, resplit = apply_cluster(pages, classes, verdicts)
    elif args.task in ("confusable", "merge"):
        changes, problems, new_separations, new_merges = apply_confusable(pages, classes, verdicts)
        problems += slate_problems
        if args.task == "merge":
            reviewed = any(str(r.get("notes", "")).startswith("slate REVIEWED-ALL") for r in verdicts)
            if reviewed and not problems:
                for meta in classes.values():
                    meta.setdefault("audit", {})["partition_reviewed"] = True
                changes.append(f"{len(classes)} class(es) marked partition_reviewed")
    elif args.task == "completeness":
        from completeness import apply_completeness  # noqa: PLC0415

        changes, problems, new_merges, new_separations, new_added_marks = apply_completeness(
            pages, classes, verdicts, reviewer=args.reviewer
        )
    elif args.task == "ucsf_classes":
        from ucsf_classes import apply_ucsf_classes  # noqa: PLC0415

        before = set(classes)
        changes, problems, new_added_marks, new_reviewed_negatives, new_exclusions = apply_ucsf_classes(
            pages, classes, verdicts, reviewer=args.reviewer
        )
        new_class_ids = sorted(set(classes) - before)
    elif args.task == "surprise":
        from surprise_review import apply_surprise  # noqa: PLC0415

        changes, problems, new_reviewed_negatives, new_exclusions = apply_surprise(
            classes, verdicts, reviewer=args.reviewer
        )
    elif args.task == "query_crops":
        from query_crops import STORE, apply_query_crops, load_store  # noqa: PLC0415

        crop_store = load_store(args.corpus / STORE)
        changes, problems = apply_query_crops(classes, verdicts, crop_store, reviewer=args.reviewer)
    elif args.task == "box_tighten":
        from box_tighten import STORE as BOX_STORE, apply_box_tighten, load_store as load_box_store  # noqa: PLC0415

        box_store = load_box_store(args.corpus / BOX_STORE)
        changes, problems, stale_queries = apply_box_tighten(
            pages, classes, verdicts, box_store, reviewer=args.reviewer
        )
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
        notes, split_merges, split_separations = resplit_classes(
            pages,
            classes,
            resplit,
            backend=args.cluster_backend,
            threshold=args.cluster_threshold,
            corpus=args.corpus,
            min_mark_px=args.min_mark_px,
        )
        for note in notes:
            print(f"  {note}")
        new_merges += split_merges
        new_separations += split_separations
        print("  re-run make_audit_slate.py --task cluster to review the new pieces")

    if new_separations or new_merges:
        from cluster_marks import load_adjudication_rows, save_adjudications

        # Verbatim rows, not the narrowed pairs: re-saving what the clusterer
        # reads would drop every note and class id already on file.
        old_same, old_diff = load_adjudication_rows(adjudications_path)
        old_same, old_diff, overturned = supersede(old_same, old_diff, new_merges, new_separations)
        if overturned and not args.supersede:
            for row in overturned:
                print(f"  PROBLEM: {describe_pair(row)} is already on file ruled the other way")
            print(
                "\nA reviewer has changed their mind, which is allowed and is not the same as a "
                "contradiction: re-run with --supersede to replace the stored ruling(s) above. "
                "Without it nothing is written, because silently keeping whichever was applied "
                "last is how a decision nobody made ends up in the ground truth."
            )
            return 1
        for row in overturned:
            print(f"  SUPERSEDED: {describe_pair(row)} — the stored ruling is replaced")
        save_adjudications(
            old_same + new_merges,
            old_diff + new_separations,
            adjudications_path,
        )
        print(f"  wrote {len(new_merges)} merge(s) and {len(new_separations)} separation(s) to {adjudications_path}")

    if new_added_marks:
        from completeness import (  # noqa: PLC0415
            ADDED_MARKS,
            dedupe_added_marks,
            load_added_marks,
            save_added_marks,
        )

        # Before classes.json and the manifest: a new box that is on a page but
        # not in the store would vanish at the next rebuild while its must-link
        # still names it, which is the drift this store exists to prevent.
        store = args.corpus / ADDED_MARKS
        # Re-applying a class proposes every box it already contributed again
        # (#4042), so count what the store actually gained rather than what the
        # pass offered.
        before = dedupe_added_marks(load_added_marks(store))
        written = save_added_marks(before + new_added_marks, store)
        already = len(new_added_marks) - (len(written) - len(before))
        print(
            f"  appended {len(written) - len(before)} hand-added mark(s) to {store}"
            + (f" ({already} already on file)" if already else "")
        )

    if new_reviewed_negatives or new_exclusions:
        import roster  # noqa: PLC0415

        # The only UCSF negatives a class gets (#3921); stored apart from
        # classes.json so a rebuild's regenerated metadata gets them back.
        store_path = args.corpus / roster.REVIEWED_NEGATIVES
        store = roster.load_reviewed_negatives(store_path)
        excluded = roster.load_reviewed_negatives(store_path, "excluded")
        for cid in set(new_reviewed_negatives) | set(new_exclusions):
            positives = set(classes.get(cid, {}).get("page_ids", []))
            excluded[cid] = sorted((set(excluded.get(cid, [])) | set(new_exclusions.get(cid, []))) - positives)
            store[cid] = sorted(
                (set(store.get(cid, [])) | set(new_reviewed_negatives.get(cid, []))) - positives - set(excluded[cid])
            )
        roster.save_reviewed_negatives(store, store_path, excluded)
        print(f"  wrote reviewed negatives for {len(store)} class(es) to {store_path}")

    if new_class_ids:
        import roster  # noqa: PLC0415

        # A new class is rebuilt only if the roster names it.
        roster_path = args.corpus / "roster.json"
        current = roster.load(roster_path)
        current.classes = list(dict.fromkeys(current.classes + new_class_ids))
        roster.save(current, roster_path)
        print(f"  added {len(new_class_ids)} class(es) to {roster_path}: {new_class_ids}")
    if args.task == "box_tighten":
        from box_tighten import STORE as BOX_STORE, recut_query_crop, save_store as save_box_store  # noqa: PLC0415

        # The store before the manifest, for the same reason as added_marks: a
        # box on the page but not in the store would revert at the next rebuild.
        save_box_store(box_store, args.corpus / BOX_STORE)
        print(f"  wrote {len(box_store)} box override(s) to {args.corpus / BOX_STORE}")
        by_page = {p.page_id: p for p in pages}
        for class_id in sorted(stale_queries):
            # The primary crop was cut from the box just replaced.
            path = recut_query_crop(classes[class_id], by_page[classes[class_id]["query_page_id"]], class_id)
            print(f"  re-cut {class_id} query crop -> {path}")

    if args.task == "query_crops":
        from query_crops import STORE, materialise, save_store  # noqa: PLC0415

        # The store first: it is what a rebuild replays, so a crop that exists
        # on disk but not in the store would silently vanish at the next build.
        save_store(crop_store, args.corpus / STORE)
        crop_pages = {p.page_id: p for p in read_manifest(manifest_path)}
        crop_warnings: list[str] = []
        n = materialise(classes, crop_store, crop_pages, args.corpus / "queries", crop_warnings)
        for w in crop_warnings:
            print(f"  WARNING: {w}")
        print(f"  wrote {n} extra query crop(s) and {args.corpus / STORE}")

    classes_path.write_text(json.dumps(classes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if mutates_pages:
        write_manifest(pages, manifest_path)
    print(f"wrote {classes_path}" + (f" and {manifest_path}" if mutates_pages else ""))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
