"""UCSF letterhead-band classes as roster-class proposals, with a member search (#3921, #3922).

#3902 found 11 SigLIP band classes that are each one printed mark.  A band class
is not a roster class: it has no query crop, and it holds only the bands SigLIP
happened to cluster -- the RJR block logo is on ~550 bands and 20 of them
clustered -- so every page it misses would be scored as a negative for its own
mark (#3922).  This turns each candidate into something an owner can rule on,
and writes nothing to the corpus but a slate under ``audit/ucsf_classes/``:

* ``ucsf_proposals.json`` (beside this script) names each proposal's band
  components and whether it **extends** a Tobacco800 roster class -- the same
  mark, so the class spans two sources instead of the mark becoming a second
  class -- or is **new**.
* ``crops`` cuts a query crop from each proposal's medoid band: SIFT-match
  member bands against it and keep the region their inliers pile onto
  (:func:`consensus_box`).  A crop that is weakly supported or wider than a mark
  is flagged ``needs_hand_crop``; a ``crop`` in the proposal overrides it.
* ``search`` runs SIFT over the top band of every UCSF tier ``s``/``m`` page,
  every Tobacco800 page and every component member, against each proposal's
  crop and every Tobacco800 roster crop, and checks each crop against the roster
  crops directly.
* ``slates`` renders one-screen sheets with ``completeness.render``: (a) is it
  one mark, from a sample of the band class, then every other band-class member
  in tiers ``s``/``m``; (b) every search hit outside the band class, by inlier
  bin; (c) every band-class member in ``s``/``m`` the search did not find.
  **A verdict covers exactly the cells its sheet shows** -- ``all`` on a sheet
  that is one of nine for a bin accepts those 18 pages and nothing else -- so a
  bin is reviewed only when every one of its sheets is answered.
* :func:`apply_ucsf_classes` (``audit_to_corrections.py --task ucsf_classes``)
  folds the owner's answers back: accepted cells become marks of the extended
  or new class, stored in ``added_marks.json`` with their ``class_id`` (UCSF is
  never clustered, so nothing else would restore it), and rejected cells become
  the class's ``reviewed_negative_page_ids``, stored in
  ``reviewed_negatives.json``.  Those are the only UCSF negatives a class gets:
  ``roster.eligible_pages`` keeps every unreviewed UCSF page out of the pool.
  Suggestions (``suggestion``, ``suggested_relation``) are never applied; only
  ``verdict`` and ``relation`` are.

Bands are resampled so every page is ``TARGET_WIDTH`` pixels wide before
detection: UCSF pages are 150 dpi and a letterhead mark on one is often under
100 px, where SIFT has little to hold; a crop is resampled by the factor of the
page it was cut from, so a crop and its source band stay at the same scale.

    python ucsf_classes.py crops  --out <dir>
    python ucsf_classes.py search --out <dir>      # CPU, many cores
    python ucsf_classes.py slates --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from datetime import date
from typing import Any, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import docmarks_config as cfg  # noqa: E402

PROPOSALS = Path(__file__).resolve().parent / "ucsf_proposals.json"
TASK = "ucsf_classes"
#: Top of the page searched.  #3902 clustered the top 22%; a little more keeps a
#: mark that sits just under that line.
SEARCH_BAND_FRAC = 0.25
#: Every page is resampled to this width before detection (a UCSF page is 1,240 px).
TARGET_WIDTH = 2480
BUDGET = 8192
#: Member bands matched against the medoid to find the mark on it.
CONSENSUS_SAMPLE = 32
#: A consensus cell must hold inliers from this share of the verified matches.
CONSENSUS_SHARE = 0.5
CONSENSUS_GRID = (64, 16)  # cells across, cells down the band
#: Fewer verified member matches than this, or a crop wider than this share of
#: the band, and the auto crop is not trusted.
MIN_SUPPORT = 8
MAX_CROP_WIDTH = 0.5
#: Hits kept in ``hits.json``; slates show hits from MIN_INLIERS (the completeness gate).
KEEP_INLIERS = 6
MIN_INLIERS = 8
#: A fit whose inlier box is under this share of the query's size on the page is
#: a collapsed RANSAC fit -- many template points mapped onto one spot, which
#: carries 80+ inliers on unrelated bands (#3912) -- not a mark.
MIN_BOX_SHARE = 0.25
#: Inlier bins for sheet (b), strongest first: [80, inf), [40, 80), ...
BIN_EDGES = (80, 40, 20, 12, 8)
PER_SHEET = 18
#: Page pixels shown around a hit at least, so a small mark is seen in its letterhead.
CONTEXT_PX = 120
SEED = 3921
#: The matcher's RANSAC tolerance and scale floor (``vtscore.media.structural``),
#: repeated because :func:`inlier_points` needs the inlier set ``verify`` discards.
RANSAC_THRESHOLD = 0.02
MIN_SCALE = 0.03


# ---------------------------------------------------------------------------
# Proposals and band components
# ---------------------------------------------------------------------------


@dataclass
class Proposal:
    name: str
    components: list[tuple[str, int]]
    relation: str  # "extends" or "new"
    roster_class: Optional[str] = None
    note: str = ""
    crop: Optional[dict[str, Any]] = None  # hand crop: {"page_id", "box": [x, y, w, h], "why"}


def load_proposals(path: Path = PROPOSALS, classes: Optional[dict[str, Any]] = None) -> list[Proposal]:
    """Read and check the proposals: an extension must name an on-roster class, a new one none.

    A proposal the owner dropped moves to ``dropped`` with its reason and is not
    returned: it gets no slate and cannot be applied.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    dropped = {row["name"] for row in raw.get("dropped", [])}
    out, errors, seen = [], [], set()
    for row in raw["proposals"]:
        p = Proposal(
            name=row["name"],
            components=[(str(a), int(r)) for a, r in row["components"]],
            relation=row["relation"],
            roster_class=row.get("roster_class"),
            note=row.get("note", ""),
            crop=row.get("crop"),
        )
        if p.name in seen:
            errors.append(f"{p.name}: duplicate proposal name")
        seen.add(p.name)
        if p.relation not in ("extends", "new"):
            errors.append(f"{p.name}: relation must be 'extends' or 'new', got {p.relation!r}")
        elif p.relation == "extends" and not p.roster_class:
            errors.append(f"{p.name}: an extension must name its roster_class")
        elif p.relation == "new" and p.roster_class:
            errors.append(f"{p.name}: a new class cannot name a roster_class")
        if p.roster_class and classes is not None and not classes.get(p.roster_class, {}).get("on_roster"):
            errors.append(f"{p.name}: {p.roster_class} is not an on-roster class")
        out.append(p)
    errors += [f"{name}: both proposed and dropped" for name in sorted(seen & dropped)]
    claimed = Counter(c for p in out for c in p.components)
    errors += [f"component {c} is claimed by {n} proposals" for c, n in claimed.items() if n > 1]
    if errors:
        raise ValueError("; ".join(errors))
    return out


def component_members(authors: np.ndarray, vecs: np.ndarray, author: str, rank: int, threshold: float) -> np.ndarray:
    """Row indices of *author*'s *rank*-th largest band class (1-based), as ``band_siglip.purity`` names them."""
    from band_siglip import components  # noqa: PLC0415

    idx = np.flatnonzero(authors == author)
    labels = components(vecs[idx], threshold)
    ranked = Counter(labels.tolist()).most_common()
    if not 1 <= rank <= len(ranked):
        raise ValueError(f"{author}: no component of rank {rank} ({len(ranked)} at {threshold})")
    return idx[labels == ranked[rank - 1][0]]


def medoid(vecs: np.ndarray) -> int:
    """Index (into *vecs*) of the row with the highest mean cosine similarity to the rest."""
    return int(np.argmax((vecs @ vecs.T).mean(axis=1)))


# ---------------------------------------------------------------------------
# Where a mark sits on a band
# ---------------------------------------------------------------------------


def consensus_box(
    point_sets: Sequence[np.ndarray],
    *,
    grid: tuple[int, int] = CONSENSUS_GRID,
    share: float = CONSENSUS_SHARE,
) -> Optional[tuple[tuple[float, float, float, float], int]]:
    """The region most matches' inliers land on, as ``((x0, y0, x1, y1), support)``.

    Each set is one verified match's inlier locations on the same band, in
    normalised coordinates.  A grid cell scores the number of *matches* with an
    inlier in it -- not the number of inliers, so one dense text match cannot
    outvote the rest.  Cells held by at least *share* of the matches form the
    mask; the answer is the bounding box of the mask's connected blob (with a
    one-cell bridge) around the best cell, padded by a cell.  ``support`` is the
    best cell's score.  ``None`` when there is nothing to vote on.
    """
    from scipy import ndimage  # noqa: PLC0415

    gx, gy = grid
    sets = [np.asarray(s, dtype=np.float64).reshape(-1, 2) for s in point_sets]
    sets = [s for s in sets if len(s)]
    if not sets:
        return None
    occ = np.zeros((gy, gx), dtype=np.int64)
    for pts in sets:
        cx = np.clip((pts[:, 0] * gx).astype(int), 0, gx - 1)
        cy = np.clip((pts[:, 1] * gy).astype(int), 0, gy - 1)
        cells = np.zeros_like(occ, dtype=bool)
        cells[cy, cx] = True
        occ += cells
    mask = occ >= max(1, math.ceil(share * len(sets)))
    if not mask.any():
        return None
    labels, _ = ndimage.label(ndimage.binary_dilation(mask, structure=np.ones((3, 3))), structure=np.ones((3, 3)))
    best = np.unravel_index(int(np.argmax(np.where(mask, occ, -1))), occ.shape)
    ys, xs = np.nonzero(mask & (labels == labels[best]))
    x0, x1 = max(0, xs.min() - 1), min(gx, xs.max() + 2)
    y0, y1 = max(0, ys.min() - 1), min(gy, ys.max() + 2)
    return (float(x0 / gx), float(y0 / gy), float(x1 / gx), float(y1 / gy)), int(occ[best])


def band_px(width: int, height: int, frac: float = SEARCH_BAND_FRAC) -> tuple[int, int]:
    """``(w, h)`` of a page's search band in page pixels."""
    return width, max(1, int(height * frac))


def norm_to_page_box(norm: Sequence[float], width: int, height: int, frac: float = SEARCH_BAND_FRAC) -> list[int]:
    """A band-normalised ``(x0, y0, x1, y1)`` as a page-pixel ``[x, y, w, h]``."""
    bw, bh = band_px(width, height, frac)
    x0, x1 = sorted((float(norm[0]), float(norm[2])))
    y0, y1 = sorted((float(norm[1]), float(norm[3])))
    x, y = int(round(x0 * bw)), int(round(y0 * bh))
    return [x, y, max(1, int(round(x1 * bw)) - x), max(1, int(round(y1 * bh)) - y)]


def crop_flags(norm: Sequence[float], support: int) -> list[str]:
    flags = []
    if support < MIN_SUPPORT:
        flags.append(f"support {support} < {MIN_SUPPORT}")
    if norm[2] - norm[0] > MAX_CROP_WIDTH:
        flags.append(f"width {norm[2] - norm[0]:.2f} of the band > {MAX_CROP_WIDTH}")
    return flags


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


def parse_verdict(raw: str, n: int) -> tuple[Optional[list[int]], Optional[str]]:
    """The shown cells that carry the proposal's mark.

    ``"all"``, ``"none"``, ``"0,3,7"``, or ``"all but 3,7"``.  Blank is not
    reviewed (``None``, no error).
    """
    text = raw.strip().lower()
    if not text:
        return None, None
    if text == "all":
        return list(range(n)), None
    if text == "none":
        return [], None
    negate = text.startswith("all but")
    body = text[len("all but") :] if negate else text
    try:
        idx = sorted({int(tok) for tok in body.replace(" ", "").split(",") if tok})
    except ValueError:
        return None, f"verdict must be 'all', 'none', 'all but N,..' or cell numbers, got {raw!r}"
    bad = [i for i in idx if not 0 <= i < n]
    if bad:
        return None, f"cell number(s) {bad} outside 0..{n - 1}"
    if negate and not idx:
        return None, f"'all but' needs cell numbers, got {raw!r}"
    return ([i for i in range(n) if i not in idx] if negate else idx), None


def bin_label(lo: int, hi: Optional[int]) -> str:
    return f"{lo}+" if hi is None else f"{lo}-{hi - 1}"


def bins(scored: Sequence[tuple[str, int]], edges: Sequence[int] = BIN_EDGES) -> list[tuple[int, Optional[int], list]]:
    """Group ``(page_id, inliers)`` by descending inlier *edges*; below the last edge is dropped."""
    out: list[tuple[int, Optional[int], list]] = []
    hi: Optional[int] = None
    for lo in edges:
        members = sorted(((p, n) for p, n in scored if n >= lo and (hi is None or n < hi)), key=lambda t: (-t[1], t[0]))
        out.append((lo, hi, members))
        hi = lo
    return out


def sample_sheet(items: Sequence[Any], k: int, rng: random.Random, key=None) -> list[Any]:
    """All of *items* if they fit a sheet, else a random *k*, back in *key* order."""
    if len(items) <= k:
        return list(items)
    picked = rng.sample(list(items), k)
    return sorted(picked, key=key) if key else picked


def spread_sample(groups: Sequence[Sequence[Any]], k: int, rng: random.Random) -> list[Any]:
    """*k* items drawn across *groups* as evenly as their sizes allow (a merged class shows both halves)."""
    pools = [list(g) for g in groups]
    for pool in pools:
        rng.shuffle(pool)
    out: list[Any] = []
    while len(out) < k and any(pools):
        for pool in pools:
            if pool and len(out) < k:
                out.append(pool.pop())
    return out


ANSWER_FIELDS = ("verdict", "suggestion", "uncertain", "note", "suggested_by", "relation")


def carry_answers(old: Sequence[dict[str, Any]], new: Sequence[dict[str, Any]]) -> int:
    """Copy answers from a previous ``verdicts.jsonl`` onto re-rendered rows that show exactly the same pages.

    A sheet whose cells changed (a new crop, a moved bin edge) starts blank:
    its old answer was about other pages.  Returns the number of rows carried.
    """

    def key(row: dict[str, Any]) -> tuple:
        if row.get("task") == TASK:
            return (row["sheet"], tuple(c["page_id"] for c in row["cells"]))
        return (row.get("task"), row.get("proposal"))

    by_key = {key(r): r for r in old}
    carried = 0
    for row in new:
        prev = by_key.get(key(row))
        if prev is None:
            continue
        for field in ANSWER_FIELDS:
            if field in prev and prev[field] not in ("", None) and (field in row or field == "suggested_by"):
                row[field] = prev[field]
        carried += 1
    return carried


def chunks(items: Sequence[Any], k: int = PER_SHEET) -> list[list[Any]]:
    return [list(items[i : i + k]) for i in range(0, len(items), k)]


def sheet_order(items: Sequence[Any], k: int, rng: random.Random, key=None) -> list[list[Any]]:
    """Every item on sheets of *k*: a random *k* first (the sheet a clean bin is judged by), the rest in *key* order.

    The first sheet is the same draw ``sample_sheet`` makes, so a sheet already
    reviewed keeps its cells when the rest of its bin is added behind it.
    """
    first = sample_sheet(items, k, rng, key=key)
    taken = {id(x) for x in first}
    rest = [x for x in items if id(x) not in taken]
    return [first] + chunks(sorted(rest, key=key) if key else rest, k) if items else []


def tally(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Per proposal, what the answered sheets say: ``(counts, problems)``.

    Counts only pages a verdict names: ``accepted`` and ``rejected`` are shown
    cells, and ``unreviewed`` is every cell on an unanswered sheet.  There is no
    extrapolation from a sheet to the bin it came from -- a page nobody saw is
    unreviewed, whatever its neighbours were.
    """
    out: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    for row in rows:
        if row.get("task") != TASK:
            continue
        entry = out.setdefault(
            row["proposal"], {"sheets": 0, "answered": 0, "accepted": 0, "rejected": 0, "unreviewed": 0}
        )
        n = len(row["cells"])
        accepted, error = parse_verdict(str(row.get("verdict", "")), n)
        entry["sheets"] += 1
        if error:
            problems.append(f"{row['sheet']}: {error}")
            continue
        if accepted is None:
            entry["unreviewed"] += n
            continue
        entry["answered"] += 1
        entry["accepted"] += len(accepted)
        entry["rejected"] += n - len(accepted)
    return out, problems


# ---------------------------------------------------------------------------
# Applying the owner's answers
# ---------------------------------------------------------------------------

PROVENANCE = "ucsf_classes"
#: A cell the search did not locate shows the page's whole band; accepting it
#: says the mark is on the page, not where, so its mark is tagged apart.
PROVENANCE_BAND = "ucsf_classes_band"
ALL_SOURCES = ("spods", "staver", "tobacco800", "ucsf", "synth")


def new_class_id(name: str) -> str:
    return f"ucsf/logo_{name}"


def _new_class_meta(class_id: str, relation_row: dict[str, Any], reviewer: Optional[str]) -> dict[str, Any]:
    """Metadata for a new UCSF class, in the shape ``build_corpus.admit_classes`` writes."""
    return {
        "class_id": class_id,
        "source": "ucsf",
        "kind": "logo",
        "n_instances": 0,
        "median_mark_px": None,
        "located_by": "box",
        "provenance": [PROVENANCE],
        "page_ids": [],
        "eligible_distractor_sources": sorted(s for s in ALL_SOURCES if cfg.eligible_distractor("ucsf", s)),
        "distinct_from": [],
        "on_roster": True,
        "caveats": [],
        "query_crop": relation_row.get("query_crop"),
        "query_page_id": relation_row.get("query_page_id"),
        "audit": {
            "distinctive": None,
            "cluster_ok": None,
            "membership_verified": False,
            "rejected_page_ids": [],
            "reviewed_by": reviewer,
            "reviewed_on": None,
            "notes": f"proposed by #3921 ({relation_row['proposal']})",
        },
    }


def _target(relation: str, name: str, classes: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """``(class_id, problem)`` for a relation answer; ``(None, None)`` for reject or blank."""
    text = relation.strip()
    if not text or text.lower() == "reject":
        return None, None
    if text.lower() == "new":
        return new_class_id(name), None
    if text.lower().startswith("extends "):
        cid = text.split(None, 1)[1].strip()
        if not classes.get(cid, {}).get("on_roster"):
            return None, f"{name}: {cid} is not an on-roster class"
        return cid, None
    return None, f"{name}: relation must be 'extends <class_id>', 'new' or 'reject', got {relation!r}"


def apply_ucsf_classes(
    pages: list[Any],
    classes: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    *,
    reviewer: Optional[str] = None,
) -> tuple[list[str], list[str], list[dict[str, Any]], dict[str, list[str]], dict[str, list[str]]]:
    """Fold answered sheets into classes: ``(changes, problems, added_marks, reviewed_negatives, excluded)``.

    Per proposal whose ``relation`` is answered: every accepted cell becomes a
    mark of the target class on its page (a new box, provenance ``ucsf_classes``
    or ``ucsf_classes_band`` when the search never located it) and the page a
    positive; every *shown* rejected cell becomes a reviewed negative of that
    class, unless the same page is accepted elsewhere.  Nothing else changes:
    an unanswered sheet leaves its pages unreviewed, which keeps them out of
    the class's pool.  A sheet answered while its proposal's relation is blank
    is a problem, not a silent no-op.

    An accepted cell on a clustered source (a Tobacco800 hit for a new UCSF
    class) cannot become a mark here -- a mark there needs a must-link to a
    class instance -- so its page is *excluded* from the class's pools rather
    than left to score as a negative carrying the mark, and reported for the
    completeness pass.
    """
    from sources._common import Mark  # noqa: PLC0415

    by_id = {p.page_id: p for p in pages}
    changes: list[str] = []
    problems: list[str] = []
    added_rows: list[dict[str, Any]] = []
    negatives: dict[str, list[str]] = {}
    exclusions: dict[str, list[str]] = {}
    relations = {r["proposal"]: r for r in rows if r.get("task") == f"{TASK}_relation"}
    sheets: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if r.get("task") == TASK:
            sheets.setdefault(r["proposal"], []).append(r)

    for name in sorted(set(relations) | set(sheets)):
        rel = relations.get(name, {"proposal": name})
        cid, problem = _target(str(rel.get("relation", "")), name, classes)
        answered = [r for r in sheets.get(name, []) if str(r.get("verdict", "")).strip()]
        if problem:
            problems.append(problem)
            continue
        if cid is None:
            if answered and not str(rel.get("relation", "")).strip():
                problems.append(f"{name}: {len(answered)} sheet(s) answered but the relation is blank")
            elif str(rel.get("relation", "")).strip():
                changes.append(f"{name}: rejected, nothing applied")
            continue
        if cid not in classes:
            classes[cid] = _new_class_meta(cid, rel, reviewer)
        meta = classes[cid]
        kind = meta.get("kind", "logo")
        accepted_pages: set[str] = set()
        rejected_pages: set[str] = set()
        bad = False
        for row in answered:
            cells = row["cells"]
            accepted, error = parse_verdict(str(row["verdict"]), len(cells))
            if error or accepted is None:
                problems.append(f"{row['sheet']}: {error}")
                bad = True
                continue
            for i, cell in enumerate(cells):
                (accepted_pages if i in accepted else rejected_pages).add(cell["page_id"])
        if bad:
            continue
        added = 0
        refused: set[str] = set()
        carried: set[str] = set()
        for row in answered:
            accepted, _ = parse_verdict(str(row["verdict"]), len(row["cells"]))
            for i in accepted or []:
                cell = row["cells"][i]
                page = by_id.get(cell["page_id"])
                if page is None:
                    problems.append(f"{row['sheet']}: cell {i} page {cell['page_id']} is not in the manifest")
                    refused.add(cell["page_id"])
                    continue
                if page.source in cfg.CLUSTERED_SOURCES:
                    carried.add(page.page_id)
                    continue
                if any(m.class_id == cid for m in page.marks):
                    continue  # accepted twice (e.g. on an (a) and a (b) sheet)
                box = tuple(int(v) for v in cell["box"])
                prov = PROVENANCE if cell.get("located", cell.get("inliers", 0) >= MIN_INLIERS) else PROVENANCE_BAND
                page.marks.append(Mark(kind, box, cid, prov))
                added_rows.append(
                    {
                        "page_id": page.page_id,
                        "kind": kind,
                        "box": list(box),
                        "provenance": prov,
                        "class_id": cid,
                        "note": f"{TASK}: {name} {row['sheet']} cell {i}, {cell.get('inliers')} SIFT inliers, reviewer {reviewer}",
                    }
                )
                added += 1
        # A page whose mark could not be added is neither a positive nor a negative: it stays unreviewed.
        positives = set(meta.get("page_ids", [])) | (accepted_pages - refused - carried)
        meta["page_ids"] = sorted(positives)
        meta["n_instances"] = len(positives)
        rejected = sorted(rejected_pages - positives - refused - carried)
        out = sorted(carried - positives)
        if out:
            meta["excluded_page_ids"] = sorted(set(meta.get("excluded_page_ids", [])) | set(out))
            exclusions[cid] = out
        meta["reviewed_negative_page_ids"] = sorted(
            set(meta.get("reviewed_negative_page_ids", [])) - positives | set(rejected)
        )
        negatives[cid] = sorted(set(rejected) - set(meta.get("excluded_page_ids", [])))
        meta["reviewed_negative_page_ids"] = sorted(
            set(meta["reviewed_negative_page_ids"]) - set(meta.get("excluded_page_ids", []))
        )
        audit = meta.setdefault("audit", {})
        audit[f"{TASK}_checked"] = {
            "proposal": name,
            "reviewed_by": reviewer,
            "reviewed_on": date.today().isoformat(),
            "sheets_answered": len(answered),
            "sheets_total": len(sheets.get(name, [])),
            "accepted": len(accepted_pages),
            "rejected": len(rejected),
        }
        changes.append(
            f"{name} -> {cid}: {len(answered)}/{len(sheets.get(name, []))} sheet(s), {added} new mark(s), "
            f"{len(negatives[cid])} reviewed negative(s); {meta['n_instances']} instance(s)"
            + (
                f"; {len(out)} page(s) on a clustered source excluded (box them with --task completeness): {out}"
                if out
                else ""
            )
        )
    return changes, problems, added_rows, negatives, exclusions


# ---------------------------------------------------------------------------
# SIFT on bands (GRID side; not exercised by the unit tests)
# ---------------------------------------------------------------------------

_MATCHER: Any = None
_FEATURES: dict[str, Any] = {}
_QUERY: Any = None


def _matcher():
    global _MATCHER
    if _MATCHER is None:
        from vtscore.media.structural import SiftMatcher  # noqa: PLC0415

        _MATCHER = SiftMatcher()
    return _MATCHER


def band_gray(path: str, width: int, height: int, frac: float = SEARCH_BAND_FRAC) -> np.ndarray:
    """The page's search band, resampled to ``TARGET_WIDTH`` across, as uint8 gray."""
    from PIL import Image  # noqa: PLC0415

    bw, bh = band_px(width, height, frac)
    with Image.open(path) as im:
        band = im.convert("L").crop((0, 0, bw, bh))
    f = TARGET_WIDTH / max(1, width)
    band = band.resize((max(1, round(bw * f)), max(1, round(bh * f))), Image.Resampling.BICUBIC)
    return np.asarray(band, dtype=np.uint8)


def crop_gray(path: str, page_width: int) -> np.ndarray:
    """A query crop resampled by the factor of the page it was cut from."""
    from PIL import Image  # noqa: PLC0415

    f = TARGET_WIDTH / max(1, page_width)
    with Image.open(path) as im:
        g = im.convert("L")
        g = g.resize((max(1, round(g.width * f)), max(1, round(g.height * f))), Image.Resampling.BICUBIC)
    return np.asarray(g, dtype=np.uint8)


def features(gray: np.ndarray, budget: int = BUDGET):
    return _matcher().detect_and_describe(gray, max_features=budget)


def inlier_points(template: Any, candidate: Any) -> Optional[np.ndarray]:
    """Normalised candidate-side inlier locations of a similarity fit, or ``None`` when it does not verify."""
    import cv2  # noqa: PLC0415

    from vtscore.media.structural import ratio_test_matches  # noqa: PLC0415

    t_desc, c_desc = template.descriptors_f32(), candidate.descriptors_f32()
    if len(t_desc) < 2 or len(c_desc) < 2:
        return None
    ((ti, ci),) = ratio_test_matches(t_desc, [c_desc], ratio=0.75)
    if len(ti) < 4:
        return None
    src = np.ascontiguousarray(template.keypoints_f32()[ti, :2])
    dst = np.ascontiguousarray(candidate.keypoints_f32()[ci, :2])
    model, mask = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESHOLD, maxIters=2000, confidence=0.99
    )
    if model is None or mask is None or not np.isfinite(model).all():
        return None
    keep = mask.ravel().astype(bool)
    if keep.sum() < 8 or float(np.hypot(model[0, 0], model[1, 0])) < MIN_SCALE:
        return None
    return dst[keep]


def _extract(item: tuple[str, str, int, int]) -> tuple[str, Any]:
    page_id, path, width, height = item
    try:
        return page_id, features(band_gray(path, width, height)).compact()
    except Exception as exc:  # noqa: BLE001 -- one unreadable page must not sink the search
        print(f"  extract failed {page_id}: {exc}", flush=True)
        return page_id, None


def _verify(page_id: str) -> Optional[tuple[str, int, list[float]]]:
    feats = _FEATURES.get(page_id)
    if feats is None:
        return None
    stats = _matcher().verify(_QUERY, feats)
    if not stats.model_ok or stats.inlier_count < KEEP_INLIERS or not stats.inlier_box:
        return None
    return page_id, int(stats.inlier_count), [round(v, 4) for v in stats.inlier_box]


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def _context(corpus: Path, bands: Path) -> tuple[dict[str, Any], dict[str, Any], list[Proposal], Any]:
    from sources._common import read_manifest  # noqa: PLC0415

    classes = json.loads((corpus / "classes.json").read_text(encoding="utf-8"))
    pages = {p.page_id: p for p in read_manifest(corpus / "corpus.jsonl")}
    return classes, pages, load_proposals(PROPOSALS, classes), np.load(bands)


def members_of(p: Proposal, bands: Any, threshold: float) -> list[list[str]]:
    vecs, authors, ids = bands["vectors"], bands["authors"], bands["page_ids"]
    return [[str(ids[i]) for i in component_members(authors, vecs, a, r, threshold)] for a, r in p.components]


def step_crops(corpus: Path, bands_path: Path, out: Path, threshold: float) -> int:
    from PIL import Image  # noqa: PLC0415

    classes, pages, proposals, bands = _context(corpus, bands_path)
    vec_of = {str(pid): i for i, pid in enumerate(bands["page_ids"])}
    rng = random.Random(SEED)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"threshold": threshold, "band_frac": SEARCH_BAND_FRAC, "target_width": TARGET_WIDTH}
    for p in proposals:
        groups = members_of(p, bands, threshold)
        flat = [m for g in groups for m in g]
        rows = np.array([vec_of[m] for m in flat])
        med = flat[medoid(bands["vectors"][rows])]
        entry: dict[str, Any] = {"members": [len(g) for g in groups], "medoid": med}
        page = pages[med]
        mfeat = features(band_gray(page.path, page.width, page.height))
        others = spread_sample([[m for m in g if m != med] for g in groups], CONSENSUS_SAMPLE, rng)
        point_sets = []
        for m in others:
            q = pages[m]
            pts = inlier_points(features(band_gray(q.path, q.width, q.height)), mfeat)
            if pts is not None:
                point_sets.append(pts)
        found = consensus_box(point_sets)
        auto: dict[str, Any] = {"verified": len(point_sets), "sampled": len(others)}
        if found is None:
            auto.update(box=None, flags=["no consensus"])
        else:
            norm, support = found
            auto.update(
                box=norm_to_page_box(norm, page.width, page.height),
                norm=[round(v, 4) for v in norm],
                support=support,
                flags=crop_flags(norm, support),
            )
        # The auto result is kept beside a hand crop, so the report can say what the rule got wrong.
        entry["auto"] = auto
        entry["needs_hand_crop"] = bool(auto["flags"])
        if p.crop:
            page = pages[p.crop["page_id"]]
            entry.update(
                page_id=page.page_id, box=[int(v) for v in p.crop["box"]], source="hand", why=p.crop.get("why")
            )
        elif auto["box"] is None:
            entry.update(page_id=med, box=None, source="auto")
            print(f"{p.name}: NO CONSENSUS ({len(point_sets)}/{len(others)} verified)", flush=True)
            report[p.name] = entry
            continue
        else:
            entry.update(page_id=med, box=auto["box"], source="auto")
        x, y, w, h = entry["box"]
        with Image.open(page.path) as im:
            im.convert("RGB").crop((x, y, x + w, y + h)).save(out / "crops" / f"{p.name}.png")
        entry["query_crop"] = str(out / "crops" / f"{p.name}.png")
        report[p.name] = entry
        print(f"{p.name}: {entry['source']} crop {entry['box']} on {entry['page_id']} auto={auto}", flush=True)
    (out / "crops.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


def step_search(
    corpus: Path, bands_path: Path, out: Path, threshold: float, workers: int, only: Sequence[str] = ()
) -> int:
    global _QUERY
    classes, pages, proposals, bands = _context(corpus, bands_path)
    crops = json.loads((out / "crops.json").read_text(encoding="utf-8"))
    comp = {m for p in proposals for g in members_of(p, bands, threshold) for m in g}
    pool = sorted(
        pid
        for pid, pg in pages.items()
        if (pg.source == "ucsf" and pg.meta.get("tier") in ("s", "m")) or pg.source == "tobacco800" or pid in comp
    )
    queries: dict[str, tuple[str, int]] = {}
    for p in proposals:
        c = crops.get(p.name, {})
        if c.get("query_crop"):
            queries[p.name] = (c["query_crop"], pages[c["page_id"]].width)
    for cid, meta in sorted(classes.items()):
        if meta.get("on_roster") and meta.get("source") == "tobacco800" and meta.get("query_crop"):
            queries[f"roster:{cid}"] = (meta["query_crop"], pages[meta["query_page_id"]].width)
    if only:
        # Re-verify just these queries (a re-cut crop) and merge into the last search.
        unknown = set(only) - set(queries)
        if unknown:
            raise ValueError(f"--only names unknown queries: {sorted(unknown)}")
        queries = {k: v for k, v in queries.items() if k in only or k.startswith("roster:")}
    print(f"=== {len(pool)} pages, {len(queries)} queries, budget {BUDGET}, {workers} workers", flush=True)

    t0 = time.time()
    items = [(pid, pages[pid].path, pages[pid].width, pages[pid].height) for pid in pool]
    with get_context("fork").Pool(workers) as mp:
        for i, (pid, feats) in enumerate(mp.imap_unordered(_extract, items, chunksize=8)):
            if feats is not None:
                _FEATURES[pid] = feats
            if (i + 1) % 2000 == 0:
                print(f"  extracted {i + 1}/{len(items)} in {time.time() - t0:.0f}s", flush=True)
    print(f"  extracted {len(_FEATURES)} bands in {time.time() - t0:.0f}s", flush=True)

    qfeats = {k: features(crop_gray(path, w)).compact() for k, (path, w) in queries.items()}
    hits: dict[str, dict[str, list]] = json.loads((out / "hits.json").read_text(encoding="utf-8")) if only else {}
    for key in queries:
        if only and key not in only:
            continue  # roster queries are unchanged; kept only for the crop-to-crop check
        t1 = time.time()
        _QUERY = qfeats[key]
        with get_context("fork").Pool(workers) as mp:
            found = [r for r in mp.map(_verify, pool, chunksize=64) if r]
        hits[key] = {pid: [n, *box] for pid, n, box in found}
        strong = sum(1 for _, n, _ in found if n >= MIN_INLIERS)
        print(f"  {key}: {strong} pages >= {MIN_INLIERS} inliers ({time.time() - t1:.0f}s)", flush=True)

    # Crop against crop: does a proposal's crop *be* a roster mark?
    cross: dict[str, dict[str, list[int]]] = (
        json.loads((out / "cross.json").read_text(encoding="utf-8")) if only else {}
    )
    for p in proposals:
        if p.name not in qfeats or (only and p.name not in only):
            continue
        cross[p.name] = {}
        for key in queries:
            if key.startswith("roster:"):
                ab = _matcher().verify(qfeats[p.name], qfeats[key])
                ba = _matcher().verify(qfeats[key], qfeats[p.name])
                cross[p.name][key[len("roster:") :]] = [
                    ab.inlier_count if ab.model_ok else 0,
                    ba.inlier_count if ba.model_ok else 0,
                ]
    (out / "hits.json").write_text(json.dumps(hits) + "\n", encoding="utf-8")
    (out / "cross.json").write_text(json.dumps(cross, indent=2) + "\n", encoding="utf-8")
    (out / "pool.json").write_text(json.dumps({"n": len(pool), "extracted": len(_FEATURES)}) + "\n", encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s")
    return 0


def plausible(norm: Sequence[float], expect: tuple[float, float], width: int, height: int) -> bool:
    """Is a hit's inlier box at least MIN_BOX_SHARE of the query's size on this page?

    *expect* is the query crop's ``(w, h)`` as fractions of its own page's width,
    so it carries across pages of different resolution.
    """
    bw, bh = band_px(width, height)
    w, h = abs(norm[2] - norm[0]) * bw, abs(norm[3] - norm[1]) * bh
    return w >= MIN_BOX_SHARE * expect[0] * width and h >= MIN_BOX_SHARE * expect[1] * width


def best_hit(
    hits: dict[str, dict[str, list]],
    keys: Sequence[str],
    page_id: str,
    expect: Optional[dict[str, tuple[float, float]]] = None,
    page_wh: Optional[tuple[int, int]] = None,
) -> tuple[int, Optional[list]]:
    """Strongest plausible hit on *page_id* over the query *keys*: ``(inliers, band-normalised box)``.

    With *expect* (per query) and *page_wh*, a collapsed fit (:func:`plausible`) does not count.
    """
    best: tuple[int, Optional[list]] = (0, None)
    for k in keys:
        row = hits.get(k, {}).get(page_id)
        if not row or row[0] <= best[0]:
            continue
        if expect is not None and page_wh is not None and k in expect and not plausible(row[1:], expect[k], *page_wh):
            continue
        best = (int(row[0]), row[1:])
    return best


def step_slates(corpus: Path, bands_path: Path, out: Path, threshold: float) -> int:
    from PIL import Image  # noqa: PLC0415

    import completeness as comp  # noqa: PLC0415

    classes, pages, proposals, bands = _context(corpus, bands_path)
    crops = json.loads((out / "crops.json").read_text(encoding="utf-8"))
    hits = json.loads((out / "hits.json").read_text(encoding="utf-8"))
    cross = json.loads((out / "cross.json").read_text(encoding="utf-8"))
    slate = corpus / "audit" / TASK
    slate.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    expect: dict[str, tuple[float, float]] = {}
    for name, c in crops.items():
        if isinstance(c, dict) and c.get("box"):
            qw = pages[c["page_id"]].width
            expect[name] = (c["box"][2] / qw, c["box"][3] / qw)
    for cid, meta in classes.items():
        if meta.get("on_roster") and meta.get("query_crop") and meta.get("query_page_id") in pages:
            with Image.open(meta["query_crop"]) as im:
                qw = pages[meta["query_page_id"]].width
                expect[f"roster:{cid}"] = (im.width / qw, im.height / qw)
    collapsed = Counter()

    def hit(pid: str, keys: Sequence[str]) -> tuple[int, Optional[list]]:
        pg = pages[pid]
        raw = best_hit(hits, keys, pid)
        got = best_hit(hits, keys, pid, expect, (pg.width, pg.height))
        if raw[0] >= MIN_INLIERS and got[0] < MIN_INLIERS:
            collapsed[keys[0]] += 1
        return got

    def refs_for(p: Proposal) -> list[tuple[str, Any]]:
        refs = []
        with Image.open(crops[p.name]["query_crop"]) as im:
            refs.append((f"CROP ({crops[p.name]['source']})", im.convert("RGB").copy()))
        if p.roster_class:
            meta = classes[p.roster_class]
            with Image.open(meta["query_crop"]) as im:
                refs.append(("ROSTER QUERY", im.convert("RGB").copy()))
            for pid in meta["page_ids"][:3]:
                page = pages[pid]
                mark = next((m for m in page.marks if m.class_id == p.roster_class), None)
                if mark:
                    x, y, w, h = mark.box
                    with Image.open(page.path) as im:
                        refs.append(("roster member", im.convert("RGB").crop((x, y, x + w, y + h))))
        return refs

    def sheet(
        p: Proposal,
        part: str,
        title: str,
        cells: list[tuple[str, int, list[int], bool]],
        extra: dict[str, Any],
        k: int = 0,
    ):
        # render() always writes <class_id>_00.png; a later sheet of the same part renders under its own
        # name first, or it would overwrite sheet 0 before being moved to _0k.
        entry = comp.ClassCandidates(
            class_id=f"{p.name}__{part}" if not k else f"{p.name}__{part}__sheet{k}",
            candidates=[comp.Candidate(page_id=pid, inliers=n, box=tuple(box)) for pid, n, box, _ in cells],
        )
        (path,) = comp.render(
            entry,
            {},
            pages,
            slate,
            per_sheet=PER_SHEET,
            refs=refs_for(p),
            title=title,
            unboxed_label="",
            min_context=CONTEXT_PX,
        )
        if k:
            path = path.rename(path.with_name(f"{p.name}__{part}_{k:02d}.png"))
        rows.append(
            {
                "task": TASK,
                "proposal": p.name,
                "part": part,
                "sheet": path.name,
                "sheet_index": k,
                **extra,
                "cells": [
                    {"index": i, "page_id": pid, "inliers": n, "box": box, "located": located}
                    for i, (pid, n, box, located) in enumerate(cells)
                ],
                # Which SHOWN cells carry this proposal's mark: all | none | 0,3,7 | all but 3,7.
                # It never reaches a page this sheet does not show.
                "verdict": "",
                # A first pass's answer, for the reviewer to confirm; never applied.
                "suggestion": "",
                "uncertain": False,
                "note": "",
            }
        )

    for p in proposals:
        if not crops.get(p.name, {}).get("query_crop"):
            print(f"{p.name}: no crop, no slate")
            continue
        rng = random.Random(f"{SEED}:{p.name}")
        keys = [p.name] + ([f"roster:{p.roster_class}"] if p.roster_class else [])
        groups = members_of(p, bands, threshold)
        in_comp = {m for g in groups for m in g}

        def cell(pid: str) -> tuple[str, int, list[int], bool]:
            n, norm = hit(pid, keys)
            pg = pages[pid]
            located = bool(norm) and n >= MIN_INLIERS
            box = norm_to_page_box(norm, pg.width, pg.height) if norm else [0, 0, *band_px(pg.width, pg.height)]
            return pid, n, box, located

        def emit(part: str, label: str, pages_by_sheet: list[list[str]], total: int, extra: dict[str, Any]) -> None:
            shown = 0
            for k, ids in enumerate(pages_by_sheet):
                what = f"sheet {k + 1}/{len(pages_by_sheet)}: pages {shown + 1}-{shown + len(ids)} of {total}"
                if k == 0 and len(pages_by_sheet) > 1:
                    what += " (a random draw)"
                sheet(
                    p,
                    part,
                    f"{p.name}  {label}  {what}",
                    [cell(pid) for pid in ids],
                    dict(extra, n_sheets=len(pages_by_sheet)),
                    k,
                )
                shown += len(ids)

        # (a) one mark?  A sample across the band class, then every other member in s/m.
        sample = spread_sample(groups, PER_SHEET, rng)
        comp_sm = sorted(m for m in in_comp if pages[m].meta.get("tier") in ("s", "m"))
        rest_sm = [m for m in comp_sm if m not in set(sample)]
        emit(
            "a_members",
            "(a) ONE MARK?",
            [sample] + chunks(rest_sm),
            len(sample) + len(rest_sm),
            {"n_band_class": len(in_comp), "note_scope": "sheet 1 samples all tiers; later sheets are the rest of s+m"},
        )
        # (b) every search hit outside the band class
        eligible_sources = ("ucsf",) if p.relation == "extends" else ("ucsf", "tobacco800")
        roster_members = set(classes.get(p.roster_class, {}).get("page_ids", [])) if p.roster_class else set()
        scored = []
        for pid in {pid for k in keys for pid in hits.get(k, {})}:
            pg = pages[pid]
            if pid in in_comp or pid in roster_members or pg.source not in eligible_sources:
                continue
            if pg.source == "ucsf" and pg.meta.get("tier") not in ("s", "m"):
                continue
            n, _ = hit(pid, keys)
            if n >= MIN_INLIERS:
                scored.append((pid, n))
        by_bin = {}
        for lo, hi, items in bins(scored):
            by_bin[bin_label(lo, hi)] = len(items)
            if not items:
                continue
            ordered = sheet_order(items, PER_SHEET, rng, key=lambda t: (-t[1], t[0]))
            emit(
                f"b_{bin_label(lo, hi)}",
                f"(b) HITS {bin_label(lo, hi)} inliers",
                [[pid for pid, _ in chunk] for chunk in ordered],
                len(items),
                {"bin": [lo, hi], "n_population": len(items)},
            )
        # (c) every band-class member in s/m the search missed
        missed = [m for m in comp_sm if hit(m, keys)[0] < MIN_INLIERS]
        if missed:
            emit(
                "c_missed",
                f"(c) MISSED s/m band-class members under {MIN_INLIERS}",
                sheet_order(missed, PER_SHEET, rng, key=lambda m: m),
                len(missed),
                {"n_population": len(missed)},
            )
        comp_all_scores = [hit(m, keys)[0] for m in sorted(in_comp)]
        summary[p.name] = {
            "relation": p.relation,
            "roster_class": p.roster_class,
            "band_class": len(in_comp),
            "band_class_sm": len(comp_sm),
            "band_class_found": sum(1 for n in comp_all_scores if n >= MIN_INLIERS),
            "band_class_sm_found": len(comp_sm) - len(missed),
            "hits_outside_by_bin": by_bin,
            "hits_outside": len(scored),
            "collapsed_fits_dropped": collapsed[p.name],
            "crop": {k: crops[p.name].get(k) for k in ("source", "page_id", "box", "needs_hand_crop", "auto")},
            "cross_roster_top": sorted(
                ((cid, max(v)) for cid, v in cross.get(p.name, {}).items()), key=lambda t: -t[1]
            )[:3],
        }
        rows.append(
            {
                "task": f"{TASK}_relation",
                "proposal": p.name,
                "suggested_relation": f"extends {p.roster_class}" if p.roster_class else "new",
                # extends <class_id> | new | reject.  Blank = not reviewed; the suggestion is never applied.
                "relation": "",
                "query_crop": crops[p.name]["query_crop"],
                "query_page_id": crops[p.name]["page_id"],
                "query_box": crops[p.name]["box"],
                "uncertain": False,
                "note": p.note,
            }
        )
        print(f"{p.name}: {json.dumps(summary[p.name])}", flush=True)

    # A re-render names its sheets by bin; a bin that emptied must not leave its old sheet behind.
    written = {r["sheet"] for r in rows if r["task"] == TASK}
    for stale in slate.glob("*.png"):
        if stale.name not in written:
            stale.unlink()
    template = slate / "verdicts.jsonl"
    if template.exists():
        old = [json.loads(line) for line in template.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(f"carried answers onto {carry_answers(old, rows)} of {len(rows)} row(s) whose pages did not change")
    comp.write_template(rows, template)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    n_sheets = sum(1 for r in rows if r["task"] == TASK)
    print(f"\n{n_sheets} sheet(s) over {len(summary)} proposal(s) -> {slate}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("crops", "search", "slates"))
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--bands", type=Path, default=cfg.OUT.parent / "bands-3902" / "bands.npz", help="band_siglip.py embed output"
    )
    ap.add_argument("--threshold", type=float, default=0.10, help="the band-class cut the proposals were named at")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--only", default="", help="search: comma-separated proposal names to re-verify and merge")
    args = ap.parse_args(argv)
    if args.out.resolve().is_relative_to(args.corpus.resolve()):
        ap.error("--out is inside the corpus; only the slate is written there")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.step == "crops":
        return step_crops(args.corpus, args.bands, args.out, args.threshold)
    if args.step == "search":
        return step_search(
            args.corpus, args.bands, args.out, args.threshold, args.workers, [n for n in args.only.split(",") if n]
        )
    return step_slates(args.corpus, args.bands, args.out, args.threshold)


if __name__ == "__main__":
    raise SystemExit(main())
