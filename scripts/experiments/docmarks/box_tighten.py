"""Tighter boxes for a class's members, chosen by a person.

A source box is the retrieval target, and a loose one moves the target: a
StaVer page whose ``Projekt / Anerkennung`` form stamp was boxed together with
the separate ``EINGEGANGEN AM`` date stamp beside it (``staver/stampds-00218``,
570x414 where the form stamp alone is ~556x282) asks a matcher to find two marks
and scores a localiser against a region only half of which is the class.

The pass follows the other audits' shape:

* :func:`propose` fits the class's query crop to each member with SIFT and a
  similarity transform, and proposes the bounds of the query's *ink* projected
  onto the page.  Not the crop's rectangle: a query crop cut from a copy
  stamped a few degrees askew carries corner margin, and projecting that
  rectangle onto an upright copy added ~25% to every box's height on the first
  slate.  The ink -- pixels far from the crop's paper colour -- is the mark.
  The inlier bounds are kept too, as a check: a projection the inliers barely
  span is a fit to distrust.
* :func:`render` draws each member on one screen: the page region with the OLD
  box in grey and the PROPOSED box in red, numbered large.
* :func:`apply_box_tighten` folds the verdict back: accepted boxes replace the
  mark's box *in place* (same page, same mark index, so every must-link and
  cannot-link keyed on ``(page_id, mark_index)`` still names the same mark) and
  go to ``box_overrides.json``, which ``build_corpus.py`` replays before
  clustering so a rebuild keeps them.

Nothing here decides; a proposal is only drawn.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Optional, Sequence

from sources._common import Mark, Page

STORE = "box_overrides.json"
#: Fewer inliers than this and the fit is flagged; the box may still be right.
MIN_INLIERS = 12
#: A proposal reaching past the old box at all is flagged, and past this share
#: of the old box's side it is clamped: the pass is for fixing a box, not for
#: following a wild fit across the page.  Generous on purpose, because a source
#: box can clip its mark too (``stampds-00237``'s cuts the stamp's left edge).
OVERSHOOT = 0.15
#: A proposal none of whose edges moves more than this share of the old box's
#: side is "unchanged".  Per edge, not IoU: an IoU of 0.9 hid ``stampds-00237``,
#: whose left edge moves 34 px (12% of the width) to take in a clipped frame.
UNCHANGED = 0.03
#: ``--loose-only`` keeps members whose proposal is at most this share of the old box.
LOOSE = 0.8
#: The inliers should span a fair part of the projected outline.
MIN_SPAN = 0.25
#: Page context around the old box searched for the mark.
CONTEXT = 0.5
#: A query pixel is ink when its RGB distance from the crop's median colour
#: (the paper) exceeds this.
INK_DISTANCE = 60
#: Projected ink bounds are taken between these percentiles, so a speck of
#: dirt in the query crop does not stretch every proposal.
INK_TRIM = (0.5, 99.5)

Box = tuple[int, int, int, int]  # x, y, w, h in page pixels


@dataclass
class Proposal:
    page_id: str
    mark_index: int
    old_box: Box
    new_box: Optional[Box] = None
    inliers: int = 0
    inlier_box: Optional[Box] = None
    flags: list[str] = field(default_factory=list)


def area(box: Optional[Sequence[int]]) -> int:
    return 0 if box is None else max(0, int(box[2])) * max(0, int(box[3]))


def edge_shift(new: Sequence[int], old: Sequence[int]) -> float:
    """Largest edge move from *old* to *new*, as a share of *old*'s side along that edge."""
    w, h = max(1, old[2]), max(1, old[3])
    return max(
        abs(new[0] - old[0]) / w,
        abs(new[0] + new[2] - old[0] - old[2]) / w,
        abs(new[1] - old[1]) / h,
        abs(new[1] + new[3] - old[1] - old[3]) / h,
    )


#: Provenance suffix of a mark located only by its letterhead band (#3953, #4088):
#: the mark is somewhere inside the box, not boxed.
BAND_SUFFIX = "_band"


def is_band(mark: Mark) -> bool:
    return str(mark.provenance).endswith(BAND_SUFFIX)


def located_provenance(provenance: str) -> str:
    """The provenance a band-located mark takes once a person accepts a tight box for it (#4109)."""
    return provenance[: -len(BAND_SUFFIX)] if provenance.endswith(BAND_SUFFIX) else provenance


def members(class_id: str, meta: dict[str, Any], pages: dict[str, Page], *, band_only: bool = False) -> list[Proposal]:
    """Every boxed instance of *class_id*, in page order, as an empty proposal.

    ``band_only`` keeps only band-located instances: the pass that gives them a
    real box (#4109).
    """
    out = []
    for page_id in sorted(meta.get("page_ids", [])):
        page = pages.get(page_id)
        if page is None:
            continue
        for i, mark in enumerate(page.marks):
            if mark.class_id == class_id and area(mark.box) > 0 and (not band_only or is_band(mark)):
                out.append(Proposal(page_id, i, tuple(int(v) for v in mark.box)))
    return out


def propose(
    prop: Proposal,
    corners: Optional[Sequence[tuple[float, float]]],
    inlier_pts: Optional[Sequence[tuple[float, float]]],
    page_size: tuple[int, int],
    within: Optional[Box] = None,
) -> Proposal:
    """Turn a fit (projected query corners and inlier points, page pixels) into a box and flags.

    The proposal is clamped near the old box -- or near *within*, when the old
    box may itself be the wrong shape and a larger region (a letterhead band)
    is known to hold the mark.
    """
    if corners is None or not inlier_pts:
        prop.flags.append("no fit")
        return prop
    pw, ph = page_size
    ox, oy, ow, oh = within or prop.old_box
    prop.inliers = len(inlier_pts)
    xs, ys = [c[0] for c in corners], [c[1] for c in corners]
    x0, y0, x1, y1 = max(0.0, min(xs)), max(0.0, min(ys)), min(float(pw), max(xs)), min(float(ph), max(ys))
    # Clamp to the old box plus a margin: a tightening pass shrinks boxes.
    overshoot = x0 < ox or y0 < oy or x1 > ox + ow or y1 > oy + oh
    lx0, ly0 = ox - OVERSHOOT * ow, oy - OVERSHOOT * oh
    lx1, ly1 = ox + ow + OVERSHOOT * ow, oy + oh + OVERSHOOT * oh
    clamped = x0 < lx0 or y0 < ly0 or x1 > lx1 or y1 > ly1
    x0, y0, x1, y1 = max(x0, lx0), max(y0, ly0), min(x1, lx1), min(y1, ly1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        prop.flags.append("no fit")
        return prop
    prop.new_box = (int(round(x0)), int(round(y0)), int(round(x1 - x0)), int(round(y1 - y0)))
    ix = [p[0] for p in inlier_pts]
    iy = [p[1] for p in inlier_pts]
    prop.inlier_box = (int(min(ix)), int(min(iy)), int(max(ix) - min(ix)), int(max(iy) - min(iy)))
    ox, oy, ow, oh = prop.old_box
    if edge_shift(prop.new_box, prop.old_box) <= UNCHANGED:
        # Nothing to decide, so no warnings to read either.
        prop.flags.append("unchanged")
        return prop
    if clamped:
        prop.flags.append("clamped: fit reaches far past old box")
    elif overshoot:
        prop.flags.append("grows the old box")
    if prop.inliers < MIN_INLIERS:
        prop.flags.append("few inliers")
    if area(prop.inlier_box) < MIN_SPAN * area(prop.new_box):
        prop.flags.append("inliers span little of the box")
    return prop


def verdict_row(class_id: str, props: Sequence[Proposal]) -> dict[str, Any]:
    return {
        "task": "box_tighten",
        "class_id": class_id,
        "members": [
            {
                "index": i,
                "page_id": p.page_id,
                "mark_index": p.mark_index,
                "old_box": list(p.old_box),
                "new_box": list(p.new_box) if p.new_box else None,
                "inliers": p.inliers,
                "flags": p.flags,
            }
            for i, p in enumerate(props)
        ],
        # "none", or the comma-separated member numbers whose PROPOSED box to accept.
        "verdict": "",
        "notes": "",
    }


def _parse(raw: str, n: int) -> tuple[Optional[list[int]], Optional[str]]:
    raw = raw.strip().lower()
    if not raw:
        return None, None
    if raw == "none":
        return [], None
    try:
        idx = [int(t) for t in raw.replace(" ", "").split(",") if t]
    except ValueError:
        return None, f"verdict must be 'none' or comma-separated member numbers, got {raw!r}"
    if len(set(idx)) != len(idx):
        return None, "a member number is repeated"
    bad = [i for i in idx if not 0 <= i < n]
    if bad:
        return None, f"member number(s) {bad} outside 0..{n - 1}"
    return sorted(idx), None


def load_store(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")).get("boxes", []) if path.exists() else []


def save_store(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.write_text(json.dumps({"boxes": list(rows)}, indent=2) + "\n", encoding="utf-8")


def apply_box_tighten(
    pages: Sequence[Page],
    classes: dict[str, Any],
    verdicts: Sequence[dict[str, Any]],
    store: list[dict[str, Any]],
    *,
    reviewer: Optional[str] = None,
) -> tuple[list[str], list[str], set[str]]:
    """Replace accepted boxes in place and upsert them into *store*.

    Returns ``(changes, problems, classes whose query page's box changed)``.  A
    member whose mark no longer has the box the sheet showed is a problem, not a
    silent overwrite: the corpus moved since the slate was drawn.
    """
    by_id = {p.page_id: p for p in pages}
    changes: list[str] = []
    problems: list[str] = []
    stale_queries: set[str] = set()
    for row in verdicts:
        class_id = row["class_id"]
        meta = classes.get(class_id)
        if meta is None:
            problems.append(f"{class_id}: not in classes.json")
            continue
        rows = row.get("members", [])
        accepted, error = _parse(str(row.get("verdict", "")), len(rows))
        if error:
            problems.append(f"{class_id}: {error}")
            continue
        if accepted is None:
            continue
        done = 0
        for i in accepted:
            m = rows[i]
            page = by_id.get(m["page_id"])
            idx = m["mark_index"]
            if not m.get("new_box"):
                problems.append(f"{class_id}: member {i} has no proposed box to accept")
                continue
            if page is None or not 0 <= idx < len(page.marks):
                problems.append(f"{class_id}: member {i} ({m['page_id']} mark {idx}) is not in the manifest")
                continue
            mark = page.marks[idx]
            if tuple(mark.box) != tuple(m["old_box"]) or mark.class_id != class_id:
                problems.append(
                    f"{class_id}: member {i} ({m['page_id']} mark {idx}) no longer has the box the sheet showed"
                )
                continue
            new_box = tuple(int(v) for v in m["new_box"])
            # An accepted tight box is a located mark, whatever located it before.
            provenance = located_provenance(mark.provenance)
            page.marks[idx] = Mark(mark.kind, new_box, mark.class_id, provenance)
            key = (page.page_id, idx)
            store[:] = [r for r in store if (r["page_id"], r["mark_index"]) != key]
            store.append(
                {
                    "page_id": page.page_id,
                    "mark_index": idx,
                    "class_id": class_id,
                    "old_box": list(m["old_box"]),
                    "new_box": list(new_box),
                    "provenance": provenance,
                    "reviewed_by": reviewer,
                }
            )
            if page.page_id == meta.get("query_page_id"):
                stale_queries.add(class_id)
            done += 1
        meta.setdefault("audit", {})["box_tighten_checked"] = {
            "reviewed_by": reviewer,
            "reviewed_on": date.today().isoformat(),
            "n_members": len(rows),
            "accepted": done,
        }
        changes.append(f"{class_id}: {done} of {len(rows)} box(es) tightened")
    return changes, problems, stale_queries


def replay_box_overrides(
    pages: Sequence[Page], rows: Sequence[dict[str, Any]], warnings: Optional[list[str]] = None
) -> int:
    """Put stored tightened boxes back on a fresh build, by mark index.

    Called by ``build_corpus.py`` after the hand-added marks (an override may
    name one) and before clustering.  The box is replaced in place, so the mark
    index -- what every adjudication names -- is unchanged.  A row whose mark
    has neither its old nor its new box is warned about and skipped: the source
    annotations changed under it and a person should look.
    """
    by_id = {p.page_id: p for p in pages}
    n = 0
    for row in rows:
        page = by_id.get(row["page_id"])
        idx = int(row["mark_index"])
        if page is None:
            if warnings is not None:
                warnings.append(f"box override on {row['page_id']}: page not in this build")
            continue
        old, new = tuple(row["old_box"]), tuple(row["new_box"])
        if 0 <= idx < len(page.marks) and tuple(page.marks[idx].box) == old:
            mark = page.marks[idx]
            page.marks[idx] = Mark(mark.kind, new, mark.class_id, row.get("provenance", mark.provenance))
            n += 1
        elif not (0 <= idx < len(page.marks) and tuple(page.marks[idx].box) == new):
            if warnings is not None:
                warnings.append(
                    f"box override on {row['page_id']} mark {idx}: the mark no longer has box {list(old)}; skipped"
                )
    return n


def recut_query_crop(meta: dict[str, Any], page: Page, class_id: str) -> Optional[Path]:
    """Re-cut a class's primary query crop from its (tightened) box on the query page."""
    from PIL import Image  # noqa: PLC0415

    from sources import _common  # noqa: PLC0415

    idx = next((i for i, m in enumerate(page.marks) if m.class_id == class_id), None)
    if idx is None or not meta.get("query_crop"):
        return None
    x, y, w, h = page.marks[idx].box
    with Image.open(page.path) as im:
        return _common.save_verified(im.convert("RGB").crop((x, y, x + w, y + h)), Path(meta["query_crop"]))


# ---------------------------------------------------------------------------
# The slate: matcher and sheets (GRID side; not exercised by the unit tests)
# ---------------------------------------------------------------------------


def ink_points(rgb, limit: int = 20000):
    """``(N, 2)`` float ``x, y`` of the query crop's ink pixels (at most *limit*, evenly subsampled)."""
    import numpy as np  # noqa: PLC0415

    arr = np.asarray(rgb, dtype=np.float32)
    paper = np.median(arr.reshape(-1, 3), axis=0)
    ys, xs = np.nonzero(np.linalg.norm(arr - paper, axis=2) > INK_DISTANCE)
    if xs.size == 0:
        h, w = arr.shape[:2]
        return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    step = max(1, xs.size // limit)
    return np.stack([xs[::step], ys[::step]], axis=1).astype(np.float32)


def sift_fitter(query_crop: str, budget: int, upsample: float = 1.0):
    """``fit(page, box) -> (ink bounds corners, inlier points)`` in page pixels, or ``(None, None)``.

    ``upsample`` enlarges both the crop and the searched region before
    detection.  A UCSF letterhead mark is often under 60 px on a 150 dpi page,
    where SIFT finds too few keypoints to fit; the geometry is unchanged, and
    the result is returned in page pixels.
    """
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    from vtscore.media.structural import _LOWE_RATIO, SiftMatcher, ratio_test_matches  # noqa: PLC0415

    matcher = SiftMatcher()
    with Image.open(query_crop) as crop:
        q = crop.convert("L")
        if upsample != 1.0:
            q = q.resize((round(q.width * upsample), round(q.height * upsample)), Image.Resampling.BICUBIC)
        qw, qh = q.size
        q_feats = matcher.detect_and_describe(np.asarray(q, dtype=np.uint8), max_features=budget)
        ink = ink_points(crop.convert("RGB")) * upsample
    q_kp = q_feats.keypoints_f32()[:, :2] * np.array([qw, qh], dtype=np.float32)

    def fit(page: Page, box: Box):
        x, y, w, h = box
        left, top = max(0, int(x - CONTEXT * w)), max(0, int(y - CONTEXT * h))
        right, bottom = min(page.width, int(x + w + CONTEXT * w)), min(page.height, int(y + h + CONTEXT * h))
        with Image.open(page.path) as im:
            region = im.convert("L").crop((left, top, right, bottom))
        if upsample != 1.0:
            region = region.resize(
                (round(region.width * upsample), round(region.height * upsample)), Image.Resampling.BICUBIC
            )
        rw, rh = region.size
        feats = matcher.detect_and_describe(np.asarray(region, dtype=np.uint8), max_features=budget)
        if q_feats.descriptors_f32().shape[0] < 2 or feats.descriptors_f32().shape[0] < 2:
            return None, None
        ((t_idx, c_idx),) = ratio_test_matches(q_feats.descriptors_f32(), [feats.descriptors_f32()], ratio=_LOWE_RATIO)
        if len(t_idx) < 4:
            return None, None
        src = np.ascontiguousarray(q_kp[t_idx], dtype=np.float32)
        dst = np.ascontiguousarray(
            feats.keypoints_f32()[c_idx, :2] * np.array([rw, rh], dtype=np.float32) / upsample + np.array([left, top]),
            dtype=np.float32,
        )
        model, mask = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=max(3.0, 0.004 * max(rw, rh) / upsample),
            maxIters=4000,
        )
        if model is None or mask is None or not np.isfinite(model).all():
            return None, None
        keep = mask.ravel().astype(bool)
        if keep.sum() < 4:
            return None, None
        projected = ink @ model[:, :2].T + model[:, 2]
        x0, x1 = np.percentile(projected[:, 0], INK_TRIM)
        y0, y1 = np.percentile(projected[:, 1], INK_TRIM)
        corners = [(float(x0), float(y0)), (float(x1), float(y0)), (float(x1), float(y1)), (float(x0), float(y1))]
        return corners, [tuple(map(float, p)) for p in dst[keep]]

    return fit


def band_of(page: Page) -> Optional[Box]:
    """The page's letterhead band: its unclassed ``candidate`` mark, if it has one."""
    for m in page.marks:
        if m.class_id is None and m.provenance == "candidate":
            return tuple(int(v) for v in m.box)
    return None


def union(a: Box, b: Box) -> Box:
    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
    return (x0, y0, x1 - x0, y1 - y0)


def padded(box: Box, pad: float, page_size: tuple[int, int]) -> Box:
    """*box* grown by *pad* of its width and height on each edge, clipped to the page."""
    x, y, w, h = box
    dx, dy = round(pad * w), round(pad * h)
    x0, y0 = max(0, x - dx), max(0, y - dy)
    x1, y1 = min(page_size[0], x + w + dx), min(page_size[1], y + h + dy)
    return (x0, y0, x1 - x0, y1 - y0)


def best_fit(fits, page: Page, box: Box):
    """The fit with the most inliers over several references, or ``(None, None)``."""
    best = (None, None)
    for fit in fits:
        corners, pts = fit(page, box)
        if corners is not None and pts and len(pts) > len(best[1] or ()):
            best = (corners, pts)
    return best


def reference_crops(class_id: str, meta: dict[str, Any], pages: dict[str, Page], out: Path, k: int) -> list[str]:
    """Crops of the class's *k* largest tightly boxed instances, written under *out*.

    A single query crop is one copy of the mark. Band-located copies vary in
    print and scan, and a second or third reference fits the ones the first
    misses.  Band-located instances are never references: their box is a band.
    """
    if k <= 0:
        return []
    from PIL import Image  # noqa: PLC0415

    tight = []
    for page_id in meta.get("page_ids", []):
        page = pages.get(page_id)
        if page is None or page_id == meta.get("query_page_id"):
            continue
        for mark in page.marks:
            if mark.class_id == class_id and not is_band(mark) and area(mark.box) > 0:
                tight.append((area(mark.box), page, mark.box))
                break
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for _a, page, (x, y, w, h) in sorted(tight, key=lambda t: -t[0])[:k]:
        dst = out / f"{page.page_id.replace('/', '_').replace('#', '_')}.png"
        with Image.open(page.path) as im:
            im.convert("RGB").crop((x, y, x + w, y + h)).save(dst)
        paths.append(str(dst))
    return paths


#: Members per sheet; one screen (the owner's rule).
PER_SHEET = 18


def render(
    class_id: str, meta: dict[str, Any], props: Sequence[Proposal], pages: dict[str, Page], out: Path
) -> list[Path]:
    """Sheets of numbered members: the page region, OLD box grey, PROPOSED box red."""
    from PIL import Image, ImageDraw  # noqa: PLC0415

    from completeness import COLS, _font  # noqa: PLC0415

    # Tiles are landscape: document stamps and logos are wider than tall, and
    # a square tile spends a third of the screen on white.
    thumb, tile_h, pad, cap = 250, 190, 10, 40
    big, small, title = _font(40, bold=True), _font(15), _font(18, bold=True)
    width = COLS * (thumb + pad) + pad
    with Image.open(meta["query_crop"]) as im:
        query = im.convert("RGB").copy()
    query.thumbnail((thumb, tile_h))
    paths = []
    chunks = [props[i : i + PER_SHEET] for i in range(0, len(props), PER_SHEET)] or [[]]
    for s, chunk in enumerate(chunks):
        rows = 1 + (len(chunk) + COLS - 1) // COLS
        sheet = Image.new("RGB", (width, 44 + rows * (tile_h + cap + pad)), "white")
        draw = ImageDraw.Draw(sheet)
        head = f"{class_id}   accept the RED box?  (grey = current box; red may be tighter OR larger)"
        if len(chunks) > 1:
            head += f"   sheet {s + 1}/{len(chunks)}"
        draw.text((pad, 10), head, fill="black", font=title)
        draw.text((pad, 44), "QUERY CROP", fill="#b00000", font=small)
        sheet.paste(query, (pad, 44 + cap))  # thumbnailed to (thumb, tile_h) above
        for k, prop in enumerate(chunk):
            number = s * PER_SHEET + k
            page = pages[prop.page_id]
            boxes = [b for b in (prop.old_box, prop.new_box) if b]
            x0 = min(b[0] for b in boxes)
            y0 = min(b[1] for b in boxes)
            x1 = max(b[0] + b[2] for b in boxes)
            y1 = max(b[1] + b[3] for b in boxes)
            margin = int(0.12 * max(x1 - x0, y1 - y0))
            left, top = max(0, x0 - margin), max(0, y0 - margin)
            with Image.open(page.path) as im:
                crop = im.convert("RGB").crop((left, top, min(page.width, x1 + margin), min(page.height, y1 + margin)))
            d = ImageDraw.Draw(crop)
            lw = max(3, crop.width // 120)
            ox, oy, ow, oh = prop.old_box
            d.rectangle([ox - left, oy - top, ox - left + ow, oy - top + oh], outline="#8c8c8c", width=lw)
            if prop.new_box:
                nx, ny, nw, nh = prop.new_box
                d.rectangle([nx - left, ny - top, nx - left + nw, ny - top + nh], outline="#d62728", width=lw)
            r, c = divmod(k, COLS)
            cx, cy = pad + c * (thumb + pad), 44 + (r + 1) * (tile_h + cap + pad)
            crop.thumbnail((thumb, tile_h))
            sheet.paste(crop, (cx, cy + cap))
            draw.rectangle([cx, cy + cap, cx + thumb - 1, cy + cap + tile_h - 1], outline="#cccccc")
            size = f"{ow}x{oh} → {prop.new_box[2]}x{prop.new_box[3]}" if prop.new_box else f"{ow}x{oh} → ?"
            line1 = f"{prop.inliers} inl · {size}"
            line2 = f"{prop.page_id.split('/')[-1].removeprefix('stampds-')} m{prop.mark_index}"
            if prop.flags:
                line2 += " · " + ", ".join(prop.flags)
            for j, text in enumerate((line1, line2)):
                while draw.textlength(text, font=small) > thumb and len(text) > 1:
                    text = text[:-1]
                draw.text((cx, cy + 2 + 18 * j), text, fill="#a05a00" if j and prop.flags else "#333333", font=small)
            nb = draw.textbbox((0, 0), str(number), font=big)
            bw, bh = nb[2] - nb[0] + 16, nb[3] - nb[1] + 14
            draw.rectangle([cx, cy + cap, cx + bw, cy + cap + bh], fill="#111111")
            draw.text((cx + 8 - nb[0], cy + cap + 7 - nb[1]), str(number), fill="#ffffff", font=big)
        path = out / f"{class_id.replace('/', '__')}_{s:02d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(path)
        paths.append(path)
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse  # noqa: PLC0415

    import docmarks_config as cfg  # noqa: PLC0415
    from sources._common import read_manifest  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--classes", default="", help="comma-separated class ids (default: every roster class of --source)")
    ap.add_argument("--source", default="staver")
    ap.add_argument("--budget", type=int, default=8192, help="SIFT keypoint budget")
    ap.add_argument(
        "--band-only",
        action="store_true",
        help="propose boxes only for marks located by their letterhead band (#4109)",
    )
    ap.add_argument("--upsample", type=float, default=1.0, help="enlarge crop and page region before SIFT")
    ap.add_argument(
        "--extra-refs",
        type=int,
        default=0,
        help="also fit the class's N largest tightly boxed instances, keeping the fit with the most inliers",
    )
    ap.add_argument("--audit-dir", default="box_tighten", help="slate directory under <corpus>/audit/")
    ap.add_argument(
        "--within-band",
        action="store_true",
        help="search and clamp within the page's letterhead band, not the current box, which may be the wrong shape",
    )
    ap.add_argument(
        "--pad",
        type=float,
        default=0.0,
        help="grow each proposal by this share of its side, per edge: a fit a few pixels off then still holds the mark",
    )
    ap.add_argument(
        "--loose-only",
        action="store_true",
        help=f"sheet only members whose proposal is at most {LOOSE:.0%} of the current box's area",
    )
    args = ap.parse_args(argv)

    classes = json.loads((args.corpus / "classes.json").read_text(encoding="utf-8"))
    pages = {p.page_id: p for p in read_manifest(args.corpus / "corpus.jsonl")}
    if args.classes:
        chosen = [c.strip() for c in args.classes.split(",") if c.strip()]
    else:
        chosen = sorted(c for c, m in classes.items() if m.get("on_roster") and m.get("source") == args.source)
    out = args.corpus / "audit" / args.audit_dir
    rows = []
    for class_id in chosen:
        meta = classes[class_id]
        refs = [meta["query_crop"]] + reference_crops(
            class_id, meta, pages, out / "refs" / class_id.replace("/", "_"), args.extra_refs
        )
        fits = [sift_fitter(r, args.budget, args.upsample) for r in refs]
        props = []
        for prop in members(class_id, meta, pages, band_only=args.band_only):
            page = pages[prop.page_id]
            within = band_of(page) if args.within_band else None
            search = union(prop.old_box, within) if within else prop.old_box
            corners, inlier_pts = best_fit(fits, page, search)
            prop = propose(prop, corners, inlier_pts, (page.width, page.height), within=search if within else None)
            if args.pad and prop.new_box and "unchanged" not in prop.flags:
                prop.new_box = padded(prop.new_box, args.pad, (page.width, page.height))
            props.append(prop)
        if args.loose_only:
            props = [p for p in props if p.new_box and area(p.new_box) <= LOOSE * area(p.old_box)]
        render(class_id, meta, props, pages, out)
        rows.append(verdict_row(class_id, props))
        loose = sum(1 for p in props if p.new_box and area(p.new_box) <= LOOSE * area(p.old_box))
        to_decide = sum(1 for p in props if p.flags != ["unchanged"])
        print(f"  {class_id}: {len(props)} member(s), {loose} loose, {to_decide} to decide", flush=True)
    (out / "verdicts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"\n{len(rows)} class(es) -> {out}; fill verdicts, then audit_to_corrections.py --task box_tighten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
