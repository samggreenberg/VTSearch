"""Several query crops per roster class, chosen by a person.

Every FullMarks number so far rests on **one** query crop per class, so a result
mixes two things: how good the method is, and how good that one crop happens to
be.  ``tobacco800/logo_aeq93a00_1``'s crop carries 211 SIFT keypoints and drags
its class to the bottom of every ranking.  Extra crops let a study average over
queries and report the spread.

The pass follows the other audits' shape:

* :func:`candidates` ranks a class's members as query material -- a large box
  (more pixels for the query are free) that matches the existing query crop
  strongly (so it is the same mark and the box holds it), one per source
  document so the choices are not the same scan twice.
* :func:`render` draws them on one screen: the current query crop, then each
  candidate *in context* with its box outlined, because a loose box is the most
  common defect and is only visible against its surroundings.
* :func:`apply_query_crops` folds the verdict back: the chosen boxes go to
  ``query_crops.json`` (the durable store, replayed by ``build_corpus.py`` after
  it writes the primary crop), each is cut to ``queries/<class>__q<n>.png``, and
  ``classes.json`` gains ``query_crops`` -- the primary first, then the extras.

The primary ``query_crop`` is untouched, so every existing study reads the same
query it always did.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from sources._common import Page

STORE = "query_crops.json"
#: Candidates shown per class; one sheet (the owner's one-screen rule).
TOP = 12
#: Extra crops suggested per class, in addition to the primary.
SUGGEST = 4


@dataclass
class CropCandidate:
    page_id: str
    mark_index: int
    box: tuple[int, int, int, int]
    inliers: int
    area: int


def document_of(page_id: str) -> str:
    """The source document a page belongs to.

    Only Tobacco800 splits one document across page ids (``aah97e00-page02_1``,
    ``bbb22b00_2``: an 8-character document id plus page/zone suffixes).  Every
    other source's page id already names one document -- StaVer's
    ``stampds-00213`` would collapse into a single "document" if split the same way.
    """
    source, _, local = page_id.partition("/")
    if source == "tobacco800":
        m = re.match(r"[a-z0-9]{8}", local)
        if m:
            return f"{source}/{m.group(0)}"
    return page_id


def candidates(
    class_id: str,
    meta: dict[str, Any],
    pages: dict[str, Page],
    inliers: dict[str, list[int]],
    *,
    top: int = TOP,
) -> list[CropCandidate]:
    """Members ranked as query material, at most one per source document."""
    primary_doc = document_of(meta.get("query_page_id", ""))
    pool = []
    for page_id in meta.get("page_ids", []):
        page = pages.get(page_id)
        if page is None or page_id == meta.get("query_page_id"):
            continue
        for i, mark in enumerate(page.marks):
            if mark.class_id == class_id and mark.area() > 0:
                pool.append(CropCandidate(page_id, i, tuple(mark.box), inliers.get(page_id, [0, 0])[0], mark.area()))
                break
    # Rank by match strength first (is it clearly this mark, with the box around
    # it), then by size (more pixels for the query); page id breaks ties.
    pool.sort(key=lambda c: (-c.inliers, -c.area, c.page_id))
    seen = {primary_doc}
    out = []
    for cand in pool:
        doc = document_of(cand.page_id)
        if doc in seen:
            continue
        seen.add(doc)
        out.append(cand)
        if len(out) >= top:
            break
    return out


def verdict_row(class_id: str, cands: Sequence[CropCandidate]) -> dict[str, Any]:
    return {
        "task": "query_crops",
        "class_id": class_id,
        "candidates": [
            {"index": i, "page_id": c.page_id, "mark_index": c.mark_index, "box": list(c.box), "inliers": c.inliers}
            for i, c in enumerate(cands)
        ],
        # "none", or the comma-separated candidate numbers to add as query crops.
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
        return None, f"verdict must be 'none' or comma-separated candidate numbers, got {raw!r}"
    if len(set(idx)) != len(idx):
        return None, "a candidate number is repeated"
    bad = [i for i in idx if not 0 <= i < n]
    if bad:
        return None, f"candidate number(s) {bad} outside 0..{n - 1}"
    return idx, None


def load_store(path: Path) -> dict[str, list[dict[str, Any]]]:
    return json.loads(path.read_text(encoding="utf-8")).get("classes", {}) if path.exists() else {}


def save_store(store: dict[str, list[dict[str, Any]]], path: Path) -> None:
    path.write_text(json.dumps({"classes": store}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def apply_query_crops(
    classes: dict[str, Any],
    verdicts: Sequence[dict[str, Any]],
    store: dict[str, list[dict[str, Any]]],
    *,
    reviewer: Optional[str] = None,
) -> tuple[list[str], list[str]]:
    """Record chosen crops in *store* (replacing a class's previous choice). Returns ``(changes, problems)``."""
    changes, problems = [], []
    for row in verdicts:
        class_id = row["class_id"]
        if class_id not in classes:
            problems.append(f"{class_id}: not in classes.json")
            continue
        cands = row.get("candidates", [])
        chosen, error = _parse(str(row.get("verdict", "")), len(cands))
        if error:
            problems.append(f"{class_id}: {error}")
            continue
        if chosen is None:
            continue
        store[class_id] = [
            {"page_id": cands[i]["page_id"], "box": list(cands[i]["box"]), "reviewed_by": reviewer} for i in chosen
        ]
        changes.append(f"{class_id}: {len(chosen)} extra query crop(s)")
    return changes, problems


def materialise(
    classes: dict[str, Any],
    store: dict[str, list[dict[str, Any]]],
    pages: dict[str, Page],
    out_dir: Path,
    warnings: Optional[list[str]] = None,
) -> int:
    """Cut every stored crop to ``<out_dir>/<class>__q<n>.png`` and set ``query_crops``.

    Called by the apply step and by ``build_corpus.py`` after the primary crops,
    so a rebuild reproduces the same list.  A stored crop whose page is not in
    this build, or whose box no longer lies on the page, is skipped with a
    warning rather than failing the build.
    """
    from PIL import Image  # noqa: PLC0415

    from sources import _common  # noqa: PLC0415

    written = 0
    for class_id, meta in classes.items():
        if not meta.get("query_crop"):
            continue
        paths = [meta["query_crop"]]
        for n, entry in enumerate(store.get(class_id, []), start=1):
            page = pages.get(entry["page_id"])
            x, y, w, h = entry["box"]
            if page is None or x + w > page.width or y + h > page.height:
                if warnings is not None:
                    warnings.append(f"{class_id}: stored query crop on {entry['page_id']} does not fit this build")
                continue
            dest = out_dir / f"{class_id.replace('/', '__')}__q{n}.png"
            with Image.open(page.path) as im:
                _common.save_verified(im.convert("RGB").crop((x, y, x + w, y + h)), dest)
            paths.append(str(dest))
            written += 1
        meta["query_crops"] = paths
    return written


# ---------------------------------------------------------------------------
# The sheet (GRID side)
# ---------------------------------------------------------------------------


def render(
    class_id: str, meta: dict[str, Any], cands: Sequence[CropCandidate], pages: dict[str, Page], out: Path
) -> Path:
    """One screen: the current query crop, then up to TOP numbered candidates in context."""
    from PIL import Image, ImageDraw  # noqa: PLC0415

    from completeness import COLS, _font  # noqa: PLC0415

    thumb, pad, cap = 250, 10, 22
    big, small, title = _font(40, bold=True), _font(15), _font(18, bold=True)
    width = COLS * (thumb + pad) + pad
    rows = 1 + (len(cands) + COLS - 1) // COLS
    sheet = Image.new("RGB", (width, 44 + rows * (thumb + cap + pad)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (pad, 10),
        f"{class_id}   which are good EXTRA query crops?  (whole mark, tight red box, clean scan)",
        fill="black",
        font=title,
    )
    with Image.open(meta["query_crop"]) as im:
        q = im.convert("RGB").copy()
    q.thumbnail((thumb, thumb))
    sheet.paste(q, (pad, 44 + cap))
    draw.text((pad, 44), "CURRENT QUERY", fill="#b00000", font=small)
    for k, cand in enumerate(cands):
        page = pages[cand.page_id]
        x, y, w, h = cand.box
        margin = int(0.3 * max(w, h))
        with Image.open(page.path) as im:
            im = im.convert("RGB")
            left, top = max(0, x - margin), max(0, y - margin)
            crop = im.crop((left, top, min(im.width, x + w + margin), min(im.height, y + h + margin)))
        d = ImageDraw.Draw(crop)
        d.rectangle([x - left, y - top, x - left + w, y - top + h], outline="#d62728", width=max(2, crop.width // 100))
        r, c = divmod(k, COLS)
        cx, cy = pad + c * (thumb + pad), 44 + (r + 1) * (thumb + cap + pad)
        crop.thumbnail((thumb, thumb))
        sheet.paste(crop, (cx, cy + cap))
        draw.rectangle([cx, cy + cap, cx + thumb - 1, cy + cap + thumb - 1], outline="#cccccc")
        caption = f"{cand.inliers} inl · {w}x{h} · {cand.page_id.split('/')[-1].removeprefix('stampds-')}"
        while draw.textlength(caption, font=small) > thumb and len(caption) > 1:
            caption = caption[:-1]  # long Tobacco800 ids ran into the next column
        draw.text((cx, cy + 2), caption, fill="#333333", font=small)
        nb = draw.textbbox((0, 0), str(k), font=big)
        bw, bh = nb[2] - nb[0] + 16, nb[3] - nb[1] + 14
        draw.rectangle([cx, cy + cap, cx + bw, cy + cap + bh], fill="#111111")
        draw.text((cx + 8 - nb[0], cy + cap + 7 - nb[1]), str(k), fill="#ffffff", font=big)
    path = out / f"{class_id.replace('/', '__')}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse  # noqa: PLC0415

    import fullmarks_config as cfg  # noqa: PLC0415
    from sources._common import read_manifest  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--inliers", type=Path, required=True, help="tier-s inliers.json from eval_sift_rank.py")
    ap.add_argument("--top", type=int, default=TOP)
    args = ap.parse_args(argv)

    classes = json.loads((args.corpus / "classes.json").read_text(encoding="utf-8"))
    pages = {p.page_id: p for p in read_manifest(args.corpus / "corpus.jsonl")}
    inliers = json.loads(args.inliers.read_text(encoding="utf-8"))
    out = args.corpus / "audit" / "query_crops"
    rows = []
    for class_id, meta in sorted(classes.items()):
        if not meta.get("on_roster") or not meta.get("query_crop"):
            continue
        cands = candidates(class_id, meta, pages, inliers.get(class_id, {}), top=args.top)
        render(class_id, meta, cands, pages, out)
        rows.append(verdict_row(class_id, cands))
        print(f"  {class_id}: {len(cands)} candidate(s)", flush=True)
    (out / "verdicts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"\n{len(rows)} class(es) -> {out}; fill verdicts, then audit_to_corrections.py --task query_crops")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
