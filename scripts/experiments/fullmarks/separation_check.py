#!/usr/bin/env python
"""Re-cluster the real corpus and check that every separation holds two classes apart (#3837).

    python separation_check.py --out <dir>                    # cluster, write CSVs + figures
    python separation_check.py --out <dir> --plot-only        # figures from the CSVs already there
    python separation_check.py --out <dir> --skip-pair 'tobacco800/afm90c00-first_1#0,tobacco800/bad45f00#1'

A separation is a verdict that two *classes* are different marks, and it is
stored as one representative mark per side.  That is the whole constraint once
both classes have been through the membership pass, because a verified class is
a must-link star and forms one group before any distance does.  Before that it
holds the two named marks apart and nothing else: single linkage merges in
increasing distance order, so a cheaper crossing edge elsewhere fuses the
classes first.  ``apply_confusable`` now also records an unverified class's
members on the row, and ``single_linkage`` keeps every one of them on its side.

This script measures both representations on the real marks and the real
thresholds, with the real descriptor, and forgets every identity first, as a
rebuild does.  It runs three arms per source:

``stored``
    ``adjudications.json`` as it is.
``before_named``
    the store as it stood before the membership pass. Every row whose note
    starts ``membership:`` or ``completeness:`` is dropped (the stars, the
    rejections, and the completeness pass that came after them), and every
    class-level separation names only its two marks, as rows did before #3837.
``before_members``
    the same rows dropped, but every class-level separation carries both
    classes' members, as ``apply_confusable`` now writes them.  The members
    are today's memberships, which is the nearest stand-in on disk for the
    class as it stood when it was ruled on.

A class-level separation is a row naming both classes (``left_class_id`` and
``right_class_id``), which is what the confusable and merge passes write.  It
is **breached** at a threshold when some group of the re-clustered partition
holds a mark of each of its two classes, with classes taken from today's
manifest.  The named marks can stay apart while the classes fuse, and that
case is the one #3837 is about.

Each threshold is the source's own (``cfg.cluster_threshold_for``) times a
scale.  At scale 1 nothing has moved.  The larger scales are the case that
matters: a loosened threshold, a different descriptor, or new pages that
bridge two classes.  Those are the rebuilds a separation exists to survive.

Writes ``separations.csv`` (one row per source, arm and scale), ``breaches.csv``
(one row per breached class pair, with the edge that joined them), ``stores/``
(the store each arm replayed), ``figures/breaches.png`` and
``figures/bridges.png``.  Descriptors are cached under ``--cache``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import fullmarks_config as cfg  # noqa: E402
from cluster_marks import (  # noqa: E402
    MarkRef,
    crop_mark,
    describe_marks,
    distance_matrix,
    load_adjudications,
    resolve_pairs,
    resolve_separations,
    single_linkage,
)
from sources._common import Page, read_manifest  # noqa: E402

ARMS = ("stored", "before_named", "before_members")
DEFAULT_SCALES = (1.0, 1.25, 1.5, 2.0, 3.0)
#: Rows written by the membership and completeness passes, which come after
#: the merge slate and so are absent from a store as it stood before membership.
LATER_PASSES = ("membership:", "completeness:")
KINDS = ("logo", "stamp")


def mark_key(text: str) -> tuple[str, int]:
    """``'spods/00001#1'`` -> ``('spods/00001', 1)``."""
    page_id, _, index = text.rpartition("#")
    return page_id, int(index)


def anchor(row: dict[str, Any], side: str) -> tuple[str, int]:
    index = row.get(f"{side}_mark_index")
    return row[f"{side}_page_id"], -1 if index is None else int(index)


def is_class_level(row: dict[str, Any]) -> bool:
    return bool(row.get("left_class_id") and row.get("right_class_id"))


def arm_store(
    arm: str,
    same: Sequence[dict[str, Any]],
    different: Sequence[dict[str, Any]],
    members_of: dict[str, list[list[Any]]],
    class_of: dict[tuple[str, int], str],
    skip: set[frozenset[tuple[str, int]]],
) -> dict[str, list[dict[str, Any]]]:
    """The adjudications one arm replays.  See the module docstring."""
    different = [r for r in different if frozenset((anchor(r, "left"), anchor(r, "right"))) not in skip]
    if arm == "stored":
        return {"same": list(same), "different": list(different)}

    def earlier(row: dict[str, Any]) -> bool:
        return not str(row.get("note") or "").startswith(LATER_PASSES)

    same = [r for r in same if earlier(r)]
    out: list[dict[str, Any]] = []
    for row in (r for r in different if earlier(r)):
        row = {k: v for k, v in row.items() if not k.endswith("_members")}
        if is_class_level(row) and arm == "before_members":
            for side in ("left", "right"):
                cid = class_of.get(anchor(row, side))
                if cid:
                    row[f"{side}_members"] = members_of[cid]
        out.append(row)
    return {"same": same, "different": out}


def refs_for(pages: Sequence[Page], source: str) -> list[MarkRef]:
    """Every logo and stamp of *source*, identity forgotten, as a rebuild sees them."""
    return [
        MarkRef(pi, mi, page.page_id, mark.kind, mark.box)
        for pi, page in enumerate(pages)
        if page.source == source
        for mi, mark in enumerate(page.marks)
        if mark.kind in KINDS
    ]


def cached_distances(pages: Sequence[Page], refs: Sequence[MarkRef], source: str, cache: Path) -> np.ndarray:
    path = cache / f"{source}-phash-{len(refs)}.npz"
    keys = np.array([f"{r.page_id}#{r.mark_index}:{','.join(map(str, r.box))}" for r in refs])
    if path.exists():
        hit = np.load(path)
        if hit["keys"].shape == keys.shape and bool((hit["keys"] == keys).all()):
            return hit["dist"]
    desc = describe_marks(pages, refs, backend="phash")
    dist = distance_matrix(desc, refs, backend="phash")
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, keys=keys, dist=dist)
    return dist


def group_edges(group: Sequence[int], dist: np.ndarray, threshold: float) -> list[tuple[float, int, int]]:
    """A group's own edges under *threshold*, cheapest first."""
    idx = np.asarray(group)
    sub = dist[np.ix_(idx, idx)]
    a, b = np.triu_indices(len(idx), 1)
    keep = sub[a, b] <= threshold
    a, b, d = idx[a[keep]], idx[b[keep]], sub[a[keep], b[keep]]
    order = np.lexsort((b, a, d))
    return [(float(d[k]), int(a[k]), int(b[k])) for k in order]


def bridge(edges: Sequence[tuple[float, int, int]], left: set[int], right: set[int]) -> tuple[int, int, float]:
    """The edge at which *left* and *right* first became connected inside one group.

    Replays plain single linkage over the group's own *edges*, so it names the
    crossing a reviewer would look at: the two crops whose closeness put two
    ruled-different classes into one group.  It can be an unclassed mark's
    edge, in which case the chain runs through that mark.
    """
    parent: dict[int, int] = {}
    has_left: dict[int, bool] = {}
    has_right: dict[int, bool] = {}

    def find(i: int) -> int:
        if i not in parent:
            parent[i], has_left[i], has_right[i] = i, i in left, i in right
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for d, a, b in edges:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        parent[rb] = ra
        has_left[ra] = has_left[ra] or has_left[rb]
        has_right[ra] = has_right[ra] or has_right[rb]
        if has_left[ra] and has_right[ra]:
            return a, b, d
    return -1, -1, float("nan")


def check(args: argparse.Namespace) -> None:
    corpus: Path = args.corpus
    out: Path = args.out
    pages = list(read_manifest(corpus / "corpus.jsonl"))
    raw = json.loads((args.adjudications or corpus / "adjudications.json").read_text(encoding="utf-8"))
    same_rows, diff_rows = raw.get("same", []), raw.get("different", [])

    class_of: dict[tuple[str, int], str] = {}
    members_of: dict[str, list[list[Any]]] = {}
    for page in pages:
        for index, mark in enumerate(page.marks):
            if mark.class_id:
                class_of[(page.page_id, index)] = mark.class_id
                members_of.setdefault(mark.class_id, []).append([page.page_id, index])

    skip = {frozenset(mark_key(m) for m in pair.split(",")) for pair in args.skip_pair}
    stores = out / "stores"
    stores.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        payload = arm_store(arm, same_rows, diff_rows, members_of, class_of, skip)
        (stores / f"{arm}.json").write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    # The class pairs a verdict ruled different, by today's classes.
    ruled: set[tuple[str, str]] = set()
    for row in diff_rows:
        if not is_class_level(row) or frozenset((anchor(row, "left"), anchor(row, "right"))) in skip:
            continue
        a, b = class_of.get(anchor(row, "left")), class_of.get(anchor(row, "right"))
        if a and b and a != b:
            ruled.add((min(a, b), max(a, b)))

    summary_rows: list[dict[str, Any]] = []
    breach_rows: list[dict[str, Any]] = []
    for source in args.sources:
        refs = refs_for(pages, source)
        if not refs:
            continue
        dist = cached_distances(pages, refs, source, args.cache)
        row_class = [class_of.get((r.page_id, r.mark_index)) for r in refs]
        rows_of: dict[str, set[int]] = {}
        for i, cid in enumerate(row_class):
            if cid:
                rows_of.setdefault(cid, set()).add(i)
        pairs_here = sorted((a, b) for a, b in ruled if a in rows_of and b in rows_of)
        base = cfg.cluster_threshold_for(source)
        print(f"{source}: {len(refs)} marks, {len(pairs_here)} class pair(s) ruled different, threshold {base}")

        for arm in ARMS:
            same, different = load_adjudications(stores / f"{arm}.json")
            try:
                must = resolve_pairs(refs, same)
                seps = resolve_separations(refs, different)
            except ValueError as exc:
                must, seps, refused = [], [], str(exc)
            else:
                refused = ""
            for scale in args.scales:
                threshold = round(base * scale, 4)
                record: dict[str, Any] = {
                    "source": source,
                    "arm": arm,
                    "scale": scale,
                    "threshold": threshold,
                    "marks": len(refs),
                    "class_pairs": len(pairs_here),
                    "separations": len(seps),
                    "on_members": sum(1 for s in seps if s.left_members or s.right_members),
                    "refused": refused,
                }
                if not refused:
                    try:
                        labels = single_linkage(dist, threshold, must_link=must, separations=seps)
                    except ValueError as exc:
                        record["refused"] = str(exc)
                if record["refused"]:
                    summary_rows.append(record)
                    print(f"  {arm:>14} x{scale:<4} REFUSED: {record['refused'][:120]}")
                    continue
                groups: dict[int, list[int]] = {}
                for i, label in enumerate(labels):
                    groups.setdefault(label, []).append(i)
                breached = 0
                edges_of: dict[int, list[tuple[float, int, int]]] = {}
                for a, b in pairs_here:
                    together = {labels[i] for i in rows_of[a]} & {labels[i] for i in rows_of[b]}
                    if not together:
                        continue
                    breached += 1
                    label = min(together, key=lambda g: (-len(groups[g]), g))
                    if label not in edges_of:
                        edges_of[label] = group_edges(groups[label], dist, threshold)
                    i, j, d = bridge(edges_of[label], rows_of[a], rows_of[b])
                    breach_rows.append(
                        {
                            "source": source,
                            "arm": arm,
                            "scale": scale,
                            "threshold": threshold,
                            "left_class": a,
                            "right_class": b,
                            "group_size": len(groups[label]),
                            "left_in_group": sum(1 for r in rows_of[a] if labels[r] == label),
                            "left_size": len(rows_of[a]),
                            "right_in_group": sum(1 for r in rows_of[b] if labels[r] == label),
                            "right_size": len(rows_of[b]),
                            "bridge_left": f"{refs[i].page_id}#{refs[i].mark_index}" if i >= 0 else "",
                            "bridge_left_class": (row_class[i] or "") if i >= 0 else "",
                            "bridge_right": f"{refs[j].page_id}#{refs[j].mark_index}" if j >= 0 else "",
                            "bridge_right_class": (row_class[j] or "") if j >= 0 else "",
                            "bridge_distance": round(d, 4),
                        }
                    )
                sizes = sorted((len(g) for g in groups.values()), reverse=True)
                record.update(
                    {
                        "groups": len(groups),
                        "largest": sizes[0],
                        "largest_share": round(sizes[0] / len(refs), 4),
                        "breached": breached,
                    }
                )
                summary_rows.append(record)
                print(
                    f"  {arm:>14} x{scale:<4} t={threshold:<6} groups {len(groups):>5}  largest {sizes[0]:>5}  "
                    f"breached {breached}/{len(pairs_here)}"
                )

    write_csv(out / "separations.csv", summary_rows)
    write_csv(out / "breaches.csv", breach_rows)
    print(f"wrote {out / 'separations.csv'} ({len(summary_rows)} rows) and {out / 'breaches.csv'} ({len(breach_rows)})")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def plot(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_csv(args.out / "separations.csv")
    figures = args.out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    sources = list(dict.fromkeys(r["source"] for r in rows))
    style = {
        "stored": ("#555555", "-", "stored store"),
        "before_named": ("#c0392b", "--", "before membership, two named marks (pre-#3837)"),
        "before_members": ("#2471a3", "-", "before membership, members recorded (#3837)"),
    }
    fig, axes = plt.subplots(1, len(sources), figsize=(4.2 * len(sources), 3.4), squeeze=False)
    for ax, source in zip(axes[0], sources):
        here = [r for r in rows if r["source"] == source]
        pairs = int(here[0]["class_pairs"]) if here else 0
        for arm in ARMS:
            pts = [(float(r["scale"]), int(r["breached"])) for r in here if r["arm"] == arm and not r["refused"]]
            if not pts:
                continue
            color, line, label = style[arm]
            ax.plot(*zip(*pts), line, color=color, marker="o", ms=3.5, label=label)
        ax.set_title(f"{source} ({pairs} class pairs ruled different)", fontsize=9)
        ax.set_xlabel("threshold / source threshold")
        ax.set_ylabel("class pairs fused")
        ax.set_ylim(bottom=-0.5)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(figures / "breaches.png", dpi=130)
    plt.close(fig)
    print(f"wrote {figures / 'breaches.png'}")


def contact_sheet(args: argparse.Namespace, limit: int = 8) -> None:
    """The crops that joined two ruled-different classes when only two marks were named.

    One row per distinct bridging edge, at the smallest scale it appears, with
    the class pair it fused in the header.  A bridge's own crops can belong to
    a third class: single linkage chains, and the chain is what the row shows.
    Each crop is framed, so a box with no ink in it still reads as a box.
    """
    from PIL import Image, ImageDraw

    rows = [r for r in read_csv(args.out / "breaches.csv") if r["arm"] == "before_named" and r["bridge_left"]]
    if not rows:
        return
    first: dict[tuple[str, str], dict[str, str]] = {}
    for row in sorted(rows, key=lambda r: (float(r["scale"]), r["source"], r["left_class"], r["right_class"])):
        first.setdefault((row["bridge_left"], row["bridge_right"]), row)
    picked = list(first.values())[:limit]
    pages = {p.page_id: p for p in read_manifest(args.corpus / "corpus.jsonl")}
    tile, head, gap = 160, 46, 14
    width = 2 * tile + 30
    sheet = Image.new("RGB", (width, (tile + head + gap) * len(picked) + 6), "white")
    draw = ImageDraw.Draw(sheet)
    for n, row in enumerate(picked):
        y = 4 + n * (tile + head + gap)
        fused = f"{row['left_class'].split('/')[-1]}  /  {row['right_class'].split('/')[-1]}"
        draw.text(
            (6, y), f"x{float(row['scale']):g}  d={float(row['bridge_distance']):.2f}  fused: {fused}", fill="#c0392b"
        )
        for k, key in enumerate(("bridge_left", "bridge_right")):
            x = 6 + k * (tile + 14)
            page_id, index = mark_key(row[key])
            page = pages[page_id]
            with Image.open(page.path) as img:
                crop = crop_mark(img.convert("RGB"), page.marks[index].box)
            crop.thumbnail((tile - 4, tile - 4))
            cls = (row[f"{key}_class"] or "unclassed").split("/")[-1]
            draw.text((x, y + 14), f"{row[key]}\n{cls}", fill="black")
            sheet.paste(crop, (x + 2, y + head + 2))
            draw.rectangle((x, y + head, x + crop.width + 3, y + head + crop.height + 3), outline="#999999")
        draw.line((0, y + tile + head + gap // 2, width, y + tile + head + gap // 2), fill="#dddddd")
    path = args.out / "figures" / "bridges.png"
    sheet.save(path)
    print(f"wrote {path} ({len(picked)} bridge(s))")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--adjudications", type=Path, default=None, help="default: <corpus>/adjudications.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=None, help="descriptor cache (default: <out>/cache)")
    ap.add_argument("--sources", default=",".join(cfg.CLUSTERED_SOURCES))
    ap.add_argument("--scales", default=",".join(str(s) for s in DEFAULT_SCALES))
    ap.add_argument(
        "--skip-pair",
        action="append",
        default=[],
        help="'page#idx,page#idx': leave this separation out of every arm (repeatable)",
    )
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args(argv)
    args.sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    args.scales = [float(s) for s in args.scales.split(",") if s.strip()]
    args.cache = args.cache or args.out / "cache"
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        check(args)
    plot(args)
    contact_sheet(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
