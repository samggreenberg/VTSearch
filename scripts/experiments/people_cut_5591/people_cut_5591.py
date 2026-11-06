#!/usr/bin/env python
"""Calibrate the People cut (#5591): how well the people partition recovers real identities.

The people partition (#5586, :mod:`vtscore.datasets.people`) merges a Face dataset's crops into people at the
face ball's beta 1 floor (0.4425 on buffalo_s), with centroid linkage on the unit sphere and hard views
attached last. Neither choice was measured as a clustering cut. This study scores the partition against
known identities, and sweeps the floor, the linkage and the hard-view rule around the shipped cut.

Every arm calls :func:`vtscore.datasets.people.fit_people` itself, so what is measured is what ships. The one
arm the module has no switch for, average linkage, swaps the module's own ``_centroid_linkage`` for
:func:`average_linkage` for the length of the fit (graph, re-assignment, hard-view attachment and ordering stay
the shipped code). The "alike" hard-view arm passes ``hard=None``, so a hard view is any face.

Subcommands:

``dump-fhibe``
    A FHIBE face cell (``fhibe_faces_ik_<size>__face.pkl``, buffalo_s) to an ``.npz`` of vectors, subject,
    photo, hard-view flag and Face Quality. FHIBE photos hold one subject each, so a face's identity is its
    photo's subject. Faces the provider found beyond that one (no subject) are dropped and counted.
``dump-demo``
    A demo loaded through the app's own ``image2face`` path (:func:`~vtscore.datasets.loader_demo.load_demo_dataset`,
    no cache) to the same ``.npz``. For ``vggface2_faces_s`` the largest face per photo carries the photo's
    label and the others are dropped and counted. For ``celebrity_together_s`` every face is kept, labelled
    with the person its photo was picked for, and the label is never used to score.
``sweep``
    Fits every arm (floor x linkage x hard rule) on each dump and scores it: pairwise and B-cubed precision,
    recall and F1, the split and merged shares, with person-bootstrap CIs on the B-cubed F1 gap to the
    shipped cut. Writes ``arms.csv`` and ``summary.md``.
``cic``
    Celebrities in Context: for each named person, the partition person in the most of their photos, and the
    share of their photos it appears in. Writes ``cic.csv`` and, with ``--sheets``, contact sheets (owner-only:
    the photos are agency photos with no licence).
``figure``
    The report's figure from ``arms.csv``.

Usage::

    python people_cut_5591.py dump-fhibe --cell <cells>/fhibe_faces_ik_1024__face.pkl --identities <k4>/identities.txt --out <dir>/fhibe_1024.npz
    python people_cut_5591.py dump-demo --demo vggface2_faces_s --out <dir>/vggface2_s.npz
    python people_cut_5591.py sweep --dump <dir>/fhibe_1024.npz [--dump ...] --out <dir>/sweep
    python people_cut_5591.py cic --dump <dir>/cic_s.npz --out <dir>/cic [--sheets]
    python people_cut_5591.py figure --arms <dir>/sweep/arms.csv --out docs/experiments/<study>/cut.webp
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: The floors swept: the issue's window (0.36 to 0.52) widened to 0.20 to 0.60, so average linkage, whose
#: cosine is a mean over pairs rather than between unit centroids, has room to find its own best cut.
FLOORS = tuple(round(x, 2) for x in np.arange(0.20, 0.6001, 0.02))
LINKAGES = ("centroid", "average")
HARD_RULES = ("last", "alike")
N_BOOT = 1000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- dumps


def _save(out: Path, **arrays) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    log(f"wrote {out} ({len(arrays['ids'])} faces)")


def _signals(media: dict) -> tuple[bool, float]:
    from vtscore.media.face.imagekit_bridge import QUALITY_METADATA_KEY, media_hard_view  # noqa: PLC0415

    q = (media.get("custom_metadata") or {}).get(QUALITY_METADATA_KEY)
    return bool(media_hard_view(media)), float(q) if isinstance(q, (int, float)) else math.nan


def _compat(medias: dict) -> str:
    from vtscore.media.face.imagekit_bridge import COMPAT_METADATA_KEY  # noqa: PLC0415

    tags = Counter((m.get("custom_metadata") or {}).get(COMPAT_METADATA_KEY) for m in medias.values())
    if len(tags) != 1:
        raise SystemExit(f"faces carry {dict(tags)} compat tags, not one")
    return str(next(iter(tags)))


def dump_fhibe(cell: Path, identities: Path, out: Path) -> None:
    sys.path.insert(0, str(REPO / "scripts" / "experiments" / "calibration"))
    from _cells_io import load_medias  # noqa: PLC0415

    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

    medias = load_medias(cell, repair=True)
    review = {ln.strip() for ln in identities.read_text().splitlines() if ln.strip() and not ln.startswith("#")}
    keep = [cid for cid in sorted(medias) if medias[cid].get("categories")]
    extra = len(medias) - len(keep)
    hard, quality = zip(*(_signals(medias[c]) for c in keep))
    labels = np.asarray([medias[c]["category"] for c in keep])
    _save(
        out,
        ids=np.asarray(keep, dtype=np.int64),
        vectors=np.stack([np.asarray(media_embedding(medias[c]), dtype=np.float32) for c in keep]),
        labels=labels,
        photos=np.asarray([int(medias[c]["source_photo_id"]) for c in keep], dtype=np.int64),
        hard=np.asarray(hard, dtype=bool),
        quality=np.asarray(quality, dtype=np.float64),
        review=np.isin(labels, sorted(review)),
        compat=_compat(medias),
        excluded=extra,
        source=str(cell),
    )
    log(f"{cell.name}: {len(keep)} subject faces, {extra} extra faces excluded, {len(set(labels))} subjects")


def dump_demo(demo: str, out: Path) -> None:
    from vtscore.datasets.co_occurrence import faces_by_photo  # noqa: PLC0415
    from vtscore.datasets.loader_demo import load_demo_dataset  # noqa: PLC0415
    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

    medias: dict[int, dict] = {}
    load_demo_dataset(demo, medias, converter_name="image2face", use_cache=False, on_progress=lambda *a, **k: None)
    by_photo = faces_by_photo(medias)
    photo_of = {f: p for p, faces in by_photo.items() for f in faces}
    keep_largest = demo.startswith("vggface2")
    keep: list[int] = []
    if keep_largest:
        for faces in by_photo.values():
            keep.append(max(faces, key=lambda f: (medias[f]["width"] * medias[f]["height"], -f)))
    else:
        keep = sorted(photo_of)
    keep.sort()
    hard, quality = zip(*(_signals(medias[c]) for c in keep))
    labels = np.asarray([str(medias[c].get("category") or "") for c in keep])
    _save(
        out,
        ids=np.asarray(keep, dtype=np.int64),
        vectors=np.stack([np.asarray(media_embedding(medias[c]), dtype=np.float32) for c in keep]),
        labels=labels,
        photos=np.asarray([photo_of[c] for c in keep], dtype=np.int64),
        hard=np.asarray(hard, dtype=bool),
        quality=np.asarray(quality, dtype=np.float64),
        review=np.ones(len(keep), dtype=bool),
        compat=_compat(medias),
        excluded=len(medias) - len(keep),
        source=demo,
        photo_paths=json.dumps({str(c): _photo_path(medias[c]) for c in keep}),
        boxes=np.asarray([medias[c].get("source_box") or [np.nan] * 4 for c in keep], dtype=np.float64),
    )
    log(f"{demo}: {len(medias)} faces in {len(by_photo)} photos, kept {len(keep)}, {len(set(labels))} labels")


def _photo_path(media: dict) -> str:
    from vtscore.datasets.co_occurrence import photo_keys  # noqa: PLC0415

    keys = photo_keys(media)
    return keys[0][1] if keys else ""


# ---------------------------------------------------------------- arms


def average_linkage(unit: np.ndarray, cutoff: float) -> list[list[int]]:
    """Average linkage over one component: merge while the mean pairwise cosine between clusters clears *cutoff*.

    The drop-in for :func:`vtscore.datasets.people._centroid_linkage` (same arguments, same return shape).
    """
    from scipy.cluster.hierarchy import fcluster, linkage  # noqa: PLC0415
    from scipy.spatial.distance import pdist  # noqa: PLC0415

    m = unit.shape[0]
    if m < 2:
        return [[i] for i in range(m)]
    dist = np.clip(pdist(unit, "cosine"), 0.0, 2.0)
    labels = fcluster(linkage(dist, "average"), t=1.0 - cutoff, criterion="distance")
    groups: dict[int, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        groups[int(lab)].append(i)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])


@contextlib.contextmanager
def _linkage(name: str):
    from vtscore.datasets import people  # noqa: PLC0415

    if name == "centroid":
        yield
        return
    shipped = people._centroid_linkage
    people._centroid_linkage = average_linkage
    try:
        yield
    finally:
        people._centroid_linkage = shipped


def fit_arm(d: dict, floor: float, linkage: str, hard_rule: str) -> np.ndarray:
    """One arm's partition of the dump *d*: a person index per row (an unplaced face gets its own)."""
    from vtscore.datasets.people import fit_people  # noqa: PLC0415

    rows = np.arange(len(d["ids"]))
    with _linkage(linkage):
        part = fit_people(
            d["vectors"],
            list(rows),
            cutoff=floor,
            compat=str(d["compat"]),
            hard=d["hard"] if hard_rule == "last" else None,
            quality=list(d["quality"]),
        )
    assign = np.full(rows.size, -1, dtype=np.int64)
    for p, person in enumerate(part.people):
        assign[list(person)] = p
    loose = np.flatnonzero(assign < 0)
    assign[loose] = len(part.people) + np.arange(loose.size)
    return assign


# ---------------------------------------------------------------- scores


def _pairs(n: np.ndarray) -> float:
    return float(np.sum(n * (n - 1) / 2))


def score(assign: np.ndarray, labels: np.ndarray) -> dict:
    """Pairwise and B-cubed P/R/F1, the split and merged shares, and the per-face B-cubed values."""
    lab_codes, lab = np.unique(labels, return_inverse=True)
    _, cl = np.unique(assign, return_inverse=True)
    joint = Counter(zip(cl.tolist(), lab.tolist()))
    n_cl = np.bincount(cl)
    n_lab = np.bincount(lab)
    n_joint = np.asarray([joint[(c, b)] for c, b in zip(cl, lab)])
    b3p_face = n_joint / n_cl[cl]
    b3r_face = n_joint / n_lab[lab]
    tp = _pairs(np.asarray(list(joint.values())))
    pred, true = _pairs(n_cl), _pairs(n_lab)
    pp = tp / pred if pred else 1.0
    pr = tp / true if true else 1.0
    ids_per_cl = Counter(c for c, _ in joint)
    cls_per_id = Counter(b for _, b in joint)
    multi = np.asarray([ids_per_cl[c] > 1 for c in range(n_cl.size)])
    big = n_cl >= 2
    largest = defaultdict(int)
    for (c, b), n in joint.items():
        largest[b] = max(largest[b], n)
    out = {
        "n_faces": int(labels.size),
        "n_identities": int(lab_codes.size),
        "n_people": int(n_cl.size),
        "n_singletons": int(np.sum(n_cl == 1)),
        "pair_p": pp,
        "pair_r": pr,
        "pair_f": 2 * pp * pr / (pp + pr) if pp + pr else 0.0,
        "b3_p": float(b3p_face.mean()),
        "b3_r": float(b3r_face.mean()),
        # identities spread over more than one partition person (an unplaced face counts as its own)
        "split_share": float(np.mean([cls_per_id[b] > 1 for b in range(lab_codes.size)])),
        "mean_parts": float(np.mean([cls_per_id[b] for b in range(lab_codes.size)])),
        # the share of an identity's faces outside its largest part, averaged over identities
        "stray_share": float(np.mean([1 - largest[b] / n_lab[b] for b in range(lab_codes.size)])),
        # partition people (2+ faces) holding more than one identity, and the faces that sit in one
        "merged_share": float(multi[big].mean()) if big.any() else 0.0,
        "merged_face_share": float(multi[cl].mean()),
        # identities with a face in a partition person that holds another identity too
        "merged_identity_share": len({b for c, b in joint if multi[c]}) / lab_codes.size,
    }
    out["b3_f"] = 2 * out["b3_p"] * out["b3_r"] / (out["b3_p"] + out["b3_r"])
    out["_b3"] = (b3p_face, b3r_face, lab)
    return out


def _boot_b3f(b3: tuple, base: tuple, rng: np.random.Generator) -> tuple[float, float]:
    """95% CI of the B-cubed F1 gap (arm - *base*), resampling identities with their faces."""
    p, r, lab = b3
    p0, r0, lab0 = base
    assert np.array_equal(lab, lab0)
    n_id = int(lab.max()) + 1
    order = np.argsort(lab, kind="stable")
    bounds = np.searchsorted(lab[order], np.arange(n_id + 1))
    sums = np.stack([np.add.reduceat(v[order], bounds[:-1]) for v in (p, r, p0, r0)])  # per identity
    cnt = np.diff(bounds).astype(np.float64)
    gaps = np.empty(N_BOOT)
    for i in range(N_BOOT):
        w = np.bincount(rng.integers(0, n_id, n_id), minlength=n_id).astype(np.float64)
        tot = (w * cnt).sum()
        mp, mr, mp0, mr0 = (sums * w).sum(axis=1) / tot
        gaps[i] = 2 * mp * mr / (mp + mr) - 2 * mp0 * mr0 / (mp0 + mr0)
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return float(lo), float(hi)


def load_dump(path: Path, population: str) -> dict:
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    keep = np.ones(len(d["ids"]), dtype=bool) if population == "all" else d["review"].astype(bool)
    keep &= d["labels"] != ""
    for k in ("ids", "vectors", "labels", "photos", "hard", "quality", "review"):
        d[k] = d[k][keep]
    return d


SHIPPED_FLOOR = 0.4425


def sweep(dumps: list[Path], out: Path, populations: list[str], linkages: list[str]) -> None:
    from vtscore.datasets.people import PEOPLE_LINKAGE, people_cutoff  # noqa: PLC0415

    shipped = people_cutoff("oss-arcface-mbf:1")
    assert shipped is not None and abs(shipped - SHIPPED_FLOOR) < 1e-9, shipped
    assert PEOPLE_LINKAGE == "centroid"
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    floors = sorted({*FLOORS, SHIPPED_FLOOR})
    rng = np.random.default_rng(5591)
    for path in dumps:
        for pop in populations:
            d = load_dump(path, pop)
            if not d["ids"].size:
                continue
            name = f"{path.stem}:{pop}"
            log(f"{name}: {d['ids'].size} faces, {len(set(d['labels']))} identities, {int(d['hard'].sum())} hard")
            base = None
            # the shipped arm goes first in every job: it is the bootstrap's base
            arms = [(SHIPPED_FLOOR, "centroid", "last")] + [
                (f, lk, h)
                for lk in linkages
                for h in HARD_RULES
                for f in floors
                if (f, lk, h) != (SHIPPED_FLOOR, "centroid", "last")
            ]
            for floor, lk, hr in arms:
                t0 = time.time()
                s = score(fit_arm(d, floor, lk, hr), d["labels"])
                b3 = s.pop("_b3")
                if base is None:
                    base, base_f = b3, s["b3_f"]
                lo, hi = (0.0, 0.0) if b3 is base else _boot_b3f(b3, base, rng)
                if lk not in linkages:
                    continue
                rows.append(
                    {
                        "dump": path.stem,
                        "population": pop,
                        "floor": floor,
                        "linkage": lk,
                        "hard": hr,
                        "shipped": (floor, lk, hr) == (SHIPPED_FLOOR, "centroid", "last"),
                        **s,
                        "d_b3_f": s["b3_f"] - base_f,
                        "d_b3_f_lo": lo,
                        "d_b3_f_hi": hi,
                        "seconds": round(time.time() - t0, 2),
                        "excluded_faces": int(d["excluded"]),
                    }
                )
                log(
                    f"  {floor:.4f} {lk:8s} {hr:5s} B3 F {s['b3_f']:.3f} split {s['split_share']:.3f} merged {s['merged_share']:.3f}"
                )
            with (out / "arms.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    write_summary(rows, out / "summary.md")


def summarize(arms: list[Path], out: Path) -> None:
    """Concatenate per-job ``arms.csv`` files into one, and write its summary."""
    rows: list[dict] = []
    for path in arms:
        with path.open() as fh:
            rows.extend(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            if k in ("dump", "population", "linkage", "hard"):
                continue
            r[k] = (v == "True") if k == "shipped" else float(v)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "arms.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_summary(rows, out / "summary.md")


def write_summary(rows: list[dict], path: Path) -> None:
    lines = []
    for key in dict.fromkeys((r["dump"], r["population"]) for r in rows):
        rs = [r for r in rows if (r["dump"], r["population"]) == key]
        ship = next((r for r in rs if r["shipped"]), None)
        if ship is None:  # a job of one other linkage: its shipped base is in the centroid job's file
            continue
        lines += [
            f"## {key[0]} ({key[1]}): {ship['n_faces']} faces, {ship['n_identities']} identities",
            "",
            "| floor | linkage | hard | B3 P | B3 R | B3 F | dF (95% CI) | pair F | split | merged | merged faces |",
            "|---:|---|---|---:|---:|---:|---|---:|---:|---:|---:|",
        ]
        for r in sorted(rs, key=lambda r: (r["linkage"], r["hard"], r["floor"])):
            mark = " **(shipped)**" if r["shipped"] else ""
            lines.append(
                f"| {r['floor']:.4g} | {r['linkage']} | {r['hard']}{mark} | {r['b3_p']:.3f} | {r['b3_r']:.3f} | {r['b3_f']:.3f} | "
                f"{r['d_b3_f']:+.3f} ({r['d_b3_f_lo']:+.3f}, {r['d_b3_f_hi']:+.3f}) | {r['pair_f']:.3f} | "
                f"{r['split_share']:.3f} | {r['merged_share']:.3f} | {r['merged_face_share']:.3f} |"
            )
        better = [
            r for r in rs if r["d_b3_f_lo"] > 0 and r["merged_share"] <= ship["merged_share"] and not r["shipped"]
        ]
        lines += ["", f"Arms whose B3 F CI clears the shipped cut without raising the merged share: {len(better)}", ""]
        for r in sorted(better, key=lambda r: -r["d_b3_f"])[:10]:
            lines.append(
                f"- {r['floor']:.4g} {r['linkage']} {r['hard']}: {r['d_b3_f']:+.3f} merged {r['merged_share']:.3f}"
            )
        lines.append("")
    path.write_text("\n".join(lines))
    log(f"wrote {path}")


# ---------------------------------------------------------------- Celebrities in Context


def cic(dump: Path, out: Path, sheets: bool, photo_root: Path | None = None) -> None:
    d = load_dump(dump, "all")
    out.mkdir(parents=True, exist_ok=True)
    assign = fit_arm(d, SHIPPED_FLOOR, "centroid", "last")
    photos_of_name: dict[str, set[int]] = defaultdict(set)
    for lab, ph in zip(d["labels"], d["photos"]):
        photos_of_name[lab].add(int(ph))
    people_in_photo: dict[int, set[int]] = defaultdict(set)
    for p, ph in zip(assign, d["photos"]):
        people_in_photo[int(ph)].add(int(p))
    size = Counter(assign.tolist())
    rows = []
    for name in sorted(photos_of_name):
        photos = photos_of_name[name]
        count = Counter(p for ph in photos for p in people_in_photo[ph])
        top, n = max(count.items(), key=lambda kv: (kv[1], -kv[0]))
        rows.append(
            {
                "name": name,
                "photos": len(photos),
                "top_person_photos": n,
                "share": n / len(photos),
                "top_person_faces": size[top],
                "top_person": top,
                "runner_up_photos": sorted(count.values())[-2] if len(count) > 1 else 0,
            }
        )
    with (out / "cic.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    shares = np.asarray([r["share"] for r in rows])
    summary = {
        "faces": int(d["ids"].size),
        "photos": int(len(set(d["photos"].tolist()))),
        "names": len(rows),
        "people": int(len(size)),
        "people_2plus": int(sum(1 for v in size.values() if v >= 2)),
        "mean_share": float(shares.mean()),
        "median_share": float(np.median(shares)),
        "names_majority": int(np.sum(shares > 0.5)),
        "names_80": int(np.sum(shares >= 0.8)),
        "excluded_faces": int(d["excluded"]),
    }
    (out / "cic_summary.json").write_text(json.dumps(summary, indent=2))
    log(json.dumps(summary))
    if sheets:
        _sheets(d, assign, rows, out / "sheets", photo_root or Path("."))


def _sheets(d: dict, assign: np.ndarray, rows: list[dict], out: Path, photo_root: Path) -> None:
    """For four names (worst, lower quartile, median and best share), the faces of their top partition person.

    One sheet per name: each face cut from its photo by its ``source_box``, up to 60, in row order. Owner-only:
    the photos are agency photos with no licence.
    """
    from PIL import Image  # noqa: PLC0415

    out.mkdir(parents=True, exist_ok=True)
    paths = json.loads(str(d["photo_paths"]))
    picks = sorted(rows, key=lambda r: (r["share"], r["name"]))
    chosen = {"worst": picks[0], "q1": picks[len(picks) // 4], "median": picks[len(picks) // 2], "best": picks[-1]}
    tile, cols = 96, 10
    # the photo key is scan-relative ("Name/file.jpg"); the files sit in id folders, so find them by name
    by_name = {f.name: f for f in photo_root.rglob("*") if f.is_file()}
    for tag, r in chosen.items():
        rows_ = np.flatnonzero(assign == r["top_person"])[:60]
        sheet = Image.new("RGB", (cols * tile, max(1, math.ceil(rows_.size / cols)) * tile), "white")
        for k, row in enumerate(rows_):
            path = paths.get(str(int(d["ids"][row])), "")
            box = d["boxes"][row]
            if not path or not np.isfinite(box).all():
                continue
            with Image.open(by_name[Path(path).name]) as im:
                w, h = im.size
                crop = im.convert("RGB").crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)))
            crop.thumbnail((tile, tile))
            sheet.paste(crop, ((k % cols) * tile, (k // cols) * tile))
        name = f"{tag}-{r['name']}".replace("/", "_").replace(" ", "_")
        sheet.save(out / f"{name}.webp", quality=80)
        log(f"sheet {name}: share {r['share']:.2f}, {rows_.size} faces")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("dump-fhibe")
    a.add_argument("--cell", type=Path, required=True)
    a.add_argument("--identities", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a = sub.add_parser("dump-demo")
    a.add_argument("--demo", required=True)
    a.add_argument("--out", type=Path, required=True)
    a = sub.add_parser("sweep")
    a.add_argument("--dump", type=Path, action="append", required=True)
    a.add_argument("--population", action="append", choices=("review", "all"))
    a.add_argument("--linkage", action="append", choices=LINKAGES)
    a.add_argument("--out", type=Path, required=True)
    a = sub.add_parser("summarize")
    a.add_argument("--arms", type=Path, action="append", required=True)
    a.add_argument("--out", type=Path, required=True)
    a = sub.add_parser("cic")
    a.add_argument("--dump", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--sheets", action="store_true")
    a.add_argument("--photo-root", type=Path, help="the folder the photo paths are relative to")
    a = sub.add_parser("figure")
    a.add_argument("--arms", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "dump-fhibe":
        dump_fhibe(args.cell, args.identities, args.out)
    elif args.cmd == "dump-demo":
        dump_demo(args.demo, args.out)
    elif args.cmd == "sweep":
        sweep(args.dump, args.out, args.population or ["review"], args.linkage or list(LINKAGES))
    elif args.cmd == "summarize":
        summarize(args.arms, args.out)
    elif args.cmd == "cic":
        cic(args.dump, args.out, args.sheets, args.photo_root)
    elif args.cmd == "figure":
        figure(args.arms, args.out)


def figure(arms: Path, out: Path) -> None:
    """B-cubed F1 and the merged share against the floor, per linkage and hard rule, for the review population."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    with arms.open() as fh:
        rows = [r for r in csv.DictReader(fh)]
    dumps = [k for k in dict.fromkeys(r["dump"] for r in rows) if k.startswith("fhibe")] + [
        k for k in dict.fromkeys(r["dump"] for r in rows) if not k.startswith("fhibe")
    ]
    fig, axes = plt.subplots(2, len(dumps), figsize=(4.2 * len(dumps), 6.4), sharex=True, squeeze=False)
    styles = {
        ("centroid", "last"): ("#2a6fdb", "-"),
        ("centroid", "alike"): ("#2a6fdb", ":"),
        ("average", "last"): ("#d9822b", "-"),
        ("average", "alike"): ("#d9822b", ":"),
    }
    for c, dump in enumerate(dumps):
        pop = "review" if any(r["dump"] == dump and r["population"] == "review" for r in rows) else "all"
        rs = [r for r in rows if r["dump"] == dump and r["population"] == pop]
        for (lk, hr), (color, ls) in styles.items():
            pts = sorted(
                (float(r["floor"]), float(r["b3_f"]), float(r["merged_share"]), float(r["split_share"]))
                for r in rs
                if r["linkage"] == lk and r["hard"] == hr
            )
            if not pts:
                continue
            x, f, m, sp = map(np.asarray, zip(*pts))
            axes[0, c].plot(x, f, color=color, ls=ls, label=f"{lk}, hard {hr}")
            axes[1, c].plot(x, m, color=color, ls=ls)
            if hr == "last":
                axes[1, c].plot(x, sp, color=color, ls="--", alpha=0.5)
        for ax in axes[:, c]:
            ax.axvline(SHIPPED_FLOOR, color="0.5", lw=0.8)
        axes[0, c].set_title(f"{dump} ({pop})", fontsize=10)
        axes[1, c].set_xlabel("floor (cosine)")
    axes[0, 0].set_ylabel("B-cubed F1")
    axes[1, 0].set_ylabel("merged share (solid)\nsplit share (dashed)")
    axes[0, 0].legend(fontsize=8, loc="lower center")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
