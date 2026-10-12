#!/usr/bin/env python3
"""Does FHIBE supply a face benchmark that tests *finding* the face, not just *matching* it?

The supply census for #4699, in the shape of #3983's ``coco_only_supply.py``.
Three stages, each writing into ``--out`` (keep it beside the FHIBE version it
came from: FHIBE's consent-revocation terms mean every derived table is deleted
with the release it was derived from):

``annotations``
    One row per (image, consenting subject) from the JSON annotations: the
    subject, how many people the photo has, the frame, the annotated face box,
    the camera-distance label and the pile band that box falls in. Banding uses
    the SHIPPED rule, :func:`pilebuild.scale_core.band_for`, so a FHIBE "small"
    is a COCO Better "small" (``BOX_BANDS``: under 1/196 of the frame, under
    1/12, the rest).

``detect --shard i/n``
    Runs the app's own :class:`~vtscore.media.image.face_localizer.FaceLocalizer`
    (MTCNN, keep-all) on each image at full resolution and at each
    ``--long-side`` (the image downscaled so its long side is that many pixels,
    standing in for a benchmark that stores smaller copies than the 12 MB
    originals). Every detection is kept with its confidence: thresholds are
    applied in ``report``.

``report``
    Usable photos per subject after Sony's two-subject exclusion; the band x
    camera-distance supply; and MTCNN recall and IoU against the annotated face
    box, per band and per resolution.

What this deliberately does not touch: FHIBE's demographic annotations. The
census needs the face box, the camera distance and the subject id, and nothing
else is read into a table.

    python fhibe_supply.py annotations --root <dir holding filepaths.csv> --out <dir>
    python fhibe_supply.py detect --root ... --out ... --shard 0/8 [--long-side 1024 --long-side 640]
    python fhibe_supply.py report --out <dir>
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pile_config as pc  # noqa: E402
from pilebuild.scale_core import band_for  # noqa: E402

BANDS = ("small", "medium", "large")
#: The app's defaults: FaceLocalizer's confidence threshold, and image2face's
#: min_size after its 0.25 padding (a crop 1.5x the box must be >= 32 px).
APP_THRESHOLD = 0.5
APP_MIN_CROP = 32
APP_PADDING = 0.25
IOU_HIT = 0.5

ANN_COLS = (
    "uid",
    "img",
    "subject_id",
    "is_primary",
    "humans",
    "W",
    "H",
    "x1",
    "y1",
    "x2",
    "y2",
    "frac",
    "band",
    "camera_distance",
    "head_pose",
)


# ---------------------------------------------------------------------------
# annotations
# ---------------------------------------------------------------------------


def stage_annotations(root: Path, out: Path) -> None:
    rows = list(csv.DictReader((root / "filepaths.csv").open()))
    table = []
    for r in rows:
        d = json.loads((root / r["json"]).read_text())
        ia = d["image_annotation"]
        W, H = int(ia["image_width"]), int(ia["image_height"])
        for s in d["subject_annotation"]:
            x, y, w, h = (float(v) for v in s["face_bbox"])  # FHIBE stores [x, y, w, h]
            box = [x, y, x + w, y + h]
            table.append(
                {
                    "uid": r["uid"],
                    "img": r["img"],
                    "subject_id": s["subject_id"],
                    "is_primary": int(bool(s["is_primary"])),
                    "humans": int(ia["humans_per_image"]),
                    "W": W,
                    "H": H,
                    "x1": box[0],
                    "y1": box[1],
                    "x2": box[2],
                    "y2": box[3],
                    "frac": round(w * h / (W * H), 6),
                    "band": band_for([box], W, H),
                    "camera_distance": ia["camera_distance"],
                    "head_pose": s.get("head_pose", ""),
                }
            )
    out.mkdir(parents=True, exist_ok=True)
    with (out / "annotations.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, ANN_COLS)
        wr.writeheader()
        wr.writerows(table)
    print(f"{len(rows)} images, {len(table)} subject rows -> {out / 'annotations.csv'}")


def read_annotations(out: Path) -> list[dict]:
    rows = list(csv.DictReader((out / "annotations.csv").open()))
    for r in rows:
        for k in ("W", "H", "humans", "is_primary"):
            r[k] = int(r[k])
        for k in ("x1", "y1", "x2", "y2", "frac"):
            r[k] = float(r[k])
    return rows


# ---------------------------------------------------------------------------
# detect
# ---------------------------------------------------------------------------


def stage_detect(root: Path, out: Path, shard: str, long_sides: list[int]) -> None:
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    from vtscore.media.image.face_localizer import FaceLocalizer  # noqa: PLC0415

    i, n = (int(v) for v in shard.split("/"))
    images = sorted({r["img"] for r in read_annotations(out)})[i::n]
    dest = out / "detect" / f"shard-{i:03d}-of-{n:03d}.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if dest.exists():  # resume: a requeued shard skips what it already wrote
        done = {json.loads(line)["img"] for line in dest.open()}
    todo = [p for p in images if p not in done]
    print(f"shard {shard}: {len(images)} images, {len(done)} done, {len(todo)} to go", flush=True)

    loc = FaceLocalizer("fhibe-census", threshold=0.0)  # keep every detection; report thresholds
    loc.load_model()

    def variants(path: str) -> list[tuple[str, bytes, float]]:
        """``(resolution, bytes, scale back to full-res pixels)`` for each resolution measured."""
        data = (root / path).read_bytes()
        got = [("full", data, 1.0)]
        with Image.open(io.BytesIO(data)) as im:
            im = im.convert("RGB")
            for L in long_sides:
                s = L / max(im.size)
                if s >= 1:
                    got.append((str(L), data, 1.0))
                    continue
                small = im.resize((round(im.width * s), round(im.height * s)), Image.BICUBIC)
                buf = io.BytesIO()
                small.save(buf, "JPEG", quality=90)
                got.append((str(L), buf.getvalue(), s))
        return got

    def prefetched(paths: list[str], ahead: int = 8):
        """``(path, variants)`` in order, decoding at most *ahead* images ahead (a 12 MB PNG is ~100 MB as RGB)."""
        with ThreadPoolExecutor(4) as pool:
            for j in range(0, len(paths), ahead):
                batch = paths[j : j + ahead]
                yield from zip(batch, pool.map(variants, batch))

    t0, k = time.time(), 0
    with dest.open("a") as fh:
        for path, vs in prefetched(todo):
            rec = {"img": path, "det": {}}
            for res, data, s in vs:
                hits = loc.localize({"media_bytes": data})
                rec["det"][res] = [[round(c / s, 1) for c in h["bbox"]] + [h["confidence"]] for h in hits]
                rec.setdefault("scale", {})[res] = s
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            k += 1
            if k % 25 == 0:
                print(
                    f"{time.strftime('%H:%M:%S')} progress: {k}/{len(todo)} ({(time.time() - t0) / k:.2f} s/image)",
                    flush=True,
                )
    print(f"shard {shard} DONE: {k} images in {time.time() - t0:.0f} s", flush=True)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def app_keeps(det: list[float], scale: float) -> bool:
    """Would the app keep this detection? Confidence threshold, then image2face's min crop.

    The crop size is judged in the pixels the app sees (the stored resolution),
    which is the full-res box times *scale*.
    """
    if det[4] < APP_THRESHOLD:
        return False
    w, h = (det[2] - det[0]) * scale, (det[3] - det[1]) * scale
    return w * (1 + 2 * APP_PADDING) >= APP_MIN_CROP and h * (1 + 2 * APP_PADDING) >= APP_MIN_CROP


def pct(a: int, b: int) -> str:
    return f"{100 * a / b:.0f}%" if b else "-"


def stage_report(out: Path) -> None:
    ann = read_annotations(out)
    lines: list[str] = []
    say = lines.append

    # -- supply ------------------------------------------------------------
    single = [r for r in ann if r["humans"] == 1]
    two = {r["uid"] for r in ann if r["humans"] == 2}
    per_subj = collections.Counter(r["subject_id"] for r in single)
    all_subj = {r["subject_id"] for r in ann}
    say("## Supply")
    say(f"- images: {len({r['uid'] for r in ann})}; one-person {len(single)}, two-person {len(two)} (excluded)")
    say(f"- subjects: {len(all_subj)} in any photo; {len(per_subj)} with a one-person photo")
    hist = collections.Counter(per_subj.values())
    say("- one-person photos per subject: " + ", ".join(f"{k}: {hist[k]}" for k in sorted(hist)))
    q = [s for s, c in per_subj.items() if c >= 2]
    say(
        f"- subjects usable as a query (>= 2 photos, so >= 1 positive): {len(q)}, "
        f"median photos {statistics.median(per_subj[s] for s in q):.0f}"
    )
    by_cd = collections.Counter(r["camera_distance"] for r in single)
    say("")
    say("### One-person photos by band x camera distance")
    cds = sorted(by_cd)
    say("| band | " + " | ".join(cds) + " | all |")
    say("|---|" + "---|" * (len(cds) + 1))
    for b in (*BANDS, "scattered", "oversize"):
        row = [r for r in single if r["band"] == b]
        if not row and b not in BANDS:
            continue
        c = collections.Counter(r["camera_distance"] for r in row)
        say(f"| {b} | " + " | ".join(str(c[cd]) for cd in cds) + f" | {len(row)} |")
    fr = sorted(r["frac"] for r in single)
    say("")
    say(
        f"- face box as a fraction of the frame: p5 {fr[len(fr) // 20]:.4f}, median {fr[len(fr) // 2]:.4f}, "
        f"p95 {fr[19 * len(fr) // 20]:.4f}; small band is < {pc.BOX_BANDS['small'][1]:.4f}"
    )
    # Per-subject depth in the small band: a query needs its positives in the band.
    small_subj = collections.Counter(r["subject_id"] for r in single if r["band"] == "small")
    say(
        f"- subjects with >= 2 small-band photos: {sum(1 for c in small_subj.values() if c >= 2)}; "
        f">= 1: {len(small_subj)}"
    )

    # -- detection ---------------------------------------------------------
    det: dict[str, dict] = {}
    for f in sorted((out / "detect").glob("shard-*.jsonl")) if (out / "detect").exists() else []:
        for line in f.open():
            rec = json.loads(line)
            det[rec["img"]] = rec
    say("")
    say("## MTCNN against the annotated face box")
    if not det:
        say("_no detect shards yet_")
    else:
        rows = [r for r in single if r["img"] in det]
        say(
            f"- {len(rows)} of {len(single)} one-person photos detected so far. Hit = an app-kept detection "
            f"(confidence >= {APP_THRESHOLD}, padded crop >= {APP_MIN_CROP} px) with IoU >= {IOU_HIT}."
        )
        resolutions = list(next(iter(det.values()))["det"])
        for res in resolutions:
            say("")
            say(f"### Resolution: {res if res == 'full' else 'long side ' + res + ' px'}")
            say(
                "| band | photos | hit | top-1 hit | median best IoU | median face px (short side) | extra faces/photo |"
            )
            say("|---|---|---|---|---|---|---|")
            for b in (*BANDS, "all"):
                sub = [r for r in rows if b == "all" or r["band"] == b]
                if not sub:
                    continue
                hit = top1 = extra = 0
                ious, px = [], []
                for r in sub:
                    rec = det[r["img"]]
                    s = rec["scale"][res]
                    kept = [d for d in rec["det"][res] if app_keeps(d, s)]
                    gt = [r["x1"], r["y1"], r["x2"], r["y2"]]
                    best = max((iou(gt, d) for d in kept), default=0.0)
                    ious.append(best)
                    px.append(min(r["x2"] - r["x1"], r["y2"] - r["y1"]) * s)
                    hit += best >= IOU_HIT
                    top1 += bool(kept) and iou(gt, kept[0]) >= IOU_HIT  # face index 0 in image2face
                    extra += sum(1 for d in kept if iou(gt, d) < IOU_HIT)
                say(
                    f"| {b} | {len(sub)} | {pct(hit, len(sub))} | {pct(top1, len(sub))} | "
                    f"{statistics.median(ious):.2f} | {statistics.median(px):.0f} | {extra / len(sub):.2f} |"
                )
        say("")
        say("### What a miss is, and recall by the face's size as stored")
        say(
            "Miss kinds: *none* = no app-kept detection at all; *off* = the best kept detection overlaps "
            "the face at IoU 0.1-0.5; *elsewhere* = kept detections, none on the face (IoU < 0.1)."
        )
        say("")
        say("| resolution | misses | none | off | elsewhere |")
        say("|---|---|---|---|---|")
        px_bins = (0, 16, 24, 32, 48, 64, 96, 128, 1 << 20)
        by_px: dict[int, list[int]] = collections.defaultdict(lambda: [0, 0])
        by_pose: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
        for res in resolutions:
            kinds = collections.Counter()
            for r in rows:
                rec = det[r["img"]]
                s = rec["scale"][res]
                kept = [d for d in rec["det"][res] if app_keeps(d, s)]
                gt = [r["x1"], r["y1"], r["x2"], r["y2"]]
                best = max((iou(gt, d) for d in kept), default=None)
                hit = best is not None and best >= IOU_HIT
                if not hit:
                    kinds["none" if best is None else "off" if best >= 0.1 else "elsewhere"] += 1
                px = min(r["x2"] - r["x1"], r["y2"] - r["y1"]) * s
                b = next(j for j in range(len(px_bins) - 1) if px < px_bins[j + 1])
                by_px[b][0] += hit
                by_px[b][1] += 1
                if res == "full":
                    by_pose[r["head_pose"]][0] += hit
                    by_pose[r["head_pose"]][1] += 1
            m = sum(kinds.values())
            say(f"| {res} | {m} | {kinds['none']} | {kinds['off']} | {kinds['elsewhere']} |")
        say("")
        say("Recall by the face box's short side in the pixels the app sees, pooled over every resolution:")
        say("")
        say("| face px | photos | hit |")
        say("|---|---|---|")
        for j in sorted(by_px):
            lo, hi = px_bins[j], px_bins[j + 1]
            label = f">= {lo}" if hi == 1 << 20 else f"{lo}-{hi - 1}"
            say(f"| {label} | {by_px[j][1]} | {pct(*by_px[j])} |")
        say("")
        say("Recall at full resolution by FHIBE's head-pose label:")
        say("")
        say("| head pose | photos | hit |")
        say("|---|---|---|")
        for p in sorted(by_pose):
            say(f"| {p or '(none)'} | {by_pose[p][1]} | {pct(*by_pose[p])} |")
    text = "\n".join(lines) + "\n"
    (out / "census.md").write_text(text)
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=("annotations", "detect", "report"))
    ap.add_argument("--root", type=Path, help="directory holding FHIBE's filepaths.csv")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--long-side", type=int, action="append", default=[])
    a = ap.parse_args()
    if a.stage in ("annotations", "detect") and not a.root:
        ap.error("--root is required for this stage")
    if a.stage == "annotations":
        stage_annotations(a.root, a.out)
    elif a.stage == "detect":
        stage_detect(a.root, a.out, a.shard, a.long_side)
    else:
        stage_report(a.out)


if __name__ == "__main__":
    os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
    main()
