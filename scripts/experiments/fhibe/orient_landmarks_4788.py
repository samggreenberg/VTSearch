#!/usr/bin/env python3
"""MTCNN on every one-person FHIBE photo turned 0, 90, 180 and 270 degrees (#4788).

A face lying on its side is often still found by MTCNN, but its five landmarks come back placed as
if it were upright, so ``image2face``'s ``roll`` alignment (#4777) reads a small angle and turns
nothing. This measures what the turned copies say about each face: whether MTCNN finds it there,
how confidently, and where it puts the landmarks, all mapped back to the stored frame.

``detect --long-side N --turn T``
    MTCNN (``image2face``'s own detector) on each stored copy turned T degrees counter-clockwise
    (an exact pixel transpose), every detection's box and landmarks mapped back to the stored
    frame. One pickle per size and turn, ``turns_<N>_<T>.pkl``, owner-only.

``table --long-side N``
    One row per photo whose subject the plain pass finds (the loader's rule at turn 0): per turn,
    the matching detection (IoU >= 0.5 with the turn-0 subject box in the stored frame), its
    confidence, and the angle the 5-point fit to ``FACE_TEMPLATE`` gives for its mapped landmarks.
    Plus how many faces the turned passes find that the plain pass does not. Writes
    ``turns_<N>.csv`` (photo id, subject, per-turn conf and angle; owner-only).

    python orient_landmarks_4788.py detect --long-side 1024 --turn 90 --out <dir>
    python orient_landmarks_4788.py table --long-side 1024 --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pile"))

import pile_config as pc  # noqa: E402
from pilebuild.loaders import fhibe  # noqa: E402

SIZES = (1024, 640)
TURNS = (0, 90, 180, 270)


def to_stored(points: np.ndarray, turn: int, w: int, h: int) -> np.ndarray:
    """Points ``(..., 2)`` in the copy turned *turn* degrees counter-clockwise, back in the stored
    ``w`` x ``h`` frame."""
    u, v = points[..., 0], points[..., 1]
    if turn == 0:
        x, y = u, v
    elif turn == 90:
        x, y = w - v, u
    elif turn == 180:
        x, y = w - u, h - v
    elif turn == 270:
        x, y = v, h - u
    else:
        raise ValueError(turn)
    return np.stack([x, y], axis=-1)


def box_to_stored(box: np.ndarray, turn: int, w: int, h: int) -> np.ndarray:
    corners = to_stored(np.asarray([[box[0], box[1]], [box[2], box[3]]], dtype=np.float64), turn, w, h)
    return np.asarray([corners[:, 0].min(), corners[:, 1].min(), corners[:, 0].max(), corners[:, 1].max()])


def turned(im, turn: int):
    from PIL import Image  # noqa: PLC0415

    op = {90: Image.Transpose.ROTATE_90, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_270}
    return im if turn == 0 else im.transpose(op[turn])


def detect(long_side: int, turn: int, out: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    from vtscore.converters.image2face import CONVERTER  # noqa: PLC0415

    detector = CONVERTER._make_detector()
    if detector is None:
        raise SystemExit("no MTCNN (facenet-pytorch missing)")
    photos = fhibe.read_photos(fhibe.data_dir())
    copies = pc.fhibe_derived_dir(long_side)
    rows, t0 = [], time.time()
    for k, p in enumerate(photos, 1):
        with Image.open(io.BytesIO((copies / f"{p.uid}.jpg").read_bytes())) as im:
            im = im.convert("RGB")
            w, h = im.size
            boxes, probs, points = detector.detect(turned(im, turn), landmarks=True)
        if boxes is not None:
            boxes = np.stack([box_to_stored(b, turn, w, h) for b in np.asarray(boxes, dtype=np.float64)])
            points = to_stored(np.asarray(points, dtype=np.float64), turn, w, h)
            probs = np.asarray(probs, dtype=np.float64)
        rows.append({"pid": p.pid, "subject": p.subject, "size": (w, h), "face": [c * w / p.width for c in p.face],
                     "boxes": boxes, "probs": probs, "points": points})  # fmt: skip
        if k % 1000 == 0:
            print(f"{long_side} turn {turn}: {k:,}/{len(photos):,} in {time.time() - t0:.0f}s", flush=True)
    old = os.umask(0o077)
    try:
        with (out / f"turns_{long_side}_{turn}.pkl").open("wb") as f:
            pickle.dump(rows, f)
    finally:
        os.umask(old)
    print(f"{long_side} turn {turn}: {len(rows):,} photos in {time.time() - t0:.0f}s")


def angle(points: np.ndarray) -> float:
    from vtscore.converters.image2face import FACE_TEMPLATE, fit_similarity  # noqa: PLC0415

    a, b, _, _ = fit_similarity(FACE_TEMPLATE, points)
    return math.degrees(math.atan2(b, a))


def table(long_side: int, out: Path) -> None:
    from vtscore.converters.image2face import CONVERTER  # noqa: PLC0415

    threshold, padding, min_size = CONVERTER._resolve_params(None)[:3]
    passes = {t: pickle.loads((out / f"turns_{long_side}_{t}.pkl").read_bytes()) for t in TURNS}  # noqa: S301
    n = len(passes[0])
    rows, extra = [], {t: 0 for t in TURNS if t}
    for i in range(n):
        base = passes[0][i]
        w, h = base["size"]
        if base["boxes"] is None:
            continue
        kept0 = CONVERTER._kept_faces(
            (base["boxes"], base["probs"], base["points"]), w, h, threshold, padding, min_size
        )
        scored = sorted(((fhibe.iou(base["face"], list(k.raw)), k) for k in kept0), key=lambda t: (-t[0], -t[1].conf))
        if not scored or scored[0][0] < fhibe.SUBJECT_IOU:
            continue
        subj = scored[0][1]
        row = {"pid": base["pid"], "subject": base["subject"]}
        for t in TURNS:
            r = passes[t][i]
            assert r["pid"] == base["pid"]
            best, best_iou = None, 0.0
            if r["boxes"] is not None:
                for b, p, pts in zip(r["boxes"], r["probs"], r["points"]):
                    v = fhibe.iou(list(subj.raw), list(b))
                    if v > best_iou:
                        best, best_iou = (float(p), angle(pts)), v
            if best is not None and best_iou >= 0.5:
                row[f"conf_{t}"], row[f"angle_{t}"] = round(best[0], 6), round(best[1], 2)
            else:
                row[f"conf_{t}"], row[f"angle_{t}"] = "", ""
        rows.append(row)
    # Faces a turned pass finds that the plain pass does not keep at all, over every photo.
    for i in range(n):
        base = passes[0][i]
        w, h = base["size"]
        plain = []
        if base["boxes"] is not None:
            kept = (base["boxes"], base["probs"], base["points"])
            plain = [list(k.raw) for k in CONVERTER._kept_faces(kept, w, h, threshold, padding, min_size)]
        for t in extra:
            r = passes[t][i]
            if r["boxes"] is None:
                continue
            found = CONVERTER._kept_faces((r["boxes"], r["probs"], r["points"]), w, h, threshold, padding, min_size)
            extra[t] += sum(1 for k in found if all(fhibe.iou(list(k.raw), b) < 0.1 for b in plain))
    with (out / f"turns_{long_side}.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"{long_side}: {len(rows):,} subject faces -> {out / f'turns_{long_side}.csv'}")
    print(f"  faces only a turned pass finds (conf >= {threshold}, IoU < 0.1 with every plain face): {extra}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("detect", "table"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--long-side", type=int, choices=SIZES, required=True)
    ap.add_argument("--turn", type=int, choices=TURNS)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.out.chmod(0o700)
    if args.stage == "detect":
        if args.turn is None:
            ap.error("detect needs --turn")
        detect(args.long_side, args.turn, args.out)
    else:
        table(args.long_side, args.out)


if __name__ == "__main__":
    main()
