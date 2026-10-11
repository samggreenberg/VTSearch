#!/usr/bin/env python3
"""MTCNN's five landmarks on every one-person FHIBE photo, and the template ``image2face`` aligns to (#4777).

Two stages, each writing into ``--out`` (owner-only, beside the release: FHIBE's
terms delete every derived table with the release it came from):

``detect --long-side N``
    Runs ``image2face``'s own MTCNN (its ``_make_detector``) on the stored copy of
    every one-person photo at that size and keeps every detection: raw box,
    confidence and the five landmarks (eyes, nose, mouth corners, image-left
    first). One pickle per size, ``landmarks_<N>.pkl``.

``template``
    Where an upright face's landmarks sit in MTCNN's box. For each photo's
    subject detection (the loader's rule: the best IoU against the annotated face
    box, at least 0.5), the landmarks as fractions of the raw box; the template is
    their median over subjects whose eye line is within ``--upright`` degrees of
    level, at ``--template-size`` px. Prints the template, the roll each subject
    face is at (the angle the 5-point fit removes), and writes ``subjects_<N>.csv``
    (photo id, subject, roll, fit scale, detection confidence) for the report.

    python align_landmarks_4777.py detect --long-side 1024 --out <dir>
    python align_landmarks_4777.py template --out <dir>
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


def detect(long_side: int, out: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    from vtscore.converters.image2face import CONVERTER  # noqa: PLC0415

    detector = CONVERTER._make_detector()
    if detector is None:
        raise SystemExit("no MTCNN (facenet-pytorch missing)")
    root = fhibe.data_dir()
    photos = fhibe.read_photos(root)
    copies = pc.fhibe_derived_dir(long_side)
    rows = []
    t0 = time.time()
    for k, p in enumerate(photos, 1):
        with Image.open(io.BytesIO((copies / f"{p.uid}.jpg").read_bytes())) as im:
            im = im.convert("RGB")
            boxes, probs, points = detector.detect(im, landmarks=True)
            size = im.size
        rows.append(
            {
                "pid": p.pid,
                "subject": p.subject,
                "size": size,
                "face": [c * size[0] / p.width for c in p.face],
                "boxes": None if boxes is None else np.asarray(boxes, dtype=np.float64),
                "probs": None if boxes is None else np.asarray(probs, dtype=np.float64),
                "points": None if boxes is None else np.asarray(points, dtype=np.float64),
            }
        )
        if k % 1000 == 0:
            print(f"{long_side}: {k:,}/{len(photos):,} in {time.time() - t0:.0f}s", flush=True)
    old = os.umask(0o077)
    try:
        with (out / f"landmarks_{long_side}.pkl").open("wb") as f:
            pickle.dump(rows, f)
    finally:
        os.umask(old)
    print(f"{long_side}: {len(rows):,} photos -> {out / f'landmarks_{long_side}.pkl'}")


def subject_detection(row: dict, threshold: float) -> int | None:
    """The loader's subject: the kept detection of best IoU with the annotated face, if >= SUBJECT_IOU."""
    if row["boxes"] is None:
        return None
    best, best_iou = None, -1.0
    for i, (box, prob) in enumerate(zip(row["boxes"], row["probs"])):
        if prob < threshold:
            continue
        v = fhibe.iou(row["face"], list(box))
        if v > best_iou or (v == best_iou and prob > row["probs"][best]):
            best, best_iou = i, v
    return best if best is not None and best_iou >= fhibe.SUBJECT_IOU else None


def template(out: Path, template_size: int, upright: float) -> None:
    from vtscore.converters.image2face import fit_similarity  # noqa: PLC0415

    for long_side in SIZES:
        path = out / f"landmarks_{long_side}.pkl"
        if not path.exists():
            continue
        rows = pickle.loads(path.read_bytes())  # noqa: S301  # the detect stage's own output
        fracs, eye_angles, table = [], [], []
        for row in rows:
            i = subject_detection(row, 0.5)
            if i is None:
                continue
            x0, y0, x1, y1 = row["boxes"][i]
            pts = row["points"][i]
            frac = np.column_stack([(pts[:, 0] - x0) / (x1 - x0), (pts[:, 1] - y0) / (y1 - y0)])
            eye = math.degrees(math.atan2(pts[1, 1] - pts[0, 1], pts[1, 0] - pts[0, 0]))
            fracs.append(frac)
            eye_angles.append(eye)
            table.append((row["pid"], row["subject"], pts, float(row["probs"][i]), (x1 - x0)))
        fracs_a = np.asarray(fracs)
        eye_a = np.asarray(eye_angles)
        level = np.abs(eye_a) <= upright
        tmpl = np.median(fracs_a[level], axis=0)
        print(
            f"\n{long_side}: {len(fracs):,} subject faces, {int(level.sum()):,} with the eye line within {upright} deg"
        )
        print("  template (median, upright):", np.round(tmpl, 3).tolist())
        print("  template (median, all):    ", np.round(np.median(fracs_a, axis=0), 3).tolist())
        if long_side != template_size:
            continue
        # Roll as the 5-point fit to this size's template sees it, for every subject face.
        rolls = []
        with (out / f"subjects_{long_side}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pid", "subject", "roll_deg", "eye_line_deg", "fit_scale_over_box", "conf"])
            for (pid, subject, pts, conf, side), eye in zip(table, eye_a):
                a, b, _tx, _ty = fit_similarity(tmpl, pts)
                roll = math.degrees(math.atan2(b, a))
                rolls.append(roll)
                w.writerow([pid, subject, f"{roll:.2f}", f"{eye:.2f}", f"{math.hypot(a, b) / side:.4f}", f"{conf:.4f}"])
        r = np.abs(np.asarray(rolls))
        for cut in (5, 10, 15, 20, 30, 45, 90):
            print(f"  |roll| > {cut:>2} deg: {int((r > cut).sum()):>5,} ({(r > cut).mean():.1%})")
        print(f"  subjects table -> {out / f'subjects_{long_side}.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("detect", "template"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--long-side", type=int, choices=SIZES)
    ap.add_argument("--template-size", type=int, default=1024, choices=SIZES)
    ap.add_argument("--upright", type=float, default=10.0, help="eye-line tilt, in degrees, still counted upright")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.out.chmod(0o700)
    if args.stage == "detect":
        if args.long_side is None:
            ap.error("detect needs --long-side")
        detect(args.long_side, args.out)
    else:
        template(args.out, args.template_size, args.upright)


if __name__ == "__main__":
    main()
