#!/usr/bin/env python
"""How each FHIBE one-person photo was taken, from FHIBE's own annotations (#4780).

#4762's face review found the app does worse for she/her people than he/him (0.73 vs 0.85 after the
check), for younger people and for middle skin tones. Those are strata of the people; #4780 asks
whether they hold once the photos are matched on how they were taken. This writes one row per
one-person photo with what FHIBE annotates about it:

* the person: ``subject``, ``pronoun``, ``age``, ``skin_tone`` (``natural_skin_color`` index 0-5);
* the camera: ``camera_distance`` (FHIBE's CD I-V as 0-4; #4709 found the labels run near to far),
  ``camera_position`` (typical / atypical high / low), ``head_pose`` (typical / atypical), indoor or
  outdoor ``scene``, ``lighting`` from behind the face, the capture month and the phone model;
* the face: its annotated box's side in px at the 1024 px copy the app reads (``face_px_1024``);
  the share of that box covered by FHIBE's segments - face skin, head hair, headwear, eyewear, a
  mask (``frac_*``), each rasterised on a 64 x 64 grid over the box; whether headwear, eyewear,
  a mask or jewellery is segmented anywhere (``has_*``); facial hair; make-up as a facial mark;
  the hairstyle; how many of FHIBE's eleven face keypoints are marked (``face_keypoints``).

The output names people (subject IDs): it stays in the release's owner-only derived directory.

Usage (on a compute node: it reads 10,901 JSONs with their polygons)::

    python face_gap_4780_covariates.py --out <runs>/2026-10-12-gap-4780/photo_covariates.csv [--workers 8]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pile"))
sys.path.insert(0, str(HERE))

from person_strata import _first, _strip_index  # noqa: E402

GRID = 64
#: FHIBE's segment classes that can cover a face, by the column they feed.
COVER = {
    "0. Face skin": "skin",
    "6. Head hair": "hair",
    "17. Headwear": "headwear",
    "19. Eyewear": "eyewear",
    "18. Mask": "mask",
}
ANYWHERE = {
    "17. Headwear": "headwear",
    "19. Eyewear": "eyewear",
    "18. Mask": "mask",
    "27. Jewelry or timepiece": "jewelry",
}
#: FHIBE's keypoints 0-10 are the face's (nose, eyes, ears, mouth corners).
FACE_KEYPOINTS = 11


def _index(value) -> int:
    """``"3. CD IV"`` -> 3; -1 when absent."""
    s = str(value or "")
    head = s.split(".", 1)[0].strip()
    return int(head) if head.lstrip("-").isdigit() else -1


def _cover(segments: list, box: tuple[float, float, float, float]) -> dict[str, float]:
    """The share of *box* (x, y, w, h) each covering class fills, rasterised on a GRID x GRID raster."""
    import cv2  # noqa: PLC0415

    x0, y0, w, h = box
    out = {}
    for cls, name in COVER.items():
        mask = np.zeros((GRID, GRID), np.uint8)
        for seg in segments:
            if seg.get("class_name") != cls or not seg.get("polygon"):
                continue
            pts = np.array([[(p["x"] - x0) / w * GRID, (p["y"] - y0) / h * GRID] for p in seg["polygon"]])
            cv2.fillPoly(mask, [np.round(pts * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
        out[f"frac_{name}"] = float(mask.mean())
    return out


def one(args: tuple[str, str, str]) -> dict | None:
    uid, json_rel, root = args
    d = json.loads((Path(root) / json_rel).read_text())
    ia, subs = d["image_annotation"], d["subject_annotation"]
    if int(ia.get("humans_per_image") or 0) != 1 or len(subs) != 1:
        return None
    s = subs[0]
    x, y, w, h = (float(v) for v in s["face_bbox"])
    W, H = int(ia["image_width"]), int(ia["image_height"])
    scale = min(1.0, 1024 / max(W, H))
    segs = s.get("segments") or []
    classes = {g.get("class_name") for g in segs}
    marks = [str(m) for m in (s.get("facial_marks") or [])]
    lighting = [str(v) for v in (ia.get("lighting") or [])]
    facial_hair = [str(v) for v in (s.get("facial_hairstyle") or [])]
    keypoints = s.get("keypoints") or {}
    row = {
        "uid": uid,
        "pid": int(uid.replace("-", "")[:12], 16),
        "subject": s["subject_id"],
        "pronoun": _strip_index(_first(s.get("pronoun"))),
        "age": s.get("age"),
        "skin_tone": str(s.get("natural_skin_color", "")).split(".")[0].strip(),
        "camera_distance": _index(ia.get("camera_distance")),
        "camera_position": _index(ia.get("camera_position")),
        "head_pose": _index(s.get("head_pose")),
        "scene_outdoor": int("Outdoor" in str(ia.get("scene", ""))),
        "light_behind": int(any("behind" in v for v in lighting)),
        "captured": str(ia.get("user_date_captured", "")),
        "hour": str(ia.get("user_hour_captured", "")),
        "phone": f"{ia.get('manufacturer', '')} {ia.get('model', '')}".strip(),
        "country": str(ia.get("location_country", "")),
        "face_px_1024": float(np.sqrt(w * h) * scale),
        "face_aspect": h / w if w > 0 else float("nan"),
        **_cover(segs, (x, y, w, h)),
        **{f"has_{name}": int(cls in classes) for cls, name in ANYWHERE.items()},
        "facial_hair": int(any(not v.startswith("0.") for v in facial_hair)),
        "makeup": int(any("Make-up" in m for m in marks)),
        "hairstyle": _strip_index(str(s.get("hairstyle", ""))),
        "face_keypoints": sum(1 for k in keypoints if _index(k) in range(FACE_KEYPOINTS)),
    }
    return row


def main() -> int:
    from person_strata import RELEASE  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, len(os.sched_getaffinity(0))))
    args = ap.parse_args()
    todo = [(r["uid"], r["json"], str(RELEASE)) for r in csv.DictReader(open(RELEASE / "filepaths.csv"))]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = [r for r in pool.map(one, todo, chunksize=32) if r is not None]
    rows.sort(key=lambda r: r["pid"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    old = os.umask(0o077)
    try:
        with open(out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    finally:
        os.umask(old)
    print(f"{len(rows):,} one-person photos of {len({r['subject'] for r in rows}):,} people -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
