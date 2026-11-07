"""#5373 sizing: re-run the app's image2face on low-quality subject faces at 0/90/180/270.

For each subject face below --qmax in the cell, the photo (the cell's own long-side copy) is
turned by PIL transpose, run through image2face._find_faces (the app path: threshold, min
size, provider), and every rotated box mapped back to the upright frame. The rotated face kept
is the one with IoU >= 0.5 against the 0-degree detection that reproduces the pile vector.
Match to own photos: mean cosine to the person's other subject faces, and AP of those among
every other face in the cell.
usage: size.py SIZE QMAX SHARD NSHARDS OUT.csv
"""

import csv
import pickle
import sys
import numpy as np
from PIL import Image

SIZE, QMAX, SHARD, NSH, OUT = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
D = "/expscratch/sgreenberg/fhibe/fhibe-full-resolution-674a7dcf-derived"
from vtscore.converters.image2face import CONVERTER as conv

with open(f"{D}/cells/fhibe_faces_ik_{SIZE}__face.pkl", "rb") as fh:
    cell = pickle.loads(fh.read())  # noqa: S301  # our own pile cell
recs = list(cell.values())
E = np.array([r["embeddings"]["face"] for r in recs], dtype=np.float64)
E /= np.linalg.norm(E, axis=1, keepdims=True)
cats = np.array([r.get("category") or "" for r in recs])
todo = [i for i, r in enumerate(recs) if cats[i] and r["custom_metadata"]["Face Quality"] < QMAX]
todo = todo[SHARD::NSH]
params = conv._resolve_params({"align": "none"})
TR = {0: None, 90: Image.Transpose.ROTATE_90, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_270}


def back(box, rot, W, H):
    """Map a box in the rotated frame back to the upright W x H frame."""
    x1, y1, x2, y2 = box
    pts = [(x1, y1), (x2, y2)]
    if rot == 90:
        pts = [(W - v, u) for u, v in pts]
    elif rot == 180:
        pts = [(W - u, H - v) for u, v in pts]
    elif rot == 270:
        pts = [(v, H - u) for u, v in pts]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def ap(scores, pos):
    o = np.argsort(-scores)
    hits = pos[o]
    if hits.sum() == 0:
        return float("nan")
    prec = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((prec * hits).sum() / hits.sum())


with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["idx", "uid", "category", "n_own", "q_pile", "rot", "n_faces", "iou", "q", "self_cos", "cos_own", "ap"])
    for i in todo:
        r = recs[i]
        uid = r["origin_name"].split(":")[1]
        img = Image.open(f"{D}/long{SIZE}/{uid}.jpg").convert("RGB")
        W, H = img.size
        mask = np.ones(len(recs), bool)
        mask[i] = False
        own = (cats == cats[i]) & mask
        ref = None
        for rot in (0, 90, 180, 270):
            im = img if TR[rot] is None else img.transpose(TR[rot])
            found = conv._find_faces(im, params, uid, strict=True)
            row = [i, uid, cats[i], int(own.sum()), r["custom_metadata"]["Face Quality"], rot, len(found)]
            cand = []
            for k, extra in found:
                e = np.asarray(extra["embeddings"]["face"], np.float64)
                e /= np.linalg.norm(e)
                cand.append((back(k.raw, rot, W, H), e, extra["custom_metadata"]["Face Quality"]))
            if rot == 0:
                if cand:
                    j = int(np.argmax([c[1] @ E[i] for c in cand]))
                    ref = cand[j][0]
                    pick, ov = cand[j], 1.0
                else:
                    pick = None
            else:
                pick, ov = None, 0.0
                if ref is not None and cand:
                    ovs = [iou(ref, c[0]) for c in cand]
                    j = int(np.argmax(ovs))
                    if ovs[j] >= 0.5:
                        pick, ov = cand[j], ovs[j]
            if pick is None:
                w.writerow(row + ["", "", "", "", ""])
                continue
            e = pick[1]
            s = E @ e
            w.writerow(
                row
                + [
                    round(ov, 3),
                    pick[2],
                    round(float(e @ E[i]), 4),
                    round(float(s[own].mean()), 4),
                    round(ap(s[mask], own[mask]), 4),
                ]
            )
        fh.flush()
