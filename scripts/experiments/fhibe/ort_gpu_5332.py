"""#5332: the imagekit face path on GPU vs CPU under buffalo_s, same photos, same node.

    IMAGEKIT_FACE_CONFIG=<cfg> python ort_gpu_5332.py run <n photos> <size> <label> <out.npz>
    python ort_gpu_5332.py compare <a.npz> <b.npz>

``run`` times image2face (provider imagekit, align roll; 5 warm-up photos), then
saves every face's box, confidence and unit embedding from the face pipeline.
``ORT_GPU_DIR`` puts an onnxruntime-gpu build ahead of the venv's (unset = the
venv's own); ``ORT_PRELOAD=1`` calls ``ort.preload_dlls()`` first.  ``compare``
pairs faces by box IoU and prints the cosine per face.
"""

import os
import sys
import time

import numpy as np

if os.environ.get("ORT_GPU_DIR"):
    sys.path.insert(0, os.environ["ORT_GPU_DIR"])


def run(n: int, size: int, label: str, out: str) -> None:
    import onnxruntime as ort

    if os.environ.get("ORT_PRELOAD") == "1":
        ort.preload_dlls()
    print("ort", ort.__version__, ort.get_available_providers(), file=sys.stderr)
    from PIL import Image

    sys.path.insert(0, "scripts/experiments/pile")
    import pile_config as pc
    from pilebuild.loaders import fhibe

    from vtscore.converters.image2face import CONVERTER as conv
    from vtscore.media.face import imagekit_bridge as bridge

    photos = fhibe.read_photos(fhibe.data_dir())[: n + 5]
    copies = pc.fhibe_derived_dir(size)
    medias = [{"media_bytes": (copies / f"{p.uid}.jpg").read_bytes(), "filename": f"{p.uid}.jpg"} for p in photos]
    params = {"align": "roll"}
    for m in medias[:5]:
        conv.convert(m, params)
    pipeline = bridge.get_face_pipeline()
    sessions = [
        s.get_providers()
        for o in vars(pipeline).values()
        for s in (vars(o).values() if hasattr(o, "__dict__") else [])
        if hasattr(s, "get_providers")
    ]
    print("sessions", sessions, file=sys.stderr)
    t = time.perf_counter()
    faces = 0
    for m in medias[5:]:
        faces += len(conv.convert(m, params))
    s = time.perf_counter() - t
    print(
        f"{label} {size}: {n} photos, {faces} faces; {1000 * s / n:.0f} ms/photo, {1000 * s / max(faces, 1):.0f} ms/face"
    )
    idx, boxes, confs, embs = [], [], [], []
    for i, p in enumerate(photos[5:]):
        img = np.asarray(Image.open(copies / f"{p.uid}.jpg").convert("RGB"))
        for rec in pipeline.process(img, confidence_threshold=0.5, media_id=p.uid):
            idx.append(i)
            boxes.append([float(v) for v in rec.bbox])
            confs.append(float(rec.confidence))
            embs.append(rec.unit_embedding())
    np.savez(out, idx=np.array(idx), boxes=np.array(boxes), confs=np.array(confs), embs=np.array(embs))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def compare(a_path: str, b_path: str) -> None:
    a, b = np.load(a_path), np.load(b_path)
    cos, ious, dconf, unmatched = [], [], [], 0
    for i in sorted(set(a["idx"]) | set(b["idx"])):
        ja, jb = np.flatnonzero(a["idx"] == i), np.flatnonzero(b["idx"] == i)
        used = set()
        for x in ja:
            best = max(((_iou(a["boxes"][x], b["boxes"][y]), y) for y in jb if y not in used), default=(0.0, -1))
            if best[0] < 0.5:
                unmatched += 1
                continue
            used.add(best[1])
            ious.append(best[0])
            cos.append(float(a["embs"][x] @ b["embs"][best[1]]))
            dconf.append(abs(a["confs"][x] - b["confs"][best[1]]))
        unmatched += len(jb) - len(used)
    cos = np.array(cos)
    print(f"{len(a['idx'])} vs {len(b['idx'])} faces; matched {len(cos)}, unmatched {unmatched}")
    print(f"cosine min {cos.min():.6f} p1 {np.percentile(cos, 1):.6f} median {np.median(cos):.6f}")
    print(f"box IoU min {min(ious):.4f}; |dconf| max {max(dconf):.5f}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5])
    else:
        compare(sys.argv[2], sys.argv[3])
