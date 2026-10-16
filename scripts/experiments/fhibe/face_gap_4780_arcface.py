#!/usr/bin/env python
"""A second face embedder on the same FHIBE faces: ArcFace in place of FaceNet, id for id (#4780).

#4762 found the app does worse for she/her people than he/him on FHIBE's face crops, and the gap is
already in the full-label ceiling. #4780 asks whether a second face embedder (an ArcFace-class model)
shows it too. This builds that arm without touching the app: the same MTCNN detections, the same
media ids, labels and metadata as a FaceNet cell (``fhibe_faces_roll_<size>__face.pkl``), with each
face's vector replaced by ArcFace's.

* **Detections.** ``align_landmarks_4777.py detect`` kept every MTCNN detection with its five
  landmarks on each stored copy. This re-applies ``image2face``'s own keep rule
  (``_kept_faces``) and the loader's labelling (``pilebuild.loaders.fhibe``: the subject is the best
  IoU >= 0.5, other faces below 0.1 take ``EXTRA_BIT | pid << 4 | k``), and refuses to write unless
  it reproduces the FaceNet cell's id set exactly.
* **Model.** insightface's ``buffalo_l`` recognition model, ``w600k_r50.onnx`` (an ArcFace-loss
  ResNet-50 trained on WebFace600K), run with onnxruntime on the CPU. Its licence is non-commercial
  research, which a bias benchmark is; it never ships.
* **Input.** ArcFace's own 112 x 112 alignment: the least-squares similarity
  (``image2face.fit_similarity``) taking MTCNN's five landmarks onto insightface's template, warped
  from the same stored copy the FaceNet crop came from, ``(x - 127.5) / 127.5``, RGB. A face whose
  landmarks MTCNN did not give falls back to its plain box, resized.
* **Output.** The vector, L2-normalised as FaceNet's are, under the embedding key ``face``: the
  harness reads a cell by name, so the cell's file goes into a private pile (``--pile``) where it
  stands in for FaceNet's, and ``VTS_PILE=<pile>`` runs the shipped face path on it. The real file
  and a provenance note stay owner-only beside the release (``--out``).

``--embedder facenet_tight`` is a control for the crop rather than the model: the app's own FaceNet
on the same tight, aligned crop (ArcFace's template scaled to FaceNet's 160 px), for the static read
only (no ``--pile``).

Usage (on a compute node)::

    python face_gap_4780_arcface.py --size 1024 --model <dir>/w600k_r50.onnx \\
        --out <release>-derived/cells-4780 --pile /expscratch/$USER/fhibe/gap-4780/pile [--threads 8]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import pickle
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pile"))

import pile_config as pc  # noqa: E402
from pilebuild.loaders import fhibe  # noqa: E402

#: insightface's five-point template for its 112 x 112 recognition crops (arcface_dst), image-left first.
ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]]
)
INPUT = 112
BATCH = 64
EMBEDDING_KEY = "face"


def detections(row: dict, conv, params: tuple) -> list[tuple[int, object]]:
    """``(media id, FaceDetection)`` for every face the loader would keep in this photo, in its order."""
    threshold, padding, min_size = params[:3]
    if row["boxes"] is None:
        return []
    w, h = row["size"]
    kept = conv._kept_faces((row["boxes"], row["probs"], row["points"]), w, h, threshold, padding, min_size)
    scored = sorted(((fhibe.iou(row["face"], list(k.raw)), k) for k in kept), key=lambda t: (-t[0], -t[1].conf))
    out = []
    if scored and scored[0][0] >= fhibe.SUBJECT_IOU:
        out.append((row["pid"], scored[0][1]))
        scored = scored[1:]
    extras = [k for v, k in scored if v < fhibe.ELSEWHERE_IOU][:16]
    out += [(fhibe.EXTRA_BIT | (row["pid"] << 4) | j, k) for j, k in enumerate(extras)]
    return out


def aligned(img, det, size: int = INPUT) -> tuple[np.ndarray, bool]:
    """The *size* x *size* RGB crop ArcFace expects (its template scaled), and whether it came from the landmarks."""
    import cv2  # noqa: PLC0415

    from vtscore.converters.image2face import fit_similarity  # noqa: PLC0415

    arr = np.asarray(img)
    if det.landmarks is None:
        x1, y1, x2, y2 = det.box
        return cv2.resize(arr[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_AREA), False
    a, b, tx, ty = fit_similarity(det.landmarks, ARCFACE_DST * size / INPUT)
    m = np.array([[a, -b, tx], [b, a, ty]], dtype=np.float64)
    return cv2.warpAffine(arr, m, (size, size), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0)), True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--size", type=int, required=True, choices=(1024, 640))
    ap.add_argument("--arm", default="roll", help="the FaceNet cell whose ids and metadata this copies")
    ap.add_argument("--model", default=None, help="the ArcFace ONNX file (--embedder arcface)")
    ap.add_argument(
        "--embedder",
        default="arcface",
        choices=("arcface", "facenet_tight"),
        help="facenet_tight: the app's FaceNet on ArcFace's tight crop (at 160 px), a control for the crop",
    )
    ap.add_argument("--landmarks", default=None, help="landmarks_<size>.pkl (default: #4777's)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pile", default=None, help="link the cell into this private pile (the arcface arm's sessions)")
    ap.add_argument("--threads", type=int, default=max(1, len(os.sched_getaffinity(0))))
    args = ap.parse_args()

    import onnxruntime as ort  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    from vtscore.converters.image2face import CONVERTER  # noqa: PLC0415

    derived = pc.fhibe_derived_dir(args.size).parent
    dataset = f"fhibe_faces_{args.arm}_{args.size}"
    cell_name = f"{dataset}__face.pkl"
    src_cell = derived / "cells" / cell_name
    lm_path = Path(args.landmarks or derived / "runs" / "2026-10-10-align-4777" / f"landmarks_{args.size}.pkl")
    cell = pickle.loads(src_cell.read_bytes())  # noqa: S301  # our own pile cell
    rows = pickle.loads(lm_path.read_bytes())  # noqa: S301  # #4777's detect stage
    params = CONVERTER._resolve_params({"align": args.arm if args.arm in ("roll", "similarity") else "none"})
    copies = pc.fhibe_derived_dir(args.size)
    uid = {p.pid: p.uid for p in fhibe.read_photos(fhibe.data_dir())}

    plan = [(mid, row["pid"], det) for row in rows for mid, det in detections(row, CONVERTER, params)]
    want, got = set(cell), {mid for mid, _, _ in plan}
    if want != got:
        raise SystemExit(
            f"{dataset}: rebuilt {len(got):,} faces, the cell holds {len(want):,} "
            f"({len(got - want)} extra, {len(want - got)} missing); refusing to write a cell that does not pair"
        )
    print(f"{dataset}: {len(plan):,} faces rebuilt id for id from {lm_path.name}", flush=True)

    if args.embedder == "arcface":
        so = ort.SessionOptions()
        so.intra_op_num_threads = args.threads
        sess = ort.InferenceSession(args.model, so, providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        size, scale, tag = INPUT, 127.5, "arcface_r50"

        def embed(x: np.ndarray) -> np.ndarray:
            return sess.run(None, {in_name: x})[0]

    else:
        import torch  # noqa: PLC0415
        from facenet_pytorch import InceptionResnetV1  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

        torch.set_num_threads(args.threads)
        net = InceptionResnetV1(pretrained="vggface2").eval()
        size, scale, tag = 160, 128.0, "facenet_tight"

        def embed(x: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                return net(torch.from_numpy(x)).numpy()

    vecs: dict[int, np.ndarray] = {}
    fallback = 0
    batch_ids: list[int] = []
    batch: list[np.ndarray] = []

    def flush() -> None:
        x = (np.stack(batch).astype(np.float32) - 127.5) / scale
        out = embed(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))
        out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
        vecs.update(zip(batch_ids, out.astype(np.float32)))
        batch_ids.clear()
        batch.clear()

    t0 = time.time()
    by_photo: dict[int, list] = {}
    for mid, pid, det in plan:
        by_photo.setdefault(pid, []).append((mid, det))
    for k, (pid, dets) in enumerate(by_photo.items(), 1):
        with Image.open(io.BytesIO((copies / f"{uid[pid]}.jpg").read_bytes())) as im:
            img = im.convert("RGB")
        for mid, det in dets:
            crop, used = aligned(img, det, size)
            fallback += not used
            batch_ids.append(mid)
            batch.append(crop)
            if len(batch) == BATCH:
                flush()
        if k % 2000 == 0:
            print(f"  {k:,}/{len(by_photo):,} photos in {time.time() - t0:.0f}s", flush=True)
    if batch:
        flush()

    new = copy.deepcopy(cell)
    for mid, media in new.items():
        media["embeddings"] = {EMBEDDING_KEY: vecs[mid]}
    mat = np.stack([vecs[m] for m in sorted(new)])
    sha = hashlib.sha256(mat.tobytes()).hexdigest()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir.chmod(0o700)
    real = out_dir / f"{dataset}__{tag}.pkl"
    old = os.umask(0o077)
    try:
        real.write_bytes(pickle.dumps(new, protocol=pickle.HIGHEST_PROTOCOL))
        git = ["git", "-C", str(HERE), "rev-parse", "HEAD"]
        commit = subprocess.run(git, capture_output=True, text=True).stdout  # noqa: S603, S607  # our own checkout
        prov = {
            "dataset": dataset,
            "stands_in_for": cell_name,
            "embedder": (
                "ArcFace R50 (insightface buffalo_l w600k_r50.onnx), under the key 'face'"
                if tag == "arcface_r50"
                else "FaceNet (facenet-pytorch InceptionResnetV1, vggface2) on the tight crop, under the key 'face'"
            ),
            "model_sha256": hashlib.sha256(Path(args.model).read_bytes()).hexdigest() if args.model else None,
            "landmarks": str(lm_path),
            "alignment": f"5-point similarity onto insightface's arcface_dst, {size} x {size}, from the stored copy",
            "fallback_box_crops": fallback,
            "n_vectors": len(new),
            "vectors_sha256": sha,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "host": socket.gethostname(),
            "code_commit": commit.strip(),
            "seconds": round(time.time() - t0, 1),
        }
        (out_dir / f"{dataset}__{tag}.provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    finally:
        os.umask(old)
    print(f"{dataset}: {len(new):,} {tag} vectors ({fallback} from plain boxes) -> {real}")
    if args.pile:
        emb = Path(args.pile) / "datadir" / "embeddings"
        emb.mkdir(parents=True, exist_ok=True)
        link = emb / cell_name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(real)
        print(f"  {link} -> it")
    print(json.dumps(prov, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
