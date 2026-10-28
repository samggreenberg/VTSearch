#!/usr/bin/env python
"""buffalo_m against buffalo_l on FHIBE's face cells: the same ArcFace, a smaller detector (#5330).

Both packs ship the same ``w600k_r50.onnx`` (byte for byte); they differ only in SCRFD, ``det_10g``
(buffalo_l) against ``det_2.5g`` (buffalo_m). Three reads, each per stored size (1024 / 640 px):

* **detections** (``detect_all.py``'s shards): per one-person photo, whether each detector finds the
  annotated face (best IoU >= 0.5 against FHIBE's face box), its confidence, how far the two
  detectors' subject boxes and five landmarks sit apart (landmarks over the inter-ocular distance),
  how many other faces each reports, and detect time per photo (both detectors timed in the same
  process, so the ratio holds whatever node a shard ran on);
* **cells** (the two ``fhibe_faces_ik_<size>__face.pkl`` built on one commit): subjects found, other
  faces kept, and the cosine between the two vectors of the same subject photo;
* **static ranking**: each subject photo queries its own arm's pool (every face, subject or other),
  scored by the AP of the person's other photos (leave-one-out), averaged per person; then the
  per-person difference m - l, and the she/her - he/him and 18-29 - 45+ gaps in it.

    python buffalo_m_compare_5330.py --cells-l <dir> --cells-m <dir> --detect <dir> --strata <person_strata.csv> --out <dir>

Landmarks are also scored against FHIBE's own five annotated points (eyes, nose, mouth corners), as
the mean error over the annotated inter-ocular distance (NME), on the photos where both detectors
find the subject.

Writes aggregates only (``summary.csv``, ``cosine_hist.csv``, ``nme_cdf.csv``); ``people.csv`` (per person, by subject
id) is owner-only.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def _cells(path: Path) -> pd.DataFrame:
    from _cells_io import load_medias  # noqa: PLC0415

    m = load_medias(str(path))
    rows = []
    for mid, v in m.items():
        (vec,) = v["embeddings"].values()
        rows.append({"mid": mid, "pid": v["source_photo_id"], "category": v["category"] or "", "vec": vec})
    return pd.DataFrame(rows)


def _loo_ap(df: pd.DataFrame, block: int = 512) -> pd.DataFrame:
    """Leave-one-out AP of every subject photo against the whole pool of *df*."""
    x = np.stack(df["vec"].to_numpy()).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    cats = df["category"].to_numpy()
    _, cat_ids = np.unique(cats, return_inverse=True)
    empty = cats == ""
    q = np.where(~empty)[0]
    out = np.full(len(q), np.nan)
    for s in range(0, len(q), block):
        idx = q[s : s + block]
        sims = x[idx] @ x.T
        sims[np.arange(len(idx)), idx] = -np.inf
        order = np.argsort(-sims, axis=1)
        rel = (cat_ids[order] == cat_ids[idx][:, None]) & ~empty[order]
        hits = np.cumsum(rel, axis=1)
        npos = rel.sum(axis=1)
        prec = hits / np.arange(1, rel.shape[1] + 1)
        ap = (prec * rel).sum(axis=1) / np.maximum(npos, 1)
        out[s : s + len(idx)] = np.where(npos > 0, ap, np.nan)
    return pd.DataFrame({"pid": df["pid"].to_numpy()[q], "category": cats[q], "ap": out})


#: FHIBE's keypoints in SCRFD's five-point order: the eye and mouth corner on the image's left first
#: (the subject's right), then the other eye, the nose, the other corner.
KEYPOINTS = ("2. Right eye", "5. Left eye", "0. Nose", "9. Mouth right", "10. Mouth left")


def _keypoints(root: Path) -> pd.DataFrame:
    """FHIBE's annotated five points per one-person photo, full-resolution pixels, NaN where absent."""
    import csv  # noqa: PLC0415
    import json  # noqa: PLC0415

    rows = []
    for row in csv.DictReader((root / "filepaths.csv").open()):
        d = json.loads((root / row["json"]).read_text())
        subs = d["subject_annotation"]
        if int(d["image_annotation"]["humans_per_image"]) != 1 or len(subs) != 1:
            continue
        kp = subs[0].get("keypoints") or {}
        r = {"uid": row["uid"]}
        for i, name in enumerate(KEYPOINTS):
            v = kp.get(name)
            ok = v is not None and len(v) >= 2 and (len(v) < 3 or v[2])
            r[f"gx{i}"], r[f"gy{i}"] = (float(v[0]), float(v[1])) if ok else (np.nan, np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


def _nme(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Mean point error over the annotated points, over the annotated inter-ocular distance."""
    iod = np.linalg.norm(gt[:, 0] - gt[:, 1], axis=1)
    err = np.nanmean(np.linalg.norm(pred - gt, axis=2), axis=1)
    return np.where(iod > 0, err / iod, np.nan)


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise IoU of two (n, 4) box arrays."""
    x0 = np.maximum(a[:, 0], b[:, 0])
    y0 = np.maximum(a[:, 1], b[:, 1])
    x1 = np.minimum(a[:, 2], b[:, 2])
    y1 = np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])  # noqa: E731
    return inter / (area(a) + area(b) - inter)


def detections(det_dir: Path, photos: pd.DataFrame, kp: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    names = ["uid", "size", "det", "ms", "conf", "x0", "y0", "x1", "y1"] + [
        f"{c}{i}" for i in range(5) for c in ("lx", "ly")
    ]
    d = pd.concat([pd.read_csv(f, header=0, names=names) for f in sorted(det_dir.glob("shard-*.csv"))])
    d = d.merge(photos, on="uid").merge(kp, on="uid", how="left")
    gcols = [f"g{c}{i}" for i in range(5) for c in "xy"]
    for size in (1024, 640):
        s = size / d["long_full"]
        s = s.clip(upper=1.0)
        mask = d["size"] == size
        for c in ("fx0", "fy0", "fx1", "fy1", *gcols):
            d.loc[mask, c + "_s"] = d.loc[mask, c] * s[mask]
    have = d["conf"].notna()
    d["iou_face"] = np.nan
    d.loc[have, "iou_face"] = _iou(
        d.loc[have, ["x0", "y0", "x1", "y1"]].to_numpy(float),
        d.loc[have, ["fx0_s", "fy0_s", "fx1_s", "fy1_s"]].to_numpy(float),
    )
    ms = d.groupby(["uid", "size", "det"])["ms"].first().unstack("det")
    subj = d[d["iou_face"] >= 0.5].sort_values("iou_face").groupby(["uid", "size", "det"]).tail(1)
    others = d[have & (d["iou_face"].fillna(0) < 0.1)].groupby(["size", "det"]).size()
    rows = []
    per_photo = []
    for size in (1024, 640):
        n = d.loc[d["size"] == size, "uid"].nunique()
        sl = subj[(subj["size"] == size) & (subj["det"] == "l")].set_index("uid")
        sm = subj[(subj["size"] == size) & (subj["det"] == "m")].set_index("uid")
        both = sl.index.intersection(sm.index)
        a, b = sl.loc[both], sm.loc[both]
        box_iou = _iou(a[["x0", "y0", "x1", "y1"]].to_numpy(float), b[["x0", "y0", "x1", "y1"]].to_numpy(float))
        lm_a = a[[f"{c}{i}" for i in range(5) for c in ("lx", "ly")]].to_numpy(float).reshape(-1, 5, 2)
        lm_b = b[[f"{c}{i}" for i in range(5) for c in ("lx", "ly")]].to_numpy(float).reshape(-1, 5, 2)
        iod = np.linalg.norm(lm_a[:, 0] - lm_a[:, 1], axis=1)
        lm_err = np.linalg.norm(lm_a - lm_b, axis=2).mean(axis=1) / np.maximum(iod, 1e-6)
        gt = a[[c + "_s" for c in gcols]].to_numpy(float).reshape(-1, 5, 2)
        nme_l, nme_m = _nme(lm_a, gt), _nme(lm_b, gt)
        split = (lm_err > 0.1) & np.isfinite(nme_l) & np.isfinite(nme_m)
        m_s = ms.xs(size, level="size")
        r = {
            "size": size,
            "photos": n,
            "subject_l": len(sl),
            "subject_m": len(sm),
            "subject_only_l": len(sl.index.difference(sm.index)),
            "subject_only_m": len(sm.index.difference(sl.index)),
            "conf_subject_l_median": round(float(sl["conf"].median()), 3),
            "conf_subject_m_median": round(float(sm["conf"].median()), 3),
            "conf_subject_l_below_0.6": int((sl["conf"] < 0.6).sum()),
            "conf_subject_m_below_0.6": int((sm["conf"] < 0.6).sum()),
            "box_iou_l_m_median": round(float(np.median(box_iou)), 3),
            "box_iou_l_m_below_0.8": int((box_iou < 0.8).sum()),
            "landmark_err_iod_median": round(float(np.median(lm_err)), 4),
            "landmark_err_iod_p95": round(float(np.quantile(lm_err, 0.95)), 4),
            "landmark_err_iod_above_0.1": int((lm_err > 0.1).sum()),
            "gt_photos": int(np.isfinite(nme_l).sum()),
            "nme_l_median": round(float(np.nanmedian(nme_l)), 4),
            "nme_m_median": round(float(np.nanmedian(nme_m)), 4),
            "nme_l_above_0.25": int((nme_l > 0.25).sum()),
            "nme_m_above_0.25": int((nme_m > 0.25).sum()),
            "split_photos": int(split.sum()),
            "split_l_closer": int((nme_l[split] < nme_m[split]).sum()),
            "split_nme_l_median": round(float(np.median(nme_l[split])), 4),
            "split_nme_m_median": round(float(np.median(nme_m[split])), 4),
            "other_faces_l": int(others.get((size, "l"), 0)),
            "other_faces_m": int(others.get((size, "m"), 0)),
            "detect_ms_l_median": round(float(m_s["l"].median()), 1),
            "detect_ms_m_median": round(float(m_s["m"].median()), 1),
            "detect_ratio_m_over_l_median": round(float((m_s["m"] / m_s["l"]).median()), 3),
        }
        rows.append(r)
        per_photo.append(
            pd.DataFrame(
                {"uid": both, "size": size, "box_iou": box_iou, "lm_err": lm_err, "nme_l": nme_l, "nme_m": nme_m}
            )
        )
    return rows, pd.concat(per_photo)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cells-l", required=True, type=Path)
    ap.add_argument("--cells-m", required=True, type=Path)
    ap.add_argument("--detect", required=True, type=Path)
    ap.add_argument("--strata", required=True, type=Path)
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    sys.path.insert(0, str(args.repo / "scripts/experiments/calibration"))
    sys.path.insert(0, str(args.repo / "scripts/experiments/pile"))
    from pilebuild.loaders.fhibe import data_dir, read_photos  # noqa: PLC0415

    ph = read_photos(data_dir())
    photos = pd.DataFrame(
        {
            "uid": [p.uid for p in ph],
            "pid": [p.pid for p in ph],
            "long_full": [max(p.width, p.height) for p in ph],
            "fx0": [p.face[0] for p in ph],
            "fy0": [p.face[1] for p in ph],
            "fx1": [p.face[2] for p in ph],
            "fy1": [p.face[3] for p in ph],
        }
    )
    strata = pd.read_csv(args.strata)
    sid = "category" if "category" in strata.columns else strata.columns[0]
    strata = strata.rename(columns={sid: "category"})
    strata["pronoun"] = strata["pronoun"].str.lower().str.extract(r"^(she/her|he/him)", expand=False)
    strata["age"] = strata["age_bracket"].map(lambda b: "45+" if b in ("45-59", "60+") else b)

    summary, per_photo = detections(args.detect, photos, _keypoints(data_dir()))
    grid = np.round(np.arange(0, 0.501, 0.01), 2)
    cdf = []
    for size, g in per_photo.groupby("size"):
        for det in ("l", "m"):
            v = g[f"nme_{det}"].dropna().to_numpy()
            cdf.append(
                pd.DataFrame(
                    {"size": size, "det": det, "nme": grid, "share_at_or_below": [(v <= t).mean() for t in grid]}
                )
            )
    hists, people = [], []
    for r in summary:
        size = r["size"]
        cl = _cells(args.cells_l / f"fhibe_faces_ik_{size}__face.pkl")
        cm = _cells(args.cells_m / f"fhibe_faces_ik_{size}__face.pkl")
        r |= {
            "cell_faces_l": len(cl),
            "cell_faces_m": len(cm),
            "cell_subjects_l": int((cl["category"] != "").sum()),
            "cell_subjects_m": int((cm["category"] != "").sum()),
            "cell_others_l": int((cl["category"] == "").sum()),
            "cell_others_m": int((cm["category"] == "").sum()),
        }
        sl = cl[cl["category"] != ""].set_index("pid")
        sm = cm[cm["category"] != ""].set_index("pid")
        both = sl.index.intersection(sm.index)
        a = np.stack(sl.loc[both, "vec"].to_numpy())
        b = np.stack(sm.loc[both, "vec"].to_numpy())
        cos = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
        r |= {
            "cos_pairs": len(both),
            "cos_median": round(float(np.median(cos)), 4),
            "cos_p05": round(float(np.quantile(cos, 0.05)), 4),
            "cos_p01": round(float(np.quantile(cos, 0.01)), 4),
            "cos_share_above_0.99": round(float((cos > 0.99).mean()), 4),
            "cos_share_below_0.9": round(float((cos < 0.9).mean()), 4),
            "cos_share_below_0.7": round(float((cos < 0.7).mean()), 4),
        }
        edges = np.array([-1, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0001])
        h, _ = np.histogram(cos, bins=edges)
        hists.append(pd.DataFrame({"size": size, "lo": edges[:-1], "hi": edges[1:], "photos": h}))
        apl = _loo_ap(cl).groupby("category")["ap"].mean().rename("ap_l")
        apm = _loo_ap(cm).groupby("category")["ap"].mean().rename("ap_m")
        p = pd.concat([apl, apm], axis=1).dropna().reset_index().merge(strata, on="category", how="left")
        p["d"] = p["ap_m"] - p["ap_l"]
        low = pd.Series(cos < 0.9, index=sl.loc[both, "category"].to_numpy()).groupby(level=0).any()
        p["low_cos"] = p["category"].map(low).fillna(False).astype(bool)
        r |= {
            "people_with_low_cos_photo": int(p["low_cos"].sum()),
            "loo_ap_d_low_cos": round(float(p.loc[p["low_cos"], "d"].mean()), 4),
            "loo_ap_d_rest": round(float(p.loc[~p["low_cos"], "d"].mean()), 4),
            "loo_ap_d_rest_se": round(float(p.loc[~p["low_cos"], "d"].sem()), 4),
            "loo_ap_d_low_cos_share": round(float(p.loc[p["low_cos"], "d"].sum() / p["d"].sum()), 3),
        }
        r |= {
            "people": len(p),
            "loo_ap_l": round(float(p["ap_l"].mean()), 4),
            "loo_ap_m": round(float(p["ap_m"].mean()), 4),
            "loo_ap_d": round(float(p["d"].mean()), 4),
            "loo_ap_d_se": round(float(p["d"].std(ddof=1) / np.sqrt(len(p))), 4),
            "people_d_below_-0.05": int((p["d"] < -0.05).sum()),
            "people_d_above_0.05": int((p["d"] > 0.05).sum()),
        }
        for key, (ga, gb) in {"pronoun": ("she/her", "he/him"), "age": ("18-29", "45+")}.items():
            for col in ("ap_l", "ap_m", "d"):
                x, y = p.loc[p[key] == ga, col], p.loc[p[key] == gb, col]
                r[f"gap_{key}_{col}"] = round(float(x.mean() - y.mean()), 4)
                r[f"gap_{key}_{col}_se"] = round(float(np.sqrt(x.var() / len(x) + y.var() / len(y))), 4)
        people.append(p.assign(size=size))
    args.out.mkdir(parents=True, exist_ok=True)
    old = os.umask(0o077)
    try:
        pd.DataFrame(summary).to_csv(args.out / "summary.csv", index=False)
        pd.concat(hists).to_csv(args.out / "cosine_hist.csv", index=False)
        pd.concat(cdf).to_csv(args.out / "nme_cdf.csv", index=False)
        pd.concat(people).to_csv(args.out / "people.csv", index=False)
    finally:
        os.umask(old)
    print(pd.DataFrame(summary).T.to_string())
    print(pd.concat(hists).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
