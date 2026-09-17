"""Is a query crop the same pixels as its page, at the same scale? (#3912)

For the Tobacco800 classes whose crop would not verify against its own page
(plus three that do), compare the crop with a re-cut of the box from the raw
page, the bounded decode ``sift_vlad`` reads, and a side-by-side panel of crop,
page and decoded box region.

    source scripts/experiments/pile/pile_env.sh
    python scripts/experiments/docmarks/diag_crop_page.py <out-dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1])
sys.path.insert(0, str(HERE))

import docmarks_config as cfg  # noqa: E402
from sources._common import read_manifest  # noqa: E402

BAD = [
    "tobacco800/logo_azb11c00_1",
    "tobacco800/logo_bqz95d00_1",
    "tobacco800/logo_cgr96c00_1",
    "tobacco800/logo_asg54f00_1",
    "tobacco800/logo_afm90c00-first_1_0",
]
GOOD = ["tobacco800/logo_ajj10e00_1", "tobacco800/logo_ciy01a00-page02_1_0", "spods/logo_00003_0"]


def main() -> int:
    from vtscore.media import get_embedder
    from vtscore.media.image.decode import decode_bounded_rgb

    OUT.mkdir(parents=True, exist_ok=True)
    classes = json.loads((cfg.OUT / "classes.json").read_text())
    wanted = {classes[c]["query_page_id"]: c for c in BAD + GOOD}
    pages = {p.page_id: p for p in read_manifest(cfg.OUT / "corpus.jsonl") if p.page_id in wanted}
    emb = get_embedder("sift_vlad")
    emb.load_models()
    m = emb.structural_matcher
    rows = []
    for cid in BAD + GOOD:
        meta = classes[cid]
        page = pages[meta["query_page_id"]]
        marks = [mk for mk in page.marks if mk.class_id == cid and mk.area() > 0]
        mark = max(marks, key=lambda mk: mk.area())
        x, y, w, h = mark.box
        with Image.open(page.path) as raw:
            raw_mode, raw_size, raw_info = raw.mode, raw.size, {k: raw.info.get(k) for k in ("dpi", "compression")}
            recut = raw.convert("RGB").crop((x, y, x + w, y + h))
        crop = Image.open(meta["query_crop"])
        crop_same = crop.size == recut.size and np.array_equal(np.asarray(crop.convert("RGB")), np.asarray(recut))
        dec, scale = decode_bounded_rgb(Path(page.path))
        gray_page = np.asarray(dec.convert("L"))
        gray_crop = np.asarray(crop.convert("L"))
        uniq_page = len(np.unique(gray_page))
        uniq_crop = len(np.unique(gray_crop))
        crop_f = m.detect_and_describe(gray_crop, max_features=emb.max_features)
        # the decoded page's own region, at the decoded scale
        sx0, sy0, sx1, sy1 = [int(round(v * scale)) for v in (x, y, x + w, y + h)]
        region = gray_page[sy0:sy1, sx0:sx1]
        res = {}
        for budget in (1024, 16384):
            pf = m.detect_and_describe(gray_page, max_features=budget)
            # Columns 0-1 are x, y normalised to [0, 1]; 2-3 are size and angle.
            xy = pf.keypoints_f32()[:, :2]
            h_px, w_px = gray_page.shape
            inside = int(
                (
                    (xy[:, 0] >= sx0 / w_px)
                    & (xy[:, 0] <= sx1 / w_px)
                    & (xy[:, 1] >= sy0 / h_px)
                    & (xy[:, 1] <= sy1 / h_px)
                ).sum()
            )
            s = m.verify(crop_f, pf)
            res[budget] = {
                "page_kp": pf.count,
                "kp_in_box": inside,
                "inliers": s.inlier_count,
                "scale": round(s.scale, 4),
                "model_ok": bool(s.model_ok),
            }
        reg_f = m.detect_and_describe(region, max_features=16384) if region.size else None
        s_reg = m.verify(crop_f, reg_f) if reg_f is not None and reg_f.count else None
        s_self = m.verify(crop_f, crop_f)
        row = {
            "class_id": cid,
            "bad": cid in BAD,
            "page": page.page_id,
            "path": page.path,
            "raw_mode": raw_mode,
            "raw_size": raw_size,
            "raw_info": {k: str(v) for k, v in raw_info.items()},
            "box": [x, y, w, h],
            "crop_size": crop.size,
            "crop_mode": crop.mode,
            "crop_equals_recut": bool(crop_same),
            "decoded_size": dec.size,
            "scale": round(scale, 4),
            "box_px_decoded": [sx1 - sx0, sy1 - sy0],
            "unique_gray_page": uniq_page,
            "unique_gray_crop": uniq_crop,
            "crop_kp": crop_f.count,
            "budget": res,
            "crop_vs_decoded_region_inliers": (s_reg.inlier_count if s_reg and s_reg.model_ok else 0),
            "crop_vs_itself_inliers": (s_self.inlier_count if s_self.model_ok else 0),
            "other_marks_on_page": [(mk.kind, mk.class_id, mk.box) for mk in page.marks if mk is not mark],
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        # figure panel: crop | page with box | decoded region
        panel_h = 260

        def fit(im):
            im = im.convert("RGB")
            r = panel_h / im.height
            return im.resize((max(1, int(im.width * r)), panel_h))

        pg = dec.copy()
        d = ImageDraw.Draw(pg)
        d.rectangle((sx0, sy0, sx1, sy1), outline=(255, 0, 0), width=max(3, int(8 * scale)))
        parts = [fit(crop), fit(pg), fit(Image.fromarray(region)) if region.size else Image.new("RGB", (10, panel_h))]
        W = sum(p.width for p in parts) + 20
        canvas = Image.new("RGB", (W, panel_h + 18), "white")
        xx = 0
        for p in parts:
            canvas.paste(p, (xx, 18))
            xx += p.width + 10
        ImageDraw.Draw(canvas).text(
            (2, 2),
            f"{cid}  crop {crop.size} {crop.mode} | page {raw_size} {raw_mode} scale {scale:.2f} | decoded box region",
            fill=(0, 0, 0),
        )
        canvas.save(OUT / f"{cid.replace('/', '__')}.png")
    (OUT / "diag_crops.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
