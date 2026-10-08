"""What Stage 2 costs per vote at a document budget (#3928, M4).

Times ``SiftMatcher.verify_many`` -- the call ``structural_rerank`` makes once per
template -- over a shortlist of *K* pages at 8,192 keypoints, for the kinds of template
a detector carries: a query crop and Good-box templates (``filter_features_to_box``).
Candidates are stored the way the app stores them (``StructuralFeatures.compact``).

Two regimes:

* **cold:** every template against all *K* pages -- a detector's first re-rank, or a
  re-rank with no verification cache;
* **per vote with a cache:** one new template against *K*, plus the pages new to the
  shortlist against every template (the design's incremental cache, #3928 plan §5).

    python bench_verify.py --tier s --k 1000 --out <dir>/bench.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fullmarks_config as cfg  # noqa: E402
import embed_corpus  # noqa: E402
import template_matrix as tm  # noqa: E402

BUDGET = 8192


def _extract(path: str) -> Any:
    from eval_splg_rank import _gray  # noqa: PLC0415
    from vtscore.media.structural import SiftMatcher  # noqa: PLC0415

    return SiftMatcher().detect_and_describe(_gray(path), max_features=BUDGET).compact()


def _timed(matcher: Any, template: Any, cands: list[Any], reps: int = 2) -> float:
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        matcher.verify_many(template, cands)
        best = min(best, time.perf_counter() - t0)
    return best


def main(argv: Optional[Sequence[str]] = None) -> int:
    import torch  # noqa: PLC0415

    from vtscore.media.structural import SiftMatcher  # noqa: PLC0415
    from vtscore.training.structural_similarity import filter_features_to_box  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--tier", default="s")
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--templates", type=int, default=6, help="Good-box templates to time")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    pages = embed_corpus.pages_for_tier(args.corpus, args.tier)
    classes = json.loads((args.corpus / "classes.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(0)
    cand_pages = [pages[i] for i in rng.choice(len(pages), size=args.k, replace=False)]
    boxed = [(p, m.class_id) for p in pages for m in p.marks if m.box and classes.get(m.class_id, {}).get("on_roster")]
    tpl_pages = [boxed[i] for i in rng.choice(len(boxed), size=args.templates, replace=False)]

    t0 = time.time()
    with get_context("fork").Pool(args.workers) as pool:
        cands = pool.map(_extract, [p.path for p in cand_pages], chunksize=8)
        tpl_feats = pool.map(_extract, [p.path for p, _ in tpl_pages])
    extract_s = time.time() - t0
    templates = [filter_features_to_box(f, tm.largest_box(p, cid)) for f, (p, cid) in zip(tpl_feats, tpl_pages)]
    crop_cid = tpl_pages[0][1]
    crop = _extract(classes[crop_cid]["query_crop"])

    matcher = SiftMatcher()
    report: dict[str, Any] = {
        "k": args.k,
        "budget": BUDGET,
        "cpus": args.workers,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "extract_s": round(extract_s, 1),
        "median_candidate_keypoints": int(np.median([c.count for c in cands])),
        "templates": [],
    }
    matcher.verify_many(crop, cands[:20])  # warm-up (CUDA context, kernels)
    for name, tpl in [("crop", crop), *[(f"box{i}", t) for i, t in enumerate(templates)]]:
        s = _timed(matcher, tpl, cands)
        report["templates"].append({"name": name, "keypoints": int(tpl.count), "seconds": round(s, 2)})
        print(f"  {name}: {tpl.count} kp -> {s:.2f} s for {args.k} pages", flush=True)
    per_tpl = [t["seconds"] for t in report["templates"]]
    report["median_s_per_template"] = float(np.median(per_tpl))
    # The ratio-test half alone, to split matching from RANSAC.
    from vtscore.media.structural import ratio_test_matches  # noqa: PLC0415

    t1 = time.perf_counter()
    ratio_test_matches(crop.descriptors_f32(), [c.descriptors_f32() for c in cands], ratio=0.75)
    report["crop_ratio_test_s"] = round(time.perf_counter() - t1, 2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "templates"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
