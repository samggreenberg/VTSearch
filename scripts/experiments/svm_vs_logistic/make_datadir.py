"""Build #3197's private datadir: the pile's cells, with every vector unit-norm.

The app L2-normalises every embedding once, at ingest
(:mod:`vtscore.embedding.normalize`), so every detector it trains sees unit
vectors.  The calibration harness reads the pile's cell pickles directly
(``_cells_io.load_medias``), which bypasses that chokepoint - harmless for every
cell that was *written* unit-norm, and wrong for ``coco_val__siglip``, whose
vectors have norms of 12-19 (it was built from the #2790 region cache by
``build_coco_pickle.py``).  A head comparison is exactly where that matters: the
SVM's ``C`` and the logistic loop's weight decay both act on ``||w||``, and a
15x input scale moves both heads along their regularisation paths by different
amounts.

So this study never reads the pile in place.  It builds a datadir of **symlinks**
to the pile's cells and writes a normalised copy of any cell that is not already
unit-norm.  The pile itself is never written (it is shared with other sessions),
and a cell that is already unit-norm is linked, not copied, so the run reads the
same bytes every other study does.

    python make_datadir.py --out /expscratch/$USER/svmlog-3197/datadir
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "calibration"))

from _cells_io import dump_medias, load_medias  # noqa: E402

PILE = Path("/expscratch/sgreenberg/vts-cache/datadir/embeddings")
DATASETS = ("caltech101_m", "coco_val", "visual_genome_m")
EMBEDDERS = ("siglip", "siglip2_l")


def _norms(medias: dict, emb: str) -> np.ndarray:
    return np.array(
        [float(np.linalg.norm(np.asarray(m["embeddings"][emb], dtype=np.float64))) for m in medias.values()]
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--pile", type=Path, default=PILE)
    args = ap.parse_args()
    emb_dir = args.out / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for ds in DATASETS:
        for emb in EMBEDDERS:
            name = f"{ds}__{emb}.pkl"
            src = args.pile / name
            dst = emb_dir / name
            prov = args.pile / f"{ds}__{emb}.provenance.json"
            medias = load_medias(src)
            n = _norms(medias, emb)
            unit = bool(np.all(np.abs(n - 1.0) < 1e-3))
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if unit:
                dst.symlink_to(src)
                action = "linked"
            else:
                for m in medias.values():
                    v = np.asarray(m["embeddings"][emb], dtype=np.float32)
                    m["embeddings"][emb] = (v / np.linalg.norm(v)).astype(np.float32)
                dump_medias(medias, dst)
                n2 = _norms(load_medias(dst), emb)
                assert np.all(np.abs(n2 - 1.0) < 1e-3), f"{name}: normalised copy is not unit-norm"
                action = "normalised-copy"
            if prov.exists():
                pdst = emb_dir / prov.name
                if pdst.exists() or pdst.is_symlink():
                    pdst.unlink()
                pdst.symlink_to(prov)
            report[name] = {
                "action": action,
                "n_medias": len(medias),
                "pile_norm_min": round(float(n.min()), 3),
                "pile_norm_median": round(float(np.median(n)), 3),
                "pile_norm_max": round(float(n.max()), 3),
            }
            print(name, report[name], flush=True)
    (args.out / "datadir_report.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
