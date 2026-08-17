"""Build per-class review slates for VTSearch, to audit the ground truth itself.

``vg_scale`` takes its labels from COCO because VG's are not exhaustive
(``coco_anchor.py``: VG recall 0.76 over *C*, 1.4% of its negatives actually
positive). COCO is far better, but "far better" is not a number, and a report
that assumes zero residual error is making the same mistake #3156 opened with —
just one level up. So this builds a small, *stratified* slate per class that a
human can label in VTSearch, from which the residual rate can be estimated
rather than assumed.

**Three strata, recorded per image, because they answer different questions:**

* ``boundary`` — the highest-scoring images among the cell's **negatives**. A
  COCO miss, if there is one, hides here: an image that looks exactly like a bus
  image and is labelled as holding none. This stratum finds errors efficiently
  and is **not** an unbiased sample of anything.
* ``random`` — a uniform sample of the same negatives. This one *is* unbiased,
  so it is what bounds the residual false-negative rate. Without it, finding
  three misses in the boundary stratum says nothing about the pool.
* ``positive`` — a sample of the cell's positives, spread across the bands.
  Checks the other direction: is the object really there, and is the box (hence
  the band) the one a user would drag?

Ranking uses the text tower, not a trained detector: it needs no votes to exist
first, it costs one text embedding per class, and for "which negatives look like
buses" it is entirely adequate.

Each class becomes a folder of JPEGs plus a manifest, ready for VTSearch's
``server_folder`` importer. Vote Good/Bad, export with ``server_json_file``, and
feed the export to ``ingest_slate.py``.

Usage::

    python make_audit_slate.py --out /expscratch/$USER/vgscale-3156/slates
    python make_audit_slate.py --classes bus,clock --boundary 25 --random 20
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

import pile_config as pc

pc.setup_env()


def log(msg: str) -> None:
    print(f"[slate] {msg}", flush=True)


def _load_cell(pkl: Path) -> dict[int, dict]:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calibration"))
    from _cells_io import load_medias  # noqa: PLC0415

    return load_medias(pkl)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", default=str(pc.EMBEDDINGS / "vg_scale__siglip.pkl"), help="scored pickle")
    ap.add_argument("--embedder", default="siglip", help="embedder the cell carries (for the text tower)")
    ap.add_argument("--out", default=str(pc.PILE / "slates"), help="output directory")
    ap.add_argument("--classes", default="", help="comma-separated subset of C (default: all)")
    ap.add_argument("--boundary", type=int, default=15, help="top-scoring negatives per class")
    ap.add_argument("--random", dest="n_random", type=int, default=10, help="uniform negatives per class")
    ap.add_argument("--positive", type=int, default=5, help="positives per class, spread over the bands")
    ap.add_argument("--seed", type=int, default=20260817, help="sampling seed")
    args = ap.parse_args()

    from vtscore.embedding import embed_text_query  # noqa: PLC0415
    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415
    from vtscore.eval.labels import media_is_positive  # noqa: PLC0415

    classes = [c.strip() for c in args.classes.split(",") if c.strip()] or list(pc.SCALE_CLASSES)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    medias = _load_cell(Path(args.cell))
    log(f"loaded {len(medias)} medias from {Path(args.cell).name}")

    # The VG file for an id — the cell carries vectors, not pixels.
    from build_pile import _vg_image_paths  # noqa: PLC0415

    paths = _vg_image_paths()

    ids = sorted(medias)
    mat = np.stack([np.asarray(media_embedding(medias[i]), dtype=np.float32) for i in ids])
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12

    rng = random.Random(args.seed)
    index: list[dict] = []
    for c in classes:
        cells = [pc.scale_cell(c, b) for b in pc.BOX_BANDS]
        # Negatives for this class are the shared pool: images COCO annotated
        # and found none of C in. They are the same images for every class,
        # which is what makes the classes comparable.
        negatives = [i for i in ids if not medias[i]["categories"]]
        positives = {cell: [i for i in ids if media_is_positive(medias[i], cell)] for cell in cells}

        tvec = embed_text_query(c, "image", embedder_name=args.embedder)
        if tvec is None:
            raise SystemExit(f"no text tower for embedder {args.embedder!r}")
        tvec = np.asarray(tvec, dtype=np.float32)
        tvec /= np.linalg.norm(tvec) + 1e-12
        pos_of = {i: n for n, i in enumerate(ids)}
        scores = {i: float(mat[pos_of[i]] @ tvec) for i in negatives}

        ranked = sorted(negatives, key=lambda i: -scores[i])
        boundary = ranked[: args.boundary]
        rest = [i for i in ranked[args.boundary :]]
        uniform = rng.sample(rest, min(args.n_random, len(rest)))
        # Spread the positive check over the bands rather than taking whichever
        # band happens to sort first: the box question is different at each size.
        per_band = max(1, args.positive // len(cells))
        chosen_pos: list[tuple[int, str]] = []
        for cell in cells:
            for i in rng.sample(positives[cell], min(per_band, len(positives[cell]))):
                chosen_pos.append((i, cell))

        cdir = out_root / c.replace(" ", "_")
        cdir.mkdir(parents=True, exist_ok=True)
        rows = []
        for stratum, items in (
            ("boundary", [(i, "") for i in boundary]),
            ("random", [(i, "") for i in uniform]),
            ("positive", chosen_pos),
        ):
            for i, cell in items:
                src = paths.get(i)
                if src is None:
                    continue
                (cdir / f"{i}.jpg").write_bytes(src.read_bytes())
                rows.append(
                    {
                        "image_id": i,
                        "class": c,
                        "stratum": stratum,
                        "cell": cell,
                        "text_score": round(scores.get(i, float("nan")), 4),
                        "coco_says": "present" if cell else "absent",
                        "n_boxes": len(medias[i].get("regions") or []) if cell else 0,
                    }
                )
        with (cdir / "manifest.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        index.append({"class": c, "dir": str(cdir), "n": len(rows)})
        log(f"  {c:<12} {len(rows):3d} images -> {cdir}")

    (out_root / "slates.json").write_text(json.dumps(index, indent=1) + "\n")
    total = sum(e["n"] for e in index)
    print(f"\n{total} images across {len(index)} classes under {out_root}")
    print("\nIn VTSearch, per class: Add Dataset -> Server Folder -> the path above,")
    print("vote Good (drag a box) / Bad, then export with Server JSON File and run:")
    print(f"  python ingest_slate.py --export <exported.json> --slates {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
