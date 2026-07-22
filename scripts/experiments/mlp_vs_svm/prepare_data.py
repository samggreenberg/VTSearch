"""Stage 0: load + embed each image dataset once, caching the SigLIP pickle.

``load_demo_dataset`` caches embeddings to ``EMBEDDINGS_DIR/<id>.pkl`` (under the
experiment ``VTSEARCH_DATA_DIR``), so running this once means every Stage A/B/C
array task loads from the warm pickle instead of re-embedding.  The SigLIP
embedder is forced explicitly (the experiment is single-embedder by design).

Usage::

    python prepare_data.py caltech101_m caltech256_a visual_genome_m [vggface2_faces_m]
"""

from __future__ import annotations

import argparse
import json

import common

common.setup_env()

EMBEDDER = "siglip"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load + embed image datasets for the MLP-vs-SVM study.")
    parser.add_argument("datasets", nargs="+", help="Demo dataset ids (e.g. caltech256_a visual_genome_m).")
    parser.add_argument("--embedder", default=EMBEDDER)
    args = parser.parse_args(argv)

    from vtscore.datasets.loader_demo import load_demo_dataset
    from vtscore.embedding import initialize_models

    initialize_models()

    info: dict[str, object] = {"embedder": args.embedder, "datasets": {}, "failed": []}
    for ds in args.datasets:
        timings: dict[str, float] = {}
        medias: dict[int, dict] = {}
        common.log(f"\n=== {ds} ===")
        try:
            with common.timed(f"load:{ds}", timings):
                load_demo_dataset(ds, medias, embedder_name=args.embedder)
        except Exception as e:  # noqa: BLE001 - one bad dataset must not lose the others
            import traceback

            common.log(f"FAILED to load {ds}: {e}")
            traceback.print_exc()
            info["failed"].append(ds)  # type: ignore[attr-defined]
            # Persist progress so a partial prepare still yields a usable info file.
            (common.RESULTS / "prepare_info.json").write_text(json.dumps(info, indent=2))
            continue
        n = len(medias)
        # Per-category prevalence (multi-label aware).
        cats: dict[str, int] = {}
        for m in medias.values():
            for c in m.get("categories") or [m.get("category")]:
                if c:
                    cats[c] = cats.get(c, 0) + 1
        dim = None
        if medias:
            from vtscore.embedding.media_vectors import media_embedding

            dim = int(len(media_embedding(next(iter(medias.values())))))
        common.log(f"{ds}: {n} medias, dim={dim}, {len(cats)} categories, took {timings.get(f'load:{ds}')}s")
        # Report the 8 easiest/hardest by prevalence so the config picker has data.
        by_prev = sorted(cats.items(), key=lambda kv: kv[1], reverse=True)
        common.log(f"  top categories by count: {by_prev[:8]}")
        info["datasets"][ds] = {  # type: ignore[index]
            "n_medias": n,
            "dim": dim,
            "n_categories": len(cats),
            "category_counts": cats,
            "load_seconds": timings.get(f"load:{ds}"),
        }
        # Sanity: at least one category has both positives and negatives.
        assert any(0 < c < n for c in cats.values()), f"{ds}: no usable category"
        del medias

    out = common.RESULTS / "prepare_info.json"
    out.write_text(json.dumps(info, indent=2))
    common.log(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
