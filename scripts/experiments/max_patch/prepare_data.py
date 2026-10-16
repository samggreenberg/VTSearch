"""Stage 0: load + embed each (dataset, embedder) pair once; cache pickles, select categories.

For every embedder in the grid, each dataset is loaded and embedded through
``load_demo_dataset`` (which also attaches the patch side-channels -
``patch_grid`` - for patch-capable embedders), then the in-memory medias are
written to a per-(dataset, embedder) cell pickle (``_cells_io.dump_medias``), so
the three embedders don't evict each other from the single demo cache slot and
the patch fields the demo cache drops survive.  Array tasks load these warm
pickles directly.

Also selects each pair's categories (:func:`experiment_config.select_categories`)
and records them, with counts and timings, in ``prepare_info.json``, which the
runner enumerates its cells from.  The runner opens on the typed query's text
sort (or known-good examples), so nothing here crops or embeds an exemplar.

DINOv3 weights are gated on Hugging Face: ``HF_TOKEN`` must be set to a token
that has accepted the licence, or the dinov3 rows are skipped with a loud
warning.

Usage::

    python prepare_data.py            # full grid from experiment_config
    python prepare_data.py --datasets visual_genome_m --embedders dinov2_patch
"""

from __future__ import annotations

import argparse
import json
import os

import common

common.setup_env()

import experiment_config as cfg  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load + embed each (dataset, embedder) pair for the Max-Patch study.")
    parser.add_argument("--datasets", nargs="+", default=cfg.DATASETS)
    parser.add_argument("--embedders", nargs="+", default=cfg.EMBEDDERS)
    args = parser.parse_args(argv)

    from vtscore.config import EMBEDDINGS_DIR
    from vtscore.datasets.loader_demo import load_demo_dataset
    from vtscore.embedding import initialize_models
    from vtscore.embedding.media_vectors import media_embedding

    initialize_models()

    common.RESULTS.mkdir(parents=True, exist_ok=True)

    info_path = common.RESULTS / "prepare_info.json"
    info: dict = {"datasets": {}, "failed": []}
    if info_path.exists():
        # Re-runs (e.g. after adding an embedder) keep earlier entries.
        info = json.loads(info_path.read_text())

    for emb_name in args.embedders:
        if emb_name.startswith("dinov3") and not os.environ.get("HF_TOKEN"):
            common.log(f"SKIPPING {emb_name}: HF_TOKEN is not set (DINOv3 weights are licence-gated on HF).")
            info["failed"].append(emb_name)
            continue
        for ds in args.datasets:
            common.log(f"\n=== {ds} x {emb_name} ===")
            timings: dict[str, float] = {}
            medias: dict[int, dict] = {}
            try:
                with common.timed(f"load:{ds}:{emb_name}", timings):
                    load_demo_dataset(ds, medias, embedder_name=emb_name)
                    # load_demo_dataset attaches the CLS vector but not the
                    # patch-region side-channels; run the same patch back-fill
                    # the app runs at dataset creation so patch_grid /
                    # patch grids exist to persist (embed_missing is a no-op
                    # for the already-present CLS vectors and only does the
                    # patch pass for this patch-capable embedder).
                    from vtscore.datasets.stages.embedding import embed_missing  # noqa: PLC0415

                    embed_missing(medias, emb_name)
            except Exception as e:  # noqa: BLE001 - one bad pair must not lose the others
                import traceback

                common.log(f"FAILED to load {ds} with {emb_name}: {e}")
                traceback.print_exc()
                info["failed"].append(f"{ds}:{emb_name}")
                info_path.write_text(json.dumps(info, indent=2))
                continue

            # Serialize the in-memory medias (which carry patch_grid /
            # patch_grid / regions / categories) directly.  The demo *cache*
            # pickle drops those fields (see _cells_io), so a copy would leave
            # every style scoring on the whole-image vector alone.
            from _cells_io import dump_medias  # noqa: PLC0415

            dst = EMBEDDINGS_DIR / cfg.pickle_name(ds, emb_name)
            nbytes = dump_medias(medias, dst)
            common.log(f"  wrote cell pickle {dst.name}: {nbytes / 1e6:.0f} MB")

            cats: dict[str, int] = {}
            for m in medias.values():
                for c in m.get("categories") or [m.get("category")]:
                    if c:
                        cats[c] = cats.get(c, 0) + 1
            selected, sel_report = cfg.select_categories(medias, cats)
            dim = int(len(media_embedding(next(iter(medias.values()))))) if medias else None
            n_patch = sum(1 for m in medias.values() if m.get("patch_grid") is not None)
            common.log(
                f"{ds} x {emb_name}: {len(medias)} medias (patch grids on {n_patch}), dim={dim}, "
                f"{len(cats)} categories -> selected {len(selected)} by {sel_report.get('mode')}"
            )
            # Report the scale sample explicitly: an under-populated band is the
            # difference between testing the crossover and merely bracketing it.
            for band, info_b in (sel_report.get("bands") or {}).items():
                lo, hi = info_b["range"]
                common.log(
                    f"  band {band:14s} [{lo * 100:5.2f}%, {hi * 100:6.2f}%): "
                    f"{len(info_b['selected'])}/{info_b['target']} from {info_b['n_candidates']} candidates"
                    f"{'  ** UNDER-POPULATED **' if info_b['under_populated'] else ''} -> {info_b['selected']}"
                )
            for cat, area in sel_report.get("dropped_above_max_voted_area") or []:
                common.log(
                    f"  dropped {cat!r}: median voted box covers {area * 100:.0f}% of the image "
                    f"(> {cfg.MAX_VOTED_AREA * 100:.0f}% cap) - its region vote is an image-level vote"
                )

            info["datasets"].setdefault(ds, {})[emb_name] = {
                "n_medias": len(medias),
                "n_patch_grids": n_patch,
                "dim": dim,
                "category_counts": cats,
                "selected_categories": selected,
                "category_selection": sel_report,
                "load_seconds": timings.get(f"load:{ds}:{emb_name}"),
                "embed_seconds_per_image": round(timings.get(f"load:{ds}:{emb_name}", 0.0) / max(len(medias), 1), 4),
                "pickle": cfg.pickle_name(ds, emb_name),
            }
            info_path.write_text(json.dumps(info, indent=2))
            del medias

    common.log(f"\nWrote {info_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
