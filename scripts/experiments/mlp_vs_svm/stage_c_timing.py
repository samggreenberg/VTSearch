"""Stage C: GPU train/inference scaling microbenchmark (torch MLP vs cuML SVM).

Independent of Stage B.  Samples real SigLIP vectors from ``caltech256_a`` +
``places365_m`` as the feature source, then measures how each trainer's fit and
inference time scale with size (median-of-7, warmup-discarded, CUDA-synced).
Also records the sklearn↔cuML score-parity correlation so the report can state
whether the GPU and CPU SVMs describe the same model.  Writes ``stage_c.csv`` and
``stage_c_parity.json``.
"""

from __future__ import annotations

import argparse
import json

import common

common.setup_env()

SOURCE_DATASETS = ["caltech256_a", "places365_m"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage C: GPU timing microbenchmark.")
    parser.add_argument("--source-datasets", nargs="+", default=SOURCE_DATASETS)
    parser.add_argument("--output", default=str(common.RESULTS / "stage_c.csv"))
    parser.add_argument("--max-source", type=int, default=20000, help="Cap on pooled source vectors.")
    args = parser.parse_args(argv)

    import numpy as np

    from vtscore.datasets.loader_demo import load_demo_dataset
    from vtscore.embedding import initialize_models
    from vtscore.embedding.media_vectors import media_embedding
    from vtscore.eval.timing_benchmark import run_timing_benchmark, svm_backend_parity

    import experiment_config as cfg

    initialize_models()

    vecs: list[np.ndarray] = []
    for ds in args.source_datasets:
        medias: dict[int, dict] = {}
        try:
            load_demo_dataset(ds, medias, embedder_name=cfg.EMBEDDER)
        except Exception as e:  # noqa: BLE001 - a missing source dataset shouldn't abort Stage C
            common.log(f"WARN: could not load {ds} for timing source ({e}); skipping")
            continue
        for m in medias.values():
            vecs.append(np.asarray(media_embedding(m), dtype=np.float32))
        common.log(f"pooled {len(vecs)} vectors after {ds}")
        if len(vecs) >= args.max_source:
            break
    x_source = np.stack(vecs[: args.max_source]) if vecs else None
    dim = int(x_source.shape[1]) if x_source is not None else 768
    common.log(f"source pool: {0 if x_source is None else len(x_source)} vectors, dim={dim}")

    common.log("running scaling benchmark ...")
    df = run_timing_benchmark(
        trainers=["mlp", "svm_linear", "svm_rbf"],
        dim=dim,
        svm_backend="auto",
        x_source=x_source,
        progress=True,
    )
    df.to_csv(args.output, index=False)
    common.log(f"wrote {len(df)} rows to {args.output}")

    parity = {
        "svm_rbf": svm_backend_parity("svm_rbf", n_train=256, dim=dim),
        "svm_linear": svm_backend_parity("svm_linear", n_train=256, dim=dim),
    }
    (common.RESULTS / "stage_c_parity.json").write_text(json.dumps(parity, indent=2))
    common.log(f"sklearn<->cuML parity (Spearman): {parity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
