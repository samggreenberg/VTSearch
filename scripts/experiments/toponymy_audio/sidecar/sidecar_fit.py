"""Sidecar half of the fallback architecture: toponymy in its OWN venv.

Runs the full Toponymy fit in a venv where toponymy was installed normally
(its own transformers 4.x pin, no torch), reading embeddings/texts from
files and encoding keyphrase strings via the app's text branch over
localhost HTTP (see clap_server.py). Proves the conflict can be fenced off
entirely if the in-venv route is ever blocked.

Usage::  sidecar_fit.py <dataset> <variant> [--server http://127.0.0.1:8763]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402

import numpy as np  # noqa: E402

from run_toponymy import CallStats, make_keyphrase_namer  # noqa: E402


class RemoteTextEncoder:
    """TextEmbedderProtocol adapter that calls the app process over HTTP."""

    def __init__(self, server: str):
        self.server = server
        self.calls = 0
        self.texts_encoded = 0

    def encode(self, texts, show_progress_bar=False, **kwargs):
        import httpx  # a toponymy dependency, present in the sidecar venv

        texts = [str(t) for t in texts]
        resp = httpx.post(self.server + "/encode", json={"texts": texts}, timeout=600)
        resp.raise_for_status()
        vecs = resp.json()["vectors"]
        self.calls += 1
        self.texts_encoded += len(texts)
        return np.asarray(vecs, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("variant")
    ap.add_argument("--server", default="http://127.0.0.1:8763")
    args = ap.parse_args()

    out = common.ds_dir(args.dataset)
    emb = np.load(out / "embeddings_clap.npy").astype(np.float32)
    texts = [str(t) for t in common.load_json(out / f"texts_{args.variant}.json")]
    n = min(len(texts), len(emb))
    emb, texts = emb[:n], texts[:n]

    import transformers
    import umap
    import toponymy as topo_pkg
    from toponymy import Toponymy
    from toponymy.clustering import ToponymyClusterer

    print(
        f"sidecar venv: toponymy from {Path(topo_pkg.__file__).parents[1]}, "
        f"transformers {transformers.__version__}, torch "
        f"{'ABSENT' if 'torch' not in sys.modules and not _has('torch') else 'present'}"
    )

    timings: dict = {}
    with common.timed("umap_5d", timings):
        umap5 = umap.UMAP(n_components=5, metric="cosine", random_state=42).fit_transform(emb)

    stats = CallStats()
    encoder = RemoteTextEncoder(args.server)
    model = Toponymy(
        llm_wrapper=make_keyphrase_namer(stats),
        text_embedding_model=encoder,
        clusterer=ToponymyClusterer(min_clusters=4, verbose=True),
        object_description="audio clips",
        corpus_description=f"a collection of short audio clips ({args.dataset})",
        verbose=True,
    )
    t0 = time.time()
    with common.timed("toponymy_fit", timings):
        model.fit(texts, emb, umap5.astype(np.float32))

    layers = [{"layer": i, "n_topics": len(model.topic_names_[i])} for i in range(len(model.cluster_layers_))]
    common.save_json(
        out / f"sidecar_{args.variant}_keyphrase.json",
        {
            "layers": layers,
            "topic_names_l1": list(model.topic_names_[1]) if len(layers) > 1 else [],
            "timings_s": timings,
            "rpc_calls": encoder.calls,
            "rpc_texts": encoder.texts_encoded,
            "namer_calls": stats.calls,
            "wall_fit_s": round(time.time() - t0, 1),
        },
    )


def _has(mod: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(mod) is not None


if __name__ == "__main__":
    main()
