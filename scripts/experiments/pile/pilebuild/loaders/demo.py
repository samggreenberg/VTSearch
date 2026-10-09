"""VTSearch demo datasets, staged in the shared demo cache and loaded by vtscore."""

from __future__ import annotations

import pile_config as pc


def load(dataset: str, medias: dict[int, dict], embedder_name: str) -> None:
    from vtscore.datasets.loader_demo import load_demo_dataset  # noqa: PLC0415

    pc.require_demo_source(dataset)
    # Never through the app's `<dataset>.pkl` cache in EMBEDDINGS_DIR, which is
    # the pile's own directory. A hit there reloaded every VG media with no
    # labels (#4117), and even with that fixed it hands back vectors embedded by
    # whichever earlier build wrote the cache -- under a provenance sidecar that
    # records THIS build's code, device and batch size. A pile build reads the
    # staged source and embeds; `--force` means exactly that.
    load_demo_dataset(dataset, medias, embedder_name=embedder_name, use_cache=False)


def check(dataset: str) -> str:
    pc.require_demo_source(dataset)
    return "demo source staged"
