"""VTSearch demo datasets, staged in the shared demo cache and loaded by vtscore."""

from __future__ import annotations

import pile_config as pc


def load(dataset: str, medias: dict[int, dict], embedder_name: str) -> None:
    from vtscore.datasets.loader_demo import load_demo_dataset  # noqa: PLC0415

    pc.require_demo_source(dataset)
    load_demo_dataset(dataset, medias, embedder_name=embedder_name)


def check(dataset: str) -> str:
    pc.require_demo_source(dataset)
    return "demo source staged"
