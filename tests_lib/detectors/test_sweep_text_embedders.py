"""``sweep.TEXT_EMBEDDERS`` must stay in sync with the embedder registry.

The sweeps decide whether a text query can seed the cold start by testing the embedder
name against a hard-coded set. A text-capable embedder missing from that set does not
error: ``query_vec`` stays ``None`` and the run silently falls back to example-seeding,
cosine against ONE randomly drawn training positive *per seed*. The damage is quiet and
easy to misread as a result - cold-start rows differ across seeds, and ``--query`` is
accepted then ignored. That is exactly how ``siglip2_l`` was mistaken for a model-level
difference from ``siglip``.

So pin the set against the live registry. When someone adds a text-capable image
embedder, this fails and names it, rather than the next sweep quietly measuring the
wrong cold start.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "sod"))


def _registry_text_image_embedders() -> set[str]:
    from vtscore.media import all_embedders

    return {
        e.name
        for e in all_embedders()
        if getattr(e, "media_type_id", "") == "image" and getattr(e, "supports_text", False)
    }


class TestTextEmbedderSet:
    def test_matches_the_registry(self):
        import sweep

        registry = _registry_text_image_embedders()
        declared = set(sweep.TEXT_EMBEDDERS)
        missing = registry - declared
        extra = declared - registry
        assert not missing, (
            f"text-capable image embedders absent from sweep.TEXT_EMBEDDERS: {sorted(missing)}. "
            "Runs naming them silently example-seed the cold start and ignore --query."
        )
        assert not extra, f"sweep.TEXT_EMBEDDERS names non-text-capable embedders: {sorted(extra)}"

    def test_includes_the_large_variants(self):
        # Regression: the `_l` checkpoints were text-capable but unlisted, so a
        # `--embedders siglip2_l --query ...` run seeded from a random positive instead.
        import sweep

        assert {"siglip_l", "siglip2_l"} <= set(sweep.TEXT_EMBEDDERS)

    def test_sweep_train_test_reuses_the_same_set(self):
        # Not a copy: a second literal would drift independently and reintroduce the bug
        # on one script only.
        import sweep
        import sweep_train_test

        assert sweep_train_test.TEXT_EMBEDDERS is sweep.TEXT_EMBEDDERS
