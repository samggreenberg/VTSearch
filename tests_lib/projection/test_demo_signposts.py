"""Library-tier tests for the ground-truth signpost builder + synthetic demo.

Covers the "cheating" signpost path (``vtscore.projection.demo_signposts``) and
the synthetic world-map demo data (``vtscore.media._toponymy_demo``): category
path cleaning, the hierarchical build (levels, anchors, layout pinning), the
hierarchical-category probe, and the demo generator's determinism + clusterable
structure.  No Flask, no models — pure numpy.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.projection.demo_signposts import (
    build_category_signposts,
    clean_category_path,
    has_hierarchical_categories,
)
from vtscore.projection.umap_projection import Projection


def _proj(ids, coords):
    return Projection("proj-xyz", list(ids), np.asarray(coords, dtype=np.float32), "pca")


class TestCleanCategoryPath:
    def test_splits_synthetic_path(self):
        assert clean_category_path("Europe/France/Île-de-France/Paris") == [
            "Europe",
            "France",
            "Île-de-France",
            "Paris",
        ]

    def test_drops_leading_single_char_index_segment(self):
        # Places365-style "/a/arena/hockey": leading empty + alpha bucket dropped.
        assert clean_category_path("/a/arena/hockey") == ["arena", "hockey"]

    def test_trims_and_drops_empties(self):
        assert clean_category_path("Asia/ / Japan /") == ["Asia", "Japan"]

    @pytest.mark.parametrize("bad", [None, "", 123, "   "])
    def test_missing_or_blank_is_empty(self, bad):
        assert clean_category_path(bad) == []

    def test_flat_category_is_single_segment(self):
        assert clean_category_path("dog") == ["dog"]


class TestHasHierarchicalCategories:
    def test_true_when_majority_are_paths(self):
        medias = {i: {"category": "A/B/C"} for i in range(10)}
        assert has_hierarchical_categories(medias)

    def test_false_for_flat_categories(self):
        medias = {i: {"category": "dog"} for i in range(10)}
        assert not has_hierarchical_categories(medias)

    def test_false_for_empty(self):
        assert not has_hierarchical_categories({})

    def test_false_when_only_a_few_are_paths(self):
        medias = {i: {"category": "dog"} for i in range(9)}
        medias[99] = {"category": "A/B"}
        assert not has_hierarchical_categories(medias)


class TestBuildCategorySignposts:
    def _hierarchical_medias(self):
        # Two continents, two countries each, two cities each; 3 items/city.
        # Globally unique names so distinct-prefix count == distinct-text count.
        paths = []
        n = 0
        for cont in ("Europe", "Asia"):
            for country in (f"{cont}Co1", f"{cont}Co2"):
                for city in (f"{country}City1", f"{country}City2"):
                    paths.extend([f"{cont}/{country}/{city}"] * 3)
                    n += 1
        medias = {i + 1: {"category": p} for i, p in enumerate(paths)}
        ids = list(medias.keys())
        # Coordinates don't matter for structure, only for anchor placement.
        rng = np.random.default_rng(0)
        coords = rng.standard_normal((len(ids), 2))
        return medias, _proj(ids, coords)

    def test_one_sign_per_distinct_prefix_across_levels(self):
        medias, proj = self._hierarchical_medias()
        ls = build_category_signposts(proj, medias)
        count_by_level = {}
        for lab in ls.labels:
            count_by_level.setdefault(round(lab.level, 2), 0)
            count_by_level[round(lab.level, 2)] += 1
        levels = sorted(count_by_level)
        assert len(levels) == 3  # continent / country / city
        # 2 continents, 4 countries, 8 cities — one sign per distinct prefix.
        assert count_by_level[levels[0]] == 2
        assert count_by_level[levels[1]] == 4
        assert count_by_level[levels[2]] == 8
        assert {lab.text for lab in ls.labels if lab.level == levels[0]} == {"Europe", "Asia"}

    def test_coarsest_level_is_zero_and_increases(self):
        medias, proj = self._hierarchical_medias()
        ls = build_category_signposts(proj, medias)
        levels = sorted({round(lab.level, 2) for lab in ls.labels})
        assert levels[0] == 0.0
        assert levels == sorted(levels)
        assert all(b > a for a, b in zip(levels, levels[1:]))

    def test_anchor_is_a_member_point(self):
        # Give one city widely separated points; its anchor must be one of them.
        medias = {1: {"category": "X/Y/Z"}, 2: {"category": "X/Y/Z"}, 3: {"category": "X/Y/Z"}}
        coords = [[0.0, 0.0], [10.0, 0.0], [5.0, 0.0]]
        proj = _proj([1, 2, 3], coords)
        ls = build_category_signposts(proj, medias)
        city = [lab for lab in ls.labels if lab.text == "Z"][0]
        assert (city.x, city.y) in {(0.0, 0.0), (10.0, 0.0), (5.0, 0.0)}

    def test_pins_to_projection_id(self):
        medias, proj = self._hierarchical_medias()
        ls = build_category_signposts(proj, medias)
        assert ls.projection_id == "proj-xyz"

    def test_prunes_singletons(self):
        # A city with a single item earns no sign (below _MIN_MEMBERS).
        medias = {1: {"category": "A/B/Solo"}, 2: {"category": "A/B/Pair"}, 3: {"category": "A/B/Pair"}}
        proj = _proj([1, 2, 3], [[0, 0], [1, 1], [2, 2]])
        ls = build_category_signposts(proj, medias)
        city_texts = {lab.text for lab in ls.labels if round(lab.level, 2) == max(round(x.level, 2) for x in ls.labels)}
        assert "Solo" not in city_texts
        assert "Pair" in city_texts

    def test_empty_projection_returns_empty_pinned_set(self):
        proj = _proj([], np.zeros((0, 2)))
        ls = build_category_signposts(proj, {})
        assert ls.projection_id == "proj-xyz"
        assert ls.labels == ()

    def test_flat_categories_produce_one_level(self):
        medias = {i + 1: {"category": "dog"} for i in range(4)}
        proj = _proj(list(medias.keys()), np.zeros((4, 2)))
        ls = build_category_signposts(proj, medias)
        assert {round(lab.level, 2) for lab in ls.labels} == {0.0}
        assert {lab.text for lab in ls.labels} == {"dog"}


class TestSyntheticToponymyDemo:
    def test_generator_is_deterministic(self):
        from vtscore.media._toponymy_demo import generate_items

        a = generate_items(dim=16, items_per_city=4)
        b = generate_items(dim=16, items_per_city=4)
        assert len(a) == len(b) == 108 * 4
        assert np.array_equal(a[0].embedding, b[0].embedding)
        assert a[10].category == b[10].category

    def test_embeddings_are_unit_norm(self):
        from vtscore.media._toponymy_demo import generate_items

        items = generate_items(dim=32, items_per_city=3)
        norms = np.linalg.norm(np.stack([it.embedding for it in items]), axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_categories_are_four_level_paths(self):
        from vtscore.media._toponymy_demo import generate_items

        items = generate_items(dim=16, items_per_city=2)
        assert all(len(clean_category_path(it.category)) == 4 for it in items)

    def test_within_city_tighter_than_across_continents(self):
        # The baked structure must nest: siblings in one city sit far closer than
        # items from different continents, so UMAP can recover the hierarchy.
        from vtscore.media._toponymy_demo import generate_items

        items = generate_items(dim=64, items_per_city=6)
        by_cat: dict[str, list] = {}
        for it in items:
            by_cat.setdefault(it.category, []).append(it.embedding)
        first_city = next(iter(by_cat.values()))
        within = np.linalg.norm(first_city[0] - first_city[1])
        europe = next(it.embedding for it in items if it.category.startswith("Europe/"))
        asia = next(it.embedding for it in items if it.category.startswith("Asia/"))
        across = np.linalg.norm(europe - asia)
        assert within < across
