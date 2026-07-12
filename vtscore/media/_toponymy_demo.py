"""Synthetic "world map" demo data for exercising the VTSBrowse signpost layer.

A friction-free stand-in for a real Toponymy run: a hand-authored 4-level
geographic taxonomy (**Continent → Country → State → City**) whose items ship
**pre-baked hierarchical embeddings** — no model, no download, no GPU.  UMAP
recovers the nested clusters, the ground-truth signpost builder
(:mod:`vtscore.projection.demo_signposts`) letters them straight from the
``category`` paths, and the browse canvas hands "Europe" off to "France" off to
"Île-de-France" off to "Paris" as you zoom — the toponymy metaphor made
literal, so the *display* can be evaluated hands-on before the real naming
pipeline exists.

This module is media-type-agnostic: it produces the taxonomy, the per-item
category paths, and the synthetic vectors.  The audio and image media types
wrap each item in their own media bytes (a tone / a colour tile) and thumbnail.
Library tier: numpy + stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The demo ``source`` id both media types dispatch on in ``load_demo_source``.
SOURCE_ID = "synthetic_toponymy"

#: Continent → Country → State → City.  Real-ish place names so the map reads
#: like an actual atlas; 4 continents × 3 × 3 × 3 = 108 leaf cities.
TAXONOMY: dict[str, dict[str, dict[str, list[str]]]] = {
    "Europe": {
        "France": {
            "Île-de-France": ["Paris", "Versailles", "Meaux"],
            "Provence": ["Marseille", "Nice", "Avignon"],
            "Brittany": ["Rennes", "Brest", "Vannes"],
        },
        "Italy": {
            "Lazio": ["Rome", "Tivoli", "Anzio"],
            "Tuscany": ["Florence", "Siena", "Pisa"],
            "Lombardy": ["Milan", "Bergamo", "Como"],
        },
        "Germany": {
            "Bavaria": ["Munich", "Nuremberg", "Augsburg"],
            "Saxony": ["Dresden", "Leipzig", "Chemnitz"],
            "Hesse": ["Frankfurt", "Wiesbaden", "Kassel"],
        },
    },
    "Asia": {
        "Japan": {
            "Kanto": ["Tokyo", "Yokohama", "Chiba"],
            "Kansai": ["Osaka", "Kyoto", "Kobe"],
            "Hokkaido": ["Sapporo", "Hakodate", "Asahikawa"],
        },
        "India": {
            "Maharashtra": ["Mumbai", "Pune", "Nagpur"],
            "Tamil Nadu": ["Chennai", "Madurai", "Coimbatore"],
            "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur"],
        },
        "China": {
            "Guangdong": ["Guangzhou", "Shenzhen", "Foshan"],
            "Sichuan": ["Chengdu", "Mianyang", "Leshan"],
            "Zhejiang": ["Hangzhou", "Ningbo", "Wenzhou"],
        },
    },
    "Africa": {
        "Egypt": {
            "Cairo": ["Cairo", "Giza", "Helwan"],
            "Alexandria": ["Alexandria", "Borg El Arab", "Abu Qir"],
            "Luxor": ["Luxor", "Karnak", "Esna"],
        },
        "Kenya": {
            "Nairobi": ["Nairobi", "Karen", "Ruiru"],
            "Coast": ["Mombasa", "Malindi", "Lamu"],
            "Rift Valley": ["Nakuru", "Eldoret", "Naivasha"],
        },
        "Nigeria": {
            "Lagos": ["Lagos", "Ikeja", "Badagry"],
            "Kano": ["Kano", "Wudil", "Gaya"],
            "Rivers": ["Port Harcourt", "Bonny", "Okrika"],
        },
    },
    "Americas": {
        "United States": {
            "California": ["Los Angeles", "San Francisco", "San Diego"],
            "Texas": ["Houston", "Austin", "Dallas"],
            "New York": ["New York City", "Buffalo", "Albany"],
        },
        "Brazil": {
            "São Paulo": ["São Paulo", "Campinas", "Santos"],
            "Rio de Janeiro": ["Rio de Janeiro", "Niterói", "Petrópolis"],
            "Bahia": ["Salvador", "Ilhéus", "Porto Seguro"],
        },
        "Mexico": {
            "Jalisco": ["Guadalajara", "Zapopan", "Puerto Vallarta"],
            "Yucatán": ["Mérida", "Valladolid", "Progreso"],
            "Oaxaca": ["Oaxaca City", "Juchitán", "Huatulco"],
        },
    },
}

#: Default items generated per leaf city.  108 cities × 12 ≈ 1,296 items —
#: enough for UMAP to form clean nested clusters and for ~160 signs across the
#: four levels (a Toponymy-like density) without a slow load.
DEFAULT_ITEMS_PER_CITY = 12

# Per-level Gaussian offset scales, coarse → fine.  Each level's cluster centre
# is its parent's plus a random offset at this scale; well-separated scales make
# the four tiers nest cleanly under UMAP's cosine metric (our vectors are
# L2-normalised, matching real embeddings at ingest).
_CONTINENT_SCALE = 6.0
_COUNTRY_SCALE = 2.4
_STATE_SCALE = 1.0
_CITY_SCALE = 0.42
_ITEM_JITTER = 0.11

# RNG stream ids kept distinct so centre-drawing and item-jitter never collide.
_SEED_CENTERS = 20260712
_SEED_ITEMS = 71207202


@dataclass(frozen=True)
class ToponymyItem:
    """One synthetic place: its category path, vector, and a stable hue index."""

    #: ``"Continent/Country/State/City"`` — the signpost builder's input.
    category: str
    #: L2-normalised synthetic embedding (dim chosen by the caller).
    embedding: np.ndarray
    #: 0-based index of the leaf city among all cities, for deterministic
    #: per-city media rendering (a tone frequency / a fill colour).
    city_index: int
    #: 0-based index of the continent, for a coarse colour/tone family.
    continent_index: int


def _center_tree(dim: int) -> dict[tuple[str, ...], np.ndarray]:
    """Draw a deterministic cluster centre for every taxonomy node.

    A node's centre is its parent's centre plus a fresh Gaussian offset scaled
    by the node's level, so siblings share a parent neighbourhood while the four
    tiers stay well separated.  Keyed by the node's path tuple.
    """
    rng = np.random.default_rng(_SEED_CENTERS)
    centers: dict[tuple[str, ...], np.ndarray] = {}
    root = np.zeros(dim, dtype=np.float64)
    for continent, countries in TAXONOMY.items():
        c_key = (continent,)
        centers[c_key] = root + rng.standard_normal(dim) * _CONTINENT_SCALE
        for country, states in countries.items():
            co_key = (*c_key, country)
            centers[co_key] = centers[c_key] + rng.standard_normal(dim) * _COUNTRY_SCALE
            for state, cities in states.items():
                s_key = (*co_key, state)
                centers[s_key] = centers[co_key] + rng.standard_normal(dim) * _STATE_SCALE
                for city in cities:
                    ci_key = (*s_key, city)
                    centers[ci_key] = centers[s_key] + rng.standard_normal(dim) * _CITY_SCALE
    return centers


def generate_items(dim: int, items_per_city: int = DEFAULT_ITEMS_PER_CITY) -> list[ToponymyItem]:
    """Generate the full synthetic item list with baked hierarchical vectors.

    Deterministic (fixed seeds) so a cached demo pickle and a fresh build agree.
    Each item is a leaf-city centre plus small Gaussian jitter, L2-normalised.
    """
    centers = _center_tree(dim)
    rng = np.random.default_rng(_SEED_ITEMS)
    items: list[ToponymyItem] = []
    city_index = 0
    for continent_index, (continent, countries) in enumerate(TAXONOMY.items()):
        for country, states in countries.items():
            for state, cities in states.items():
                for city in cities:
                    center = centers[(continent, country, state, city)]
                    for _ in range(items_per_city):
                        vec = center + rng.standard_normal(dim) * _ITEM_JITTER
                        norm = float(np.linalg.norm(vec))
                        if norm > 0:
                            vec = vec / norm
                        items.append(
                            ToponymyItem(
                                category=f"{continent}/{country}/{state}/{city}",
                                embedding=vec.astype(np.float32),
                                city_index=city_index,
                                continent_index=continent_index,
                            )
                        )
                    city_index += 1
    return items


def total_cities() -> int:
    """Number of leaf cities in the taxonomy (108) — for media rendering math."""
    return sum(
        len(cities) for countries in TAXONOMY.values() for states in countries.values() for cities in states.values()
    )


__all__ = [
    "SOURCE_ID",
    "TAXONOMY",
    "DEFAULT_ITEMS_PER_CITY",
    "ToponymyItem",
    "generate_items",
    "total_cities",
]
