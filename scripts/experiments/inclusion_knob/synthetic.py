"""Synthetic controlled arm: two classes on the unit sphere with tunable overlap.

Real embeddings (the AG News arm) show what production geometry does; this arm
controls *separability* directly so we can see how each knob design behaves as
the task moves from trivially separable to heavily overlapping.  Points are
drawn as ``normalize(class_center + noise_scale * gaussian)`` - the same
"tight-ish cluster on a sphere" shape CLAP/SigLIP/E5 concept clusters have.

Overlap levels (empirically tuned with a balanced logistic probe trained on
600 items, tested on the rest, 3 seeds):

* ``easy``   - balanced probe error ~0%: clusters fully separable
* ``medium`` - balanced probe error ~3%
* ``hard``   - balanced probe error ~6-8%: real irreducible overlap
"""

from __future__ import annotations

import numpy as np

DIM = 256
POOL_SIZE = 3000
PREVALENCE = 0.10

#: noise_scale (relative to the inter-center distance) per difficulty level.
LEVELS: dict[str, float] = {"easy": 2.0, "medium": 3.5, "hard": 4.5}


def make_synthetic(level: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X, y)``: unit-norm float32 ``(POOL_SIZE, DIM)`` and binary int8 labels."""
    noise_scale = LEVELS[level]
    rng = np.random.default_rng(10_000 + seed)

    center_pos = rng.standard_normal(DIM)
    center_pos /= np.linalg.norm(center_pos)
    center_neg = rng.standard_normal(DIM)
    center_neg /= np.linalg.norm(center_neg)

    n_pos = int(POOL_SIZE * PREVALENCE)
    n_neg = POOL_SIZE - n_pos
    sep = np.linalg.norm(center_pos - center_neg)
    sigma = noise_scale * sep / np.sqrt(DIM)

    X_pos = center_pos + sigma * rng.standard_normal((n_pos, DIM))
    X_neg = center_neg + sigma * rng.standard_normal((n_neg, DIM))
    X = np.vstack([X_pos, X_neg]).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = np.concatenate([np.ones(n_pos, dtype=np.int8), np.zeros(n_neg, dtype=np.int8)])

    perm = rng.permutation(POOL_SIZE)
    return X[perm], y[perm]
