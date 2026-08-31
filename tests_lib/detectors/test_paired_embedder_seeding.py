"""Tests for the paired ``"<text>+<learn>"`` embedder (#3276).

Autopilot asks an embedding space for two unrelated things: a **text sort** to
open on, and a **media space** to learn in.  ``dinov3_patch`` is the only
patch-capable embedder in the pile and it has no text tower, so on its own it
can never take the app's real opening — every DINOv3 cell fell back to the
three-random-known-goods start while the whole-image arms opened on a typed
query, putting a seeding difference *inside* the voting-mode axis the #3156
scale study exists to measure.

``siglip+dinov3_patch`` splits the two: SigLIP ranks the query, DINOv3 does the
learning.  These tests pin the two halves of that — which half each config
lookup reads, and that the opening really is built from the *text* half's
vectors over the *learn* half's media ids.

Both modules are loose scripts, not package members, so they are loaded by
path.  ``run_cells`` is executed against a stub ``common``: the real one mutates
``sys.meta_path`` and mkdirs an experiment tree at import time, neither of which
belongs in a unit test.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_CALIB = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "calibration"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cfg():
    return _load("_calib_experiment_config", _CALIB / "experiment_config.py")


@pytest.fixture(scope="module")
def run_cells(cfg):
    """``run_cells`` with a stub ``common`` so importing it has no side effects."""
    # `Any` rather than `ModuleType`: a stub module is built by assigning
    # attributes that the type does not declare, and pyright is right to object
    # to that on a real ModuleType.
    stub: Any = types.ModuleType("common")
    stub.setup_env = lambda: None
    stub.log = lambda _msg: None
    stub.Path = Path
    stub.RESULTS = Path(".")
    saved = {k: sys.modules.get(k) for k in ("common", "experiment_config")}
    sys.modules["common"] = stub
    sys.modules["experiment_config"] = cfg
    try:
        return _load("_calib_run_cells", _CALIB / "run_cells.py")
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


PAIR = "siglip+dinov3_patch"


# --- Which half each lookup reads -------------------------------------------


def test_split_embedder_plain_name_is_both_halves(cfg):
    assert cfg.split_embedder("siglip") == ("siglip", "siglip")
    assert not cfg.is_paired("siglip")


def test_split_embedder_pair(cfg):
    assert cfg.split_embedder(PAIR) == ("siglip", "dinov3_patch")
    assert cfg.text_embedder(PAIR) == "siglip"
    assert cfg.learn_embedder(PAIR) == "dinov3_patch"
    assert cfg.is_paired(PAIR)


def test_patchness_is_decided_by_the_learn_half(cfg):
    """The patch grid comes from the space the detector learns in, not the sort."""
    assert cfg.is_patch_embedder(PAIR)
    # Reversed, the pair learns in SigLIP space and has no patches to pool.
    assert not cfg.is_patch_embedder("dinov3_patch+siglip")


def test_pickles_split_by_role(cfg):
    assert cfg.pickle_name("vg_scale", PAIR) == "vg_scale__dinov3_patch.pkl"
    assert cfg.text_pickle_name("vg_scale", PAIR) == "vg_scale__siglip.pkl"
    assert cfg.crops_basename("vg_scale", PAIR) == "vg_scale__dinov3_patch__crops"


def test_unpaired_pickle_names_are_unchanged(cfg):
    """The paired path is a generalisation, not a branch: plain names still work."""
    assert cfg.pickle_name("vg_scale", "siglip") == "vg_scale__siglip.pkl"
    assert cfg.text_pickle_name("vg_scale", "siglip") == "vg_scale__siglip.pkl"
    assert cfg.crops_basename("vg_scale", "siglip") == "vg_scale__siglip__crops"


def test_pair_region_votes_on_a_boxed_dataset(cfg):
    assert cfg.region_voting_for("vg_scale", PAIR)
    assert cfg.styles_for("vg_scale", PAIR) == cfg.PATCH_STYLES
    # Both halves of the premise still required: no boxes, no region voting.
    assert not cfg.region_voting_for("caltech101_m", PAIR)
    assert cfg.styles_for("caltech101_m", PAIR) == cfg.SINGLE_STYLES


# --- The opening itself ------------------------------------------------------

_DS = "_paired_fixture"
_CAT = "boat"
_QUERY = "a boat on the water"


def _media(vectors: dict[str, list[float]], primary: str) -> dict:
    return {
        "embeddings": {k: np.asarray(v, dtype=np.float32) for k, v in vectors.items()},
        "embedder": primary,
    }


@pytest.fixture
def queried(cfg, monkeypatch):
    """A dataset the query table knows about, so ``seed_query_text`` resolves."""
    monkeypatch.setitem(cfg.EXPERIMENT_QUERIES, _DS, {_CAT: _QUERY})


@pytest.fixture
def text_tower(monkeypatch):
    """``embed_text_query`` answering only for SigLIP, as the real towers do."""
    import vtscore.embedding.helpers as helpers

    calls: list[tuple[str, str | None, bool]] = []

    def _embed(text, media_type, enrich=False, embedder_name=None):  # noqa: ARG001
        calls.append((text, embedder_name, enrich))
        if embedder_name != "siglip":
            return None  # DINOv3 has no text tower
        return np.asarray([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(helpers, "embed_text_query", _embed)
    return calls


#: SigLIP says media 1 is the best match; DINOv3 says the exact opposite.  Any
#: implementation that ranks in the wrong space inverts the order, so the
#: assertion below is a real discriminator rather than a smoke test.
_SIGLIP = {1: [1.0, 0.0], 2: [0.0, 1.0]}
_DINOV3 = {1: [0.0, 1.0], 2: [1.0, 0.0]}


def test_pair_ranks_in_the_text_space_over_the_learn_ids(cfg, run_cells, queried, text_tower, monkeypatch, tmp_path):
    """SigLIP orders the opening; the ids are the DINOv3 cell's own."""
    learn = {cid: _media({"dinov3_patch": v}, "dinov3_patch") for cid, v in _DINOV3.items()}
    text = {cid: _media({"siglip": v}, "siglip") for cid, v in _SIGLIP.items()}

    from vtscore.datasets import loader as _loader

    monkeypatch.setattr(_loader, "EMBEDDINGS_DIR", tmp_path)
    (tmp_path / "_paired_fixture__siglip.pkl").write_bytes(b"")  # existence only
    monkeypatch.setitem(sys.modules, "_cells_io", _cells_io_stub(text))

    scores = run_cells._text_seed_scores(_DS, PAIR, _CAT, learn)

    assert set(scores) == set(learn)
    assert scores[1] > scores[2], "the opening must follow SigLIP, not DINOv3"
    assert ("a boat on the water", "siglip", False) in text_tower
    # #3341: the opening must be embedded the way the app embeds it, not
    # however `embed_text_query` happens to default.  The harness passes
    # cfg.SEED_ENRICH explicitly, which tracks the app's shipped
    # `enrich_descriptions` default (off).
    assert all(enrich is cfg.SEED_ENRICH for _text, _name, enrich in text_tower)


def test_pair_prefers_vectors_already_on_the_media(run_cells, queried, text_tower, monkeypatch, tmp_path):
    """A multi-vector media needs no second pickle — the production shape."""
    both = {cid: _media({"dinov3_patch": _DINOV3[cid], "siglip": _SIGLIP[cid]}, "dinov3_patch") for cid in _DINOV3}

    from vtscore.datasets import loader as _loader

    monkeypatch.setattr(_loader, "EMBEDDINGS_DIR", tmp_path)  # empty: no side pickle exists
    monkeypatch.setitem(sys.modules, "_cells_io", _cells_io_stub(None))

    vectors, provenance = run_cells._text_seed_vectors(_DS, PAIR, both)

    assert provenance == "multi_vector"
    assert vectors is both
    scores = run_cells._text_seed_scores(_DS, PAIR, _CAT, both)
    assert scores[1] > scores[2]


def test_pair_refuses_a_text_side_that_misses_medias(run_cells, monkeypatch, tmp_path):
    """A partial opening is a different experiment, so it fails rather than runs."""
    learn = {cid: _media({"dinov3_patch": v}, "dinov3_patch") for cid, v in _DINOV3.items()}
    partial = {1: _media({"siglip": _SIGLIP[1]}, "siglip")}

    from vtscore.datasets import loader as _loader

    monkeypatch.setattr(_loader, "EMBEDDINGS_DIR", tmp_path)
    (tmp_path / "_paired_fixture__siglip.pkl").write_bytes(b"")
    monkeypatch.setitem(sys.modules, "_cells_io", _cells_io_stub(partial))

    with pytest.raises(ValueError, match="missing 1 of 2 medias"):
        run_cells._text_seed_vectors(_DS, PAIR, learn)


def test_pair_reports_a_missing_text_pickle_by_name(run_cells, monkeypatch, tmp_path):
    learn = {cid: _media({"dinov3_patch": v}, "dinov3_patch") for cid, v in _DINOV3.items()}

    from vtscore.datasets import loader as _loader

    monkeypatch.setattr(_loader, "EMBEDDINGS_DIR", tmp_path)
    monkeypatch.setitem(sys.modules, "_cells_io", _cells_io_stub(None))

    with pytest.raises(FileNotFoundError, match="_paired_fixture__siglip.pkl"):
        run_cells._text_seed_vectors(_DS, PAIR, learn)


def test_unpaired_embedder_still_ranks_on_its_own_medias(run_cells, queried, text_tower, monkeypatch):
    """No pickle is opened and the primary vector is used — the pre-#3276 path."""
    medias = {cid: _media({"siglip": v}, "siglip") for cid, v in _SIGLIP.items()}
    monkeypatch.setitem(sys.modules, "_cells_io", _cells_io_stub(None))

    vectors, provenance = run_cells._text_seed_vectors(_DS, "siglip", medias)
    assert provenance == "cell"
    assert vectors is medias

    scores = run_cells._text_seed_scores(_DS, "siglip", _CAT, medias)
    assert scores[1] > scores[2]


def test_no_text_tower_means_no_text_sort(run_cells, queried, text_tower):
    """Bare DINOv3 still falls back to the known-good start rather than pretending."""
    medias = {cid: _media({"dinov3_patch": v}, "dinov3_patch") for cid, v in _DINOV3.items()}
    assert run_cells._text_seed_scores(_DS, "dinov3_patch", _CAT, medias) is None


def _cells_io_stub(text_medias):
    """A ``_cells_io`` whose ``load_medias`` returns *text_medias* (or refuses)."""
    module: Any = types.ModuleType("_cells_io")

    def load_medias(_path):
        if text_medias is None:
            raise AssertionError("load_medias must not be called on this path")
        return text_medias

    module.load_medias = load_medias
    return module
