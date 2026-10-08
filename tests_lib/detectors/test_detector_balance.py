"""A detector's balance, kept on the detector (#4665).

The balance (F-beta's beta) is asked for when a detector is created and kept
in its JSON, so Autopilot - which has no Threshold control - runs at the
balance the user chose.  A detector that keeps none takes the user's balance.
These tests pin the store (``vtscore.detectors.balance``), the seeding
(``vtscore.state.detector_beta`` / ``seed_detector_beta``), the write a
Threshold pick makes, and that a cold context trains at its own balance.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

import vtscore.detectors.training as training
import vtscore.state.core as core
from vtscore import config
from vtscore.config import DEFAULT_BETA
from vtscore.datasets.labelset import LabeledElement, LabelSet
from vtscore.detectors.balance import (
    BETA_KEY,
    detector_stored_beta,
    keep_beta,
    stored_beta,
    valid_beta,
)
from vtscore.detectors.store import _detector_path, _read_detector, _write_detector
from vtscore.state import (
    detector_beta,
    get_beta,
    register_setting_persister,
    seed_detector_beta,
    set_beta,
)
from vtscore.state.core import (
    DetectorContext,
    recompute_detector_thresholds,
    register_detector_context,
    set_thread_detector_context,
)


def _labelset() -> LabelSet:
    return LabelSet([LabeledElement(md5=f"m{n}", label="good") for n in range(3)])


@pytest.fixture
def scratch_dir(tmp_path):
    """Point the detectors dir at a scratch folder (``reset_contexts`` restores the builder)."""
    scratch = dataclasses.replace(config.CoreConfig.from_settings(), detectors_dir=tmp_path / "detectors")
    config.register_core_config_builder(lambda path=None: scratch)
    return tmp_path


def _registered(name: str, beta: float | None = None) -> tuple[DetectorContext, object]:
    """A registered detector with a file; *beta* kept on it when given."""
    from vtscore.detectors.registry import register_detector

    entry = register_detector(name=name, media_type="audio")
    path = _detector_path(name)
    data: dict = {"name": name, "media_type": "audio", "text_query": "kept", "labelset": _labelset().to_dict()}
    if beta is not None:
        data[BETA_KEY] = beta
    _write_detector(path, data)
    return DetectorContext(entry["id"], name=name), path


class TestTheStoredValue:
    @pytest.mark.parametrize(("raw", "expected"), [(1, 1.0), ("4", 4.0), (0.25, 0.25), (2.5, 2.5)])
    def test_a_number_in_range_is_a_balance(self, raw, expected):
        assert valid_beta(raw) == expected

    @pytest.mark.parametrize("raw", [None, True, False, "", "lean", math.nan, math.inf, 0.2, 5.0, -1.0])
    def test_anything_else_is_none(self, raw):
        assert valid_beta(raw) is None

    def test_a_detector_json_keeps_it_at_the_top(self):
        assert stored_beta({"name": "d", BETA_KEY: 4}) == 4.0
        assert stored_beta({"name": "d"}) is None
        assert stored_beta({BETA_KEY: 9}) is None
        assert stored_beta(None) is None

    def test_an_unregistered_id_keeps_none(self, scratch_dir):
        assert detector_stored_beta("") is None
        assert detector_stored_beta("no-such-detector") is None


class TestKeepingIt:
    def test_writes_the_balance_and_keeps_the_rest_of_the_file(self, scratch_dir):
        ctx, path = _registered("keep-det")
        assert keep_beta(ctx, 0.25) is True
        data = _read_detector(path)
        assert data is not None
        assert data[BETA_KEY] == 0.25
        assert data["text_query"] == "kept" and data["labelset"] == _labelset().to_dict()
        assert detector_stored_beta(ctx.detector_id) == 0.25
        # The write re-pointed the cached labelset, so it does not read as an outside edit.
        assert ctx.cached_labelset is not None and ctx.cached_labelset_mtime > 0

    def test_an_unchanged_balance_is_not_rewritten(self, scratch_dir):
        ctx, _path = _registered("same-det", beta=4.0)
        assert keep_beta(ctx, 4.0) is False

    def test_nowhere_to_keep_it(self, scratch_dir):
        ctx, path = _registered("bad-det")
        assert keep_beta(DetectorContext(""), 1.0) is False
        assert keep_beta(DetectorContext("unregistered"), 1.0) is False
        assert keep_beta(ctx, 9.0) is False
        data = _read_detector(path)
        assert data is not None and BETA_KEY not in data


class TestSeeding:
    def test_a_detector_seeds_from_the_balance_it_keeps(self, scratch_dir):
        ctx, _path = _registered("seed-kept", beta=4.0)
        set_thread_detector_context(ctx)
        assert get_beta() == 4.0
        assert ctx.beta_seeded and ctx.beta == 4.0

    def test_a_detector_that_keeps_none_takes_the_users(self, scratch_dir):
        ctx, _path = _registered("seed-none")
        set_thread_detector_context(ctx)
        assert get_beta() == DEFAULT_BETA

    def test_any_context_reads_its_own_not_the_active_ones(self, scratch_dir):
        active, _ = _registered("active-det", beta=0.25)
        other, _ = _registered("other-det", beta=4.0)
        set_thread_detector_context(active)
        assert detector_beta(other) == 4.0
        assert get_beta() == 0.25

    def test_a_seeded_context_keeps_its_own(self, scratch_dir):
        ctx, _path = _registered("seeded-det", beta=4.0)
        ctx.beta, ctx.beta_seeded = 0.25, True
        assert detector_beta(ctx) == 0.25

    def test_a_context_the_registry_cannot_name_is_seeded_from_its_json(self):
        ctx = DetectorContext("")
        assert seed_detector_beta(ctx, {"name": "cold", BETA_KEY: 0.25}) == 0.25
        assert ctx.beta_seeded and ctx.beta == 0.25
        # Already seeded: a second seed keeps it.
        assert seed_detector_beta(ctx, {BETA_KEY: 4.0}) == 0.25
        assert seed_detector_beta(DetectorContext(""), {"name": "cold"}) == DEFAULT_BETA


class TestAThresholdPick:
    def test_keeps_it_on_the_detector_and_as_the_users(self, scratch_dir):
        persisted: list = []
        register_setting_persister("beta", persisted.append)
        ctx, path = _registered("pick-det")
        set_thread_detector_context(ctx)
        register_detector_context(ctx)
        set_beta(4.0)
        data = _read_detector(path)
        assert data is not None and data[BETA_KEY] == 4.0
        assert persisted == [4.0]
        assert get_beta() == 4.0

    def test_a_loaded_detector_that_has_not_read_its_own_is_cut_at_the_one_it_keeps(self, scratch_dir, monkeypatch):
        cuts: dict[str, float | None] = {}

        def _recut(ctx, _inclusion, *, beta=None):
            cuts[ctx.detector_id] = beta
            return None

        monkeypatch.setattr(core, "recut_detector_threshold", _recut)
        kept, _ = _registered("recut-kept", beta=0.25)
        bare, _ = _registered("recut-bare")
        seeded, _ = _registered("recut-seeded", beta=0.25)
        seeded.beta, seeded.beta_seeded = 4.0, True
        for ctx in (kept, bare, seeded):
            register_detector_context(ctx)
        recompute_detector_thresholds(1.0)
        ours = {ctx.detector_id for ctx in (kept, bare, seeded)}
        assert {k: v for k, v in cuts.items() if k in ours} == {
            kept.detector_id: 0.25,
            bare.detector_id: 1.0,
            seeded.detector_id: 4.0,
        }


DIM = 8
EMB = "dinov3_patch"


def _snap(n: int = 40) -> dict[int, dict]:
    rng = np.random.default_rng(4665)
    snap: dict[int, dict] = {}
    for cid in range(1, n + 1):
        vec = rng.standard_normal(DIM).astype(np.float32)
        vec /= np.linalg.norm(vec)
        snap[cid] = {
            "id": cid,
            "media_type": "image",
            "embedder": EMB,
            "embeddings": {EMB: vec},
            "filename": f"m{cid}.png",
            "md5": f"m{cid}",
        }
    return snap


def test_a_cold_context_trains_at_its_own_balance_not_the_active_detectors(monkeypatch):
    """A throwaway context (AutoFind, the CLI, a cold Find) cuts at the balance its detector keeps."""
    seen: list[float | None] = []
    real = training._fused_threshold

    def _spy(*args, **kwargs):
        seen.append(kwargs.get("beta"))
        return real(*args, **kwargs)

    monkeypatch.setattr(training, "_fused_threshold", _spy)
    active = DetectorContext("active-det")
    active.beta, active.beta_seeded = 0.25, True
    set_thread_detector_context(active)
    cold = DetectorContext("")
    seed_detector_beta(cold, {"name": "cold", BETA_KEY: 4.0})

    snap = _snap()
    ids = sorted(snap)[:12]
    X = [snap[cid]["embeddings"][EMB] for cid in ids]
    y = [1.0 if i % 2 == 0 else 0.0 for i in range(12)]
    training.train_and_threshold(X, y, snap=snap, embedder_name=EMB, det_ctx=cold, voted_ids=set(ids))
    training.train_and_threshold(X, y, snap=snap, embedder_name=EMB, voted_ids=set(ids))
    assert seen == [4.0, 0.25]
