"""Layout persistence and the freshness guard (``vtscore.projection.store``).

These used to live in the app-tier route tests because the code did: the
persisted-projection guard and the container path resolution were private
helpers of ``vtsearch/routes/projection.py``, so exercising them meant
importing a Flask blueprint.  They are library code now, and this is a
library test.
"""

from __future__ import annotations

import pickle as _pickle
from unittest.mock import patch

import numpy as np

from vtscore.datasets.container import read_projection, write_container
from vtscore.projection import Projection, build_pyramid
from vtscore.projection.params import ProjectionParams
from vtscore.projection.store import (
    load_any_persisted_layout,
    load_persisted_layout,
    persist_projection,
    pkl_path_for,
    projection_params_match,
    remove_persisted_projections,
)
from vtscore.state.core import DatasetContext


def _params(n_neighbors: int, min_dist: float, compact: bool):
    """Patch what the active configuration would resolve for any dataset."""
    return patch(
        "vtscore.projection.params.resolve_projection_params",
        lambda ctx=None: ProjectionParams(n_neighbors, min_dist, compact),
    )


def _projection(pid: str, ids: list[int], method: str = "pca", *knobs) -> Projection:
    coords = np.zeros((len(ids), 2), dtype=np.float32)
    return Projection(pid, ids, coords, method, *knobs)


class TestProjectionParamsMatch:
    def test_invalidates_a_layout_whose_umap_params_changed(self):
        ids = [0, 1, 2]
        umap_default = _projection("p", ids, "umap", 15, 0.1, False)
        umap_changed = _projection("p", ids, "umap", 30, 0.1, False)
        pca = _projection("p", ids, "pca", None, None, None)
        legacy = _projection("p", ids, "umap", None, None, None)

        # Active settings at the config defaults.
        with _params(15, 0.1, False):
            assert projection_params_match(umap_default) is True
            assert projection_params_match(umap_changed) is False
            assert projection_params_match(pca) is True
            # Legacy None UMAP knobs are assumed to be the config defaults — but
            # an unstamped ``compact`` means "compacted", which today's default
            # is not.
            assert projection_params_match(legacy) is False

        # Operator tuned the setting away from the default -> stale layouts
        # recompute.
        with _params(30, 0.1, False):
            assert projection_params_match(legacy) is False
            assert projection_params_match(umap_changed) is True
            assert projection_params_match(pca) is True

    def test_detects_compaction_mismatch(self):
        """A layout compacted under the old default is refit, not silently served.

        ``compact`` used not to be stamped at all, so a compacted layout was
        indistinguishable from an uncompacted one and the mismatch could never
        force a recompute (issue #3056).
        """
        ids = [0, 1, 2]
        compacted = _projection("p", ids, "umap", 15, 0.1, True)
        uncompacted = _projection("p", ids, "umap", 15, 0.1, False)
        unstamped = _projection("p", ids, "umap", 15, 0.1, None)

        with _params(15, 0.1, False):
            assert projection_params_match(uncompacted) is True
            assert projection_params_match(compacted) is False
            # Unstamped reads as compacted — what it was when nothing recorded it.
            assert projection_params_match(unstamped) is False

        # ...and the guard is symmetric: with compaction back on, the
        # uncompacted layout is the stale one.
        with _params(15, 0.1, True):
            assert projection_params_match(compacted) is True
            assert projection_params_match(unstamped) is True
            assert projection_params_match(uncompacted) is False


class TestPklPathFor:
    def test_reads_the_registry_entry(self):
        with patch("vtscore.datasets.registry.get_dataset", return_value={"pkl_path": "/tmp/x.pkl"}):
            assert pkl_path_for("ds") == "/tmp/x.pkl"

    def test_unregistered_or_pathless_entries_read_as_absent(self):
        with patch("vtscore.datasets.registry.get_dataset", return_value=None):
            assert pkl_path_for("nope") is None
        with patch("vtscore.datasets.registry.get_dataset", return_value={"pkl_path": ""}):
            assert pkl_path_for("pathless") is None


class TestPersistProjection:
    def _container(self, tmp_path):
        pkl = tmp_path / "ds.pkl"
        write_container(pkl, _pickle.dumps({"medias": {}}), {"format_version": 1})
        return pkl

    def test_appends_to_registered_container(self, tmp_path):
        pkl = self._container(tmp_path)
        proj = _projection("persist-pid", [1, 2, 3])
        pyr = build_pyramid(proj, n_levels=1)

        with patch("vtscore.datasets.registry.get_dataset", return_value={"pkl_path": str(pkl)}):
            persist_projection("ds-xyz", proj, pyr)

        loaded = read_projection(str(pkl))
        assert loaded is not None
        assert loaded[0].projection_id == "persist-pid"

    def test_missing_registry_entry_is_noop(self):
        proj = _projection("x", [1])
        pyr = build_pyramid(proj, n_levels=1)
        with patch("vtscore.datasets.registry.get_dataset", return_value=None):
            # Must not raise even though there is no container to write to.
            persist_projection("nonexistent", proj, pyr)
            remove_persisted_projections("nonexistent")

    def test_removal_clears_every_stored_shape(self, tmp_path):
        """A forced re-projection must not leave the other shape behind.

        The coordinates are shared across bin shapes, so a surviving square
        pyramid would resurrect exactly the arrangement the user asked to
        replace the next time the hex map was opened.
        """
        pkl = self._container(tmp_path)
        proj = _projection("shared-pid", [1, 2, 3])
        with patch("vtscore.datasets.registry.get_dataset", return_value={"pkl_path": str(pkl)}):
            persist_projection("ds", proj, build_pyramid(proj, bin_shape="hex", n_levels=1))
            persist_projection("ds", proj, build_pyramid(proj, bin_shape="square", n_levels=1))
            assert read_projection(str(pkl), "hex") is not None
            assert read_projection(str(pkl), "square") is not None

            remove_persisted_projections("ds")
            assert read_projection(str(pkl), "hex") is None
            assert read_projection(str(pkl), "square") is None


class TestLoadPersistedLayout:
    def _stored(self, tmp_path, ids, shapes=("hex",), method="pca", knobs=()):
        pkl = tmp_path / "ds.pkl"
        write_container(pkl, _pickle.dumps({"medias": {}}), {"format_version": 1})
        proj = _projection("stored-pid", list(ids), method, *knobs)
        with patch("vtscore.datasets.registry.get_dataset", return_value={"pkl_path": str(pkl)}):
            for shape in shapes:
                persist_projection("ds", proj, build_pyramid(proj, bin_shape=shape, n_levels=1))
        return pkl

    def test_round_trips_a_matching_layout(self, tmp_path):
        pkl = self._stored(tmp_path, [1, 2, 3])
        ctx = DatasetContext("ds")
        with patch("vtscore.datasets.registry.get_dataset", return_value={"pkl_path": str(pkl)}):
            loaded = load_persisted_layout(ctx, [1, 2, 3], "hex")
        assert loaded is not None
        assert loaded[0].projection_id == "stored-pid"
        assert loaded[1].bin_shape == "hex"

    def test_id_set_mismatch_reads_as_absent(self, tmp_path):
        """A dataset that gained or lost items must be re-fit, not re-served."""
        pkl = self._stored(tmp_path, [1, 2, 3])
        ctx = DatasetContext("ds")
        with patch("vtscore.datasets.registry.get_dataset", return_value={"pkl_path": str(pkl)}):
            assert load_persisted_layout(ctx, [1, 2, 3, 4], "hex") is None

    def test_stale_params_read_as_absent(self, tmp_path):
        pkl = self._stored(tmp_path, [1, 2, 3], method="umap", knobs=(15, 0.1, False))
        ctx = DatasetContext("ds")
        with patch("vtscore.datasets.registry.get_dataset", return_value={"pkl_path": str(pkl)}):
            with _params(15, 0.1, False):
                assert load_persisted_layout(ctx, [1, 2, 3], "hex") is not None
            with _params(30, 0.1, False):
                assert load_persisted_layout(ctx, [1, 2, 3], "hex") is None

    def test_unstored_shape_reads_as_absent(self, tmp_path):
        pkl = self._stored(tmp_path, [1, 2, 3], shapes=("hex",))
        ctx = DatasetContext("ds")
        with patch("vtscore.datasets.registry.get_dataset", return_value={"pkl_path": str(pkl)}):
            assert load_persisted_layout(ctx, [1, 2, 3], "square") is None

    def test_any_shape_yields_the_shared_coordinates(self, tmp_path):
        """Only hex is stored, but a square request still gets the layout back.

        The 2-D coordinates are shared across bin shapes, so the caller can
        re-bin them instead of paying for a second UMAP fit.
        """
        pkl = self._stored(tmp_path, [1, 2, 3], shapes=("hex",))
        ctx = DatasetContext("ds")
        with patch("vtscore.datasets.registry.get_dataset", return_value={"pkl_path": str(pkl)}):
            loaded = load_any_persisted_layout(ctx, [1, 2, 3], prefer="square")
        assert loaded is not None
        assert loaded[1].bin_shape == "hex"

    def test_prefer_wins_when_both_shapes_are_stored(self, tmp_path):
        pkl = self._stored(tmp_path, [1, 2, 3], shapes=("hex", "square"))
        ctx = DatasetContext("ds")
        with patch("vtscore.datasets.registry.get_dataset", return_value={"pkl_path": str(pkl)}):
            square = load_any_persisted_layout(ctx, [1, 2, 3], prefer="square")
            hexed = load_any_persisted_layout(ctx, [1, 2, 3], prefer="hex")
        assert square is not None and square[1].bin_shape == "square"
        assert hexed is not None and hexed[1].bin_shape == "hex"

    def test_unregistered_dataset_reads_as_absent(self):
        ctx = DatasetContext("ds")
        with patch("vtscore.datasets.registry.get_dataset", return_value=None):
            assert load_persisted_layout(ctx, [1], "hex") is None
            assert load_any_persisted_layout(ctx, [1]) is None
