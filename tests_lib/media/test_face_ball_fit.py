"""Fitting a face space's ball (#5497): the cells, the floors, the go/no-go check, and the script that writes them.

On FHIBE the fit reproduces #5478's shipped buffalo_s floors and #5486's buffalo_l and FaceNet refits exactly,
with FaceNet coming out no-go (``docs/experiments/2026-11-06-face-ball-per-embedder-5497``).  Here a small
synthetic set stands in: people who are tight balls far from everyone else, and people who are not.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from tests_shared.face_ball import face_ball_space  # noqa: F401
from vtscore.media.face import face_ball_fit as F
from vtscore.media.face.face_ball import face_ball_entry, write_entry

REPO = Path(__file__).resolve().parents[2]
D = 24


def _faces(noise: float, *, people: int = 16, photos: int = 10, strangers: int = 120, seed: int = 0) -> dict:
    """*people* people of *photos* faces each, ``noise`` from their own centre, and unlabelled strangers."""
    rng = np.random.default_rng(seed)
    medias: dict[int, dict] = {}

    def add(vec: np.ndarray, cats: list[str]) -> None:
        cid = len(medias) + 1
        medias[cid] = {
            "id": cid,
            "embeddings": {"face": (vec / np.linalg.norm(vec)).astype(np.float32)},
            "categories": cats,
        }

    for p in range(people):
        centre = rng.standard_normal(D)
        centre /= np.linalg.norm(centre)
        for _ in range(photos):
            add(centre + noise * rng.standard_normal(D) / np.sqrt(D), [f"p{p:02d}"])
    for _ in range(strangers):
        add(rng.standard_normal(D), [])
    return medias


PEOPLE = [f"p{p:02d}" for p in range(16)]


class TestTheCells:
    def test_each_person_has_a_cell_per_k_with_the_withheld_half_scored(self):
        cells = F.build_cells({"set": _faces(0.3)}, PEOPLE)
        assert {c.k for c in cells} == set(F.FIT_KS)
        assert {c.person for c in cells} == set(PEOPLE)
        for c in cells:
            assert c.n_pos == 5 and c.scores.dtype == np.float64 and np.isfinite(c.spread)
            assert c.kept_g[0] == int((c.scores >= 0.0).sum()) and np.all(np.diff(c.kept_g) <= 0)

    def test_the_cells_are_in_file_order(self):
        cells = F.build_cells({"b": _faces(0.3), "a": _faces(0.3, seed=1)}, PEOPLE[:3])
        keys = [(c.corpus, "all" if c.k == 0 else str(c.k)) for c in cells]
        assert keys == sorted(keys)

    def test_the_folds_split_the_people(self):
        assert {F.fold_of(p) for p in PEOPLE} == {0, 1}
        assert F.fold_of("002b6487-cb73-4fb3-97dc-b045d8b8ba06") == F.fold_of("002b6487-cb73-4fb3-97dc-b045d8b8ba06")


class TestTheFit:
    def test_balls_far_apart_are_used(self):
        ball = F.fit_face_ball("synthetic:balls", {"set": _faces(0.3)}, PEOPLE)
        assert ball.use and ball.check["use"]
        assert ball.check["clean_gap"] >= F.MIN_CLEAN_GAP
        assert all(g >= -F.LEVEL for g in ball.check["gain"].values())
        assert set(ball.betas) == set(F.FIT_BETAS)
        for floor, slope in ball.betas.values():
            assert 0.0 < floor < 1.0 and slope in F.SLOPES
        assert ball.betas[4.0][0] <= ball.betas[1.0][0], "a higher balance keeps more"
        assert ball.provenance["people"] == 16 and ball.provenance["seeds"] == [0]

    def test_people_who_are_not_balls_are_not(self):
        ball = F.fit_face_ball("synthetic:mist", {"set": _faces(4.0)}, PEOPLE)
        assert not ball.use and ball.check["clean_gap"] < F.MIN_CLEAN_GAP

    def test_no_cells_is_refused(self):
        with pytest.raises(ValueError, match="no cells"):
            F.fit_face_ball("synthetic:none", {"set": _faces(0.3)}, ["nobody"])


def _script():
    spec = importlib.util.spec_from_file_location("fit_face_ball", REPO / "scripts" / "data" / "fit_face_ball.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTheScript:
    def _set(self, tmp_path: Path, tag: str | None) -> tuple[Path, Path]:
        medias = _faces(0.3)
        if tag:
            for m in medias.values():
                m["custom_metadata"] = {"Face Compat": tag}
        pkl = tmp_path / "set__face.pkl"
        pkl.write_bytes(pickle.dumps(medias))
        people = tmp_path / "people.txt"
        people.write_text("# the synthetic people\n" + "\n".join(PEOPLE) + "\n")
        return pkl, people

    def test_it_writes_the_medias_space_beside_the_other_entries(self, tmp_path, face_ball_space):  # noqa: F811
        pkl, people = self._set(tmp_path, "synthetic:1")
        out = tmp_path / "face_ball.json"
        write_entry(face_ball_space, out)
        assert _script().main(["--medias", str(pkl), "--people", str(people), "--out", str(out)]) == 0
        spaces = json.loads(out.read_text())["spaces"]
        assert set(spaces) == {"synthetic:1", face_ball_space.compat}
        entry = spaces["synthetic:1"]
        assert entry["use"] is True and entry["provenance"]["people"] == 16
        assert entry["provenance"]["tool"] == "scripts/data/fit_face_ball.py" and "commit" in entry["provenance"]

    def test_an_untagged_set_needs_its_space_named(self, tmp_path):
        pkl, people = self._set(tmp_path, None)
        out = tmp_path / "face_ball.json"
        with pytest.raises(SystemExit):
            _script().main(["--medias", str(pkl), "--people", str(people), "--out", str(out)])
        args = ["--medias", str(pkl), "--people", str(people), "--out", str(out), "--compat", "synthetic:named"]
        assert _script().main(args) == 0
        assert "synthetic:named" in json.loads(out.read_text())["spaces"]

    def test_a_named_space_must_agree_with_the_tag(self, tmp_path):
        pkl, people = self._set(tmp_path, "synthetic:1")
        with pytest.raises(SystemExit):
            _script().main(["--medias", str(pkl), "--people", str(people), "--compat", "other:1", "--dry-run"])

    def test_a_dry_run_writes_nothing(self, tmp_path, face_ball_space):  # noqa: F811
        pkl, people = self._set(tmp_path, "synthetic:1")
        assert _script().main(["--medias", str(pkl), "--people", str(people), "--dry-run"]) == 0
        assert face_ball_entry("synthetic:1") is None
