"""The State of the App analyzer under the precision floor (#4357).

Since #4272 the default arm's line is the floor's set and every run ends on a
spot check whose rows sit past ``max_steps``.  What is pinned here is the part
a re-run cannot show by looking at it:

* **No FPR + FNR** reaches the analyzer's tables (owner, 2026-09-30).
* **F1 is the returned set's** (owner, 2026-09-30: "Using the returned-set
  threshold, show F1 over time, too"): the F1 of the top *K* the floor keeps,
  read off the rank frames, at every recorded click as well as the checkpoints.
* **The check is not a click:** "final" is the last ordinary step, and no check
  pick is credited to an image.
* **The line is read off the rank frames** by one definition
  (``_rank_metrics``), which agrees with brute force and with sklearn's AP.
* **Missing data stays missing:** a run recorded without rank frames reports
  its line at click 0 only, never a neighbour's value.

The cells are written by the real harness on synthetic clusters, so they carry
the floor-era shape (check rows, check picks, rank frames) without the GRID.

Meta-group: the subject is repo tooling under ``scripts/experiments/``.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_CALIB = _ROOT / "scripts" / "experiments" / "calibration"
_SOTA = _ROOT / "scripts" / "experiments" / "state_of_app"

MAX_STEPS = 20
CATS = ("cat0@small", "cat1@large")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rm():
    return _load("_rank_metrics", _CALIB / "_rank_metrics.py")


# ---------------------------------------------------------------- _rank_metrics


def test_average_precision_is_sklearns(rm) -> None:
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(42)
    scores = rng.normal(size=300)
    labels = (rng.random(300) < 0.1).astype(float)
    ranks = rm.ranks_from_scores(list(range(300)), scores, labels)
    assert rm.average_precision(ranks, int(labels.sum())) == pytest.approx(average_precision_score(labels, scores))


def test_oracle_recall_is_the_best_cut_by_brute_force(rm) -> None:
    rng = np.random.default_rng(42)
    for _ in range(20):
        n = 200
        labels = (rng.random(n) < 0.15).astype(int)
        ranks = rm.ranks_from_scores(list(range(n)), rng.normal(size=n) + labels, labels)
        n_pos = int(labels.sum())
        for floor in rm.FLOORS:
            best = 0.0
            for k in range(1, n + 1):
                right = int(np.count_nonzero(ranks < k))
                if right / k >= floor:
                    best = max(best, right / n_pos)
            assert rm.oracle_recall(ranks, n_pos, floor) == pytest.approx(best)


def test_oracle_f1_is_the_best_cut_by_brute_force(rm) -> None:
    rng = np.random.default_rng(7)
    for _ in range(20):
        n = 200
        labels = (rng.random(n) < 0.15).astype(int)
        ranks = rm.ranks_from_scores(list(range(n)), rng.normal(size=n) + labels, labels)
        n_pos = int(labels.sum())
        best = max(2 * int(np.count_nonzero(ranks < k)) / (k + n_pos) for k in range(1, n + 1))
        assert rm.oracle_f1(ranks, n_pos) == pytest.approx(best)


def test_the_line_keeps_the_floors_unchecked_candidate(rm) -> None:
    """The top 128 at 10%, 32 at 50% and above (#4272), never more than the corpus."""
    assert [rm.kept_count(x, 10_000) for x in rm.FLOORS] == [128, 32, 32]
    assert rm.kept_count(0.1, 50) == 50
    ranks = np.arange(0, 64, 2)  # every other item of the top 64 is a positive: 32 of them
    m = rm.line_metrics(ranks, 1000, 40, 0.5)
    assert (m["k"], m["precision"], m["meets"], m["shortfall"]) == (32, 0.5, 1.0, 0.0)
    assert m["recall"] == pytest.approx(16 / 40)
    assert m["f1"] == pytest.approx(2 * 0.5 * 0.4 / (0.5 + 0.4)), "F1 of the kept set, its own precision and recall"
    m = rm.line_metrics(ranks, 1000, 40, 0.9)
    assert (m["precision"], m["meets"], m["shortfall"]) == (0.5, 0.0, pytest.approx(0.4))


def test_a_frame_that_recorded_the_shipped_line_is_read_at_its_count(rm) -> None:
    """#4389: the unchecked line keeps the smaller of the schedule's count and the mixture's; frames record it."""
    ranks = np.array([0, 1, 2, 3, 10, 20])
    frame = {"test_line_k_p50": 4, "test_line_k_p10": -1}
    assert rm.frame_k(frame, 0.5) == 4
    assert rm.frame_k(frame, 0.1) is None, "-1: no session drew a line here"
    assert rm.frame_k({}, 0.9) is None, "a run before the column reads the schedule's count"
    m = rm.line_metrics(ranks, 1000, 6, 0.5, rm.frame_k(frame, 0.5))
    assert (m["k"], m["precision"], m["recall"]) == (4, 1.0, 4 / 6)
    assert rm.line_metrics(ranks, 1000, 6, 0.5)["k"] == 32
    assert rm.line_metrics(ranks, 3, 3, 0.5, 40)["k"] == 3, "never more than the corpus"


def test_ties_break_by_id_as_the_line_does(rm) -> None:
    # Two items tie; the lower id ranks first, as LineRanking sorts.
    assert rm.ranks_from_scores([7, 3], [0.5, 0.5], [1.0, 0.0]).tolist() == [1]


# ---------------------------------------------------------------- the analyzer


def _clips(seed: int = 0, n_per_cat: int = 80, dim: int = 16) -> dict:
    """Overlapping clusters, so the check has something to find and AP is not 1."""
    rng = np.random.RandomState(seed)
    cats = [*CATS, "other@small", "other@large", "other@medium"]
    centres = np.eye(len(cats), dim, dtype=np.float32)
    medias, mid = {}, 1
    for c, name in enumerate(cats):
        for _ in range(n_per_cat):
            emb = (centres[c] + rng.normal(0, 0.45, dim)).astype(np.float32)
            medias[mid] = {"id": mid, "embeddings": {"emb": emb}, "category": name}
            mid += 1
    return medias


def _write_cells(exp: Path) -> None:
    from vtscore.eval.voting_columns import CALIBRATION_COLUMNS, PICK_COLUMNS, RANK_FRAME_COLUMNS
    from vtscore.eval.voting_iterations import simulate_voting_iterations

    cells = exp / "results" / "cells"
    cells.mkdir(parents=True)
    clips = _clips()
    for idx, cat in enumerate(CATS):
        picks: list = []
        frames: list = []
        rows = simulate_voting_iterations(
            clips,
            cat,
            seed=0,
            dataset_name="coco_better",
            max_steps=MAX_STEPS,
            style="whole_image",
            safe_thresholds=True,
            emit_calibration_metrics=True,
            skyline_arms=["skyline_train_full"],
            pick_sink=picks,
            rank_frame_sink=frames,
            rank_frame_steps=(10, 20),
        )
        for table in (rows, picks, frames):
            for r in table:
                r["embedder"] = "siglip"
        pd.DataFrame(rows, columns=pd.Index([*CALIBRATION_COLUMNS, "embedder"])).to_csv(
            cells / f"task_{idx:04d}.csv", index=False
        )
        pd.DataFrame(picks, columns=pd.Index([*PICK_COLUMNS, "embedder"])).to_csv(
            cells / f"task_{idx:04d}__picks.csv", index=False
        )
        pd.DataFrame(frames, columns=pd.Index([*RANK_FRAME_COLUMNS, "embedder"])).to_csv(
            cells / f"task_{idx:04d}__rankframes.csv", index=False
        )
    # A run that never found a positive: no metric row, only its (all-Bad) clicks.
    idx = len(CATS)
    pd.DataFrame(columns=pd.Index([*CALIBRATION_COLUMNS, "embedder"])).to_csv(
        cells / f"task_{idx:04d}.csv", index=False
    )
    starved = [
        {"seed": 0, "dataset": "coco_better", "category": "cat9@small", "style": "whole_image", "t": t,
         "phase": "good", "picked_id": 1000 + t, "picked_label": 0, "embedder": "siglip"}
        for t in (1, 2, 3)
    ]  # fmt: skip
    pd.DataFrame(starved, columns=pd.Index([*PICK_COLUMNS, "embedder"])).to_csv(
        cells / f"task_{idx:04d}__picks.csv", index=False
    )
    pd.DataFrame(columns=pd.Index([*RANK_FRAME_COLUMNS, "embedder"])).to_csv(
        cells / f"task_{idx:04d}__rankframes.csv", index=False
    )


def _baseline(path: Path, rm) -> dict:
    """A text baseline in ``text_baseline.py``'s shape, with made-up but distinct values."""
    rows, want = [], {}
    for i, cat in enumerate([*CATS, "cat9@small"]):
        row = {"dataset": "coco_better", "embedder": "siglip", "category": cat, "seed": 0, "supports_text": 1}
        row["text_AP"] = 0.1 + i / 10
        for x in rm.FLOORS:
            for j, m in enumerate(
                ("k", "precision", "shortfall", "meets", "recall", "oracle_recall", "f1", "oracle_f1")
            ):
                row[f"text_{m}_{rm.floor_tag(x)}"] = round(0.01 * (i + 1) + j / 100 + x, 6)
        rows.append(row)
        want[cat] = row
    pd.DataFrame(rows).to_csv(path, index=False)
    return want


def _analyze(exp: Path, baseline: Path) -> Path:
    out = exp / "analysis"
    subprocess.run(  # noqa: S603  # fixed argv, repo-local script path, no shell
        [sys.executable, str(_SOTA / "analyze.py"), "--exp", str(exp), "--baseline", str(baseline), "--out", str(out)],
        check=True,
        capture_output=True,
        text=True,
    )
    return out


@pytest.fixture(scope="module")
def run(tmp_path_factory, rm):
    exp = tmp_path_factory.mktemp("sota") / "exp"
    _write_cells(exp)
    text = _baseline(exp / "text_baseline.csv", rm)
    out = _analyze(exp, exp / "text_baseline.csv")
    read = lambda name: pd.read_csv(out / name, dtype={"point": str} if name == "lines.csv" else None)  # noqa: E731
    return {
        "exp": exp,
        "text": text,
        "cells": read("cells.csv").set_index("category"),
        "lines": read("lines.csv"),
        "influence": read("influence.csv"),
        "curves": read("curves.csv"),
        "steps": read("line_steps.csv"),
        "summary": (out / "summary.md").read_text(),
    }


def _main_frame(exp: Path, idx: int) -> pd.DataFrame:
    df = pd.read_csv(exp / "results" / "cells" / f"task_{idx:04d}.csv")
    return df.loc[df["gmm_variant"].isna() & (df["pool_variant"] == "max")]


def _frames(exp: Path, idx: int) -> pd.DataFrame:
    return pd.read_csv(
        exp / "results" / "cells" / f"task_{idx:04d}__rankframes.csv",
        dtype={"test_pos_ranks": str, "pool_pos_ranks": str},
    )


def test_no_fpr_fnr_reaches_the_tables(run) -> None:
    for name in ("cells", "lines", "influence", "curves", "steps"):
        cols = [c for c in run[name].columns if "cost" in c or c in ("fpr", "fnr")]
        assert not cols, f"{name}.csv still carries {cols}"
    assert "cost" not in run["summary"].lower()


def test_f1_is_the_returned_sets_at_every_recorded_click(run, rm) -> None:
    """cells, lines, line_steps and curves all read one F1: the kept set's, off the frames."""
    lines, steps, curves = run["lines"], run["steps"], run["curves"]
    for idx, cat in enumerate(CATS):
        frames = _frames(run["exp"], idx)
        row = run["cells"].loc[cat]
        final = lines[(lines["category"] == cat) & (lines["point"] == "final") & (lines["floor"] == 0.5)].iloc[0]
        assert row["final_f1"] == pytest.approx(final["f1"])
        text = run["text"][cat]
        assert row["text_f1"] == pytest.approx(text["text_f1_p50"])
        c = curves[curves["category"] == cat].set_index("t")
        assert c.loc[0, "f1"] == pytest.approx(text["text_f1_p50"]), "click 0 is the text sort's line"
        assert c.loc[0, "f1_p10"] == pytest.approx(text["text_f1_p10"])
        for f in frames.query("kind == 'step'").to_dict("records"):
            want = rm.line_metrics(
                rm.parse_ranks(f["test_pos_ranks"]), int(f["n_test"]), int(f["n_test_pos"]), 0.5, rm.frame_k(f, 0.5)
            )
            got = steps[(steps["category"] == cat) & (steps["t"] == f["t"]) & (steps["floor"] == 0.5)].iloc[0]
            assert got["f1"] == pytest.approx(want["f1"])
            assert c.loc[int(f["t"]), "f1"] == pytest.approx(want["f1"]), "the curve at a frame IS the frame"
    assert "F1 of the line at P = 50%" in run["summary"]


def test_final_is_the_last_ordinary_step_not_the_check(run) -> None:
    for idx, cat in enumerate(CATS):
        main = _main_frame(run["exp"], idx)
        ordinary = main.loc[main["phase"] != "check"]
        assert (main["phase"] == "check").any() and main.loc[main["phase"] == "check", "t"].min() > MAX_STEPS
        last = ordinary.sort_values("t").iloc[-1]
        row = run["cells"].loc[cat]
        assert row["final_t"] == last["t"] == MAX_STEPS
        assert row["final_ap"] == pytest.approx(last["average_precision"])
        assert row["ap_150"] == pytest.approx(last["average_precision"]), "a checkpoint past the run keeps its end"


def test_no_check_pick_is_credited_as_a_click(run) -> None:
    inf = run["influence"]
    assert not inf.empty
    assert (inf["phase"] != "check").all()
    assert inf["t"].max() <= MAX_STEPS
    assert "help_ap" in inf.columns


def test_the_check_is_reported_against_its_own_truth(run, rm) -> None:
    for idx, cat in enumerate(CATS):
        row = run["cells"].loc[cat]
        assert row["check_status"] in ("confirmed", "short")
        last = _frames(run["exp"], idx).query("kind == 'last'").iloc[0]
        k = min(int(row["check_k"]), int(last["n_pool"]))
        truth = np.count_nonzero(rm.parse_ranks(last["pool_pos_ranks"]) < k) / k
        assert row["check_truth"] == pytest.approx(truth)
        assert row["check_covered"] == float(row["range_lo"] - 1e-9 <= truth <= row["range_hi"] + 1e-9)
        assert row["check_votes"] > 0


def test_the_line_is_read_off_the_rank_frames(run, rm) -> None:
    lines = run["lines"]
    for idx, cat in enumerate(CATS):
        frames = _frames(run["exp"], idx)
        for point, kind in (("final", "last"), ("ceiling", "skyline_train_full")):
            f = frames[frames["kind"] == kind].iloc[0]
            for x in rm.FLOORS:
                want = rm.line_metrics(
                    rm.parse_ranks(f["test_pos_ranks"]), int(f["n_test"]), int(f["n_test_pos"]), x, rm.frame_k(f, x)
                )
                got = lines[(lines["category"] == cat) & (lines["point"] == point) & (lines["floor"] == x)].iloc[0]
                for m, v in want.items():
                    assert got[m] == pytest.approx(v), (cat, point, x, m)
        text = lines[(lines["category"] == cat) & (lines["point"] == "text") & (lines["floor"] == 0.5)].iloc[0]
        assert text["precision"] == pytest.approx(run["text"][cat]["text_precision_p50"])


def test_a_run_that_never_trained_reads_the_text_sort_throughout(run) -> None:
    row = run["cells"].loc["cat9@small"]
    assert bool(row["never_trained"])
    assert row["final_ap"] == row["ap_10"] == row["text_ap"] == pytest.approx(run["text"]["cat9@small"]["text_AP"])
    assert row["positives_found"] == 0
    lines = run["lines"]
    at = lines[(lines["category"] == "cat9@small") & (lines["floor"] == 0.5)].set_index("point")
    for point in ("10", "50", "150", "final"):
        assert at.loc[point, "precision"] == pytest.approx(run["text"]["cat9@small"]["text_precision_p50"])


def test_without_rank_frames_the_line_is_known_at_click_0_only(run, tmp_path) -> None:
    exp = tmp_path / "old"
    shutil.copytree(run["exp"] / "results", exp / "results")
    for f in (exp / "results" / "cells").glob("*__rankframes.csv"):
        f.unlink()
    out = _analyze(exp, run["exp"] / "text_baseline.csv")
    lines = pd.read_csv(out / "lines.csv", dtype={"point": str})
    trained = lines.loc[lines["category"].isin(CATS)]
    assert bool(trained.loc[trained["point"] == "text", "precision"].notna().all())
    assert bool(trained.loc[trained["point"] != "text", "precision"].isna().all())
    cells = pd.read_csv(out / "cells.csv")
    assert bool(cells["final_ap"].notna().all()), "AP, harvest and the check need no rank frame"
    assert "no rank frames" in (out / "summary.md").read_text()
