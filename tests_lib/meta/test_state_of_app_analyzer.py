"""The State of the App analyzer under the precision floor (#4357).

Since #4272 the default arm's line is the floor's set and every run ends on a
spot check whose rows sit past ``max_steps``.  What is pinned here is the part
a re-run cannot show by looking at it:

* **No FPR + FNR** reaches the analyzer's tables (owner, 2026-09-30).
* **F1 is the returned set's** (owner, 2026-09-30: "Using the returned-set
  threshold, show F1 over time, too"): the F1 of the top *K* the floor keeps,
  read off the rank frames, at every recorded click as well as the checkpoints.
* **Precision and recall at P** (owner, 2026-10-01, #4408): the returned set
  at each floor, scored against the floor it aimed for, at every recorded
  click; each run records the floor its sessions aimed at, and ``perp.py``
  reads each P off its own run.
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
            min_precision=0.5,  # the floor arm: the per-P machinery's fixture (#4408); the balance is the default
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
        # The text sort's own line in the app (the blind GMM cut), as text_baseline.py records it (#4474).
        row["n_test"] = 1000
        row["n_test_pos"] = 10 + i
        row["text_precision"] = round(0.05 + i / 50, 6)
        row["text_recall"] = round(0.9 - i / 50, 6)
        row["text_fpr"] = round(0.1 + i / 100, 6)
        for x in rm.FLOORS:
            for j, m in enumerate(
                ("k", "precision", "shortfall", "meets", "recall", "oracle_recall", "f1", "oracle_f1")
            ):
                row[f"text_{m}_{rm.floor_tag(x)}"] = round(0.01 * (i + 1) + j / 100 + x, 6)
        for b in rm.BETAS:  # the balance's columns (#4413), distinct from the floors'
            for j, m in enumerate(("k", "precision", "recall", "fbeta", "oracle_fbeta", "fb_share")):
                row[f"text_{m}_{rm.beta_tag(b)}"] = round(0.02 * (i + 1) + j / 100 + b / 10, 6)
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
        "balances": read("balances.csv"),
        "balance_steps": read("balance_steps.csv"),
        "pools": read("pools.csv"),
        "pool_steps": read("pool_steps.csv"),
        "thresholds": read("thresholds.csv"),
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


def test_precision_and_recall_at_each_p_are_the_returned_sets_at_every_recorded_click(run, rm) -> None:
    """#4408: the set the app returns when it aims for P, scored at P, per floor, off the same frames as F1."""
    curves, steps = run["curves"], run["steps"]
    for idx, cat in enumerate(CATS):
        frames = _frames(run["exp"], idx)
        c = curves[curves["category"] == cat].set_index("t")
        for x in rm.FLOORS:
            tag = rm.floor_tag(x)
            assert c.loc[0, f"precision_{tag}"] == pytest.approx(run["text"][cat][f"text_precision_{tag}"])
            for f in frames.query("kind == 'step'").to_dict("records"):
                want = rm.line_metrics(
                    rm.parse_ranks(f["test_pos_ranks"]), int(f["n_test"]), int(f["n_test_pos"]), x, rm.frame_k(f, x)
                )
                got = steps[(steps["category"] == cat) & (steps["t"] == f["t"]) & (steps["floor"] == x)].iloc[0]
                for m in ("precision", "recall", "oracle_recall"):
                    assert got[m] == pytest.approx(want[m])
                    assert c.loc[int(f["t"]), f"{m}_{tag}"] == pytest.approx(want[m]), (cat, x, m)
    assert "## The returned set at each P: precision against P, recall against the oracle" in run["summary"]
    assert "These sessions aimed at P = 50%" in run["summary"]


def test_fbeta_share_is_the_returned_sets_at_every_recorded_click(run, rm) -> None:
    """#4413: the set the app returns at each balance, scored as F-beta over the best cut, off the same frames."""
    curves, balances, steps = run["curves"], run["balances"], run["balance_steps"]
    for idx, cat in enumerate(CATS):
        frames = _frames(run["exp"], idx)
        c = curves[curves["category"] == cat].set_index("t")
        for b in rm.BETAS:
            tag = rm.beta_tag(b)
            text = balances[(balances["category"] == cat) & (balances["point"] == "text") & (balances["beta"] == b)]
            # Two rules per point (#4474, owner 2026-10-04): the text sort's top-K is the baseline's cap reading;
            # its app line is the blind GMM cut; the curves carry the app's line.
            assert set(text["rule"]) == {"app line", "top-K"}
            top = text[text["rule"] == "top-K"].iloc[0]
            assert top["fb_share"] == pytest.approx(run["text"][cat][f"text_fb_share_{tag}"])
            app = text[text["rule"] == "app line"].iloc[0]
            tb = run["text"][cat]
            want_f = (
                (1 + b * b)
                * tb["text_precision"]
                * tb["text_recall"]
                / (b * b * tb["text_precision"] + tb["text_recall"])
            )
            assert app["fbeta"] == pytest.approx(want_f), "the text sort's app line is its GMM cut"
            assert app["k"] == pytest.approx(
                tb["text_recall"] * tb["n_test_pos"] + tb["text_fpr"] * (tb["n_test"] - tb["n_test_pos"])
            )
            assert app["fb_share"] == pytest.approx(want_f / tb[f"text_oracle_fbeta_{tag}"])
            assert c.loc[0, f"fb_share_{tag}"] == pytest.approx(app["fb_share"]), "the curves carry the app's line"
            cap = 32 if b <= 1 else 128
            for f in frames.query("kind == 'step'").to_dict("records"):
                top_step = steps[
                    (steps["category"] == cat)
                    & (steps["t"] == f["t"])
                    & (steps["beta"] == b)
                    & (steps["rule"] == "top-K")
                ].iloc[0]
                assert top_step["k"] == min(cap, int(f["n_test"])), "top-K: the cap on the detector too"
                steps_app = steps[steps["rule"] == "app line"]
                want = rm.balance_metrics(
                    rm.parse_ranks(f["test_pos_ranks"]),
                    int(f["n_test"]),
                    int(f["n_test_pos"]),
                    b,
                    rm.frame_beta_k(f, b),
                )
                got = steps_app[
                    (steps_app["category"] == cat) & (steps_app["t"] == f["t"]) & (steps_app["beta"] == b)
                ].iloc[0]
                for m in ("k", "fbeta", "fb_share", "precision", "recall"):
                    assert got[m] == pytest.approx(want[m]), (cat, b, m)
                assert c.loc[int(f["t"]), f"fb_share_{tag}"] == pytest.approx(want["fb_share"])
                assert 0.0 <= want["fb_share"] <= 1.0 + 1e-9
            final = balances[(balances["category"] == cat) & (balances["point"] == "final") & (balances["beta"] == b)]
            final = final[final["rule"] == "app line"]
            last = frames.query("kind == 'last'").iloc[0]
            want = rm.balance_metrics(
                rm.parse_ranks(last["test_pos_ranks"]),
                int(last["n_test"]),
                int(last["n_test_pos"]),
                b,
                rm.frame_beta_k(last, b),
            )
            assert final["fbeta"].iloc[0] == pytest.approx(want["fbeta"])
    assert "## The returned set at each balance: F-beta over the best cut" in run["summary"]
    assert "These sessions aimed at a floor, not a balance" in run["summary"]
    assert run["cells"]["session_beta"].isna().all(), "a floor run records no session beta"


def test_each_rule_thresholds_both_sorts_the_same_way() -> None:
    """Owner, 2026-10-04: a text-vs-detector comparison uses one rule on both sides."""
    sa = _load("sota_analyze_rules", _SOTA / "analyze.py")
    # The text sort: its app line is the blind GMM cut (precision, recall, fpr); its top-K the cap.
    text = {"text_gmm_precision": 0.5, "text_gmm_recall": 0.8, "text_gmm_fpr": 0.004, "n_test": 1000.0, "n_test_pos": 10.0,
            "text_oracle_fbeta_b1": 0.9, "text_k_b1": 32, "text_fbeta_b1": 0.4, "text_precision_b1": 0.2,
            "text_recall_b1": 0.64, "text_fb_share_b1": 0.4 / 0.9}  # fmt: skip
    rows = dict(sa._balance_rows(text, None, True, 1.0))
    assert set(rows) == {"app line", "top-K"}
    assert rows["app line"]["k"] == pytest.approx(0.8 * 10 + 0.004 * 990)
    assert rows["app line"]["fbeta"] == pytest.approx(2 * 0.5 * 0.8 / 1.3)
    assert rows["app line"]["fb_share"] == pytest.approx(rows["app line"]["fbeta"] / 0.9)
    assert rows["top-K"]["k"] == 32 and rows["top-K"]["fbeta"] == pytest.approx(0.4)
    # The detector: its app line is what the frame recorded; its top-K the cap, whatever the frame says.
    frame = {
        "test_pos_ranks": "0 1 5 40 300",
        "n_test": 1000,
        "n_test_pos": 5,
        "test_line_k_b1": 3,
        "test_line_k_b4": 3,
    }
    rows = dict(sa._balance_rows(text, frame, False, 1.0))
    assert rows["app line"]["k"] == 3 and rows["top-K"]["k"] == 32
    assert rows["app line"]["oracle_fbeta"] == rows["top-K"]["oracle_fbeta"], "one ranking, one best cut"
    rows4 = dict(sa._balance_rows(text, frame, False, 4.0))
    assert rows4["top-K"]["k"] == 128
    # The ceiling (#4486): its app line is Find's labels line, as its frame recorded it; a frame from
    # before #4486 recorded none (-1), and the row is blank but for the ranking's best cut.
    sky = {"precision": 1.0, "recall": 0.6, "fpr": 0.0, "n_test_pos": 5.0, "n_test_neg": 995.0}
    rows = dict(sa._balance_rows(text, frame, False, 1.0, sky))
    assert rows["app line"]["k"] == 3, "the line the ceiling's frame recorded, not the skyline's oracle cut"
    assert rows["top-K"]["k"] == 32
    old = {**frame, "test_line_k_b1": -1}
    rows = dict(sa._balance_rows(text, old, False, 1.0, sky))
    assert all(np.isnan(rows["app line"][m]) for m in ("k", "fbeta", "precision", "recall", "fb_share"))
    assert rows["app line"]["oracle_fbeta"] == rows["top-K"]["oracle_fbeta"]


def test_the_objective_is_the_withheld_set_above_the_threshold(run) -> None:
    """#4427 (owner): F-beta of the withheld images above the app's threshold, off the session's own rows -
    the last ordinary row (unchecked) and the last check row (after the walk); every ordinary click in
    thresholds.csv and the curve; the summary's first section."""
    cells, thr, curves = run["cells"], run["thresholds"], run["curves"]

    def f1(p, r):
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    for idx, cat in enumerate(CATS):
        main = _main_frame(run["exp"], idx).sort_values("t")
        ordinary = main[main["phase"].astype(str) != "check"]
        check = main[main["phase"].astype(str) == "check"]
        last = ordinary.iloc[-1]
        want_unchecked = f1(float(last["precision"]), float(last["recall"]))
        assert cells.loc[cat, "thr_fbeta_unchecked"] == pytest.approx(want_unchecked)
        end = check.iloc[-1] if len(check) else last
        want_final = f1(float(end["precision"]), float(end["recall"]))
        assert cells.loc[cat, "thr_fbeta_final"] == pytest.approx(want_final)
        assert cells.loc[cat, "thr_walk_effect"] == pytest.approx(want_final - want_unchecked)
        assert cells.loc[cat, "thr_returned_final"] == pytest.approx(
            float(end["recall"]) * float(end["n_test_pos"]) + float(end["fpr"]) * float(end["n_test_neg"])
        )
        steps = thr[(thr["category"] == cat) & (thr["point"] == "step")].set_index("t")
        c = curves[curves["category"] == cat].set_index("t")
        # Plain numpy columns rather than rows: `iterrows` and `to_dict` each
        # type their cells differently across pandas-stubs releases, and pyright
        # has rejected each of them in turn.
        rows = zip(np.asarray(ordinary["t"]), np.asarray(ordinary["precision"]), np.asarray(ordinary["recall"]))
        for t, precision, recall in rows:
            want = f1(float(precision), float(recall))
            assert steps.loc[int(t), "thr_fbeta"] == pytest.approx(want)
            assert c.loc[int(t), "thr_fbeta"] == pytest.approx(want)
        assert np.isnan(c.loc[0, "thr_fbeta"]), "no threshold before the first trained click"
        fin = thr[(thr["category"] == cat) & (thr["point"] == "final")].iloc[0]
        assert fin["thr_fbeta"] == pytest.approx(want_final) and fin["beta"] == 1.0
    s = run["summary"]
    assert s.index("## The objective: the withheld set above the app's threshold") < s.index("## The ranking: AP"), (
        "the objective comes first"
    )
    assert "fbeta_unchecked" in s and "walk_effect" in s


def test_positives_in_hand_are_the_goods_plus_the_kept_sets_on_the_users_own_corpus(run, rm) -> None:
    """#4427 (a diagnostic, not the objective): at each recorded click, Goods voted so far + the positives
    inside the line the app kept over the session's unvoted pool, the line being the harness's own
    ``floor_count`` at that click."""
    cells, pools, steps = run["cells"], run["pools"], run["pool_steps"]
    for idx, cat in enumerate(CATS):
        frames = _frames(run["exp"], idx)
        main = _main_frame(run["exp"], idx)
        picks = pd.read_csv(run["exp"] / "results" / "cells" / f"task_{idx:04d}__picks.csv")
        clicks = picks[picks["phase"].astype(str) != "check"] if "phase" in picks else picks
        own_beta = 1.0  # a floor run: the pool-side set is scored as F1
        for f in frames.query("kind == 'step'").to_dict("records"):
            t = int(f["t"])
            got = steps[(steps["category"] == cat) & (steps["t"] == t)].iloc[0]
            count = main.loc[(main["t"] == t) & (main["phase"].astype(str) != "check"), "floor_count"].iloc[-1]
            want = rm.balance_metrics(
                rm.parse_ranks(f["pool_pos_ranks"]), int(f["n_pool"]), int(f["n_pool_pos"]), own_beta, int(count)
            )
            goods = int(clicks.loc[clicks["t"] <= t, "picked_label"].sum())
            assert got["beta"] == own_beta and got["pool_k"] == want["k"]
            assert got["pool_precision"] == pytest.approx(want["precision"])
            assert got["pool_fb_share"] == pytest.approx(want["fb_share"], nan_ok=True)
            assert got["goods"] == goods
            assert got["in_hand"] == goods + round(want["precision"] * want["k"])
        final = pools[(pools["category"] == cat) & (pools["point"] == "final")].iloc[0]
        assert cells.loc[cat, "final_in_hand"] == final["in_hand"]
        assert cells.loc[cat, "final_pool_positives"] == final["pool_positives"]
        text = pools[(pools["category"] == cat) & (pools["point"] == "text")].iloc[0]
        assert text["goods"] == 0 and np.isnan(text["in_hand"]), "nothing is in hand before the first click"
    assert "## Diagnostic: the user's own corpus and the positives in hand" in run["summary"]
    assert run["summary"].index("## Diagnostic: the user's own corpus") > run["summary"].index(
        "## The returned set at each balance"
    ), "a diagnostic, after the withheld half's reading (the objective)"
    assert "fresh_fb_share" in run["summary"]


def test_the_balance_metric_peaks_at_the_balance_by_construction(rm) -> None:
    """The oracle F-beta is the best cut's, by brute force; the returned set's share is <= 1."""
    ranks = np.array([0, 2, 3, 7, 30])
    n_pos = 5
    for b in rm.BETAS:
        brute = max(rm.fbeta_of(int((ranks < k).sum()), k, n_pos, b) for k in range(1, 2001))
        assert rm.oracle_fbeta(ranks, n_pos, b) == pytest.approx(brute)
        m = rm.balance_metrics(ranks, 2000, n_pos, b, 8)
        assert m["fb_share"] <= 1.0 + 1e-9 and m["k"] == 8
    assert rm.balance_metrics(ranks, 2000, n_pos, 2.0, None)["k"] == 128, "no recorded count: the balance's cap"
    assert rm.balance_metrics(ranks, 2000, n_pos, 0.5, None)["k"] == 32


def test_a_line_that_keeps_nothing_is_read_as_empty_not_as_the_cap(rm) -> None:
    """#4471: the labels line (#4452) can keep nothing; a recorded 0 is an empty set, -1 or no column is unrecorded."""
    ranks = np.array([0, 2, 3, 7, 30])
    tag = rm.beta_tag(1.0)
    assert rm.frame_beta_k({f"test_line_k_{tag}": 0}, 1.0) == 0
    assert rm.frame_beta_k({f"test_line_k_{tag}": -1}, 1.0) is None
    assert rm.frame_beta_k({}, 1.0) is None
    m = rm.balance_metrics(ranks, 2000, 5, 1.0, 0)
    assert (m["k"], m["recall"], m["fbeta"], m["fb_share"]) == (0, 0.0, 0.0, 0.0)
    assert np.isnan(m["precision"]), "no returned set, no precision"
    assert m["oracle_fbeta"] == pytest.approx(rm.oracle_fbeta(ranks, 5, 1.0))
    assert np.isnan(rm.balance_metrics(np.array([], dtype=int), 2000, 0, 1.0, 0)["fbeta"]), "no positives: undefined"


def test_the_review_reads_the_apps_presets(rm) -> None:
    """#4471: the analyzer's betas are the harness's, which are the app's presets (1/4, 1, 4 since #4448)."""
    from vtscore.training.thresholds import BALANCE_PRESETS

    assert tuple(rm.BETAS) == BALANCE_PRESETS
    assert [rm.beta_tag(b) for b in rm.BETAS] == ["b025", "b1", "b4"]


def test_each_run_records_the_floor_its_sessions_aimed_at(run) -> None:
    """``session_floor`` is the run's CALIB_MIN_PRECISION: pinned to 0.5 here (the default arm is the balance, #4413)."""
    trained = run["cells"][~run["cells"]["never_trained"].astype(bool)]
    assert not trained.empty and (trained["session_floor"] == 0.5).all()


def test_perp_reads_each_p_off_its_own_run(run, tmp_path) -> None:
    """``perp.py`` puts one run per P side by side; with this run given for every P, each row is its own floor."""
    out = tmp_path / "perp"
    a = run["exp"] / "analysis"
    subprocess.run(  # noqa: S603  # fixed argv, repo-local script path, no shell
        [sys.executable, str(_SOTA / "perp.py"), "--run", f"0.1={a}", "--run", f"0.5={a}", "--run", f"0.9={a}",
         "--out", str(out)],
        check=True, capture_output=True, text=True,
    )  # fmt: skip
    assert (out / "returned_at_own_p.png").stat().st_size > 0
    md = (out / "perp_summary.md").read_text()
    assert "| 10% |" in md and "| 90% |" in md and "50% (read at 90%)" in md


def test_perp_reads_each_beta_off_its_own_run(run, tmp_path) -> None:
    """``perp.py --kind balance`` (#4474): the objective first, then the returned set, the early dip and the check."""
    out = tmp_path / "perbeta"
    a = run["exp"] / "analysis"
    subprocess.run(  # noqa: S603  # fixed argv, repo-local script path, no shell
        [sys.executable, str(_SOTA / "perp.py"), "--kind", "balance", "--run", f"0.25={a}", "--run", f"1={a}",
         "--run", f"4={a}", "--out", str(out)],
        check=True, capture_output=True, text=True,
    )  # fmt: skip
    for png in ("returned_at_own_beta.png", "objective_at_own_beta.png"):
        assert (out / png).stat().st_size > 0
    md = (out / "perbeta_summary.md").read_text()
    sections = ["## The objective", "## The returned set at each balance", "## The early dip", "## The spot check"]
    assert all(s_ in md for s_ in sections) and [md.index(s_) for s_ in sections] == sorted(
        md.index(s_) for s_ in sections
    )
    assert "| 0.25 |" in md and "| 4 |" in md


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


def test_an_advisory_check_is_scored_on_the_set_it_audited() -> None:
    """#4474: under the advisory check the range describes the audited set, which is not the set the line keeps."""
    sa = _load("sota_analyze", _SOTA / "analyze.py")
    check = pd.DataFrame(
        [{"t": 160, "floor_count": 8, "check_audited": 16, "range_lo": 0.4, "range_hi": 0.9, "floor_status": "checked"}]
    )
    last = {"n_pool": 100, "pool_pos_ranks": "0 1 2 3 9 12"}  # 4 positives in the top 8, 6 in the top 16
    out = sa._check_columns(check, last, None)
    assert out["check_k"] == 16 and out["check_truth"] == pytest.approx(6 / 16)
    assert out["check_covered"] == 0.0, "0.375 lies below the range"
    floor = check.drop(columns="check_audited").assign(floor_status="confirmed")
    out = sa._check_columns(floor, last, None)
    assert out["check_k"] == 8 and out["check_truth"] == pytest.approx(4 / 8), "a floor walk's end is the kept set"


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
