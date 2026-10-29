"""One session cell with the beta > 1 line's reach logged, and optionally a hard-view Good kept out of it (#5409).

#5376: one hard-view starting face (Face Quality under #5372's cut) costs a four-photo session about 0.1 F at
50 votes at beta 4, and the line returns about twice the faces. #5389 and #5413 ruled out the example centroid
and recovered a third through the trained head's fit. A face session is on the head's count line all session
(:func:`vtscore.training.thresholds.labels_line.head_count_cut`), and above a balance of 1 that line reaches down
to the head's excess-curve depth whenever one of the class model's held-out Goods ranks below the beta 1 count
(#5039). A hard Good held out in a calibration fold scores low. This runner tags each held-out Good with its face
and logs every beta > 1 cut: the held-out Goods, which are hard, the count without the reach, and the cut with
and without the hard ones in it.

``--arm off`` changes nothing (the control, logging); ``--arm out`` leaves the hard-view Goods out of the reach's
lowest-Good test, while they stay Goods everywhere else (the fit, the class model, the counts).

    python hard_good_line_5409.py --index N [--arm off|out] [--log PATH]

Run from ``scripts/experiments/calibration`` in a run dir's environment (``sota_face.sh cell`` with
``SOTA_FACE_RUNNER`` pointing here, ``HARD_GOOD_LINE`` the arm and ``HARD_GOOD_LOG`` the JSON-lines log).
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CALIB = HERE.parent / "calibration"


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


class Tags:
    """The latest calibration folds' held-out Goods as ``(logit, is_hard)``, and the session's vote count."""

    def __init__(self, hard_of: dict[bytes, bool]) -> None:
        self.hard_of = hard_of
        self.goods: list[tuple[float, bool]] = []
        self.n_labels: int | None = None
        self.stats = {"fold_calls": 0, "cuts": 0, "reach_hard_only": 0, "unmatched": 0, "missing_rows": 0}

    def folds(self, original):
        def compute_fold_orderings(X_list, y_list, *args, holdout_sink=None, **kwargs):
            sink = [] if holdout_sink is None else holdout_sink
            start = len(sink)
            orderings, fallback = original(X_list, y_list, *args, holdout_sink=sink, **kwargs)
            self.stats["fold_calls"] += 1
            goods: list[tuple[float, bool]] = []
            for (scores, labels), rows in zip(orderings, sink[start:]):
                for s, lab, r in zip(scores, labels, rows):
                    if float(lab) < 0.5 or not (0.0 <= float(s) <= 1.0):
                        continue
                    key = np.asarray(X_list[r], dtype=np.float32).ravel().tobytes()
                    if key not in self.hard_of:
                        self.stats["missing_rows"] += 1
                    goods.append((float(_logit(np.array([s]))[0]), bool(self.hard_of.get(key, False))))
            self.goods = goods
            return orderings, fallback

        return compute_fold_orderings

    def hard_mask(self, goods: np.ndarray) -> np.ndarray:
        """Which of the class model's held-out Good logits *goods* are hard views (nearest tagged logit)."""
        if not self.goods:
            self.stats["unmatched"] += int(goods.size)
            return np.zeros(goods.size, dtype=bool)
        tagged = np.array([g for g, _ in self.goods])
        flags = np.array([h for _, h in self.goods])
        out = np.zeros(goods.size, dtype=bool)
        for i, g in enumerate(goods):
            j = int(np.argmin(np.abs(tagged - g)))
            if abs(tagged[j] - g) > 1e-6:
                self.stats["unmatched"] += 1
            else:
                out[i] = flags[j]
        return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--arm", choices=("off", "out"), default=os.environ.get("HARD_GOOD_LINE", "off"))
    ap.add_argument("--log", default=os.environ.get("HARD_GOOD_LOG"))
    args = ap.parse_args(argv)
    sys.path.insert(0, str(CALIB))
    sys.path.insert(0, str(HERE))
    import common  # noqa: PLC0415
    import experiment_config as cfg  # noqa: PLC0415
    import run_cells  # noqa: PLC0415
    from hard_face_train_weight_5413 import _hard_by_vector  # noqa: PLC0415

    import vtscore.training.thresholds as thresholds  # noqa: PLC0415
    from vtscore.eval import step_trainers, voting_iterations  # noqa: PLC0415
    from vtscore.training.thresholds import conformal, labels_line  # noqa: PLC0415

    info = json.loads((common.RESULTS / "prepare_info.json").read_text())
    cell = cfg.array_cells(run_cells._categories_by_dataset(info))[args.index]
    tags = Tags(_hard_by_vector(cell["dataset"]))
    log = open(args.log, "w") if args.log else None  # noqa: SIM115

    folds = tags.folds(conformal.compute_fold_orderings)
    for mod in (conformal, thresholds, step_trainers):
        if hasattr(mod, "compute_fold_orderings"):
            mod.compute_fold_orderings = folds

    fit_line = voting_iterations.fit_labels_line

    def fit_labels_line(orderings, corpus_scores, corpus_ids=None, labels=None):
        tags.n_labels = len(labels or {})
        return fit_line(orderings, corpus_scores, corpus_ids, labels)

    voting_iterations.fit_labels_line = fit_labels_line

    cut = labels_line.head_count_cut

    def head_count_cut(scores_desc, beta, held_out_goods=None):
        if held_out_goods is None or not float(beta) > 1.0:
            return cut(scores_desc, beta, held_out_goods)
        goods = np.asarray(held_out_goods, dtype=np.float64).ravel()
        hard = tags.hard_mask(goods)
        clear = goods[~hard]
        s = np.asarray(scores_desc, dtype=np.float64)
        x = _logit(s)

        def kept(thr: float) -> int:
            return int(np.count_nonzero(s >= thr))

        t_all = cut(scores_desc, beta, goods)
        t_out = cut(scores_desc, beta, clear)
        from vtscore.detectors.centroid_head import centroid_cut  # noqa: PLC0415

        standouts = int(np.count_nonzero(x >= centroid_cut(x, beta=1.0))) if x.size else 0
        k_all, k_out = kept(t_all), kept(t_out)
        tags.stats["cuts"] += 1
        if k_all != k_out:
            tags.stats["reach_hard_only"] += 1
        if log is not None:
            rec = {
                "n_labels": tags.n_labels,
                "beta": float(beta),
                "n_unvoted": int(s.size),
                "standouts": standouts,
                "k_noreach": kept(cut(scores_desc, beta, None)),
                "k_all": k_all,
                "k_out": k_out,
                "goods": [round(float(g), 4) for g in goods],
                "hard": [bool(h) for h in hard],
                "good_ranks": [int(np.count_nonzero(x >= g)) for g in goods],
            }
            log.write(json.dumps(rec) + "\n")
        return t_out if args.arm == "out" else t_all

    labels_line.head_count_cut = head_count_cut

    def done() -> None:
        if log is not None:
            log.close()
        print(
            "hard_good_line_5409: " + json.dumps({"index": args.index, "arm": args.arm, **tags.stats}),
            file=sys.stderr,
            flush=True,
        )

    atexit.register(done)
    return run_cells.main(["--index", str(args.index)])


if __name__ == "__main__":
    sys.exit(main())
