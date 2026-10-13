"""#4816: price a hand-over judge for the typed query's hold that reads the votes where the two sets disagree.

A typed-query session shows the typed query's set until 30 votes on Binary (#4736) and the head's from the quota
on Region. What is shown changes no pick, so one session set per preset prices every rule: a rule picks, per run
and vote, which set is on screen, and the vote's objective is that set's F-beta on the run's withheld half.
Sessions run with ``CALIB_SET_OVERLAP=1`` (``overlap_4816.sh``) also record, at every click a trained head scored
the pool, how the head's set D and the query's Q split the unvoted media (``__overlap.csv``), and where each pick
sat in them at the previous click (the pick log's ``picked_in_head`` ...). A judge reads those:

* **rate**: each part's share of positives from the votes that fell in it, shrunk toward a prior (the head's own
  posterior over the part, or 1/2) with *a* pseudo-votes; the parts' sizes turn the shares into each set's
  expected positives, and the total P into each set's expected F-beta on the voting collection;
* **calib**: the same, but each part's expected positives are the head's posterior summed over its unvoted media,
  scaled by how its votes came out against what the head predicted for them (observed / predicted Goods). The
  votes then say how wrong the head is in a part, not what the part's rate is, so a pick's position in the part
  (Hard picks sit at the head's line) is read through the head's own posterior;
* **pv**: #4736's progressive-validation judge (each set's F-beta on the votes), for reference;
* **oracle_pool**: the true F-beta of each set on the unvoted pool (the truth columns): what a judge could reach if
  its estimates on the voting collection were exact. Never a rule the app can run.

Every run is in every average (#4631); a run that never trains shows the typed query throughout. Session means are
over votes 1-150; the gain is paired against today, SE clustered on class, and each family's parameters are chosen
on one class half and scored on the other.

    python judge_4816.py extract --cells <run>/results/cells --beta 1 --out X.npz [--region]
    python judge_4816.py price --arm b025=X025.npz --arm b1=X1.npz ... --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "calibration"))
import _cells_io  # noqa: E402

T = 150
GRID = np.arange(T + 1)
PARTS = ("dq", "d_only", "q_only", "none")
HOLD_TAG = "hold:head"
OPENING = ("good", "bad", "more", "example")
MAIN_COLS = ["category", "seed", "t", "phase", "detector_tier", "gmm_variant", "schedule", "pool_variant", "beta",
             "precision", "recall", "average_precision", "app_trained"]  # fmt: skip
PICK_COLS = ["t", "phase", "picked_label", "picked_in_head", "picked_in_query", "picked_head_rank",
             "picked_part_rank", "picked_head_posterior", "n_good"]  # fmt: skip
OV_COLS = ["t", "n_pool", "n_d", "n_q", "n_dq", "est_positives", "head_threshold", "query_cut",
           *[f"pos_{p}" for p in PARTS], *[f"post_{p}" for p in PARTS],
           *[f"good_{p}" for p in PARTS], *[f"bad_{p}" for p in PARTS]]  # fmt: skip
LC_COLS = ["t", "separation", "model_n_pos", "model_n_neg"]
SPANS = {"1-10": (1, 10), "1-25": (1, 25), "1-50": (1, 50), "51-150": (51, 150), "1-150": (1, 150)}


def fbeta_pr(p: np.ndarray, r: np.ndarray, beta: float) -> np.ndarray:
    """F-beta from precision and recall; an empty set (precision NaN, recall 0) scores 0."""
    p = np.asarray(p, dtype=float)
    r = np.asarray(r, dtype=float)
    b2 = beta * beta
    with np.errstate(invalid="ignore", divide="ignore"):
        den = b2 * p + r
        f = np.where(den > 0, (1 + b2) * p * r / den, 0.0)
    return np.where(np.isfinite(p) & np.isfinite(r), f, np.where(r == 0, 0.0, np.nan))


def fbeta_counts(tp, size, pos, beta: float) -> np.ndarray:
    """F-beta of a set of *size* holding *tp* of *pos* positives: (1+b2) tp / (b2 pos + size)."""
    b2 = beta * beta
    tp, size, pos = (np.asarray(x, dtype=float) for x in (tp, size, pos))
    den = b2 * pos + size
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (1 + b2) * tp / den, 0.0)


def carry(t, v) -> np.ndarray:
    """Values at votes *t* on 0..T, carried forward over votes with none; NaN before the first."""
    out = np.full(T + 1, np.nan)
    t = np.asarray(t, dtype=int)
    keep = (t >= 0) & (t <= T)
    out[t[keep]] = np.asarray(v, dtype=float)[keep]
    return pd.Series(out).ffill().to_numpy()


def _side(path: str, suffix: str) -> Path:
    p = Path(path)
    return p.with_name(p.name.replace(".csv", f"__{suffix}.csv", 1))


def _read(path: Path, cols: list[str]) -> pd.DataFrame:
    try:
        return pd.read_csv(path, usecols=lambda c: c in cols, low_memory=False)
    except (pd.errors.EmptyDataError, OSError, ValueError):
        return pd.DataFrame(columns=cols)


def read_cell(args: tuple[str, float]) -> dict | None:
    path, beta = args
    df = _read(Path(path), MAIN_COLS)
    if df.empty:
        return None
    picks = _read(_side(path, "picks"), PICK_COLS)
    ov = _read(_side(path, "overlap"), OV_COLS)
    lc = _read(_side(path, "linecounts"), LC_COLS)
    category, seed = str(df["category"].iloc[0]), int(df["seed"].iloc[0])
    df = df[~_cells_io.check_rows(df) & np.isclose(df["beta"].astype(float), beta) & (df["t"] <= T)]
    df = df.assign(F=fbeta_pr(df["precision"].to_numpy(), df["recall"].to_numpy(), beta))
    tag = df["gmm_variant"].fillna("").astype(str).str.strip()
    base = _cells_io._base_rows(df).drop_duplicates("t", keep="last").sort_values("t")
    head = df[(df["detector_tier"].astype(str) == "trained") & (tag.isin(("", "nan", HOLD_TAG)))]
    head = head[_cells_io._blank(head["schedule"])] if "schedule" in head.columns else head
    head = head.drop_duplicates("t", keep="last").sort_values("t")
    query = base[base["detector_tier"].astype(str) == "query"]
    out: dict = {"category": category, "seed": seed}
    out["tq"] = float(query["F"].iloc[0]) if len(query) else np.nan
    out["tq_sets"] = int(query["F"].round(9).nunique())
    out["det"] = carry(head["t"], head["F"])
    out["ap"] = carry(head["t"], head["average_precision"])
    out["ft"] = float(head["t"].min()) if len(head) else np.inf
    # Today: the untagged row at each vote (the query's set until the hold ends, the head's after), carried.
    today = carry(base["t"], base["F"])
    out["today"] = np.where(np.isnan(today), out["tq"], today)
    out["today_head"] = carry(base["t"], (base["detector_tier"].astype(str) == "trained").astype(float))
    p = picks[picks["t"] <= T].copy()
    last = p.drop_duplicates("t", keep="last")
    out["n_good"] = carry(last["t"], last["n_good"]) if "n_good" in last else np.full(T + 1, np.nan)
    o = ov[ov["t"] <= T].drop_duplicates("t", keep="last")
    for c in OV_COLS[1:]:
        out[f"ov_{c}"] = carry(o["t"], o[c]) if c in o else np.full(T + 1, np.nan)
    lc = lc[lc["t"] <= T].drop_duplicates("t", keep="last")
    for c in LC_COLS[1:]:
        out[f"lc_{c}"] = carry(lc["t"], lc[c]) if c in lc else np.full(T + 1, np.nan)
    out["picks"] = p[["t", "phase", "picked_label", "picked_in_head", "picked_in_query", "picked_head_rank",
                      "picked_part_rank", "picked_head_posterior"]].to_numpy(dtype=object)  # fmt: skip
    return out


SERIES = ("det", "ap", "today", "today_head", "n_good", *[f"ov_{c}" for c in OV_COLS[1:]],
          *[f"lc_{c}" for c in LC_COLS[1:]])  # fmt: skip


def extract(a: argparse.Namespace) -> None:
    cells = [str(p) for p in _cells_io.main_frame_files(a.cells)]
    with ProcessPoolExecutor(a.workers) as ex:
        got = list(ex.map(read_cell, [(c, a.beta) for c in cells], chunksize=4))
    runs = [g for g in got if g is not None]
    print(f"{a.cells}: {len(cells)} cells, {len(runs)} read; query rows not one set: "
          f"{sum(r['tq_sets'] > 1 for r in runs)}; no query row: {sum(np.isnan(r['tq']) for r in runs)}")  # fmt: skip
    pk = [(i, *row) for i, r in enumerate(runs) for row in r["picks"]]
    pk_a = np.array(pk, dtype=object) if pk else np.empty((0, 9), dtype=object)
    np.savez_compressed(
        a.out,
        beta=a.beta,
        region=a.region,
        category=np.array([r["category"] for r in runs]),
        seed=np.array([r["seed"] for r in runs]),
        tq=np.array([r["tq"] for r in runs]),
        ft=np.array([r["ft"] for r in runs]),
        pk_run=pk_a[:, 0].astype(int),
        pk_t=pk_a[:, 1].astype(int),
        pk_phase=pk_a[:, 2].astype(str),
        pk_label=pk_a[:, 3].astype(float),
        pk_in_d=pk_a[:, 4].astype(float),
        pk_in_q=pk_a[:, 5].astype(float),
        pk_rank=pk_a[:, 6].astype(float),
        pk_part_rank=pk_a[:, 7].astype(float),
        pk_post=pk_a[:, 8].astype(float),
        **{k: np.stack([r[k] for r in runs]) for k in SERIES},
    )
    print(f"wrote {a.out}")


def class_half(category: str) -> int:
    return int(hashlib.md5(category.split("@")[0].encode()).hexdigest(), 16) % 2


def first_true(cond: np.ndarray) -> np.ndarray:
    any_ = cond.any(axis=1)
    return np.where(any_, cond.argmax(axis=1).astype(float), np.inf)


class Arm:
    """One preset's sessions: both sets' F per run and vote, the parts' sizes, and the votes in each part."""

    def __init__(self, path: str, *, picks: str = "all") -> None:
        z = np.load(path, allow_pickle=False)
        self.beta = float(z["beta"])
        self.region = bool(z["region"])
        self.category = z["category"]
        self.n = len(self.category)
        self.cls = np.array([c.split("@")[0] for c in self.category])
        self.half = np.array([class_half(c) for c in self.category])
        self.ft = z["ft"].astype(float)
        self.tq = z["tq"].astype(float)
        self.det = z["det"]
        self.has_det = np.isfinite(self.det)
        self.today = z["today"]
        self.today_head = np.nan_to_num(z["today_head"]) > 0.5
        self.trained = GRID[None, :] >= self.ft[:, None]
        self.ov = {c: z[f"ov_{c}"] for c in OV_COLS[1:]}
        self.sep = z["lc_separation"]
        self.npos_model = z["lc_model_n_pos"]
        self.n_good = np.nan_to_num(z["n_good"])
        n = self.n
        # The parts' unvoted sizes at each vote.
        size_d, size_q, both, pool = (self.ov[k] for k in ("n_d", "n_q", "n_dq", "n_pool"))
        self.size = {
            "dq": both,
            "d_only": size_d - both,
            "q_only": size_q - both,
            "none": pool - size_d - size_q + both,
        }
        self.n_d, self.n_q = size_d, size_q
        # The votes in each part, cumulative over votes: count, Goods, and the head's posterior summed.
        run, t = z["pk_run"], z["pk_t"]
        ind, inq, lab, post = z["pk_in_d"], z["pk_in_q"], z["pk_label"], z["pk_post"]
        ok = np.isfinite(ind) & np.isfinite(inq) & (t <= T)
        if picks == "past_opening":
            ok &= ~np.isin(z["pk_phase"], OPENING)
        elif picks == "past_quota":
            ok &= (t - 1) >= self.ft[run]
        part = np.where((ind == 1) & (inq == 1), 0, np.where(ind == 1, 1, np.where(inq == 1, 2, 3)))
        self.m = np.zeros((n, T + 1, 4))
        self.y = np.zeros((n, T + 1, 4))
        self.p = np.zeros((n, T + 1, 4))
        self.p_ok = np.zeros((n, T + 1, 4))
        has_post = np.isfinite(post)
        np.add.at(self.m, (run[ok], t[ok], part[ok]), 1.0)
        np.add.at(self.y, (run[ok], t[ok], part[ok]), lab[ok])
        np.add.at(self.p, (run[ok & has_post], t[ok & has_post], part[ok & has_post]), post[ok & has_post])
        np.add.at(self.p_ok, (run[ok & has_post], t[ok & has_post], part[ok & has_post]), 1.0)
        for arr in (self.m, self.y, self.p, self.p_ok):
            np.cumsum(arr, axis=1, out=arr)
        self.n_picks_read = int(ok.sum())
        self.pk = {"run": run[ok], "t": t[ok], "part": part[ok], "label": lab[ok], "post": post[ok],
                   "phase": z["pk_phase"][ok], "part_rank": z["pk_part_rank"][ok]}  # fmt: skip

    def bias(self, preset: str) -> pd.DataFrame:
        """Per part: the votes' Good rate against the part's true rate at the click before, and the head's posterior.

        The gap between the first two is how far the picks' position in a part moves a rate read off its votes;
        the gap between the first and third is how wrong the head's own posterior is about those votes.
        """
        pk = self.pk
        prev = np.clip(pk["t"] - 1, 0, T)
        rows = []
        for k, name in enumerate(PARTS):
            size = self.size[name][pk["run"], prev]
            true_rate = self.ov[f"pos_{name}"][pk["run"], prev] / np.where(size > 0, size, np.nan)
            depth = pk["part_rank"] / np.where(size > 0, size, np.nan)
            for group, sel in (("all", np.ones(len(prev), bool)), ("hard", pk["phase"] == "hard"),
                               ("not hard", pk["phase"] != "hard")):  # fmt: skip
                at = (pk["part"] == k) & sel & np.isfinite(true_rate)
                rows.append({
                    "preset": preset, "part": name, "picks": group, "votes": int(at.sum()),
                    "votes_per_run": float(at.sum() / self.n),
                    "vote_good_rate": float(pk["label"][at].mean()) if at.any() else np.nan,
                    "part_true_rate": float(true_rate[at].mean()) if at.any() else np.nan,
                    "head_posterior": float(np.nanmean(pk["post"][at])) if at.any() else np.nan,
                    "median_depth_in_part": float(np.nanmedian(depth[at])) if at.any() else np.nan,
                    "part_size_median": float(np.nanmedian(size[at])) if at.any() else np.nan,
                })  # fmt: skip
        return pd.DataFrame(rows)

    # ---- what each judge estimates ------------------------------------------------------------------------
    def estimate(
        self, kind: str, a: float, prior: str, pmode: str, scope: str = "pool"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimated F-beta of D and of Q on the voting collection, per run and vote.

        *scope* ``pool`` reads the unvoted media only; ``all`` adds the voted ones at their known labels, in the
        part this click's sets put them (in sample: the head was fitted to them).
        """
        tp = {}
        for k, name in enumerate(PARTS):
            n_k = self.size[name]
            m, y = self.m[:, :, k], self.y[:, :, k]
            post_sum = self.ov[f"post_{name}"]
            if kind == "rate":
                if prior == "head":
                    pi = np.where(n_k > 0, post_sum / np.maximum(n_k, 1), 0.5)
                    pi = np.where(np.isfinite(pi), pi, 0.5)
                else:
                    pi = np.full_like(m, 0.5)
                r = (y + a * pi) / (m + a) if a > 0 else np.where(m > 0, y / np.maximum(m, 1), pi)
                tp[name] = n_k * r
            elif kind == "calib":
                # Observed Goods against the head's posterior for the same votes, shrunk to 1 with *a* pseudo-votes.
                pv = self.p[:, :, k]
                c = (y + a * 0.5) / (pv + a * 0.5) if a > 0 else np.where(pv > 0, y / np.maximum(pv, 1e-9), 1.0)
                base = np.where(np.isfinite(post_sum), post_sum, 0.5 * n_k)
                tp[name] = c * base
            elif kind in ("oracle_pool", "oracle_all"):
                tp[name] = self.ov[f"pos_{name}"]
            else:
                raise ValueError(kind)
        size_d, size_q = self.n_d, self.n_q
        if kind == "oracle_all" or scope == "all":
            g = {k: np.nan_to_num(self.ov[f"good_{k}"]) for k in PARTS}
            b = {k: np.nan_to_num(self.ov[f"bad_{k}"]) for k in PARTS}
            tp = {k: tp[k] + g[k] for k in PARTS}
            size_d = size_d + g["dq"] + g["d_only"] + b["dq"] + b["d_only"]
            size_q = size_q + g["dq"] + g["q_only"] + b["dq"] + b["q_only"]
        tp_d = tp["dq"] + tp["d_only"]
        tp_q = tp["dq"] + tp["q_only"]
        if kind in ("oracle_pool", "oracle_all") or pmode == "parts":
            pos = tp_d + tp["q_only"] + tp["none"]
        elif pmode == "union":
            pos = tp_d + tp["q_only"]
        elif pmode == "model":
            pos = np.where(np.isfinite(self.ov["est_positives"]), self.ov["est_positives"], tp_d + tp["q_only"])
        else:
            raise ValueError(pmode)
        pos = np.maximum(pos, np.maximum(tp_d, tp_q))
        if pmode == "model" and (kind == "oracle_all" or scope == "all"):
            pos = pos + self.n_good
        return fbeta_counts(tp_d, size_d, pos, self.beta), fbeta_counts(tp_q, size_q, pos, self.beta)

    def pv(self) -> tuple[np.ndarray, np.ndarray]:
        """#4736's judge: each set's F-beta on the votes themselves (progressive validation)."""
        m, y = self.m, self.y
        tp_d = y[:, :, 0] + y[:, :, 1]
        fp_d = (m[:, :, 0] - y[:, :, 0]) + (m[:, :, 1] - y[:, :, 1])
        tp_q = y[:, :, 0] + y[:, :, 2]
        fp_q = (m[:, :, 0] - y[:, :, 0]) + (m[:, :, 2] - y[:, :, 2])
        pos = y.sum(axis=2)
        return fbeta_counts(tp_d, tp_d + fp_d, pos, self.beta), fbeta_counts(tp_q, tp_q + fp_q, pos, self.beta)

    # ---- the sets on screen ---------------------------------------------------------------------------------
    def show(self, head: np.ndarray) -> np.ndarray:
        """Per run and vote, the shown set's F: the head's where *head* says and one exists, else the query's."""
        return np.where(head & self.has_det, self.det, self.tq[:, None])

    def handover(self, h: np.ndarray) -> np.ndarray:
        return self.show(GRID[None, :] >= h[:, None])

    def rules(self):
        """Every rule as ``(name, family, param, per-run per-vote F)``; no rule shows the head before the quota."""
        yield "today", "today", None, self.today
        yield "at_quota", "fixed", "quota", self.handover(self.ft)
        yield "never", "fixed", "never", self.handover(np.full(self.n, np.inf))
        for v in (20, 30, 40, 50):
            yield f"vote_{v}", "vote", v, self.handover(np.maximum(self.ft, float(v)))
        yield (
            "ceiling_switch",
            "ceiling",
            "switch",
            np.where(self.has_det, np.maximum(self.det, self.tq[:, None]), self.tq[:, None]),
        )
        gain = np.where(self.has_det, self.det - self.tq[:, None], 0.0)[:, 1:]
        tail = np.cumsum(gain[:, ::-1], axis=1)[:, ::-1]
        best = np.argmax(np.concatenate([tail, np.zeros((self.n, 1))], axis=1), axis=1) + 1.0
        best = np.where(best > T, np.inf, np.maximum(best, self.ft))
        yield "ceiling_once", "ceiling", "once", self.handover(best)
        mdis = self.m[:, :, 1] + self.m[:, :, 2]
        judges = [
            (kind, (a, prior, pmode, scope))
            for kind, a, prior, pmode, scope in itertools.product(
                ("rate", "calib"), (0.0, 1.0, 3.0, 10.0), ("half", "head"), ("parts", "union", "model"), ("pool", "all")
            )
            if not (kind == "calib" and prior == "head")
        ]
        judges += [("pv", None), ("oracle_pool", None), ("oracle_all", None)]
        # Before a judge is ready: today's rule (the hold on Binary), or the head from the quota.
        fallback = {"today": self.today_head, "quota": self.trained}
        for kind, param in judges:
            fd, fq = self.pv() if kind == "pv" else self.estimate(kind, *(param or (0.0, "half", "parts", "pool")))
            say_raw = fd - fq
            for margin, mmin, fb, mode in itertools.product(
                (-0.05, 0.0, 0.05), (0, 3, 6), ("today", "quota"), ("switch", "once")
            ):
                if kind.startswith("oracle") and (margin != 0.0 or mmin != 0 or fb != "today"):
                    continue
                ready = self.trained & (mdis >= mmin) & np.isfinite(say_raw) & np.isfinite(self.ov["n_pool"])
                say = say_raw >= margin
                if mode == "switch":
                    head = np.where(ready, say, fallback[fb])
                else:
                    # The fallback until the judge is first ready; from then the query until it first says the head.
                    h = first_true(ready & say)
                    r0 = first_true(ready)
                    head = (GRID[None, :] >= h[:, None]) | ((GRID[None, :] < r0[:, None]) & fallback[fb])
                fam = f"{kind}:{mode}"
                yield f"{fam}|{param}|m{margin:+.2f}|n{mmin}|{fb}", fam, (param, margin, mmin, fb), self.show(head)


def _mean_se(x: np.ndarray, cls: np.ndarray) -> tuple[float, float]:
    m = float(np.mean(x))
    by = pd.Series(x - m).groupby(cls).sum().to_numpy()
    n, g = len(x), len(by)
    return m, float(np.sqrt(g / max(g - 1, 1) * np.sum(by**2)) / n)


def span_mean(F: np.ndarray, span: tuple[int, int]) -> np.ndarray:
    lo, hi = span
    return F[:, lo : hi + 1].mean(axis=1)


def reduce(arm: Arm) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Every rule's per-run span means (``{name: (n, spans)}``) and its table row against today."""
    means: dict[str, np.ndarray] = {}
    meta: dict[str, tuple[str, str]] = {}
    changed: dict[str, float] = {}
    today = None
    for name, fam, param, F in arm.rules():
        if today is None:
            today = F
        means[name] = np.stack([span_mean(F, sp) for sp in SPANS.values()], axis=1)
        meta[name] = (fam, str(param))
        changed[name] = float((np.abs(F - today)[:, 1:].max(axis=1) > 1e-12).mean())
    base = means["today"]
    rows = []
    for name, M in means.items():
        r = {"rule": name, "family": meta[name][0], "param": meta[name][1]}
        for j, s in enumerate(SPANS):
            r[f"F_{s}"] = float(M[:, j].mean())
            r[f"d_{s}"], r[f"se_{s}"] = _mean_se(M[:, j] - base[:, j], arm.cls)
        r["runs_changed"] = changed[name]
        rows.append(r)
    return pd.DataFrame(rows), means


def crossfit(arm: Arm, tab: pd.DataFrame, means: dict[str, np.ndarray], fam: str, span: str = "1-150"):
    """The family's rule chosen on one half of the classes and scored on the other, both ways."""
    j = list(SPANS).index(span)
    names = list(tab.loc[tab["family"] == fam, "rule"])
    base = means["today"]
    M = np.empty_like(base)
    picked = []
    for h in (0, 1):
        train = arm.half != h
        best = max(names, key=lambda k: float((means[k][train, j] - base[train, j]).mean()))
        picked.append(best)
        M[arm.half == h] = means[best][arm.half == h]
    return M, picked


def diagnose(arm: Arm, preset: str) -> pd.DataFrame:
    """How well each estimator calls the better set: sign agreement with the withheld truth, weighted by the gap."""
    rows = []
    truth = arm.det - arm.tq[:, None]
    ok_t = arm.trained & arm.has_det & np.isfinite(arm.ov["n_pool"])
    pool_d, pool_q = arm.estimate("oracle_pool", 0.0, "half", "parts")
    cands = {"oracle_pool": (pool_d, pool_q), "oracle_all": arm.estimate("oracle_all", 0.0, "half", "parts"),
             "pv": arm.pv()}  # fmt: skip
    for kind, a, prior, pmode in (("rate", 1.0, "half", "parts"), ("rate", 3.0, "head", "parts"), ("rate", 3.0, "head", "model"),
                                  ("calib", 1.0, "half", "parts"), ("calib", 3.0, "half", "model")):  # fmt: skip
        for scope in ("pool", "all"):
            cands[f"{kind}|{a}|{prior}|{pmode}|{scope}"] = arm.estimate(kind, a, prior, pmode, scope)
    for lo, hi in ((1, 30), (31, 150), (1, 150)):
        win = ok_t & (GRID[None, :] >= lo) & (GRID[None, :] <= hi)
        for name, (fd, fq) in cands.items():
            est = fd - fq
            w = win & np.isfinite(est)
            agree = np.sign(est[w]) == np.sign(truth[w])
            wt = np.abs(truth[w])
            rows.append({
                "preset": preset, "votes": f"{lo}-{hi}", "estimator": name, "n": int(w.sum()),
                "agree": float(agree.mean()) if w.any() else np.nan,
                "agree_weighted": float((agree * wt).sum() / wt.sum()) if w.any() and wt.sum() > 0 else np.nan,
                "corr": float(np.corrcoef(est[w], truth[w])[0, 1]) if w.sum() > 2 else np.nan,
                "says_head": float((est[w] >= 0).mean()) if w.any() else np.nan,
                "head_better": float((truth[w] >= 0).mean()) if w.any() else np.nan,
            })  # fmt: skip
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, digits: int = 3) -> str:
    """A markdown table (pandas' ``to_markdown`` needs tabulate, which the venv lacks)."""

    def cell(v) -> str:
        if isinstance(v, float):
            return "" if np.isnan(v) else f"{v:.{digits}f}"
        return str(v)

    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def price(a: argparse.Namespace) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tables, xf_rows, diags, biases = [], [], [], []
    for spec in a.arm:
        preset, path = spec.split("=", 1)
        arm = Arm(path, picks=a.picks)
        print(f"{preset}: {arm.n} runs; trained {np.isfinite(arm.ft).mean():.3f}; picks read {arm.n_picks_read}; "
              f"votes in D only / Q only at 150: {arm.m[:, T, 1].mean():.1f} / {arm.m[:, T, 2].mean():.1f}")  # fmt: skip
        tab, means = reduce(arm)
        tab.insert(0, "preset", preset)
        tables.append(tab)
        diags.append(diagnose(arm, preset))
        biases.append(arm.bias(preset))
        base = means["today"]
        for fam in sorted(set(tab["family"]) - {"today", "ceiling", "fixed"}):
            M, picked = crossfit(arm, tab, means, fam)
            r = {"preset": preset, "family": fam, "picked": " / ".join(picked)}
            for j, s_ in enumerate(SPANS):
                r[f"d_{s_}"], r[f"se_{s_}"] = _mean_se(M[:, j] - base[:, j], arm.cls)
            xf_rows.append(r)
    tab = pd.concat(tables)
    tab.to_csv(out / "rules.csv", index=False)
    xf = pd.DataFrame(xf_rows)
    xf.to_csv(out / "crossfit.csv", index=False)
    dg = pd.concat(diags)
    dg.to_csv(out / "diagnose.csv", index=False)
    bs = pd.concat(biases)
    bs.to_csv(out / "bias.csv", index=False)
    fmt = lambda v: f"{v:+.4f}"  # noqa: E731
    with open(out / "summary.md", "w") as f:
        f.write(f"picks read: {a.picks}\n")
        for preset, g in tab.groupby("preset", sort=False):
            f.write(f"\n## {preset}: rules vs today (paired, SE clustered on class)\n\n")
            g = g.sort_values("d_1-150", ascending=False)
            f.write(
                "| rule | F 1-150 | d 1-10 | d 1-25 | d 1-50 | d 51-150 | d 1-150 | runs changed |\n|---|---|---|---|---|---|---|---|\n"
            )
            keep = g.head(a.top).to_dict("records") + g[
                g["family"].isin(["ceiling", "fixed", "vote", "today"])
            ].to_dict("records")
            for r in keep:
                f.write(f"| {r['rule']} | {r['F_1-150']:.4f} | {fmt(r['d_1-10'])} | {fmt(r['d_1-25'])} | {fmt(r['d_1-50'])} | "
                        f"{fmt(r['d_51-150'])} | {fmt(r['d_1-150'])} ± {r['se_1-150']:.4f} | {r['runs_changed']:.2f} |\n")  # fmt: skip
        f.write("\n## Cross-fitted families (chosen on one class half, scored on the other)\n\n")
        f.write("| preset | family | d 1-25 | d 1-50 | d 1-150 | picked |\n|---|---|---|---|---|---|\n")
        for r in xf.to_dict("records"):
            f.write(f"| {r['preset']} | {r['family']} | {fmt(r['d_1-25'])} ± {r['se_1-25']:.4f} | {fmt(r['d_1-50'])} ± {r['se_1-50']:.4f} | "
                    f"{fmt(r['d_1-150'])} ± {r['se_1-150']:.4f} | {r['picked']} |\n")  # fmt: skip
        f.write("\n## Estimators: sign agreement with the withheld truth (head minus query)\n\n")
        f.write(md_table(dg))
        f.write("\n\n## The votes in each part against the part's truth and the head's posterior\n\n")
        f.write(md_table(bs))
        f.write("\n")
    print((out / "summary.md").read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--cells", required=True)
    e.add_argument("--beta", type=float, required=True)
    e.add_argument("--region", action="store_true")
    e.add_argument("--workers", type=int, default=16)
    e.add_argument("--out", required=True)
    p = sub.add_parser("price")
    p.add_argument("--arm", action="append", required=True, help="PRESET=extract.npz, repeatable")
    p.add_argument("--picks", default="all", choices=("all", "past_opening", "past_quota"))
    p.add_argument("--out", required=True)
    p.add_argument("--top", type=int, default=15)
    a = ap.parse_args()
    extract(a) if a.cmd == "extract" else price(a)


if __name__ == "__main__":
    main()
