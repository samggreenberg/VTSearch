"""#4736: price a hand-over rule from the typed query's set to the trained head, offline, on full sessions.

A typed-query session shows the typed query's own set at its display line (#4603, #4758) until its labels give
the trained head (the label quota: 3 Goods + 4 Bads, or a Good + 16 Bads, #4643/#4731), and the head's set from
that vote on. What is shown changes no pick: the opening's picks come off the text sort, Hard's and New's off the
head, which the harness trains at every vote from the quota whether it is shown or not. So one session set per
preset prices every hand-over rule exactly: a rule picks, per run and vote, which of the two sets is on screen,
and the vote's objective is that set's F-beta on the run's withheld half.

- **the head's set** is the run's untagged ``trained`` rows (``fbeta`` at the run's beta), carried over votes with
  no row (a check round writes one row for its five votes).
- **the typed query's set** is ``text_baseline.csv``'s ``text_line_*_b<beta>`` for the cell and seed: the display
  line at the run's preset, drawn over the haystack and scored on the withheld half. It does not move with votes.

Every run is in every average (#4631): a run that never reaches the quota shows the typed query throughout, under
every rule. Session means are over votes 1-150 (the runs carry the last row to 150 after a stop).

    python handover_4736.py extract --cells <run>/results/cells --baseline <text_baseline.csv> --beta 1 --out X.npz
    python handover_4736.py price --arm b025=X025.npz --arm b1=X1.npz --arm b4=X4.npz --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
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
TRAINED = "trained"
BASE_COLS = [
    "category", "seed", "t", "phase", "n_good", "n_bad", "app_trained", "detector_tier", "beta", "threshold",
    "fbeta", "precision", "recall", "average_precision", "n_flagged", "gmm_variant", "pool_variant", "schedule",
]  # fmt: skip
PICK_COLS = ["t", "phase", "picked_label", "picked_seed_score", "picked_detector_score", "n_good", "n_bad"]
LC_COLS = ["t", "separation", "model_n_pos", "model_n_neg", "est_positives"]
SERIES = ("det", "ap", "thr", "sep", "npos_model", "nneg_model", "est_pos", "n_good", "n_bad")


def btag(beta: float) -> str:
    return {0.25: "b025", 1.0: "b1", 4.0: "b4"}[float(beta)]


def fbeta_pr(p: np.ndarray, r: np.ndarray, beta: float) -> np.ndarray:
    """F-beta from precision and recall; an empty set (precision NaN, recall 0) scores 0."""
    p = np.asarray(p, dtype=float)
    r = np.asarray(r, dtype=float)
    b2 = beta * beta
    with np.errstate(invalid="ignore", divide="ignore"):
        den = b2 * p + r
        f = np.where(den > 0, (1 + b2) * p * r / den, 0.0)
    return np.where(np.isfinite(p) & np.isfinite(r), f, np.where(r == 0, 0.0, np.nan))


def counts_fbeta(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, beta: float) -> np.ndarray:
    b2 = beta * beta
    den = (1 + b2) * tp + b2 * fn + fp
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (1 + b2) * tp / den, 0.0)


def carry(t: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Values at votes *t* on 0..T, carried forward over votes with none; NaN before the first."""
    out = np.full(T + 1, np.nan)
    t = np.asarray(t, dtype=int)
    keep = (t >= 0) & (t <= T)
    out[t[keep]] = np.asarray(v, dtype=float)[keep]
    return pd.Series(out).ffill().to_numpy()


def read_cell(args: tuple[str, float]) -> dict | None:
    path, beta = args
    try:
        df = pd.read_csv(path, usecols=lambda c: c in BASE_COLS, low_memory=False)
        picks = pd.read_csv(path.replace(".csv", "__picks.csv"), usecols=lambda c: c in PICK_COLS)
    except (pd.errors.EmptyDataError, OSError, ValueError):
        return None
    if df.empty:
        return None
    lc_path = Path(path.replace(".csv", "__linecounts.csv"))
    try:
        lc = pd.read_csv(lc_path, usecols=lambda c: c in LC_COLS) if lc_path.exists() else pd.DataFrame()
    except (pd.errors.EmptyDataError, ValueError):
        lc = pd.DataFrame()
    category, seed = str(df["category"].iloc[0]), int(df["seed"].iloc[0])
    df = _cells_io._base_rows(df)
    df = df[~_cells_io.check_rows(df) & np.isclose(df["beta"].astype(float), beta)]
    df = df[df["t"] <= T].drop_duplicates("t", keep="last").sort_values("t")
    tr = df[df["detector_tier"].astype(str) == TRAINED]
    f = fbeta_pr(tr["precision"].to_numpy(), tr["recall"].to_numpy(), beta)
    out: dict = {"category": category, "seed": seed}
    out["det"] = carry(tr["t"], f)
    out["ap"] = carry(tr["t"], tr["average_precision"])
    out["thr"] = carry(tr["t"], tr["threshold"])
    out["first_trained"] = float(tr["t"].min()) if len(tr) else np.inf
    on = df.loc[df["app_trained"] == 1, "t"]
    out["app_from"] = float(on.min()) if len(on) else np.inf
    p = picks[picks["t"] <= T]
    last = p.drop_duplicates("t", keep="last")
    out["n_good"] = carry(last["t"], last["n_good"])
    out["n_bad"] = carry(last["t"], last["n_bad"])
    if len(lc):
        lc = lc[lc["t"] <= T].drop_duplicates("t", keep="last")
        out["sep"] = carry(lc["t"], lc["separation"])
        out["npos_model"] = carry(lc["t"], lc["model_n_pos"])
        out["nneg_model"] = carry(lc["t"], lc["model_n_neg"])
        out["est_pos"] = carry(lc["t"], lc["est_positives"])
    else:
        for k in ("sep", "npos_model", "nneg_model", "est_pos"):
            out[k] = np.full(T + 1, np.nan)
    # First-trained tier's source: the first quota (3G+4B) or the dry exit's (a Good + 16 Bads, #4731).
    ft = out["first_trained"]
    out["dry_exit"] = bool(np.isfinite(ft) and out["n_good"][int(ft)] < 3)
    out["picks"] = p[["t", "phase", "picked_label", "picked_seed_score", "picked_detector_score"]].to_numpy()
    return out


def extract(a: argparse.Namespace) -> None:
    cells = sorted(str(p) for p in Path(a.cells).glob("task_[0-9]*.csv") if "__" not in p.name)
    with ProcessPoolExecutor(a.workers) as ex:
        got = list(ex.map(read_cell, [(c, a.beta) for c in cells], chunksize=4))
    runs = [g for g in got if g is not None]
    print(f"{a.cells}: {len(cells)} cells, {len(runs)} read")
    base = pd.read_csv(a.baseline)
    base = base[base["embedder"] == a.text_embedder].drop_duplicates(["category", "seed"])
    tag = btag(a.beta)
    key = {(c, int(s)): r for c, s, r in zip(base["category"], base["seed"], base.to_dict("records"))}
    n = len(runs)
    tq = np.full(n, np.nan)
    tq_cut = np.full(n, np.nan)
    for i, r in enumerate(runs):
        b = key.get((r["category"], r["seed"]))
        if b is not None:
            tq[i] = fbeta_pr(
                np.array([b[f"text_line_precision_{tag}"]]), np.array([b[f"text_line_recall_{tag}"]]), a.beta
            )[0]
            tq_cut[i] = b[f"text_line_cut_{tag}"]
    print(f"typed query missing for {int(np.isnan(tq).sum())} runs")
    arrays = {k: np.stack([r[k] for r in runs]) for k in SERIES}
    pk_rows = []
    for i, r in enumerate(runs):
        for t, ph, lab, ss, ds in r["picks"]:
            pk_rows.append((i, int(t), str(ph), float(lab), float(ss), float(ds)))
    pk = np.array(pk_rows, dtype=object)
    np.savez_compressed(
        a.out,
        beta=a.beta,
        category=np.array([r["category"] for r in runs]),
        seed=np.array([r["seed"] for r in runs]),
        first_trained=np.array([r["first_trained"] for r in runs]),
        app_from=np.array([r["app_from"] for r in runs]),
        dry_exit=np.array([r["dry_exit"] for r in runs]),
        tq=tq,
        tq_cut=tq_cut,
        pk_run=pk[:, 0].astype(int),
        pk_t=pk[:, 1].astype(int),
        pk_phase=pk[:, 2].astype(str),
        pk_label=pk[:, 3].astype(float),
        pk_seed=pk[:, 4].astype(float),
        pk_det=pk[:, 5].astype(float),
        **arrays,
    )
    print(f"wrote {a.out}")


SPANS = {"1-10": (1, 10), "1-25": (1, 25), "1-50": (1, 50), "51-150": (51, 150), "1-150": (1, 150)}


def class_half(category: str) -> int:
    return int(hashlib.md5(category.split("@")[0].encode()).hexdigest(), 16) % 2


def first_true(cond: np.ndarray) -> np.ndarray:
    """Per row, the first vote where *cond* holds; inf where it never does."""
    any_ = cond.any(axis=1)
    return np.where(any_, cond.argmax(axis=1).astype(float), np.inf)


class Arm:
    """One preset's sessions: the two sets' F-beta per run and vote, and the votes a judge may read."""

    def __init__(self, path: str) -> None:
        z = np.load(path, allow_pickle=False)
        self.beta = float(z["beta"])
        self.category = z["category"]
        self.seed = z["seed"]
        self.n = len(self.category)
        self.ft = z["first_trained"].astype(float)
        self.dry = z["dry_exit"].astype(bool)
        self.tq = z["tq"].astype(float)
        self.det = z["det"]
        self.thr = z["thr"]
        self.sep = z["sep"]
        self.npos_model = z["npos_model"]
        self.n_good = np.nan_to_num(z["n_good"])
        self.half = np.array([class_half(c) for c in self.category])
        self.cls = np.array([c.split("@")[0] for c in self.category])
        ok = np.isfinite(self.tq)
        if not ok.all():
            print(f"dropping {int((~ok).sum())} runs with no typed-query baseline")
        self.has_det = np.isfinite(self.det)
        self._votes(z)

    def _votes(self, z) -> None:
        """Cumulative F-beta of each set on the picks the head scored before their vote (progressive validation).

        A pick at vote s was ranked by the head trained after vote s-1, so it is judged against that head's line;
        the typed query's call is its cosine against the display line's cut. Only picks after the first trained
        vote count: the opening's picks came off the typed query's own sort.
        """
        n = self.n
        run, t = z["pk_run"], z["pk_t"]
        lab, ss, ds = z["pk_label"], z["pk_seed"], z["pk_det"]
        prev = np.clip(t - 1, 0, T)
        th = self.thr[run, prev]
        judged = np.isfinite(ds) & np.isfinite(th) & (prev >= self.ft[run])
        dcall = (ds >= th).astype(float)
        qcall = (ss >= z["tq_cut"][run]).astype(float)
        acc = {k: np.zeros((n, T + 1)) for k in ("m", "pos", "tpd", "fpd", "fnd", "tpq", "fpq", "fnq")}
        sel = judged
        r, tt, y, d, q = run[sel], t[sel], lab[sel], dcall[sel], qcall[sel]
        for k, v in (
            ("m", np.ones_like(y)),
            ("pos", y),
            ("tpd", y * d),
            ("fpd", (1 - y) * d),
            ("fnd", y * (1 - d)),
            ("tpq", y * q),
            ("fpq", (1 - y) * q),
            ("fnq", y * (1 - q)),
        ):
            np.add.at(acc[k], (r, tt), v)
        acc = {k: v.cumsum(axis=1) for k, v in acc.items()}
        self.v_m, self.v_pos = acc["m"], acc["pos"]
        self.v_fd = counts_fbeta(acc["tpd"], acc["fpd"], acc["fnd"], self.beta)
        self.v_fq = counts_fbeta(acc["tpq"], acc["fpq"], acc["fnq"], self.beta)
        self.v_judged = int(judged.sum())
        self.v_dcall_rate = float(dcall[judged].mean()) if judged.any() else np.nan
        self.v_label_rate = float(lab[judged].mean()) if judged.any() else np.nan

    def handover(self, h: np.ndarray) -> np.ndarray:
        """Per run and vote: the head's F-beta from hand-over vote *h* on (once the head exists), else the query's."""
        use = (GRID[None, :] >= h[:, None]) & self.has_det
        return np.where(use, self.det, self.tq[:, None])

    def switch(self, show_head: np.ndarray) -> np.ndarray:
        return np.where(show_head & self.has_det, self.det, self.tq[:, None])

    def rules(self) -> dict[str, tuple[str, object, np.ndarray]]:
        """``{name: (family, param, per-run per-vote F)}``; every rule hands over no earlier than the quota."""
        ft = self.ft
        out: dict[str, tuple[str, object, np.ndarray]] = {}
        out["today"] = ("today", None, self.handover(ft))
        out["never"] = ("never", None, self.handover(np.full(self.n, np.inf)))
        out["ceiling_switch"] = (
            "ceiling",
            "switch",
            np.where(self.has_det, np.maximum(self.det, self.tq[:, None]), self.tq[:, None]),
        )
        # The best single hand-over vote per run, with hindsight.
        gain = np.where(self.has_det, self.det - self.tq[:, None], 0.0)[:, 1:]
        tail = np.cumsum(gain[:, ::-1], axis=1)[:, ::-1]  # tail[:, j] = gain summed over votes j+1..150
        best = np.argmax(np.concatenate([tail, np.zeros((self.n, 1))], axis=1), axis=1) + 1.0
        best = np.where(best > T, np.inf, np.maximum(best, ft))
        out["ceiling_once"] = ("ceiling", "once", self.handover(best))
        for k in (1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100):
            out[f"after_{k}"] = ("after", k, self.handover(ft + k))
        for v in (10, 15, 20, 25, 30, 40, 50, 75, 100):
            out[f"vote_{v}"] = ("vote", v, self.handover(np.maximum(ft, float(v))))
        for g in (4, 5, 6, 8, 10, 12, 15, 20, 25, 30):
            out[f"goods_{g}"] = ("goods", g, self.handover(np.maximum(ft, first_true(self.n_good >= g))))
        trained = GRID[None, :] >= ft[:, None]
        for p in (2, 3, 4, 5, 6, 8, 10):
            cond = trained & (np.nan_to_num(self.npos_model) >= p)
            out[f"model_goods_{p}"] = ("model_goods", p, self.handover(first_true(cond)))
        for d in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
            cond = trained & (np.nan_to_num(self.sep) >= d)
            out[f"sep_{d:g}"] = ("sep", d, self.handover(first_true(cond)))
        for margin in (-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2):
            for m in (1, 3, 5, 10):
                for p in (1, 2, 3):
                    cond = trained & (self.v_m >= m) & (self.v_pos >= p) & (self.v_fd >= self.v_fq + margin)
                    out[f"votes_m{margin:+.2f}_n{m}_p{p}"] = ("votes", (margin, m, p), self.handover(first_true(cond)))
                    if m in (3, 5) and p in (1, 3):
                        ready = (self.v_m >= m) & (self.v_pos >= p)
                        show = trained & np.where(ready, self.v_fd >= self.v_fq + margin, False)
                        out[f"votesw_m{margin:+.2f}_n{m}_p{p}"] = ("votes_switch", (margin, m, p), self.switch(show))
        return out


def _mean_se(x: np.ndarray, cls: np.ndarray) -> tuple[float, float]:
    """Mean of per-run *x* and its SE clustered on the class (a class's bands and seeds share a target)."""
    m = float(np.mean(x))
    by = pd.Series(x - m).groupby(cls).sum().to_numpy()
    n = len(x)
    g = len(by)
    se = float(np.sqrt(g / max(g - 1, 1) * np.sum(by**2)) / n)
    return m, se


def span_mean(F: np.ndarray, span: tuple[int, int]) -> np.ndarray:
    lo, hi = span
    return F[:, lo : hi + 1].mean(axis=1)


def summarize(arm: Arm, rules: dict, preset: str) -> pd.DataFrame:
    today = rules["today"][2]
    rows = []
    for name, (fam, param, F) in rules.items():
        r = {"preset": preset, "rule": name, "family": fam, "param": str(param)}
        for s, span in SPANS.items():
            r[f"F_{s}"] = float(span_mean(F, span).mean())
            m, se = _mean_se(span_mean(F, span) - span_mean(today, span), arm.cls)
            r[f"d_{s}"], r[f"se_{s}"] = m, se
        diff = np.abs(F - today)[:, 1:].max(axis=1) > 1e-12
        r["runs_changed"] = float(diff.mean())
        rows.append(r)
    return pd.DataFrame(rows)


def crossfit(arm: Arm, rules: dict, fam: str, span: str = "1-150") -> tuple[np.ndarray, list]:
    """The family's rule chosen on one half of the classes and scored on the other, both ways."""
    names = [k for k, v in rules.items() if v[0] == fam]
    today = rules["today"][2]
    F = np.empty_like(today)
    picked = []
    for h in (0, 1):
        train = arm.half != h
        gains = {
            k: float((span_mean(rules[k][2], SPANS[span]) - span_mean(today, SPANS[span]))[train].mean()) for k in names
        }
        best = max(gains, key=gains.get)
        picked.append(best)
        F[arm.half == h] = rules[best][2][arm.half == h]
    return F, picked


def price(a: argparse.Namespace) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tables, xf_rows, curves = [], [], []
    for spec in a.arm:
        preset, path = spec.split("=", 1)
        arm = Arm(path)
        print(
            f"{preset}: {arm.n} runs; trained {np.isfinite(arm.ft).mean():.3f}; dry exit {arm.dry.mean():.3f}; "
            f"judged picks {arm.v_judged} (head says yes {arm.v_dcall_rate:.2f}, Goods {arm.v_label_rate:.2f})"
        )
        rules = arm.rules()
        tab = summarize(arm, rules, preset)
        tables.append(tab)
        today = rules["today"][2]
        for fam in sorted({v[0] for v in rules.values()} - {"today", "never", "ceiling"}):
            F, picked = crossfit(arm, rules, fam)
            r = {"preset": preset, "family": fam, "picked": " / ".join(picked)}
            for s, span in SPANS.items():
                r[f"d_{s}"], r[f"se_{s}"] = _mean_se(span_mean(F, span) - span_mean(today, span), arm.cls)
            xf_rows.append(r)
        for sub, mask in (
            ("dry_exit", arm.dry),
            ("quota", np.isfinite(arm.ft) & ~arm.dry),
            ("all", np.ones(arm.n, bool)),
        ):
            for name in ("never", "ceiling_switch", "ceiling_once"):
                d = rules[name][2][mask] - today[mask]
                for s, span in SPANS.items():
                    m, se = _mean_se(span_mean(d, span), arm.cls[mask]) if mask.any() else (np.nan, np.nan)
                    xf_rows.append(
                        {
                            "preset": preset,
                            "family": f"{sub}:{name}",
                            "picked": f"n={int(mask.sum())}",
                            f"d_{s}": m,
                            f"se_{s}": se,
                        }
                    )
        for name in a.curve_rules.split(","):
            if name in rules:
                c = rules[name][2].mean(axis=0)
                curves.append(pd.DataFrame({"preset": preset, "rule": name, "t": GRID, "F": c}))
        curves.append(
            pd.DataFrame({"preset": preset, "rule": "share_head_exists", "t": GRID, "F": arm.has_det.mean(axis=0)})
        )
    tab = pd.concat(tables)
    tab.to_csv(out / "rules.csv", index=False)
    xf = pd.DataFrame(xf_rows)
    xf = xf.groupby(["preset", "family", "picked"], as_index=False, sort=False).first()
    xf.to_csv(out / "crossfit.csv", index=False)
    pd.concat(curves).to_csv(out / "curves.csv", index=False)
    fmt = lambda v: f"{v:+.4f}"  # noqa: E731
    with open(out / "summary.md", "w") as f:
        for preset, g in tab.groupby("preset", sort=False):
            f.write(f"\n## {preset}: every rule vs today (paired, SE clustered on class)\n\n")
            g = g.sort_values("d_1-150", ascending=False)
            f.write(
                "| rule | F 1-150 | d 1-10 | d 1-25 | d 1-50 | d 51-150 | d 1-150 | runs changed |\n|---|---|---|---|---|---|---|---|\n"
            )
            for r in g.head(a.top).to_dict("records") + g[g["rule"].isin(["today", "never"])].to_dict("records"):
                f.write(
                    f"| {r['rule']} | {r['F_1-150']:.4f} | {fmt(r['d_1-10'])} | {fmt(r['d_1-25'])} | {fmt(r['d_1-50'])} | "
                    f"{fmt(r['d_51-150'])} | {fmt(r['d_1-150'])} ± {r['se_1-150']:.4f} | {r['runs_changed']:.2f} |\n"
                )
        f.write("\n## Cross-fitted families (chosen on one class half, scored on the other)\n\n")
        f.write("| preset | family | picked | d 1-25 | d 1-50 | d 1-150 |\n|---|---|---|---|---|---|\n")
        for r in xf.to_dict("records"):
            f.write(
                f"| {r['preset']} | {r['family']} | {r['picked']} | {fmt(r['d_1-25'])} ± {r['se_1-25']:.4f} | "
                f"{fmt(r['d_1-50'])} ± {r['se_1-50']:.4f} | {fmt(r['d_1-150'])} ± {r['se_1-150']:.4f} |\n"
            )
    print((out / "summary.md").read_text())


HOLD_TAG = "hold:head"
QUERY = "query"
ARM_COLS = [*BASE_COLS, "gmm_variant"]


def _rows_by_t(df: pd.DataFrame, beta: float) -> pd.DataFrame:
    df = df[~_cells_io.check_rows(df) & np.isclose(df["beta"].astype(float), beta) & (df["t"] <= T)]
    return df.assign(F=fbeta_pr(df["precision"].to_numpy(), df["recall"].to_numpy(), beta))


def read_arm_pair(args: tuple[str, str, float, int]) -> dict | None:
    """One run of the hold's arm against the same run of today's sessions: the checks, and both F curves."""
    arm_path, ctl_path, beta, hold = args
    try:
        arm = pd.read_csv(arm_path, usecols=lambda c: c in ARM_COLS, low_memory=False)
        ctl = pd.read_csv(ctl_path, usecols=lambda c: c in BASE_COLS, low_memory=False)
        apk = pd.read_csv(arm_path.replace(".csv", "__picks.csv"))
        cpk = pd.read_csv(ctl_path.replace(".csv", "__picks.csv"))
    except (pd.errors.EmptyDataError, OSError, ValueError):
        return None
    if arm.empty or ctl.empty:
        return None
    key = (str(arm["category"].iloc[0]), int(arm["seed"].iloc[0]))
    if key != (str(ctl["category"].iloc[0]), int(ctl["seed"].iloc[0])):
        return {"error": f"{arm_path}: {key} is not the control's run"}
    tag = arm["gmm_variant"].fillna("").astype(str) == HOLD_TAG
    tagged = _rows_by_t(arm[tag], beta).drop_duplicates("t", keep="last")
    abase = _rows_by_t(_cells_io._base_rows(arm), beta).drop_duplicates("t", keep="last").sort_values("t")
    cbase = _rows_by_t(_cells_io._base_rows(ctl), beta).drop_duplicates("t", keep="last").sort_values("t")
    # Through the hold: past it the arm is today's code, and a run cut short of 150 votes picks differently near
    # its end (the prompted check starts only with its rounds' votes left in the budget).
    horizon = min(int(abase["t"].max()), int(hold))
    abase = abase[abase["t"] <= horizon]
    tagged = tagged[tagged["t"] <= horizon]
    # The picks, click for click, through the horizon (a check round past it is not a click here).
    ap = apk[(apk["t"] <= horizon) & (apk["phase"] != "check")]
    cp = cpk[(cpk["t"] <= horizon) & (cpk["phase"] != "check")]
    picks_same = len(ap) == len(cp) and bool((ap["picked_id"].to_numpy() == cp["picked_id"].to_numpy()).all())
    ctrained = cbase[cbase["detector_tier"].astype(str) == TRAINED].set_index("t")["F"]
    head_same = all(abs(f - ctrained.get(t, np.nan)) < 1e-9 for t, f in zip(tagged["t"], tagged["F"]))
    aq = abase[abase["detector_tier"].astype(str) == QUERY]
    qf = aq["F"].unique()
    atrained = abase[abase["detector_tier"].astype(str) == TRAINED].set_index("t")["F"]
    after_same = all(abs(f - ctrained.get(t, np.nan)) < 1e-9 for t, f in atrained.items())
    ft = float(ctrained.index.min()) if len(ctrained) else np.inf
    held_past_quota = aq[aq["t"] >= ft]
    q = float(qf[0]) if len(qf) else np.nan
    # Today: the typed query's set until the quota (#4758), the head from it.  The arm: its own rows up to its
    # last vote, today's head after it.
    head = carry(ctrained.index.to_numpy(), ctrained.to_numpy())
    today = np.where(GRID >= ft, head, q)
    arm_curve = carry(abase["t"].to_numpy(), abase["F"].to_numpy())
    arm_curve = np.where(GRID <= horizon, arm_curve, today)
    return {
        "category": key[0],
        "seed": key[1],
        "horizon": horizon,
        "first_trained": ft,
        "picks_same": picks_same,
        "head_same": head_same,
        "after_same": after_same,
        "query_sets": len(qf),
        "held_app_trained": int(held_past_quota["app_trained"].fillna(0).astype(int).sum()),
        "n_held_past_quota": len(held_past_quota),
        "last_held": float(held_past_quota["t"].max()) if len(held_past_quota) else np.nan,
        "query_F": q,
        "today": today,
        "arm": arm_curve,
    }


def arm_check(a: argparse.Namespace) -> None:
    """The hold's arm against today's sessions, run for run: the checks that it is today's but for what Test gives."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for spec in a.arm:
        preset, rest = spec.split("=", 1)
        arm_dir, ctl_dir, beta_s, offline = rest.split("::")
        beta = float(beta_s)
        cells = sorted(p.name for p in Path(arm_dir).glob("task_[0-9]*.csv") if "__" not in p.name)
        jobs = [(str(Path(arm_dir) / c), str(Path(ctl_dir) / c), beta, a.hold) for c in cells]
        with ProcessPoolExecutor(a.workers) as ex:
            got = [g for g in ex.map(read_arm_pair, jobs, chunksize=4) if g is not None]
        errors = [g["error"] for g in got if "error" in g]
        runs = [g for g in got if "error" not in g]
        cls = np.array([r["category"].split("@")[0] for r in runs])
        today = np.stack([r["today"] for r in runs])
        armF = np.stack([r["arm"] for r in runs])
        r = {
            "preset": preset,
            "cells": len(cells),
            "runs": len(runs),
            "mismatched_runs": len(errors),
            "picks_same": sum(x["picks_same"] for x in runs),
            "head_rows_same": sum(x["head_same"] for x in runs),
            "trained_rows_same": sum(x["after_same"] for x in runs),
            "one_query_set": sum(x["query_sets"] == 1 for x in runs),
            "held_rows_app_trained": sum(x["held_app_trained"] for x in runs),
            "runs_held_past_quota": sum(x["n_held_past_quota"] > 0 for x in runs),
            "last_held_vote": float(np.nanmax([x["last_held"] for x in runs])),
        }
        for s, span in SPANS.items():
            r[f"today_{s}"] = float(span_mean(today, span).mean())
            r[f"d_{s}"], r[f"se_{s}"] = _mean_se(span_mean(armF, span) - span_mean(today, span), cls)
        seeds = np.array([x["seed"] for x in runs])
        d150 = span_mean(armF, (1, 150)) - span_mean(today, (1, 150))
        r["d_1-150_by_seed"] = " ".join(f"{d150[seeds == s].mean():+.4f}" for s in sorted(set(seeds)))
        if offline:
            # The offline price on the text baseline, on the same runs: what the arm replaces.
            off = Arm(offline)
            idx = {(c, int(s)): i for i, (c, s) in enumerate(zip(off.category, off.seed))}
            sel = np.array([idx[(x["category"], x["seed"])] for x in runs])
            F = off.handover(np.maximum(off.ft, float(a.hold)))[sel]
            base = off.handover(off.ft)[sel]
            for s, span in SPANS.items():
                r[f"offline_{s}"] = float((span_mean(F, span) - span_mean(base, span)).mean())
            r["query_F_minus_baseline"] = float(np.nanmean(np.array([x["query_F"] for x in runs]) - off.tq[sel]))
        summary.append(r)
        if errors:
            print("\n".join(errors[:5]))
    tab = pd.DataFrame(summary)
    tab.to_csv(out / "arm.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        print(tab.T.to_string())


HOLDS = tuple(range(0, 81, 5))


def hold_table(arms: dict[str, Arm]) -> pd.DataFrame:
    """Per preset and hold V: each span's paired gain of showing the typed query until V labels."""
    rows = []
    for preset, arm in arms.items():
        today = arm.handover(arm.ft)
        for v in HOLDS:
            F = arm.handover(np.maximum(arm.ft, float(v)))
            r = {"preset": preset, "hold": v, "n_runs": arm.n}
            for s, span in SPANS.items():
                r[f"d_{s}"], r[f"se_{s}"] = _mean_se(span_mean(F, span) - span_mean(today, span), arm.cls)
            for sub, mask in (("quota", ~arm.dry), ("dry", arm.dry)):
                r[f"d_1-150_{sub}"] = float((span_mean(F, (1, 150)) - span_mean(today, (1, 150)))[mask].mean())
            rows.append(r)
    return pd.DataFrame(rows)


def figures(a: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, surface, grid = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
    slots = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": ink2,
            "axes.labelcolor": ink2,
            "xtick.color": ink2,
            "ytick.color": ink2,
            "text.color": ink,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": surface,
            "axes.facecolor": surface,
        }
    )
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = {}
    titles = {}
    for spec in a.arm:
        preset, rest = spec.split("=", 1)
        path, title = rest.split("::", 1)
        arms[preset] = Arm(path)
        titles[preset] = title

    def spread(ys: list[float], gap: float) -> list[float]:
        order = sorted(range(len(ys)), key=lambda i: ys[i])
        o = list(ys)
        for i, j in zip(order, order[1:]):
            if o[j] - o[i] < gap:
                o[j] = o[i] + gap
        shift = (sum(o) - sum(ys)) / max(len(ys), 1)
        return [y - shift for y in o]

    # Figure 1: mean F-beta at every vote, per preset.
    ncol = 2 if len(arms) > 2 else len(arms)
    nrow = -(-len(arms) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.7 * nrow), squeeze=False)
    for ax, (preset, arm) in zip(axes.flat, arms.items()):
        hold = float(a.hold)
        # Today is drawn last: from the hold on it is the same set, and its line lies on the other's.
        series = [
            (arm.handover(np.maximum(arm.ft, hold)), f"query to {a.hold} votes", slots[1], "-"),
            (arm.handover(np.full(arm.n, np.inf)), "query throughout", slots[2], "-"),
            (
                np.where(arm.has_det, np.maximum(arm.det, arm.tq[:, None]), arm.tq[:, None]),
                "better of the two",
                slots[3],
                "--",
            ),
            (arm.handover(arm.ft), "today", slots[0], "-"),
        ]
        ends = []
        for F, label, color, ls in series:
            m = F[:, 1:].mean(axis=0)
            ax.plot(GRID[1:], m, color=color, lw=1.6 if label == "today" else 2, ls=ls)
            ends.append((float(m[-1]), label, color))
        ax.set_xlim(0, T)
        ax.axvline(hold, color=grid, lw=1, zorder=0)
        ax.grid(axis="y", color=grid, lw=0.8)
        lo, hi = ax.get_ylim()
        for (_y, label, color), yy in zip(ends, spread([e[0] for e in ends], 0.07 * (hi - lo))):
            ax.text(T + 3, yy, label, color=color, va="center", fontsize=8, clip_on=False)
        ax.set_title(titles[preset], fontsize=10)
        ax.set_xlabel("vote")
        ax.set_ylabel("mean F-beta of the set on screen")
    for ax in list(axes.flat)[len(arms) :]:
        ax.set_visible(False)
    fig.tight_layout(w_pad=8.0, h_pad=1.5)
    fig.savefig(out / "fig_by_vote.png", dpi=130)
    plt.close(fig)

    # Figure 2: the gain over votes 1-150 against the hold, per preset.
    tab = hold_table(arms)
    tab.to_csv(out / "holds.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    ends = []
    for (preset, arm), color in zip(arms.items(), slots):
        g = tab[tab["preset"] == preset]
        ax.plot(g["hold"], g["d_1-150"], color=color, lw=2)
        ax.fill_between(
            g["hold"], g["d_1-150"] - 2 * g["se_1-150"], g["d_1-150"] + 2 * g["se_1-150"], color=color, alpha=0.12, lw=0
        )
        ends.append((float(g["d_1-150"].iloc[-1]), titles[preset], color))
    ax.axhline(0, color=ink2, lw=0.8)
    ax.axvline(float(a.hold), color=grid, lw=1, zorder=0)
    ax.grid(axis="y", color=grid, lw=0.8)
    ax.set_xlim(0, HOLDS[-1])
    lo, hi = ax.get_ylim()
    for (_y, label, color), yy in zip(ends, spread([e[0] for e in ends], 0.07 * (hi - lo))):
        ax.text(HOLDS[-1] + 1.5, yy, label, color=color, va="center", fontsize=8, clip_on=False)
    ax.set_xlabel("typed query's set on screen until this many votes (0 = today)")
    ax.set_ylabel("gain in mean F-beta, votes 1-150")
    fig.tight_layout(rect=(0, 0, 0.8, 1))
    fig.savefig(out / "fig_hold.png", dpi=130)
    plt.close(fig)
    print(f"wrote {out}/fig_by_vote.png, fig_hold.png, holds.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--cells", required=True)
    e.add_argument("--baseline", required=True)
    e.add_argument("--beta", type=float, required=True)
    e.add_argument("--text-embedder", default="siglip")
    e.add_argument("--workers", type=int, default=16)
    e.add_argument("--out", required=True)
    p = sub.add_parser("price")
    p.add_argument("--arm", action="append", required=True, help="PRESET=extract.npz, repeatable")
    p.add_argument("--out", required=True)
    p.add_argument("--top", type=int, default=12)
    p.add_argument("--curve-rules", default="today,never,ceiling_switch,ceiling_once")
    f = sub.add_parser("figures")
    f.add_argument("--arm", action="append", required=True, help="PRESET=extract.npz::panel title, repeatable")
    f.add_argument("--hold", type=int, default=30)
    f.add_argument("--out", required=True)
    c = sub.add_parser("arm")
    c.add_argument(
        "--arm", action="append", required=True, help="PRESET=ARM_CELLS::CONTROL_CELLS::BETA::OFFLINE.npz, repeatable"
    )
    c.add_argument("--hold", type=int, default=30)
    c.add_argument("--workers", type=int, default=16)
    c.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a)
    elif a.cmd == "figures":
        figures(a)
    elif a.cmd == "arm":
        arm_check(a)
    else:
        price(a)


if __name__ == "__main__":
    main()
