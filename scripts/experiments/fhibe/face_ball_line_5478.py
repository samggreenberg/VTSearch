"""#5478: the face crops' ball as the session's line, priced on the review's own sessions.

#5474 found that the centroid of a person's four starting photos and one cosine floor per balance scored level
with the 150-vote session, from vectors alone. This prices the ball as the session's line: at every click, the
centroid of the Goods voted so far, cut at a floor, scored on the withheld half beside Find's own line on the
same click. The ball never picks (Autopilot asks by the trained head), so no session is rerun: each cell's
``__picks`` and the review's split (``stratified_split`` at the cell's seed) give the Goods at every click.

Subcommands::

    fit       the floors from #5474's dumps (balls_*.npz): A, one floor per beta (#5474's ``global_one``), and B,
              that floor moved by slope * (spread - centre) (#5474's ``adaptive``, both sizes, K >= 2); each
              fitted on one fold of people (fold0 / fold1, scored on the other) and on all (``all``, what ships)
    replay    per cell and click of a review set: the ball's F at every constant, and the app's own centroid
              rule redrawn at the centroid tier's clicks, which must reproduce the cell's F (the check)
    analyze   ball - Find's line, paired by person: the opening, votes 25/50/150, the mean over votes 1-150,
              by pronoun; and the shipped form (``gated``: the ball from beta 1 up with two Goods, else the line)
    ship      the shipped code's cells against the replay, click by click: same picks, and F = the gated ball
    figures   ship.webp: mean F per vote and the paired difference per person
    examples  literal examples: a person's withheld cosines to their Goods' centroid at vote 150

Usage (in the review's run environment, as ``sota_face.sh`` sets it; people are named: owner-only outputs)::

    python face_ball_line_5478.py fit --dumps <balls-5474>/dumps --out constants.json
    python face_ball_line_5478.py replay --runs <runs> --set 2026-10-30-5473nodiv-ik --constants constants.json --out replay
    python face_ball_line_5478.py analyze --replay replay --strata <person_strata.csv> --out analysis
    python face_ball_line_5478.py ship --runs <runs> --ship 2026-10-30-5478ship-ik --dev 2026-10-30-5473nodiv-ik --replay replay
    python face_ball_line_5478.py figures --analysis analysis --replay replay --out ship.webp
    python face_ball_line_5478.py examples --runs <runs> --set 2026-10-30-5473nodiv-ik --replay replay --out examples.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BTAG = {"025": 0.25, "1": 1.0, "4": 4.0}
SIZE = {"fhibe_faces_ik_1024": "1024", "fhibe_faces_ik_640": "640"}
#: Find's centroid line on face crops before #5478 (#5325), which the dev review's cells were drawn by.
DEV_RULE = "count_gap_floor_skip25_b1_z5.5"
ARMS = {"A": "A_x", "B": "B_x", "A_all": "A_all", "B_all": "B_all"}
_BALL = re.compile(r"ball(?P<floor>-?\d+(?:\.\d+)?)(?:_s(?P<slope>-?\d+(?:\.\d+)?)_c(?P<centre>-?\d+(?:\.\d+)?))?")


def fold_of(cat: str) -> int:
    """#5474's fold of a person: half the people fit a floor, the other half are scored by it."""
    return int(hashlib.sha256(cat.encode()).hexdigest(), 16) % 2


def ball_numbers(rule: str) -> tuple[float, float, float]:
    m = _BALL.fullmatch(rule)
    if m is None:
        raise ValueError(rule)
    return float(m["floor"]), float(m["slope"] or 0.0), float(m["centre"] or 0.0)


def spread_of(G: np.ndarray) -> float | None:
    """Each unit Good's cosine to the unit centroid of the others, averaged (#5474's goods_loo); None under two."""
    if len(G) < 2:
        return None
    rest = (G.sum(axis=0)[None, :] - G) / (len(G) - 1)
    rest /= np.maximum(np.linalg.norm(rest, axis=1, keepdims=True), 1e-12)
    return float(np.mean(np.sum(G * rest, axis=1)))


def fbeta(kept: int, tp: int, npos: int, beta: float) -> tuple[float, float, float]:
    p = tp / kept if kept else 0.0
    r = tp / npos if npos else 0.0
    b2 = beta * beta
    return p, r, ((1 + b2) * p * r / (b2 * p + r) if p + r > 0 else 0.0)


# ---------------------------------------------------------------- fit


def fit(dumps: Path, out: Path) -> None:
    sys.path.insert(0, str(HERE))
    import face_balls_5474 as fb  # noqa: PLC0415

    cells = []
    for p in sorted(dumps.glob("balls_*_k*.npz")):
        cells += fb.cell_table(p)
    for c in cells:
        c["fold"] = fold_of(c["category"])
    res = {}
    for beta in fb.BETAS:
        rec = {}
        for name, pick in (
            ("fold0", lambda c: c["fold"] != 0),
            ("fold1", lambda c: c["fold"] != 1),
            ("all", lambda c: True),
        ):
            a = fb.fit_floor([c for c in cells if pick(c)], beta)
            g = [c for c in cells if pick(c) and c["k"] != 1 and np.isfinite(c["goods_loo"])]
            f0, slope, centre = fb.fit_adaptive(g, beta)
            rec[name] = {"A": f"ball{a:g}", "B": f"ball{f0:g}_s{slope:g}_c{centre:.4f}"}
        res[f"{beta:g}"] = rec
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


# ---------------------------------------------------------------- replay


def _load_pools(run: Path):
    sys.path.insert(0, str(HERE.parent / "calibration"))
    import common  # noqa: PLC0415

    common.setup_env()
    import experiment_config as cfg  # noqa: PLC0415
    from _cells_io import load_medias  # noqa: PLC0415

    from vtscore.config import EMBEDDINGS_DIR  # noqa: PLC0415
    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

    prepare = json.loads((run / "results/prepare_info.json").read_text())
    data = {}
    for ds in cfg.DATASETS:
        for emb in prepare.get("datasets", {}).get(ds, {}):
            medias = load_medias(EMBEDDINGS_DIR / cfg.pickle_name(ds, emb), repair=True)
            ids = list(medias)
            u = np.stack([np.asarray(media_embedding(medias[c]), dtype=np.float64) for c in ids])
            u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
            data[ds] = (medias, {cid: i for i, cid in enumerate(ids)}, u)
            common.log(f"{ds} x {emb}: {len(medias)} medias")
    return cfg, common, data


def _cell_inputs(cfg, data, pk):
    """The cell's withheld half (vectors, labels) and its Goods in vote order, from its picks."""
    from vtscore.eval.example_opening import stratified_split  # noqa: PLC0415
    from vtscore.eval.labels import evaluable_pool, media_is_positive  # noqa: PLC0415

    ds, cat, seed = pk["dataset"].iloc[0], pk["category"].iloc[0], int(pk["seed"].iloc[0])
    medias, row, u = data[ds]
    pool = evaluable_pool(medias, cat)
    _sim, test_ids = stratified_split(pool, cat, cfg.SIM_FRACTION, np.random.RandomState(seed))
    T = u[[row[c] for c in test_ids]]
    y = np.asarray([1 if media_is_positive(pool[c], cat) else 0 for c in test_ids], dtype=np.uint8)
    pk = pk.sort_values("t", kind="stable")
    good_t = pk.loc[pk["picked_label"] == 1, ["t", "picked_id"]].to_numpy()
    return ds, cat, seed, row, u, T, y, good_t


def _picks(run: Path, stem: str):
    import pandas as pd  # noqa: PLC0415

    p = run / "results/cells" / f"{stem}__picks.csv.gz"
    p = p if p.exists() else p.with_suffix("")
    return pd.read_csv(p, usecols=["seed", "dataset", "category", "t", "picked_id", "picked_label"])


def _main_rows(cell: Path):
    """The cell's one row per click: the untagged rows (no line variant, no schedule)."""
    import pandas as pd  # noqa: PLC0415

    m = pd.read_csv(
        cell,
        usecols=["t", "detector_tier", "n_good", "gmm_variant", "schedule", "fbeta", "precision", "recall"],
        dtype=str,
        keep_default_na=False,
    )
    m = m[(m["gmm_variant"] == "") & (m["schedule"] == "")]
    return m.assign(**{c: m[c].astype(float) for c in ("fbeta", "precision", "recall")}, t=m["t"].astype(int))


def replay(runs: Path, set_: str, constants: Path, out: Path, only: str = "", limit: int = 0) -> None:
    import pandas as pd  # noqa: PLC0415

    from vtscore.detectors.centroid_head import _gap_centre, centroid_cut  # noqa: PLC0415

    consts = json.loads(constants.read_text())
    out.mkdir(parents=True, exist_ok=True)
    dirs = sorted(p for p in runs.glob(f"{set_}-k*-b*") if p.is_dir())
    if only:
        dirs = [p for p in dirs if p.name in {f"{set_}-{x}" for x in only.split(",")}]
    cfg, common, data = _load_pools(dirs[0])
    for run in dirs:
        K = int(run.name.split("-k")[-1].split("-b")[0])
        beta = BTAG[run.name.split("-b")[-1]]
        cb = consts[f"{beta:g}"]
        rules = {f"{arm}_{fold}": ball_numbers(cb[fold][arm]) for fold in ("fold0", "fold1", "all") for arm in "AB"}
        cells = sorted((run / "results/cells").glob("task_[0-9][0-9][0-9][0-9].csv*"))[: limit or None]
        rows, n_val, n_ok = [], 0, 0
        for cell in cells:
            stem = cell.name.split(".")[0]
            ds, cat, seed, row, u, T, y, good_t = _cell_inputs(cfg, data, _picks(run, stem))
            npos = int(y.sum())
            cache: dict[int, dict] = {}
            for m in _main_rows(cell).itertuples(index=False):
                g_ids = [int(cid) for tt, cid in good_t if tt <= m.t]
                ng = len(g_ids)
                rec = {"dataset": ds, "K": K, "beta": beta, "category": cat, "seed": seed, "task": stem, "t": m.t,
                       "tier": m.detector_tier, "n_good_row": int(float(m.n_good)), "n_good": ng, "n_pos": npos,
                       "app_f": m.fbeta, "app_p": m.precision, "app_r": m.recall}  # fmt: skip
                if ng and ng not in cache:
                    G = u[[row[c] for c in g_ids]]
                    q = G.mean(axis=0)
                    s = T @ (q / max(np.linalg.norm(q), 1e-12))
                    spread = spread_of(G)
                    c = {"spread": np.nan if spread is None else spread}
                    for name, (floor, slope, centre) in rules.items():
                        cut = floor + (slope * (spread - centre) if spread is not None else 0.0)
                        keep = s >= cut
                        p, r, f = fbeta(int(keep.sum()), int((keep & (y == 1)).sum()), npos, beta)
                        c.update({f"{name}_f": f, f"{name}_p": p, f"{name}_r": r, f"{name}_kept": int(keep.sum()),
                                  f"{name}_cut": cut})  # fmt: skip
                    keep = s >= _gap_centre(s, centroid_cut(s, rule=DEV_RULE, beta=beta))
                    c["redraw_f"] = fbeta(int(keep.sum()), int((keep & (y == 1)).sum()), npos, beta)[2]
                    cache[ng] = c
                if ng:
                    rec.update(cache[ng])
                    if m.detector_tier == "centroid":
                        n_val += 1
                        n_ok += abs(rec["redraw_f"] - rec["app_f"]) < 1e-4
                rows.append(rec)
        df = pd.DataFrame(rows)
        path = out / f"{run.name}.csv.gz"
        df.to_csv(path, index=False)
        path.chmod(0o600)
        common.log(f"{run.name}: {len(cells)} cells, {len(df)} clicks; the centroid tier's line redrawn = the cell's "
                   f"F at {n_ok}/{n_val}; Goods agree {(df['n_good'] == df['n_good_row']).mean():.4f}")  # fmt: skip


# ---------------------------------------------------------------- analyze


def load_replay(replay_dir: Path):
    import pandas as pd  # noqa: PLC0415

    df = pd.concat([pd.read_csv(p) for p in sorted(replay_dir.glob("*.csv.gz"))], ignore_index=True)
    df["fold"] = df["category"].map(fold_of)
    for arm in "AB":
        for m in ("f", "p", "r", "kept", "cut"):
            df[f"{arm}_x_{m}"] = np.where(df["fold"] == 0, df[f"{arm}_fold0_{m}"], df[f"{arm}_fold1_{m}"])
    # The shipped form: the ball from beta 1 up with two Goods (pooled constants), else Find's line.
    holds = (df["beta"] >= 1) & (df["n_good"] >= 2)
    df["gated_f"] = np.where(holds, df["B_all_f"], df["app_f"])
    df["gated_x_f"] = np.where(holds, df["B_x_f"], df["app_f"])
    df["size"] = df["dataset"].map(SIZE)
    return df


def reads(df):
    import pandas as pd  # noqa: PLC0415

    key = ["size", "K", "beta", "category", "seed"]
    df = df[df["t"] <= 150]
    last = df.groupby(key)["t"].transform("max")
    out = []
    for name, T in (("opening", None), ("vote 25", 25), ("vote 50", 50), ("vote 150", 150)):
        h = df[df["t"] == df["K"]] if T is None else df[(df["t"] == T) | ((df["t"] == last) & (last < T))]
        out.append(h.drop_duplicates(key, keep="first").assign(read=name))
    cols = ["app_f", "gated_f", "gated_x_f"] + [f"{c}_f" for c in ARMS.values()]
    out.append(df.groupby(key)[cols].mean().reset_index().assign(read="mean 1-150"))
    return pd.concat(out, ignore_index=True)


def paired(r, by: list[str], arms: dict[str, str]):
    import pandas as pd  # noqa: PLC0415

    rows = []
    for g, h in r.groupby(by, sort=False):
        g = g if isinstance(g, tuple) else (g,)
        for arm, col in arms.items():
            d = (h[f"{col}_f"] - h["app_f"]).to_numpy()
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
            rows.append({**dict(zip(by, g)), "arm": arm, "people": len(d), "app_f": h["app_f"].mean(),
                         "ball_f": h[f"{col}_f"].mean(), "diff": d.mean(), "se": se, "lo": d.mean() - 1.96 * se,
                         "hi": d.mean() + 1.96 * se, "wins": float((d > 0.005).mean()),
                         "loses": float((d < -0.005).mean())})  # fmt: skip
    return pd.DataFrame(rows)


def analyze(replay_dir: Path, strata: Path, out: Path) -> None:
    import pandas as pd  # noqa: PLC0415

    out.mkdir(parents=True, exist_ok=True)
    df = load_replay(replay_dir)
    cen = df[df["tier"] == "centroid"]
    val = (
        cen.assign(ok=(cen["redraw_f"] - cen["app_f"]).abs() < 1e-4)
        .groupby(["size", "K", "beta"])["ok"]
        .agg(["size", "mean"])
        .rename(columns={"size": "centroid_clicks", "mean": "redraw_matches"})
    )
    val["goods_agree"] = df.groupby(["size", "K", "beta"]).apply(lambda h: (h["n_good"] == h["n_good_row"]).mean())
    val.reset_index().to_csv(out / "validation.csv", index=False)
    r = reads(df)
    arms = {**ARMS, "gated": "gated", "gated_x": "gated_x"}
    paired(r, ["size", "K", "beta", "read"], arms).round(4).to_csv(out / "paired.csv", index=False)
    st = pd.read_csv(strata, usecols=["category", "pronoun"]).drop_duplicates("category")
    rp = r.merge(st, on="category", how="left")
    rp = rp[rp["pronoun"].isin(["She/her/hers", "He/him/his"])]
    paired(rp, ["size", "K", "beta", "read", "pronoun"], arms).round(4).to_csv(out / "pronoun.csv", index=False)
    h = df[(df["t"] == 150) & (df["n_good"] >= 2)]
    h.groupby(["size", "K", "beta"])[["B_all_cut", "B_all_kept", "spread", "n_good"]].median().round(3).to_csv(
        out / "at150.csv"
    )
    for f in out.iterdir():
        f.chmod(0o600)
    print(val)


# ---------------------------------------------------------------- ship


def ship(runs: Path, ship_set: str, dev_set: str, replay_dir: Path) -> None:
    """Each shipped cell: the dev cell's picks, and at every click the gated ball's F the replay predicts."""
    import pandas as pd  # noqa: PLC0415

    n_cells = n_rows = n_ok = same = 0
    for run in sorted(runs.glob(f"{ship_set}-k*-b*")):
        dev = runs / run.name.replace(ship_set, dev_set)
        rep = pd.read_csv(replay_dir / f"{dev.name}.csv.gz")
        for cell in sorted((run / "results/cells").glob("task_[0-9][0-9][0-9][0-9].csv*")):
            stem = cell.name.split(".")[0]
            m = _main_rows(cell)[["t", "fbeta"]]
            r = rep[rep["task"] == stem]
            r = r.assign(want=np.where((r["beta"] >= 1) & (r["n_good"] >= 2), r["B_all_f"], r["app_f"]))
            j = m.merge(r[["t", "want"]], on="t", how="outer", indicator=True)
            ok = (j["_merge"] == "both") & ((j["fbeta"] - j["want"]).abs() < 1e-4)
            n_cells, n_rows, n_ok = n_cells + 1, n_rows + len(j), n_ok + int(ok.sum())
            cols = ["t", "picked_id", "picked_label"]
            same += int(
                _picks(run, stem)[cols].reset_index(drop=True).equals(_picks(dev, stem)[cols].reset_index(drop=True))
            )
            if not ok.all():
                print(run.name, stem, j[~ok].head(3).to_dict("records"))
    print(f"{n_cells} shipped cells: picks identical to dev {same}/{n_cells}; clicks at the replay's F {n_ok}/{n_rows}")


# ---------------------------------------------------------------- figures and examples


def figures(analysis: Path, replay_dir: Path, out: Path) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    df = load_replay(replay_dir)
    df = df[df["t"] <= 150]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    colors = {1: "#9aa7b8", 4: "#1b2f52"}
    for j, beta in enumerate((0.25, 1.0, 4.0)):
        a = ax[0, j]
        for K, col in colors.items():
            h = df[(df["size"] == "1024") & (df["K"] == K) & (df["beta"] == beta)]
            m = h.groupby("t")[["app_f", "gated_f"]].mean()
            a.plot(
                m.index,
                m["app_f"],
                color=col,
                lw=1.2,
                ls=":",
                label=f"{K} photo{'s' if K > 1 else ''}: Find's line (dev)",
            )
            a.plot(m.index, m["gated_f"], color=col, lw=2, label=f"{K} photo{'s' if K > 1 else ''}: the shipped line")
        a.set(xscale="log", ylim=(0.6, 1.0), xlabel="vote", ylabel=f"mean F-beta (beta {beta:g})",
              title=f"beta {beta:g}, 1024 px" + (" (the ball does not hold: same line)" if beta < 1 else ""))  # fmt: skip
        if j == 0:
            a.legend(frameon=False, fontsize=8, loc="lower right")
        b = ax[1, j]
        if beta < 1:
            p = pd.read_csv(analysis / "paired.csv")
            p = p[(p["beta"] == beta) & (p["read"] == "mean 1-150") & (p["arm"] == "B")]
            b.bar(np.arange(len(p)), p["diff"], yerr=1.96 * p["se"], color="#c0392b")
            b.set_xticks(np.arange(len(p)), [f"{s} px\nK {k}" for s, k in zip(p["size"], p["K"], strict=True)])
            b.axhline(0, color="k", lw=0.8)
            b.set(ylabel="ball B - Find's line, votes 1-150", title="beta 1/4: why the ball stays off (cross-fitted)")
            continue
        for K, col in colors.items():
            h = df[(df["size"] == "1024") & (df["K"] == K) & (df["beta"] == beta)]
            d = np.sort((h.groupby("category")["gated_x_f"].mean() - h.groupby("category")["app_f"].mean()).to_numpy())
            b.plot(np.linspace(0, 1, d.size), d, color=col, lw=2, label=f"{K} photo{'s' if K > 1 else ''}")
        b.axhline(0, color="k", lw=0.8)
        b.set(ylim=(-0.25, 0.25), xlabel="people, sorted", ylabel="ball - Find's line, mean over votes 1-150",
              title=f"Per person, beta {beta:g}, 1024 px (cross-fitted floors)")  # fmt: skip
        b.legend(frameon=False, fontsize=8)
    for a in ax.flat:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)


def examples(runs: Path, set_: str, replay_dir: Path, out: Path, top: int = 12) -> None:
    """At vote 150, beta 1, 4 photos, 1024 px: the people the ball gains most and loses most against Find's line."""
    df = load_replay(replay_dir)
    run = runs / f"{set_}-k4-b1"
    h = df[(df["size"] == "1024") & (df["K"] == 4) & (df["beta"] == 1.0)]
    last = h.groupby("task")["t"].transform("max")
    h = h[(h["t"] == 150) | ((h["t"] == last) & (last < 150))].drop_duplicates("task")
    h = h.assign(d=h["gated_f"] - h["app_f"]).sort_values("d")
    pick = {"the ball loses most": h.head(3), "the ball gains most": h.tail(3)}
    cfg, _common, data = _load_pools(run)
    md = ["# #5478 literal examples (owner-only: names people)", "",
          "Vote 150, beta 1, four starting photos, 1024 px. The withheld half's top cosines to the centroid of the Goods",
          "voted by then, own photos starred; `|` marks the ball's floor. Find's line is the trained head's, on its",
          "own scores, so it has no cosine: its precision and recall stand beside it.", ""]  # fmt: skip
    for kind, rows in pick.items():
        md += [f"## {kind}", ""]
        for r in rows.itertuples():
            _ds, cat, _seed, row, u, T, y, good_t = _cell_inputs(cfg, data, _picks(run, r.task))
            G = u[[row[int(c)] for tt, c in good_t if tt <= r.t]]
            q = G.mean(axis=0)
            s = T @ (q / np.linalg.norm(q))
            order = np.argsort(-s)[:top]
            cells = []
            for i in order:
                if s[i] < r.B_all_cut and (not cells or "|" not in cells):
                    cells.append("|")
                cells.append(f"{s[i]:.2f}{'*' if y[i] else ''}")
            md.append(f"- `{cat[:8]}` ({len(G)} Goods, spread {r.spread:.2f}, floor {r.B_all_cut:.2f}, "
                      f"{int(y.sum())} own photos withheld): ball F {r.B_all_f:.2f} (kept {int(r.B_all_kept)}), "
                      f"Find's line F {r.app_f:.2f} (P {r.app_p:.2f}, R {r.app_r:.2f})  ")  # fmt: skip
            md.append("  " + " ".join(cells))
        md.append("")
    out.write_text("\n".join(md) + "\n")
    out.chmod(0o600)
    print(out.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--dumps", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("replay")
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--set", required=True)
    p.add_argument("--constants", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--only", default="")
    p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("analyze")
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--strata", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("ship")
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--ship", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--replay", type=Path, required=True)
    p = sub.add_parser("figures")
    p.add_argument("--analysis", type=Path, required=True)
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("examples")
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--set", required=True)
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "fit":
        fit(a.dumps, a.out)
    elif a.cmd == "replay":
        replay(a.runs, a.set, a.constants, a.out, a.only, a.limit)
    elif a.cmd == "analyze":
        analyze(a.replay, a.strata, a.out)
    elif a.cmd == "ship":
        ship(a.runs, a.ship, a.dev, a.replay)
    elif a.cmd == "figures":
        figures(a.analysis, a.replay, a.out)
    else:
        examples(a.runs, a.set, a.replay, a.out)


if __name__ == "__main__":
    main()
