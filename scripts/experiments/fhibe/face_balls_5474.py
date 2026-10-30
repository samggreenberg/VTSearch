#!/usr/bin/env python
"""Face Balls (#5474): is finding one person a centre and a radius, and is the radius the same for everyone?

A face review looks for one person. If the face embedder puts each person's photos in a tight cap on
the sphere and everyone else outside it, a detector needs only the cap's centre (the mean of a few
Good photos, as the example sort already takes it) and a radius (a cosine floor). If that floor is
also the same for every person, there is nothing left to fit per session: average the Goods, cut at
the global floor, done.

On unit vectors a ball around the centroid ``c`` (``cos(x, c) >= r``) is the half-space ``c . x >= r``,
a hyperplane whose normal is the centroid. So the ball is the special case of the session's linear
head where the normal is not fitted, and the question is what fitting it, and fitting the offset per
person, buys over a fixed rule.

``dump`` (in a FHIBE run's environment, as ``centroid_count_4759.py dump`` runs) rebuilds every cell's
split with the harness's own pieces (``evaluable_pool``, ``stratified_split`` on ``RandomState(seed)``,
``choose_examples``) and writes, for K = 1, 2, 3, 4 Goods and for all of the voting half's Goods
(``K = 0`` in the files, "all" in the tables), each withheld-half media's cosine to the Goods' centroid
and its label. The centroid is ``example_sort``'s (the mean of the L2-normalised vectors, normalised).
Goods are taken in the opening's draw order, so K = 1 and K = 4 are the review's own example sets.

``price`` reads those files and scores, per dataset x K x balance (beta 1/4, 1, 4), on the withheld
half as the review scores:

* ``oracle`` - each person's best cut on the labels (their own radius, chosen with the answers);
* ``global`` - one cosine floor for everyone at that size, K and beta, fitted on the other half of the
  people (two folds by person) so it is never scored on the people it was fitted on;
* ``global_anyk`` - one floor per size and beta whatever K is;
* ``global_one`` - one floor for every size, K and beta.

It also writes each person's interval of perfect floors (above every other face, at or below every
own photo), how many people have one, and how many of those contain the global floor. Writes
``cells.csv`` (names people: owner-only), ``curves.csv`` (mean F-beta against the floor), ``summary.csv``
and ``summary.md``.

``compare`` joins the ball to the app's own sessions on the same people: the review's click-0 row (the
example sort at its shipped line) and the session at 25, 50 and 150 votes, from a review's analysis
dir (``curves.csv`` per run dir), paired by person.

Usage::

    python face_balls_5474.py dump --results <run>/results --out <dir>
    python face_balls_5474.py price --dumps <dir> --out <dir>
    python face_balls_5474.py compare --price <dir> --analysis <runs>/<set>-analysis --strata <person_strata.csv> --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BETAS = (0.25, 1.0, 4.0)
KS = (1, 2, 3, 4, 0)  # 0 = every Good of the voting half
GRID = np.round(np.arange(0.0, 0.9001, 0.0025), 4)


def ktag(k: int) -> str:
    return "all" if k == 0 else str(k)


# ---------------------------------------------------------------- dump


def dump(results: Path, out: Path) -> None:
    sys.path.insert(0, str(HERE.parent / "calibration"))
    import common  # noqa: PLC0415

    common.setup_env()
    import experiment_config as cfg  # noqa: PLC0415
    from _cells_io import load_medias  # noqa: PLC0415

    from vtscore.config import EMBEDDINGS_DIR  # noqa: PLC0415
    from vtscore.eval.example_opening import choose_examples, stratified_split  # noqa: PLC0415
    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415
    from vtscore.eval.labels import evaluable_pool, media_is_positive  # noqa: PLC0415

    prepare = json.loads((results / "prepare_info.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    for ds in cfg.DATASETS:
        for emb, info in prepare.get("datasets", {}).get(ds, {}).items():
            cats = info.get("selected_categories") or []
            medias = load_medias(EMBEDDINGS_DIR / cfg.pickle_name(ds, emb), repair=True)
            common.log(f"{ds} x {emb}: {len(medias)} medias, {len(cats)} people")
            ids = list(medias)
            row = {cid: i for i, cid in enumerate(ids)}
            u = np.stack([np.asarray(media_embedding(medias[c]), dtype=np.float64) for c in ids])
            u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
            per_k: dict[int, dict[str, list]] = {
                k: {"s": [], "y": [], "off": [0], "cat": [], "seed": [], "nvp": [], "loo": []} for k in KS
            }
            for cat in cats:
                pool = evaluable_pool(medias, cat)
                for seed in cfg.SEEDS:
                    rng = np.random.RandomState(seed)
                    sim_ids, test_ids = stratified_split(pool, cat, cfg.SIM_FRACTION, rng)
                    n_vote_pos = sum(1 for c in sim_ids if media_is_positive(pool[c], cat))
                    if n_vote_pos == 0:
                        continue
                    order = choose_examples(sim_ids, pool, cat, n_vote_pos, seed)
                    t = np.asarray([row[c] for c in test_ids])
                    y = np.asarray([1 if media_is_positive(pool[c], cat) else 0 for c in test_ids], dtype=np.uint8)
                    if not y.any():
                        continue
                    for k in KS:
                        if k > n_vote_pos:
                            continue
                        ex = order if k == 0 else order[:k]
                        q = u[[row[c] for c in ex]].mean(axis=0)
                        q /= max(np.linalg.norm(q), 1e-12)
                        d = per_k[k]
                        # how close each Good sits to the centroid of the others: the spread the session can see
                        g = u[[row[c] for c in ex]]
                        if len(ex) > 1:
                            rest = (g.sum(axis=0)[None, :] - g) / (len(ex) - 1)
                            rest /= np.maximum(np.linalg.norm(rest, axis=1, keepdims=True), 1e-12)
                            d["loo"].append(float(np.mean(np.sum(g * rest, axis=1))))
                        else:
                            d["loo"].append(np.nan)
                        d["s"].append((u[t] @ q).astype(np.float32))
                        d["y"].append(y)
                        d["off"].append(d["off"][-1] + len(t))
                        d["cat"].append(cat)
                        d["seed"].append(seed)
                        d["nvp"].append(n_vote_pos)
            for k, d in per_k.items():
                path = out / f"balls_{ds}_k{ktag(k)}.npz"
                np.savez_compressed(
                    path,
                    scores=np.concatenate(d["s"]),
                    labels=np.concatenate(d["y"]),
                    offsets=np.asarray(d["off"], dtype=np.int64),
                    category=np.asarray(d["cat"]),
                    seed=np.asarray(d["seed"], dtype=np.int32),
                    n_vote_pos=np.asarray(d["nvp"], dtype=np.int32),
                    goods_loo=np.asarray(d["loo"], dtype=np.float64),
                    dataset=ds,
                    embedder=emb,
                    k=k,
                )
                common.log(f"K={ktag(k)}: {len(d['cat'])} cells -> {path}")


# ---------------------------------------------------------------- price


def fbeta_arr(tp, kept, npos, beta):
    p = np.where(kept > 0, tp / np.maximum(kept, 1), 0.0)
    r = tp / npos
    b2 = beta * beta
    f = np.where((p > 0) | (r > 0), (1 + b2) * p * r / np.maximum(b2 * p + r, 1e-12), 0.0)
    return p, r, f


def fold_of(cat: str) -> int:
    return int(hashlib.sha256(cat.encode()).hexdigest(), 16) % 2


def cell_table(path: Path) -> list[dict]:
    """One dict per cell: sorted cosines, labels, F-beta on GRID per beta, the oracle cut, the perfect interval."""
    z = np.load(path)
    off = z["offsets"]
    out = []
    for i in range(len(off) - 1):
        s = z["scores"][off[i] : off[i + 1]].astype(np.float64)
        y = z["labels"][off[i] : off[i + 1]].astype(np.int64)
        order = np.argsort(-s, kind="stable")
        ss, ys = s[order], y[order]
        npos = int(y.sum())
        # kept and tp at each grid floor (cos >= r)
        kept_g = np.searchsorted(-ss, -GRID, side="right")
        ctp = np.concatenate([[0], np.cumsum(ys)])
        tp_g = ctp[kept_g]
        pos, neg = s[y == 1], s[y == 0]
        rec = {
            "dataset": str(z["dataset"]),
            "k": int(z["k"]),
            "category": str(z["category"][i]),
            "seed": int(z["seed"][i]),
            "n_vote_pos": int(z["n_vote_pos"][i]),
            "goods_loo": float(z["goods_loo"][i]),
            "n_test": int(s.size),
            "n_pos": npos,
            "pos_min": float(pos.min()),
            "pos_median": float(np.median(pos)),
            "neg_max": float(neg.max()),
            "separable": bool(pos.min() > neg.max()),
            "kept_g": kept_g,
            "tp_g": tp_g,
        }
        k_all = np.arange(1, s.size + 1)
        for beta in BETAS:
            _, _, f_g = fbeta_arr(tp_g, kept_g, npos, beta)
            rec[f"f_grid_{beta:g}"] = f_g
            p, r, f = fbeta_arr(ctp[1:], k_all, npos, beta)
            b = int(np.argmax(f))
            rec[f"oracle_f_{beta:g}"] = float(f[b])
            rec[f"oracle_p_{beta:g}"] = float(p[b])
            rec[f"oracle_r_{beta:g}"] = float(r[b])
            rec[f"oracle_kept_{beta:g}"] = b + 1
            # the oracle's floor: midway between the last kept and the first left out
            nxt = ss[b + 1] if b + 1 < ss.size else ss[b] - 0.01
            rec[f"oracle_floor_{beta:g}"] = float((ss[b] + nxt) / 2)
        out.append(rec)
    return out


def fit_floor(cells: list[dict], beta: float) -> float:
    """The grid floor with the best mean F-beta over *cells*."""
    m = np.mean([c[f"f_grid_{beta:g}"] for c in cells], axis=0)
    return float(GRID[int(np.argmax(m))])


SLOPES = (0.0, 0.25, 0.5, 0.75, 1.0)
STEP = float(GRID[1] - GRID[0])


def fit_adaptive(cells: list[dict], beta: float) -> tuple[float, float, float]:
    """``(floor, slope, centre)``: floor_i = floor + slope * (goods_loo_i - centre), best mean F-beta on *cells*.

    ``goods_loo`` is how close each Good sits to the centroid of the others, so a person whose own photos
    are spread out gets a lower floor. Slope 0 is the global floor.
    """
    centre = float(np.median([c["goods_loo"] for c in cells]))
    f = np.stack([c[f"f_grid_{beta:g}"] for c in cells])
    best = (-1.0, 0.0, 0.0)
    for b in SLOPES:
        shift = np.rint(np.asarray([b * (c["goods_loo"] - centre) for c in cells]) / STEP).astype(int)
        idx = np.clip(np.arange(GRID.size)[None, :] + shift[:, None], 0, GRID.size - 1)
        m = np.take_along_axis(f, idx, axis=1).mean(axis=0)
        j = int(np.argmax(m))
        if m[j] > best[0]:
            best = (float(m[j]), float(GRID[j]), b)
    return best[1], best[2], centre


def at_floor(c: dict, r: float, beta: float) -> tuple[float, float, float, int]:
    j = int(np.clip(np.rint(r / STEP), 0, GRID.size - 1))
    kept, tp = int(c["kept_g"][j]), int(c["tp_g"][j])
    p, rr, f = fbeta_arr(np.float64(tp), np.float64(kept), c["n_pos"], beta)
    return float(p), float(rr), float(f), kept


def price(dumps: Path, out: Path) -> None:
    import pandas as pd  # noqa: PLC0415

    cells: list[dict] = []
    for path in sorted(dumps.glob("balls_*_k*.npz")):
        t = cell_table(path)
        print(f"{path.name}: {len(t)} cells", file=sys.stderr)
        cells += t
    for c in cells:
        c["fold"] = fold_of(c["category"])

    def group(pred):
        return [c for c in cells if pred(c)]

    rows, curves = [], []
    datasets = sorted({c["dataset"] for c in cells})
    for beta in BETAS:
        # one floor for everything, fitted per fold on the other fold
        one = {f: fit_floor(group(lambda c, f=f: c["fold"] != f), beta) for f in (0, 1)}
        for ds in datasets:
            anyk = {f: fit_floor(group(lambda c, f=f: c["dataset"] == ds and c["fold"] != f), beta) for f in (0, 1)}
            for k in KS:
                g = group(lambda c: c["dataset"] == ds and c["k"] == k)
                if not g:
                    continue
                own = {f: fit_floor([c for c in g if c["fold"] != f], beta) for f in (0, 1)}
                adapt = {f: fit_adaptive([c for c in g if c["fold"] != f], beta) for f in (0, 1)} if k != 1 else {}
                m = np.mean([c[f"f_grid_{beta:g}"] for c in g], axis=0)
                for r, v in zip(GRID, m):
                    curves.append({"dataset": ds, "k": ktag(k), "beta": beta, "floor": float(r), "fbeta": float(v)})
                for c in g:
                    base = {kk: c[kk] for kk in ("dataset", "category", "seed", "n_vote_pos", "n_test", "n_pos",
                                                 "pos_min", "pos_median", "neg_max", "separable", "fold", "goods_loo")}  # fmt: skip
                    base.update(k=ktag(k), beta=beta)
                    rows.append({**base, "rule": "oracle", "floor": c[f"oracle_floor_{beta:g}"],
                                 "precision": c[f"oracle_p_{beta:g}"], "recall": c[f"oracle_r_{beta:g}"],
                                 "fbeta": c[f"oracle_f_{beta:g}"], "kept": c[f"oracle_kept_{beta:g}"]})  # fmt: skip
                    for rule, floors in (("global", own), ("global_anyk", anyk), ("global_one", one)):
                        r = floors[c["fold"]]
                        p, rr, f, kept = at_floor(c, r, beta)
                        rows.append({**base, "rule": rule, "floor": r, "precision": p, "recall": rr, "fbeta": f,
                                     "kept": kept})  # fmt: skip
                    if adapt:
                        g0, b, centre = adapt[c["fold"]]
                        r = g0 + b * (c["goods_loo"] - centre)
                        p, rr, f, kept = at_floor(c, r, beta)
                        rows.append({**base, "rule": "adaptive", "floor": r, "slope": b, "precision": p,
                                     "recall": rr, "fbeta": f, "kept": kept})  # fmt: skip
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "cells.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "curves.csv", index=False)
    summ = summarize_price(df)
    summ.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(price_md(df, summ))
    print((out / "summary.md").read_text())


def summarize_price(df):
    import pandas as pd  # noqa: PLC0415

    rows = []
    key = ["dataset", "category", "seed", "k", "beta"]
    ora = df[df["rule"] == "oracle"].set_index(key)
    for (ds, k, beta, rule), h in df.groupby(["dataset", "k", "beta", "rule"], sort=False):
        hh = h.set_index(key)
        gap = (hh["fbeta"] - ora["fbeta"].reindex(hh.index)).groupby(level="category").mean()
        o = ora.reindex(hh.index)
        inside = ((hh["floor"] > o["neg_max"]) & (hh["floor"] <= o["pos_min"]))[o["separable"].astype(bool)]
        rows.append({
            "dataset": ds, "k": k, "beta": beta, "rule": rule, "people": h["category"].nunique(),
            "floor_median": float(h["floor"].median()),
            "floor_iqr": float(h["floor"].quantile(0.75) - h["floor"].quantile(0.25)),
            "precision": float(h["precision"].mean()), "recall": float(h["recall"].mean()),
            "fbeta": float(h["fbeta"].mean()), "kept_median": float(h["kept"].median()),
            "vs_oracle": float(gap.mean()), "vs_oracle_se": float(gap.std(ddof=1) / np.sqrt(len(gap))),
            "perfect": float((h["fbeta"] >= 0.9999).mean()),
            "separable": float(h["separable"].mean()),
            "floor_in_interval": float(inside.mean()) if len(inside) else np.nan,
        })  # fmt: skip
    return pd.DataFrame(rows).round(4)


def price_md(df, summ) -> str:
    lines = ["# #5474 Face Balls: the Goods' centroid and one cosine floor, on the withheld half", ""]
    one = df.drop_duplicates(["dataset", "k", "category", "seed"])
    lines += ["## Is a person a ball at all? (does any floor separate them perfectly)", "",
              "| dataset | K | people | separable | own photos' lowest cosine (median) | closest other face (median) |",
              "|---|---|---|---|---|---|"]  # fmt: skip
    for (ds, k), g in one.groupby(["dataset", "k"], sort=False):
        lines.append(f"| {ds} | {k} | {len(g)} | {g['separable'].mean():.0%} | {g['pos_min'].median():.2f} "
                     f"| {g['neg_max'].median():.2f} |")  # fmt: skip
    lines.append("")
    for beta in BETAS:
        lines += [f"## beta {beta:g}", "",
                  "| dataset | K | rule | floor (median) | floor IQR | precision | recall | F-beta | vs own radius "
                  "| perfect | global floor inside a separable person's interval |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for _, r in summ[summ["beta"] == beta].iterrows():
            lines.append(
                f"| {r['dataset']} | {r['k']} | {r['rule']} | {r['floor_median']:.3f} | {r['floor_iqr']:.3f} "
                f"| {r['precision']:.2f} | {r['recall']:.2f} | {r['fbeta']:.3f} "
                f"| {r['vs_oracle']:+.3f} ± {r['vs_oracle_se']:.3f} | {r['perfect']:.0%} | {r['floor_in_interval']:.0%} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- compare

DS_SIZE = {"1024": "fhibe_faces_ik_1024", "640": "fhibe_faces_ik_640"}
BTAG = {0.25: "025", 1.0: "1", 4.0: "4"}


def compare(price_dir: Path, analysis: Path, strata: Path | None, out: Path, votes: list[int]) -> None:
    import pandas as pd  # noqa: PLC0415

    cells = pd.read_csv(price_dir / "cells.csv", dtype={"k": str})
    sess = []
    for K in (1, 4):
        for beta, bt in BTAG.items():
            for size, ds in DS_SIZE.items():
                f = analysis / f"k{K}-b{bt}-{size}" / "curves.csv"
                if not f.exists():
                    print(f"missing {f}", file=sys.stderr)
                    continue
                c = pd.read_csv(f, usecols=["category", "seed", "t", f"fbeta_b{bt}", "goods"])
                c = c.rename(columns={f"fbeta_b{bt}": "fbeta"})
                last = c.groupby(["category", "seed"])["t"].transform("max")
                for t in [K, *votes]:
                    # a session that stopped before t keeps its last row
                    h = c[(c["t"] == t) | ((c["t"] == last) & (last < t))].drop_duplicates(
                        ["category", "seed"], keep="first"
                    )
                    sess.append(h.assign(K=K, beta=beta, dataset=ds, at=("opening" if t == K else f"vote {t}")))
    sess = pd.concat(sess, ignore_index=True)
    key = ["dataset", "category", "seed", "beta"]
    rows = []
    for K in (1, 4):
        for (at,), s in sess[sess["K"] == K].groupby(["at"], sort=False):
            s = s.set_index(key)
            for kb, rule in (
                (str(K), "global"),
                (str(K), "adaptive"),
                (str(K), "oracle"),
                ("all", "global"),
                ("all", "adaptive"),
                ("all", "oracle"),
            ):
                b = cells[(cells["k"] == kb) & (cells["rule"] == rule)].set_index(key)["fbeta"]
                j = s.join(b.rename("ball"), how="inner")
                for (ds, beta), h in j.groupby(level=["dataset", "beta"]):
                    d = (h["ball"] - h["fbeta"]).groupby(level="category").mean()
                    rows.append({"dataset": ds, "beta": beta, "session_K": K, "session_at": at,
                                 "ball": f"K={kb} {rule}", "people": len(d), "session_f": h["fbeta"].mean(),
                                 "ball_f": h["ball"].mean(), "ball_minus_session": d.mean(),
                                 "se": d.std(ddof=1) / np.sqrt(len(d)), "ball_wins": float((d > 0.005).mean()),
                                 "ball_loses": float((d < -0.005).mean()),
                                 "session_goods": h["goods"].mean()})  # fmt: skip
    res = pd.DataFrame(rows).round(4)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "vs_session.csv", index=False)
    md = ["# #5474 ball vs the app's session, same people, paired by person", "",
          "| dataset | beta | K | session at | ball | people | session F | ball F | ball - session | ball better / worse |",
          "|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for _, r in res.iterrows():
        md.append(f"| {r['dataset']} | {r['beta']:g} | {r['session_K']} | {r['session_at']} | {r['ball']} | {r['people']} "
                  f"| {r['session_f']:.3f} | {r['ball_f']:.3f} | {r['ball_minus_session']:+.3f} ± {r['se']:.3f} "
                  f"| {r['ball_wins']:.0%} / {r['ball_loses']:.0%} |")  # fmt: skip
    if strata is not None and strata.exists():
        st = pd.read_csv(strata, usecols=["category", "pronoun"])
        g = cells[cells["rule"].isin(["global", "oracle"])].merge(st, on="category", how="left")
        md += ["", "## By pronoun (ball F-beta; global floor vs own radius)", "",
               "| dataset | K | beta | pronoun | people | global | own radius | own floor (median) |",
               "|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for (ds, k, beta, pr), h in g.groupby(["dataset", "k", "beta", "pronoun"]):
            if pr not in ("He/him/his", "She/her/hers"):
                continue
            gl, orc = h[h["rule"] == "global"], h[h["rule"] == "oracle"]
            md.append(f"| {ds} | {k} | {beta:g} | {pr} | {gl['category'].nunique()} | {gl['fbeta'].mean():.3f} "
                      f"| {orc['fbeta'].mean():.3f} | {orc['floor'].median():.3f} |")  # fmt: skip
    (out / "vs_session.md").write_text("\n".join(md) + "\n")
    print((out / "vs_session.md").read_text())


# ---------------------------------------------------------------- figure and examples


def figure(price_dir: Path, compare_dir: Path, out: Path, dataset: str = "fhibe_faces_ik_1024") -> None:
    """Three panels at beta 1: mean F against one floor; each person's own floor; ball vs session."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    cur = pd.read_csv(price_dir / "curves.csv", dtype={"k": str})
    cells = pd.read_csv(price_dir / "cells.csv", dtype={"k": str})
    vs = pd.read_csv(compare_dir / "vs_session.csv")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    colors = {"1": "#9aa7b8", "4": "#4a6fa5", "all": "#1b2f52"}
    for k, col in colors.items():
        h = cur[(cur["dataset"] == dataset) & (cur["k"] == k) & (cur["beta"] == 1.0)]
        o = cells[
            (cells["dataset"] == dataset) & (cells["k"] == k) & (cells["beta"] == 1.0) & (cells["rule"] == "oracle")
        ]
        ax[0].plot(h["floor"], h["fbeta"], color=col, lw=2)
        ax[0].axhline(o["fbeta"].mean(), color=col, ls=":", lw=1)
        j = h["fbeta"].idxmax()
        if k == "4":
            continue  # under the every-Good curve: labelled with it
        label = "4 Goods, or every Good" if k == "all" else "1 Good"
        ax[0].annotate(f"{label}: best {h.loc[j, 'fbeta']:.2f} at {h.loc[j, 'floor']:.2f}", (h.loc[j, "floor"], h.loc[j, "fbeta"]),
                       xytext=(0.55, h.loc[j, "fbeta"] - (0.08 if k == "1" else 0.04)), color=col, fontsize=9, va="center")  # fmt: skip
    ax[0].set(xlim=(0.15, 0.85), ylim=(0.4, 1.0), xlabel="cosine floor (one for everyone)", ylabel="mean F1",
              title="F1 against one floor (dotted: each person's own best)")  # fmt: skip
    o = cells[
        (cells["dataset"] == dataset) & (cells["k"] == "all") & (cells["beta"] == 1.0) & (cells["rule"] == "oracle")
    ]
    g = cells[
        (cells["dataset"] == dataset) & (cells["k"] == "all") & (cells["beta"] == 1.0) & (cells["rule"] == "global")
    ]
    gf = g["floor"].median()
    o = o.assign(mid=(o["pos_min"] + o["neg_max"]) / 2).sort_values("mid").reset_index(drop=True)
    sep = o["pos_min"] > o["neg_max"]
    x = np.arange(len(o))
    ax[1].vlines(x[sep], o["neg_max"][sep], o["pos_min"][sep], color="#4a6fa5", lw=1)
    ax[1].vlines(x[~sep], o["pos_min"][~sep], o["neg_max"][~sep], color="#e08a2c", lw=1)
    ax[1].axhline(gf, color="#c0392b", lw=1.5)
    inside = (sep & (o["neg_max"] < gf) & (o["pos_min"] >= gf)).sum()
    ax[1].text(
        len(o) * 0.35,
        0.13,
        f"one floor {gf:.2f}: inside {inside} of {int(sep.sum())} gaps",
        color="#c0392b",
        fontsize=9,
    )
    ax[1].text(5, 0.88, f"blue: closest other face to lowest own photo ({int(sep.sum())} people)\n"
               f"orange: no gap, an other face above an own photo ({int((~sep).sum())})", fontsize=8, va="top")  # fmt: skip
    ax[1].set(ylim=(0.1, 0.9), xlabel="people, by the middle of their gap", ylabel="cosine to every Good's centroid",
              title="Each person's gap of perfect floors")  # fmt: skip
    h = vs[(vs["dataset"] == dataset) & (vs["session_K"] == 4) & (vs["ball"].isin(["K=4 global", "K=all global"]))]
    ats = ["opening", "vote 25", "vote 50", "vote 150"]
    for i, (beta, col) in enumerate(((0.25, "#9aa7b8"), (1.0, "#4a6fa5"), (4.0, "#1b2f52"))):
        hb = h[(h["beta"] == beta) & (h["ball"] == "K=4 global")].set_index("session_at").reindex(ats)
        x = np.arange(len(ats)) + (i - 1) * 0.25
        ax[2].bar(x, hb["ball_minus_session"], 0.24, yerr=1.96 * hb["se"], color=col, label=f"beta {beta:g}")
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_xticks(np.arange(len(ats)), ["click 0\n(shipped line)", "25 votes", "50 votes", "150 votes"])
    ax[2].set(ylabel="ball F - session F (paired)", title="Ball (4 Goods, one floor) vs session, K = 4")
    ax[2].legend(frameon=False, fontsize=9)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{dataset}, buffalo_s", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)


def examples(dumps: Path, price_dir: Path, out: Path, dataset: str = "fhibe_faces_ik_1024", top: int = 10) -> None:
    """Literal examples: the withheld half's top cosines to every Good's centroid, for three kinds of person."""
    import pandas as pd  # noqa: PLC0415

    cells = pd.read_csv(price_dir / "cells.csv", dtype={"k": str})
    sel = cells[(cells["dataset"] == dataset) & (cells["k"] == "all") & (cells["beta"] == 1.0)]
    gl = sel[sel["rule"] == "global"].set_index("category")
    orc = sel[sel["rule"] == "oracle"].set_index("category")
    floor = float(gl["floor"].median())
    z = np.load(dumps / f"balls_{dataset}_kall.npz")
    off = z["offsets"]
    where = {str(c): i for i, c in enumerate(z["category"])}
    kinds = {
        "a ball at the global floor": orc.index[(orc["fbeta"] >= 0.999) & (gl["fbeta"].reindex(orc.index) >= 0.999)],
        "own photos spread out (a gap, all of it below the global floor)": orc.index[
            (orc["pos_min"] < floor) & orc["separable"]
        ],
        "a look-alike close in (a gap, all of it above the global floor)": orc.index[
            (orc["neg_max"] >= floor) & orc["separable"]
        ],
        "no floor separates them": orc.index[~orc["separable"].astype(bool)],
    }
    rng = np.random.default_rng(5474)
    lines = [f"# #5474 examples: {dataset}, every voting-half Good, beta 1; global floor {floor:.2f}", "",
             f"Top {top} of the withheld half by cosine to the Goods' centroid; * marks the person's own photos.", ""]  # fmt: skip
    for kind, cats in kinds.items():
        lines += [f"## {kind} ({len(cats)} of {len(orc)} people)", "",
                  "| person | Goods | Goods' spread | gap | F at own best / global | top cosines |", "|---|---|---|---|---|---|"]  # fmt: skip
        for cat in rng.choice(np.asarray(cats), size=min(3, len(cats)), replace=False):
            i = where[cat]
            sc = z["scores"][off[i] : off[i + 1]]
            y = z["labels"][off[i] : off[i + 1]]
            idx = np.argsort(-sc, kind="stable")[:top]
            s = " ".join(f"{sc[j]:.2f}{'*' if y[j] else ''}" for j in idx)
            lines.append(f"| {cat[:8]} | {int(z['n_vote_pos'][i])} | {float(z['goods_loo'][i]):.2f} "
                         f"| {orc.loc[cat, 'neg_max']:.2f} to {orc.loc[cat, 'pos_min']:.2f} "
                         f"| {orc.loc[cat, 'fbeta']:.2f} / {gl.loc[cat, 'fbeta']:.2f} | {s} |")  # fmt: skip
        lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--results", type=Path, required=True)
    d.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("price")
    p.add_argument("--dumps", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--price", type=Path, required=True)
    c.add_argument("--analysis", type=Path, required=True)
    c.add_argument("--strata", type=Path)
    c.add_argument("--votes", default="25,50,150")
    c.add_argument("--out", type=Path, required=True)
    f = sub.add_parser("figure")
    f.add_argument("--price", type=Path, required=True)
    f.add_argument("--compare", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("examples")
    e.add_argument("--dumps", type=Path, required=True)
    e.add_argument("--price", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "figure":
        figure(args.price, args.compare, args.out)
    elif args.cmd == "examples":
        examples(args.dumps, args.price, args.out)
    elif args.cmd == "dump":
        dump(args.results, args.out)
    elif args.cmd == "price":
        price(args.dumps, args.out)
    else:
        compare(args.price, args.analysis, args.strata, args.out, [int(v) for v in args.votes.split(",") if v])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
