"""#5490: does SAM3 suffer more than SigLIP on a class no model knows? Grading half.

``zeroshot``: every model's sort at every prompt level of ``pilebuild/loaders/fgvc_aircraft.py``, graded per cell
(a variant with siblings) at four depths:

- ``ap_variant``: the cell's 100 photos among the whole pool (20,000 images), what the opening has to find;
- ``ap_sibling``: the cell's photos among its family's photos only, so whether the text tells a 737-400 from a
  737-300 (chance is 1 / the family's variant count);
- ``ap_family``: the family's photos among the whole pool, so whether the text at least finds the right family;
- ``auroc_aircraft``: FGVC's 10,000 photos against the 10,000 COCO distractors, so whether it finds aircraft at all.

SAM3 and GroundingDINO score from their shards (``sam3_score_5213.py`` / ``gdino_score_5395.py`` run with
``--dataset fgvc_aircraft``: one prompt per level per cell); SigLIP scores as the app's typed sort does
(``embed_text_query`` into the whole-image style's ``exemplar_sims``, as ``text_baseline.py``). Two more columns
show a detector's collapse: ``n_cut`` (how many images clear its own cut, 0.5 for SAM3 and 0.35 for GroundingDINO)
and ``pos_median`` (the median score of the cell's photos).

A family's cells share every level's text but ``variant``, so a paired difference is averaged over cells and its
SE taken over the 16 families (cluster SE): 46 cells are not 46 independent prompts.

``votes``: the seven vote arms of ``unknown_class_5490.sh`` (seed 0, 46 cells a path, beta 1, 150 clicks), F1
shown and AP by click, the Goods each opening finds, every arm paired with ``sig-family`` (today's app when the user
can only type the family), and what losing the name costs each opener: ``X:variant-family`` is the name's worth to
opener X, ``sam3-sig:variant-family`` whether SAM3 loses more of it than SigLIP does. SAM3 arms are read under four
hand-over rules (``hold``, ``head``, ``g20`` = the head from the 20th Good, #5312's route, and ``best``, a bound).

    python unknown_class_5490.py zeroshot --results <grid>/results --scores sam3=DIR gdino=DIR --out DIR
    python unknown_class_5490.py votes --out DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

#: Each detector's own cut (#5213: SAM3 0.5; #5395: GroundingDINO's 0.35).
NATIVE_CUT = {"sam3": 0.5, "gdino": 0.35}


def _shards(root: Path):
    """``(ids, top1 matrix, {prompt: column})`` from a scorer's full shards."""
    shards = sorted(p for p in root.glob("shard_*_of_*.npz") if "limit" not in p.name)
    n = int(shards[0].name.split("_of_")[1].split(".")[0])
    if len(shards) != n:
        raise SystemExit(f"{root}: {len(shards)} of {n} shards present")
    ids, top1, prompts = [], [], None
    for p in shards:
        z = np.load(p, allow_pickle=True)
        prompts = list(z["prompts"]) if prompts is None else prompts
        if list(z["prompts"]) != prompts:
            raise SystemExit(f"{p}: prompt list differs from the first shard's")
        ids.extend(z["ids"].tolist())
        top1.append(z["top1"].astype(np.float32))
    return ids, np.concatenate(top1), {q: j for j, q in enumerate(prompts)}


def _cluster_se(diff: pd.Series, fam: pd.Series) -> float:
    """SE of a cell mean, clustered by family: the SE of the family means, weighted by family size."""
    g = pd.DataFrame({"d": diff.values, "f": fam.values}).groupby("f")["d"]
    k = g.ngroups
    if k < 2:
        return float("nan")
    resid = (g.sum() - g.count() * diff.mean()).to_numpy()
    return float(np.sqrt(k / (k - 1) * (resid**2).sum()) / len(diff))


def zeroshot(a) -> int:
    import common  # noqa: PLC0415

    common.setup_env()
    from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: PLC0415

    import experiment_config as cfg  # noqa: PLC0415
    from _cells_io import load_medias  # noqa: PLC0415
    from vtscore.config import EMBEDDINGS_DIR  # noqa: PLC0415
    from vtscore.embedding import embed_text_query  # noqa: PLC0415
    from vtscore.eval.patch_styles import resolve_style  # noqa: PLC0415

    sys.path.append(str(Path(cfg.__file__).resolve().parent.parent / "pile"))
    from pilebuild.loaders import fgvc_aircraft as fa  # noqa: PLC0415

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prepare = json.loads((Path(a.results) / "prepare_info.json").read_text())
    cells = sorted(next(iter(prepare["datasets"]["fgvc_aircraft"].values()))["selected_categories"])
    if cells != fa.cells():
        raise SystemExit("the grid's cells are not the loader's")

    medias = load_medias(EMBEDDINGS_DIR / "fgvc_aircraft__siglip.pkl")
    ids = sorted(medias)
    lab = [medias[i].get("fgvc") or {} for i in ids]
    variant = np.asarray([x.get("variant", "") for x in lab])
    family = np.asarray([x.get("family", "") for x in lab])
    aircraft = np.asarray([bool(x) for x in lab])

    # Every model's score matrix over `ids`, one column per prompt.
    texts = sorted({t for c in cells for t in fa.prompt_levels(c).values()})
    style = resolve_style("whole_image")
    sig = np.zeros((len(ids), len(texts)), np.float32)
    for j, t in enumerate(texts):
        vec = embed_text_query(t, "image", enrich=cfg.SEED_ENRICH, embedder_name="siglip")
        sims = style.exemplar_sims(medias, np.asarray(vec, dtype=np.float32))
        sig[:, j] = [float(sims[i]) for i in ids]
    models = {"siglip": (sig, {t: j for j, t in enumerate(texts)})}
    del medias
    for spec in a.scores:
        name, root = spec.split("=", 1)
        if not Path(root).is_dir():
            common.log(f"SKIP {name}: no {root}")
            continue
        s_ids, mat, col = _shards(Path(root))
        row = {i: k for k, i in enumerate(s_ids)}
        missing = [i for i in ids if i not in row]
        if missing:
            raise SystemExit(f"{name}: {len(missing)} pool images unscored (e.g. {missing[:3]})")
        models[name] = (mat[[row[i] for i in ids]], col)

    rows = []
    for c in cells:
        fam = fa.family_of(c)
        pos = variant == c
        in_fam = family == fam
        for level, text in fa.prompt_levels(c).items():
            for name, (mat, col) in models.items():
                s = mat[:, col[text]]
                cut = NATIVE_CUT.get(name)
                rows.append(
                    {
                        "model": name,
                        "level": level,
                        "cell": c,
                        "family": fam,
                        "n_siblings": len(fa.FAMILIES[fam].variants),
                        "text": text,
                        "ap_variant": average_precision_score(pos, s),
                        "ap_sibling": average_precision_score(pos[in_fam], s[in_fam]),
                        "ap_family": average_precision_score(in_fam, s),
                        "auroc_aircraft": roc_auc_score(aircraft, s),
                        "n_cut": int((s > cut).sum()) if cut is not None else np.nan,
                        "pos_median": float(np.median(s[pos])),
                        "neg_median": float(np.median(s[~aircraft])),
                    }
                )
    d = pd.DataFrame(rows)
    d.to_csv(out / "zeroshot_per_cell.csv", index=False)

    metrics = ["ap_variant", "ap_sibling", "ap_family", "auroc_aircraft", "n_cut", "pos_median"]
    summ = []
    for (m, lv), g in d.groupby(["model", "level"], sort=False):
        r = {"model": m, "level": lv, "n_cells": len(g)}
        for k in metrics:
            r[k] = g[k].mean()
            r[f"{k}_se"] = _cluster_se(g[k], g["family"])
        summ.append(r)
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "zeroshot_summary.csv", index=False)

    # Chance for each depth, per cell, averaged the same way.
    n_pool = len(ids)
    chance = pd.DataFrame(
        {
            "ap_variant": [100 / n_pool for _ in cells],
            "ap_sibling": [1 / len(fa.FAMILIES[fa.family_of(c)].variants) for c in cells],
            "ap_family": [float((family == fa.family_of(c)).sum()) / n_pool for c in cells],
        }
    ).mean()
    chance.to_csv(out / "zeroshot_chance.csv", header=["chance"])

    paired = []
    for other in [m for m in models if m != "siglip"]:
        for lv in fa.LEVELS:
            x = d[(d.model == other) & (d.level == lv)].set_index("cell")
            y = d[(d.model == "siglip") & (d.level == lv)].set_index("cell").loc[x.index]
            for k in ("ap_variant", "ap_sibling", "ap_family", "auroc_aircraft"):
                diff = x[k] - y[k]
                paired.append(
                    {
                        "model": other,
                        "vs": "siglip",
                        "level": lv,
                        "metric": k,
                        "diff": diff.mean(),
                        "se": _cluster_se(diff, x["family"]),
                        "cells_ahead": float((diff > 0).mean()),
                    }
                )
    pd.DataFrame(paired).to_csv(out / "zeroshot_paired.csv", index=False)

    _zeroshot_figure(summ, chance, out / "zeroshot_levels.png")

    pd.set_option("display.width", 200)
    print(summ[["model", "level"] + metrics].round(3).to_string(index=False))
    print("\nchance:", chance.round(4).to_dict())
    print(pd.DataFrame(paired).round(3).to_string(index=False))
    return 0


def _zeroshot_figure(summ: pd.DataFrame, chance: pd.Series, path: Path) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from early_check_4664 import MUTED, _axes_style  # noqa: PLC0415

    colors = {"siglip": "#8a8985", "sam3": "#2a78d6", "gdino": "#d65f2a"}
    metrics = (("ap_variant", "AP, the variant in the pool"), ("ap_sibling", "AP, the variant among its family"),
               ("ap_family", "AP, the family in the pool"))  # fmt: skip
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (k, title) in zip(axes, metrics, strict=True):
        for m, g in summ.groupby("model", sort=False):
            g = g.set_index("level").reindex(list(LEVELS_ORDER))
            x = np.arange(len(g))
            ax.errorbar(x, g[k], yerr=g[f"{k}_se"], color=colors.get(m, "k"), marker="o", lw=1.6, capsize=3, label=m)
        if k in chance:
            ax.axhline(chance[k], color=MUTED, ls=":", lw=1)
            ax.text(len(LEVELS_ORDER) - 1, chance[k], " chance", color=MUTED, fontsize=8, va="bottom", ha="right")
        ax.set_xticks(np.arange(len(LEVELS_ORDER)), LEVELS_ORDER, rotation=30, fontsize=8)
        ax.set_title(title, loc="left", fontsize=10)
        _axes_style(ax)
    axes[0].set_ylabel("mean over 46 cells", color=MUTED)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


#: Prompt levels, most to least specific.
LEVELS_ORDER = ("variant", "family", "manufacturer", "generic", "fake_number", "fake_name")


#: The vote arms (``unknown_class_5490.sh``) and what each shows. A SAM3 arm's untagged row is SAM3's set at 0.5
#: held through the session and its ``hold:head`` row the head the same session trains, so a hand-over rule is a
#: choice, per click, between two rows of one session (#5241). ``g20`` is #5312's planned route: the head from the
#: 20th Good.
ARMS = ("sig-family", "sam3-family", "sig-variant", "sam3-variant", "sig-fake", "sam3-fake", "ex4")
HAND_OVER = 20
AREAS = ((1, 50), (1, 150))
AT = (10, 25, 50, 100, 150)
VCOLOR = {"sig": "#8a8985", "sam3": "#2a78d6", "ex4": "#3a9a5b"}


def _goods(cells: Path, T: int) -> dict[tuple[str, str], np.ndarray]:
    """Goods labelled by each click, per (cell, path), off the session's own picks (check votes included)."""
    import glob  # noqa: PLC0415

    out = {}
    for pp in sorted(glob.glob(f"{cells}/task_*__picks.csv*")):
        pk = pd.read_csv(pp, usecols=["category", "embedder", "t", "picked_label"])
        if pk.empty:
            continue
        g = np.zeros(T + 1)
        for t in pk.loc[(pk["picked_label"] == 1) & (pk["t"] <= T), "t"].astype(int):
            g[t:] += 1
        out[(str(pk["category"].iloc[0]), str(pk["embedder"].iloc[0]))] = g
    return out


def votes(a) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.append(str(Path(__file__).resolve().parent.parent / "pile"))
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from early_check_4664 import MUTED, _axes_style, area  # noqa: PLC0415
    from pilebuild.loaders import fgvc_aircraft as fa  # noqa: PLC0415
    from sam3_opening_5241 import HOLD_TAG, PATHS, T, curves, load  # noqa: PLC0415

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # shown[(arm, rule)][(cell, emb)] = (F1 curve, AP curve); goods[arm][(cell, emb)] = Goods by click.
    shown: dict[tuple[str, str], dict] = {}
    goods: dict[str, dict] = {}
    for arm in ARMS:
        cells = Path(a.root) / f"{a.date}-uc5490-{arm}-b1" / "results" / "cells"
        if not any(cells.glob("task_*.csv*")):
            print(f"SKIP {arm}: no cells in {cells}")
            continue
        rows = load(cells, a.procs)
        goods[arm] = _goods(cells, T)
        if arm.startswith("sam3-"):
            hold, head = curves(rows, ""), curves(rows, HOLD_TAG)
            shown[(arm, "hold")] = hold
            shown[(arm, "head")] = head
            g20, best = {}, {}
            for k in set(hold) & set(head) & set(goods[arm]):
                has = np.isfinite(head[k][0])
                take = has & (goods[arm][k] >= HAND_OVER)
                g20[k] = tuple(np.where(take, head[k][j], hold[k][j]) for j in (0, 1))
                best[k] = tuple(np.where(has, np.fmax(head[k][j], hold[k][j]), hold[k][j]) for j in (0, 1))
            shown[(arm, "g20")] = g20
            shown[(arm, "best")] = best
        else:
            shown[(arm, "shown")] = curves(rows, "")
    if ("sig-family", "shown") not in shown:
        raise SystemExit("sig-family (the comparison every row is paired with) has no cells")

    summary, paired, per_click, info = [], [], [], {}
    for emb, path in PATHS.items():
        base = shown[("sig-family", "shown")]
        keys = sorted(k for k in base if k[1] == emb)
        keys = [k for k in keys if all(k in v for v in shown.values())]
        cats = [k[0] for k in keys]
        fam = pd.Series([fa.family_of(c) for c in cats])
        info[path] = {"cells": len(keys), "families": int(fam.nunique())}

        def mat(arm_rule, j, keys=keys):
            return np.vstack([shown[arm_rule][k][j] for k in keys])

        base_f = mat(("sig-family", "shown"), 0)
        for arm_rule in shown:
            f, ap_ = mat(arm_rule, 0), mat(arm_rule, 1)
            row = {"path": path, "arm": arm_rule[0], "rule": arm_rule[1], "cells": len(keys)}
            for lo, hi in AREAS:
                x = pd.Series(area(f, lo, hi))
                row[f"F_{lo}_{hi}"], row[f"F_{lo}_{hi}_se"] = x.mean(), _cluster_se(x, fam)
                d = x - pd.Series(area(base_f, lo, hi))
                paired.append({"path": path, "arm": arm_rule[0], "rule": arm_rule[1], "vs": "sig-family",
                               "metric": f"F_{lo}_{hi}", "diff": d.mean(), "se": _cluster_se(d, fam),
                               "cells_ahead": float((d > 0).mean())})  # fmt: skip
            for t in AT:
                row[f"F@{t}"] = float(np.nanmean(f[:, t]))
                row[f"AP@{t}"] = float(np.nanmean(ap_[:, t]))
            g = np.vstack([goods[arm_rule[0]][k] for k in keys]) if all(k in goods[arm_rule[0]] for k in keys) else None
            if g is not None:
                for t in AT:
                    row[f"goods@{t}"] = float(g[:, t].mean())
                row[f"cells_reach_{HAND_OVER}_goods"] = float((g[:, T] >= HAND_OVER).mean())
                first = np.where((g > 0).any(axis=1), (g > 0).argmax(axis=1), T + 1)
                row["first_good_median"] = float(np.median(first))
            summary.append(row)
            for t in range(1, T + 1):
                per_click.append({"path": path, "arm": arm_rule[0], "rule": arm_rule[1], "t": t,
                                  "F": float(np.nanmean(f[:, t])), "AP": float(np.nanmean(ap_[:, t])),
                                  "goods": float(g[:, t].mean()) if g is not None else np.nan})  # fmt: skip

        # What losing the name costs each opener, and whether SAM3 pays more (a difference of paired differences).
        for opener, rule in (("sig", "shown"), ("sam3", "g20"), ("sam3", "hold"), ("sam3", "head")):
            for named, unnamed in (("variant", "family"), ("variant", "fake"), ("family", "fake")):
                ka, kb = (f"{opener}-{named}", rule), (f"{opener}-{unnamed}", rule)
                if ka not in shown or kb not in shown:
                    continue
                for lo, hi in AREAS:
                    d = pd.Series(area(mat(ka, 0), lo, hi) - area(mat(kb, 0), lo, hi))
                    paired.append({"path": path, "arm": f"{opener}:{named}-{unnamed}", "rule": rule, "vs": "",
                                   "metric": f"F_{lo}_{hi}", "diff": d.mean(), "se": _cluster_se(d, fam),
                                   "cells_ahead": float((d > 0).mean())})  # fmt: skip
                    sk = (("sig-" + named, "shown"), ("sig-" + unnamed, "shown"))
                    if opener == "sam3" and all(s in shown for s in sk):
                        ds = pd.Series(area(mat(sk[0], 0), lo, hi) - area(mat(sk[1], 0), lo, hi))
                        dd = d - ds
                        paired.append({"path": path, "arm": f"sam3-sig:{named}-{unnamed}", "rule": rule, "vs": "",
                                       "metric": f"F_{lo}_{hi}", "diff": dd.mean(), "se": _cluster_se(dd, fam),
                                       "cells_ahead": float((dd > 0).mean())})  # fmt: skip

    summ = pd.DataFrame(summary)
    summ.to_csv(out / "votes_summary.csv", index=False)
    pr = pd.DataFrame(paired)
    pr.to_csv(out / "votes_paired.csv", index=False)
    pc = pd.DataFrame(per_click)
    pc.to_csv(out / "votes_per_click.csv", index=False)
    (out / "votes_info.json").write_text(json.dumps(info, indent=1) + "\n")
    _votes_figure(pc, out / "votes_shown.png", "F", "F1 shown (mean over cells)", plt, MUTED, _axes_style)
    _votes_figure(pc, out / "votes_goods.png", "goods", "Goods labelled (mean over cells)", plt, MUTED, _axes_style)

    pd.set_option("display.width", 220)
    print(summ.round(3).to_string(index=False))
    print(pr.round(3).to_string(index=False))
    return 0


def _votes_figure(pc, path, col, ylabel, plt, muted, style) -> None:
    lines = (("sig-family", "shown", "SigLIP, family typed", "-"), ("sig-variant", "shown", "SigLIP, name typed", ":"),
             ("sig-fake", "shown", "SigLIP, invented name", "--"), ("sam3-family", "g20", "SAM3, family typed", "-"),
             ("sam3-variant", "g20", "SAM3, name typed", ":"), ("sam3-fake", "g20", "SAM3, invented name", "--"),
             ("ex4", "shown", "4 example photos", "-"))  # fmt: skip
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (name, g) in zip(axes, pc.groupby("path", sort=False), strict=False):
        for arm, rule, label, ls in lines:
            s = g[(g.arm == arm) & (g.rule == rule)]
            if s.empty:
                continue
            ax.plot(s["t"], s[col], color=VCOLOR[arm.split("-")[0]], ls=ls, lw=1.8, label=label)
        ax.set_title(name, loc="left", fontsize=11)
        ax.set_xlabel("votes", color=muted)
        style(ax)
    axes[0].set_ylabel(ylabel, color=muted)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    z = sub.add_parser("zeroshot")
    z.add_argument("--results", required=True, help="the grid's results dir (prepare_info.json names the cells)")
    z.add_argument("--scores", nargs="*", default=[], help="name=dir of a scorer's shards")
    z.add_argument("--out", required=True)
    v = sub.add_parser("votes")
    v.add_argument("--root", default="/expscratch/sgreenberg/state-of-the-app")
    v.add_argument("--date", default="2026-11-02")
    v.add_argument("--out", required=True)
    v.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    return {"zeroshot": zeroshot, "votes": votes}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
