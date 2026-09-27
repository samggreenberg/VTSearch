"""Vote curves on FullMarks: does stamp-finding *learn* from Good/Bad clicks? (#4162)

Replays the template x page matrices ``template_matrix.py`` wrote, so every arm
sees identical SIFT matching and differs only in its learning rule.  Arms
(pre-registered on #4162):

``a0_exemplar``     inliers against the query crop; never learns
``a1_max``          max inliers over the crop and every Good's box template (the control)
``a1p_goods_only``  max over the Good templates only, the crop dropped once a Good exists
``a2_repick``       the single template, crop or Good, with the highest median inliers
                    against the *other* Goods (the crop until two Goods exist)
``a3_vlad_svm``     the production linear SVM head on page VLAD (crop + Goods vs Bads);
                    VLAD cosine to the crop before a Bad exists
``a3s_production``  a3's top 50 re-ranked as the app does: ``structural_rerank`` over the
                    Good templates, match-statistic MLP from 3 votes, cold gate before
``a4_siglip_svm``   the SVM head on SigLIP; cosine to the crop before a Bad exists
``a4r_siglip_sift`` a4's top 1,000 re-ranked by a1's inliers, the tail in SVM order
``a5_mlp``          the match-statistic MLP (``train_verification_classifier``'s recipe)
                    over a1's best-template statistics, from 3 votes; a1 before

One vote per step: the top unlabelled page of the ranking, labelled from ground
truth.  Two readouts:

* **closed loop** -- each arm picks its own votes; positives found after *v* votes.
* **shared sequence** -- every arm trains on the same *v* votes (the top *v* of
  ``a0_exemplar``) and is scored by AP and P@10 on the same unlabelled remainder.

    python vote_curve.py --matrix <dir>/matrix-s --tier s --out <dir>/curves-s
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fullmarks_config as cfg  # noqa: E402

CHECKPOINTS = (0, 3, 5, 10, 20, 40)
#: Production Stage-2 shortlist (``DEFAULT_RERANK_TOP_K``) and the SigLIP arm's.
PRODUCTION_TOP_K = 50
SIGLIP_TOP_K = 1000
#: ``train_verification_classifier`` trains from this many votes.
MIN_MLP_VOTES = 3
#: The cold gate: ``min(1, inliers / (2 * DEFAULT_MIN_INLIERS))``.
MIN_INLIERS = 8

#: Arms that read the page VLAD / SigLIP vectors rather than the template matrix alone.
VECTOR_ARMS = {"a3_vlad_svm", "a3s_production", "a4_siglip_svm", "a4r_siglip_sift"}

ARMS = (
    "a0_exemplar",
    "a1_max",
    "a1p_goods_only",
    "a2_repick",
    "a3_vlad_svm",
    "a3s_production",
    "a4_siglip_svm",
    "a4r_siglip_sift",
    "a5_mlp",
)


# --------------------------------------------------------------------------
# Ranking primitives
# --------------------------------------------------------------------------


def order_by(*keys: np.ndarray) -> np.ndarray:
    """Pool indices by descending *keys* (first key primary); the pool is sorted by page id,
    so index order breaks the remaining ties exactly as ``rank_splg`` does."""
    n = len(keys[0])
    return np.lexsort((np.arange(n),) + tuple(-np.asarray(k, dtype=np.float64) for k in reversed(keys)))


def shortlist(stage1: np.ndarray, k: int, *keys: np.ndarray) -> np.ndarray:
    """*stage1*'s top *k* re-ordered by *keys* (Stage-1 order breaking ties), then the tail."""
    head = stage1[:k]
    pos = np.arange(len(head))
    sub = np.lexsort((pos,) + tuple(-np.asarray(key, dtype=np.float64)[head] for key in reversed(keys)))
    return np.concatenate([head[sub], stage1[k:]])


def average_precision(order: np.ndarray, positive: np.ndarray) -> float:
    hits = positive[order]
    n = int(hits.sum())
    if n == 0:
        return float("nan")
    cum = np.cumsum(hits)
    return float((cum[hits] / (np.flatnonzero(hits) + 1)).sum() / n)


# --------------------------------------------------------------------------
# One class
# --------------------------------------------------------------------------


class ClassData:
    """A class's matrix plus the vectors the SVM arms need, indexed by pool position."""

    def __init__(self, npz: Path, vlad: np.ndarray, siglip: np.ndarray, qvlad: np.ndarray, qsiglip: np.ndarray):
        z = np.load(npz)
        self.pool_ids = [str(p) for p in z["pool_ids"]]
        self.template_ids = [str(t) for t in z["template_ids"]]
        self.stats = z["stats"].astype(np.float32)  # [T, N, 9]
        self.inliers = z["inliers"].astype(np.int32)  # [T, N]
        self.tentative = z["tentative"].astype(np.int64)
        self.ratio = self.stats[..., 1]
        self.positive = z["positives"].astype(bool)
        col = {p: i for i, p in enumerate(self.pool_ids)}
        #: Pool column of each template's own page; the crop's page is not in the pool.
        self.tcol = np.array([col.get(t, -1) for t in self.template_ids])
        self.template_of = {int(c): t for t, c in enumerate(self.tcol) if c >= 0}
        self.vlad, self.siglip, self.qvlad, self.qsiglip = vlad, siglip, qvlad, qsiglip

    @property
    def n(self) -> int:
        return len(self.pool_ids)

    def templates_for(self, goods: Sequence[int]) -> list[int]:
        return [self.template_of[g] for g in goods if g in self.template_of]

    def best(self, templates: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per page: the strongest template's ``(inliers, tentative, template index)``.

        "Strongest" is the production key ``(model_ok, inlier_count, inlier_ratio)``
        (``best_match_stats``); inliers are 0 unless the model is sane, so the
        count carries ``model_ok``.
        """
        t = np.asarray(templates)
        inl = self.inliers[t]
        key = inl.astype(np.float64) + np.clip(self.ratio[t], 0, 1) * 0.5
        arg = key.argmax(axis=0)
        cols = np.arange(self.n)
        return inl[arg, cols], self.tentative[t][arg, cols], t[arg]


def _svm_scores(X: np.ndarray, q: np.ndarray, goods: list[int], bads: list[int]) -> np.ndarray:
    """The production linear SVM head's decision on every page, crop + Goods vs Bads.

    Before a Bad exists there is nothing to separate from, so the ranking is the
    cosine to the query crop -- what example sort shows first.
    """
    if not bads:
        return X @ q
    from vtscore.training.svm import fit_linear_svm_head  # noqa: PLC0415

    import torch  # noqa: PLC0415

    Xt = np.vstack([q[None, :], X[goods], X[bads]]).astype(np.float32)
    y = np.array([1.0] * (1 + len(goods)) + [0.0] * len(bads), dtype=np.float32)
    model = fit_linear_svm_head(Xt, y, Xt.shape[1])
    with torch.no_grad():
        dev = next(model.parameters()).device
        return model(torch.from_numpy(X.astype(np.float32)).to(dev)).squeeze(1).cpu().numpy()


def _mlp(cd: ClassData, templates: list[int], goods: list[int], bads: list[int]) -> Optional[Any]:
    """``train_verification_classifier`` on the matrix: each Good's best fit over the
    *other* templates (leave-one-out), each Bad's over all of them."""
    if len(goods) + len(bads) < MIN_MLP_VOTES or not templates:
        return None
    feats, labels = [], []
    for g in goods:
        others = [t for t in templates if cd.tcol[t] != g]
        if not others:
            continue
        key = cd.inliers[others, g] + np.clip(cd.ratio[others, g], 0, 1) * 0.5
        feats.append(cd.stats[others[int(key.argmax())], g])
        labels.append(1.0)
    for b in bads:
        key = cd.inliers[templates, b] + np.clip(cd.ratio[templates, b], 0, 1) * 0.5
        feats.append(cd.stats[templates[int(key.argmax())], b])
        labels.append(0.0)
    if len(set(labels)) < 2:
        return None
    import torch  # noqa: PLC0415

    from vtscore.training.mlp import train_model  # noqa: PLC0415

    X = torch.from_numpy(np.stack(feats).astype(np.float32))
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    try:
        return train_model(X, y, X.shape[1])
    except ValueError:
        return None


def _mlp_scores(model: Any, stats: np.ndarray) -> np.ndarray:
    import torch  # noqa: PLC0415

    from vtscore.utils.scores import sigmoid_to_finite_scores  # noqa: PLC0415

    with torch.no_grad():
        dev = next(model.parameters()).device
        prob = np.asarray(sigmoid_to_finite_scores(model(torch.from_numpy(stats).to(dev))), dtype=np.float64)
    return np.clip(prob.reshape(-1), 0.0, None)


def rank(arm: str, cd: ClassData, goods: list[int], bads: list[int]) -> np.ndarray:
    """Pool indices, best first, for *arm* given the votes so far (labelled pages included)."""
    good_t = cd.templates_for(goods)
    if arm == "a0_exemplar":
        return order_by(cd.inliers[0], cd.tentative[0])
    if arm in ("a1_max", "a5_mlp", "a4r_siglip_sift"):
        inl, tent, arg = cd.best([0, *good_t])
        if arm == "a1_max":
            return order_by(inl, tent)
        if arm == "a4r_siglip_sift":
            stage1 = order_by(_svm_scores(cd.siglip, cd.qsiglip, goods, bads))
            return shortlist(stage1, SIGLIP_TOP_K, inl, tent)
        model = _mlp(cd, [0, *good_t], goods, bads)
        if model is None:
            return order_by(inl, tent)
        best_stats = cd.stats[arg, np.arange(cd.n)]
        return order_by(_mlp_scores(model, best_stats), inl, tent)
    if arm == "a1p_goods_only":
        inl, tent, _ = cd.best(good_t or [0])
        return order_by(inl, tent)
    if arm == "a2_repick":
        pick = 0
        if len(good_t) >= 2:
            best_med = -1.0
            for t in [0, *good_t]:
                others = [g for g in goods if g != cd.tcol[t]]
                med = float(np.median(cd.inliers[t, others]))
                if med > best_med:
                    best_med, pick = med, t
        return order_by(cd.inliers[pick], cd.tentative[pick])
    if arm == "a4_siglip_svm":
        return order_by(_svm_scores(cd.siglip, cd.qsiglip, goods, bads))
    if arm in ("a3_vlad_svm", "a3s_production"):
        stage1 = order_by(_svm_scores(cd.vlad, cd.qvlad, goods, bads))
        if arm == "a3_vlad_svm" or not good_t:
            # No Good template yet: ``maybe_structural_rerank`` returns Stage 1 unchanged.
            return stage1
        inl, tent, arg = cd.best(good_t)
        model = _mlp(cd, good_t, goods, bads)
        if model is None:
            gate = np.minimum(1.0, inl / (2.0 * MIN_INLIERS))
        else:
            gate = _mlp_scores(model, cd.stats[arg, np.arange(cd.n)])
        return shortlist(stage1, PRODUCTION_TOP_K, np.round(gate, 4))
    raise KeyError(arm)


def remainder(order: np.ndarray, labelled: set[int]) -> np.ndarray:
    return np.array([i for i in order if i not in labelled], dtype=np.int64)


def run_class(
    cd: ClassData, arms: Sequence[str], checkpoints: Sequence[int], readouts: Sequence[str] = ("shared", "closed")
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    vmax = max(checkpoints)
    n_pos = int(cd.positive.sum())

    def residual(order: np.ndarray, labelled: set[int]) -> tuple[float, float, int]:
        rest = remainder(order, labelled)
        left = int(cd.positive[rest].sum())
        p10 = float(cd.positive[rest[:10]].mean()) if len(rest) else float("nan")
        return average_precision(rest, cd.positive), p10, left

    # Shared sequence: the top v of a0's (static) ranking.
    a0 = rank("a0_exemplar", cd, [], [])
    for arm in arms if "shared" in readouts else ():
        for v in checkpoints:
            seq = [int(i) for i in a0[:v]]
            goods = [i for i in seq if cd.positive[i]]
            bads = [i for i in seq if not cd.positive[i]]
            ap, p10, left = residual(rank(arm, cd, goods, bads), set(seq))
            rows.append(
                {"readout": "shared", "arm": arm, "v": v, "found": len(goods), "ap": ap, "p10": p10, "left": left}
            )

    # Closed loop: each arm chooses its next vote from its own current ranking.
    for arm in arms if "closed" in readouts else ():
        goods: list[int] = []
        bads: list[int] = []
        labelled: set[int] = set()
        for v in range(vmax + 1):
            order = rank(arm, cd, goods, bads)
            if v in checkpoints:
                ap, p10, left = residual(order, labelled)
                rows.append(
                    {"readout": "closed", "arm": arm, "v": v, "found": len(goods), "ap": ap, "p10": p10, "left": left}
                )
            if v == vmax or len(labelled) == cd.n:
                continue
            nxt = next(int(i) for i in order if int(i) not in labelled)
            labelled.add(nxt)
            (goods if cd.positive[nxt] else bads).append(nxt)
    for r in rows:
        r["n_positive"] = n_pos
    return rows


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def load_vectors(
    matrix: Path, tier: str
) -> tuple[dict[str, int], np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Page VLADs (identical in every shard), the SigLIP cell, and each class's query vectors."""
    import embed_corpus  # noqa: PLC0415
    import eval_retrieval as ev  # noqa: PLC0415

    qvlad: dict[str, np.ndarray] = {}
    qsiglip: dict[str, np.ndarray] = {}
    page_vlad: Optional[np.ndarray] = None
    ids: list[str] = []
    for f in sorted(matrix.glob("vectors-*.npz")):
        z = np.load(f)
        if page_vlad is None:
            ids, page_vlad = [str(p) for p in z["page_ids"]], z["page_vlad"]
        for k in z.files:
            if k.startswith("qvlad__"):
                qvlad[k[len("qvlad__") :]] = z[k]
            elif k.startswith("qsiglip__"):
                qsiglip[k[len("qsiglip__") :]] = z[k]
    assert page_vlad is not None, f"no vectors-*.npz under {matrix}"
    row = {p: i for i, p in enumerate(ids)}
    sig_ids, sig = ev.read_vectors(embed_corpus.cell_path(tier, "siglip"), "siglip")
    sig_by = {p: i for i, p in enumerate(sig_ids)}
    siglip = np.zeros((len(ids), sig.shape[1]), dtype=np.float32)
    for p, i in row.items():
        if p in sig_by:
            siglip[i] = sig[sig_by[p]]
    return row, page_vlad, siglip, qvlad, qsiglip


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

CONTROL = "a1_max"
#: The pre-registered verdict point: shared-sequence AP at 10 votes, paired over classes.
VERDICT_V = 10
BOOTSTRAP = 10000


def classes_of_version(version: str) -> set[str]:
    """The roster a frozen version scored: its manifest names one query crop per class."""
    manifest = json.loads(
        (Path(__file__).resolve().parent / "versions" / f"{version}.json").read_text(encoding="utf-8")
    )
    return {k.split(":", 1)[1] for k in manifest["files"] if k.startswith("query_crop:")}


def paired_ci(diffs: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    """Mean paired difference and its bootstrap 95% interval over classes."""
    rng = np.random.default_rng(seed)
    boots = rng.choice(diffs, size=(BOOTSTRAP, len(diffs)), replace=True).mean(axis=1)
    return float(diffs.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def summarise(rows: list[dict[str, Any]], subset: Optional[set[str]] = None) -> str:
    """Markdown: per-arm means at each vote count for both readouts, then the paired test."""
    if subset is not None:
        rows = [r for r in rows if r["class_id"] in subset]
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    vs = sorted({int(r["v"]) for r in rows})
    val = {(r["readout"], r["arm"], int(r["v"]), r["class_id"]): r for r in rows}
    classes = sorted({r["class_id"] for r in rows})

    def mean(readout: str, arm: str, v: int, key: str) -> tuple[float, int]:
        # A class-step with nothing left to find has no remainder to score (AP is nan, P@10 a
        # meaningless 0), and vote_stoplist.py writes no row for it at all.
        xs = [
            float(val[(readout, arm, v, c)][key])
            for c in classes
            if (readout, arm, v, c) in val and (readout == "closed" or int(val[(readout, arm, v, c)]["left"]) > 0)
        ]
        xs = [x for x in xs if not np.isnan(x)]
        return (float(np.mean(xs)) if xs else float("nan")), len(xs)

    out = [f"{len(classes)} classes.", ""]
    for readout, key, label in (
        ("closed", "found", "Closed loop: positives found after v votes (mean over classes)"),
        ("shared", "ap", "Shared sequence: AP on the unlabelled remainder"),
        ("shared", "p10", "Shared sequence: P@10 on the unlabelled remainder"),
    ):
        if not any(r["readout"] == readout for r in rows):
            continue
        out += [f"### {label}", "", "| arm | " + " | ".join(f"v={v}" for v in vs) + " |", "|---|" + "---:|" * len(vs)]
        for arm in arms:
            if not any(r["arm"] == arm and r["readout"] == readout for r in rows):
                continue
            cells = []
            for v in vs:
                m, n = mean(readout, arm, v, key)
                cells.append(f"{m:.2f}" + (f" ({n})" if readout == "shared" and n < len(classes) else ""))
            out.append(f"| {arm} | " + " | ".join(cells) + " |")
        out.append("")
    out += [
        f"### Paired against `{CONTROL}` (mean difference, bootstrap 95% interval over classes)",
        "",
        "| arm | readout | v | classes | mean diff | 95% interval |",
        "|---|---|---:|---:|---:|---|",
    ]
    for arm in arms:
        if arm == CONTROL:
            continue
        for readout, key, v in (
            ("shared", "ap", VERDICT_V),
            ("shared", "ap", 20),
            ("closed", "found", 10),
            ("closed", "found", 40),
        ):
            pairs = [
                (float(val[(readout, arm, v, c)][key]), float(val[(readout, CONTROL, v, c)][key]))
                for c in classes
                if (readout, arm, v, c) in val and (readout, CONTROL, v, c) in val
            ]
            pairs = [(a, b) for a, b in pairs if not (np.isnan(a) or np.isnan(b))]
            if not pairs:
                continue
            d = np.array([a - b for a, b in pairs])
            m, lo, hi = paired_ci(d)
            out.append(f"| {arm} | {readout} {key} | {v} | {len(d)} | {m:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    return "\n".join(out) + "\n"


def write_summary(out: Path) -> None:
    rows: list[dict[str, Any]] = []
    # rows.csv from this script, rows-a6-*.csv from vote_stoplist.py.
    for f in [out / "rows.csv", *sorted(out.glob("rows-a6-*.csv"))]:
        with f.open(encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    text = "## All classes\n\n" + summarise(rows)
    old = classes_of_version("v4.3")
    text += "\n## The 27 classes v4.3 had (the nine v5.0 added are easy for SIFT)\n\n" + summarise(rows, old)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print(text)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--tier", default="s")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--classes", default="", help="comma-separated subset (default: every matrix)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--readouts", default="shared,closed")
    ap.add_argument("--max-v", type=int, default=max(CHECKPOINTS), help="last checkpoint (tier m's matrix reaches 20)")
    ap.add_argument("--summarise", action="store_true", help="only rewrite summary.md from <out>/rows.csv")
    args = ap.parse_args(argv)
    if args.out.resolve().is_relative_to(cfg.OUT.resolve()):
        ap.error("--out is inside the corpus; an eval never writes there")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.summarise:
        write_summary(args.out)
        return 0
    arms = [a for a in args.arms.split(",") if a]
    unknown = set(arms) - set(ARMS)
    if unknown:
        ap.error(f"unknown arms: {sorted(unknown)}")

    readouts = [r for r in args.readouts.split(",") if r]
    checkpoints = tuple(v for v in CHECKPOINTS if v <= args.max_v)
    if any(a in VECTOR_ARMS for a in arms):
        row, page_vlad, siglip, qvlad, qsiglip = load_vectors(args.matrix, args.tier)
    else:
        row, page_vlad, siglip, qvlad, qsiglip = None, None, None, {}, {}
    wanted = {c.replace("/", "__") for c in args.classes.split(",") if c}
    files = [f for f in sorted(args.matrix.glob("*.npz")) if not f.name.startswith("vectors-")]
    files = [f for f in files if not wanted or f.stem in wanted]
    all_rows: list[dict[str, Any]] = []
    for f in files:
        t0 = time.time()
        slug = f.stem
        z = np.load(f)
        if row is None:
            none = np.zeros((len(z["pool_ids"]), 1), dtype=np.float32)
            cd = ClassData(f, none, none, np.zeros(1, np.float32), np.zeros(1, np.float32))
        else:
            idx = [row[str(p)] for p in z["pool_ids"]]
            cd = ClassData(f, page_vlad[idx], siglip[idx], qvlad[slug], qsiglip[slug])
        if not cd.positive.any():
            print(f"  {slug}: no positive in the pool, skipped", flush=True)
            continue
        rows = run_class(cd, arms, checkpoints, readouts)
        cid = slug.replace("__", "/", 1)
        for r in rows:
            r["class_id"] = cid
        all_rows.extend(rows)
        found = {r["arm"]: r["found"] for r in rows if r["readout"] == "closed" and r["v"] == max(checkpoints)}
        print(
            f"  {cid}: {cd.n} pages, {cd.positive.sum()} positives, {time.time() - t0:.0f}s; found@{max(checkpoints)} {found}",
            flush=True,
        )
        # Rewrite after every class so a partial run is still readable.
        with (args.out / "rows.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(
                fh, fieldnames=["class_id", "readout", "arm", "v", "found", "ap", "p10", "left", "n_positive"]
            )
            w.writeheader()
            w.writerows(all_rows)
    (args.out / "run.json").write_text(
        json.dumps(
            {
                "matrix": str(args.matrix),
                "tier": args.tier,
                "arms": arms,
                "checkpoints": checkpoints,
                "readouts": readouts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_summary(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
