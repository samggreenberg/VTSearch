"""Score Toponymy signpost runs against ground-truth categories.

For every ``topo_*.json`` in a dataset's results dir, computes per layer:

- **cluster quality** (independent of names): ARI / NMI / purity of the
  cluster assignment vs. ground-truth categories, coverage (1 - noise).
- **name quality**: for each topic, its members' majority category; the
  CLAP text-space cosine between the topic name and that category name
  ("name-label agreement"), plus a random-name baseline for calibration.
  A topic "hits" when its name is closer to the majority label than 95% of
  random name-label pairs.
- **distinctiveness**: duplicate-name fraction; mean pairwise name
  similarity among siblings (lower = more distinguishing).

Writes ``eval_<run>.json`` per run and prints a compact comparison table.

Usage::

    python evaluate.py esc50 [--embedder clap]
"""

from __future__ import annotations

import argparse
from collections import Counter

import common

common.setup_env()

import numpy as np  # noqa: E402

# ESC-50's standard 5 major groups (by target ranges 0-9 / 10-19 / ... / 40-49).
ESC50_GROUPS = {
    "animals": ["dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects", "sheep", "crow"],
    "natural": [
        "rain",
        "sea_waves",
        "crackling_fire",
        "crickets",
        "chirping_birds",
        "water_drops",
        "wind",
        "pouring_water",
        "toilet_flush",
        "thunderstorm",
    ],
    "human": [
        "crying_baby",
        "sneezing",
        "clapping",
        "breathing",
        "coughing",
        "footsteps",
        "laughing",
        "brushing_teeth",
        "snoring",
        "drinking_sipping",
    ],
    "interior": [
        "door_wood_knock",
        "mouse_click",
        "keyboard_typing",
        "door_wood_creaks",
        "can_opening",
        "washing_machine",
        "vacuum_cleaner",
        "clock_alarm",
        "clock_tick",
        "glass_breaking",
    ],
    "exterior": [
        "helicopter",
        "chainsaw",
        "siren",
        "car_horn",
        "engine",
        "train",
        "church_bells",
        "airplane",
        "fireworks",
        "hand_saw",
    ],
}
CAT_TO_GROUP = {c: g for g, cats in ESC50_GROUPS.items() for c in cats}


def purity(labels: np.ndarray, gt: np.ndarray) -> float:
    mask = labels >= 0
    if not mask.any():
        return 0.0
    total = 0
    for c in np.unique(labels[mask]):
        members = gt[labels == c]
        total += Counter(members.tolist()).most_common(1)[0][1]
    return total / mask.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--embedder", default="clap")
    args = ap.parse_args()

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    out = common.ds_dir(args.dataset)
    meta = common.load_json(out / "meta.json")
    gt_all = np.array([m["category"] for m in meta])
    # Coarse ground truth (ESC-50 only): the 5 standard major groups. Coarse
    # map layers should agree with THIS, not with the 50 fine categories.
    gt_coarse_all = np.array([CAT_TO_GROUP.get(c, "other") for c in gt_all]) if args.dataset == "esc50" else None

    from run_toponymy import ClapTextEncoder

    enc = ClapTextEncoder(args.embedder)

    # Independent text encoder: CLAP-space similarity structurally favors
    # names built from CLAP vocabulary terms, so name↔label agreement is also
    # measured with a neutral sentence encoder.
    from sentence_transformers import SentenceTransformer

    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed_norm(strings):
        v = enc.encode([s.replace("_", " ") for s in strings])
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    def embed_norm_st(strings):
        v = st_model.encode([s.replace("_", " ") for s in strings])
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    cat_names = sorted(set(gt_all.tolist()))
    cat_vecs = embed_norm(cat_names)
    cat_vecs_st = embed_norm_st(cat_names)
    cat_idx = {c: i for i, c in enumerate(cat_names)}

    rows = []
    for run_path in sorted(out.glob("topo_*.json")):
        run = common.load_json(run_path)
        if "error" in run:
            print(f"{run_path.stem}: FAILED — {run['error']}")
            continue
        n = run["n_clips"]
        gt = gt_all[:n]
        per_layer = []
        for layer in run["layers"]:
            labels = np.array(layer["cluster_labels"])
            names = layer["topic_names"]
            mask = labels >= 0
            ari = adjusted_rand_score(gt[mask], labels[mask]) if mask.any() else 0.0
            nmi = normalized_mutual_info_score(gt[mask], labels[mask]) if mask.any() else 0.0
            pur = purity(labels, gt)
            coarse = {}
            if gt_coarse_all is not None and mask.any():
                gtc = gt_coarse_all[:n]
                coarse = {
                    "ari_coarse": round(float(adjusted_rand_score(gtc[mask], labels[mask])), 4),
                    "nmi_coarse": round(float(normalized_mutual_info_score(gtc[mask], labels[mask])), 4),
                    "purity_coarse": round(float(purity(labels, gtc)), 4),
                }

            name_strs = [str(x) or "unnamed" for x in names]
            name_vecs = embed_norm(name_strs)
            name_vecs_st = embed_norm_st(name_strs)
            agreements, agreements_st, majorities = [], [], []
            hit, hit_st = 0, 0
            rng = np.random.default_rng(0)
            null_sims = []
            for c, name in enumerate(names):
                members = gt[labels == c]
                if len(members) == 0:
                    majorities.append(None)
                    agreements.append(None)
                    agreements_st.append(None)
                    continue
                maj = Counter(members.tolist()).most_common(1)[0][0]
                majorities.append(maj)
                sim = float(name_vecs[c] @ cat_vecs[cat_idx[maj]])
                agreements.append(round(sim, 4))
                null = name_vecs[c] @ cat_vecs[rng.permutation(len(cat_names))[:20]].T
                null_sims.extend(null.tolist())
                if sim > np.quantile(null, 0.95):
                    hit += 1
                sim_st = float(name_vecs_st[c] @ cat_vecs_st[cat_idx[maj]])
                agreements_st.append(round(sim_st, 4))
                null_st = name_vecs_st[c] @ cat_vecs_st[rng.permutation(len(cat_names))[:20]].T
                if sim_st > np.quantile(null_st, 0.95):
                    hit_st += 1
            valid = [a for a in agreements if a is not None]
            valid_st = [a for a in agreements_st if a is not None]

            dup_frac = 1 - len({str(x).lower() for x in names}) / max(len(names), 1)
            sib_sims = []
            for parent, children in run["cluster_tree"].items():
                idxs = [int(ch.split(":")[1]) for ch in children if int(ch.split(":")[0]) == layer["layer"]]
                if len(idxs) > 1:
                    sub = name_vecs[idxs]
                    sims = sub @ sub.T
                    iu = np.triu_indices(len(idxs), 1)
                    sib_sims.extend(sims[iu].tolist())

            per_layer.append(
                {
                    "layer": layer["layer"],
                    "n_topics": layer["n_topics"],
                    "coverage": round(1 - layer["noise_frac"], 4),
                    "ari": round(float(ari), 4),
                    "nmi": round(float(nmi), 4),
                    "purity": round(float(pur), 4),
                    "name_label_sim_mean": round(float(np.mean(valid)), 4) if valid else None,
                    "name_label_null_mean": round(float(np.mean(null_sims)), 4) if null_sims else None,
                    "name_hit_rate": round(hit / max(len(valid), 1), 4) if valid else None,
                    "name_label_sim_st_mean": round(float(np.mean(valid_st)), 4) if valid_st else None,
                    "name_hit_rate_st": round(hit_st / max(len(valid_st), 1), 4) if valid_st else None,
                    "dup_name_frac": round(dup_frac, 4),
                    "sibling_name_sim_mean": round(float(np.mean(sib_sims)), 4) if sib_sims else None,
                    **coarse,
                    "topics": [
                        {"name": str(nm), "majority": mj, "size": int((labels == c).sum()), "sim": ag}
                        for c, (nm, mj, ag) in enumerate(zip(names, majorities, agreements))
                    ],
                }
            )
        ev = {
            "run": run_path.stem,
            "n_clips": n,
            "llm_calls": run.get("llm_calls"),
            "timings_s": run.get("timings_s"),
            "per_layer": per_layer,
        }
        common.save_json(out / f"eval_{run_path.stem}.json", ev)
        rows.append(ev)

    print(f"\n=== {args.dataset} summary ===")
    hdr = (
        f"{'run':38s} {'lyr':3s} {'top':4s} {'cov':5s} {'ARI':6s} {'NMI':6s} {'pur':5s} "
        f"{'sim':6s} {'hit':5s} {'simST':6s} {'hitST':6s} {'dup':5s}"
    )
    print(hdr)
    for ev in rows:
        for pl in ev["per_layer"]:
            print(
                f"{ev['run']:38s} {pl['layer']:3d} {pl['n_topics']:4d} {pl['coverage']:.2f}  "
                f"{pl['ari']:.3f}  {pl['nmi']:.3f}  {pl['purity']:.2f}  "
                f"{(pl['name_label_sim_mean'] or 0):.3f}  {(pl['name_hit_rate'] or 0):.2f}  "
                f"{(pl['name_label_sim_st_mean'] or 0):.3f}  {(pl['name_hit_rate_st'] or 0):.2f}  "
                f"{pl['dup_name_frac']:.2f}"
            )


if __name__ == "__main__":
    main()
