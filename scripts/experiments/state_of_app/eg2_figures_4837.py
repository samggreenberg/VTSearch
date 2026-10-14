#!/usr/bin/env python
"""Figures and tables for #4837's side measurements: runtime, the other image sets, the audio half, click 0.

Reads the study dir's raw outputs and writes small CSVs (committed with the report) and PNGs:

* ``timing.csv`` / ``runtime.png``: images a second, SigLIP vs EmbeddingGemma 2 at each vision budget,
  on the V100S that built both pile cells and on 8 CPU threads (``embeddinggemma2/timing.py``, interleaved repeats);
* ``image_evals.csv``: ``vtscore.eval`` text-sort mAP and learned-sort F1 on four more image sets;
* ``audio.csv`` / ``audio.png``: the audio half against CLAP-general, text-sort mAP and clip 1-NN by domain;
* ``click0_by_class.csv`` / ``click0.png``: the typed query's AP per class, each embedder's own, and the
  query prefix's share of the gain.

    python eg2_figures_4837.py --study <study dir> --grid <...-eg2-4837-grid> --out <report dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK, MUTED, SIG, EG2 = "#0b0b0b", "#52514e", "#8a8985", "#2a78d6"
IMAGE_SETS = ("caltech101_m", "caltech256_a", "visual_genome_m", "enrico_m")


def runtime(study: Path, out: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in (study / "timing.jsonl").read_text().splitlines() if line.strip()]
    t = pd.DataFrame([r for r in rows if r["what"] == "bulk"])
    t["where"] = np.where(t["device"].str.startswith("cuda"), t["gpu"].str.replace("Tesla ", ""), "CPU, 8 threads")
    t["arm"] = np.where(
        t["embedder"] == "siglip", "SigLIP", "EmbeddingGemma 2, " + t["budget"].fillna(0).astype(int).astype(str)
    )
    agg = t.groupby(["where", "arm"], as_index=False).agg(
        per_s=("per_s", "median"), reps=("rep", "count"), n=("n", "first")
    )
    sig = agg[agg["arm"] == "SigLIP"].set_index("where")["per_s"]
    agg["slower_than_siglip"] = (agg["where"].map(sig) / agg["per_s"]).round(1)
    agg.to_csv(out / "timing.csv", index=False)

    order = ["SigLIP", "EmbeddingGemma 2, 70", "EmbeddingGemma 2, 140", "EmbeddingGemma 2, 280"]
    places = list(agg["where"].unique())
    fig, axes = plt.subplots(1, len(places), figsize=(5.4 * len(places), 3.2), squeeze=False)
    for ax, where in zip(axes[0], places, strict=True):
        a = agg[agg["where"] == where].set_index("arm").reindex([o for o in order if o in set(agg["arm"])])
        a = a.dropna(subset=["per_s"])
        colors = [SIG if i == "SigLIP" else EG2 for i in a.index]
        ax.barh(range(len(a)), a["per_s"], color=colors)
        ax.set_yticks(
            range(len(a)),
            [i.replace("EmbeddingGemma 2, ", "EG2, ") + (" tokens" if i != "SigLIP" else "") for i in a.index],
        )
        ax.set_xscale("log")
        ax.set_xlabel("images a second (log)")
        ax.set_title(where, fontsize=10)
        for i, v in enumerate(a["per_s"]):
            ax.text(v * 1.08, i, f"{float(f'{v:.2g}'):g}", va="center", fontsize=9, color=INK)
        ax.set_xlim(right=a["per_s"].max() * 4)
    fig.tight_layout()
    fig.savefig(out / "runtime.png", dpi=130)
    plt.close(fig)
    return agg


def image_evals(study: Path, out: Path) -> pd.DataFrame:
    rows = []
    for ds in IMAGE_SETS:
        for emb in ("siglip", "embeddinggemma2"):
            p = study / "eval" / f"{ds}__{emb}.json"
            if not p.exists():
                continue
            (r,) = json.loads(p.read_text())
            rows.append(
                {
                    "dataset": ds,
                    "embedder": emb,
                    "text_mAP": r["text_sort"]["mAP"],
                    "learned_F1": r["learned_sort"]["mean_f1"],
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out / "image_evals.csv", index=False)
    return df


def audio(study: Path, out: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in (study / "eval" / "audio_zeroshot.jsonl").read_text().splitlines()]
    for emb in ("clap_general", "embeddinggemma2_audio"):
        p = study / "eval" / f"esc50_m__{emb}.json"
        if p.exists():
            (r,) = json.loads(p.read_text())
            rows.append({"dataset": "esc50_m (vtscore.eval)", "embedder": emb, "text_mAP": r["text_sort"]["mAP"],
                         "learned_F1": r["learned_sort"]["mean_f1"]})  # fmt: skip
    df = pd.DataFrame(rows)
    df.to_csv(out / "audio.csv", index=False)
    sets = list(dict.fromkeys(df["dataset"]))
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    x = np.arange(len(sets))
    for k, (emb, color, label) in enumerate(
        (("clap_general", SIG, "CLAP-general (the default)"), ("embeddinggemma2_audio", EG2, "EmbeddingGemma 2"))
    ):
        v = df[df["embedder"] == emb].set_index("dataset").reindex(sets)["text_mAP"]
        ax.bar(x + (k - 0.5) * 0.38, v, width=0.38, color=color, label=label)
    chance = df.dropna(subset=["chance_AP"]).groupby("dataset")["chance_AP"].first() if "chance_AP" in df else {}
    for i, s in enumerate(sets):
        if s in chance:
            ax.plot([i - 0.4, i + 0.4], [chance[s]] * 2, color=MUTED, lw=1, ls=":")
    ax.set_xticks(x, [s.replace("_", " ") for s in sets], fontsize=9)
    ax.set_ylabel("typed-query mAP")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "audio.png", dpi=130)
    plt.close(fig)
    return df


def click0(grid: Path, out: Path) -> pd.DataFrame:
    b = pd.read_csv(grid / "text_baseline.csv")
    n = pd.read_csv(grid / "text_baseline_eg2_noprefix.csv")
    n = n[n["embedder"] == "embeddinggemma2"].assign(embedder="eg2_noprefix")
    d = pd.concat([b, n])
    d["class"] = d["category"].str.split("@").str[0]
    w = d.pivot_table(index="class", columns="embedder", values="text_AP")
    w["d_eg2"] = w["embeddinggemma2"] - w["siglip"]
    w["d_eg2_noprefix"] = w["eg2_noprefix"] - w["siglip"]
    w = w.sort_values("d_eg2")
    w.round(4).to_csv(out / "click0_by_class.csv")
    fig, ax = plt.subplots(figsize=(6.5, 9))
    y = np.arange(len(w))
    ax.barh(y, w["d_eg2"], color=EG2, label="EmbeddingGemma 2, query prefix (shipped)")
    ax.scatter(w["d_eg2_noprefix"], y, color=INK, s=10, zorder=3, label="without the prefix")
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(y, w.index, fontsize=7)
    ax.set_xlabel("typed query's AP: EmbeddingGemma 2 minus SigLIP")
    ax.legend(frameon=False, fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(out / "click0.png", dpi=130)
    plt.close(fig)
    return w


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--study", type=Path, required=True)
    ap.add_argument("--grid", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    (args.out / "figures").mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    print(runtime(args.study, args.out).to_string(index=False))
    print(image_evals(args.study, args.out).to_string(index=False))
    print(audio(args.study, args.out).to_string(index=False))
    w = click0(args.grid, args.out)
    print(w[["siglip", "embeddinggemma2", "eg2_noprefix"]].mean().round(4).to_string())
    for name in ("runtime.png", "audio.png", "click0.png"):
        (args.out / name).replace(args.out / "figures" / name)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
