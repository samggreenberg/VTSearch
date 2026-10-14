"""#4837: the audio half beyond ESC-50 -- urban sound, music genre, spoken words -- against CLAP-general.

Per dataset, N clips per class (seeded), embedded through the shipped embedders. For each embedder:
  * text sort: AP of each class's typed query over all the sampled clips, mean over classes;
  * zero-shot top-1: a clip's best-scoring class query;
  * clip-to-clip 1-NN accuracy (leave-one-out): how well the space separates the classes without text.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

import vtscore.media as vm

D = Path("/exp/scale26/datasets/external/vtsearch-demos")  # the GRID's shared demo cache
N = 30
rng = random.Random(0)


def urbansound():
    rows = list(csv.DictReader(open(D / "UrbanSound8K/metadata/UrbanSound8K.csv")))
    by = {}
    for r in rows:
        by.setdefault(r["class"], []).append(D / "UrbanSound8K/audio" / f"fold{r['fold']}" / r["slice_file_name"])
    return {c.replace("_", " "): rng.sample(sorted(v), N) for c, v in sorted(by.items())}


def gtzan():
    out = {}
    for g in sorted(p.name for p in (D / "gtzan/genres").iterdir() if p.is_dir()):
        files = sorted(
            p for p in (D / "gtzan/genres" / g).iterdir() if p.suffix == ".wav" and not p.name.startswith("._")
        )
        out[f"{g} music"] = rng.sample(files, N)
    return out


def speech():
    words = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "dog", "bird", "happy", "house"]
    out = {}
    for w in words:
        files = sorted(p for p in (D / "speech_commands_v2" / w).iterdir() if p.suffix == ".wav")
        out[f'someone saying "{w}"'] = rng.sample(files, N)
    return out


def score(emb_name, classes):
    emb = vm.get_embedder(emb_name)
    names = list(classes)
    paths = [p for c in names for p in classes[c]]
    lab = np.array([i for i, c in enumerate(names) for _ in classes[c]])
    vecs = [emb.embed_media({"media_path": str(p)}) for p in paths]
    ok = np.array([v is not None for v in vecs])  # GTZAN ships a corrupt jazz.00054.wav
    lab = lab[ok]
    A = np.stack([v for v in vecs if v is not None]).astype(np.float32)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    T = np.stack([emb.embed_text(c) for c in names]).astype(np.float32)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    S = A @ T.T
    aps = []
    for ci in range(len(names)):
        o = np.argsort(-S[:, ci])
        rel = (lab[o] == ci).astype(float)
        prec = np.cumsum(rel) / np.arange(1, len(rel) + 1)
        aps.append(float((prec * rel).sum() / rel.sum()))
    G = A @ A.T
    np.fill_diagonal(G, -9)
    return {
        "embedder": emb_name,
        "classes": len(names),
        "clips": int(ok.sum()),
        "text_mAP": round(float(np.mean(aps)), 4),
        "zero_shot_top1": round(float((S.argmax(1) == lab).mean()), 4),
        "nn1": round(float((lab[G.argmax(1)] == lab).mean()), 4),
        "chance_AP": round(1 / len(names), 4),
    }


ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, help="JSON lines, appended")
out = open(ap.parse_args().out, "a")
for ds, make in (("urbansound8k", urbansound), ("gtzan", gtzan), ("speech_commands", speech)):
    classes = make()
    for emb in ("clap_general", "embeddinggemma2_audio"):
        r = {"dataset": ds, **score(emb, classes)}
        print(r, flush=True)
        out.write(json.dumps(r) + "\n")
        out.flush()
print("DONE", flush=True)
