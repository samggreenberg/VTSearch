"""#5395: does MM-GDINO-L score small objects low on transformers' own path too, or only in ``gdino_score_5395.py``?

On the grid, MM-GDINO-L's noun prompts rank small objects far worse than the two tiny checkpoints (airplane@small
AP 0.34 against 0.89), and the median small positive scores 0.26 against 0.74. This probe takes 30 small positives of
each class and 30 images with neither class, and scores them the plain way a transformers user would: the processor
on the image and one prompt ("airplane."), one forward pass, fp32, no backbone cache, no packing. The score is the
scorer's (best query of the phrase-token mean). It does so at the checkpoints' own resize (shortest edge 800) and at
1200, in case small objects only need more pixels. Writes one row per model x class x size x positive/negative.

    python gdino_small_probe_5395.py --out CSV
"""

from __future__ import annotations

import argparse
import io

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sam3_score_5213 import _open_member  # noqa: E402

MODELS = {
    "tiny": "IDEA-Research/grounding-dino-tiny",
    "mmt": "openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det",
    "mml": "openmmlab-community/mm_grounding_dino_large_o365v2_oiv6_goldg",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="coco_better")
    ap.add_argument("--classes", nargs="+", default=["airplane", "kite"])
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()

    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor  # noqa: PLC0415

    from _cells_io import load_medias  # noqa: PLC0415
    from vtscore.config import EMBEDDINGS_DIR  # noqa: PLC0415

    med = load_medias(EMBEDDINGS_DIR / f"{a.dataset}__siglip.pkl")
    rng = np.random.default_rng(0)
    cats = {i: m.get("categories") or [] for i, m in med.items()}
    pick = {
        c: list(rng.choice(sorted(i for i, cs in cats.items() if f"{c}@small" in cs), a.n, replace=False))
        for c in a.classes
    }
    clean = sorted(i for i, cs in cats.items() if not any(x.split("@")[0] in a.classes for x in cs))
    neg = list(rng.choice(clean, a.n, replace=False))
    zips = {}
    images = {
        i: Image.open(io.BytesIO(_open_member(med[i]["origin_name"], zips))).convert("RGB")
        for i in {*neg, *(i for v in pick.values() for i in v)}
    }
    del med

    rows = []
    for tag, name in MODELS.items():
        proc = AutoProcessor.from_pretrained(name)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(name, dtype=torch.float32).to("cuda").eval()
        special = set(proc.tokenizer.all_special_ids) | {proc.tokenizer.convert_tokens_to_ids(".")}
        for c in a.classes:
            for size in (800, 1200):
                for kind, ids in (("pos", pick[c]), ("neg", neg)):
                    for i in ids:
                        inp = proc(
                            images=images[i],
                            text=f"{c}.",
                            return_tensors="pt",
                            size={"shortest_edge": size, "longest_edge": round(size * 1333 / 800)},
                        ).to("cuda")
                        with torch.inference_mode():
                            p = model(**inp).logits.sigmoid()[0]
                        toks = [k for k, t in enumerate(inp["input_ids"][0].tolist()) if t not in special]
                        rows.append(
                            {
                                "model": tag,
                                "class": c,
                                "size": size,
                                "kind": kind,
                                "id": i,
                                "score": float(p[:, toks].mean(-1).max()),
                            }
                        )
            d = pd.DataFrame(rows)
            d = d[(d["model"] == tag) & (d["class"] == c)]
            common.log(
                f"{tag} {c}: " + d.groupby(["size", "kind"])["score"].median().round(3).to_string().replace("\n", "; ")
            )
        del model
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(a.out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
