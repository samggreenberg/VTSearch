"""Image → text variants: the ``object_to_text`` gap in the Toponymy plan.

Each variant produces ``RESULTS/<dataset>/texts_<variant>.json``: a list of
short strings, one per image, aligned with ``meta.json``. Toponymy mines its
contrastive keyphrases from these strings and shows them to the naming LLM
as exemplars — they are the only place "what the image shows" enters the
pipeline.

Variants
--------
``tags_oi600``
    Zero-shot tags: SigLIP similarity between each image and the ~600
    OpenImages V7 boxable class names ("a photo of <label>."), top-k terms.
    The medium generic vocabulary — the direct analog of the audio study's
    AudioSet-527 default.
``tags_in21k``
    Same, against the ~21k ImageNet-21k WordNet lemmas. The huge generic
    vocabulary: includes dog breeds, flowers, furniture… tests whether
    "bigger vocab" keeps winning or drowns in near-duplicate terms.
``tags_oracle``
    Oracle-vocabulary upper bound: the dataset's own category names
    (label leakage — never a production option).
``caption_blip``
    Salesforce BLIP base captioner (~1 GB): one short sentence per image.
    The light generic captioner.
``caption_florence`` / ``ocr_florence``
    Microsoft Florence-2-base (~0.5 GB, task-prompted): ``<CAPTION>`` and
    ``<OCR>`` passes. OCR is the born-digital/document signal a photo
    captioner can't see. (Runs via trust_remote_code; a load failure is
    itself a result.)
``caption_qwen3b``
    Qwen2.5-VL-3B-Instruct: one instructed line per image, told to state
    the type (photo/screenshot/document) and quote key visible text. The
    heavy do-everything option.
``blip_plus_ocr``
    Offline combine (no GPU): BLIP caption + " Text: <ocr_florence>" when
    OCR found anything. The cheap hybrid.

Usage::

    python make_texts.py caltech101 tags_oi600 [--topk 5]
    python make_texts.py rvl_cdip caption_qwen3b [--limit 200]
    python make_texts.py mixed blip_plus_ocr
"""

from __future__ import annotations

import argparse
import csv
import io

import common

common.setup_env()

import numpy as np  # noqa: E402

OI_CSV = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv"
IN21K_LEMMAS = "https://storage.googleapis.com/bit_models/imagenet21k_wordnet_lemmas.txt"


def _fetch(url: str, cache_name: str) -> str:
    cache = common.WORK / cache_name
    if cache.exists():
        return cache.read_text()
    import requests

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    cache.write_text(resp.text)
    return resp.text


def openimages_labels() -> list[str]:
    rows = list(csv.reader(io.StringIO(_fetch(OI_CSV, "oi600.csv"))))
    # LabelName,DisplayName - no header in the boxable file historically;
    # tolerate either by skipping rows whose first col doesn't look like /m/.
    labels = [r[1].strip() for r in rows if len(r) >= 2 and r[0].startswith("/")]
    return sorted(set(labels))


def in21k_labels() -> list[str]:
    lines = _fetch(IN21K_LEMMAS, "in21k.txt").splitlines()
    # each line: "tench, Tinca tinca" - take the first lemma
    labels = [ln.split(",")[0].strip().replace("_", " ") for ln in lines if ln.strip()]
    return sorted(set(labels))


def siglip_tags(emb, vocab, topk, embedder_name, template, vocab_name) -> list[str]:
    """Top-k zero-shot vocabulary terms per image by SigLIP text-image cosine."""
    from vtscore import media as media_registry

    embedder = media_registry.get_embedder(embedder_name)
    embedder.load_models()

    cache = common.WORK / f"vocabmat_{vocab_name}_{embedder_name}.npy"
    if cache.exists():
        tmat = np.load(cache)
    else:
        tvecs = []
        for i, term in enumerate(vocab):
            tvecs.append(embedder.embed_text(template.format(term)))
            if (i + 1) % 2000 == 0:
                print(f"  vocab embed {i + 1}/{len(vocab)}", flush=True)
        tmat = np.stack(tvecs).astype(np.float32)
        np.save(cache, tmat)
    tmat = tmat / (np.linalg.norm(tmat, axis=1, keepdims=True) + 1e-9)
    emb_n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    sims = emb_n @ tmat.T  # (n_images, n_vocab)
    order = np.argsort(-sims, axis=1)[:, :topk]
    return [", ".join(vocab[j] for j in row) for row in order]


def _load_rgb(path, max_side=None):
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if max_side and max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
    return img


def blip_captions(meta, batch_size=32) -> list[str]:
    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor

    model_id = "Salesforce/blip-image-captioning-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id).to(device).eval()

    texts = []
    for i in range(0, len(meta), batch_size):
        chunk = meta[i : i + batch_size]
        images = [_load_rgb(m["img_path"]) for m in chunk]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30)
        texts.extend(processor.decode(row, skip_special_tokens=True).strip() for row in out)
        print(f"  blip {min(i + batch_size, len(meta))}/{len(meta)}", flush=True)
    return texts


def florence_generate(meta, task, batch_size=8, max_new_tokens=64) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    model_id = "microsoft/Florence-2-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float32)
        .to(device)
        .eval()
    )

    texts = []
    for i in range(0, len(meta), batch_size):
        chunk = meta[i : i + batch_size]
        images = [_load_rgb(m["img_path"]) for m in chunk]
        inputs = processor(text=[task] * len(images), images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
            )
        for row, img in zip(out, images):
            raw = processor.batch_decode([row], skip_special_tokens=False)[0]
            parsed = processor.post_process_generation(raw, task=task, image_size=(img.width, img.height))
            texts.append(str(parsed.get(task, "")).strip())
        print(f"  florence{task} {min(i + batch_size, len(meta))}/{len(meta)}", flush=True)
    return texts


QWEN_PROMPT = (
    "Describe this image in one concise line (at most 20 words) for a browsing catalog. "
    "If it is a screenshot, document, form, or scanned page, say its type and quote the most "
    "important visible text. Otherwise describe the subject specifically (species, breed, "
    "object type). Output only the line."
)


def qwen_captions(meta, batch_size=8, model_id="Qwen/Qwen2.5-VL-3B-Instruct") -> list[str]:
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()
    # Bound the vision-token budget so document pages don't explode the batch.
    processor = AutoProcessor.from_pretrained(model_id, min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28)
    # Batched generation on a decoder-only model requires LEFT padding —
    # right-padding makes the model continue from pad tokens.
    processor.tokenizer.padding_side = "left"

    texts = []
    for i in range(0, len(meta), batch_size):
        chunk = meta[i : i + batch_size]
        images = [_load_rgb(m["img_path"], max_side=1280) for m in chunk]
        messages = [
            [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": img}, {"type": "text", "text": QWEN_PROMPT}],
                }
            ]
            for img in images
        ]
        prompts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
        inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=48, do_sample=False)
        trimmed = [o[len(i_) :] for i_, o in zip(inputs.input_ids, out)]
        decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        texts.extend(d.strip() for d in decoded)
        print(f"  qwen {min(i + batch_size, len(meta))}/{len(meta)}", flush=True)
    return texts


def compose_mixed(variant: str) -> list[str]:
    """Assemble mixed-corpus texts by slicing the part datasets' texts.

    The mixed corpus is a seeded sample of the four part datasets
    (``part_indices`` in its prepare_info), so any per-image text variant
    already computed on the parts can be reused without re-running a
    captioner over the same images.
    """
    info = common.load_json(common.ds_dir("mixed") / "prepare_info.json")
    texts: list[str] = []
    for part in info["parts"]:
        part_texts = common.load_json(common.ds_dir(part) / f"texts_{variant}.json")
        texts.extend(str(part_texts[i]) for i in info["part_indices"][part])
    return texts


def combine_blip_ocr(out_dir) -> list[str]:
    caps = common.load_json(out_dir / "texts_caption_blip.json")
    ocrs = common.load_json(out_dir / "texts_ocr_florence.json")
    texts = []
    for cap, ocr in zip(caps, ocrs):
        ocr = " ".join(str(ocr).split())[:200]
        texts.append(f"{cap}. Text: {ocr}" if ocr else str(cap))
    return texts


VARIANTS = [
    "tags_oi600",
    "tags_in21k",
    "tags_oracle",
    "caption_blip",
    "caption_florence",
    "ocr_florence",
    "caption_qwen3b",
    "blip_plus_ocr",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("variant", choices=VARIANTS)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--embedder", default="siglip")
    ap.add_argument("--limit", type=int, default=None, help="only first N images (probe)")
    ap.add_argument("--out-suffix", default="", help="append to output name, e.g. _k3")
    ap.add_argument(
        "--compose",
        action="store_true",
        help="mixed only: slice the part datasets' texts instead of re-running the model",
    )
    args = ap.parse_args()

    out = common.ds_dir(args.dataset)
    meta = common.load_json(out / "meta.json")
    emb = np.load(out / f"embeddings_{args.embedder}.npy")
    if args.limit:
        meta = meta[: args.limit]
        emb = emb[: args.limit]

    timings: dict = {}
    with common.timed(f"texts_{args.variant}", timings):
        if args.compose:
            if args.dataset != "mixed":
                raise SystemExit("--compose only applies to the mixed dataset")
            texts = compose_mixed(args.variant)
        elif args.variant == "tags_oi600":
            texts = siglip_tags(emb, openimages_labels(), args.topk, args.embedder, "a photo of {}.", "oi600")
        elif args.variant == "tags_in21k":
            texts = siglip_tags(emb, in21k_labels(), args.topk, args.embedder, "a photo of {}.", "in21k")
        elif args.variant == "tags_oracle":
            vocab = sorted({m["category"].split(":")[-1].replace("_", " ").replace("-", " ") for m in meta})
            texts = siglip_tags(
                emb, vocab, min(args.topk, 3), args.embedder, "a photo of {}.", f"oracle_{args.dataset}"
            )
        elif args.variant == "caption_blip":
            texts = blip_captions(meta)
        elif args.variant == "caption_florence":
            texts = florence_generate(meta, "<CAPTION>")
        elif args.variant == "ocr_florence":
            texts = florence_generate(meta, "<OCR>", max_new_tokens=128)
        elif args.variant == "caption_qwen3b":
            texts = qwen_captions(meta)
        elif args.variant == "blip_plus_ocr":
            texts = combine_blip_ocr(out)

    common.save_json(out / f"texts_{args.variant}{args.out_suffix}.json", texts)
    common.save_json(
        out / f"texts_{args.variant}{args.out_suffix}_info.json",
        {
            "variant": args.variant,
            "n": len(texts),
            "empty_frac": round(sum(1 for t in texts if not str(t).strip()) / len(texts), 4),
            "timings_s": timings,
            "params": vars(args),
        },
    )
    print("sample texts:")
    for t, m in list(zip(texts, meta))[:8]:
        print(f"  [{m['category']}] {str(t)[:110]}")


if __name__ == "__main__":
    main()
