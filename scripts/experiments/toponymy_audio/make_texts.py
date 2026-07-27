"""Audio → text variants: the ``object_to_text`` gap in the Toponymy plan.

Each variant produces ``RESULTS/<dataset>/texts_<variant>.json``: a list of
short strings, one per clip, aligned with ``meta.json``. Toponymy mines its
contrastive keyphrases from these strings and shows them to the naming LLM
as exemplars — they are the only place "what the audio sounds like" enters
the pipeline.

Variants
--------
``clap_audioset``
    Zero-shot tags: CLAP similarity between each clip and the 527 AudioSet
    class names (embedded as "The sound of <label>"), top-k terms. No new
    models — reuses the dataset's own CLAP embeddings + text branch.
``clap_esc50vocab``
    Oracle-vocabulary upper bound: same, but the vocabulary is the 50
    ESC-50 category names. Shows what a perfectly matched vocab gives
    (label leakage — never a production option).
``whisper``
    OpenAI Whisper (small) transcripts. Expected to work for speech and
    produce noise/hallucination for non-speech.
``caption``
    A dedicated audio-captioning model (MU-NLPC/whisper-small-audio-captioning,
    trained on AudioCaps+Clotho): one natural-language sentence per clip.

Usage::

    python make_texts.py esc50 clap_audioset [--topk 5] [--embedder clap]
    python make_texts.py esc50 whisper [--limit 400]
    python make_texts.py esc50 caption
"""

from __future__ import annotations

import argparse
import csv
import io

import common

common.setup_env()

import numpy as np  # noqa: E402

AUDIOSET_CSV = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"

ESC50_CATEGORIES: list[str] | None = None  # loaded from meta.json categories


def audioset_labels() -> list[str]:
    cache = common.WORK / "audioset_labels.txt"
    if cache.exists():
        return cache.read_text().splitlines()
    import requests

    resp = requests.get(AUDIOSET_CSV, timeout=60)
    resp.raise_for_status()
    rows = list(csv.DictReader(io.StringIO(resp.text)))
    labels = [row["display_name"].strip().strip('"') for row in rows]
    cache.write_text("\n".join(labels))
    return labels


def clap_tags(meta, emb, vocab, topk, embedder_name, template) -> list[str]:
    """Top-k zero-shot vocabulary terms per clip by CLAP text-audio cosine."""
    from vtscore import media as media_registry

    embedder = media_registry.get_embedder(embedder_name)
    embedder.load_models()
    tvecs = []
    for term in vocab:
        v = embedder.embed_text(template.format(term))
        tvecs.append(v)
    tmat = np.stack(tvecs).astype(np.float32)
    tmat /= np.linalg.norm(tmat, axis=1, keepdims=True) + 1e-9
    emb_n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    sims = emb_n @ tmat.T  # (n_clips, n_vocab)
    order = np.argsort(-sims, axis=1)[:, :topk]
    return [", ".join(vocab[j] for j in row) for row in order]


def whisper_transcripts(meta, model_size="small") -> list[str]:
    import numpy as np
    import torch
    import whisper

    from vtscore.media.audio.decode import decode_audio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_size, device=device)
    texts = []
    for i, m in enumerate(meta):
        # Decode via vtscore's helper: it prefers libsndfile and only shells
        # out to ffmpeg for codecs libsndfile can't parse, whereas whisper's own
        # loader always needs an ffmpeg grid compute nodes don't have.
        audio, _ = decode_audio(m["wav_path"], sr=16000, mono=True)
        result = model.transcribe(
            audio.astype(np.float32),
            temperature=0.0,
            no_speech_threshold=0.4,
            fp16=(device == "cuda"),
        )
        txt = " ".join(seg["text"].strip() for seg in result["segments"]).strip()
        texts.append(txt)
        if (i + 1) % 100 == 0:
            print(f"  whisper {i + 1}/{len(meta)}", flush=True)
    return texts


def captions(meta, batch_size=16) -> list[str]:
    """MU-NLPC/whisper-small-audio-captioning — a real audio captioner."""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperTokenizer, WhisperFeatureExtractor

    from vtscore.media.audio.decode import decode_audio

    model_id = "MU-NLPC/whisper-small-audio-captioning"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device).eval()
    tokenizer = WhisperTokenizer.from_pretrained(model_id, language="en", task="transcribe")
    fe = WhisperFeatureExtractor.from_pretrained(model_id)
    style_prefix = "clotho > caption: "
    style_ids = tokenizer("", text_target=style_prefix, return_tensors="pt").labels[:, :-1].to(device)

    texts = []
    for i in range(0, len(meta), batch_size):
        chunk = meta[i : i + batch_size]
        feats = []
        for m in chunk:
            audio, sr = decode_audio(m["wav_path"], sr=fe.sampling_rate, mono=True)
            feats.append(fe(audio, sampling_rate=fe.sampling_rate, return_tensors="pt").input_features[0])
        batch = torch.stack(feats).to(device)
        with torch.no_grad():
            # The checkpoint's custom class takes forced_ac_decoder_ids; with
            # the vanilla WhisperForConditionalGeneration we force the same
            # style prefix via decoder_input_ids.
            out = model.generate(
                inputs=batch,
                decoder_input_ids=style_ids.repeat(len(chunk), 1),
                max_length=80,
            )
        for row in out:
            t = tokenizer.decode(row, skip_special_tokens=True)
            t = t.replace(style_prefix.strip(), "").strip(" :>")
            texts.append(t)
        print(f"  caption {min(i + batch_size, len(meta))}/{len(meta)}", flush=True)
    return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("variant", choices=["clap_audioset", "clap_esc50vocab", "whisper", "caption"])
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--embedder", default="clap")
    ap.add_argument("--limit", type=int, default=None, help="only first N clips (probe)")
    ap.add_argument("--whisper-size", default="small")
    ap.add_argument("--out-suffix", default="", help="append to output name, e.g. _k3 → texts_clap_audioset_k3.json")
    args = ap.parse_args()

    out = common.ds_dir(args.dataset)
    meta = common.load_json(out / "meta.json")
    emb = np.load(out / f"embeddings_{args.embedder}.npy")
    if args.limit:
        meta = meta[: args.limit]
        emb = emb[: args.limit]

    timings: dict = {}
    with common.timed(f"texts_{args.variant}", timings):
        if args.variant == "clap_audioset":
            texts = clap_tags(meta, emb, audioset_labels(), args.topk, args.embedder, "The sound of {}")
        elif args.variant == "clap_esc50vocab":
            vocab = sorted({m["category"].replace("_", " ") for m in meta})
            texts = clap_tags(meta, emb, vocab, min(args.topk, 3), args.embedder, "The sound of {}")
        elif args.variant == "whisper":
            texts = whisper_transcripts(meta, args.whisper_size)
        elif args.variant == "caption":
            texts = captions(meta)

    common.save_json(out / f"texts_{args.variant}{args.out_suffix}.json", texts)
    common.save_json(
        out / f"texts_{args.variant}{args.out_suffix}_info.json",
        {
            "variant": args.variant,
            "n": len(texts),
            "empty_frac": round(sum(1 for t in texts if not t.strip()) / len(texts), 4),
            "timings_s": timings,
            "params": vars(args),
        },
    )
    print("sample texts:")
    for t, m in list(zip(texts, meta))[:8]:
        print(f"  [{m['category']}] {t[:110]}")


if __name__ == "__main__":
    main()
