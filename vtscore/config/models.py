"""Hugging Face identifiers and per-checkpoint constants for every embedder.

Identifiers *only*: nothing here downloads or loads anything at import time.
The actual fetch is lazy, driven by each embedder's ``load_models()``.  Sample
rates, embedding dimensions and normalisation statistics live beside the id they
belong to, because they are properties of that specific checkpoint rather than
tunables anyone should change.
"""

from __future__ import annotations

# Model IDs
CLAP_MODEL_ID = "laion/clap-htsat-unfused"
CLAP_SAMPLE_RATE = 48000  # CLAP model expected input sample rate
XCLIP_MODEL_ID = "microsoft/xclip-base-patch32"
E5_MODEL_ID = "intfloat/e5-base-v2"
SIGLIP_MODEL_ID = "google/siglip-base-patch16-224"
SIGLIP2_MODEL_ID = "google/siglip2-base-patch16-224"
# SigLIP2-L: the SigLIP 2 SO400M/384 checkpoint, loaded through ``transformers``
# like its base sibling (SigLIP 2 has a first-party HF port, so there is no
# reason to route it through open_clip the way ``SIGLIP_L_MODEL_ID`` is).  The
# fixed-resolution ``patch14-384`` variant, not the NaFlex one, so the standard
# ``AutoProcessor`` image pipeline applies.  Emits 1152-d vectors, so its
# galleries are *not* interchangeable with the 768-d base SigLIP 2.
SIGLIP2_L_MODEL_ID = "google/siglip2-so400m-patch14-384"
# SigLIP-L: the SO400M/384 checkpoint, loaded via ``open_clip`` (not
# transformers) so its 1152-d vectors match galleries produced by open_clip's
# own ``ViT-SO400M-14-SigLIP-384`` model.  The arch name is the open_clip
# model key; the ``webli`` tag selects the WebLI-pretrained weights.
SIGLIP_L_MODEL_ID = "ViT-SO400M-14-SigLIP-384"
SIGLIP_L_PRETRAINED = "webli"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
# CLIP ViT-L/14, **evaluation only** (#3292): a single-vector, language-aligned
# encoder from a different lineage than SigLIP, used to test whether #3287's
# `calibration_fraction` optimum follows single-vector geometry or just the
# SigLIP family.  Chosen over the base/32 checkpoint above because it emits
# **768-d** vectors, exactly like `SIGLIP_MODEL_ID`, so a difference between the
# two cannot be "CLIP's vectors are narrower".  Not offered in the app -- see
# `ImageClipLargeEmbedder.eval_only`.
CLIP_L_MODEL_ID = "openai/clip-vit-large-patch14"
DINOV2_MODEL_ID = "facebook/dinov2-base"
DINOV3_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
EUPE_MODEL_ID = "https://huggingface.co/facebook/EUPE-ViT-B/resolve/main/EUPE-ViT-B.pt"
"""Direct URL to the real EUPE ViT-B/16 weights on Hugging Face.

Loaded via :func:`torch.hub.load` from the ``facebookresearch/EUPE`` GitHub
repo with this URL passed as the ``weights`` kwarg.  The HF repo
``facebook/EUPE-ViT-B`` is ungated; the underlying weights are released
under Meta's FAIR Noncommercial Research Licence (surfaced to users via
``MediaEmbedder.license_notice`` on the EUPE embedder).

Not the same model as ``facebook/PE-Core-B16-224``; that was Meta's
Perception Encoder Core, which the dev "eupe" slug was confusingly
aliased to via a broken ``AutoModel.from_pretrained`` path (the PE-Core
HF repo has no ``config.json`` so ``AutoModel`` could never load it).
"""
CLAP_MUSIC_MODEL_ID = "laion/larger_clap_music_and_speech"
CLAP_GENERAL_MODEL_ID = "laion/larger_clap_general"
AST_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_SAMPLE_RATE = 16000  # AST expects 16 kHz mono
# BEATs: Microsoft's self-supervised audio encoder (MIT, part of ``microsoft/unilm``).
# The official weights are published as loose ``.pt`` files on Azure blob storage
# rather than on the Hub, so we pull the ``iter3+`` AudioSet-2M checkpoint from an
# MIT-licensed Hub mirror of that release. ``iter3_plus_AS2M`` is the
# self-supervised encoder, *not* one of the AudioSet-finetuned classifier
# variants: it has no prediction head, which is what we want for embeddings.
BEATS_CHECKPOINT_REPO = "lpepino/beats_ckpts"
BEATS_CHECKPOINT_FILE = "BEATs_iter3_plus_AS2M.pt"
BEATS_SAMPLE_RATE = 16000  # BEATs expects 16 kHz mono
BEATS_EMBED_DIM = 768
BEATS_MAX_SAMPLES = 16000 * 10  # cap clips at the 10 s AudioSet window BEATs was trained on
BEATS_MIN_SAMPLES = 16000  # zero-pad anything shorter, so short clips still yield patches
# Global fbank normalisation constants baked into the released BEATs
# checkpoints; the encoder expects ``(fbank - mean) / (2 * std)``.
BEATS_FBANK_MEAN = 15.41663
BEATS_FBANK_STD = 6.55582
WHISPER_MODEL_ID = "openai/whisper-base"
WHISPER_SAMPLE_RATE = 16000  # Whisper expects 16 kHz mono
# ParaSpeechCLAP: dual-encoder speech↔text "style" CLAP (MIT-licensed).
# Unlike the AST / Whisper speech embedders, it has a paired text tower, so
# text queries like "a deep, raspy voice" or "a whispered, anxious style" land
# in the same space as the speech embeddings.  Reconstructed from the upstream
# checkpoint via ``_paraspeechclap_model.py`` (WavLM speech + Granite text +
# projection heads); the ``combined`` variant covers both speaker-level
# (pitch/texture/clarity) and utterance-level (emotion/speaking-style) attributes.
PARASPEECHCLAP_SPEECH_MODEL_ID = "microsoft/wavlm-large"
PARASPEECHCLAP_TEXT_MODEL_ID = "ibm-granite/granite-embedding-278m-multilingual"
PARASPEECHCLAP_CHECKPOINT_REPO = "ajd12342/paraspeechclap-combined"
# Upstream renamed the released weights to ``slap-combined.pth.tar``; the old
# ``paraspeechclap-combined.pth.tar`` was removed and now 404s (issue #2635).
PARASPEECHCLAP_CHECKPOINT_FILE = "slap-combined.pth.tar"
PARASPEECHCLAP_EMBED_DIM = 768
PARASPEECHCLAP_SAMPLE_RATE = 16000  # WavLM expects 16 kHz mono
PARASPEECHCLAP_MAX_SAMPLES = 16000 * 30  # cap clips at 30 s to bound CPU memory/latency
BGE_MODEL_ID = "BAAI/bge-base-en-v1.5"
LANGUAGEBIND_VIDEO_MODEL_ID = "LanguageBind/LanguageBind_Video_V1.5_FT"
VIDEOMAE_MODEL_ID = "OpenGVLab/VideoMAEv2-Base"
"""Hugging Face repo for VideoMAE v2 Base weights.

Loaded via ``AutoModel.from_pretrained(..., trust_remote_code=True)``.
Vision-only encoder with no paired text tower, so the embedder
sets ``supports_text=False`` and :meth:`embed_text` returns ``None``.
The masked-autoencoder objective produces unusually strong action /
motion features compared to image-only encoders applied per frame.
"""
