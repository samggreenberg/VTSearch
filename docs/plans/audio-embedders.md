# Audio embedder candidates: CLAP / SpeechCLAP / AudioCLIP / Omni-Embed

Status: **ParaSpeechCLAP shipped** (speech-style audio embedder). Other three
candidates evaluated and declined; see below.

## Background

VTSearch shipped five audio embedders before this work:

| Embedder | Model | Dim | Text queries? |
|---|---|---|---|
| `clap` (default) | `laion/clap-htsat-unfused` | 512 | ✅ |
| `clap_general` | `laion/larger_clap_general` | 512 | ✅ |
| `clap_music` | `laion/larger_clap_music_and_speech` | 512 | ✅ |
| `ast` | `MIT/ast-finetuned-audioset-10-10-0.4593` | 768 | ❌ |
| `whisper_encoder` | `openai/whisper-base` | 512 | ❌ |

The four candidates raised were CLAP, SpeechCLAP, AudioCLIP, and Omni-Embed-Audio.

## Decisions

- **CLAP** — already covered by three laion variants; nothing to add.
- **AudioCLIP** — declined. 2021 ESResNeXt+CLIP, not `transformers`-native,
  weaker than CLAP for audio↔text, and its distinctive image branch is
  irrelevant to audio search. High friction, ~no marginal value.
- **Omni-Embed-Audio** (`nvidia/omni-embed-nemotron-3b`) — declined. ~5B params,
  GPU-only (A100/H100), bf16 + flash-attention, NVIDIA non-commercial licence,
  2048-dim. Conflicts with the CPU-runnable embedder model; non-commercial
  licence is a dealbreaker.
- **msclap** (Microsoft CLAP) — declined. The `msclap` pip package pins
  `numpy<2.0.0` and pulls `torchaudio`, forcing an app-wide numpy downgrade,
  and is not `transformers`-native. Capability overlaps heavily with the
  existing laion CLAP variants. Harder integration for a smaller gain.
- **ParaSpeechCLAP** — **shipped.** The only candidate that fills a genuine
  gap: a text-queryable *speech-style* embedder (the existing `ast` /
  `whisper_encoder` speech embedders have no paired text tower).

## What shipped

`paraspeechclap` audio embedder (`vtscore/media/audio/embedder_paraspeechclap.py`):

- Maps speech clips and rich style descriptions ("a deep, raspy voice", "a
  whispered, anxious style") into a shared 768-dim space.
- Reconstructed from the released `ajd12342/paraspeechclap-combined` checkpoint
  (MIT) via a vendored minimal architecture
  (`vtscore/media/audio/_paraspeechclap_model.py`): WavLM-Large speech encoder
  (`microsoft/wavlm-large`, MIT) + Granite text encoder
  (`ibm-granite/granite-embedding-278m-multilingual`, Apache-2.0) + two
  projection heads. No new pip dependencies (uses torch / transformers /
  librosa / huggingface_hub, all already present).
- `combined` variant chosen as the most versatile (covers both speaker-level
  intrinsic attributes and utterance-level situational attributes).
- Opt-in, not the default. Clips capped at 30 s to bound CPU memory/latency.

## Open follow-ups

- **Download size.** The embedder loads the WavLM + Granite base weights via
  `from_pretrained` and then overlays the full fine-tuned checkpoint, so the
  on-disk footprint is ~4.5 GB (base weights are downloaded then largely
  overwritten). If the checkpoint is confirmed to carry the complete encoder
  weights (upstream `inference.py` attempts `strict=True`, which implies it
  does), the base loads could be switched to `AutoModel.from_config` to skip
  ~2.3 GB of redundant downloads. Kept on `from_pretrained` for now for
  robustness against a projections-only checkpoint.
- **Variants.** Only `combined` is wired. The `intrinsic` (speaker-level) and
  `situational` (utterance-level) checkpoints exist on HF and could be added as
  separate embedders if users want sharper specialisation.
- **Real-weights integration test.** Property/registration tests run without
  downloading weights (mocked). A GPU-marked end-to-end test that loads the
  real checkpoint and checks speech↔text similarity ordering would be valuable
  but pulls ~4.5 GB, so it is deferred.
