# SAM3 research notes for #5213 (agent survey, 2026-10-25; unverified items flagged)
- Exemplars: box + pos/neg label, ROI-pooled, concatenated with text tokens. SAME IMAGE ONLY (paper's 1-exemplar eval
  samples the box from that image's GT). Cross-image: open issue facebookresearch/sam3#426; workaround = paste reference
  + target on one canvas (FSS-SAM3; arXiv 2604.05433, which reports negative prompts often HURT / collapse few-shot).
  => a Good-vote box on image A is not a native exemplar for pool image B; canvas trick ~2x encoder cost per pool image.
- Paper exemplar gains (same image): 1-exemplar AP+ COCO text 58.8 vs image 76.8 vs text+img 78.1 (mapping unverified);
  interactive +21.6 cgF1 after 3 exemplar clicks, plateau ~4.
- Few-shot FT (paper, full FT, 40 ep): ODinW13 BoxAP 0/1/3/5/10-shot/full 61.0/66.1/69.1/70.2/71.8/75.8 (GDINO1.5Pro
  58.7..72.4). RF100-VL 15.2 zero-shot -> 36.5 10-shot. Official README_TRAIN.md. LoRA: Sompote/sam3_lora (text enc
  frozen), MedSAM3 LoRA arXiv 2608.18731, defects 2609.00469; NTIRE26 CD-FSOD team tuned only text encoder (2604.11998).
  No linear probe / light head on frozen SAM3 features found.
- SAM/SAM2 masks + DINO: No Time to Train (2507.02798) SAM2-L proposals + DINOv2-L mask-avg prototypes, COCO FSOD nAP
  36.6 @10-shot, beats fine-tuned CD-ViTO/DE-ViT; DINOv3 WORSE than DINOv2 there (32.8-33.8 vs 35.7). Matcher (2305.13310)
  52.7 1-shot mIoU COCO-20i. Region-Based Representations Revisited (2402.02352): mask-pooled DINOv2 for retrieval.
  No head-to-head mask-pool vs patch-max for retrieval ranking found -> ours to measure.
- Cost: ~848M params (vision ~450M, text ~300M), fixed 1008x1008 input; 30 ms/img H200 (paper). Vision features
  cacheable across prompts. SAM3.1 = video only (Object Multiplex), image detector unchanged. EfficientSAM3 (~90M,
  2511.15833) + SAM3-LiteText (2602.12173, text enc only) are lighter variants.
- Weaknesses: simple noun phrases only (Appendix B), long/referring phrases unsupported; fine-grained/OOD weak;
  "dog, cat, frisbee" read as one phrase (third party) -> one NP per call; small/dense objects not analysed.
