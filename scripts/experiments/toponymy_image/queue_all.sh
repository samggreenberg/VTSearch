#!/bin/bash
# Serialized experiment queue — the grid GPU nodes run in Exclusive_Process
# compute mode, so CUDA stages must not overlap. Each step logs and continues
# on failure (a variant that crashes the pipeline is itself a result).
set -uo pipefail
RUN=/exp/sgreenberg/experiments/toponymy-image/run.sh
LOG=/exp/sgreenberg/experiments/toponymy-image/queue_all.log

step() {
    echo "=== STEP: $* — $(date +%H:%M:%S)" | tee -a "$LOG"
    "$RUN" "$@" >>"$LOG" 2>&1 || echo "=== STEP FAILED: $*" | tee -a "$LOG"
}

# ---- prepare (download + SigLIP embed) --------------------------------
# caltech101 prepared during smoke test
step prepare_dataset.py stanford_dogs --per-cat 15
step prepare_dataset.py enrico
step prepare_dataset.py rvl_cdip --per-cat 100
step prepare_dataset.py mixed --per-source 500

# ---- image -> text variants -------------------------------------------
for ds in caltech101 stanford_dogs enrico rvl_cdip; do
    step make_texts.py $ds tags_oi600
    step make_texts.py $ds tags_in21k
    step make_texts.py $ds tags_oracle
    step make_texts.py $ds caption_blip
    step make_texts.py $ds caption_florence
    step make_texts.py $ds ocr_florence
    step make_texts.py $ds caption_qwen3b
    step make_texts.py $ds blip_plus_ocr
done
# mixed reuses the part datasets' texts (same images, seeded sample)
for v in tags_oi600 tags_in21k caption_blip caption_florence ocr_florence caption_qwen3b blip_plus_ocr; do
    step make_texts.py mixed $v --compose
done
step make_texts.py mixed tags_oracle

# ---- toponymy fits: all variants under the no-LLM namer ---------------
# (blip_plus_ocr only where OCR has signal: screens/docs/mixed)
for ds in caltech101 stanford_dogs; do
    for v in tags_oi600 tags_in21k tags_oracle caption_blip caption_florence caption_qwen3b; do
        step run_toponymy.py $ds $v keyphrase
    done
done
for ds in enrico rvl_cdip mixed; do
    for v in tags_oi600 tags_in21k tags_oracle caption_blip caption_florence blip_plus_ocr caption_qwen3b; do
        step run_toponymy.py $ds $v keyphrase
    done
done

# ---- subset scenario: browsing the RESULTS of a Find ------------------
# "dog photo detector" -> 10 breeds x 20 images
for v in tags_oi600 tags_in21k caption_blip caption_qwen3b; do
    step run_toponymy.py stanford_dogs $v keyphrase --n-cats 10 --per-cat 20 --out-suffix _sub10x20
    step run_toponymy.py stanford_dogs $v hf --n-cats 10 --per-cat 20 --out-suffix _sub10x20
done
# "faxed document detector" -> 4 doc types x 50 pages
for v in tags_oi600 tags_in21k caption_blip caption_qwen3b; do
    step run_toponymy.py rvl_cdip $v keyphrase --n-cats 4 --per-cat 50 --out-suffix _sub4x50
    step run_toponymy.py rvl_cdip $v hf --n-cats 4 --per-cat 50 --out-suffix _sub4x50
done

# ---- LLM namer on the main contenders (last: expendable if time runs out)
for ds in caltech101 stanford_dogs enrico rvl_cdip mixed; do
    for v in tags_oi600 tags_in21k caption_blip caption_qwen3b; do
        step run_toponymy.py $ds $v hf
    done
done

# ---- score + render ----------------------------------------------------
for ds in caltech101 stanford_dogs enrico rvl_cdip mixed; do
    step evaluate.py $ds
done
step summarize.py

step visualize.py mixed topo_caption_qwen3b_hf
step visualize.py stanford_dogs topo_tags_in21k_hf
step visualize.py rvl_cdip topo_caption_qwen3b_hf
step visualize.py caltech101 topo_tags_oi600_hf

echo "=== QUEUE DONE — $(date +%H:%M:%S)" | tee -a "$LOG"
