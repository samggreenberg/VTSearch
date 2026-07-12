#!/bin/bash
# Serialized experiment queue — the grid a100 nodes run in Exclusive_Process
# compute mode, so CUDA stages must not overlap. Each step logs and continues
# on failure (a variant that crashes the pipeline is itself a result).
set -uo pipefail
RUN=/exp/sgreenberg/experiments/toponymy-audio/run.sh
LOG=/exp/sgreenberg/experiments/toponymy-audio/queue_all.log

step() {
    echo "=== STEP: $* — $(date +%H:%M:%S)" | tee -a "$LOG"
    "$RUN" "$@" >>"$LOG" 2>&1 || echo "=== STEP FAILED: $*" | tee -a "$LOG"
}

step make_texts.py esc50 caption
step make_texts.py esc50 whisper
step run_toponymy.py esc50 caption keyphrase
step run_toponymy.py esc50 caption hf
step run_toponymy.py esc50 clap_esc50vocab keyphrase
step run_toponymy.py esc50 clap_esc50vocab hf
step run_toponymy.py esc50 whisper keyphrase

step make_texts.py speech_commands whisper
step make_texts.py speech_commands clap_audioset
step run_toponymy.py speech_commands whisper keyphrase
step run_toponymy.py speech_commands whisper hf
step run_toponymy.py speech_commands clap_audioset keyphrase
step run_toponymy.py speech_commands clap_audioset hf

step prepare_dataset.py clotho
step make_texts.py clotho clap_audioset
step make_texts.py clotho caption
step run_toponymy.py clotho clap_audioset hf --corpus-desc "'a collection of real-world Freesound recordings (Clotho)'"
step run_toponymy.py clotho caption hf --corpus-desc "'a collection of real-world Freesound recordings (Clotho)'"
step run_toponymy.py clotho caption keyphrase --corpus-desc "'a collection of real-world Freesound recordings (Clotho)'"

step evaluate.py esc50
step evaluate.py speech_commands

step visualize.py esc50 topo_clap_audioset_hf
step visualize.py esc50 topo_caption_hf
step visualize.py speech_commands topo_whisper_hf
step visualize.py clotho topo_caption_hf

echo "=== QUEUE DONE — $(date +%H:%M:%S)" | tee -a "$LOG"
