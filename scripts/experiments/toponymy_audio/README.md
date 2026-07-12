# Toponymy audio-signpost experiments

Experimental framework for `docs/plans/vtsbrowse-toponymy.md`: can the
[Tutte Institute `toponymy`](https://github.com/TutteInstitute/toponymy)
library produce useful browse-map "street signs" for **audio** datasets,
and which `object_to_text` (audio → a little text) strategy should feed it?

Runs on the HLTCOE grid against the `vtscore` library tier (no Flask app).
See the experiment report for findings.

## Pipeline

```
prepare_dataset.py <ds>              # download + CLAP-embed via vtscore demo loader
make_texts.py <ds> <variant>         # object_to_text: clap_audioset | clap_esc50vocab | whisper | caption
run_toponymy.py <ds> <variant> <namer>   # namer: keyphrase (no-LLM) | hf (local LLM)
evaluate.py <ds>                     # metrics vs ground-truth categories + table
```

Each stage writes JSON/npy under `$TOPO_RESULTS/<ds>/` and is independently
re-runnable; stages only communicate through those files.

## Grid usage

```bash
# one-time, inside a GPU allocation (node-local scratch):
sbatch --job-name=topo-audio --gres=gpu:a100:1 --mem=64G --cpus-per-task=12 \
       --time=12:00:00 --wrap "sleep 43200"          # park an allocation
srun --jobid=<JOBID> --overlap bash setup_node.sh    # venv on /scratch/$USER

# each experiment step:
srun --jobid=<JOBID> --overlap bash -c \
  'PYTHONPATH=/exp/sgreenberg/projects/VTSearch /scratch/$USER/topo-audio/venv/bin/python prepare_dataset.py esc50'
```

Environment knobs (see `common.py`): `VTS_REPO` (VTSearch checkout),
`TOPO_WORK` (node-local scratch), `TOPO_RESULTS` (durable results dir).

## Datasets used

| dataset | clips | length | ground truth | why |
|---|---|---|---|---|
| `esc50` | 2000 | 5 s | 50 categories in 5 major groups | labeled hierarchy → objective signpost metrics |
| `speech_commands` | ~1400 (40/cat) | 1 s | 35 spoken words | the speech case (Whisper's home turf) |
| `clotho` | 1045 | 15–30 s | none (5 human captions/clip) | uncurated realism check |
