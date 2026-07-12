# Toponymy image-signpost experiments

Experimental framework for `docs/plans/vtsbrowse-toponymy.md` (Phase 3:
image datasets): can the [Tutte Institute `toponymy`](https://github.com/TutteInstitute/toponymy)
library produce useful browse-map "street signs" for **image** datasets, and
which `object_to_text` (image → a little text) strategy should feed it —
especially when the corpus spans photos, screenshots, and scanned documents,
and when the user browses the **results of a Find** (a fine-grained subset
where a fixed tag vocabulary is either too coarse or full of distractors)?

Runs on the HLTCOE grid against the `vtscore` library tier (no Flask app).
Companion to `scripts/experiments/toponymy_audio/` (same stage layout,
JSON/npy handoff). See the experiment report for findings.

## Pipeline

```
prepare_dataset.py <ds>              # download + SigLIP-embed via vtscore demo loader
make_texts.py <ds> <variant>         # object_to_text: tags_* | caption_* | ocr_florence | blip_plus_ocr
run_toponymy.py <ds> <variant> <namer> [--n-cats N --per-cat M]   # keyphrase | hf; subset flags = Find→Browse
evaluate.py <ds>                     # metrics vs ground-truth categories + table
summarize.py                         # RESULTS/summary.json
visualize.py <ds> <run>              # browse-map mockup PNGs
```

Each stage writes JSON/npy under `$TOPO_RESULTS/<ds>/` and is independently
re-runnable; stages only communicate through those files.

## Grid usage

```bash
# one-time, inside a GPU allocation (node-local scratch):
sbatch --job-name=topo-image --gres=gpu:a100:1 --mem=64G --cpus-per-task=12 \
       --time=12:00:00 --wrap "sleep 43200"          # park an allocation
srun --jobid=<JOBID> --overlap bash setup_node.sh    # venv on /scratch/$USER

# each experiment step:
srun --jobid=<JOBID> --overlap bash -c \
  'PYTHONPATH=/exp/sgreenberg/projects/VTSearch /scratch/jobs/$USER/topo-image/venv/bin/python prepare_dataset.py caltech101'
```

Environment knobs (see `common.py`): `VTS_REPO` (VTSearch checkout),
`TOPO_WORK` (node-local scratch), `TOPO_RESULTS` (durable results dir).

## Datasets used

| dataset | images | ground truth | why |
|---|---|---|---|
| `caltech101` | ~2000 (20/cat) | 101 object categories | generic photo baseline |
| `stanford_dogs` | ~1800 (15/breed) | 120 dog breeds | fine-grained photos; the "dog detector" Find→Browse case |
| `enrico` | 1460 | 20 UI design topics | born-digital screenshots |
| `rvl_cdip` | 1600 (100/class) | 16 document types | scanned/faxed documents |
| `mixed` | 2000 (500/source) | `domain:class`, 2 levels | heterogeneous corpus; coarse layer should find the domains |

## Text variants (the `object_to_text` candidates)

| variant | model / vocab | the hypothesis it tests |
|---|---|---|
| `tags_oi600` | SigLIP zero-shot vs ~600 OpenImages classes | audio study's default, ported |
| `tags_in21k` | SigLIP zero-shot vs ~21k ImageNet-21k lemmas | huge vocab: has breeds/species, or drowns in near-dupes? |
| `tags_oracle` | dataset's own class names | leakage upper bound |
| `caption_blip` | BLIP base (~1 GB) | light generic captioner |
| `caption_florence` / `ocr_florence` | Florence-2-base (~0.5 GB) | task-prompted caption + real OCR |
| `caption_qwen3b` | Qwen2.5-VL-3B-Instruct | instructed VLM: states type + quotes visible text |
| `blip_plus_ocr` | combine (no GPU) | cheap hybrid: photo caption + document text |
