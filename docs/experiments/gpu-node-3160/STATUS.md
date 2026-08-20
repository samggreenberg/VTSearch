# #3160 — resume note

Written 2026-08-19, after §5.1 landed. **The study is complete**; this file
tracks only what is owed outside the report. Delete it when both PRs merge.

## State

- **PR #3178** (`claude/gpu-provenance-3160`) — study + pile-side code. Suite
  green post-merge with the 2026-08-18 release (job `521271`, 8728 passed).
  `REPORT.md` is complete through §5.1.
- **PR #3182** (`claude/embedding-stack-pin`) — `embedding_stack` in dataset
  `meta.json`. Suite green (job `521373`, 8736 passed).
- Issue #3160 commented and labelled `solved`.

## Approved and queued: rebuild the two AVX-512 bands

Owner approved 2026-08-19 ("rebuild, just for future simplicity"). **The target
is not the band the confound was found in.** `vg_box_small` was built on
`rack8n06`, an AVX2-only host, and the shipped pin *is* `avx2` — rebuilding it
reproduces what it already has. It is `vg_box_medium` and `vg_box_large`
(built on `rack7n03`, AVX-512) that differ from the go-forward standard.

**Check this first.** Job `523950` was rebuilding `vg_box_small × siglip2_l`
into a side pile under the pin, to test whether a pinned rebuild reproduces the
2026-08-12 cell bit-for-bit:

```bash
ssh grid 'sacct -X -j 523950 --format=State,Elapsed -n
  tail -5 /expscratch/$USER/gpu-node-3160/logs/rebuild-verify-523950.out
  python3 -c "import json;print(json.load(open(\"/expscratch/$USER/gpu-node-3160/rebuild/datadir/embeddings/vg_box_small__siglip2_l.provenance.json\"))[\"fingerprint\"][\"vectors_sha256\"])"'
# live cell for comparison: 728f14f8beeb...
```

- **Hashes match** → the pin reproduces builds exactly; proceed with the two
  bands below.
- **Hashes differ** → something *else* moved since 2026-08-12 (transformers has
  gone to 5.12.1 since; the original build recorded no version). Rebuilding
  medium/large would then import that drift as well, which is a different
  decision than the one approved. Stop and report the delta instead.

Then, to rebuild the two bands into the shared pile (~10 min each on an L40S,
12,000 images per cell):

```bash
ssh grid 'W=/exp/$USER/projects/vts-gpu-3160; P=/expscratch/$USER/vts-cache
  cd $W && git pull --ff-only
  for ds in vg_box_medium vg_box_large; do
    sbatch --job-name=rebuild-$ds --partition=gpu --gres=gpu:l40s:1 \
      --cpus-per-task=8 --mem=24G --time=2:00:00 \
      --output=/expscratch/$USER/gpu-node-3160/logs/rebuild-$ds-%j.out \
      --wrap "bash -lc \"module load python/3.12.3 && source /exp/$USER/projects/VTSearch/.venv/bin/activate \
        && export VTS_REPO=$W VTS_PILE=$P VTSEARCH_DATA_DIR=$P/datadir \
           VTSEARCH_MODELS_DIR=$P/models HF_HOME=$P/models \
           ATEN_CPU_CAPABILITY=avx2 VTSEARCH_TORCH_THREADS=8 OMP_NUM_THREADS=8 \
        && cd $W/scripts/experiments/pile && python build_pile.py --datasets $ds --embedders siglip2_l --force\""
  done'
```

`--force` is required (the cells exist). Afterwards:

```bash
ssh grid 'cd /exp/$USER/projects/vts-gpu-3160/scripts/experiments/pile
  source pile_env.sh && source ../../../gridenv.sh
  python build_pile.py --verify && python build_pile.py --provenance'
```

`--provenance` should then show all three bands on one dispatch (`AVX2`) with
real hostnames rather than `rack7n03?`, and the mixed-environment warning should
name one environment instead of several.

**Two things to say out loud when it lands**, because they are the cost of the
change rather than side effects:

- The old cells' `vectors_sha256` (`a1d274acf47a…` medium, `b1bf5d6dc8b0…` large)
  are recorded in their current sidecars. After the rebuild those hashes are
  history — which is exactly what the fingerprint was added for: the change is
  *detectable*, not silent. Anyone re-running #3129/#3156 against the new cells
  is reading different vectors than the published run did.
- Only `siglip2_l` is rebuilt. The `siglip` and `dinov3_patch` cells of all three
  bands are 224px, measured bit-identical across the ISA split, and are left
  alone deliberately — rebuilding them would churn 3.5 GB apiece to change
  nothing.

## Housekeeping owed

- Five worktrees under `/exp/$USER/projects/` from this work: `vts-gpu-3160`,
  `vts-3160-tests`, `vts-3160-merge`, `vts-stackpin`, `vts-stackpin-tests`.
  Remove the `*-tests`/`*-merge` ones once both PRs land — `/exp` is a 50 G quota.
  Note `vts-gpu-3160` is several commits behind its branch: it was deliberately
  not pulled while arrays were reading code from it.
- Artifacts on scratch: `/expscratch/$USER/gpu-node-3160/{census,mechanism,
  cpuinfo,backend,bench,bench-rep,figures}`. Purgeable; every number that
  matters is in `REPORT.md`, but the census and mechanism JSONs are the only
  copies of measurements that cost GPU time.

## One open question, not owed by this issue

Whether transformers 4.x `use_fast=True` produces bit-identical pixels to 5.x
`backend="torchvision"`. If yes, naming the backend (#3173/#3176) closes the
version axis completely; if no, `>=4.49` still admits two answers even with the
backend named. Needs a second venv on 4.x, which was never built. Recorded on
#3173 by the #3146 session, as a question rather than a proposal.
