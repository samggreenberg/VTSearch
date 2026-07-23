# CD-ViTO stand-up (separate env)

Bring up **CD-ViTO** (`github.com/lovelyqian/CDFSOD-benchmark`, ECCV 2024,
arXiv 2402.03094) in its own project dir + conda env, build it, and validate it runs
— the same way DE-ViT was brought up (`scripts/vg/`). **No VTSearch/sweep integration
yet**; this is just groundwork so we can decide from a working baseline.

## Why CD-ViTO
It's DE-ViT plus **learnable instance features** + an **instance-reweighting MLP**,
fine-tuned per support set — so unlike DE-ViT (frozen mean prototype, flat annotation
curve) more/better annotations actually help (monotonic 1/5/10-shot gains). Same frozen
DINOv2 backbone. Eventual goal: judge whether this learnable-prototype architecture is
worth porting to **VTSearch patch mode**.

## Environment (GPU-only build)
CD-ViTO inherits DE-ViT's stack: **torch 1.13.1 / cu117 + a vendored detectron2** whose
CUDA ops are compiled at install. cu117 → **V100 / A100 only** (Ampere sm_86), **NOT
L40S / H100 / H200**.

Build on a compute node (login nodes can't compile CUDA ops). From a login node:

```bash
srun --partition=gpu --gres=gpu:v100:1 --cpus-per-task=8 --mem=32G --time=3:00:00 --pty bash -l
bash /exp/mlucio/projects/VTSearch/scripts/cdfsod/cdvito_env_setup.sh
```

The script clones to `/exp/$USER/projects/cdfsod`, creates conda env `cdfsod` (py3.9),
installs `cudatoolkit-dev=11.7` + torch cu117, then `pip install -e` the repo
(`FORCE_CUDA=1 --no-build-isolation`). If the repo doesn't vendor detectron2, it falls
back to building DE-ViT's from `/exp/$USER/projects/devit`. Success prints:
`torch 1.13.1+cu117  cuda True  detectron2 <ver>  _C OK`.

Build gotchas (carried over from DE-ViT, see the `devit-integration-status` notes):
don't `module load` a system 12.x/13.x cuda; the env's `cudatoolkit-dev=11.7` +
`CUDA_HOME=$CONDA_PREFIX` is what nvcc must use. `--no-build-isolation` is required
because detectron2's `setup.py` imports torch.

## Weights (what CD-ViTO actually loads)
`main_results.sh` runs, per (dataset, shot):
`MODEL.WEIGHTS weights/trained/few-shot/vitl_0089999.pth` +
`DE.CLASS_PROTOTYPES prototypes_init/<dataset>_<shot>shot.vitl14...pkl` +
`DE.BG_PROTOTYPES weights/initial/background/background_prototypes.vitl14.pth` +
an offline R-50 RPN (`detectron2://ImageNetPretrained/MSRA/R-50.pkl`, auto-downloaded).

So of the DE-ViT weights you already have at `/exp/mlucio/projects/devit/weights/`:
- **`background_prototypes.vitl14.pth` — reusable directly** (symlink it in).
- **`trained/open-vocabulary/lvis/vitl_0069999.pth` — NOT usable.** That's the
  *open-vocabulary* model; CD-ViTO wants the *few-shot* **`vitl_0089999.pth`**, which
  you must **download** (DE-ViT Box folder `rutgers.box.com/s/2lco6ab66pn3ufq6rh4gmyfzg9vfkm23`,
  path `release/weights/trained/few-shot/vitl_0089999.pth`, 1.2G — see
  `github.com/mlzxy/devit/blob/main/Downloads.md`, rclone per devit issue #7).
- The **class prototypes ship precomputed** in `prototypes_init/` (ArTaxOr/DIOR/…​ ×
  {1,5,10}-shot × {s,b,l}), so no prototype build is needed for the benchmark.

Reuse the background prototypes:
```bash
cd /exp/mlucio/projects/cdfsod
mkdir -p weights/initial/background weights/trained/few-shot
ln -s /exp/mlucio/projects/devit/weights/initial/background/background_prototypes.vitl14.pth \
      weights/initial/background/background_prototypes.vitl14.pth
# then drop the downloaded few-shot model at weights/trained/few-shot/vitl_0089999.pth
```

## Validate (smoke test)
Download a benchmark dataset (links in the repo README) into `datasets/`. Sizes aren't
published; Clipart1k (1,000 cartoon images) is likely the lightest download, but confirm
with `du -sh datasets/*/` after pulling rather than trusting a guess. Registration
(`detectron2/data/datasets/builtin.py::register_all_CD`) expects this COCO layout, and
the dataset names in the config map to it:
```
datasets/ArTaxOr/
  train/                      # ArTaxOr_5shot + ArTaxOr_train images
  test/                       # ArTaxOr_test images
  annotations/{train.json,test.json,1_shot.json,5_shot.json,10_shot.json}
```
(`ArTaxOr_5shot` → `train/` + `annotations/5_shot.json`; `ArTaxOr_test` → `test/` +
`annotations/test.json`.) The repo's dataset downloads come pre-converted to this layout.

**Gotcha:** `register_all_CD` runs at import and `open()`s the json for **every** name in
`builtin.py`'s `datasets_name = ('ArTaxOr', 'clipart1k')` with no error handling, so
`train_net.py` **crashes on any missing json**. Either stage **both** ArTaxOr and
clipart1k, or trim that line to `datasets_name = ('ArTaxOr',)` for an ArTaxOr-only smoke.

`main_results.sh` hardcodes **4 GPUs** (`--num-gpus 4 CUDA_VISIBLE_DEVICES=2,5,6,7`) and
loops all datasets × shots — for a single-V100 smoke, run one cell directly, and drop the
batch (config default `IMS_PER_BATCH: 16` OOMs ViT-L on one 16 GB card):

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --num-gpus 1 \
  --config-file configs/artaxor/vitl_shot5_artaxor_finetune.yaml \
  MODEL.WEIGHTS weights/trained/few-shot/vitl_0089999.pth \
  DE.OFFLINE_RPN_CONFIG configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
  SOLVER.IMS_PER_BATCH 4 \
  OUTPUT_DIR output/vitl/artaxor_5shot/
```

Expect it to fine-tune (MAX_ITER 100) on the 5-shot support and print a non-degenerate
AP — the CD-ViTO analog of DE-ViT's YCB-demo check. If ViT-L still OOMs, use a
`vitb`/`vits` config (prototypes for those ship too).

## Run the published detector on our SOD classes (`sodcoco`)
Goal: measure CD-ViTO's resolution-floor (per-size AP) + annotation-scaling (1/5/10-shot)
on our own classes before deciding whether to port the architecture into VTSearch.

**1. Export (main venv, login node — no GPU).** `export_sod_cocofsod.py` turns
`SodDataset` classes into CD-ViTO's COCO layout (`datasets/sodcoco/{train,test}/` +
`annotations/{train,test,1_shot,5_shot,10_shot}.json`). Small + large contrast set:
```bash
.venv/bin/python scripts/cdfsod/export_sod_cocofsod.py --dataset coco --name sodcoco \
  --classes "traffic light,stop sign,car,person,bus" \
  --max-pos-per-class 200 --k-values 1,5,10 \
  --out-root /exp/mlucio/projects/cdfsod/datasets
```
(`--min-box-frac 0` keeps all boxes so small objects are fully exposed; the sweep uses
0.01. Categories are 1-indexed in the given order. Already run → dataset present.)

**2. Register — TWO files, both required:**
- `detectron2/data/datasets/builtin.py`: append the name to `datasets_name`
  (registers the *images*). `register_all_CD` opens each name's jsons at import, so only
  list names whose files exist.
- `lib/categories.py`: append the name to its own `datasets_name` tuple **and** add a
  `'<name>_classes': [...]` entry to `CLASS_NAME`, in **category_id 1..N order** matching
  the export (registers the *class names* into `SEEN_CLS_DICT`/`ALL_CLS_DICT`). Skipping
  this file gives `KeyError: '<name>_Kshot'` from `extract_instance_prototypes.py`
  (`ALL_CLS_DICT[...]`) and `train_net.py` (`SEEN_CLS_DICT[...]`).
  For `sodcoco`: `'sodcoco_classes': ['traffic light','stop sign','car','person','bus']`.

**3. Build prototypes (GPU, cdfsod env).** Mirrors `build_prototypes.sh`:
```bash
for shot in 1 5 10; do
  python3 ./tools/extract_instance_prototypes.py --dataset sodcoco_${shot}shot \
      --out_dir prototypes_init --model vitl14 --epochs 1 --use_bbox yes --without_mask True
  python3 ./tools/run_sinkhorn_cluster.py --inp prototypes_init/sodcoco_${shot}shot.vitl14.bbox.pkl \
      --epochs 30 --momentum 0.002 --num_prototypes ${shot}
done
```
→ `prototypes_init/sodcoco_${shot}shot.vitl14.bbox.p${shot}.sk.pkl` (what the configs point at).
`extract_instance_prototypes.py` pulls DINOv2 via `torch.hub` — if compute nodes lack
internet, pre-cache on a login node: `python -c "import torch; torch.hub.load('facebookresearch/dinov2','dinov2_vitl14')"`.

**4. Configs** — already generated: `configs/sodcoco/vitl_shot{1,5,10}_sodcoco_finetune.yaml`
(clones of the clipart ViT-L config; `TOPK 5` = our 5 classes).

**5. Fine-tune + eval per K (GPU).** Read the per-category + per-size AP table:
```bash
for shot in 1 5 10; do
  CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --num-gpus 1 \
    --config-file configs/sodcoco/vitl_shot${shot}_sodcoco_finetune.yaml \
    MODEL.WEIGHTS weights/trained/few-shot/vitl_0089999.pth \
    DE.OFFLINE_RPN_CONFIG configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
    SOLVER.IMS_PER_BATCH 4 \
    OUTPUT_DIR output/vitl/sodcoco_${shot}shot/
done
```
The signals: **APs vs APl** (resolution floor on our small classes) and **AP vs shot**
(does more annotation help). Per-category AP separates the small (traffic light, stop
sign) from large (car, person, bus). Use an A100 to run at the config's `IMS_PER_BATCH 16`.

## Next step (undecided — decide after it's validated)
Two integration paths for the eventual VTSearch-patch-mode question:
- **Port the idea as an in-loop head** — a learnable-prototype + instance-reweighting
  scoring head in the main env, slotted into `evaluate_realistic_curve` over the
  patch/HAC region sources (realistic good→bad→hard→new order, x = total annotations,
  uses negatives). Directly answers the patch-mode question; not the published pipeline.
- **Run the published detector** — this env, its own RPN+ROIAlign, positives-only,
  controlled K-shot curve. Faithful to the paper; doesn't touch patch mode.

Also carry forward the known **resolution-floor** caveat: the DINOv2-ROI family
(DE-ViT/CD-ViTO) is context-contaminated on sub-patch objects, so expect it to struggle
on the smallest classes regardless — worth measuring against the crop/zoom methods.
