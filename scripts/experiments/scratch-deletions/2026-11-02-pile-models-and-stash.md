# 2026-11-02: unused model weights and a stash cell from the pile (8.2 GB by `du`, owner-approved)

All under `/expscratch/sgreenberg/vts-cache` (the pile, 98 -> 90 GB). Every model below is a Hugging Face
cache entry, so it downloads again the first time anything asks for it. No job was running against any of them.

- **`models/models--timm--ViT-SO400M-14-SigLIP-384`** (3.3 GB) and **`models/models--google--siglip2-base-patch16-224`**
  (1.5 GB): the weights behind `siglip_l` and `siglip2`, the "middle" embedders dropped from the pile on 2026-08-12.
  `pile_config.EMBEDDERS` no longer lists either. The app still has both embedders (`embedder_siglip_l.py`,
  `embedder_siglip2.py`). **Rebuild:** `scripts/experiments/pile/prefetch_models.py`, or let the embedder download them.
- **`models/hub/models--IDEA-Research--grounding-dino-tiny`**, **`…--mm_grounding_dino_large_o365v2_oiv6_goldg`**,
  **`…--mm_grounding_dino_tiny_o365v1_goldg_v3det`** (2.6 GB): #5395's GroundingDINO zero-shot study, which ships
  nothing. The only code on `dev` that names them is the study's launcher,
  `scripts/experiments/state_of_app/gdino_5395.sh`. **Rebuild:** that launcher downloads them on first use.
- **`datadir/models/models--google--siglip-base-patch16-224`** (782 MB): a second copy of the shipped SigLIP. It was
  downloaded into the datadir by a run that set `VTSEARCH_DATA_DIR` without `VTSEARCH_MODELS_DIR`. The pile's own
  copy in `models/` stays.
- **`stash/vg_scale_cocoanchored__siglip.pkl`** (22 MB): a 2026-08-17 side cell of the retired `vg_scale`, read by
  nothing (`git grep cocoanchored` is empty). Not rebuildable: Visual Genome was retired in #4038.
- **`logs/`**: 43 build logs of the retired Visual Genome datasets (`pile-vg*`, `pile-visual_genome_m-*`,
  `vgscan-*`, `verify-570298.out`), a few KB each.

**Kept on purpose:** `lvis/lvis_v1_train.json` (1.1 GB). It is the only copy on the cluster (`/exp/scale26` has
only the val split), and `boundary_contest.py` reads it by default through `pile_config.LVIS_DIR`. Also kept:
`stash/corrections.pre-3926.json` (168 KB), a backup of a human-record file. Offered and not chosen: the
`coco_better_full_s0`…`_s7` × `dinov3_patch` shards (36 GB) and the FullMarks `sift_vlad` cells (8.8 GB).
