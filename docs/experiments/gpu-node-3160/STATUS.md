# #3160 run status

Written 2026-08-18 08:25 EDT so the run can be picked up after a disconnect.
Delete this file when the report lands.

## In flight on the GRID

| what | job | where | expected |
|---|---|---|---|
| bench arm `fp32_v100_rack7n03` (2048 cells) | array `511229` | `/expscratch/$USER/gpu-node-3160/bench/fp32_v100_rack7n03/results/cells` | ~09:55 EDT |
| bench arm `fp32_v100` (2048 cells) | array `511258` | `.../bench/fp32_v100/results/cells` | ~09:55 EDT |
| analysis, chained `afterany` on both | `511709` | writes `BENCH_STATUS.txt` + `BENCH_TABLES.txt` in the study root | on drain |

The analysis is a **dependency job**, not a login-shell waiter, so it survives a
dropped VPN. Nothing else is queued; the GPU probes are finished.

## Done

- **Premise verified from the pickles** (not assumed): between the two piles,
  `siglip2_l` median 1−cos **1.5e-04** (max 1.1e-02, 0% of rows bit-identical);
  `siglip` median **exactly 0** (max 1.3e-15, 78% of rows bit-identical).
- **Census**: 21 GPU nodes probed → `/expscratch/$USER/gpu-node-3160/census/<node>/`.
  Analyse with `bash launch_census.sh analyze`.
- **Mechanism probe v1**: `/expscratch/$USER/gpu-node-3160/mechanism/<node>/`.
  Two defects found in the probe itself, both fixed in v2 (below).
- **Code**: per-cell provenance + `VTS_GPU_NODE` pinning are committed.

## The finding that changed the question

The v1 mechanism probe recorded a **preprocessing** difference: on `rack5n03`
the `pixel_values` tensor entering the tower is **not** the one `rack7n03` and
the L40S produce — and those two agree with each other exactly. At the same
time the bare GEMM fingerprints are **bit-identical between the two V100s**
(and it is the *L40S* that differs there, by ~6e-07, while its embeddings agree
with `rack7n03` to 2.7e-12).

That inverts #3143's leading hypothesis. If it holds up, the 1.5e-04 is not GEMM
tiling on the GPU at all; it enters before the GPU, and "which V100" is standing
in for something about the host.

**Next step (v2 probe, the decisive test):** ship the reference node's
`pixel_values` to the other nodes and run the forward on *identical* input. Same
pixels + different node → the GPU math genuinely differs. Same pixels + same
output → the whole effect is preprocessing, and the provenance fields to record
are the host's, not the card's.

v1 also lost its per-layer data to a bug: `get_image_features` returns a
`BaseModelOutputWithPooling` on this transformers version, and stamping it threw
inside the per-backend loop, so every backend recorded an error instead of its
layers. Use `extract_tensor` from `vtscore.media.embedder`.
