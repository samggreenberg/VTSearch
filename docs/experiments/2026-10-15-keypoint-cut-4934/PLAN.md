# Can a CPU Good verify each page on fewer keypoints? (#4934) — plan

**Pre-registered 2026-10-15, before any tier-`m` run.** The owner chose the margin and the arms.

## Question

After #4933, the CPU ratio test is exact, and what is left of it is the GEMM itself. That is 2·m·n·128 FLOPs per shortlisted page, ~2 s of a ~4.5 s CPU Good at 5k pages. It falls only if each page is verified on fewer of its keypoints (n: median ~5,900, up to 8,192). That changes which pairs pass. What does it cost in ranking AP and in F1 at the line?

## Arms

Each cut applies to the **candidate pages only**: a Good's template keeps every keypoint in its box. A page with no more keypoints than the cut keeps is verified exactly as in the control.

| arm | keeps per page | ratio-test work vs control |
|---|---|---:|
| `all` (control) | every keypoint | 1 |
| `strong2048` | the 2,048 strongest SIFT responses | ~0.35 |
| `strong1024` | the 1,024 strongest | ~0.18 |
| `rows2048` | the first 2,048 stored rows: what a dataset ingested today would get without re-ingesting, since stored order is not by strength | ~0.35 |
| `strongfrac035` | the strongest 35% of each page: the same average work as 2,048 (tier s: a 2,048 cap keeps 1,923 of 5,502 keypoints on average), but sparse pages are cut too | ~0.35 |

The strength order comes from `strength_order.py`. It re-detects each page with the venv's OpenCV (5.0.0) and attaches every response to its cached row. Today's detection reproduces the cached keypoints in the cached row order. On 40 tier-s pages, descriptors differed by at most 1 in an element. Every tier-s page matched ≥ 99.8% of its rows.

## Design

- **Path:** the app's CPU path, with the GPU hidden: Stage 1 on the CPU and a 1,000-page shortlist (`TILED_TOP_K_CPU`).
- **Harness:** `sota_documents.py` through `sota_documents_arms.py`, which loads the tier once and interleaves the arms class by class. 50 clicks, both replicates (the second swaps the click and test halves).
- **Pilot (not decision-bearing):** tier `s`, all 36 classes, every arm, plus `scale2048` (the 2,048 largest SIFT scales) as an exploratory arm. Its job is to show whether tier `m` is worth its ~15 process-hours. The owner decides on tier `m` after the pilot.
- **Decision run:** tier `m` (~50k pages a pool), 36 classes × 2 replicates, the arms the owner keeps after the pilot.

## Readouts

Per session, at 0 / 10 / 25 / 50 clicks: ranking AP on the test half, and the returned set's F1 at the line (`gate_f1`; F-beta at 1/4 and 4 beside it). Also the retrain's wall clock after a Good and after a Bad. A session that runs out of unlabelled click-half pages holds its last state.

Each arm is paired against `all` by (class, replicate). Each class's difference is averaged over its two replicates and bootstrapped over the 36 classes (`vote_curve.paired_ci`, 95%). Script: `keypoint_cut_analyze.py`.

## Ship rule (owner, 2026-10-15)

- An arm **passes** if, at 25 clicks, the 95% lower bound of its paired difference is ≥ −0.01 on **both** AP and F1 at the line.
- The passing arm that verifies the fewest keypoints ships for the CPU path.
- If no arm passes, nothing ships and the CPU path keeps every keypoint.
- `strongfrac035` is judged by the same rule. If both it and `strong2048` pass, the one with the better F1 at 25 clicks ships (they cost the same).

## Known before the runs

The smoke test (tier s, `tobacco800/logo_ajj10e00_1`, 4 clicks) already lost heavily under every cut. `strong1024` was worse than `rows1024`: AP 0.43 against 0.63 at click 4, with the control at 0.93. A page's strongest responses are small-scale (ρ(size, response) ≈ −0.44 over 40 pages), which on a document means text strokes rather than the mark.
