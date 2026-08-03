# Linear (logistic) detector head — finish & validate

**Task shape:** the code is already written and pushed on this branch
(`claude/linear-default-head`, commit `20b015e9`). This is **not** a from-scratch
implementation — **finish and validate** it: get `./run-tests.sh` green and open
the PR. Run this in a real test environment (Claude Code on the web / a container
where `./run-tests.sh` installs deps and runs the full suite incl. the frontend
build) — it cannot be validated in the origin session's local env (no Python/torch).

## Background / why

Threshold-stability #2790: the production **MLP** detector head spikes on sparse
positives — with only ~3–5 labeled positives it's under-determined, so each retrain
wobbles the scores and the decision cut lurches over the unlabeled/test positives
(FNR→1 transient). A **linear (logistic)** head has no such flexibility and removes
the spikes. Validated experimentally:

- COCO / SigLIP2: deep-spike rate `mlp` 0.055 → `linear` 0.025 (~55% cut), lower cost + FNR.
- VG / **both** SigLIP1 & SigLIP2: deep-spike ~0.05 → **0.014 (~73% cut)**, FNR
  0.48→0.32, lower cost — a Pareto win, and `mlp`'s FNR is non-monotone (spike
  signature) while `linear`'s improves monotonically.

Decision: make the linear head the production default. **No UI, no config selector,
no "backend" setting** — a clean swap. The MLP survives only as an internal
`build_model` primitive for the eval harness/tests; it is not reachable from the app.

## What's already implemented (reference — don't redo)

- `vtscore/training/mlp.py`: `LINEAR_HEAD = 0` sentinel; `build_model(hidden_dim=0)`
  builds `nn.Sequential(nn.Linear(input_dim, 1))` (no hidden layer); trained through
  the **unchanged** `train_model` balanced-BCE + label-smoothing loop it *is*
  `LogisticRegression(class_weight="balanced")`. `build_model_from_weights` gained a
  1-layer branch. `train_model`'s default (`hidden_dim=None`) is **still the MLP**.
- `vtscore/detectors/training.py`: `train_and_threshold`, `_train_and_score_xy`,
  `train_detector_from_origins` → `hidden_dim=LINEAR_HEAD`, **uniformly** (the single
  threaded `hidden_dim` sizes both the final model and the cross-calibration fold
  sub-models).
- `vtscore/detectors/labeling_progress.py`: the stability preview → `LINEAR_HEAD`
  (so the preview reflects the real head).
- `vtscore/detectors/portable_bundle.py`: `embedding_dim_from_weights` reads
  `0.weight` shape (layer-count agnostic); `mlp_weights_to_onnx` gained a 1-layer
  linear branch (`sigmoid(Gemm(x, W, b))`, no Relu).
- **Left as MLP on purpose:** `vtscore/training/structural_similarity.py` (the Stage-2
  geometric-verification classifier — a distinct feature, not the detector head).

Since the head stays a torch `nn.Module` emitting logits, the scoring / device /
sigmoid / conformal-calibration paths are untouched.

## Remaining work

1. `./run-tests.sh`; fix **every** failure and diagnostic (ruff, ruff-format,
   codespell, deptry, pyright, pytest all tiers, frontend build). Commit+push before
   each re-run. Read only the final `==== ====` summary block for pass/fail.
2. **Reconcile the tests that assert the old 2-layer MLP geometry on the PRODUCTION
   path** (votes→detector, find, labelset, `train_detector_from_origins`, portable
   export of a *trained* detector, calibration harness asserting
   `hidden_dim == _auto_hidden_dim(...)`, labeling_progress). Update them to expect the
   **linear head**: one Linear layer; state-dict keys `0.weight`/`0.bias`; ONNX graph
   `sigmoid(Gemm)` with no Relu; `input_dim` from `0.weight` shape[1].
   - Tests that **directly** call `build_model(hidden_dim=X>0)` or
     `train_model(..., hidden_dim=X>0)` stay MLP — keep them.
   - Parameterize over head type where a test genuinely needs both.
   - **Do not** "fix" a failure by forcing production back to the MLP.
   - Candidate files (verify against reality; the recon may be stale): `test_sorting.py`,
     `test_mlp_training.py`, `test_torch_config.py`, `test_gpu.py`,
     `test_portable_detector_exporter.py`, `test_portable_bundle.py`, `test_votes.py`,
     `test_conformal_threshold.py`, `test_calibration_harness.py`,
     `test_calibration_inference_geometry.py`, `test_score_sanitization.py`,
     `test_eval_voting_iterations.py`, `test_max_patch_style.py`,
     `test_eval_routes_failures.py`.
3. **Add a fidelity test** (`tests_lib/detectors/`, library tier): on seeded synthetic
   2-class data, the torch linear head (`build_model(d, hidden_dim=0)` trained via
   `train_model`) must rank items in strong agreement with
   `sklearn.linear_model.LogisticRegression(C=1.0, class_weight="balanced")` — assert
   Spearman ≥ 0.95 on the two score vectors (proves the shipped head *is* logistic
   regression). Plus a small structural test: `build_model(d, 0)` has exactly one
   Linear layer and round-trips through `build_model_from_weights` and
   `mlp_weights_to_onnx`.
4. Green → `gh pr create --base dev` (base `dev` mandatory). Title: "Linear (logistic)
   detector head as the production default". Body: the #2790 motivation, that a torch
   `Linear(d,1)` via the existing balanced-BCE loop = LogisticRegression, the
   experimental backing above, the two judgment calls (labeling_progress flipped;
   structural_similarity left MLP), and that the MLP primitive is retained for the eval
   harness/tests. Link `Part of #2790` (non-closing — #2790 is the umbrella experiment).
   Do not close any issue; do not `subscribe_pr_activity`. Delete this plan file in the
   same PR (it's fully shipped once the PR is up).

## Follow-up (separate, not blocking this PR)

- **GRID confirmation:** run the `scripts/sod` sweep with a **torch-linear** arm
  (`build_model(0)`) to confirm the *shipped* head reproduces the spike reduction the
  sklearn-linear experiment showed end-to-end (the unit fidelity test only proves it's
  logistic regression, not the spike behavior in the full loop).
- Parked: anneal linear→MLP keyed on `n_good` (would reintroduce a head switch for the
  high-label regime where the MLP eventually overtakes).
