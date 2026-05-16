# Pyright Type-Checking

Adopt **pyright in basic mode** as a hard CI gate across the `vtsearch/`
package. Rolled out in stages so each PR stays reviewable.

## Status

- **Stage 0 + Stage 1:** ✅ shipped in PR #1349. The gated CI job covers
  `auth/`, `cli.py`, `concurrency/`, `config.py`, `exporters/`, `labels/`,
  `plugins/`, `settings_io/`, `sync/`, `utils/`. The advisory job runs over
  the full `vtsearch/` package on every PR and prints the residual error
  count to the GitHub step summary.
- **Stage 2:** ✅ shipped. Adds `settings.py`, `settings_factory.py`,
  `state/`, `security/` to the gated scope (31 real errors fixed).
- **Stage 3:** ✅ shipped. Adds `datasets/`, `detectors/`, `eval/`,
  `embedding/`, `training/` to the gated scope (38 real errors fixed).
- **Stage 4:** ✅ shipped. Adds `routes/`, `converters/` to the gated
  scope (40 real errors fixed).
- **Stage 5:** ✅ shipped. Adds `media/` to the gated scope (97 real
  errors fixed). Also clears the stale `vtsearch/settings_factory.py`
  entry from `include` (the module was merged into
  `vtsearch/settings_models.py` during an unrelated reorg).
- **Stage 6:** ⏳ next — the whole `vtsearch/` package. Post-Stage-5
  advisory is **0 errors**, so Stage 6 is the cleanup PR: switch
  `include` to a single `"vtsearch"` entry (or add the remaining
  top-level files: `__init__.py`, `achievements.py`, `cli_pipeline.py`,
  `cli_progress.py`, `logging_config.py`, `openapi.py`,
  `settings_models.py`, and `vtsearch/schemas/`) and delete the
  advisory job from `.github/workflows/pyright.yml`.
- **Stage 7 (`tests/`):** 📋 optional, deferred.

## Goal

- `pyrightconfig.json` at repo root with `typeCheckingMode: "basic"`.
- A CI job (`.github/workflows/pyright.yml`) that fails the build on any
  pyright error in the **gated** scope.
- An **advisory** CI step that runs pyright over the entire `vtsearch/`
  package on every PR (does not fail the build) so we always know the
  residual error count for the next stage.
- Each stage expands the gated scope by editing `include` in
  `pyrightconfig.json`. Once stage 6 lands, the gated scope == the
  advisory scope and the advisory job goes away.

## Why pyright, why basic mode

- **Pyright over mypy**: faster, better inference, fewer plugins
  required to handle Flask/numpy/torch idioms.
- **Basic mode over strict**: basic flags real bugs (None-attribute
  access, wrong arg counts, undefined names, override mismatches)
  without forcing exhaustive annotations on a 49k-line codebase.
  Strict can come later if we want it.

## Baseline (2026-05-15)

`pyright vtsearch/` from a clean working tree reports **482 errors** and
1 warning, broken down by directory:

| Dir | Total | Non-import |
|---|---:|---:|
| `media/` | 211 | 68 |
| `routes/` | 96 | 57 |
| `security/` | 20 | 20 |
| `models/` | 33 | 9 |
| `eval/` | 29 | 1 |
| `datasets/` | 26 | 13 |
| `detectors/` | 18 | 7 |
| `converters/` | 15 | 1 |
| `state/` | 13 | 9 |
| `utils/` | 6 | 0 |
| `settings.py` | 5 | 5 |
| `auth/` | 3 | 0 |
| `labels/` | 2 | 2 |
| other | ~5 | ~2 |

The 288 `reportMissingImports` errors come from heavy third-party
packages (`torch`, `transformers`, ...) that aren't installed in a bare
shell. CI installs them via `requirements/base.txt`, so import errors
will resolve there. **Local dev needs `bash scripts/install-cpu.sh`
for clean reports.**

### Post-Stage-2 advisory count

After Stage 2 shipped, `pyright vtsearch/` with deps installed reports
**174 errors** across the remaining out-of-gate scopes:

| Dir | Errors |
|---|---:|
| `media/` | 96 |
| `routes/` | 37 |
| `datasets/` | 19 |
| `detectors/` | 12 |
| `converters/` | 3 |
| `embedding/` | 3 |
| `eval/` | 2 |
| `training/` | 2 |

(`models/` no longer exists — its contents were redistributed into
`detectors/`, `training/`, and `embedding/` during a separate reorg.)

### Post-Stage-3 advisory count

After Stage 3 shipped, `pyright vtsearch/` with deps installed reports
**137 errors** across the remaining out-of-gate scopes:

| Dir | Errors |
|---|---:|
| `media/` | 97 |
| `routes/` | 37 |
| `converters/` | 3 |

(`media/` ticked up by one against the Stage-2 snapshot because of an
unrelated refactor between the two stages — not new regression from
Stage 3's fixes.)

### Post-Stage-4 advisory count

After Stage 4 shipped, `pyright vtsearch/` with deps installed reports
**97 errors**, all in `media/`. The advisory job now exactly matches the
remaining work for Stage 5.

| Dir | Errors |
|---|---:|
| `media/` | 97 |

### Post-Stage-5 advisory count

After Stage 5 shipped, `pyright vtsearch/` with deps installed reports
**0 errors**. The advisory job is now a no-op — Stage 6 deletes it.

## Stages

Each stage is a separate PR. The "errors to fix" column counts real
(non-import) errors at the start of the stage.

| # | Scope added to `include` | Errors to fix | Status |
|---|---|---:|---|
| 0 | (no scope; config + advisory CI only) | 0 | ✅ shipped (PR #1349) |
| 1 | `utils/`, `auth/`, `plugins/`, `sync/`, `concurrency/`, `exporters/`, `labels/`, `settings_io/`, `cli.py`, `config.py` | 4 | ✅ shipped (PR #1349) |
| 2 | `settings.py`, `settings_factory.py`, `state/`, `security/` | 31 | ✅ shipped |
| 3 | `datasets/`, `detectors/`, `eval/`, `embedding/`, `training/` | 38 | ✅ shipped |
| 4 | `routes/`, `converters/` | 40 | ✅ shipped |
| 5 | `media/` (heaviest — turned out to be the HF-stub gap pattern, not stubs) | 97 | ✅ shipped |
| **6** | Whole `vtsearch/` (incl. `achievements.py`, `logging_config.py`, `openapi.py`, `schemas/`) — advisory job removed | 0 | ⏳ next |
| 7 *(optional)* | `tests/` | TBD | 📋 |

**Stage 0 + Stage 1 landed together in PR #1349.** Subsequent stages are
separate PRs; each one bumps `include` in `pyrightconfig.json` and fixes
the real errors that the gate then surfaces.

### Stage 1 fixes (shipped)

Documenting these because the patterns are likely to recur in later
stages:

1. **Forward references that pyright can't resolve at runtime.**
   `vtsearch/labels/sources/base.py` referenced `LabelSet` only in type
   annotations; the runtime import had been removed. Fix: add a
   `TYPE_CHECKING`-guarded import so the symbol resolves for the
   type-checker without re-introducing the runtime dependency.
2. **Override signature mismatches.** `SettingsSource.save()` and
   `LabelsetSource.save()` rename the data argument to `settings` /
   `labelset` for readability. Pyright treats that as a parameter-name
   override violation against `SyncSource.save(self, data)`. Fix: make
   the base-class parameter positional-only (`def save(self, data, /)`)
   so the override contract is on type and position, not name.
3. **`__file__` is `Optional[str]`.** Namespace packages have
   `__file__ is None`, but `Path(...)` requires `str`. The
   `PluginRegistry._discover` walker hit this. Fix: explicit early-return
   when `__file__ is None` instead of letting `Path` raise on `None`.
4. **`importlib.metadata.entry_points(group=...)` requires `str`, not
   `str | None`.** Same shape as #3 — narrow the optional via an early
   return at the call site (e27bd8e). Worth checking other `entry_points`
   callers as later stages widen scope.

### Stage 2 fixes (shipped)

Patterns that recurred enough to document:

1. **Dynamically generated module accessors.** `vtsearch/settings.py`
   creates `get_<key>` / `set_<key>` at import time via
   `globals()[f"get_{key}"] = ...`. Pyright can't see those, so
   `settings.get_inclusion()` from `state/__init__.py` errored with
   *"is not a known attribute of module"*. Fix: declare each accessor's
   signature in an `if TYPE_CHECKING:` block at the top of the module.
   That documents the public API for free and stays correct as long as
   the spec table and the block stay in sync.
2. **Private CPython implementation details (`pickle._Unpickler`).**
   `_PeekUnpickler` extends the private pure-Python unpickler and pokes
   at its internals (`read`, `append`, `stack`, `pop_mark`, `dispatch`).
   None of those are in typeshed. Fix: declare the touched attributes
   inside an `if TYPE_CHECKING:` class block, and explicitly type
   `dispatch: dict[int, Any]` so the Self-typed dispatch methods stop
   failing the contravariance check against `Callable[[Unpickler], None]`.
3. **`socket.getaddrinfo` sockaddr field is `str | int` at index 0.**
   IPv4/IPv6 union means typeshed widens `sockaddr[0]` to `str | int`,
   even though it's always `str` at runtime. Wrap with `str(...)` at
   the call site rather than asserting.
4. **Multi-step nullability narrowing doesn't survive intermediate
   booleans.** `has_origin_key = origin is not None and bool(...)` then
   `if has_origin_key:` does *not* narrow `origin` to `dict`. Inline
   the condition (`if origin is not None and origin_name:`) so the
   narrow applies to the type-checker on the actual access.
5. **`sklearn.cluster.KMeans` has mistyped kwargs in stubs.** `n_init=1`
   is valid at runtime; the stub claims it must be `str`. Use
   `# pyright: ignore[reportArgumentType]` rather than `int -> str`
   gymnastics. `KMeans.inertia_` is typed `Optional[float]`; assign to
   a local and narrow before comparing.

### Stage 3 fixes (shipped)

Patterns that recurred enough to document:

1. **Methods only on a subclass, not the ABC.** `embed_pil_image` lives
   on every image-embedder subclass but not on `MediaEmbedder`; same
   for `_get_model_and_processor` / `_get_model` on the audio/video/text
   subclasses. Callers know their embedder is the right kind by
   construction (`embedders_for_type("image")[0]`, `get_embedder("clap")`).
   Fix at the call site: `cast(Any, emb).embed_pil_image(...)`. Lifting
   the method onto the ABC isn't right either — it isn't part of the
   contract for non-image embedders.
2. **`Optional` ABC params losing all attribute info.** A
   `det_ctx: object` parameter blocks `det_ctx.model = ...` assignments
   because `object` has no `model`. Use a `TYPE_CHECKING`-guarded
   import of the real class (`DetectorContext`) and annotate with the
   forward reference; avoids a runtime import cycle while restoring
   attribute access for the checker.
3. **`elem.origin or {}` does not narrow `elem.origin` for downstream
   code.** Pyright narrows the expression itself, not the original
   attribute. Bind to a local first
   (`origin = elem.origin or {}; ... call(origin)`) so the narrow
   applies to the value flowing into the call.
4. **`ProgressCallback` is `Callable[[str, str, int, int], None]` — pass
   all four args.** Several loader call sites passed only two
   (`on_progress("idle", "Done")`), which pyright caught as missing
   positional arguments. Pass `0, 0` for the count/total of a no-op
   completion message.
5. **sklearn return-type unions driven by overloads.** `fetch_20newsgroups`
   has a `return_X_y` overload that widens the return type to
   `Bunch | tuple`, so `.data` / `.target` / `.target_names` access
   fails. With `return_X_y` defaulted to `False` (the only mode we use)
   the runtime value is always a Bunch. `cast(Any, fetch_20newsgroups(...))`
   is the pragmatic fix.
6. **sklearn `GaussianMixture.means_` is `Optional[ndarray]`.** It's
   always set after `.fit()`. `assert gmm.means_ is not None` before
   use; pyright narrows from there.
7. **Pillow `Image.frombytes(mode, size, data)` expects
   `tuple[int, int]`, not `list[int]`.** `[pix.width, pix.height]` →
   `(pix.width, pix.height)`.
8. **Pillow `Image.crop(box)` expects
   `tuple[float, float, float, float]`.** `tuple(some_list)` is
   `tuple[int, ...]` with indeterminate length — too loose. Build the
   tuple from indexed elements (`(parts[0], parts[1], parts[2], parts[3])`)
   after a length check.
9. **Optional dict args called with `None` from a sibling helper.**
   `_load_embedder_with_progress(media_dict: dict, ...)` was called by
   `_load_embedder_for_clips()` as `(None, ...)`. The body already
   handles `None`; widen the param type to `dict | None`.
10. **Pandas `Series` row scalars confuse builtin coercions.**
    `int(row["n_labels"])` errors because pandas stubs widen the
    cell type to `ndarray | Series | Any`. The value is a scalar at
    runtime; a localized `# pyright: ignore[reportArgumentType]` is
    the right escape hatch rather than wrapping every cell access in
    `cast`.

### Stage 4 fixes (shipped)

Patterns that recurred enough to document:

1. **Discriminated tuple unions don't narrow after unpacking.** Helpers
   like `get_plugin_or_404` and `_extract_importer_fields` return
   `(value, None)` on success and `(None, error)` on failure. Typing the
   return as `tuple[T, None] | tuple[None, E]` doesn't help: pyright's
   `is not None` check on the unpacked `err` variable does NOT narrow
   the sibling `value` variable. Confirmed with a minimal repro.
   Fix at every call site: add `assert value is not None  # narrowed by
   err check` right after the early-return. Explicit, pythonic, and
   safe at runtime (catches helper-contract drift). Stage 4 added ~13
   such asserts across `routes/datasets`, `routes/detectors`,
   `routes/labels`, and `routes/settings`.
2. **`hasattr` does not narrow attribute access for non-Protocol
   classes.** Two sites used
   `if hasattr(mt, "image_response"): mt.image_response(...)` to call
   an optional method on a `MediaType` ABC; pyright still flagged the
   call as accessing an unknown attribute. Fix: switch to
   `fn = getattr(mt, "image_response", None); if fn is not None:
   fn(...)`. The local binding has type `Any`, no narrowing needed, and
   the runtime check is unchanged.
3. **`dict[str, callable]` is a typo — use
   `Callable[..., Any]`.** `callable` is a builtin function (the
   `isinstance`-style predicate), not a type. Pyright's diagnostic is
   *"Expected class but received '(obj: object, /) -> TypeIs[(...) ->
   object]'"* — cryptic until you spot the lowercase `c`. Fix:
   `from typing import Callable; dict[str, Callable[[Any], Any]]`.
4. **Trained-model parameter type returned as `object | None` blocks
   `.parameters()` / call.** `_resolve_or_train_detector(...) ->
   tuple[object | None, float, dict | None]` made callers fail with
   *"Cannot access attribute 'parameters' for class 'object'"* and
   *"Object of type 'object' is not callable"*. The runtime value is a
   `torch.nn.Module`. Annotate as `Any | None` (matches
   `DetectorContext.model: Any` in `state/core.py`) — pyright stops
   complaining and callers can do `next(mlp.parameters())` and `mlp(x)`
   freely.
5. **PyMuPDF `Page.get_text(option)` widens to `str | list | dict` in
   the stub even when the option is the string literal `"text"`** (no
   overload picks up the literal). The runtime value is always `str`
   in text mode. Fix: `cast(str, page.get_text("text")).strip()`.
6. **Soft-dependency imports inside `try/except ImportError` still
   trip `reportMissingImports`** when the package isn't pinned in
   `requirements/base.txt`. Stage 4 saw this with `paddleocr`. Fix:
   `from paddleocr import PaddleOCR  # pyright:
   ignore[reportMissingImports]` — the runtime check already handles
   absence, and we don't want to drag the package into CI install just
   for the type-checker.
7. **`result.get("metric")` is `Unknown | None`; passing it as a dict
   key with `dict.get` fails.** Common shape: an untyped JSON
   container yields `Any | None`, and the next call wants `str`.
   Coerce at the boundary: `metric = result.get("metric") or ""` (or
   `str(result.get("metric") or "")` when downstream needs `str`
   specifically).
8. **`run_plugin_or_error` returning `(result, None) | (None, err)`
   leaks `None` into a `**outcome` unpack.** Same shape as #1 but the
   helper's untyped return makes pyright keep `outcome` as
   `Any | None` even after `if err: return err`. Two fixes work:
   `**(outcome or {})` (defensive; also fixes the runtime crash if the
   plugin actually returned `None`) or the explicit assert. Stage 4
   used `or {}` for the dict-unpack site since the value is genuinely
   optional at the plugin contract level.

### Stage 5 fixes (shipped)

`media/` is the package with the heaviest interaction with the
HuggingFace `transformers` stubs (CLAP, CLIP, SigLIP, SigLIP 2, X-CLIP,
DINOv2, DINOv3, EUPE — eight backbones, each with model + processor
load + forward calls). All 97 errors collapsed onto these patterns:

1. **`Optional[<ConcreteHFClass>]` for `self._model` / `self._processor`
   is a footgun.** The combination of `load_pretrained_local_first`
   returning `Any` and HF stubs' `from_pretrained` returning
   `Optional[T]` means pyright keeps the attribute's static type as
   `Optional[T]` even right after assignment. Then `self._model.to(...)`
   fails with *"to" is not a known attribute of "None"*. Worse, the HF
   processor stubs' `__call__` doesn't list the kwargs we actually pass
   (`sampling_rate=`, `return_tensors=`, `padding=`, `max_length=`,
   `truncation=`), so even the call sites fail with
   `reportCallIssue: No parameter named "X"`. Both problems vanish if
   you change the annotation to `Any`. Stage 5 made
   `self._model: Any = None` / `self._processor: Any = None` the
   standard pattern across every embedder, with a one-line comment
   pointing at the stub gap. The runtime `if self._model is None`
   guards already in place are the safety net.
2. **String forward references that aren't imported at module scope
   fail under pyright.** `vtsearch/media/embedder.py` declared
   `def patch_forward(self, media: dict) -> Optional["PatchEmbedOutput"]`
   five times, but `PatchEmbedOutput` was only imported inside the
   function body. `from __future__ import annotations` made this work
   at runtime but pyright still resolves the forward reference. Fix:
   add a `TYPE_CHECKING`-guarded `from vtsearch.media.patch_embed
   import PatchEmbedOutput` at the top of the module. (Same shape as
   Stage 1 fix #1, recurring in a different module.)
3. **ABC attributes that subclasses set dynamically must be declared
   on the ABC.** `vtsearch/media/__init__.py` does
   `mt._on_progress = callback` for every `MediaType` in the registry,
   but `MediaType` didn't declare `_on_progress`. Fix: add
   `_on_progress: ProgressCallback = _noop_progress` as a class-level
   default on `MediaType` (mirroring what `MediaEmbedder` already does
   in its `__init__`). This also lets media-type subclasses drop their
   ad-hoc `self._on_progress = _noop_progress` lines in `__init__`,
   though Stage 5 left the existing inits untouched.
4. **`torch.hub.load` returns `object`, which propagates through every
   attribute access.** EUPE loads its backbone via
   `torch.hub.load("facebookresearch/EUPE", ...)`. The stub return type
   is `object`, so `self._model.to("cpu")`, `.parameters()`,
   `.forward_features(...)` all fail. Same `self._model: Any = None`
   fix applies — covered by pattern #1. Also `torch.hub.load(...,
   trust_repo=True, ...)` errors because the stub claims `trust_repo`
   must be `str`; runtime accepts `bool` per the PyTorch docs.
   Stage 5 used `# pyright: ignore[reportArgumentType]` with a one-line
   comment rather than touching the stub.
5. **`transformers.modeling_utils.set_module_tensor_to_device` isn't
   in the stubs.** It exists at runtime (re-exported from `accelerate`
   under HF's lazy-import scheme) but pyright errors on the attribute
   read. The surrounding code already catches `AttributeError`, so
   `# pyright: ignore[reportAttributeAccessIssue]` is the right escape
   hatch — adding a runtime check is redundant.
6. **`SentenceTransformer.encode()` returns `Tensor | ndarray |
   list[Tensor]` per stubs.** Even when you pass
   `convert_to_numpy=True` (the default), the overload picks the union.
   The text embedders' `_embed_media_impl` / `embed_text_passage` /
   `embed_text` return types are `Optional[np.ndarray]`, so the bare
   `return self._model.encode(...)` errors with
   `reportReturnType: "Tensor" is not assignable to "ndarray"`. Fix:
   wrap the call with `np.asarray(...)` — cheap at runtime (no-op for
   already-ndarray values, conversion for Tensor).
7. **Soft-dependency imports for `whisper`, `mediapipe`, `paddleocr`
   still trip `reportMissingImports`** — same shape as Stage 4 fix #6
   (`paddleocr`). These are optional plugin deps not pinned in
   `requirements/base.txt`, so the per-import
   `# pyright: ignore[reportMissingImports]` is the right answer.
8. **`from ultralytics import YOLO` is the documented public API but
   the stub considers it private.** The stub diagnostic suggests
   "Import from `.yolo` instead", but the runtime module re-exports
   `YOLO` at the package root and that's what the upstream README
   tells users to write. `# pyright:
   ignore[reportPrivateImportUsage]` is the right answer here too —
   moving to the private path would just create maintenance debt the
   next time ultralytics rearranges internals.
9. **Annotation drift in `_frame_at_time`.** The helper's signature
   was `def _frame_at_time(cap, time_seconds: float) -> "tuple |
   None":`, but at runtime it returns the raw numpy frame from
   `cap.read()`. Two callers feed the result straight into
   `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`, which then fails because
   pyright thinks `frame` is a tuple. Fix: correct the annotation to
   `"np.ndarray | None"` with a `TYPE_CHECKING` import for numpy.
10. **`cast(Any, embedder).embed_pil_image(...)` /
    `cast(Any, embedder).embed_text_passage(...)`.** Same shape as
    Stage 3 fix #1: image and text demo loaders know they got an
    embedder of the right kind but the method only exists on the
    subclass. Two extra call sites (`_demo_sources.py` x2,
    `media_type.py` x1) needed the cast.
11. **Stale `include` entry for a renamed module.** Cleaning out the
    `vtsearch/settings_factory.py` line from `pyrightconfig.json`
    (the module was merged into `settings_models.py` between Stages 2
    and 5). Pyright doesn't fail on missing `include` paths — it
    silently prints *"File or directory ... does not exist."* and
    moves on — so the stale entry could have sat there forever.
    Worth a `grep` of `include` against the on-disk layout when bumping
    the gated scope.

### Operational notes from Stage 1

- The CI workflow tees the install-step output and the gated pyright
  output into `$GITHUB_STEP_SUMMARY`. When a stage's PR fails, the
  summary tab on the run is the fastest place to see what broke — no
  need to expand the raw step log.
- Pyright pins to `1.1.408` in the workflow. Bump deliberately when
  taking a new stage so the version change is reviewable on its own,
  not bundled with error fixes.

## Configuration choices

`pyrightconfig.json`:

```json
{
  "include": ["..."],
  "exclude": ["**/__pycache__", "**/node_modules", "tests", "frontend", "scripts", "data"],
  "pythonVersion": "3.10",
  "typeCheckingMode": "basic",
  "useLibraryCodeForTypes": true,
  "reportMissingImports": "error",
  "reportMissingTypeStubs": "none"
}
```

- `useLibraryCodeForTypes: true` lets pyright infer from installed
  package code when stubs are missing — important for `torch`,
  `transformers`, `laion_clap`.
- `reportMissingTypeStubs` is silenced; we don't ship stubs.
- `tests/` is excluded for now (Stage 7).

## CI shape

Single workflow `.github/workflows/pyright.yml`:

1. **`gated` job** — installs deps, runs `pyright` (uses
   `pyrightconfig.json` `include`). Hard gate.
2. **`advisory` job** — installs deps, runs
   `pyright vtsearch/ || true`. Posts the error count in the job
   summary. Does not fail the build.

The advisory job is deleted in Stage 6.

## Non-goals (now)

- Strict mode.
- Annotating every public function.
- Type-checking `tests/`, `frontend/`, or `scripts/`.
- Wiring pyright into `./run-tests.sh` (CI-only — keeps the local
  test loop fast since pyright on a fresh repo takes ~30 s).

## Local usage

```bash
pip install pyright              # or use pre-built CI binary
bash scripts/install-cpu.sh      # so reportMissingImports passes
pyright                          # gated scope (matches CI hard gate)
pyright vtsearch/                # full-package check (matches advisory job)
```

## Backwards compatibility

This is additive — no runtime behavior changes. Type-only fixes
applied during each stage may change function signatures (e.g.
parameter renames to match base classes); those are noted in each
stage's PR description.
