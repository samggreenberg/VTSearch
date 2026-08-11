# Documentation accuracy and structure

## Background

A full audit of every documentation surface in the repo (134 markdown files across
`docs/`, `vtscore/docs/`, `README`/`CLAUDE`, the plan and experiment archives, plus
mechanical sweeps for links, anchors, code-path references and orphans) produced 324
evidence-backed findings. This file tracks what is still owed. The issues listed below
own the concrete fixes; everything under "Open findings not yet promoted" is recorded
here because it has no issue yet.

**The systemic cause is measurable, and it is not carelessness.** Most of the doc set was
written in one batch on 2026-07-20 (the vtscore-extraction docs drop). Since then 557
commits landed, 168 of them touching `vtscore/` code — against 20 touching `vtscore/docs/`.
Docs are written in bursts and never revisited, because nothing ties a code change to the
prose that describes it.

Three things follow from that, and they shape how this work should be done:

- **Inventory drift dominates.** The single most repeated defect across all 15 audit areas
  was a hand-maintained list of registry contents (embedders, plugin families, exporters,
  demo datasets, env vars, settings keys) disagreeing with the code and with the other nine
  copies of the same list. This is a generation problem, not ten editing mistakes.
- **The mechanical defects are mechanically detectable.** Dead links, dead anchors, dead
  file paths and leaked absolute paths accounted for roughly 30 findings, and `run-tests.sh`
  gates ten things today without checking any of them.
- **Prefer invariants over generation over pinning.** The repo already has all three shapes
  — `wiring-check.py` (invariant), the OpenAPI snapshot (generation), `check-eval-app-sync.py`
  (digest pinning). A noisy gate gets `--update`'d blindly, a failure mode CLAUDE.md already
  names for the eval pins, so reach for the cheapest shape that catches the class.

Fixing individual docs without the first two bullets buys about six weeks.

## Tracked as issues

<!-- item-sep -->

<!-- item-sep -->

- [ ] #2978 — Personal Gmail address committed as the `support_email` example in DEPLOYMENT.md (Haiku 4.5)

<!-- item-sep -->

<!-- item-sep -->

- [ ] #2980 — DEPLOYMENT.md has no security section; file-browser.md misdescribes the single-user browse root (Opus 4.8)

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

- [ ] #2983 — Add a docs drift gate to run-tests.sh: links, anchors, code paths, absolute-path leaks (Sonnet 5)

<!-- item-sep -->

- [ ] #2984 — Generate registry inventories instead of hand-maintaining them in ~10 documents (Opus 4.8)

<!-- item-sep -->

- [ ] #2985 — Every runnable snippet in the vtscore docs fails (Opus 4.8)

<!-- item-sep -->

<!-- item-sep -->

- [ ] #2987 — Retire docs/HANDOFF.md (the stalest doc hub) and fix doc orphans (Opus 4.8)

<!-- item-sep -->

- [ ] #2988 — docs/api/medias.md vote endpoint is wrong in all three parts; windowed sort contract undocumented (Sonnet 5)

<!-- item-sep -->

- [ ] #2989 — EXTENDING docs teach contracts that raise at runtime, and omit the normalize.py validation layer (Opus 4.8)

<!-- item-sep -->

- [ ] #2990 — EXTENDING-processors.md omits the registration story, so its recipes cannot be wired in (Sonnet 5)

<!-- item-sep -->

- [ ] #2991 — ARCHITECTURE.md: false flask.g claim about the library tier, plus directory-map drift (Sonnet 5)

<!-- item-sep -->

- [ ] #2992 — CLI.md: detector paths are slugged, and the plugin-flag discovery command cannot work (Sonnet 5)

<!-- item-sep -->

- [ ] #2993 — style-guide.md Section 2 names classes that no longer exist; teaches hand-copied modal markup (Sonnet 5)

<!-- item-sep -->

- [ ] #2994 — USER_GUIDE.md: dead table of contents, hidden importers described, shipped features missing (Sonnet 5)

<!-- item-sep -->

- [ ] #2995 — DEPLOYMENT.md settings schema predates the server/user split; env-var tables omit VTSEARCH_DATA_DIR (Sonnet 5)

<!-- item-sep -->

<!-- item-sep -->

- [ ] #2997 — CLAUDE.md Commands and Test Markers sections are out of date with run-tests.sh (Sonnet 5)

<!-- item-sep -->

- [ ] #2998 — The Angular frontend has no architecture documentation (Opus 4.8)

<!-- item-sep -->

- [ ] #2999 — vtscore/docs/packages/: half-covered module tables, six undocumented subpackages, stale features (Opus 4.8)

<!-- item-sep -->


## Open findings not yet promoted

Grouped by document. Each bullet is a defect with no issue of its own yet; promote one to
an issue (and replace the body here with a pointer line) when it becomes worth shipping
separately.

<!-- item-sep -->

- **The `run-tests.sh` gate list is still hand-maintained in three places.** CLAUDE.md's "What
  `run-tests.sh` gates" table, `docs/HANDOFF.md`'s quality-tools paragraph and the script's own
  usage header each restate the chain by hand; the first two were six gates stale before #2997.
  This is the same inventory-drift shape as #2984, and it wants the same treatment — but note the
  cheaper shape fits better here: rather than *generating* the table, an invariant check could
  assert that the set of `echo "…"` stage banners in `run-tests.sh` matches the rows in CLAUDE.md's
  table, which is a few lines inside whatever #2983's docs-drift gate becomes. Worth folding into
  that gate rather than shipping its own script.

<!-- item-sep -->

- **README.md and SETUP.md — install-path drift.** `SETUP.md:205-207` describes GPU detection
  as "nvidia-smi absent → CPU wheel", but `scripts/install.sh:1137-1173` only falls back to CPU
  when no NVIDIA PCI hardware exists; with a card present and no driver it prompts for a sudo
  driver install, and non-interactively it **stops with exit 1** (`:768-779`). Headless callers
  need `VTSEARCH_AUTO_DRIVER=1` / `VTSEARCH_ASSUME_CPU=1`, documented nowhere. `SETUP.md:226-232`
  omits that the GPU path also installs multi-gigabyte cuML/RAPIDS (skippable with
  `VTSEARCH_SKIP_CUML=1`), runs a GPU smoke test, and installs toponymy, facenet-pytorch and the
  pre-commit hook. `SETUP.md:599-601` claims `install.sh` provides the Angular build tools; it
  never touches npm. Both files show a startup banner (`* Running on http://0.0.0.0:5000`) that
  no code emits. `SETUP.md`'s env table omits `VTSEARCH_DATA_DIR`; the Docker section skips the
  image-embedders variant; README's top-level tree omits directories a newcomer meets immediately.

<!-- item-sep -->

- **README.md — the media-type count is five, the code ships six.** The intro omits the `face`
  convert-in half type (`vtscore/media/face/media_type.py`, FaceNet identity space, provisioned by
  `install.sh:855-867`). The same undercount appears in `docs/HANDOFF.md` and the vtscore docs.

<!-- item-sep -->

- **CHANGELOG.md has no owner and no readers.** Last touched 2026-07-20; zero inbound links from
  any doc; `docs/RELEASE.md`'s seven steps never mention it. Its `Unreleased` section records only
  the library extraction, while breaking user-facing changes since (the `safe_thresholds` setting
  deletion, the fold-anchored threshold, exporter open-URL, datasource importers, importer-named
  datasets) went unrecorded. Under the repo's own "every commit on `dev` is a release" model,
  nothing can stay `Unreleased`. Decide whether the file has a job — if it does, add it to the
  release runbook and link it; if it doesn't, delete it rather than leave a rotting promise.
  `vtscore/CHANGELOG.md` by contrast is actively maintained and should stay.

<!-- item-sep -->

- **ARCHITECTURE.md — inventory and description drift beyond the flask.g claim in #2991.**
  `settings_models.py` is described as Marshmallow when it is pydantic; the theme enum is given as
  three modes when there are four; `vtscore/media/base.py` is credited with ABCs that live
  elsewhere; the exporter inventory omits `open_url` and `portable_detector`; request hooks and
  error handlers are attributed to `app.py` after moving to `vtsearch/hooks.py` and `errors.py`.
  Missing from the map: `gpu_backends.py`, `io.py`, `single_instance.py`, the signpost
  (region-labeling) subsystem under `projection/`, half of `datasets/` (`archive_stream.py`,
  `clipper_chain.py`), 13 of 18 `eval/` modules including `autopilot_flow.py`, and
  `state/sort_results_cache.py`. Five server-tier settings keys are missing. `PluginRegistry`'s
  entry-point discovery is described as directory-scan only. The auth section presents
  `DefaultLoginProvider` as the sole built-in, omitting `TrivialLoginProvider` and
  `ApiKeyLoginProvider`. Context resolution also accepts `dataset_id`/`detector_id` query params
  as a header fallback, undocumented. Minor: a duplicated word in the ownership-tracking sentence;
  the `DatasetContext` key-state table understates what a context holds.

<!-- item-sep -->

- **docs/api/ — undocumented endpoint families.** Beyond #2988: the four processor execution
  endpoints (`/api/extract`, `/api/auto-extract`, `/api/localize`, `/api/auto-localize`); the
  datasource-importer family (`GET /api/datasource-importers`, `POST /api/datasource-import/{name}`);
  three Find endpoints (`/api/find/queue-ids`, boundary, evidence-coverage); the saved-labelset
  element vote (`POST /api/detectors/{name}/labels/{element_id}/vote`); the labelset-source
  move-file endpoint; and the `server` and heartbeat channels of `GET /api/events`.

<!-- item-sep -->

- **docs/api/ — wrong or incomplete contracts.** `settings.md` documents the theme enum and default
  wrongly and omits roughly twenty real `PUT /api/settings` keys; `labeling.md` omits the resolved
  detector fields both `/api/inclusion` verbs return; `medias.md` omits four always-present keys on
  `GET /api/votes`, the `label_filter` param on label export, the crop fields on server-media
  upload, and `"none"` as a `vote-bulk` target; `detectors.md`'s second create example omits the
  required `media_type`; `io.md`'s exporter list omits `holder` and `portable_detector`;
  `datasets.md` shows load responses in a shape the routes do not return; `auth.md` omits the SPA
  deep-link routes. `API.md` undersells the error envelope (`{error, detail, request_id}`, plus the
  422 marshmallow shape).

<!-- item-sep -->

- **ML.md and EVAL.md — narrow but real.** `docs/EVAL.md:254-266` tells the reader to pass
  `acq_inclusion_offset=0` to `run_voting_iterations_eval`, which accepts no such parameter and
  forwards none (`vtscore/eval/voting_iterations.py:2774-2792`, `:2880-2899`) — following the doc
  at the documented entry point raises `TypeError`; the parameter exists only on
  `simulate_voting_iterations`. The eval-dataset table is missing 6 of 23 datasets, and the torch
  thread-configuration file pointer is stale. Everything else checked — every algorithmic constant
  the audit sampled (Adam lr/weight-decay, 200-epoch cap, label smoothing, conformal BASE/QPOS_MAX,
  kappa/mid_tilt, acquisition offset, autopilot quorum, atlas k/min-node-size) matches the source.

<!-- item-sep -->

- **vtscore/docs — faq.md contradicts the code and concepts.md.** The FAQ's description of the
  Inclusion knob disagrees with both. The context-resolution chain is documented with a nonexistent
  `override_dataset_context` and a `None` terminal case that never occurs. Three real subpackages
  (`projection`, `timing`, `datasource_importers`) and the sixth media type are absent from every
  vtscore inventory. (The broken snippets are #2985; the package-doc coverage gaps are #2999.)

<!-- item-sep -->

- **vtscore/docs/extending — stale contracts beyond #2989.** The media-dict key is `media_type`,
  not `type`. The clipper naming convention shown (`sound_tiling_2.0s`) carries a parameter suffix
  real names do not have. `dataset-importers.md` points four times at
  `../../datasets/importers/base.py`, which is a package now — in the doc most likely to be
  copy-pasted from. About a dozen `file.py:NNN` anchors are stale.

<!-- item-sep -->

- **Leaked absolute machine paths.** `vtscore/docs/packages/cli.md:14,360` and `config.md:15` carry
  `/home/user/VTSearch/...` in visible link text (the link *targets* are correct relative paths;
  only the label leaks). Clear artifact of an agent-authored docs session. Covered by the #2983
  gate once it exists, but worth fixing directly.

<!-- item-sep -->

- **Hand-maintained line-number anchors have a 100% rot rate.** Nearly every `file.py:NNN`
  reference in `vtscore/docs/packages/` is wrong, often by hundreds of lines, and the pattern
  recurs in the extending guides and plans. Needs a policy call, not one more sweep: either stop
  citing line numbers in prose and reference module-and-symbol instead (stable, greppable), or
  generate them. Recommend the former.

<!-- item-sep -->

<!-- item-sep -->

- **Screenshot staleness is unverifiable.** `scripts/screenshots/wiring-check.py` validates id and
  asset wiring only — nothing compares a shot against the current UI. So the reshoot queue's
  "empty table means no known-stale shots" resting state is an unfalsifiable claim, and framed
  surfaces changed after the last reshoot (all 40 PNGs committed 2026-07-20) with no rows filed.
  Either downgrade the claim in `docs/user/screenshots-reshoot-queue.md`, or add a cheap staleness
  signal — e.g. queue a shot automatically when a commit touches a component named in its
  `embeddedIn`/`caption` fields.

<!-- item-sep -->

- **The two audit areas that disagreed with each other.** Independent reviewers "corrected" the
  plugin-family count to ten and to eleven, and a sentinel-grep gives a third answer — because no
  doc states which registry it is counting. The authoritative inventory is
  `vtscore/plugins/inventory.py:228-241` (11 library families) plus 3 app families from
  `vtsearch.shim.register_app_plugin_families`. The fix is a stated counting rule or a generated
  list, not another number; folded into #2984. Recording it here because it is the one place the
  audit contradicted itself, and the next person to "fix" the count will hit the same fork.

<!-- item-sep -->

- **`media_sources` is a real extension point with no authoring guide.** Eleven library plugin
  families, and `vtscore/datasets/sources/` (8 built-in plugins, third-party-extensible via the
  `vtscore.media_sources` entry-point group) has no "Adding a…" section in `EXTENDING-plugins.md`
  or `vtscore/docs/extending/`. Every other family has one.
