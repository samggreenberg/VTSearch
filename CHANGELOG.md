# Changelog: `vtsearch`

User-facing changes to the `vtsearch` Flask + Angular application. The
companion `vtscore` library has its own CHANGELOG at
[`vtscore/CHANGELOG.md`](vtscore/CHANGELOG.md).

`vtsearch` is **not** versioned with traditional semver. The `__version__`
attribute is the UTC timestamp of `HEAD`'s commit (ISO 8601, Z-terminated),
computed from git at import time in `vtsearch/__init__.py`. Every commit on
`dev` is effectively a new release; there is no tracked version constant to
bump and no per-release tag.

This CHANGELOG is therefore a **curated** record of notable changes and does
not list every commit. Use `git log` for the full history.

## Unreleased

### Changed

- **Too few labels give the Goods' centroid, not a half-trained detector** (issue #4643). Test,
  AutoFind, Find and the CLI used to train a detector from the first Good and Bad, so a detector
  with 3 Goods and 1 Bad could be tested or exported. Below 3 Goods and 4 Bads (Autopilot's own
  opening quorum), they now give the Goods' centroid: everything ranked by how close it is to the
  average of the Goods, cut where the scores split, with no Bad needed. One Good is enough to test
  a detector, and the Test view and AutoFind results say when the centroid was used and how many
  more Goods and Bads a trained detector needs. The Threshold control does not move the centroid's
  line. Labels still save on every vote, and Export labels is not gated: whatever imports the labels
  gets the same rule. The Train view's own sort is unchanged.
- **Autopilot's Hard and New picks go where the detector is least sure** (issue #3546). They now
  sample where the detector's own model puts even odds on an image being a match, instead of a fixed
  depth below the line that had stopped following the detector. In simulation that finds about 3 more
  matches per 150 votes at every balance setting; 38% of Hard picks are matches (was 19-24%), and the
  final detector is as good or better. The `hard` phase's help text now says what it does.
- **Autopilot stays Done once it gets there** (issue #4621). Labeling on after **Done!** used to
  drop the phase panel back to **Refine Boundary** whenever a vote knocked an indicator off green,
  then jump it forward to Done again when the indicator recovered. Done now stays checked, and a
  seventh step, **Keep Improving.**, takes over: its light shows the lowest of Smart, Stable and
  Span, and its line says whether Autopilot is offering boundary items or diverse ones, or that all
  indicators are green. What Autopilot picks is unchanged.

### Added

- **The command line can delete what it imports once AutoFind has run** (issue #4674).
  Settings › **AutoFind** gains **Delete the dataset after AutoFind**, under **Command Line**,
  off by default. With it on, `python app.py --autodetect` (or `--pipeline`) deletes the dataset it
  imported once its detectors have run and the results are exported, so a nightly run no longer
  fills the dashboard. A run with no AutoFind detector to run, one that fails, and one pointed at a
  dataset already on the dashboard keep it, and AutoFind started inside the app never deletes. The
  setting is per user (`autofind_cli_delete_dataset`; a `--settings` file can set it for a run
  without `--user`), and a deleting run reports a `dataset_deleted` progress event.

- **AutoRun is now AutoFind; its ⋯ run opens no dialog, and Find Results
  gains Browse** (issue #4615). The Dashboard's **AutoRun** tab, the detector
  ⋯ menu's **Move to AutoRun**, the dataset ⋯ menu's **Run AutoRun** and the
  Add Dataset dialog's checkbox all say **AutoFind**, as the Settings tab
  (formerly *Auto-Find*) and its exporter do. AutoFind is the unattended path,
  the app's twin of the command line's run: a dataset's ⋯ **Run AutoFind** now
  ends with the same notice an import's run gives (*AutoFind finished on
  "Birds": 12 hits*, with **View results**) instead of opening the results
  dialog. The big **Find** button still opens its results as they land, and
  its runs are called Find on the dataset row and in their notices. The
  dialog, now **Find Results**, gains a **Browse** button that lays the listed
  items (Good, Bad or Both) out in Browse as a map of their own; **← Back to
  Find Results** returns to the Dashboard with them open again. The API
  follows the name: `POST /api/datasets/registry/<id>/autofind`,
  `GET /api/autofind/runs/<run_id>`, the import form's `autofind` flag, the
  `autofind` block on a run's `loading-tasks` row (whose `trigger` is now
  also `find` for the Find button's runs) and the `autofind_on_import`
  setting, which starts back at its default (on) for anyone who had turned
  `autorun_on_import` off. The autorun *processors* (`/api/autorun-extractors`,
  `/api/autorun-localizers`) are a different thing and keep their names.

- **Run your own function when an import finishes** (issue #4616). A server
  admin can name functions with `--on-dataset-imported module:function` (or
  `VTSEARCH_ON_DATASET_IMPORTED`, comma-separated), and each is called with a
  `DatasetImported` event whenever a user's import from the web app succeeds or
  fails, for example to email them. The code lives outside this repository; the
  module only has to be importable on the server. A bad spec stops the server at
  startup (the env form warns instead), and there is no settings-file key for it.
  See [EXTENDING.md § Dataset-Import Hooks](docs/EXTENDING.md#dataset-import-hooks).

- **A finished test's verdict is kept on the detector** (issue #4526). Reaching
  Done in Find's Test autopilot saves the verdict with the detector, one per
  collection it was tested on: the share right and the share found as ranges,
  the balance, the date and the picks, never a score. The detector's **Stats**
  gain a *Tested on* section listing them, and the Dashboard's **AutoRun** tab
  shows each detector's latest under its name (or *Untested*). A retrain marks
  a verdict *out of date*; it stays. Opening Find again on a tested collection,
  with the detector and its ranking unchanged, resumes the test from the kept
  picks (back in the Review tab's piles). New route `POST /api/line-test/forget`
  resets a kept verdict, for the screenshot harness.

- **Find opens on a Test autopilot** (issue #4524). The Find view's left panel
  now carries two tabs, the Train view's Autopilot / Manual split applied to
  testing. **Autopilot** measures the detector's line on the collection in
  front of you with random picks from rank bands on either side of it, five a
  round, through the phases Score, Check the matches, Check the misses and
  Done, each with a light; the right pane shows the result as it forms (the
  balance's F-beta, the likely share right as a number, the share of all the
  matches found in words, a precision-by-count chart at band resolution, the
  picks by band) and at Done a verdict with three exits: **Move to AutoRun**,
  **Lean the Threshold** (what each balance would ship, from the same picks)
  and **Add Corrections and retrain**. **Review** is Find as it was: the ranked
  list under the line, the boundary walk, the Verified Good / Verified Bad
  piles, To Dataset, Export and Browse, with the test's picks already in the
  piles. The Threshold is frozen while a test phase runs, and its note reads
  this collection's result (or *untested*), never the check Train ran. Test
  votes never train the detector; **Add Corrections** marks the result out of
  date. New routes under `/api/line-test`.

### Removed

- **Imports no longer show a time-left estimate, and the switches around it are gone**
  (issue #4667). The estimate on dataset imports, staging imports and labelset
  missing-media fetches swung too wildly to trust, so it is gone for everyone; those bars
  still fill, count and name their step, and every other progress bar keeps its estimate.
  With it go the operator switch that hid it (`--hide-ingest-eta`,
  `VTSEARCH_HIDE_INGEST_ETA`, `"hide_ingest_eta"` and its Settings ▸ Server row) and the
  per-deployment timing profile that tuned the progress bars' pacing
  (`VTSEARCH_TIMING_PROFILE`, `VTSEARCH_TIMING_RECORD`,
  `scripts/profiling/tune_timing_profile.py`). A server started with `--hide-ingest-eta`
  must drop the flag, which is no longer accepted; the two environment variables and the
  settings key are ignored. Every bar now paces from the weights the app ships with.

- **The Find view's Stats modal** (issue #4524). Its Training-domain overlap
  and Evidence coverage chips, its 2×2 of your checks and its precision chart
  now live in the Autopilot tab's result pane; the chart draws the test's
  ranges at every band edge instead of the *Checked by you* curve, which
  counted only what the boundary walk happened to serve.

### Changed

- **A text sort's green region follows the balance** (issue #4603). At a
  balance of beta 1 or below, a typed query's line now keeps about as many
  images as the sort has matches, times 3/8 at the 1/4 preset and 1 at the
  balanced one. The count is estimated from the scores alone: what stands
  above the bulk's median + 4 robust sigmas beyond a Gaussian bulk's share.
  The guarded line kept about the same set at every balance, about four times
  the matches at COCO Better's prevalence. Priced on 144 classes at 0.1%,
  0.44% and 2% prevalence, the typed query's set gains +0.13 to +0.30 F-beta
  at beta 1/4 and +0.01 to +0.13 at beta 1. At beta 4 and above 1 the guarded
  line stays. That set is what a user sees through Autopilot's whole opening,
  a median ~40 clicks (#4605). Autopilot's sampling cut (`acq_threshold`) does
  not move, so a session's picks are the same at any balance.

- **A text sort's green region ends at the guarded line, and Autopilot still
  samples at the midpoint** (issue #4136; the rule is #3826's). A typed query's
  cosine sort is usually one broad mode with the matches as a thin shoulder, so
  the two-Gaussian midpoint every text sort used to draw split the mode and
  painted a median 43% of the list green, 12–54× the true matches. The line is
  now the guarded rule: the mixture's midpoint only when its two components are
  separated, otherwise the bulk's median + 3 robust sigmas, which paints about
  the matches (F1 0.17 → 0.39 on 1,120 labelled sorts; better at the 1/4 and 1
  balances and a tie at 4). On a query that matches most of a collection and
  that the embedder does not separate ("a person" on a photo set), the line
  admits only the top of the matches; that trade is documented, not fixed. The
  sort response now carries `acq_threshold`, the midpoint, beside `threshold`,
  and Autopilot's Bad phase reads it, so the opening's picks are exactly what
  they were before this change (the guarded line as the *sampling* position
  made the first detectors worse in an A/B). `VTSEARCH_TEXT_SORT_CUT=gmm_midpoint`
  restores the plain midpoint for display.

- **Test takes exactly one dataset and one detector** (the Dashboard's action bar).
  With two datasets or two detectors ticked, Test used to stay enabled and quietly open on
  the first ticked pair; it is now greyed out with *Select exactly 1 dataset* (or
  *detector*), as Train is. Find still takes any number. The reasons shown under the
  Train / Test / Find buttons were also reworded to fit under them: two had been cut off
  with an ellipsis (*Detector has no training labels*, and Train's *Frozen: move to Drafts
  to retrain* for an AutoRun detector), and the tooltips now name the Train and Test views.

- **Testing a detector checks deeper below the line** (issue #4542). When the detector
  has a class model, the Test autopilot's *Check the misses* step now keeps drawing picks
  from deeper bands until it has spent its 40, instead of stopping at the first band that
  turns up nothing: the share of all the matches found was usually overstated before, and
  is now right three times in four or better, for about 30 more picks. A detector with no
  class model (a structural or document one) still stops at the first empty band, and its
  result now gives the share found in words only and says it is unmeasured below the
  bands checked.

- **Find is now Test, and the AutoRun button is now Find** (issue #4525, the
  last slice of #4520). The Dashboard's big button that scores a dataset and
  tests a detector's line on it reads **Test**, and its view lives at
  `/test/<dataset>/<detector>`; an old `/find/...` link lands on the
  Dashboard. The third big button, which runs every ticked detector on every
  ticked dataset as AutoRun does, reads **Find**. Test carries a new flask
  icon, and Find the magnifier Test used to carry. The keyboard help's *Train /
  Find* tab is *Train / Test*, and the Browser's way back is **← Back to
  Test**. In the user guide, *Find: testing and reviewing* is now *Testing a
  detector*, and the step-by-step's Step 4 tests the detector. The API keeps
  its names (`/api/find-label`, `find_mode`, the `find` SSE channel).

- **Document search without a GPU is ~9x faster per click at 50,000 pages** (issue #4514).
  A vote on a document collection served from a CPU took about 2 minutes at 50,000 pages:
  each click converted the whole tile matrix to float32. The conversion now happens once
  when the collection's pages are first scored, so a Good takes ~14 s and a Bad ~3 s
  (8 cores; ~10 s and 0.1 s at 5,000 pages). That first search is ~30 s slower at 50,000
  pages, and the CPU server holds twice the tile matrix in memory. Rankings are unchanged.

- **A document ranking no longer depends on the machine the server runs on** (issue #4481).
  The same detector used to order near-equal pages differently on the GRID's AMD and Intel
  nodes, because their math libraries rounded the page vectors differently. The server now
  uses one set of OpenBLAS kernels on every x86 machine (`OPENBLAS_CORETYPE`, see
  DEPLOYMENT.md), and orders Stage 1 on an exactly recomputed score. Which pages are returned
  did not change; only the order among near-ties did.

- **Autopilot checks the line when your labels still overlap** (issue #4496).
  When the detector scores your Good and Bad answers close together (the
  labels line's separation below 1.5, from 10 votes on), Autopilot opens the
  spot check itself, once it has started learning, with a line saying why, and
  asks again 25 votes after the last check if they still overlap. In Manual,
  the **Check 5 picks** button turns primary with a note. Priced at equal
  clicks, prompted sessions returned far fewer wrong images and found more
  right ones (`docs/experiments/2026-10-05-weak-check-4496/REPORT.md`). The
  balance payload carries `separation` and `check_due`.

- **Document collections stop on a dry run** (issue #4488). On a collection
  of document pages (`sift_vlad_doc`), Autopilot runs four phases: after the
  initial goods and bads it offers the detector's own best matches, re-ranked
  after every vote and with no 20-good target, until 16 in a row are not good;
  then it is **Done!** and says **Detector Trained**. There is no Refine Boundary
  or Explore Diversity phase there. On the Manual tab the Smart, Stable and
  Diverse indicators give way to one readout, **Dry run n/16**. Those
  indicators scored a page-vector model the document ranking does not use, at
  the structural detector's threshold. `GET /api/labeling-status` gains
  `stop_rule` and `dry_run`.

- **Find returns a real set from the first clicks** (issue #4492). Early in a
  session the Threshold's line used to keep a single image for about 15 clicks,
  because the model of your labels was held to a fixed minimum width that an
  early detector's scores are narrower than. The minimum now follows the
  collection's own spread, so after a handful of votes the line keeps a set of
  sensible size: at balanced, around 30 to 45 images where it kept 1.
- **The Threshold's outer radios lean further** (issue #4448). The False
  Positives radio is now F-beta at 4 (was 2) and the False Negatives radio
  F-beta at 1/4 (was 1/2); the middle stays balanced. On the same detector the
  three now keep about 81, 47 and 26 images where they kept 58, 47 and 38, so
  the choice makes a visible difference. A detector saved on an old outer
  radio shows on the new one on the same side.
- **The Threshold's line comes from your labels alone, in Train and in Find**
  (issue #4452). The line under a balance is no longer a count of the top of
  the ranking: VTSearch learns from your votes' held-out scores how high a
  match and a non-match tend to score, estimates how common matches are in the
  collection being searched, and draws the line where the balance your radio
  asks for is best for a collection like that. It keeps every item above the
  line, possibly none - Find on 200 images with nothing like the target used
  to return the top 26-128 of them, all wrong, and now returns next to nothing.
  Nothing is stored but the labels, so an exported labelset draws the same
  line on another collection or with another embedder. The spot check never
  moves the line any more (it is advisory at every radio): it measures, and
  its picks train the detector like any vote. The balance state's `count` is
  what the line keeps of the ranking scored last, and `shape` is always
  `advisory`.
- **The spot check's effect on the line follows the balance** (issue #4427).
  At the precision-leaning and balanced presets (beta 1 and below) the check
  is advisory: the walk runs as an audit and reports its ranges, its votes
  stay ordinary votes, and the line keeps the balance's own count instead of
  moving to the walk's end. At the recall-leaning preset (beta 2) the walk
  may only trim the line - it audits the bands holding it and steps shallower,
  never deeper - and the line keeps the set it ends on. Priced on the withheld
  images above the threshold the app holds: the old walk lost 0.06 / 0.03 of
  F-beta at beta 0.5 / 1 by buying recall with bands that were mostly wrong;
  advisory gains 0.09 / 0.04 over it there, and trimming is never worse and the
  best at beta 2. The balance state and `GET /api/balance` carry `shape`
  (`advisory` or `trim`) and `audited` (the set the walk ended on); the check
  modal and the balance control say which set the ranges describe.

- **Reverted the day after: Autopilot's picks sample at the line − 4 re-cut
  again under the balance** (issue #4427). The half-argmax cut below was priced
  on the rank-count reading of the line; on the objective (the withheld images
  above the threshold the app holds) it was worse at every preset, because its
  harvest thins the unvoted top and the kept set's edge score climbs.
- **Autopilot's picks sample higher under the balance** (issue #4409, the
  pricing in `docs/experiments/2026-10-01-acquisition-fbeta-4409/REPORT.md`).
  The acquisition cut the Hard / New picks sample around is now the score at
  half the depth of the mixture's F-beta argmax over the unvoted ranking
  (`ACQUISITION_ARGMAX_FACTOR`), carried on the learned sort's
  `acq_threshold` as before; it can sit below the line. On the bench that
  finds about nine more positives per 150 clicks than the old cut four
  inclusion steps above the line, with the returned set's F-beta up and AP up
  0.02–0.03 at every preset. With no mixture estimate, and under the
  deprecated floor, the old cut stands.
- **The Threshold control is a balance, not a precision floor** (issue #4413,
  the owner's ruling of 2026-10-01 after #4411 priced it). The three radios
  under the False Positives … False Negatives spectrum now pick how to weigh
  precision against recall - F-beta's beta 2 (lean to recall), 1 (balanced,
  the default) or 0.5 (lean to precision) - and the line keeps the set with
  the best estimated F-beta: before any spot check the vote-anchored
  mixture's F-beta argmax, capped at the top 128 (beta 2) or 32 (beta 1 and
  0.5); after one, the band where the check's F-beta estimate peaked. The
  check walks the ranking in bands as before but ends at the peak instead of
  at a floor, and reports the kept set's likely precision *and* recall
  ("Checked · likely 55–80% right, about half of them found (checked 15) ·
  48 kept"); nothing is "confirmed" or "short" any more. The Find Stats chart
  no longer draws a floor line. Wire: `GET|POST /api/balance` (`beta`,
  `status` unchecked | checked, `count`, `precision`, `recall`, `fbeta`,
  `schedule`, `threshold`, `n_returned`, `line_preference`); every response
  that carries a detector's line carries `balance` beside `floor`; a
  `/api/precision-check` under the balance finishes `checked` and carries
  `beta`, `fbeta` and `recall`. Settings: `beta` (0.25–4, default 1) and
  `line_preference` (`balance`, the default, or `floor`). **Deprecated:** the
  `min_precision` setting, `GET|POST /api/min-precision` and the floor's
  `confirmed` / `short` states draw the line only under
  `line_preference: "floor"`; they stay for one release and then go. Headless
  runs (the CLI and AutoRun) export the balance's unchecked set and say so
  ("exports its top 32 unchecked (at F1)"), with `beta` on the
  `detector_unchecked` event and `balance` beside `floor` in their results.
- **The GPU stack moves to CUDA 12.9, RAPIDS 26.8 and pandas 3** (issue
  #4390). `pyproject.toml` now pins `pandas>=3` (it was `<3`, #4381), and the
  GPU install moves with it, because cudf, which cuML depends on, only takes
  pandas 3 from RAPIDS 26.8, and RAPIDS 26.x needs the CUDA 12.9 libraries
  that only torch's `cu129` wheel pins. `scripts/install.sh` now picks `cu129`
  on Turing-or-newer cards with a driver at CUDA 12.9 or later, and installs
  `cuml-cu12>=26.8` on that tag only; on `cu118`..`cu128` the cuML step skips
  itself with a message and the app runs UMAP / k-means on the CPU. A Volta
  card (V100) keeps `cu124` torch and loses GPU UMAP / k-means; a venv built
  before this change keeps working as it is, but its next `install.sh` run
  needs a driver at CUDA 12.9 to keep cuML. The two CUDA Docker images move to
  `nvidia/cuda:12.9.1-runtime-ubuntu24.04` with `cu129` torch, which raises
  their host-driver floor to CUDA 12.9 (575+) or the R535..R570 branches and
  drops Volta hosts (see [Choosing an image](docs/DEPLOYMENT.md#choosing-an-image)).
  pandas 3 turns on copy-on-write and the string dtype by default.
- **Python 3.11 or later is now required** (issue #4385). Python 3.10 reaches
  end of life in October 2026, so `scripts/install.sh` now refuses it, and pip
  will not install VTSearch on it. A 3.10 venv needs rebuilding on 3.11+
  before its next install. The Docker images move with it: the CPU images run
  on `python:3.12-slim`, and the two CUDA images on
  `nvidia/cuda:12.5.1-runtime-ubuntu24.04` with Python 3.12, which starts only on
  a host driver new enough for CUDA 12.5 (555+) or from the R535 or R550
  branches (see [Choosing an image](docs/DEPLOYMENT.md#choosing-an-image)).
- **The Find Stats chart no longer claims an "at least"** (issue #4360). Its
  **Estimated (at least)** curve is gone, with its legend entry, its dot and
  the *No estimate yet* notes: the estimate behind it could not back the bound
  it was labelled with. The chart now draws only what you checked by hand, the
  Threshold, the line, and the spot check's likely range for the set the line
  keeps. `GET /api/find/stats` drops `estimated_precision` from each curve
  point and `estimate_status`, `calibration_positives` and
  `min_calibration_positives` from the response.
- **The Dashboard's RAM / Disk bars appear only when they matter.** On the
  **Default** setting each bar now stays hidden until its free space is
  running low *for your datasets*: when it would hold fewer than three more
  datasets the size of your largest one (1 GB is assumed before you have
  any). A nearly full but enormous disk no longer raises the bar, and a
  half-empty small one can. Hovering a bar says how many more datasets fit.
  The bars no longer wait for your first detector, and **View** / **Hide**
  still show them always or never.
- **The precision floor is now the Threshold, a spectrum with three radios**
  (issue #4317). The **Lean: Complete / Centered / Correct** pulldown in Train
  and Find is gone. In its place, **Threshold:** heads a spectrum from **False
  Positives** to **False Negatives**, with a radio button under each third of
  it; the radios carry no words or numbers, and hovering one says what it
  does. The floors behind them are unchanged (10, 50 and 90%, the middle one
  the default). The **?** beside it is two short sentences, not a paragraph,
  and the note under it no longer names the floor: it reads *Top 32 kept,
  unchecked*, *Confirmed · …* or *Fell short · …*. Find no longer offers
  **Check 5 picks**: Find tests the threshold you set in Train, so the spot
  check lives in Train only. The Find **Stats** chart calls its floor line
  **Threshold**.
- **Each Autopilot phase has one progress light** (issue #4319). The current
  phase's marker is now a single light: a red circle, a yellow one past
  halfway, then a green check once the phase finishes, lined up with the
  checks the finished phases keep. The Find phases are paced by
  their vote target, Refine Boundary shows the lower of Smart and Stable
  (replacing its two dots), and Explore Diversity is paced against its
  coverage goal, now shown as *Diversity: 13/40*. The labeling-status span
  indicator reports that goal as `target`.
- **Required fields you haven't filled in are outlined in red** (issue
  #4311). A form that stars a field as required now marks the box you answer
  in as well: until it holds a value, it has a red border over a faint red
  tint. It looks normal again while you type in it, and stays normal once
  filled. This covers every plugin form (Add Dataset, Import Labels, Export,
  AutoRun, and the Settings importer and exporter), the drop zones for a
  required folder or file, **Combine Detectors**' name, and **New
  Detector**'s example. Add Dataset's **Folder to import**, always required
  but never starred, now says so. New Detector's **Detector name** stays
  unmarked: the app fills it in from the example.
- **Check the line from the floor control, and see how close it got** (issue
  #4273, the app half of #4272). Beside the precision floor's note, **Check 5
  picks** (29 at Correct) opens the spot check: a random pick at a time
  from the set the line keeps, in the order it was drawn, with no rank or
  score. Vote each one with → / ← or the Good / Bad buttons (↓ goes back to
  change one); the last vote sends the round. A round that falls short says
  *Not there yet: checking a shorter list* and draws a fresh one, and the
  check ends on the result. The note then reads *Confirmed · likely 55–100%
  right (checked 5) · 32 kept*, or *Aimed at Centered: likely 19–92% right
  (checked 5) · top 32 kept*, naming no cause. The Find **Stats** chart stands
  the same likely range at the line where it meets the floor. A range that
  later votes have left stale looks the same everywhere; only its tooltip says
  it was measured before them. The threshold line in the list and on the
  minimap is no longer dashed or marked *unpromised*: it keeps a set in every
  state. Cancelling a check leaves the floor as it was. A Find pass that
  reuses the detector's cached head (re-entering Find, or Find straight after
  training) now draws the floor's set too, rather than the head's old score
  cut, and can be checked.
- **A spot check, not an estimate, decides a detector's line and says how
  close it got** (issue #4272, the backend of #4267; the check's step in the
  app is #4273). Under a precision floor the line now always keeps a set: the
  top of the ranking, sized to the floor - the top 128 unvoted items at 10%,
  the top 64 at 25%, the top 32 at 50% and above. Nothing falls back to the old
  Inclusion 0 cut any more, and no floor empties the results. A **spot check**
  measures the set: you vote on a few random picks from it (5 a round at
  10-50%, 11 at 75%, 29 at 90%), and a bound on those picks either confirms
  the floor or, round by round, halves the set down to the top 32 and reports
  a **likely range** for how much of it is right - from your picks alone,
  never from the model. The check's votes are ordinary votes and train the
  detector. A finished result stays with the detector: later votes move the
  line along the new ranking at the same count, and the range is only marked
  *stale*. The floor's state is now `unchecked`, `confirmed` or `short`, with
  the count kept and the range, in every response that carries a line
  (`/api/min-precision`, the learned sort, `/api/find-label`,
  `/api/auto-detect`, `/api/find/stats`, the CLI's results); the old
  `promised` / `unreachable` / `insufficient_evidence` states and the
  calibration-positive gate are gone from it. New endpoints:
  `GET /api/precision-check`, `POST /api/precision-check/start`, `.../votes`
  and `.../cancel`; see
  [the spot check](docs/api/labeling.md#the-spot-check). AutoRun and
  command-line runs, where nobody can vote, export the unchecked starting
  set and say so (the CLI's `detector_unpromised` event is now
  `detector_unchecked`, with the set's size).

- **Tighter New Detector and Add Dataset dialogs** (#4305). The collapsed
  **Advanced ▾** toggle no longer takes a line of its own above
  **Cancel** / **Create** (or **Import**): it sits at the left end of that
  row, and the options it opens appear at the foot of the form, scrolled into
  view. In New Detector, a typed description now names the detector with your
  own words, first letter capitalised and "detector" on the end ("large
  books" becomes **Large books detector**, "NASA rockets" **NASA rockets
  detector**), instead of title-casing every word.

### Added

- **An AutoRun button on the Dashboard runs the selected detectors on the selected datasets**
  (issue #4529). A third big button beside **Train** and **Find**, enabled on Find's rule (a
  dataset and a detector ticked, one media type, every detector trained), starts a background
  AutoRun on every ticked dataset with every ticked detector, loading a dataset first if
  needed. Drafts run as they are, without moving to the AutoRun tab. The first run to finish
  opens the AutoRun Results dialog and later ones offer **View results**. The API route behind
  it, `POST /api/datasets/registry/<id>/autorun`, takes an optional `detector_ids` list; without
  it, the dataset ⋯ **Run AutoRun** still runs the AutoRun tab.

- **A deployment can list its own docs in the Help modal** (issue #4310). An
  operator who adds plugins or extensions points users at the docs for them
  with a new `docs_links` key in `data/settings.json`: an ordered list of
  `{"label": ..., "url": ...}` objects. The Help modal lists them under **Docs
  for this server**, above the *Email us* line, on every tab; each opens in a
  new browser tab. A URL must be an absolute `http(s)` URL or a `/path` on the
  same host; an entry that isn't, or has no label, is left out, and the startup
  log names it. Read-only over the API, like the other operator settings.
- **AutoRun detectors really run on what you import, and on demand** (issue
  #4252). The Dashboard's AutoRun tab and the user guide promised that AutoRun
  detectors run on every imported dataset, but only the CLI's `--autodetect`
  ever ran them. Now a web import runs the importing user's AutoRun detectors
  on the new dataset once it is saved: the run shows on the dataset's row
  (cancellable), its results go to the Auto-Find exporter if one is set, and a
  notice with **View results** opens them. The Add Dataset dialog has a **Run
  AutoRun detectors on this dataset** checkbox (shown once you have an AutoRun
  detector) that remembers how you left it, as the new `autorun_on_import`
  setting. A dataset's ⋯ menu gains **Run AutoRun**, which runs them on an
  existing dataset right away and opens the results when done. The **AutoRun
  Results** dialog (the old Auto-Detect Results dialog, which nothing opened)
  now has a working **Export** button that sends the rows it lists.

- **Detectors draw their line at a precision floor: "show me what's at least
  half right"** (issue #4245, the backend of #4224). A detector's line is now
  the cut that returns as much as it can while at least a set share of it is
  estimated right. Every detector starts at **50%**, taken from your new
  `min_precision` setting, and each keeps its own. The estimate is cautious: it
  only promises once it has seen about ten positives among the votes it holds
  back to check itself, and it only counts votes you made off the learned sort
  (Autopilot's Hard picks, or working down a list sorted by the detector). Until
  then, or when no cut can reach the floor, the line stays exactly where
  Inclusion 0 puts it, so nothing you see empties or jumps. A set floor wins
  over Inclusion: while the Inclusion stepper is still on screen it does not
  move a floored line. Clear the floor (`POST /api/min-precision` with `null`,
  or `min_precision: null` in `PUT /api/settings`) to get the stepper back. The
  on-screen control that replaces the stepper arrives in #4246. New endpoint:
  `GET|POST /api/min-precision` (removed with the floor, #4421).

- **An unpromised line says so** (issue #4247). When a detector can't yet
  promise its precision floor - too little evidence, or no cut on the
  dataset reaches it - its line stays at the Inclusion 0 cut and is now
  labelled: the threshold line in the Find and Label lists is dashed and
  reads *unpromised*, the minimap marker is dashed too, and hovering says
  why. Nothing stops working on it: the Unverified Good count, the Find
  review walk, To Dataset, Export and Browse all act on the items above the
  line as before. AutoRun and command-line runs export the same set and
  record that it was unpromised, in the log and as a `floor` entry beside
  each detector's `threshold`. Every response that carries a detector's
  line (the learned sort, `/api/find-label`, `/api/auto-detect`) now
  carries that `floor` too (replaced by `balance`, #4413, #4421).

- **Step-by-step how-to pages, readable in the Help panel.** Seventeen new
  pages under `docs/user/howto/` each walk through one task click by click,
  in the style of the user guide's *Step by step*, on the same Synthetic
  Media drawings and `Yellow Smileys` detector: checking and correcting Find's
  calls, borderline matches and Inclusion, how far to trust a detector,
  exporting matches, starting from an example picture, region voting,
  getting Autopilot unstuck, Manual mode, moving a detector, importing
  labels, AutoRun from the command line, combining, Browse, dataset and
  detector stats, import options, demo datasets, and saving settings. The
  guide lists them under **How-to guides**, and the in-app Help panel now
  opens a linked page in place, with **← Back** to return.

- **Synthetic Media draws cartoon smiley faces, and takes a Seed** (issue
  #4240). The **Demo → Synthetic Media** image generator used to draw one
  smiley or a few flat shapes on a plain background. It now draws round
  cartoon faces in seven colours and seven expressions, piles of shapes and
  busy little scenes, on plain, polka-dot, striped, checked or gradient
  backgrounds, with enough near-misses (a frowning yellow face, a smiling
  orange one, a yellow disc) that "find the yellow smiley faces" is a real
  search. A new **Seed** field picks which set is made: the same seed always
  makes the same media, and two seeds make two sets with nothing in common.
  A synthetic dataset imported before this keeps its old pictures; import
  Synthetic Media again for the new ones.

- **`--create-detector` makes the detector `--import-labels-into` names**
  (issue #4238). A label file and a dataset are now enough for a headless run:
  `--autodetect --import-labels-into NAME --create-detector --label-importer-file …`
  creates NAME from the imported labels when it doesn't exist, then scores the
  dataset with it. The detector is registered like one made with **New
  Detector**, so it shows up on the Dashboard's Drafts tab for the user the run
  ran as. Its media type comes from the source (a pickle's recorded type, or the
  importer's `--media-type`), or from `--detector-media-type`. Without the flag
  a missing detector still fails, now saying which flag would create it.
  Pipeline files take `import_labels.create: true` and
  `import_labels.media_type`.

### Removed

- **The precision floor** (issue #4421). The precision/recall balance
  (#4413) is the only preference the line is drawn at, so the floor it
  replaced goes now rather than a release later. Removed: the `min_precision`
  and `line_preference` settings (a saved settings file that still holds them
  loads as before and the keys are ignored), `GET|POST /api/min-precision`,
  and the `floor` object every response that carries a detector's line sent
  beside `balance` - the learned sort's result, `/api/find-label`,
  `/api/find/stats`, the auto-detect, AutoRun and command-line results, and
  every `/api/precision-check` verb. Those carry `balance` alone, and the spot
  check runs only the balance's walk (its `check` object has no
  `min_precision`, and its status is `running`, `checked` or `cancelled`).
  `GET|POST /api/balance` no longer reports `line_preference`, and the
  OpenAPI spec now types `POST /api/balance`'s `beta` as a number.

### Fixed

- **A structural detector's Threshold line says how many pass its gate**
  (issue #4505). On a `sift_vlad` / `sift_vlad_doc` collection the line is
  the verification gate's boundary, but the state under the Threshold control
  read "Top 32 kept, unchecked" with a yellow dot however many items the gate
  kept. It now reads "N pass the verification gate", with no dot, since no
  spot check applies there; Find's stats chart says the same. The balance
  state every response carries reports `status: "gate"` with that count.
- **A structural detector no longer offers a spot check it cannot run**
  (issue #4489). On a `sift_vlad` / `sift_vlad_doc` collection the line is
  the verification gate's boundary, not a cut on a ranking, so there is
  nothing for a check to walk: **Check 5 picks** used to show in Train and
  fail every time ("No ranking to check"). The Threshold control now leaves
  it out wherever a check cannot start, and the balance state every response
  carries gains `checkable`.
- **On a small dataset, a spot check under the precision/recall balance no
  longer keeps every unvoted item whatever your picks said** (issue #4424).
  When the unvoted items numbered no more than the check's starting count
  (32, or 128 when leaning toward recall), the check ended on all of them at
  its first verdict. It now compares that set with the smaller one it has
  already audited and keeps whichever scores better. The smaller set wins
  a tie. Its recall is also no longer reported as everything found when
  the score model counts no match left among the unvoted items.
- **The media list no longer leaves a gap beside its last column of
  thumbnails** (issue #4347). Train fits the left panel to its thumbnail grid
  as it opens, but it measured the grid before the minimap strip beside it
  appeared. The strip then took its width from the grid, so the panel often
  showed one column fewer than it had room for, with empty space where the
  missing column should be. Dragging the divider fixed it only until your
  next visit to Train. The minimap now holds its place, as an empty strip,
  until there is a ranking to draw, and the list keeps room for its
  scrollbar before it needs one. Separately, the first time you open Manual
  in a new install (Train starts on Autopilot, which has no grid), the panel
  is now fitted to the grid as soon as Manual shows it.
- **The stall watchdog can no longer crash the app** (issue #4345). When the
  heartbeat missed its 1 s threshold, `faulthandler` dumped every thread's
  frames from a thread that holds no GIL, while those threads kept running.
  During a CPU import that read a frame another thread was popping, and the
  app segfaulted partway through the dump, so a harmless 1 s GC pause became
  a lost import. The watchdog now takes every thread's stack itself, holding
  the GIL, the moment the heartbeat wakes. It writes them just above the
  `stall:` line as before, with the thread that burned the most CPU first,
  and the stall report is unchanged. `faulthandler`'s dump from *during* the
  stall is still available as `VTSEARCH_STALL_LIVE_DUMP=1` for a diagnostic
  session that accepts the risk, and the startup `diagnostics config:` line
  says whether it is on. The screenshot harness no longer turns the watchdog
  off.

- **Train switches to the detector's own ranking even when the text-hint sort
  is slow** (issue #4326). Entering Train with a detector that already had
  Good and Bad labels, but too few Goods on this dataset to move Autopilot
  past its first step (labels made on another dataset, say), could leave the
  list on the text hint's ranking. It happened when the hint's sort was still
  running as the window noticed the detector was trained, or when the last
  session had ended on the Learned sort. Both now end on the detector's ranking.
- **Opening Train shows the same first item every time** (issue #4318).
  Entering Train with a detector that already had labels could show one of
  several items, depending on which of the sorts it starts on entry answered
  first. Autopilot's text-hint sort could even land after the detector's own
  ranking and replace it. A newer sort now always wins over an older one
  still running, and until you click, vote or step, the first item follows
  the ranking the window settles on.
- **Undo gets you out of "Nothing left in this ranking" in the New select
  mode** (issue #4312). Once New had no unseen items left to offer, Cmd/Ctrl-Z
  undid the vote but left the message on screen. The undone item now comes
  straight back, as it already did under Top and Hard, and your next vote
  checks for unseen items again.
- **`↓` then `↑` returns you to the item you were on** (issue #4306). `↑`
  re-ran the advance instead, and the ranking has often moved since the item
  was picked (in Train the re-sort a vote triggers lands after it; in Find each
  advance switches sides of the cutoff), so it landed on an item you had not
  seen. `↑` now goes back to where the first `↓` started, and only takes the
  usual advance once you have voted or picked something else in between.
- **Voting in Train no longer flashes the item you just voted on** (issue
  #4307). In the New select mode, and in Autopilot's Explore Diversity phase,
  the next item is fetched from the server after each vote. While that
  request was out, the item you had just swiped away slid back into view,
  then snapped to the next one. It now stays off-screen until the next item
  arrives.
- **The Inclusion stepper no longer jumps the line early in a session.** With
  too few votes for the calibration splits, the first change of the stepper
  replaced the trained cutoff with a fixed 0.5, so the matches could change in
  either direction, even shrinking on a step toward lenient. The line now stays
  put until there are enough votes for the stepper to move it.
- **Find with a detector that isn't loaded now uses your Inclusion and
  calibration settings.** It always cut at Inclusion 0 with two calibration
  splits, so the same detector could return different matches depending on
  whether it happened to be loaded.
- **Changing Inclusion on one detector no longer moves another detector's
  line.** Switching to a detector afterwards showed its own Inclusion value
  over a line cut at the value you'd set elsewhere.

- **Combined detectors appear on the Dashboard.** **Combine selected
  detectors** wrote the new detector but never registered it, so it showed up
  nowhere, and trying the same name again failed as taken. It now lands on the
  Drafts tab like any new detector.

- **"Dropped N item(s) whose embedding failed" now says which items, and why**
  (issue #4232). The warning ended with "See the server log for which embedder
  declined", which a GUI user has no way to do. It now names the embedder and
  what went wrong (it returned no vector for those items, crashed on the batch,
  returned the wrong number of vectors, or none is installed for the media
  type), and its new **Details** button lists every dropped item by file name,
  with **Copy list** to put the list on your clipboard, one item per line.

- **Find no longer runs a detector hidden on the other Dashboard tab**
  (issue #4228). With a single detector on the AutoRun tab and none in
  Drafts, the Dashboard selected that detector even while Drafts was
  showing, and reselected it on every registry refresh after you switched
  away from AutoRun, so Find stayed enabled with nothing visible selected.
  Moving your only draft to AutoRun did the same. The detector selection
  now only ever holds rows on the visible tab: Drafts stays empty, Find is
  disabled until you select a detector you can see, and picking a detector
  from the top bar switches to its tab.

- **The folder importer's Browse opens at the folder you typed** (issue
  #4207). In **Add Dataset → Files → Folder**, clicking **Browse** after
  typing a path opened the browser at the server root, replaced the path with
  `/`, and re-ran media-type detection on the whole filesystem. The browser
  now opens inside the typed folder, and opening it leaves the field and the
  detection alone; only navigating in the browser changes them. A typed path
  that doesn't exist opens the browser at the root, still without touching
  the field.

- **Find scores with the detector's current labels, not the ones it had when
  it last trained** (issue #4204). After you changed a detector's labels
  without a Learned sort in between (voting under a text, example or random
  sort, clearing the votes, or flipping labels in the dashboard's saved-label
  review), Find kept giving the old detector's verdicts until the app was
  restarted. A new browser page didn't help, because the cached model lives on
  the server. Find now reuses the cached model only while it was trained from
  the detector's saved labels, and retrains when they have changed. The
  legacy multi-dataset Find does the same. Learned sort's own cache now also
  counts a redrawn region as a label change.

- **Find (and Train) on an unloaded detector waits for the load to finish**
  (issue #4187). The route sometimes opened as soon as the detector load
  started, showing an empty page and a stream of "Detector is not loaded"
  errors until you went back and pressed Find again. A new loading task used
  to be published as "idle" for an instant before it reported "loading", and
  the navigation guard read that first frame as the load already finishing.
  The guard also now ignores a finished or failed row left on the progress
  channel by an earlier load of the same dataset or detector, so a retry soon
  after a cancelled load is no longer refused.

- **Pipeline files and `--import-labels-into` accept any label importer's
  fields** (issue #4174). `import_labels:` required a `file:` and passed it as
  the importer's only field, so a label importer that reads no file could not
  be driven from the CLI. `import_labels.importer` now takes the same
  `{name, fields}` mapping as `importer:` / `exporter:` (the old
  `importer: <name>` + `file:` form still works), and the flag CLI gains a
  repeatable `--label-importer-field KEY=VALUE`.

- **Sorting a SIFT/VLAD dataset by several examples now geometrically
  verifies against every example** (issue #4161). The example sort ran its
  Stage-2 RANSAC re-rank only when given exactly one example; with two or
  more it silently fell back to the VLAD centroid, which on a structural
  dataset ranks at chance. Every example is now a template and a candidate
  scores as the max over templates, the same rule a detector applies to its
  Good-vote templates, so several crops of one mark, or of several marks,
  each find their own instances. Also `vtscore`:
  `maybe_structural_rerank_example` accepts a sequence of templates
  (a single `StructuralFeatures` still works).
- **A tab left open on a deleted detector or dataset now says so, and can move
  off it** (issue #4086). Requests naming an id the registry no longer lists
  answered 409 "Detector is not loaded", which sent you looking for a load
  that could never succeed. They now answer 404 with `error_code`
  `detector_not_found` / `dataset_not_found` and a message telling you to
  reload. Loading a different detector from such a tab no longer fails on the
  stale one either.
- **Opening Train on a new dataset/detector pair now re-runs the sort you
  left it on, instead of showing a stale one** (issue #4092). The sort
  controls (Sort mode, Select mode, the text query) carried over from the last
  Train session, but nothing re-ran them. You could see "aaa" in the Text box
  above a ranking that had nothing to do with it, or above the previous
  dataset's ranking. The rule now is that the controls carry over and the
  ranking is re-run for the new pair. This applies both when you open Train
  from the dashboard and when you switch pairs in the top bar. If the new pair
  can't run that sort, the controls fall back to Text with your query: Learned
  needs a good and a bad label on the new detector, and "Sort by this" needs
  its own dataset. When Autopilot is running, it still picks the sort, as
  before.
- **Pressing Enter on a Text sort query now hands focus back, so you can vote
  with the arrow keys straight away** (issue #3935). In Manual mode, typing a
  keyword and hitting Enter resorted the left panel and selected a new item in
  the centre, but focus stayed in the query box - and keyboard shortcuts are
  deliberately suppressed while focus sits in a text field, so left/right did
  nothing until you clicked elsewhere. Submitting the sort now blurs the box.
  A query you are still typing keeps focus, as before.

### Changed

- **The precision floor offers three named floors: Lean: Complete, Centered or
  Correct** (issue #4298). The picker read **At least [50%] right**, with five
  percentages to pick from, which claimed a precision the cautious estimate
  behind the line rarely delivers exactly. It now reads **Lean: [Centered]**,
  with **Complete** (the old 10%), **Centered** (50%, still the default) and
  **Correct** (90%). The number is gone everywhere the floor is named: the
  note under the picker says *Confirmed*, *Aimed at Correct: likely 11–73%
  right (checked 5)* or *Top 32 kept, unchecked · aiming at Centered*, and
  Find's **Stats** legend reads *Floor: Correct*. What a check measured - its
  likely range - and the chart's axis stay numbers. A floor
  that is not one of the three - a 25% or 75% picked before, or one set from
  the command line or `POST /api/min-precision` - shows as the nearest of
  them, and the picker moves the detector to it once no sort is running. The
  CLI's `--min-precision` and the API still take any value.

- **The Export window says whether it is sending a detector's labels or
  Find's results** (issue #4079). The same window, with the same
  destinations, opens from the labeling view, the dashboard and Find, which
  read as one list mixing two kinds of exporter. Its title is now **Export
  Detector Labels** or **Export Results**, the button of a destination that
  opens a website says **Open Labels in …** or **Open Results in …** (it said
  **Open Labelset in …** everywhere), and the export messages count *labels*
  or *results* to match.

- **Inclusion is gone as a setting; every detector has a precision floor**
  (issue #4269). The floor replaced the Inclusion stepper in #4246, and nothing
  in the app wrote Inclusion after that, so the setting and its endpoint are
  removed: `GET|POST /api/inclusion` answers 404, `PUT /api/settings` drops an
  `inclusion` key like any unknown key, and `GET /api/settings` no longer
  reports one. A detector whose floor can promise nothing draws its line at
  the Inclusion 0 cut, as before; one that promises draws the floor's. There is
  no "no floor" any more: `POST /api/min-precision` and `PUT /api/settings`
  refuse `min_precision: null` with a 422, a `null` left in a settings file
  reads as the default 50%, and the floor picker's *No floor* entry is gone.
  If you had cleared the floor through the API or the settings file to let
  Inclusion draw your line, your detectors are back on the 50% floor.

- **The Inclusion stepper is gone: pick a precision floor instead** (issue
  #4246). Where the Manual tab and Find's left pane had the -10..10
  Inclusion box, they now read **At least [50%] right**, with **10%**,
  **25%**, **50%**, **75%** and **90%** to pick from. A note under it says what the
  floor is doing to the line: *At least 50% right* with how many items it
  returns, *Can't reach 50% on this dataset*, or *Not enough evidence yet*
  with the Good votes it has - the last two showing the default cut, as the
  dashed *unpromised* line already said. The floor is the detector's own,
  seeded from the last one you picked. In Find, **Stats** draws the floor
  across its precision chart and says whether the line keeps it; the chart's
  "Current cut (incl N)" legend is gone, and so are the unused `sweep` and
  `inclusion` fields of `GET /api/find/stats`, which gains the line's
  `floor`. Inclusion itself is retired too (issue #4269, below). The how-to
  *Catch the borderline matches* now covers the floor.

- **Find Stats charts precision against how many items are returned** (issue
  #4242). The chart that plotted wrong and missed matches at each Inclusion
  stop now reads down the ranked list: for the top N items, on a log-scale
  count axis, it draws the precision VTSearch estimates (a cautious lower
  bound from the detector's own held-out votes, shown once they include 10
  Good ones) and the precision of the items you have checked. A dashed line
  marks the current cut, the line under the chart reads both numbers there,
  and hovering reads them at any count. **Kept rate** now counts only the
  matches you checked, with the count beside it ("7 of 10 checked"); it used
  to count every unchecked match as right, so it read close to 100% however
  the checks went.

- **The Smart indicator measures every detector at Inclusion 0** (issue
  #4243). Smart asks whether the detector is still getting better, by
  re-scoring the recent detectors against your current votes. It used to
  price their mistakes at your Inclusion and measure each at the line it
  showed you. It now counts a false alarm and a miss equally, at the line
  each detector would draw at Inclusion 0, whatever Inclusion you have set.
  Nothing changes at the default Inclusion. This keeps the light steady once
  a precision floor, rather than Inclusion, sets the line (#4224).

- **The User Guide is illustrated with the yellow smiley example** (issue
  #4240). Every screenshot in [the guide](docs/user/USER_GUIDE.md) now follows
  a detector learning to find the yellow smiley faces among Synthetic Media's
  drawings, instead of books in COCO photographs, and *Step by step* says which
  Size and Seed make the very same pictures, so you can follow along without
  any data of your own.

- **`--import-labels-into` runs the detector it imports into, and only that
  one** (issue #4235). Importing labels from the command line used to merge
  them into the detector and then score with whatever was on the settings
  file's Auto-Find list, so the detector you had just labelled only ran if you
  had first opened the UI and moved it to **AutoRun**. Now
  `--autodetect --import-labels-into NAME --label-importer-file …` scores with
  NAME alone, whether or not it is on AutoRun, and nothing else on AutoRun
  runs with it. A pipeline file's `import_labels:` block does the same unless
  the file also lists `detectors:`. `--dry-run` shows the detector under
  `Detectors (1; overrides the settings' Auto-Find list)`.

- **`--autodetect` saves the dataset it imports to the dashboard** (issue
  #4226). A CLI run used to import a dataset, score it, and throw it away. It
  now imports through the same pipeline as **Add dataset**, saves the result,
  and scores that saved copy, so the next time the UI is opened the dataset is
  there (owned by `--user`, or the default user). With no Auto-Find detector
  for the dataset, the import still succeeds and the run exits 0 with a
  `Detection skipped` note, which makes `--autodetect` a plain headless import
  too. **Add `--tempimport` to keep the old import-and-discard behaviour** —
  cron jobs that should not grow the dashboard need it. `--tempimport` implies
  `--autodetect`. `--stream-results` now requires `--tempimport`, since a
  streamed source is never held whole and so cannot be saved. Pipeline files
  follow the same default and take a `tempimport: true` key.

- **The Autopilot "Update Sort Example?" prompt says what it is asking**
  (issue #4200). It used to show the current example and ask whether to keep
  it. It now reports how the sort has gone ("You've clicked 10 times and only
  found 1 positive while sorting based on …") and lays out the two answers
  side by side: **Keep clicking** that sort for the next interval on the
  left, **Supply a different sort** (text or media example) on the right.

- **Structural (instance-matching) search is ~3x faster on both of its hot
  paths** (#3900). Ingest with the `sift_vlad` embedder no longer runs SIFT
  detection at the source's full resolution: detection cost scales with pixel
  count while the keypoint set is capped regardless, so a high-resolution
  source was paying many times over for the same descriptors. Detection is now
  bounded by `VTSEARCH_MAX_STRUCTURAL_DETECT_PIXELS` (default 2 MP, `0` opts
  out) — measured 2.8x faster on a 4.5 MP corpus, and it *improves* the
  verified-pair rate, because the keypoints an uncapped detection spends its
  budget on sit in fine texture that does not survive a rescale. Separately,
  the Stage-2 geometric re-rank now matches the whole shortlist in one batched
  `torch` computation instead of a `cv2.BFMatcher` call per pair — 2.9x on CPU
  with identical ranking, and it is the one part of the pipeline that uses a
  GPU when there is one.

### Added

- **Operators can hide the ETA on import progress bars** (issue #4233). On
  some servers an import's speed is too erratic to predict, and its
  remaining-time estimate could climb from "About 10 sec left" to "About
  45 min left" in a single import. Setting `--hide-ingest-eta`,
  `VTSEARCH_HIDE_INGEST_ETA=1` or `"hide_ingest_eta": true` in the server
  settings file removes the estimate from dataset imports, staging imports
  and labelset missing-media fetches. Those bars still fill and show their
  counts, and every other progress bar keeps its estimate. Settings ▸ Server
  shows whether the switch is on.

- **A friendlier first run on the Dashboard** (issue #4227). An empty
  Datasets or Detectors panel now shows a working **+** inside its "Click + to
  add one." message, with an arrow to the real **+** in the panel header so
  you know where it lives next time. While there are no detectors, the
  **Drafts** / **AutoRun** tabs are dimmed and locked to Drafts, and the
  disabled Combine and Delete icons beside **+** are fainter. Once a new
  detector with no labels is selected next to a matching dataset, a
  "Click Train to teach your new detector." hint points at **Train**. The
  RAM / Disk bars now stay hidden until you have a detector; **Settings →
  Appearance → RAM / Disk bars** switches them to always (**View**) or never
  (**Hide**). In the New Detector dialog the examples no longer assume sound
  ("e.g. large books", "e.g. Large Book Detector"), the hint under the example
  tabs names only what that tab takes, a typed description becomes a
  title-cased "… Detector" name, and Enter in the name field creates the
  detector.

- **A click-by-click walkthrough in the user guide** (#4202). The guide opens
  with *Step by step: your first search* — load a folder of photos, make a
  detector, train it, load a second folder, and Find — with a screenshot per
  step whose red numbered markers show exactly where to click, in order. Small
  pictures of the buttons now sit in the sentences that name them ("click the
  **+**"), in the in-app Help panel as well as on GitHub, and every screenshot
  in the guide now shows real photographs (the slide deck's books example)
  instead of synthetic shapes.

- **Double-click the image to zoom in** (#3934). Looking closer at a borderline
  item meant reaching for the zoom control below the image, which breaks the
  rhythm of keyboard voting. A double-click on the image in the Train / Find
  centre panel now zooms 2x about the point you clicked, the way it already
  does on the Browse map. Repeat to go deeper; because the viewer caps at 5x,
  a double-click *at* the cap returns to fit instead of doing nothing, so the
  mouse alone gets you both in and out. Any rotation you applied is kept, and
  the gesture stands aside while a region draw owns it (Shift held, or the
  Marquee toggle on).

- **Stall diagnostics, on by default** (#3853). A rare 5-20 s freeze during
  labeling in which every in-flight request finishes at once could not be
  told apart from a slow endpoint by the request timer alone. The app now runs
  a heartbeat watchdog that, when the interpreter cannot run it for
  `VTSEARCH_STALL_WATCHDOG_MS` (default 1 s), logs which thread burned the
  wall clock (or that none did, pointing at memory pressure) and has
  `faulthandler` dump every thread's frames from inside the stall; logs any GC
  pause over `VTSEARCH_GC_WARN_MS`; and logs a phase breakdown of the
  learned-sort retrain, the per-vote labelset rewrite, the labeling-status
  replay and a vote rehydrate - plus waits on the locks they share - when
  one exceeds `VTSEARCH_SLOW_PHASE_MS`. `VTSEARCH_LOG_FILE` appends the log
  (and the dump) to a file, and the SLURM launcher sets it under
  `data/logs/`, so the pane scrolling away no longer loses the evidence. See
  `docs/DEPLOYMENT.md` → "Diagnosing a stall".

- **`gc.freeze()` after the model preload** (#3870). A labeling session logged
  a gen-2 collection of ~300 ms every ~2 minutes - 35 of them over 2,400 votes,
  flat with label count - and each one holds the GIL, so whatever was in flight
  froze with it (a vote POST to 333 ms, a learned sort to 351 ms). The pause is
  dominated by the object graph the process starts with: `transformers`,
  `torch`, `sklearn`, `cuml`/`numba` contribute millions of tracked containers
  that every full collection traverses and never frees. The app now freezes
  that graph into the permanent generation once the preload finishes, which
  full collections skip. Measured on the GRID over an otherwise identical
  600-vote run: **ten pauses of 290-360 ms became zero**, and vote POST max
  fell 377 ms → 121 ms. Datasets and detectors load lazily afterwards and stay
  collectable, so unloading one still frees its cycles. `VTSEARCH_GC_FREEZE=0`
  skips it.

- **One switch for a diagnostic session, and a log that says what its bars
  were** (#3853). `VTSEARCH_DIAGNOSE=1` sets the whole set together — INFO
  level, a 400 ms request bar, a 150 ms phase bar, and by the coupling a 75 ms
  GC bar — each as a default, so a variable you set yourself still wins. It
  deliberately does not pin `VTSEARCH_GC_WARN_MS`, since that would bypass the
  coupling. Two sessions in that issue produced inconclusive logs for
  configuration reasons alone: one ran at the shipped 1 s request bar, so a
  600–900 ms vote was invisible to it, and one got lower bars only through
  uncommitted edits to two source files. Every run now also logs a
  `diagnostics config:` line at startup (at WARNING, so a stock deployment has
  it), because a log that omits its own thresholds makes every absence in it
  ambiguous — "no slow requests" reads as *nothing was slow* and as *the bar
  was a second* equally well.

### Changed

- **Every request and phase now reports CPU and GC time beside wall time**
  (#3853). A 4918 ms vote POST that did 12 ms of work and one that did 4900 ms
  are the same number to a `perf_counter` pair, which is why the captured stall
  could be seen but not explained: six of the eight slow votes in that trace
  were slow *alone*, so something released the GIL, and nothing recorded
  whether it had blocked or merely been descheduled. `slow request` and
  `slow phase` lines now carry `cpu=` and `gc=`, and below the slow bar (at
  `VTSEARCH_LOG_LEVEL=INFO`) every request is logged as `request trace:` with
  the same figures, so a diagnostic run can add up a vote cycle instead of
  hunting for an outlier in it -- the shape the remaining felt pauses actually
  have. `VTSEARCH_GC_WARN_MS`, left unset, now tracks `VTSEARCH_SLOW_PHASE_MS`
  rather than sitting at a fixed 200 ms: a collection under the GC bar is
  invisible but still lands inside whatever phase was running, so the old
  default silently inflated phases whenever the phase bar was lowered below it.

- **The Smart indicator no longer flaps on a category that has plateaued**
  (#3832). Smart called the error cost "still declining" whenever a line
  fitted through the last ten models sloped down by more than 1.5% of the mean
  per step - with no notion of how much that cost bounces around between
  retrains. On a category the embedding cannot resolve the cost is flat on
  average but noisy, so the test fired on a quarter to a third of windows with
  nothing having changed about the detector, and Autopilot's phase display
  bounced between Done and Boundary with it. A decline now has to be bigger
  than the window's own scatter (the slope at least two standard errors below
  zero) before it holds the light yellow. A genuine improvement still does:
  a run losing 5% of its cost per step reads yellow on 99% of windows, and the
  synthetic dice reproduction's cleanly separable control reaches Done on the
  same click it did before and never leaves. The indicator's tooltip says
  which kind of green it is, and the status carries `slope_t` beside `slope`.

- **The Stable indicator no longer waits for a category the embedding cannot
  separate, and no longer degenerates as the haystack fills up** (#3831). It
  now counts only *confident* flips - items that sat clear of the cut under
  both the previous detector and this one - and divides by the whole pool
  rather than the shrinking unlabeled remainder. A pool of dice labeled by
  their roll used to hold Autopilot in the boundary phase for 127 of 150
  clicks because the d8s flipped every retrain; it now reaches Done and
  stays there for most of the run, while a cleanly separable category stops
  at exactly the click it did before. When Stable goes green with items
  still wobbling across the cut, the indicator tooltip and the Autopilot
  Done step say so ("the remaining ambiguity looks irreducible in this
  embedding") instead of implying the pool converged.

- **Opening a saved dataset paces its progress bar for the branch its coverage
  atlas actually takes.** That step either restores the atlas cached in the
  dataset's pickle (~10 ms) or rebuilds a hierarchical k-means from scratch
  (0.0026 s/item, so seconds at the sizes anybody has swept and minutes only
  near the auto-build threshold) - measured 110-700x apart on the same
  datasets - and a
  timing profile could hold only one number for both, so whichever it held made
  the other case's bar up to 0.94 of a bar wrong. A profile can now carry
  coefficients for each branch, and the load route names the branch it is taking
  as soon as it knows, before the expensive part starts. It also remembers on
  the registry entry which branch that dataset took last time, so the weights
  are right from the first update rather than from the middle of the load.
  Pacing only: nothing about what is loaded or stored changes.

### Fixed

- **A `PluginField`'s `default` is now a value everywhere, not just on the CLI**
  (issue #3874). A plugin declaring `default=DEFAULT_EMAIL` on a field showed
  an empty box in the GUI, and saving the form without touching the field
  stored a blank - so a default the plugin author could set programmatically
  was invisible to the user and absent from what ran. Three layers disagreed
  about what a blank meant. Marshmallow's `load_default` fires on a *missing*
  key, but a form posts every input it rendered, an untouched one as `""`: the
  plugin-arg schema now drops a blank for a defaulted field before loading, so
  a blank loads exactly as an omitted key does (which also lets a blank
  `number` take its default instead of failing to parse). `normalize_field_values`
  - the only pass a schema-less caller such as the Auto-Find results exporter
  gets - fills a missing or blank field from its default before the required
  check, so a declared default now satisfies `required` rather than raising
  `"<Label> is required."`; the CLI's presence check defers to it too, so
  `--email-address ""` no longer fails where omitting the flag succeeds. And
  the Auto-Find settings tab seeds defaults on every arrival at an exporter
  rather than only on the first pick, so an exporter restored from saved
  settings - or a field that *gained* a default after the exporter was first
  configured - shows and persists it.

- **A plugin's `checkbox` field now renders as a checkbox everywhere, not as a
  text box you were invited to type `true` into** (issue #3851). The SPA has
  ten near-duplicate blocks that pick a widget from a `PluginField`'s
  `field_type`, one per surface that shows plugin configuration. When
  `"checkbox"` was added to `FieldType`, only two of them grew a branch for it
  - so on the other eight (results exporters, settings importers and
  exporters, label importers, the New Model form, Auto-Find, the autodetect
  results modal, the source-specs picker) a checkbox field fell through to the
  generic text input. Every checkbox field shipped in this repo happens to
  belong to a dataset importer, which is rendered by one of the two surfaces
  that worked, so the bug was only reachable from a third-party plugin. All
  ten now delegate to one shared `<vt-plugin-checkbox>`, which draws the box
  inside the field's label (so clicking the label toggles it) and reads
  `default` the way `vtscore.plugins.parse_checkbox` does - case-insensitively,
  so a Python-style `default="False"` no longer renders as ticked.

- **Docker images now ship the `toponymy` signpost labeler** (issue #3852). A
  Docker build never runs `scripts/install.sh`, and that script is the only
  place `toponymy` was installed - it cannot be declared in `pyproject.toml`
  or a requirements file, because its `transformers<5.0.0` pin would drag the
  app's transformers stack backwards and pip rejects `--no-deps` inside a
  requirements file. So every image built from `docker/` shipped without it,
  and VTSBrowse logged `VTSBrowse signposts are disabled: the required
  'toponymy' library is not importable` and rendered every map unlettered.
  Each Dockerfile now carries a dedicated `--no-deps` install step mirroring
  the script, and the two slim requirements files (`labbench.txt`,
  `image-embedders.txt`) declare toponymy's real dependencies on its behalf,
  since those images install VTSearch itself with `--no-deps -e .`. A new meta
  test fails the suite if an image drops the step, installs toponymy with its
  deps, or pins a version the other install paths do not.

- **A dataset imported with some of its vectors already computed no longer
  fails every Browse, Train and text sort afterwards** (issue #3798). An
  importer plugin that ships pre-computed vectors for part of a dataset
  (`content_vectors` / `custom_metadata_map`, with no embedder name) and leaves
  the rest for VTSearch to embed produced a dataset that was one embedding
  space in fact and two by name: the items VTSearch embedded were keyed under
  the embedder it resolved, the shipped vectors stayed nameless. The first
  request for that embedder's matrix then raised
  `media N has no embedding for embedder '<name>'` on every shipped item. This
  happened whenever the import named no embedder - the CLI never does, and the
  GUI need not - and became visible with the mixed-space guard added on
  2026-09-04, which stopped reading a nameless vector as "the same space" on
  the strength of the first media alone. The embed step now stamps the
  embedder it resolved onto the nameless vectors whenever it embeds alongside
  them (or the dataset already carries that embedder's name), so the dataset
  leaves the import keyed under one name. A vector whose width contradicts
  that embedder's declared dimension is rejected at import, naming both widths,
  instead of surfacing later as a missing vector. A fully pre-computed nameless
  import with nothing left to embed is unchanged.

- **An import whose embedder produced nothing now says so.** Items the embedder
  returned no vector for were dropped at the end of the load with one log line
  and a progress message that the next stage overwrote within seconds, so a
  dataset that silently shrank looked like a healthy one. The embed step now
  logs which embedder declined and for how many items (and when a bulk
  embedder returns the wrong number of vectors, or no embedder is registered
  for the media type at all), and the drop raises a warning toast that stays
  until dismissed.

- **Startup no longer stalls for minutes on a cold NFS install** (issue #3715).
  `transformers` builds `importlib.metadata.packages_distributions()` when it is
  imported, and the stdlib version of that stats every file recorded by every
  installed distribution - ~85,000 of them in a venv carrying torch, onnx and
  RAPIDS. With the venv's metadata in page cache that is milliseconds; on an
  NFS-mounted venv whose dentries have been evicted it is 85,000 serialized
  round trips, measured at 78/sec on the GRID: **16 minutes 23 seconds** during
  which the process sat in `D` state after printing
  `Running in PRODUCTION mode` and looked hung. Startup now installs a
  stat-free replacement that reads each distribution's `top_level.txt` (falling
  back to parsing `RECORD` as text) before anything can import transformers, so
  the walk never happens.

- **A mixed-media dataset is no longer scored against another embedding space's
  vectors.** Asking the matrix layer for a specific embedder checked only the
  *first* media to decide whether that name was the dataset's primary - and if
  it was, every media then contributed whatever vector it happened to carry.
  On a dataset holding, say, native images in one space and videos in another,
  that stacked the videos' vectors into the images' matrix and scored them
  through the images' detector head: no error, plausible-looking numbers, and a
  different answer if the same items were loaded in a different order. It was
  reachable whenever the two spaces share a dimension, which is common. The
  check now requires every media to agree, so a request for one space reads
  only that space and reports the media it had to leave out.

- **A CLI Auto-Find run that converts or re-clips its dataset now calibrates the
  threshold on what it actually scores.** The cut was fitted on the loaded
  medias while scoring read the converted frames, pages or clips those medias
  fan out into - and those are systematically different populations, because a
  media's score is the *max* over its sub-items and is therefore never below its
  own whole-item score. A cut chosen as a quantile of the first distribution
  lands far lower in the second, so the run reported more hits than the
  algorithm ever chose, with nothing in the output looking wrong. On a dataset
  the detector needs no conversion or re-clip for - the common case, and the
  only one anyone had noticed - the two populations are the same set and
  thresholds are unchanged. Everywhere else the threshold moves, generally up.
  A pure converter route was the sharper edge of the same bug: none of the
  loaded medias carried a vector in the detector's space yet, so the population
  estimator saw an empty haystack and silently did not run at all, shipping the
  plain cross-calibration cut. Both now read the routed snapshot, which is
  prepared once and shared between calibration and scoring rather than built
  twice.

- **The Dashboard's `⋯` › Export labels now exports the detector whose row you
  clicked.** It exported the *active* pair's live labels instead - whatever the
  top-bar pulldown was on, which the row's checkbox does not change - so the
  same row gave three different answers depending only on where the app had
  been: the right labels when the pulldown happened to agree, nothing at all
  after a refresh left no detector active, and the entire dataset after a Find
  run, because Find fills that pair's votes with the detector's own call for
  every item (they are presumptions, deliberately kept out of the labelset, and
  the export was reading them as labels). The row action now names its detector
  and reads that detector's persisted labelset - the exact artefact that
  re-imports as the detector - so it is right whatever else the app is pointed
  at. The Find and Train windows' exports are unchanged: there the live session
  *is* what you are exporting.

- **The text-sort progress bar no longer reserves most of itself for a model
  load that has already happened.** `text_sort` paces three steps -- load the
  embedder, embed the query, score every media -- and the first one is either
  seconds (the first sort a server process runs) or *exactly zero* (every sort
  after it, since the encoder stays resident). One static weight vector cannot
  serve both, and every vector the app has ever shipped or measured was the
  cold one: [#3521](docs/experiments/2026-09-03-drive-cold-3521/REPORT.md)
  scored both fitted timing profiles and the shipped defaults at 0.80-0.85
  *bar error* on this task -- the fraction of the bar budgeted to the wrong
  step -- while their per-step predictions were off by only ~20 %. So the most
  frequently run bar in the app spent three quarters of itself parked on a step
  that was already over. The sort route now asks whether the encoder is
  resident (it is a lookup, not a guess) and drops the load step out of the
  pacing when it is, which is the overwhelmingly common case.

- **CLI autodetect no longer drops media the GUI keeps, and its detector
  threshold no longer moves as a result.** The CLI forced *thin* loading on
  every import -- storing a path reference in place of each media's bytes --
  while the GUI passed the user's "Reference files in place" choice. Thin is
  only a deferral when something outside the source can reproduce the bytes; a
  self-contained pickle, or an importer whose items have no local file, lost
  its only copy. Those media could not be embedded, so they were skipped at
  scoring, and because a detector's threshold is calibrated against the
  population actually being scored, the smaller survivor set also moved the
  cut. The same dataset and detector therefore returned more hits, at a lower
  threshold, from the CLI than from the GUI. Two fixes: the pickle loader now
  keeps inline bytes that nothing can re-read (thin still drops bytes that are
  also on disk, in an archive member, or behind a URL), and the CLI honours the
  importer's own `--reference-files` / `--no-reference-files` flag -- which
  previously did nothing at all -- defaulting to off, as the GUI does.

- **Staging a dataset for the combine flow now queues behind the same
  concurrency limits a regular import does, and a cancelled staging no longer
  reports itself as a failure.** Staging ran its importer and its embed pass on
  a bare background thread that acquired neither the download nor the embed
  gate, so N stagings fetched and embedded fully in parallel with each other
  *and* with gated imports -- exactly the pressure the
  `max_concurrent_dataset_downloads` / `max_concurrent_dataset_embeddings`
  settings exist to bound. On a RAM-constrained host, staging a handful of
  image datasets at once multiplied resident model weights past the configured
  budget with nothing to stop it. Staging now makes the same download -> embed
  gate handoff as an import, and shows the same "Waiting for other datasets..."
  message while queued. Separately, staging's error handling had no branch for
  a user-requested cancel, so stopping one surfaced the raw exception text and
  the dashboard and toasts read it as a red failure; both import paths now
  share one exception taxonomy and a cancel reads as `Cancelled` either side.

- **Autopilot opens on the detector you already trained, instead of on its text
  hint.** Whether a detector arrives already trained -- and so whether Autopilot
  ranks with the model from its first screen instead of seeding from the text or
  example hint -- was decided the instant the Autopilot panel mounted. On entry
  to the Train window that instant is always two round trips before
  `/api/votes` can answer (the window ends any live Find session before it reads
  the votes), so the labelset counts it read were the not-yet-loaded zeroes and
  the answer was always "untrained", however trained the detector was. Only the
  Manual -> Autopilot tab switch, which rebuilds the panel with the counts
  already loaded, ever got it right. The mount-time reading is now a guess that
  the run's first *real* reading of the labelset corrects, and the ranking
  follows that correction. It matters most for a detector trained on one dataset
  and opened on another, where there are no votes to move the phase and so
  nothing else would ever have corrected it. Two smaller things read the same
  flag and were deciding from a value that was false for reasons of timing
  rather than of fact: the re-sort prompt, and the sort mode Autopilot leaves
  selected when you stop it.

- **Find now calibrates each detector against the corpus it is searching, and
  scores each head the way that head was trained.** Two independent faults met
  in the cross-dataset Find route. A detector that is not loaded against the
  dataset being searched is retrained on the fly, and that retrain was handed no
  haystack -- so its Good/Bad line came from the pooled cross-calibration rule,
  which caps false negatives and only floors false positives, i.e. sits below
  the cut the data supports and returns more matches than the detector actually
  makes (on a synthetic 2k-media patch corpus it called all 2000 a match, where
  the population-fitted cut called 728). The corpus was in memory the whole
  time; it is now what the threshold is fitted on. Separately, both scorers read
  one image-level vector per media, which is right for the on-the-fly head
  (trained on image-level label vectors) but wrong for an already-loaded one,
  whose threshold was calibrated on max-pooled patch scores -- so on a patch
  dataset a loaded detector was compared against a line drawn for a higher
  quantity and under-returned. Each path now scores at the geometry its own head
  was trained and calibrated in. A media with no usable vector in the scored
  space is also reported as `N/A` instead of falling out of both result tables.

- **Switching dataset or detector in the Train window now re-ranks the pair you
  land on.** Opening the window seeds a ranking -- Autopilot activates, its
  seed sort runs, and an item lands in the centre. A pair *switch* re-ran none
  of that: it reloaded the media list and the votes and left the new pair with
  the empty ranking the reset had just installed, so you arrived at an empty
  work queue and a placeholder centre with no way back short of leaving and
  re-entering the window. Two faults had to meet. The reload spends a moment
  with the vote cache cleared, and Autopilot read that moment as "this detector
  has no labels" rather than "the labels have not arrived yet" -- on a dataset
  whose embedder cannot search by text that dropped you into Manual for good,
  and rewrote the sort mode on the way out. And the only re-sort a switch could
  ever produce was an Autopilot *phase change*, which a pair that lands in the
  phase you left it in never fires. Autopilot now waits for the vote read
  before deciding it has nothing to work with, and a switch that nothing else
  ranked falls back to the sort a fresh entry would have run -- learned sort
  when the detector has both label classes, otherwise Autopilot's text or
  example seed. The re-rank never moves the centre off an item you have already
  started labelling.

- **Switching dataset or detector no longer leaves the previous pair's item in
  the centre viewer.** Media ids are per-dataset, so the pair switch cleared the
  ranking, the threshold and the vote cache but left the *selection* pointing at
  an item that only existed under the pair you just left. Because the viewer
  stamps the dataset id into the media URL when it builds it, the stranded item
  kept loading from its own dataset and rendered normally -- so the grid showed
  the new pair while the centre showed the old one, with nothing on screen
  saying so. The selection is now dropped with the rest of the pair's state; the
  centre reads "Select a media item to view" until the new pair's first pick
  lands, exactly as it does on a fresh entry.

- **The "arranging your items" wait now shows a remaining-time estimate.**
  Preparing a subset map from Find results renders an ETA chip beside the
  progress bar, but the projection build's status payload never carried an
  estimate for it to show, so the chip was always blank. Background jobs are
  now backed by the same progress tracker the dataset loads use, which derives
  one from the rate the build is actually sustaining (rebased at each phase
  boundary, published on a coarse sticky ladder so the figure doesn't twitch),
  and the projection status payload passes it through as `eta_seconds`.

### Added

- **New demo dataset: Rico Icons -- screenshots with boxed, labelled icons.**
  Four new image demos (`rico_icons_s/m/l/a`) built on the Rico UI-semantics
  corpus: 66,261 Android screenshots whose annotations sit on the *elements*
  rather than the screen. Each media carries the icon classes visible on it as
  multi-label categories (32 curated semantics -- Search, Back arrow, Overflow
  menu, Notification bell, ...) plus one ground-truth bounding box per icon
  instance. This is the first demo in the tree that boxes something *inside* a
  screenshot: every other born-digital source (Enrico, RICO App UIs) labels the
  screen as a whole, and the only two boxed sources (Visual Genome, OpenLogo)
  are natural photographs. Boxes are stored normalised, exactly like Visual
  Genome's.

  Unlike every other demo, the four size variants advertise **different**
  download figures (~0.9 / 1.1 / 1.5 / 8.3 GB). The corpus's screenshots run to
  ~7.7 GB across 67 shard folders, so the loader fetches the annotation manifest
  first, slices it, and then pulls only the shard folders that slice actually
  lands in -- an (S) load costs two folders, not sixty-seven. Re-loading, or
  moving from (S) to (M), pays only for the shards it adds.

### Removed

- **Four REST endpoints with no consumer are gone.** `GET /api/dashboard/dataset-info`,
  `PUT /api/dashboard/dataset-rename`, `POST /api/dataset/load-folder` and
  `POST /api/votes/seed-from-examples` had no caller anywhere in the app -- the
  SPA reads dataset metadata and renames through `/api/datasets/registry`,
  loads folders through the generic importer flow, and seeds examples as part
  of loading a detector. Nothing in the UI changes. An out-of-repo script
  calling one of them directly will now get a 404; the same work is available
  through `GET /api/datasets/registry`,
  `PUT /api/datasets/registry/{id}/rename`,
  `POST /api/dataset/import/server_folder`, and detector load respectively.
  (Issue #3438.)

- **Old settings-file shapes are no longer migrated forward.** VTSearch used to
  carry three shims for settings written by older versions: a one-shot rewrite
  that split a pre-tier-split `data/settings.json` across the two files, and a
  pair of coercions that read a pre-enum boolean `show_animations` as
  `"show"` / `"hide"`. `CLAUDE.md`'s backwards-compatibility policy allows
  breaking saved data freely and forbids exactly these shims, so they are gone.
  A value the settings models reject -- one written before a field changed
  shape, or a hand-edit out of range -- is now **ignored on load** and the
  field's default applies. Your file is never rewritten, so nothing is lost:
  fix the value and it takes effect on the next start. Concretely, a boolean
  `show_animations` now reads as `"show"` (the default) rather than mapping
  True-ish to `"show"` and False-ish to `"hide"`, and `PUT /api/settings` now
  rejects the boolean with a 422 instead of silently rewriting it. Per-user
  keys sitting in `data/settings.json` are inert rather than migrated into the
  default user's file. The one deliberate tier exception is unchanged: the
  built-in `default` user still reads `autofind_detectors`,
  `autofind_exporter` and `autofind_exporter_field_values` through to
  `data/settings.json`, which is what keeps the CLI's `--settings` flat-file
  workflow working. (Issue #3413.)

### Changed

- **Your Dashboard selection now survives leaving the Dashboard.** Going to
  Train and back used to drop the highlighted rows and blank the top bar to
  "Select a dataset" -- even though the pair you had just opened was still
  loaded -- because the selection lived on the Dashboard component and died
  with it. It now lives in `DashboardSelectionService`, so the rows come back
  highlighted and the top bar keeps naming them. The detector grid's
  Drafts/AutoRun tab travels with the selection it scopes, so returning can no
  longer leave a hidden AutoRun row feeding the section actions. (Issue #3445.)

- **Switching between two loaded detectors no longer throws away the labeling
  indicators' cached work.** The Smart / Stable per-step cache retrains one MLP
  per label-history step, and it used to be a single slot stamped with whichever
  `(dataset, detector)` pair last touched it -- so re-selecting a detector you
  had already been labeling in dropped the other one's cache outright, and the
  next `/api/labeling-status` poll rebuilt it from step zero. The cache is now
  keyed by the pair, with the three most recent kept warm, so an A-to-B-and-back
  switch costs nothing. The stability pool tensor -- the largest thing the cache
  holds -- is keyed by dataset instead and shared across that dataset's
  detectors, so keeping several pairs warm does not multiply memory. (Issue
  #3390.)

- **"Enrich descriptions" is now a per-model choice, and can no longer make
  your search worse.** The setting averages a typed query over several
  phrasings ("the sound of a dog", "a recording of a dog", "a dog", ...)
  instead of embedding it as typed. Measured across 22 evaluation collections
  and 560 queries, that helps on some models and hurts on others: it is a gain
  on CLAP General (audio) and X-CLIP (video), and a loss on SigLIP (images),
  E5 and BGE (text) and the faster CLAP -- for text it ranked *worse on all 45
  categories tested*. The phrasings are now attached to the models they
  actually help, so turning the setting on is simply a no-op for the rest,
  rather than a small silent cost. The default stays **off**. (Issue #3341,
  following #3127.)

- **The Settings -> Sorting "Enrich descriptions" tooltip described a
  different feature.** It read "Prepend item filenames to text-sort queries to
  improve matching for named items"; the setting has never touched filenames.
  It averages the text-sort query over the embedder's phrasing templates
  ("a photo of ...", "the sound of ...") instead of embedding it as typed. The
  tooltip now says that, and says where it helps: the #3127 measurement across
  every media-type default found it worth +0.014 AP on CLAP audio search,
  inert on images, and -0.057 on text search -- so the default stays **off**.
  (Issue #3127.)

- **The calibrated cutoff no longer counts your own votes in its picture of
  the collection.** The threshold estimator reads score distributions over
  the whole collection to place the Good/Bad line; the voted items' scores
  in those distributions are optimistically shifted (the models were trained
  on them), so they are now dropped before the line is placed. On large
  collections nothing visibly changes; on small ones (demo-sized, or heavy
  voting) the cutoff gets slightly more accurate. The correction steps
  aside when almost everything has been voted — a tiny leftover pool is a
  worse guide than the full collection — so no regime pays for it.
  (Issue #3308.)

- **The Tuning-fraction default is now per-model.** With no explicit setting,
  the Train/Calibrate split of each calibration fold is 0.3 (70% Train / 30%
  Calibrate) for detectors that learn in a single-vector space and stays 0.5
  for patch-grid models — the #3287 measurement found more Train buys
  −0.012 to −0.013 cost on single-vector embedders in every vote band, while
  patch embedders want the incumbent 0.5 in both their voting styles. The
  Settings → Sorting "Tuning fraction" field now reads empty ("auto") by
  default; typing a value pins it for every detector as before, and clearing
  the field returns to the automatic per-model default. Stored settings that
  already carry an explicit `calibration_fraction` keep winning unchanged.
  (Issues #3287/#3290.)

- **The calibration deck is a talk about ideas again, and every reveal has a
  name.** `slides/decks/hold-the-line.deck` was rebuilt end to end: the
  measurement slides are parked after the Questions slide as backup, so the
  main line is what we found rather than how we ran the study; the two slides
  that argued the threshold's double duty in the abstract are gone, and the
  point is now made on the field of items where it can be seen — one detector
  curve, then the same curve cut looser and tighter, then the item the loose
  and tight cuts disagree about, which is also the item the user gets asked. A
  new slide sets up what the Inclusion knob is *for* before the section that
  shows it failing, and the epilogue's two compressed bullets became two
  slides: why the midpoint of two means is not the Bayes-optimal cut, and why
  a region-voted score is an extreme-value statistic. Every page of a build now
  carries a letter after the shared page number (5a, 5b, 5c), the speaker view
  shows the whole build as a lettered contact sheet under the slide, and
  presenter notes that would have been clipped spill onto a continuation page
  instead.

- **The Add Dataset dialog has one consistent vertical rhythm, and Advanced
  really is hidden.** The Folder, Manifest and Demo forms all laid their fields
  out differently: a field's own controls could sit further apart than two
  unrelated fields, the "Folder to import" path and its **Browse** button were
  separated as if they were different questions while the checkboxes below ran
  together with no gap at all, and the folder browser opened as loose rows with
  no frame. Every field in every importer now shares the same spacing, the path
  input and **Browse** sit on one line (matching the Manifest importer's file
  field), and the browser opens in a framed panel under it. Separately, the
  **Advanced** section now shows *nothing* until you open it: **Embedder** and
  the Demo importer's **Convert to** used to appear on the collapsed form
  whenever their value differed from the default — which, on a demo dataset
  that picks its own embedder, was most of the time. Any non-default choice is
  still disclosed, in the **Advanced** toggle's tooltip. (#3215)

- **Autopilot's "you're done" hand-off is a dialog you answer, and it appears
  once.** Reaching the end of training used to raise a toast that counted down
  and then returned you to the Dashboard unless you cancelled it — and because
  the countdown re-armed on every entry to the Train window, anyone who thought
  their detector needed more work had to dismiss the same redirect each time
  they came back. It is now a **Detector Trained** dialog with two plain
  buttons, **Continue Training** and **Head to Dashboard**, and nothing happens
  until you pick one. It is also raised only for the autopilot run that
  actually trained the detector: continuing afterwards, or picking up a
  detector already trained on another dataset, no longer announces anything.
  (#3201)

- **The detector head is now a linear SVM.** Every trained detector — new
  detectors, saved ones re-derived from their labels, and the per-step models
  behind the labeling-progress indicators — is fitted to the class-balanced
  maximum-margin boundary between your Good and Bad votes (scikit-learn's
  `LinearSVC`) instead of by logistic regression. It is the same shape of model
  as before, a single linear boundary over the embedding space, so nothing
  about detector files, exports, or the ONNX bundle changes: only where the
  boundary lands. Measurements in a separate environment put the SVM's ranking
  clearly ahead of the logistic head's, and while *why* is still under
  investigation, the best-measured head is the one that ships. Detector scores
  will move — a detector retrained after this change can put items in a
  different order and cut at a different threshold than the same votes did
  before. The regularisation strength is tunable via `VTSEARCH_SVM_HEAD_C`
  (default `1.0`), and `VTSEARCH_TRAIN_EPOCHS` / `VTSEARCH_TRAIN_PATIENCE` no
  longer affect a detector fit. See [`docs/ML.md`](docs/ML.md).

### Fixed

- **A results exporter's `dynamic_options` select is now filled in, in both
  places an exporter is configured.** A `"select"` field declared with
  `dynamic_options=True` is supposed to have its options computed at runtime by
  the plugin's `get_field_options()`, which is how an exporter whose
  destinations are only knowable then (mailboxes, buckets, remote queues) offers
  real choices. Exporters never had the route the importer families have, so the
  dropdown rendered the (usually empty) list frozen into the plugin definition —
  the only way to get options was to hard-code them at definition time. Both the
  Export modal and Settings › Auto-Find's **Results Exporter** tab now fetch the
  live list (`POST /api/exporters/field-options/<name>`), re-fetch a field when
  something it declares in `depends_on` changes, show fetch failures inline, and
  honour `allow_free_text` by rendering a combobox. (Issue #3360.)

- **One file a host won't serve no longer sinks a whole multi-file demo
  download.** The Apollo 11 Mission Audio demo pulls its tracks as separate
  files from the Internet Archive, which serves each one from a data node that
  can answer HTTP 500 for minutes at a time while its siblings stay healthy.
  A single such track spent its six retries, then failed the entire dataset
  load with a raw `500 Server Error` naming a `dn721903.ca.archive.org` URL the
  user had never asked for. A file the host refuses is now set aside, retried
  once after the rest of the set (by which point a wobbling node has usually
  recovered), and otherwise skipped with a note in the progress line — the
  download only fails when more than a quarter of the set is missing, and a
  skipped track is fetched the next time the dataset loads. When a server does
  keep returning 500/502/503/504/429 until the retries run out, that now reads
  as *"archive.org kept returning an internal server error (HTTP 500) on all 6
  attempts. That is a problem on the server's side, not yours…"* rather than a
  raw `HTTPError` ending in a CDN node URL. (#3227)

- **A demo download that can't reach its host now says so in a sentence, and
  tries harder before giving up.** Loading the Apollo 11 Mission Audio demo
  while archive.org was refusing connections failed with a hundred-line
  `MaxRetryError` traceback pasted into the Dataset-load-failed box, after
  six identical 10-second connection attempts — and the progress bar said
  "resuming" a file that had never downloaded a single byte. The failure now
  reads *"Couldn't reach archive.org: the connection timed out before the
  server answered, on all 6 attempts. The site may be down or blocked by your
  network/proxy…"*, the connect budget escalates across retries (10 s → 30 s)
  so a host that is merely slow to accept isn't written off six times over,
  the retry notice says "retrying" until there are bytes to resume, and the
  track-list fetch that precedes the download gets the same retry budget
  instead of dying on a single unlucky handshake. (#3216)

- **CLI autodetect no longer reports every media as a hit, and no longer
  disagrees with the GUI's Find.** A run against a dataset and detector that
  the GUI cut at ~0.475 came back with a threshold of `-0.375` and a positive
  hit for every image. Two separate defects were behind it. First, a media the
  detector head cannot score (a corrupt vector, a destabilised fit) is recorded
  at the `-1.0` sentinel — deliberately outside the sigmoid range so it can
  never clear a threshold — and those sentinels were being fed to the threshold
  estimators as if they were scores. A spike a full unit below the range pulls
  the fitted cut under zero, at which point every real score clears it: 2%
  unscorable media moved a threshold from `+0.58` to `-0.50` and a 30-of-600
  shortlist to 588 of 600. Every estimator now fits on the scorable population
  only — fold haystacks, held-out anchors, the pooled conformal orderings, and
  the blend's GMM — and the run warns, naming the count, when media had to be
  excluded. Second, CLI scoring forwarded each media's image-level vector while
  the threshold it compared against (and the GUI) max-pool the media's patch
  rows; on a patch dataset that is a different distribution, enough to turn a
  GUI Find's dozens of hits into zero CLI hits. Both now build their rows
  through the same builder. (#3180)

- **One image that fails to embed no longer aborts a CLI run.** Every CLI
  scoring path fed the raw media chunk straight to the embedding-matrix
  builders, which raise on a media with no vector (`ValueError`, from
  `vtscore/embedding/matrix.py`) or one whose vector is the wrong width
  (`MismatchedVectorError`, from `vtscore/embedding/precomputed.py`). Those
  raises are correct for the dataset's own matrix — the load pipeline has
  already dropped vector-less media by then — but not for a snapshot handed to
  the scorer, so a single corrupt or undecodable file took the whole run down
  with it. The scoring paths now filter first and score what is left, matching
  the drop-and-log policy the load pipeline and converter routing already used,
  and each skip is announced on the CLI event stream (`medias_skipped`) so a
  short hit count is never silent. This also fixes
  `--autodetect --importer …`, which failed on *every* run: the safe-threshold
  population pass scored the chunk before anything had been embedded, so it hit
  the same raise on the first media. (#3179)
- **Rotated photos are no longer processed sideways.** A JPEG carrying an EXIF
  orientation tag — the ordinary output of a phone camera, which stores the
  sensor frame plus "rotate me" rather than rotating pixels — reached the
  embedder, OCR, face detection and every crop path 90/180/270 degrees from the
  way the browser (and every other photo viewer) displays it. Nothing errored;
  the picture was simply the wrong way up in a space nobody looks at directly,
  so a sideways photo embedded as a sideways photo and landed in the wrong part
  of the vector space. Orientation is now applied once, in the image decode
  layer, so every consumer sees the same upright pixels; a media's stored
  `width`/`height` are the displayed dimensions to match, and cropping, lazy
  clip resolution and cross-dataset origin resolution decode upright so a box
  drawn on screen cuts the region the user drew. The `image_exif_orient`
  cleaner still exists for anyone who wants the rotation baked into the *stored
  bytes* (for tools outside VTSearch that ignore EXIF), but it is now off by
  default: it costs a lossy re-encode and no longer changes anything VTSearch
  itself sees. Images already imported keep the dimensions and embeddings they
  were ingested with; re-import a rotated corpus to pick up the fix. (#3172)

- **A detector's labelset no longer stores the same media twice.** After one
  pass over a 300-image dataset a detector reported `num_training: 356` — 300
  distinct images, 56 of them held as a duplicate pair. The two halves of the
  labelset round-trip disagreed on what "the same media" means: loading a
  detector turned an entry into a vote when it matched a dataset item by
  origin **or** content hash, while saving decided an entry belonged to the
  active dataset by comparing origins alone. Anything that matched only on its
  hash — an exemplar carrying the `example_media` sentinel, a label imported
  from a plain md5 list, a label saved under another dataset's importer — was
  restored as a vote, re-emitted as a fresh element, and kept beside its
  original. Both writers now use the same origin-or-hash resolution, so a
  re-vote updates the existing element instead of appending, and a labelset
  that already carries duplicates collapses on the next write. Two related
  leaks went with it: the Find "add corrections to detector" fold left the
  stale entry in place beside its correction (one media, two contradicting
  labels), and a drawn Good region was erased whenever a detector reload
  resynced the labelset, because the restored vote came back image-level.
  Duplicated entries were also double-weighted at training time. (#3174)

- **A finished dataset import no longer looks like a wedged one.** An import
  that had already succeeded — pickle written, dataset registered — kept the
  progress channel parked on its last message ("Loading SigLIP processor…")
  indefinitely, with no loader thread left in the process. Three things fed
  that: a model load taken through the app-wide progress sink narrated itself
  on the dataset channel and never said it had finished, the load task's
  success path wrote no terminal state (only its failure paths did), and
  `POST /api/dataset/cancel` answered `{"ok": true}` while doing nothing,
  because cancellation is cooperative and there was no worker left to observe
  the flag. Model loads now terminate whatever channel they borrowed
  (background warm-ups are silent, and an import's model load reports on the
  import's own row); a load parks its tracker when the work ends, whichever
  way it ended; and cancel reports what it actually reached — `409` with
  `ok: false` when it reached nothing, clearing the stale progress on the way
  out. The dataset registry also re-reads its manifest when the file on disk
  changes, so a dataset registered by another process (a CLI `--autodetect`
  run against the same data dir) is listed without a restart. (#3167)

- **Detector load can no longer hang forever at "Preparing".** Loading a
  detector while the selected dataset was not yet (re)loaded — typical right
  after an app restart, while the dataset was still being read from its
  pickle — left the dashboard row stuck at "Loading detector · Step 1 of 3 ·
  Preparing" indefinitely: Cancel did nothing, every retry reported "Detector
  load already in progress", voting returned 409, and only a restart
  recovered. The load now fails fast with a clean 409 (`dataset_not_loaded`)
  in that case, any other failure before the background worker starts
  releases the load reservation and surfaces an error on the row, and load
  failures are written to the server log. (#3139)

### Changed

- **A seed importer's results now appear on its own tab.** Running a seed
  importer in the New Detector modal used to switch the user to the media
  tab, where the batch had been appended — an odd jump mid-import, and one
  that hid the form they were still working in. The example stack is now
  mirrored under each seed importer's form, so seeds land in view where they
  were added. It stays one list: the mirrored rows carry the same Seed badges
  and Remove buttons, and edits from either tab hit the same stack. Picking an
  exemplar by hand still lands on the media tab, which is where the picker
  lives. (#3192)

- **Image preprocessing now names its backend instead of inheriting one.**
  Every image embedder builds its processor by asking `transformers` for the
  `torchvision` backend outright (`VTSEARCH_IMAGE_PROCESSOR_BACKEND`, new
  default `torchvision`; set `auto` for the previous behaviour). Nothing in the
  code used to say which implementation resized and normalised an image, and
  the answer changed *inside* the version range we pin: `transformers` 5
  removed the `Fast` suffix, so the bare `SiglipImageProcessor` means the PIL
  implementation below 5 and the torchvision one at 5+, while
  `requirements/image-embedders.txt` asks only for `>=4.49`. The two are not
  interchangeable — they disagree on 53–59% of pixel elements and by a median
  `1 − cos` of ~1.5e-04 on `siglip2_l`, 50× the perturbation half precision
  causes — so two hosts resolving different wheels produced different vectors
  from identical code and weights, with nothing recording which. **On a
  `transformers` 5 host this changes nothing** (the pre-embedded pile is
  torchvision-built, reproduced to 7.6e-13). **On a 4.x host it changes the
  vectors**, which is the point: that host was quietly disagreeing with the
  pile and now agrees with it. Because a backend request is a request and not a
  guarantee — DINOv3 ships no PIL implementation, and `transformers` warns and
  falls back rather than raising — each embedder now reads back the class it
  actually loaded and logs a warning naming itself when it differs. (#3173)

- **Bad pre-computed vectors are now rejected at import, with an error that says
  what is wrong.** Importing an `.npz` manifest of pre-computed embeddings used
  to accept anything: vectors of the wrong width for the embedder the manifest
  named, `NaN`/infinite rows from a failed embed, ragged archives, `float64` or
  half-precision exports. None of those failed at import. A wrong-width row
  surfaced later as `could not broadcast input array from shape (768,) into
  shape (1152,)` on an unrelated search, naming neither the file nor the
  manifest; a non-finite row never raised at all and silently corrupted every
  score and threshold it touched. Manifests are now checked as they are read —
  including against the declared embedder's own dimension, so an archive that
  says `siglip2_l` while shipping 768-dim rows is caught immediately — and
  vectors are widened to `float32`, so a half-precision or double-precision
  export imports cleanly instead of leaving the dataset mixed. If a dataset
  still ends up holding two widths, sorting and training now name the offending
  item and both dimensions instead of failing with a bare numpy shape error.
- **Audio now defaults to the larger CLAP checkpoint.** New audio datasets and
  text queries use `clap_general` (`laion/larger_clap_general`, shown as "CLAP
  (general, larger)") instead of `clap`. It wins every measured retrieval
  comparison on ESC-50, at roughly 2.1x the embedding time. The old checkpoint
  is still selectable as "CLAP (general, faster)" for large collections where
  ingest speed matters more, and existing datasets and detectors built with it
  keep working. Cached demo-dataset pickles built with `clap` are re-embedded
  the next time they are loaded with the new default.
- **Library extracted.** The reusable core of VTSearch was carved out into a
  separate `vtscore/` package. The user-facing application surface (the Flask
  app, the Angular SPA, the settings system, the auth layer) is unchanged.
  Internally, every library-candidate import path moved from `vtsearch.<lib>`
  to `vtscore.<lib>`; `vtsearch/state/__init__.py` is now a thin app-tier shim
  that re-exports `vtscore.state` and layers the proxy view (`medias`,
  `good_votes`, …) on top. See
  [`vtscore/docs/architecture.md`](vtscore/docs/architecture.md) for the
  seven seams the refactor introduced.
- **Plugin entry-point groups renamed.** Library-tier plugin families now
  register under `vtscore.<family>` instead of `vtsearch.<family>`
  (`vtscore.importers`, `vtscore.label_importers`, `vtscore.labelset_sources`,
  `vtscore.media_sources`, `vtscore.converters`). Settings-related families
  remain under `vtsearch.<family>` because they stay app-side
  (`vtsearch.settings_importers`, `vtsearch.settings_exporters`,
  `vtsearch.settings_sources`). Third-party plugin authors targeting the
  library tier need to update their `pyproject.toml` entry-point group
  names.

### Added

- **Three long-form audio demos: Apollo 11, BirdVox Full Night, and the Nixon
  White House Tapes.** Every audio demo so far was a corpus of short labelled
  clips (ESC-50, GTZAN, UrbanSound8K) or, in TUT's case, 32 four-minute street
  soundscapes. These three are hours-long *unlabelled* recordings where the
  interesting content is discrete events scattered through the runtime — the
  Quindar beeps, master alarms and MOCR applause in 174 hours of NASA mission
  loops; the sub-second bird flight calls in six ten-hour night recordings from
  BirdVox-full-night; the telephone rings, laughter and room noise under 12
  tapes' worth of Nixon's secret taping system. Each loads as one
  undifferentiated bucket, so you clip it yourself, vote on a handful of hits,
  and let the detector rank the rest. All three sources are freely
  redistributable (CC PD Mark, Creative Commons, and US federal public domain
  respectively).

  Unlike the older demos, **each size variant downloads only its own slice** of
  the source rather than the whole thing — at 5-10 GB apiece that difference
  matters, so (S) costs a twelfth (Apollo, Nixon) or a sixth (BirdVox) of the
  figure shown in [`docs/demos.md`](docs/demos.md). BirdVox's ten-hour FLAC
  units are segmented into 10-minute chunks as they download, since a ten-hour
  file cannot be handed to the clipper as a single item.

- **Seed importers: a new plugin family for unlabeled seed media.** An
  external package can now contribute its own tab to the New Detector modal's
  **Blank** flow, beside Text and the media picker, by registering a
  `SeedImporter` in the `vtscore.seed_importers` entry-point group. Where a
  label importer imports media that already carry a good/bad verdict, a seed
  importer imports a *batch* of media with **no verdict** — items that are
  "close but not quite" what the user is hunting for. Seeds are stored on the
  detector as `{"type": "media", "value": …, "labeled": false}`: they steer
  the first sort (Autopilot ranks against the centroid of every media
  example) but never become a Good label or vote, so a detector seeded this
  way starts untrained. Nothing ships in-tree, so an install with no such
  plugin looks exactly as before. New endpoints: `GET /api/seed-importers`,
  `POST /api/seed-import/<name>`, `POST /api/seed-import/<name>/options`.
- **Server-side code can raise a toast.** A new `notify()`
  (`vtscore/concurrency/notifications.py`) lets any backend code — most
  usefully a plugin that hit a recoverable problem — tell the user something
  happened *without* failing the operation: "skipped 3 unreadable files",
  "the remote API rate-limited us, results are partial". The message is
  broadcast over a new `notification` channel on `/api/events` and rendered
  as a toast; toasts gained `warning` and `info` levels alongside the
  existing `error` and `success`. Plugin subclasses get `self.notify(...)`
  with their display name attached. Headless runs print the same messages
  (stderr in text mode, `notification` NDJSON records under
  `--progress-format json`). Delivery is live-only — there is no replay for
  a client that connects afterwards. See
  [`docs/EXTENDING-plugins.md`](docs/EXTENDING-plugins.md#notifying-the-user-toasts).

- **The app now tells you when your browser is running an out-of-date build.**
  `static/` is a build artifact that git does not track, so pulling new code and
  restarting the server used to leave the browser loading whichever bundle was
  last built — silently, since the version in Settings is the *server's* and
  looks current regardless. The bundle now carries the commit it was built from,
  and a mismatch raises a toast naming both versions and the rebuild command,
  plus a `⚠ bundle v …` chip beside the version in the Settings footer.

- `vtscore` library distribution with its own [README](vtscore/README.md) and
  [CHANGELOG](vtscore/CHANGELOG.md). See the
  [package reference](vtscore/docs/README.md#package-reference) for the
  documented public surface.
