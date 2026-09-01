# DocMarks on the GRID

The full-scale run: every source, all three tiers, embedding cells cached in the
shared pile. Read [`README.md`](README.md) first for what the corpus *is*; this
is only how to build it at size on the cluster.

A tier-`s` SPODS-only build fits comfortably on a laptop or a cloud container
(SPODS is 2.94 GB down, ~8 GB unpacked). What needs the GRID is tiers `m` and
`l`: 50k–200k UCSF pages to fetch and render, and `sift_vlad` local features
over all of them.

## Sizing

Measured or estimated per page: a rendered UCSF page at 150 dpi is ~250 KB PNG;
`sift_vlad` adds 128–256 KB of local features per image in its cell; SigLIP adds
3 KB.

| tier | pages | images on disk | `sift_vlad` cell | `siglip` cell |
|---|---:|---:|---:|---:|
| `s` | 5k | ~1.5 GB | ~1 GB | ~20 MB |
| `m` | 50k | ~13 GB | ~10 GB | ~160 MB |
| `l` | 200k | ~50 GB | ~40 GB | ~640 MB |

Plus raw sources: SPODS 2.94 GB archive → ~8 GB unpacked; the UCSF PDF cache is
roughly the same size again as the rendered pages.

**Put `VTS_DOCMARKS_RAW` and `VTS_DOCMARKS_OUT` on `/expscratch`, never on the
50 GB mount.** `GRID-PLAYBOOK.md` describes that mount as chronically full, and
tier `l` alone would fill it twice over. `df` the experiment path itself, not a
parent — the free space differs.

```bash
export VTS_DOCMARKS_RAW=/expscratch/$USER/docmarks/raw
export VTS_DOCMARKS_OUT=/expscratch/$USER/docmarks/corpus
source scripts/experiments/pile/pile_env.sh    # VTS_PILE, models, HF_HOME
```

## Preflight

```bash
python build_corpus.py --probe
```

Every source fails differently and the probe is the cheapest place to find out
which: a decommissioned hostname (SPODS's own page still advertises the dead
`ernet.in` host), a missing Kaggle token (StaVer, Tobacco800), an absent RAR
extractor, or a UCSF endpoint that is down. Seconds now, a queue slot later.

**The probe downloads nothing**, so it is safe to run repeatedly on a login
node: a `HEAD` for SPODS, a file listing for the Kaggle mirrors, a result count
for UCSF. It also sweeps away any `_probe_*` directories left under
`$VTS_DOCMARKS_RAW` by the version that *did* fetch (~2 GB of duplicated
StaVer/Tobacco800 bytes), reporting what it reclaimed.

**RAR extractor.** SPODS is RAR4. The builder tries `bsdtar`, `7z`, `unar`,
`unrar` in that order; `bsdtar` (libarchive) is the one most likely to be on a
compute node, and reads RAR4 fine. If none is present the probe says so.

**Kaggle token.** `~/.kaggle/kaggle.json`, or `KAGGLE_USERNAME` +
`KAGGLE_KEY` in the job environment. Needed only for StaVer and Tobacco800; a
SPODS-only roster does not touch Kaggle.

Then the standard gate, which checks the things a script cannot:

```bash
bash scripts/experiments/preflight.sh --exp "$VTS_DOCMARKS_OUT" \
  --job-name docmarks-build --mem 16G --conc 8
```

## Stage 1 — sources and clustering (CPU, one job)

Fetching is network-bound and single-threaded per source; rendering is
CPU-bound. This is one long job rather than an array, because the sources are
sequential and the clustering needs every mark in memory at once.

```bash
python build_corpus.py \
  --sources spods,staver,tobacco800,ucsf \
  --ucsf-distractors 200000 \
  --ucsf-letterhead-per-author 2000
```

Ask for **16 GB and a wall clock in hours**, not minutes: 200k PDF fetches at a
polite rate dominate, and the UCSF endpoint is a shared public service — do not
parallelise the pull across many jobs to go faster. The builder skips a dead id
rather than aborting, and reports the count at the end; a pull that skips a few
hundred out of 200k is normal, and a pull that skips tens of thousands means
something is wrong with the endpoint, not the data.

**Resume is free.** Downloads are atomic (temp + rename), rendered pages are
skipped when present, and the Solr cursor order is stable, so a killed job
restarts where it stopped. Before resuming, delete zero-byte outputs — they
count as "done" and resume cannot see that they are empty.

## Stage 2 — roster (interactive, off-cluster)

```bash
python shortlist.py --corpus $VTS_DOCMARKS_OUT --write-roster --name docmarks-v1
```

Copy `shortlist.png` and `roster.json` somewhere you can look at them, pick your
two dozen, edit the file, then rebuild in roster mode:

```bash
python build_corpus.py --sources spods,staver,tobacco800,ucsf --roster $VTS_DOCMARKS_OUT/roster.json
```

Rebuilding is cheap — the sources are cached, so only clustering and manifest
writing re-run.

## Stage 3 — the human passes (interactive)

```bash
python make_audit_slate.py --task membership --corpus $VTS_DOCMARKS_OUT
python make_audit_slate.py --task confusable --corpus $VTS_DOCMARKS_OUT
# fill in the verdict fields, then:
python audit_to_corrections.py --task membership --corpus $VTS_DOCMARKS_OUT --apply
python audit_to_corrections.py --task confusable --corpus $VTS_DOCMARKS_OUT --apply
```

Do this **before** stage 4. A membership rejection changes which pages are
positives, and the embedding cells carry the labels; embedding first means
rebuilding every cell afterwards.

## Stage 4 — embedding cells (GPU)

```bash
python embed_corpus.py --tier s --embedders sift_vlad,siglip
python embed_corpus.py --tier m --embedders sift_vlad,siglip
python embed_corpus.py --tier l --embedders sift_vlad,siglip
```

Small tier first, always — it is the same corpus with fewer distractors, so a
tier-`s` cell is a complete usable artifact in minutes and proves the pipeline
before an hours-long tier-`l` run. **Time a real cell and multiply**; do not
quote an ETA from a tier-`s` cell for tier `l` without accounting for
`sift_vlad`'s per-image feature extraction, which dominates and is roughly
linear in page count.

Cells land in `$VTS_PILE/embeddings/docmarks_<tier>__<embedder>.pkl`. Verify
before trusting:

```bash
python embed_corpus.py --verify
```

`siglip2_l` is available and not in the default list; add it when a study needs
the premium-embedder column, not by default.

## After launching

A submission is not a launch. Confirm each job came back with a numeric id and
that output starts appearing, then arm a completion notification rather than
watching:

```bash
ssh grid 'until [ "$(squeue -u $USER -h -n docmarks-build -o %i | wc -l)" -eq 0 ]; do sleep 120; done; echo DONE; ls -la '"$VTS_DOCMARKS_OUT"
```

Poll the real signal — page counts in `build_report.json`, cell sizes in
`embed_corpus.py --list` — not just `squeue`. A drained queue with missing
output means failures, not completion.

## What to check when it finishes

- `build_report.json`: `tier_cumulative` hits the budgets, `warnings` is empty
  or explained, `rejection_reasons` holds nothing surprising.
- Roster drift: a warning naming classes "absent from this build" means the
  roster and the clustering have diverged — the eval would silently shrink.
- `separations_honoured` matches the number of adjudicated pairs.
- `needs_hand_crop`: band-located classes still owe a hand-drawn query crop.
- `embed_corpus.py --verify`: every cell loads, every media has a vector.

## Growing the corpus later

A build over a *different* page set is a new corpus version unless you pin the
tier cutoffs:

```bash
python build_corpus.py ... --pin-tiers $VTS_DOCMARKS_OUT/build_report.json
```

Pinning keeps tier membership stable so the new numbers stay comparable to the
old ones, at the cost of letting the page counts drift. Without it, say plainly
that it is a new version and re-run the baselines.
