# `human_record/` — the answers no rebuild can bring back

Every file here is a copy of something a person decided, or of the record that
makes such a decision interpretable. The pile itself is purgeable by design —
`scripts/experiments/pile/pile_config.py` requires every cell to be rebuildable
from sources that are not on scratch — and that rule never covered its one
un-rebuildable input.

**Do not edit these by hand.** They are written by
`scripts/experiments/pile/verdict_store.py` from the working copies on the
cluster, and the inventory that decides what belongs here is
`pile_config.HUMAN_RECORD`, one row per artifact with the reason it is listed.

```bash
python verdict_store.py check      # do the working copies still match these?
python verdict_store.py export     # update these from the working copies
python verdict_store.py restore    # write these back out after a purge
```

Each name is `ROOT__path__to__file`, where `ROOT` is the working location the
copy came from — a key of `pile_config.RECORD_ROOTS` (`PILE`, `WORK`,
`LABELSETS`, `WORK3588`, `EXP`). Two roots hold files with the same basename, so
the prefix is what keeps them from overwriting each other.

Not every file here is managed by the store:

- **`MANIFEST.json` lists only what `verdict_store.py export` wrote.** Its
  `CLEARED__*` rows (source `{DETECTORS}`) and `VLM3720__slates.json` (source
  `{VLM3720}`) name roots that are no longer declared, so `restore` skips them
  with a note; the committed copies are the record.
- **Some files were committed directly and are read by name.**
  `COCO_BETTER__label_review_4179.json` is the owner's `coco_better` label
  review, read through `pile_config.COCO_BETTER_CORRECTIONS`; the per-class
  `LABELSETS__{slate,belowcut,boxcheck,seatcheck,pass25,prominent,recheck}__*`
  files are read by the pile scripts (`recut.py`, `retune.py`,
  `make_belowcut.py` and `silence_rate.py` glob the `slate`, `pass25` and
  `belowcut` sets). None of them is in `MANIFEST.json`, so `check` does not
  watch them.

`MANIFEST.json` carries each file's tier, its source, its hash, and why it is
kept. Three tiers, and only the first fails the check:

| tier | means |
|---|---|
| `human` | a record of someone having looked; not reproducible at any price |
| `support` | machine-written, but what makes a human answer interpretable — which class, which cell, which stratum a row was about |
| `derived` | rebuilt by a build from the two above; kept so a purge is a restore rather than a re-derivation nobody has the flags for |
