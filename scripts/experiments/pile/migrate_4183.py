#!/usr/bin/env python3
"""#4183: rename COCO Quarry -> COCO Better and DocMarks -> FullMarks on disk, in place.

    python scripts/experiments/pile/migrate_4183.py            # dry run
    python scripts/experiments/pile/migrate_4183.py --apply    # do it

The owner's rulings on #4183 are to rename in place, not to keep both names,
and to rewrite the paths the corpus stores (2026-09-25).

**COCO Better**, per cell under ``pile_config.EMBEDDINGS``:

* ``coco_quarry*.pkl`` -> ``coco_better*.pkl``. Each cell's pickled
  ``origin.importer`` also holds the old name. ``"coco_quarry"`` and
  ``"coco_better"`` are the same length, so the string is overwritten byte for
  byte and nothing is re-serialised.
* ``*.provenance.json`` -> the new name, with ``dataset`` and ``cell`` rewritten.
* The archive ``keep/coco-quarry-25-20260920`` -> ``keep/coco-better-25-20260920``.

**FullMarks**:

* ``docmarks_*.pkl`` -> ``fullmarks_*.pkl``, re-serialised object by object in
  the same shape (one-shot or chunked). The only changes are to strings:
  ``importer`` ``"docmarks"`` becomes ``"fullmarks"``, and absolute corpus
  paths move with the directory. The names differ in length, so an in-place
  overwrite would shift the pickle. Every array is hashed on the way in and
  re-hashed from the written file, and the two must match.
* Every text file under the corpus directory that stores its absolute path
  (JSON, manifests, logs, scripts) is rewritten to the new path. Files with
  the old name in their own name are renamed.
* ``/expscratch/<you>/docmarks`` -> ``/expscratch/<you>/fullmarks``, last.

**Run it only when nothing reads the old names**: after the rename PR merges
and no queued or running job was launched from older code. ``--apply``
refuses while any of your SLURM jobs is queued or running, unless given
``--ignore-queue``.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import mmap
import os
import pickle
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pile_config as pc

USER = getpass.getuser()
OLD, NEW = b"coco_quarry", b"coco_better"
assert len(OLD) == len(NEW)  # the in-place overwrite depends on it
KEEP = [(Path(f"/expscratch/{USER}/keep/coco-quarry-25-20260920"), "coco-better-25-20260920")]
FM_OLD_DIR = Path(f"/expscratch/{USER}/docmarks")  # rename: keep
FM_NEW_DIR = Path(f"/expscratch/{USER}/fullmarks")
_FM_PATH = re.compile(re.escape(str(FM_OLD_DIR)) + r"(?=[/\"'\s:,)\]]|$)")
_FM_PATH_B = re.compile(re.escape(str(FM_OLD_DIR)).encode() + rb"(?=[/\"'\s:,)\]]|$)")
#: Pixels and documents never hold a path; skipping them keeps the scan to minutes.
_BINARY = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".pdf", ".gz", ".tar", ".zip", ".npz", ".npy", ".pkl"}


def patch_pickle(path: Path, apply: bool) -> int:
    """Occurrences of the old COCO name inside *path*; overwritten when *apply*."""
    with path.open("r+b" if apply else "rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_WRITE if apply else mmap.ACCESS_READ)
        try:
            hits, at = [], mm.find(OLD)
            while at != -1:
                hits.append(at)
                at = mm.find(OLD, at + 1)
            if apply:
                for at in hits:
                    mm[at : at + len(NEW)] = NEW
                mm.flush()
            return len(hits)
        finally:
            mm.close()


def _objects(path: Path):
    """Every pickled object in *path*, in order: one for a one-shot cell, a header and chunks for a chunked one."""
    with path.open("rb") as fh:
        while True:
            try:
                yield pickle.load(fh)  # noqa: S301 - our own cells
            except EOFError:
                return


def _swap(x: Any, counts: dict[str, int], digest: Any) -> Any:
    """*x* with the FullMarks strings renamed; every array is fed to *digest* unchanged."""
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k == "importer" and v == "docmarks":  # rename: keep
                counts["importer"] += 1
                out[k] = "fullmarks"
            else:
                out[k] = _swap(v, counts, digest)
        return out
    if isinstance(x, list):
        return [_swap(v, counts, digest) for v in x]
    if isinstance(x, tuple):
        return tuple(_swap(v, counts, digest) for v in x)
    if isinstance(x, str) and _FM_PATH.search(x):
        counts["paths"] += 1
        return _FM_PATH.sub(str(FM_NEW_DIR), x)
    if isinstance(x, np.ndarray):
        digest.update(np.ascontiguousarray(x).tobytes())
    return x


def _array_digest(path: Path) -> str:
    digest = hashlib.sha256()
    counts = {"importer": 0, "paths": 0}
    for obj in _objects(path):
        _swap(obj, counts, digest)
    return digest.hexdigest()


def repickle(old: Path, new: Path, apply: bool) -> str:
    counts = {"importer": 0, "paths": 0}
    digest = hashlib.sha256()
    tmp = new.with_name(new.name + ".part")
    n = 0
    with tmp.open("wb") if apply else open(os.devnull, "wb") as out:
        for obj in _objects(old):
            pickle.dump(_swap(obj, counts, digest), out, protocol=pickle.HIGHEST_PROTOCOL)
            n += 1
    note = f"{n} object(s), {counts['importer']} importer(s), {counts['paths']} path(s)"
    if apply:
        if _array_digest(tmp) != digest.hexdigest():
            tmp.unlink()
            raise SystemExit(f"{new.name}: arrays changed in the rewrite; refusing (old cell kept)")
        tmp.replace(new)
        old.unlink()
        note += ", arrays verified"
    return note


def rewrite_corpus_paths(apply: bool) -> int:
    changed = 0
    for path in sorted(FM_OLD_DIR.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.suffix.lower() in _BINARY:
            continue
        data = path.read_bytes()
        if not _FM_PATH_B.search(data):
            continue
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            print(f"  skip (not utf-8): {path}")
            continue
        changed += 1
        if apply:
            tmp = path.with_name(path.name + ".rename-4183")
            tmp.write_bytes(_FM_PATH_B.sub(str(FM_NEW_DIR).encode(), data))
            os.chmod(tmp, path.stat().st_mode)
            tmp.replace(path)
    return changed


def coco_better(apply: bool) -> None:
    olds = sorted(p for p in pc.EMBEDDINGS.iterdir() if p.name.startswith("coco_quarry"))  # rename: keep
    for old in olds:
        new = old.with_name(old.name.replace("coco_quarry", "coco_better", 1))  # rename: keep
        if new.exists():
            raise SystemExit(f"{new.name} already exists; refusing to overwrite")
        if old.suffix == ".pkl":
            print(f"{old.name} -> {new.name}: {patch_pickle(old, apply)} embedded name(s)")
        elif old.name.endswith(".provenance.json"):
            text = json.dumps(json.loads(old.read_text()), indent=2).replace(
                "coco_quarry", "coco_better"
            )  # rename: keep
            print(f"{old.name} -> {new.name}: dataset/cell rewritten")
            if apply:
                new.write_text(text + "\n")
                old.unlink()
            continue
        else:
            print(f"{old.name} -> {new.name}")
        if apply:
            old.rename(new)
    for old, name in KEEP:
        if old.exists():
            print(f"{old} -> {old.with_name(name)}")
            if apply:
                old.rename(old.with_name(name))
    print(f"COCO Better: {len(olds)} cell files")


def fullmarks(apply: bool) -> None:
    olds = sorted(p for p in pc.EMBEDDINGS.iterdir() if p.name.startswith("docmarks"))  # rename: keep
    for old in olds:
        new = old.with_name(old.name.replace("docmarks", "fullmarks", 1))  # rename: keep
        if new.exists():
            raise SystemExit(f"{new.name} already exists; refusing to overwrite")
        if old.suffix == ".pkl":
            print(f"{old.name} -> {new.name}: {repickle(old, new, apply)}")
        elif old.name.endswith(".provenance.json"):
            text = old.read_text().replace("docmarks", "fullmarks")  # rename: keep
            print(f"{old.name} -> {new.name}: rewritten")
            if apply:
                new.write_text(text)
                old.unlink()
        else:
            print(f"{old.name} -> {new.name}")
            if apply:
                old.rename(new)
    if not FM_OLD_DIR.exists():
        print(f"FullMarks: {len(olds)} cell files; {FM_OLD_DIR} already moved")
        return
    if FM_NEW_DIR.exists():
        raise SystemExit(f"{FM_NEW_DIR} already exists; refusing to overwrite")
    print(f"{rewrite_corpus_paths(apply)} corpus text files store {FM_OLD_DIR}; rewritten to {FM_NEW_DIR}")
    for path in sorted(FM_OLD_DIR.rglob("*docmarks*"), reverse=True):  # rename: keep
        new = path.with_name(path.name.replace("docmarks", "fullmarks"))  # rename: keep
        print(f"  {path.relative_to(FM_OLD_DIR)} -> {new.name}")
        if apply:
            path.rename(new)
    print(f"{FM_OLD_DIR} -> {FM_NEW_DIR}")
    if apply:
        FM_OLD_DIR.rename(FM_NEW_DIR)
    print(f"FullMarks: {len(olds)} cell files")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--ignore-queue", action="store_true")
    args = ap.parse_args()
    if args.apply and not args.ignore_queue:
        queue = subprocess.run(["squeue", "-u", USER, "-h"], capture_output=True, text=True).stdout  # noqa: S603, S607
        if queue.strip():
            raise SystemExit("jobs are queued or running and may read the old names; wait, or pass --ignore-queue")
    coco_better(args.apply)
    fullmarks(args.apply)
    print("done" if args.apply else "(dry run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
