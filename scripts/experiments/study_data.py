#!/usr/bin/env python3
"""Keep a study's bulk data out of the tree: archive it to a release asset, fetch it back.

A study under ``docs/experiments/<study>/`` keeps in the tree only what a reader
of its report needs on GitHub: the markdown, the figures and pages the markdown
links to, the viewer and report pages, and the scripts that made them. Its bulk
data (cell CSVs, measurement tables, JSON dumps, unlinked figures) leaves the
tree and lives in one tarball per study on the rolling ``experiments-data``
release, so every checkout, grep and file search stops paying for it (#5404).

Nothing is lost by archiving. The study's ``archived-data.json`` names each file
it moved, its git blob id, and the commit that last held it, so the same bytes
come back either from the release or straight from git history.

Subcommands::

    study_data.py plan [STUDY ...]            what `archive` would move (dry run)
    study_data.py webp STUDY ...              linked PNG figures -> lossless WebP, links repointed
    study_data.py archive STUDY ...           move the bulk data out of the tree (git rm)
    study_data.py fetch STUDY ... | --all     put archived data back in place
    study_data.py clean STUDY ... | --all     remove fetched data again
    study_data.py publish                     upload missing tarballs (CI; needs gh)

A study is named by its directory (``2026-08-27-good-mining-3267``) or its
path. Finishing a study is::

    python scripts/experiments/study_data.py webp <study>     # figures -> WebP
    git add docs/experiments/<study> && git commit              # data now in a commit
    python scripts/experiments/study_data.py archive <study>  # data -> archived-data.json
    git commit                                                  # and out of the tree

``archive`` records ``HEAD`` as the commit the files came from, so the data must
be committed, unmodified, first. When the branch merges into ``dev``,
``.github/workflows/publish-experiment-data.yml`` packs each new tarball from
that commit and uploads it. Until then, ``fetch --from-git`` restores the files
from history.

**What stays in the tree** (everything else under the study is archived):

- markdown, and code (``.py``, ``.sh``, ``.sbatch``, ...);
- ``viewer*.html`` and ``report*.html``, the study's interactive record;
- any file a tracked markdown file links to or embeds, anywhere in the repo, or
  names by its full repo path in backticks (``scripts/check-docs.py`` checks
  those), and any file a kept page links to;
- any file live code reads: code outside ``docs/experiments/`` and
  ``scripts/experiments/`` (tests, slide figure sources, the app) that names
  the study and the file's name, or the file's full path.

Experiment analysis scripts do not keep data in the tree. They re-read a
finished study's tables only to rebuild its figures, so run ``fetch`` first.

Fetched files are untracked, so ``fetch`` lists them in the checkout's
``.git/info/exclude``. Without that, ``git status`` would show them and
``git add -A`` would commit them back. ``clean`` deletes the files and takes
them off the list.
"""

from __future__ import annotations

import argparse
import fnmatch
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = "docs/experiments"
MANIFEST = "archived-data.json"
RELEASE = "experiments-data"
REPO = "samggreenberg/VTSearch"
DOWNLOAD_URL = "https://github.com/{repo}/releases/download/{release}/{asset}"

#: Prose and code: what a study's directory is *for*, never archived.
KEEP_SUFFIXES = frozenset({".md", ".py", ".sh", ".sbatch", ".slurm", ".patch", ".mjs", ".js", ".ts", ".ipynb"})
#: The interactive record a long study carries beside its REPORT.md.
KEEP_NAMES = ("viewer*.html", "report*.html")
#: Code that counts as a live reader of a study's data. Experiment analysis
#: code under these prefixes does not: it re-reads a finished study only to
#: rebuild it, and ``fetch`` serves that.
CODE_SUFFIXES = frozenset({".py", ".sh", ".mjs", ".js", ".ts", ".yml", ".yaml"})
NOT_LIVE_PREFIXES = (f"{EXPERIMENTS}/", "scripts/experiments/")

STUDY_RE = re.compile(r"\d{4}-\d{2}-\d{2}-[a-z0-9]+(?:-[a-z0-9]+)*")
# `[text](target)`, `![alt](target)`, `[k]: target`, and HTML `src=` / `href=`.
MD_LINK_RE = re.compile(r"\]\(\s*<?([^)\s>]+)")
REF_DEF_RE = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*<?(\S+?)>?(?:\s|$)", re.MULTILINE)
HTML_REF_RE = re.compile(r"""(?:src|href)\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
CODE_SPAN_RE = re.compile(r"`([^`\n]+)`")
FULL_PATH_RE = re.compile(rf"{EXPERIMENTS}/({STUDY_RE.pattern})/([^\s\"'`)\],;:]+)")
# A data file's name as code spells it, f-string fields and globs included:
# `"steps.csv"`, `f"betaq-rep{r}-steps.csv"`, `"cells_*.json"`.
DATA_NAME_RE = re.compile(
    r"[\w.{}*\[\]-]*\.(?:csv|tsv|json|jsonl|txt|gz|npz|npy|pkl|parquet|png|jpg|jpeg|webp|svg|html)\b"
)


def git(*args: str, data: bytes | None = None) -> bytes:
    """Run git in the repo root and return stdout; a failure is fatal."""
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", "-C", str(ROOT), *args],  # noqa: S607 - git resolves on PATH, as everywhere in this repo
        input=data,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed:\n{proc.stderr.decode(errors='replace').strip()}")
    return proc.stdout


def blob_oid(data: bytes) -> str:
    """The git blob id of `data`, so a fetched file is checked against the id `archive` recorded."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def tracked(*paths: str) -> list[str]:
    out = git("-c", "core.quotepath=off", "ls-files", "-z", "--", *paths)
    return [p for p in out.decode().split("\0") if p]


# --------------------------------------------------------------------------
# What stays in the tree
# --------------------------------------------------------------------------


@dataclass
class Tree:
    """The tracked files, and the experiment files something in the tree points at."""

    files: list[str]
    studies: frozenset[str]
    by_study: dict[str, list[str]] = field(default_factory=dict)
    kept: set[str] = field(default_factory=set)
    #: The subset of `kept` that live code reads, which `webp` must not rename.
    read_by_code: set[str] = field(default_factory=set)

    @classmethod
    def load(cls) -> Tree:
        files = tracked()
        by_study: dict[str, list[str]] = {}
        for path in files:
            parts = path.split("/")
            if len(parts) > 3 and path.startswith(f"{EXPERIMENTS}/"):
                by_study.setdefault(parts[2], []).append(path)
        tree = cls(files=files, studies=frozenset(by_study), by_study=by_study)
        tree._mark_kept()
        return tree

    def _keep(self, path: str) -> None:
        """Keep `path`, or every study file under it when it names a directory inside a study."""
        parts = path.rstrip("/").split("/")
        if len(parts) < 4 or parts[2] not in self.studies:
            return  # outside every study, or the study directory itself (a citation, not a read)
        if any(ch in path for ch in "*?["):
            hits = {f for f in self.by_study[parts[2]] if fnmatch.fnmatchcase(f, path)}
        else:
            prefix = path.rstrip("/") + "/"
            hits = {f for f in self.by_study[parts[2]] if f == path or f.startswith(prefix)}
        self.kept |= hits

    def _resolve(self, doc: str, target: str) -> str | None:
        target = unquote(target.split("#", 1)[0].split("?", 1)[0]).strip()
        if not target or "://" in target or target.startswith(("mailto:", "data:", "/")):
            return None
        return os.path.normpath(str(PurePosixPath(doc).parent / target)).replace(os.sep, "/")

    def _mark_kept(self) -> None:
        pages: list[str] = []
        for doc in self.files:
            if not doc.endswith(".md"):
                continue
            text = (ROOT / doc).read_text(encoding="utf-8", errors="replace")
            targets = MD_LINK_RE.findall(text) + REF_DEF_RE.findall(text) + HTML_REF_RE.findall(text)
            for target in targets:
                resolved = self._resolve(doc, target)
                if resolved:
                    self._keep(resolved)
            for span in CODE_SPAN_RE.findall(text):
                if span.startswith(f"{EXPERIMENTS}/"):
                    self._keep(re.sub(r":\d[\d,-]*$", "", span).rstrip(".,;:"))
        for study, paths in self.by_study.items():
            for path in paths:
                if any(fnmatch.fnmatchcase(path.rsplit("/", 1)[-1], pat) for pat in KEEP_NAMES):
                    self.kept.add(path)
            pages += [p for p in paths if p.endswith(".html")]
        # A kept page's own images and links (the narrative report.html pages).
        for page in pages:
            if page not in self.kept:
                continue
            text = (ROOT / page).read_text(encoding="utf-8", errors="replace")
            for target in HTML_REF_RE.findall(text):
                resolved = self._resolve(page, target)
                if resolved:
                    self._keep(resolved)
        for path in self.files:
            if path.startswith(NOT_LIVE_PREFIXES) or PurePosixPath(path).suffix not in CODE_SUFFIXES:
                continue
            if path == "scripts/experiments/study_data.py":
                continue
            self._mark_read_by(path)

    def _mark_read_by(self, code: str) -> None:
        try:
            text = (ROOT / code).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        # A named file or directory stays in the tree. Only a file whose own name
        # the code spells counts as read by it, for `webp`; a directory mention
        # alone does not say which of its files the code opens.
        for study, rest in FULL_PATH_RE.findall(text):
            self._keep(f"{EXPERIMENTS}/{study}/{rest}")
        studies = {s for s in STUDY_RE.findall(text) if s in self.studies}
        if not studies:
            return
        globs = {re.sub(r"\{[^}]*\}", "*", name.rsplit("/", 1)[-1]) for name in DATA_NAME_RE.findall(text)}
        for study in studies:
            for path in self.by_study[study]:
                base = path.rsplit("/", 1)[-1]
                if any(fnmatch.fnmatchcase(base, g) for g in globs):
                    self.kept.add(path)
                    self.read_by_code.add(path)

    def archivable(self, study: str) -> list[str]:
        """The study's tracked files that `archive` moves out of the tree."""
        out = []
        for path in self.by_study.get(study, []):
            name = path.rsplit("/", 1)[-1]
            if name == MANIFEST or PurePosixPath(name).suffix.lower() in KEEP_SUFFIXES or path in self.kept:
                continue
            out.append(path)
        return out


# --------------------------------------------------------------------------
# Manifests
# --------------------------------------------------------------------------


def study_name(arg: str) -> str:
    name = PurePosixPath(arg.rstrip("/")).name
    if not STUDY_RE.fullmatch(name):
        sys.exit(f"{arg!r} is not a study directory name (YYYY-MM-DD-<slug>)")
    return name


def manifest_path(study: str) -> Path:
    return ROOT / EXPERIMENTS / study / MANIFEST


def load_manifest(study: str) -> dict:
    path = manifest_path(study)
    if not path.is_file():
        return {"release": RELEASE, "batches": []}
    return json.loads(path.read_text(encoding="utf-8"))


def manifests() -> Iterator[tuple[str, dict]]:
    for path in sorted((ROOT / EXPERIMENTS).glob(f"*/{MANIFEST}")):
        yield path.parent.name, json.loads(path.read_text(encoding="utf-8"))


def archived_studies(args: argparse.Namespace) -> list[str]:
    if args.all:
        return [study for study, _ in manifests()]
    if not args.studies:
        sys.exit("name at least one study, or pass --all")
    return [study_name(s) for s in args.studies]


# --------------------------------------------------------------------------
# plan / archive
# --------------------------------------------------------------------------


def human(n: int) -> str:
    return f"{n / 1e6:.1f} MB" if n >= 1e5 else f"{n / 1e3:.0f} kB"


def cmd_plan(args: argparse.Namespace) -> int:
    tree = Tree.load()
    studies = [study_name(s) for s in args.studies] or sorted(tree.studies)
    total_n = total_b = 0
    for study in studies:
        paths = tree.archivable(study)
        if not paths:
            continue
        size = sum((ROOT / p).stat().st_size for p in paths)
        total_n += len(paths)
        total_b += size
        print(f"{human(size):>9} {len(paths):5d}  {study}")
        if args.files:
            for path in paths:
                print(f"            {path.split('/', 3)[3]}")
    print(f"{human(total_b):>9} {total_n:5d}  total")
    return 0


def cmd_archive(args: argparse.Namespace) -> int:
    tree = Tree.load()
    head = git("rev-parse", "HEAD").decode().strip()
    for study in [study_name(s) for s in args.studies]:
        paths = tree.archivable(study)
        if not paths:
            print(f"{study}: nothing to archive")
            continue
        dirty = git("status", "--porcelain", "--", *paths).decode().strip()
        if dirty:
            sys.exit(f"{study}: commit these first; `archive` records HEAD as their source:\n{dirty}")
        oids = {}
        for line in git("-c", "core.quotepath=off", "ls-tree", "-r", "-z", "HEAD", "--", *paths).decode().split("\0"):
            if line:
                meta, path = line.split("\t", 1)
                oids[path] = meta.split()[2]
        prefix = f"{EXPERIMENTS}/{study}/"
        manifest = load_manifest(study)
        manifest["batches"].append(
            {
                "asset": f"{study}.{head[:12]}.tar.gz",
                "source_commit": head,
                "files": {
                    p.removeprefix(prefix): {"bytes": (ROOT / p).stat().st_size, "oid": oids[p]} for p in sorted(paths)
                },
            }
        )
        manifest_path(study).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
        git("rm", "-q", *(["--cached"] if args.keep_local else []), "--", *paths)
        git("add", "--", str(manifest_path(study).relative_to(ROOT)))
        if args.keep_local:
            exclude_paths(paths, add=True)
        size = sum(f["bytes"] for f in manifest["batches"][-1]["files"].values())
        print(f"{study}: archived {len(paths)} files ({human(size)})")
    return 0


# --------------------------------------------------------------------------
# fetch / clean
# --------------------------------------------------------------------------

EXCLUDE_BEGIN = "# >>> study_data.py fetch: archived experiment data restored in place"
EXCLUDE_END = "# <<< study_data.py fetch"


def exclude_paths(paths: Iterable[str], *, add: bool) -> None:
    """Add `paths` to (or drop them from) this checkout's private ignore list."""
    exclude = ROOT / git("rev-parse", "--git-path", "info/exclude").decode().strip()
    text = exclude.read_text(encoding="utf-8") if exclude.is_file() else ""
    head, _, rest = text.partition(EXCLUDE_BEGIN + "\n")
    block, _, tail = rest.partition(EXCLUDE_END + "\n")
    listed = {line for line in block.splitlines() if line}
    wanted = {"/" + re.sub(r"([*?\[\\])", r"\\\1", p) for p in paths}
    listed = listed | wanted if add else listed - wanted
    middle = f"{EXCLUDE_BEGIN}\n" + "".join(f"{p}\n" for p in sorted(listed)) + f"{EXCLUDE_END}\n" if listed else ""
    exclude.parent.mkdir(parents=True, exist_ok=True)
    exclude.write_text(head + middle + tail, encoding="utf-8")


def have_commit(commit: str) -> bool:
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", "-C", str(ROOT), "cat-file", "-e", f"{commit}^{{commit}}"],  # noqa: S607 - git on PATH
        capture_output=True,
        check=False,
    )
    return proc.returncode == 0


def ensure_commit(commit: str) -> None:
    if not have_commit(commit):
        git("fetch", "-q", "--depth=1", "origin", commit)


def blobs_from_git(study: str, batch: dict) -> Iterator[tuple[str, bytes]]:
    """Each file of `batch`, read out of the commit that held it."""
    ensure_commit(batch["source_commit"])
    names = list(batch["files"])
    specs = "".join(f"{batch['source_commit']}:{EXPERIMENTS}/{study}/{n}\n" for n in names).encode()
    out = git("cat-file", "--batch", data=specs)
    pos = 0
    for name in names:
        header_end = out.index(b"\n", pos)
        header = out[pos:header_end].split()
        if header[-1] == b"missing" or header[1] != b"blob":
            sys.exit(f"{study}/{name} is not in {batch['source_commit']}")
        size = int(header[2])
        yield name, out[header_end + 1 : header_end + 1 + size]
        pos = header_end + 1 + size + 1


def blobs_from_tarball(study: str, batch: dict, data: bytes) -> Iterator[tuple[str, bytes]]:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile() or member.name not in batch["files"]:
                sys.exit(f"{batch['asset']}: unexpected member {member.name!r}")
            handle = tar.extractfile(member)
            assert handle is not None
            yield member.name, handle.read()


def download(asset: str, release: str) -> bytes:
    url = DOWNLOAD_URL.format(repo=REPO, release=release, asset=asset)
    try:
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - fixed https URL
            return response.read()
    except OSError as exc:
        sys.exit(
            f"could not download {url}: {exc}\n"
            "If the branch that archived it has not merged into dev yet, its tarball is not published; "
            "pass --from-git to read the files out of git history instead."
        )


def cmd_fetch(args: argparse.Namespace) -> int:
    for study in archived_studies(args):
        manifest = load_manifest(study)
        if not manifest["batches"]:
            print(f"{study}: nothing archived")
            continue
        written: list[str] = []
        for batch in manifest["batches"]:
            study_dir = ROOT / EXPERIMENTS / study
            missing = {
                n: f
                for n, f in batch["files"].items()
                if not (study_dir / n).is_file() or blob_oid((study_dir / n).read_bytes()) != f["oid"]
            }
            if not missing:
                continue
            if args.from_git:
                blobs = blobs_from_git(study, batch)
            elif args.asset_dir:
                blobs = blobs_from_tarball(study, batch, (Path(args.asset_dir) / batch["asset"]).read_bytes())
            else:
                blobs = blobs_from_tarball(study, batch, download(batch["asset"], manifest.get("release", RELEASE)))
            for name, data in blobs:
                if name not in missing:
                    continue
                if blob_oid(data) != missing.pop(name)["oid"]:
                    sys.exit(f"{study}/{name}: content does not match the id archive recorded")
                target = study_dir / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                written.append(f"{EXPERIMENTS}/{study}/{name}")
            if missing:
                sys.exit(f"{batch['asset']} lacks {len(missing)} of its files, e.g. {next(iter(missing))}")
        exclude_paths(written, add=True)
        print(f"{study}: restored {len(written)} files")
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    for study in archived_studies(args):
        removed = []
        in_git = set(tracked(f"{EXPERIMENTS}/{study}"))
        for batch in load_manifest(study)["batches"]:
            for name in batch["files"]:
                path = f"{EXPERIMENTS}/{study}/{name}"
                if (ROOT / path).is_file() and path not in in_git:
                    (ROOT / path).unlink()
                    removed.append(path)
        exclude_paths(removed, add=False)
        for directory in sorted({(ROOT / p).parent for p in removed}, key=lambda d: -len(d.parts)):
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
        print(f"{study}: removed {len(removed)} fetched files")
    return 0


# --------------------------------------------------------------------------
# publish
# --------------------------------------------------------------------------


def tarball(study: str, batch: dict) -> bytes:
    """The batch's tarball, byte-for-byte reproducible from git: sorted, zero mtimes, no owners."""
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as tar:
        for name, data in sorted(blobs_from_git(study, batch)):
            if blob_oid(data) != batch["files"][name]["oid"]:
                sys.exit(f"{study}/{name}: {batch['source_commit']} holds other bytes than the manifest records")
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def gh(*args: str) -> str:
    proc = subprocess.run(["gh", *args], capture_output=True, text=True, check=False)  # noqa: S603, S607
    if proc.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)}: {proc.stderr.strip()}")
    return proc.stdout


def cmd_publish(args: argparse.Namespace) -> int:
    batches = [(study, batch) for study, m in manifests() for batch in m["batches"]]
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for study, batch in batches:
            (out / batch["asset"]).write_bytes(tarball(study, batch))
        print(f"wrote {len(batches)} tarballs to {out}")
        return 0
    try:
        listed = gh("release", "view", RELEASE, "--json", "assets", "--jq", ".assets[].name")
    except RuntimeError:
        if not batches:
            return 0
        gh(
            "release",
            "create",
            RELEASE,
            "--prerelease",
            "--target",
            batches[0][1]["source_commit"],
            "--title",
            "Experiment data",
            "--notes",
            "The bulk data behind the reports under docs/experiments/, one tarball per archive batch. "
            "Not a version of VTSearch. Restore a study's files in place with "
            "`python scripts/experiments/study_data.py fetch <study>`.",
        )
        listed = ""
    present = set(listed.split())
    todo = [(s, b) for s, b in batches if b["asset"] not in present]
    print(f"{len(batches)} archive batches, {len(todo)} to upload")
    with tempfile.TemporaryDirectory() as tmp:
        for study, batch in todo:
            path = Path(tmp) / batch["asset"]
            path.write_bytes(tarball(study, batch))
            gh("release", "upload", RELEASE, str(path))
            print(f"  uploaded {batch['asset']} ({human(path.stat().st_size)})")
            path.unlink()
    return 0


# --------------------------------------------------------------------------
# webp
# --------------------------------------------------------------------------


def to_webp(png: Path) -> Path | None:
    """Re-encode `png` as lossless WebP beside it and delete it; None if that would lose anything."""
    from PIL import Image  # here, not at the top: the rest of the script is stdlib, for CI

    with Image.open(png) as image:
        image.load()
        if image.mode not in ("RGB", "RGBA", "L", "LA", "P", "PA", "1"):
            return None  # 16-bit and float PNGs have no lossless WebP form
        if max(image.size) > 16383:
            return None
        source = image.convert("RGBA" if "A" in image.getbands() or "transparency" in image.info else "RGB")
    webp = png.with_suffix(".webp")
    source.save(webp, "WEBP", lossless=True, quality=100, method=4, exact=True)
    with Image.open(webp) as check:
        if check.convert(source.mode).tobytes() != source.tobytes():
            webp.unlink()
            return None
    png.unlink()
    return webp


def repoint(text: str, doc: str, renamed: dict[str, str], pattern: re.Pattern[str], root_too: bool = False) -> str:
    """Rewrite each target `pattern` captures in `text` that resolves to a renamed file.

    Targets resolve against `doc`'s directory, as links do; with `root_too`, a
    target spelled from the repo root (a backticked path) resolves there too.
    """

    def swap(match: re.Match[str]) -> str:
        target = match.group(1)
        path_part = target.split("#", 1)[0].split("?", 1)[0]
        if not path_part.lower().endswith(".png"):
            return match.group(0)
        resolved = os.path.normpath(str(PurePosixPath(doc).parent / unquote(path_part))).replace(os.sep, "/")
        if resolved not in renamed and root_too and path_part in renamed:
            resolved = path_part
        if resolved not in renamed:
            return match.group(0)
        new = path_part[: -len(".png")] + PurePosixPath(renamed[resolved]).suffix + target[len(path_part) :]
        return match.group(0).replace(target, new, 1)

    return pattern.sub(swap, text)


def cmd_webp(args: argparse.Namespace) -> int:
    tree = Tree.load()
    studies = [study_name(s) for s in args.studies]
    renamed: dict[str, str] = {}
    skipped: list[str] = []
    converted = 0
    for study in studies:
        for path in tree.by_study.get(study, []):
            if path.lower().endswith(".png") and path in tree.kept:
                webp = None if path in tree.read_by_code else to_webp(ROOT / path)
                if webp is None:
                    skipped.append(path)
                    continue
                converted += 1
                renamed[path] = webp.relative_to(ROOT).as_posix()
                git("rm", "-q", "--cached", "--", path)
                git("add", "--", renamed[path])
        # A link a report generator wrote to a PNG that an earlier run already converted.
        for path in tree.by_study.get(study, []):
            if path.lower().endswith(".webp"):
                renamed.setdefault(path[: -len(".webp")] + ".png", path)
    docs = [f for f in tree.files if f.endswith((".md", ".html")) and (ROOT / f).is_file()]
    edited = 0
    for doc in docs:
        if doc.endswith(".html") and not doc.startswith(f"{EXPERIMENTS}/"):
            continue
        text = (ROOT / doc).read_text(encoding="utf-8")
        new = repoint(text, doc, renamed, MD_LINK_RE if doc.endswith(".md") else HTML_REF_RE)
        if doc.endswith(".md"):
            new = repoint(new, doc, renamed, HTML_REF_RE)
            new = repoint(new, doc, renamed, CODE_SPAN_RE, root_too=True)
        if new != text:
            (ROOT / doc).write_text(new, encoding="utf-8")
            edited += 1
    print(f"converted {converted} figures, repointed links in {edited} files")
    for path in skipped:
        why = "live code reads it" if path in tree.read_by_code else "no lossless WebP form"
        print(f"  kept as PNG ({why}): {path}")
    return 0


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="what `archive` would move (dry run)")
    plan.add_argument("studies", nargs="*")
    plan.add_argument("--files", action="store_true", help="list every file, not only the totals")
    plan.set_defaults(run=cmd_plan)

    webp = sub.add_parser("webp", help="re-encode a study's linked PNG figures as lossless WebP")
    webp.add_argument("studies", nargs="+")
    webp.set_defaults(run=cmd_webp)

    archive = sub.add_parser("archive", help="move a study's bulk data out of the tree")
    archive.add_argument("studies", nargs="+")
    archive.add_argument("--keep-local", action="store_true", help="untrack the files but leave them on disk")
    archive.set_defaults(run=cmd_archive)

    for name, run, text in (("fetch", cmd_fetch, "put archived data back"), ("clean", cmd_clean, "remove it again")):
        p = sub.add_parser(name, help=text)
        p.add_argument("studies", nargs="*")
        p.add_argument("--all", action="store_true", help="every study with an archived-data.json")
        if name == "fetch":
            p.add_argument("--from-git", action="store_true", help="read the files out of git history")
            p.add_argument("--asset-dir", help="read tarballs from this directory instead of the release")
        p.set_defaults(run=run)

    publish = sub.add_parser("publish", help="upload the tarballs the release is missing (needs gh)")
    publish.add_argument("--out", help="write every tarball here instead of uploading")
    publish.set_defaults(run=cmd_publish)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
