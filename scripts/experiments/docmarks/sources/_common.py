"""Shared record types and pure helpers for the DocMarks source adapters.

Every adapter splits the same way:

* ``fetch_*`` touches the network and the filesystem.  Not unit-tested; it is
  exercised for real on the GRID and guarded by ``--probe``.
* everything else is a pure function of bytes already on disk, and *is* unit
  tested against small fixtures.

That split is the reason the corpus builder can be developed and verified in an
environment that cannot reach Kaggle or hold a 3 GB archive.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Mark:
    """One ground-truth mark on one page.

    ``class_id`` is ``None`` until the identity-clustering pass runs: SPODS and
    StaVer ship *where* a mark is without shipping *which* mark it is.
    """

    kind: str  # "logo" | "stamp" | "signature" | "text" | "icon"
    box: tuple[int, int, int, int]  # x, y, w, h in page pixels
    class_id: Optional[str] = None
    #: "gt" (shipped by the source), "clustered" (derived, needs audit),
    #: "weak" (metadata-implied, unverified) or "synthetic" (true by construction).
    provenance: str = "gt"

    def area(self) -> int:
        return self.box[2] * self.box[3]

    def longest_side(self) -> int:
        return max(self.box[2], self.box[3])


@dataclass
class Page:
    """One page image plus everything known about it."""

    page_id: str  # globally unique, "<source>/<local id>"
    source: str  # spods | staver | tobacco800 | ucsf | synth
    path: str  # relative to the corpus root
    width: int
    height: int
    marks: list[Mark] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["marks"] = [{**asdict(m), "box": list(m.box)} for m in self.marks]
        return d

    @staticmethod
    def from_json(d: dict[str, Any]) -> "Page":
        marks = [
            Mark(
                kind=m["kind"],
                box=tuple(m["box"]),  # type: ignore[arg-type]
                class_id=m.get("class_id"),
                provenance=m.get("provenance", "gt"),
            )
            for m in d.get("marks", [])
        ]
        return Page(
            page_id=d["page_id"],
            source=d["source"],
            path=d["path"],
            width=d["width"],
            height=d["height"],
            marks=marks,
            meta=d.get("meta", {}),
        )


def write_manifest(pages: Iterable[Page], path: Path) -> int:
    """Write ``corpus.jsonl``.  Returns the number of records written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for page in pages:
            fh.write(json.dumps(page.to_json(), sort_keys=True) + "\n")
            n += 1
    tmp.replace(path)
    return n


def read_manifest(path: Path) -> Iterator[Page]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield Page.from_json(json.loads(line))


# --------------------------------------------------------------------------
# Deterministic sampling
# --------------------------------------------------------------------------


def stable_rank(key: str, salt: str) -> float:
    """A stable pseudo-random rank in ``[0, 1)`` for *key*.

    Used to pick which distractors land in which tier.  It must be a pure
    function of the id, never of iteration order or of how many pages happened
    to be fetched — otherwise a tier reshuffles every time the corpus grows, and
    two studies that both say "tier s" are not comparable.
    """
    h = hashlib.sha256(f"{salt}\x00{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / float(1 << 64)


# --------------------------------------------------------------------------
# Masks -> boxes
# --------------------------------------------------------------------------


def mask_to_boxes(
    mask: Any,
    min_area_frac: float = 0.0002,
    *,
    polarity: str = "auto",
) -> list[tuple[int, int, int, int]]:
    """Connected components of a binary *mask* as ``(x, y, w, h)`` boxes.

    SPODS and StaVer both ship per-category pixel masks rather than boxes, so
    this is the first step for either.  Components below *min_area_frac* of the
    page are dropped as scanning speckle.

    **Polarity is detected, not assumed.**  SPODS ships 1-bit masks with the
    mark in *black* on white paper, so a naive "non-zero is foreground" reads
    99.8% of every page as one enormous mark — which is not a crash, it is 1,088
    page-sized boxes that cluster into a single class and look superficially
    like a working corpus. Taking the minority phase as foreground is safe
    because a ground-truth mask marks a *mark*: on real SPODS pages the marked
    fraction runs 0.2–1.1%, and any mask where most of the page is "on" is
    inverted by definition of the task.

    Pass ``polarity="light"`` or ``"dark"`` to force it when a source's masks are
    genuinely dense (a text mask on a very full page can approach half, though
    none observed comes close).
    """
    import numpy as np

    arr = np.asarray(mask)
    if arr.ndim == 3:  # RGB(A) mask -> any channel lit
        arr = arr[..., :3].max(axis=2)

    threshold = arr.max() / 2 if arr.max() > 1 else 0
    lit = arr > threshold
    if polarity == "auto":
        foreground = lit if lit.mean() <= 0.5 else ~lit
    elif polarity == "light":
        foreground = lit
    elif polarity == "dark":
        foreground = ~lit
    else:
        raise ValueError(f"unknown polarity {polarity!r} (expected auto|light|dark)")

    binary = foreground.astype("uint8")
    if not binary.any():
        return []

    page_area = float(binary.shape[0] * binary.shape[1])
    min_area = max(1.0, min_area_frac * page_area)

    boxes: list[tuple[int, int, int, int]] = []
    try:
        import cv2

        n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, n):  # 0 is background
            x, y, w, h, area = (int(v) for v in stats[i])
            if area >= min_area:
                boxes.append((x, y, w, h))
    except ImportError:
        from scipy import ndimage  # type: ignore[import-untyped]

        labels, n = ndimage.label(binary)
        for i, sl in enumerate(ndimage.find_objects(labels), start=1):
            if sl is None:
                continue
            ys, xs = sl
            area = int((labels[sl] == i).sum())
            if area >= min_area:
                boxes.append((int(xs.start), int(ys.start), int(xs.stop - xs.start), int(ys.stop - ys.start)))

    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def merge_overlapping(boxes: list[tuple[int, int, int, int]], gap: int = 6) -> list[tuple[int, int, int, int]]:
    """Merge boxes that touch or nearly touch, within *gap* pixels.

    A rubber stamp's mask usually breaks into a dozen components — the ring, the
    text inside it, a broken arc where the ink did not take.  Left unmerged,
    every fragment becomes its own "mark" and the class inventory is nonsense.
    """
    remaining = list(boxes)
    out: list[tuple[int, int, int, int]] = []
    while remaining:
        x, y, w, h = remaining.pop()
        changed = True
        while changed:
            changed = False
            for other in list(remaining):
                ox, oy, ow, oh = other
                if x - gap < ox + ow and ox - gap < x + w and y - gap < oy + oh and oy - gap < y + h:
                    nx, ny = min(x, ox), min(y, oy)
                    x, y, w, h = nx, ny, max(x + w, ox + ow) - nx, max(y + h, oy + oh) - ny
                    remaining.remove(other)
                    changed = True
        out.append((x, y, w, h))
    out.sort(key=lambda b: (b[1], b[0]))
    return out


# --------------------------------------------------------------------------
# Fetch helpers
# --------------------------------------------------------------------------


class FetchError(RuntimeError):
    """A source could not be fetched, with an actionable reason."""


def require_kaggle_credentials(slug: str) -> None:
    """Raise :class:`FetchError` unless a Kaggle credential is in place.

    The Kaggle CLI reads ``~/.kaggle/kaggle.json``, ``~/.kaggle/access_token``
    or the ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` environment pair.  Checking here
    means a missing token is reported with the setup instruction, rather than as
    a 403 halfway through a grid job.

    ``access_token`` is in that list because it is what Kaggle's own "Create New
    Token" now writes, and what ``kagglesdk`` reads natively (its
    ``kaggle_creds.py`` / ``kaggle_oauth.py``).  Accepting only ``kaggle.json``
    made a *working* credential look like a missing one: on the GRID the probe
    reported both Kaggle sources BLOCKED while ``kaggle datasets download``
    succeeded from the very same shell, against the very same token.  Since the
    whole point of this gate is to fail fast rather than 403 mid-job, a false
    BLOCKED is the one way it can be worse than having no gate at all.
    """
    has_env = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    kaggle_dir = Path.home() / ".kaggle"
    has_file = (kaggle_dir / "kaggle.json").exists() or (kaggle_dir / "access_token").exists()
    if not (has_env or has_file):
        raise FetchError(
            f"Kaggle credentials not found, needed for '{slug}'. "
            "Put a token at ~/.kaggle/kaggle.json or ~/.kaggle/access_token "
            "(Kaggle > Settings > Create New Token writes one of the two), "
            "or export KAGGLE_USERNAME and KAGGLE_KEY."
        )


def kaggle_probe(slug: str) -> None:
    """Check that *slug* is reachable with the credential in place, fetching nothing.

    This is the reachability half of :func:`kaggle_download`, and it exists so
    that ``build_corpus.py --probe`` costs seconds rather than gigabytes: it
    lists the dataset's files (a metadata call) instead of pulling the bundle.
    A missing token, a revoked token and a slug that has been renamed or taken
    down all surface here, which is every Kaggle failure mode the real fetch has.

    **The CLI's exit code is not enough on its own.** ``kaggle datasets files``
    catches API errors itself, prints them, and still exits 0 — so a 403 would
    read as a pass. Success is therefore recognised positively: a CSV listing
    with a ``name`` column and at least one row. Anything else is a failure and
    the raw output is quoted back.
    """
    require_kaggle_credentials(slug)

    cmd = ["kaggle", "datasets", "files", "-d", slug, "--csv"]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
    except FileNotFoundError as exc:
        raise FetchError("The 'kaggle' CLI is not installed (pip install kaggle).") from exc
    except subprocess.CalledProcessError as exc:
        raise FetchError(f"kaggle metadata call for '{slug}' failed: {exc.stderr.strip()[:400]}") from exc

    rows = [line for line in (proc.stdout or "").splitlines() if line.strip()]
    header = rows[0].lower().split(",") if rows else []
    if "name" not in header or len(rows) < 2:
        detail = " ".join((proc.stdout or "").split())[:400] or "(no output)"
        raise FetchError(f"kaggle could not list '{slug}': {detail}")


def kaggle_download(slug: str, dest: Path, *, unzip: bool = True) -> Path:
    """Download a Kaggle dataset *slug* (``owner/name``) into *dest*.

    Uses the Kaggle CLI; see :func:`require_kaggle_credentials` for how the
    token is found.  Use :func:`kaggle_probe` when you only want to know whether
    the source is reachable — this one transfers the whole bundle.
    """
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        return dest

    require_kaggle_credentials(slug)

    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest)]
    if unzip:
        cmd.append("--unzip")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
    except FileNotFoundError as exc:
        raise FetchError("The 'kaggle' CLI is not installed (pip install kaggle).") from exc
    except subprocess.CalledProcessError as exc:
        raise FetchError(f"kaggle download of '{slug}' failed: {exc.stderr.strip()[:400]}") from exc
    return dest


def http_download(url: str, dest: Path, *, chunk: int = 1 << 20, session: Any = None) -> Path:
    """Stream *url* to *dest*, resuming a partial file and writing atomically.

    Pass *session* to reuse one connection across a long pull.  Measured on the
    UCSF endpoint it is worth only ~1.09x (307ms -> 281ms per PDF, so the cost
    is the archive generating and sending the file, not the TLS handshake), but
    it is free and it is strictly *less* load on a shared public service than
    re-handshaking once per document across 216,000 of them.
    """
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    have = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    get = session.get if session is not None else requests.get
    with get(url, headers=headers, stream=True, timeout=(20, 120)) as resp:
        if resp.status_code not in (200, 206):
            raise FetchError(f"{url} returned HTTP {resp.status_code}")
        mode = "ab" if have and resp.status_code == 206 else "wb"
        with tmp.open(mode) as fh:
            for block in resp.iter_content(chunk_size=chunk):
                fh.write(block)
    tmp.replace(dest)
    return dest


def extract_rar(archive: Path, dest: Path) -> Path:
    """Unpack a RAR archive, trying the tools most likely to exist on a cluster.

    SPODS ships as RAR4, which Python cannot read from the standard library.
    ``bsdtar`` (libarchive) handles it and is far more commonly installed on a
    compute node than ``unrar`` is.
    """
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        return dest

    for cmd in (
        ["bsdtar", "-x", "-f", str(archive), "-C", str(dest)],
        ["7z", "x", f"-o{dest}", "-y", str(archive)],
        ["unar", "-o", str(dest), str(archive)],
        ["unrar", "x", "-y", str(archive), str(dest) + "/"],
    ):
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
            return dest
        except FileNotFoundError:
            continue
        except subprocess.CalledProcessError as exc:
            raise FetchError(f"{cmd[0]} failed on {archive.name}: {exc.stderr.strip()[:300]}") from exc
    raise FetchError(
        f"No RAR extractor found for {archive.name}. Install one of: bsdtar (libarchive), 7z (p7zip), unar, unrar."
    )


def extract_zip(archive: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        return dest
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            # Reject absolute paths and traversal before writing anything.
            name = Path(member.filename)
            if name.is_absolute() or ".." in name.parts:
                raise FetchError(f"unsafe path in {archive.name}: {member.filename}")
        zf.extractall(dest)  # noqa: S202 - paths validated above
    return dest
