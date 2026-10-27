"""Checkpoints for resumable streamed autodetect runs (``--resume``, #5280).

A ``--stream-results`` run over a very large source can take hours, and without
this a run that dies (a preempted node, an OOM kill, Ctrl-C) starts over from
its first chunk.  With ``--resume`` the run records a checkpoint beside its
output after every chunk it writes, and the same command run again continues
after the last chunk the checkpoint names.

The checkpoint holds three things:

- **The run's identity** (:func:`run_identity`): a digest per section - the
  source, the chunk size, each detector's labels, the settings, what is
  scored and where the output goes.  A run whose identity differs in any
  section is refused (:class:`ResumeRefusedError`), naming the sections,
  rather than appending one run's hits to another's.  Only digests are kept,
  so an importer's credentials never land in the file.
- **How far it got**: ``chunks_done``, the exporter's ``position`` after them
  (for a file, its byte length; see
  :class:`~vtscore.exporters.base.StreamResume`), and one fingerprint per
  completed chunk.  A resumed run walks the source to where it stopped,
  without embedding or scoring the completed chunks, and checks each against
  its fingerprint, so a source that changed underneath the run - a file
  added, removed or rewritten in the part already done, which would shift
  every chunk after it - is refused at the first chunk that differs.  A chunk
  whose importer can say what it holds without reading it (the
  ``server_folder`` walk: each file's path, size and mtime) is fingerprinted
  by that (:func:`key_digest`) and passed without reading its files; any
  other is re-read and fingerprinted by its media (:func:`chunk_digest`).
- **Each detector's threshold**: the cut the written chunks were judged at.
  A resumed run retrains its detectors on the same first chunk, keeps the
  recorded cut so every chunk is judged by one threshold, and refuses a
  retrained one that moved further than numerical noise can explain.

Only digests, offsets and thresholds are stored: no embedding and no model
weight, which are re-derived on resume as everywhere else.

Writes go through :func:`~vtscore.io.atomic_write_json`, after the exporter
has made the chunk durable, so the checkpoint never claims more than the
output holds.  It can claim less (the process died between the two writes);
the exporter then cuts the output back to the recorded position and that
chunk is written again, once.  The checkpoint is removed when the run
completes.  While the run writes, it holds a lock beside the output
(:func:`exclusive_run`), so a second run on the same file is refused rather
than interleaving its chunks with the first's.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vtscore.io import atomic_write_json

#: The ``format`` tag of a checkpoint file.  ``v2`` fingerprints a chunk by its
#: importer's key where it has one (#5314); a ``v1`` checkpoint is refused.
CHECKPOINT_FORMAT = "vtsearch-stream-checkpoint/v2"

#: Appended to the output path to name its checkpoint.
CHECKPOINT_SUFFIX = ".resume.json"

#: Appended to the output path to name the lock a running resumable run holds.
LOCK_SUFFIX = ".resume.lock"

#: The media fields a chunk fingerprint is taken over, in this order.  Enough
#: to tell the same files in the same order from anything else: the ``id``
#: (renumbered across the run, so it moves when an earlier chunk changed
#: size), the names the file is known by, the content ``md5``, and the clip
#: bounds of a clipped source.
_FINGERPRINT_FIELDS = ("id", "filename", "origin_name", "md5", "clip_start", "clip_end")

#: How far a retrained threshold may drift from the checkpoint's before the
#: resume is refused.  A detector retrained on the same chunk, labels and
#: settings lands on the same cut up to floating-point noise (a different GPU
#: on a requeued node); anything wider means something that decides the cut
#: changed.
THRESHOLD_DRIFT_LIMIT = 1e-3


class ResumeRefusedError(ValueError):
    """A ``--resume`` run cannot continue the run its checkpoint describes.

    A :class:`ValueError`, so the CLI reports it like any other refused input.
    The message names what differs and the checkpoint to delete to start over.
    """


def checkpoint_path_for(output_path: str | Path) -> Path:
    """The checkpoint that sits beside *output_path*."""
    p = Path(output_path)
    return p.with_name(p.name + CHECKPOINT_SUFFIX)


@contextlib.contextmanager
def exclusive_run(output_path: str | Path) -> Iterator[None]:
    """Hold *output_path* for one resumable run, refusing a second at the same time.

    Two runs appending to one file in place would interleave their chunks, a
    failure an atomic export never had.  So the run takes a non-blocking
    ``flock`` on a sibling ``<output>.resume.lock`` for its whole length, and a
    second run on the same output is refused at once rather than left to wait
    and then start over on a finished file.  The kernel drops the lock when the
    process dies, so a killed run never blocks the next; the file is removed
    when the run completes.  Without ``fcntl`` (Windows) there is no guard.
    """
    try:
        import fcntl  # noqa: PLC0415
    except ImportError:  # pragma: no cover - Windows
        yield
        return
    p = Path(output_path)
    lock_path = p.with_name(p.name + LOCK_SUFFIX)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ResumeRefusedError(
                f"Another --resume run is writing {p} (it holds {lock_path}). Let it finish, or stop it first."
            ) from None
        yield
        with contextlib.suppress(OSError):
            os.unlink(lock_path)
    finally:
        os.close(fd)


def digest(obj: Any) -> str:
    """A SHA-256 hex digest of *obj*'s canonical JSON (sorted keys, no whitespace)."""
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_digest(chunk: dict[int, dict[str, Any]]) -> str:
    """A short fingerprint of *chunk*: which media it holds, in order.

    Taken over :data:`_FINGERPRINT_FIELDS` of each media, before embedding,
    so it reads only what the loader already produced.  64 bits is plenty to
    tell one chunk from a changed one.
    """
    rows = [[media.get(key) for key in _FINGERPRINT_FIELDS] for media in chunk.values()]
    return digest(rows)[:16]


def key_digest(count: int, key: Any) -> str:
    """A short fingerprint of a chunk that was not built: its size and its importer's *key*.

    The *key* is the :class:`~vtscore.datasets.chunks.DeferredChunk` one, what
    the walk knows of the chunk's items without reading them (for a folder,
    each file's relative path, size and mtime).  It pins the chunk's
    boundaries as :func:`chunk_digest` does, and catches a rewrite as far as
    the file's size and mtime show one.  The *count* rides along for a key
    that is not one row per media.
    """
    return digest({"count": count, "key": key})[:16]


def detector_identity(det: dict[str, Any] | None) -> dict[str, Any]:
    """The parts of a detector's JSON that decide the head it trains and its cut.

    Its labels, its type, the clipper it was trained at, its typed query and
    its balance.  Not the whole file: derived caches (an evidence baseline, a
    line verdict) are written to it as a matter of course, and a resume must
    not be refused for those.  Anything else that moves the cut is caught by
    the threshold check instead.
    """
    from vtscore.detectors.balance import BETA_KEY  # noqa: PLC0415
    from vtscore.detectors.query_head import detector_text_query  # noqa: PLC0415

    det = det or {}
    return {
        "media_type": det.get("media_type") or "",
        "labelset": det.get("labelset") or {},
        "input_spec": det.get("input_spec"),
        "text_query": detector_text_query(det),
        "beta": det.get(BETA_KEY),
    }


def run_identity(
    *,
    source: dict[str, Any],
    chunk_size: int,
    detectors: dict[str, dict[str, Any]],
    settings: dict[str, Any],
    scoring: dict[str, Any],
    output: dict[str, Any],
) -> dict[str, Any]:
    """Assemble a run's identity: a digest per section, plus the chunk size as it stands.

    Each argument is that section's plain content; :func:`digest` reduces it,
    so nothing a section holds (an importer's password field, say) is written
    to the checkpoint.
    """
    return {
        "source": digest(source),
        "chunk_size": chunk_size,
        "detectors": digest(detectors),
        "settings": digest(settings),
        "scoring": digest(scoring),
        "output": digest(output),
    }


#: What each identity section means, for the refusal message.
_SECTION_NAMES = {
    "source": "the source (importer or dataset and its parameters)",
    "chunk_size": "the chunk size",
    "detectors": "the detectors or their labels",
    "settings": "the settings",
    "scoring": "what is scored (--keep-negatives, --override-clipper)",
    "output": "the exporter or its fields",
}


@dataclass
class StreamCheckpoint:
    """One resumable run's checkpoint: who it is and how far it got."""

    path: Path
    identity: dict[str, Any]
    chunks_done: int = 0
    chunk_digests: list[str] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)
    position: Any = None

    @classmethod
    def load(cls, path: Path) -> StreamCheckpoint | None:
        """Read the checkpoint at *path*; ``None`` when there is none.

        Raises :class:`ResumeRefusedError` for a file that is not a readable
        checkpoint, rather than silently starting over on top of the output
        it describes.
        """
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        try:
            data = json.loads(text)
            if data.get("format") != CHECKPOINT_FORMAT:
                raise ValueError(f"unknown format {data.get('format')!r}")
            checkpoint = cls(
                path=path,
                identity=dict(data["identity"]),
                chunks_done=int(data["chunks_done"]),
                chunk_digests=[str(d) for d in data["chunk_digests"]],
                thresholds={str(k): float(v) for k, v in data["thresholds"].items()},
                position=data["position"],
            )
            if len(checkpoint.chunk_digests) != checkpoint.chunks_done:
                raise ValueError(
                    f"{checkpoint.chunks_done} chunk(s) done but {len(checkpoint.chunk_digests)} fingerprinted"
                )
            return checkpoint
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ResumeRefusedError(
                f"The resume checkpoint {path} cannot be read ({exc}). Delete it to start the run over."
            ) from exc

    def save(self) -> None:
        """Write the checkpoint atomically."""
        atomic_write_json(
            self.path,
            {
                "format": CHECKPOINT_FORMAT,
                "identity": self.identity,
                "chunks_done": self.chunks_done,
                "chunk_digests": self.chunk_digests,
                "thresholds": self.thresholds,
                "position": self.position,
            },
            indent=1,
        )

    def clear(self) -> None:
        """Remove the checkpoint: the run it described is complete."""
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass

    def refuse(self, reason: str) -> ResumeRefusedError:
        """A :class:`ResumeRefusedError` for *reason*, telling the user how to start over."""
        return ResumeRefusedError(
            f"Cannot resume from {self.path}: {reason} Delete the checkpoint to start the run over."
        )

    def check_identity(self, identity: dict[str, Any]) -> None:
        """Raise :class:`ResumeRefusedError` unless *identity* is this checkpoint's run."""
        differ = [key for key in _SECTION_NAMES if self.identity.get(key) != identity.get(key)]
        if differ:
            changed = "; ".join(_SECTION_NAMES[key] for key in differ)
            raise self.refuse(f"this run is not the one the checkpoint records. Changed: {changed}.")

    def check_chunk(self, position: int, fingerprint: str) -> None:
        """Raise :class:`ResumeRefusedError` unless completed chunk *position* (1-based) has *fingerprint*."""
        if self.chunk_digests[position - 1] != fingerprint:
            raise self.refuse(
                f"the source changed since the checkpoint was written: chunk {position} no longer "
                "holds the same media (a file in the part already done was added, removed or rewritten)."
            )

    def check_reached(self, chunks_seen: int) -> None:
        """Raise :class:`ResumeRefusedError` if the source ended before the completed chunks did."""
        if chunks_seen < self.chunks_done:
            raise self.refuse(
                f"the source now holds {chunks_seen} chunk(s), but the checkpoint records "
                f"{self.chunks_done} already written."
            )

    def pin_thresholds(self, detector_mlps: dict[str, dict[str, Any]]) -> dict[str, tuple[float, float]]:
        """Judge the resumed chunks at the thresholds the written ones were judged at.

        Sets each detector's ``threshold`` in *detector_mlps* to the recorded
        one and returns ``{name: (retrained, recorded)}`` for those that moved
        at all.  Raises :class:`ResumeRefusedError` when the detectors that ran
        differ from the recorded ones, or one moved more than
        :data:`THRESHOLD_DRIFT_LIMIT`.
        """
        if set(detector_mlps) != set(self.thresholds):
            raise self.refuse(
                f"the detectors that run are {sorted(detector_mlps)}, but the checkpoint's chunks were "
                f"scored by {sorted(self.thresholds)}."
            )
        moved: dict[str, tuple[float, float]] = {}
        for name, info in detector_mlps.items():
            retrained, recorded = float(info["threshold"]), self.thresholds[name]
            if abs(retrained - recorded) > THRESHOLD_DRIFT_LIMIT:
                raise self.refuse(
                    f"detector '{name}' retrained to threshold {retrained:.4f}, but the chunks already written "
                    f"were judged at {recorded:.4f}, so something that decides its cut changed."
                )
            if retrained != recorded:
                moved[name] = (retrained, recorded)
            info["threshold"] = recorded
        return moved
