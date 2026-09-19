"""Where a slate's ``manifest.csv`` is, in either layout it is stored in.

A review campaign writes ``slates/<class>/manifest.csv`` in its working
directory. #3729's committed record flattens the same files to
``<CAMPAIGN>__slates__<class>__manifest.csv``, because a record is one directory
of named files rather than a tree. Both are the same artefact, and after #4001
deleted the working directories the record is usually the only copy left.

Two callers need it in different shapes -- `verdicts_to_corrections.py` wants
every manifest, `make_contact_sheets.py` wants one class's -- which is exactly
how a layout rule ends up spelled twice and drifting. It is spelled here once
(#4006).

**A working directory wins over the record.** If a campaign is mid-flight its
`slates/<class>/manifest.csv` is the live answer and the committed copy is a
snapshot; the same precedence `shipped_pool_error.py` documents for detectors
("both directories are read and the live copy wins").
"""

from __future__ import annotations

from pathlib import Path

_SUFFIX = "__manifest.csv"
_INFIX = "__slates__"


def _flat_name(path: Path) -> str:
    """The slate name inside a flattened record filename."""
    stem = path.name[: -len(_SUFFIX)]
    return stem.split(_INFIX)[-1] if _INFIX in stem else stem.split("__")[-1]


def slate_manifests(root: Path | str) -> list[Path]:
    """**Every** slate manifest under *root*, both layouts, deduplicated by path.

    A list rather than a name-keyed dict, deliberately. The record holds several
    campaigns' slates side by side -- ``WORK3588__slates__Bench__manifest.csv``
    and ``WORK__slates_pos2__Bench__manifest.csv`` are different slates of the
    same class -- so keying on the extracted name silently merges them and drops
    one. Measured on the real record: 64 files collapse to 41. A caller that
    wants all of them (the corrections recipe) must get all of them; dropping 23
    manifests from that input is the #3732 shape.
    """
    root = Path(root)
    found = list(root.glob("*/manifest.csv")) + list(root.glob(f"*{_SUFFIX}"))
    return sorted(set(found))


def slate_manifest(root: Path | str, name: str) -> Path | None:
    """The manifest for one slate *name*, or ``None``.

    Nested first: a campaign mid-flight is the live answer and the committed copy
    is a snapshot, the precedence ``shipped_pool_error.py`` documents for
    detectors. Among flattened candidates the primary ``__slates__`` set wins
    over a secondary one (``slates_pos2``, ``slates_audit``), because that is the
    set the nested layout corresponded to.
    """
    root = Path(root)
    nested = root / name / "manifest.csv"
    if nested.exists():
        return nested
    flat = [p for p in root.glob(f"*{_SUFFIX}") if _flat_name(p) == name]
    if not flat:
        return None
    primary = [p for p in flat if _INFIX in p.name]
    return sorted(primary or flat)[0]
