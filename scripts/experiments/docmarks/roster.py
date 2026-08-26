"""The roster — the hand-picked classes an eval actually runs on.

DocMarks is not trying to be an exhaustive inventory of every mark in every
source.  It is trying to answer one question — *can this pipeline find a given
stamp in a pile of documents* — and that question is answered better by two
dozen classes whose every instance a human has checked than by four hundred
classes assembled by a threshold nobody validated.

So the corpus has two populations with completely different standards of
evidence:

* **roster classes** — a small, named, checked-in set.  Every instance is
  adjudicated in or out by hand, and every confusable pair is adjudicated same
  or different.  Nothing enters by heuristic.
* **distractors** — everything else, unlabelled and unexamined, in whatever
  quantity the tier budget allows.  They need no labels; they only need to be
  safe to score against, which ``docmarks_config.CONTAMINATES`` decides.

That split is what makes the eval trustworthy at a cost a person can actually
pay.  Verifying 24 classes exhaustively is an afternoon; verifying 400 is not,
and a benchmark whose labels nobody checked is a benchmark whose numbers nobody
should quote.

The roster file is small and human-editable on purpose — it is a decision, not
an artifact:

    {
      "name": "spods-v1",
      "notes": "why these",
      "classes": ["spods/logo_00042_0", "spods/stamp_00117_1", ...]
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence


@dataclass
class Roster:
    """The chosen classes, plus provenance for why they were chosen."""

    name: str
    classes: list[str] = field(default_factory=list)
    notes: str = ""

    def __contains__(self, class_id: str) -> bool:
        return class_id in set(self.classes)

    def __len__(self) -> int:
        return len(self.classes)


def load(path: Path) -> Roster:
    payload = json.loads(path.read_text(encoding="utf-8"))
    classes = list(dict.fromkeys(payload.get("classes", [])))
    return Roster(name=payload.get("name", path.stem), classes=classes, notes=payload.get("notes", ""))


def save(roster: Roster, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": roster.name,
        "notes": roster.notes,
        "classes": sorted(roster.classes),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def check(roster: Roster, available: Sequence[str]) -> tuple[list[str], list[str]]:
    """``(present, missing)`` — roster entries that do and do not exist.

    A missing entry is reported rather than ignored.  Class ids move when the
    clustering threshold moves, so a roster naming a class that no longer exists
    is the signal that the roster and the corpus have drifted apart — which is
    exactly the failure that would otherwise show up as a quietly smaller eval.
    """
    have = set(available)
    present = [c for c in roster.classes if c in have]
    missing = [c for c in roster.classes if c not in have]
    return present, missing


def starter(name: str, candidates: Sequence[dict[str, Any]], size: int = 24) -> Roster:
    """A first-draft roster from the top *size* shortlist candidates.

    Meant as a starting point for a human to edit, never as the final word: the
    ranking can order candidates by every measurable proxy and still cannot tell
    whether a mark is *interesting*.  ``shortlist.py`` prints the table and the
    contact sheet that make that judgement possible.
    """
    return Roster(
        name=name,
        classes=[c["class_id"] for c in candidates[:size]],
        notes=(
            f"first draft from the top {size} shortlist candidates; edit by hand before running the membership audit"
        ),
    )


def eligible_pages(
    class_meta: dict[str, Any],
    pages_by_source: dict[str, list[str]],
    verified_negative_sources: Optional[Sequence[str]] = None,
) -> dict[str, list[str]]:
    """Split the corpus into what may be scored against this class.

    Three populations, and the distinction matters for what a number means:

    * ``positive`` — adjudicated instances of the mark.
    * ``known_negative`` — pages from a source that was exhaustively checked for
      this class, so their *absence* of the mark is a verified fact.  These are
      the valuable negatives: same scanner, same paper, same era, and known
      clean.  A SPODS page carrying a different mark is the hardest possible
      negative for a SPODS class, and exhaustive verification is what makes it
      usable instead of a contamination risk.
    * ``presumed_negative`` — pages from a contamination-safe source that nobody
      checked individually.  Fine in bulk, and the only way to reach 200k.
    """
    positives = set(class_meta.get("page_ids", []))
    verified = set(verified_negative_sources or ())
    eligible = set(class_meta.get("eligible_distractor_sources", []))

    known: list[str] = []
    presumed: list[str] = []
    for source, page_ids in sorted(pages_by_source.items()):
        for page_id in page_ids:
            if page_id in positives:
                continue
            if source in verified:
                known.append(page_id)
            elif source in eligible:
                presumed.append(page_id)
    return {
        "positive": sorted(positives),
        "known_negative": sorted(known),
        "presumed_negative": sorted(presumed),
    }
