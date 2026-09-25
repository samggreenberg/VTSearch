"""The UCSF-class marks a box pass left without a reviewed box (#4125).

The v4.3 pass (#4109) proposed a tight box for every UCSF-class instance and a
person accepted, redrew or rejected each one.  Three kinds of mark were left:

* **query pages**, held back on purpose.  The right box on a class's query page
  is the query crop's own extent, ``query_box`` in the admission slate, and a
  crop is by definition tight on its mark.  Written as an already-answered
  ``box_tighten`` slate under ``audit/box_query``.
* **no fit** -- SIFT could not place the query crop.
* **rejected** -- the reviewer said the proposal was wrong.

The last two go into ``audit/box_draw`` as questions for a person to *draw* the
box on (``binary_review.py emit --task box_draw``): the current box outlined,
Good with a drawn box to replace it, Good alone to keep it, Bad when the mark
is not on the page.

    python box_leftovers.py --slate box_tighten_band_banked
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fullmarks_config as cfg  # noqa: E402

RELATIONS = Path("audit") / "ucsf_classes_banked" / "verdicts.jsonl"


def leftovers(
    rows: Sequence[dict[str, Any]], classes: dict[str, Any], query_boxes: dict[str, list[int]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(query-page rows, draw rows)`` from a banked ``box_tighten`` slate.

    A member is left over when it was neither accepted nor already tight
    (``unchanged``).  A query page's member is answered with its ``query_box``;
    every other left-over member becomes a question to draw.
    """
    query_rows, draw_rows = [], []
    for r in rows:
        cid = r["class_id"]
        verdict = str(r.get("verdict", ""))
        accepted = {int(i) for i in verdict.split(",")} if verdict not in ("", "none") else set()
        left = [m for m in r["members"] if m["index"] not in accepted and "unchanged" not in (m.get("flags") or [])]
        qpage = classes.get(cid, {}).get("query_page_id")
        on_query = [m for m in left if m["page_id"] == qpage and cid in query_boxes]
        rest = [m for m in left if m not in on_query]
        if on_query:
            members = [
                dict(m, index=i, new_box=list(query_boxes[cid]), flags=["query crop extent"])
                for i, m in enumerate(on_query)
            ]
            query_rows.append(
                {
                    "task": "box_tighten",
                    "class_id": cid,
                    "members": members,
                    "verdict": ",".join(str(i) for i in range(len(members))),
                    "verdict_source": "query crop extent (#4125)",
                    "notes": "the crop was cut from this box, so it is tight by construction",
                }
            )
        if rest:
            draw_rows.append(
                {
                    "task": "box_tighten",
                    "class_id": cid,
                    "members": [
                        {
                            "index": i,
                            "page_id": m["page_id"],
                            "mark_index": m["mark_index"],
                            "old_box": m["old_box"],
                            "new_box": None,
                            "why": "no fit" if "no fit" in (m.get("flags") or []) else "rejected",
                        }
                        for i, m in enumerate(rest)
                    ],
                    "verdict": "",
                    "notes": "",
                }
            )
    return query_rows, draw_rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--slate", required=True, help="banked box_tighten slate directory under <corpus>/audit/")
    args = ap.parse_args(argv)

    classes = json.loads((args.corpus / "classes.json").read_text(encoding="utf-8"))
    rel = [json.loads(line) for line in (args.corpus / RELATIONS).read_text(encoding="utf-8").splitlines() if line]
    query_boxes = {
        f"ucsf/logo_{r['proposal']}": r["query_box"]
        for r in rel
        if r.get("task") == "ucsf_classes_relation" and r.get("query_box")
    }
    rows = [
        json.loads(line)
        for line in (args.corpus / "audit" / args.slate / "verdicts.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    query_rows, draw_rows = leftovers(rows, classes, query_boxes)
    for name, out in (("box_query", query_rows), ("box_draw", draw_rows)):
        d = args.corpus / "audit" / name
        if (d / "verdicts.jsonl").exists():
            raise SystemExit(f"{d / 'verdicts.jsonl'} exists; it may hold answers, so it is never overwritten")
        d.mkdir(parents=True, exist_ok=True)
        (d / "verdicts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out), encoding="utf-8")
        print(f"{name}: {sum(len(r['members']) for r in out)} member(s) in {len(out)} class(es) -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
